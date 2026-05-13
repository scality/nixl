/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/*
 * Scality AI Connector client unit tests — HTTP wire-format verification
 *
 * Goal
 * ----
 * Confirm that ScalityClient sends correctly-formed HTTP requests to the
 * Scality AI Connector endpoint WITHOUT requiring any RDMA hardware, cuObject,
 * or the nvidia-fs kernel module.  These tests are the lowest-level sanity
 * check for the connector integration: if they pass, the HTTP layer is correct
 * and the only remaining unknowns are the RDMA-side pieces.
 *
 * Strategy
 * --------
 * Each test:
 *  1. Starts a minimal single-request TCP server (TcpServer) that listens on
 *     a random loopback port, reads the raw HTTP bytes sent by libcurl, and
 *     replies "200 OK" so curl considers the request successful.
 *  2. Instantiates ScalityClient pointing at that server.
 *  3. Calls putObjectRdmaAsync or getObjectRdmaAsync with known inputs and a
 *     hardcoded fake RDMA descriptor (in production this comes from cuObjPut/Get).
 *  4. Asserts on the raw HTTP text captured by the server.
 *
 * What is NOT tested here
 * -----------------------
 * - The cuObject RDMA descriptor generation (requires nvidia-fs + RDMA NIC).
 * - The full engine pipeline (registerMem / prepXfer / postXfer) — those
 *   require cuObjClient and are covered separately once hardware is available.
 */

#include <gtest/gtest.h>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <netinet/in.h>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "nixl_types.h"
#include "rest_accel/scality_ai_connector/client.h"

namespace gtest::obj {

// ─── TcpServer ───────────────────────────────────────────────────────────────
//
// A throwaway single-request HTTP server.
//
// Constructor: binds to localhost:0 (OS picks a free port), starts a
//   background thread that blocks on accept().
// capturedRequest(): waits for the background thread to finish reading one
//   full HTTP request (terminated by "\r\n\r\n") and returns the raw text.
//   Returns "" on timeout.
// Destructor: closes the listening socket (unblocks accept if no client ever
//   connected) and joins the thread.
//
class TcpServer {
public:
    TcpServer() {
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        EXPECT_GE(listen_fd_, 0) << "socket() failed: " << strerror(errno);

        // Allow rapid port reuse between tests
        int opt = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};
        addr.sin_family      = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port        = 0; // let the OS assign a free port

        EXPECT_EQ(bind(listen_fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)), 0)
            << "bind() failed: " << strerror(errno);
        EXPECT_EQ(listen(listen_fd_, 1), 0) << "listen() failed: " << strerror(errno);

        // Discover the assigned port so the test can point the client at it
        socklen_t len = sizeof(addr);
        getsockname(listen_fd_, reinterpret_cast<struct sockaddr *>(&addr), &len);
        port_ = ntohs(addr.sin_port);

        // Launch background thread — runs acceptAndRead()
        future_ = promise_.get_future();
        thread_ = std::thread(&TcpServer::acceptAndRead, this);
    }

    ~TcpServer() {
        // Closing the listening socket will unblock accept() if no client came
        if (listen_fd_ >= 0)
            close(listen_fd_);
        if (thread_.joinable())
            thread_.join();
    }

    // Port the server is listening on (pass this to ScalityClient)
    int
    port() const {
        return port_;
    }

    // Block until the server has captured one full HTTP request (or timeout).
    // Returns the raw request text, or "" on timeout / accept failure.
    std::string
    capturedRequest(int timeout_sec = 5) {
        if (future_.wait_for(std::chrono::seconds(timeout_sec)) ==
            std::future_status::timeout) {
            return "";
        }
        return future_.get();
    }

private:
    // Thread body: accept one connection, read headers, reply 200, done.
    void
    acceptAndRead() {
        int client_fd = accept(listen_fd_, nullptr, nullptr);
        if (client_fd < 0) {
            // listen_fd_ was closed — no request came, signal empty
            promise_.set_value("");
            return;
        }

        // Read bytes until we see the blank line that ends HTTP headers
        std::string request;
        char        buf[4096];
        while (request.find("\r\n\r\n") == std::string::npos) {
            ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
            if (n <= 0)
                break;
            request.append(buf, static_cast<std::string::size_type>(n));
        }

        // Reply with a minimal 200 so libcurl treats the transfer as successful
        const char *response = "HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n";
        send(client_fd, response, strlen(response), 0);
        close(client_fd);

        promise_.set_value(request);
    }

    int                       listen_fd_ = -1;
    int                       port_      = 0;
    std::thread               thread_;
    std::promise<std::string> promise_;
    std::future<std::string>  future_;
};

// ─── Helpers ─────────────────────────────────────────────────────────────────

// Build the nixl_b_params_t that ScalityClient's constructor expects.
static nixl_b_params_t
makeScalityParams(const std::string &endpoint) {
    return {{"endpoint_override", endpoint}};
}

// Spin-wait until the async callback fires (or until timeout_ms elapses).
// Needed because putObjectRdmaAsync / getObjectRdmaAsync dispatch work to an
// internal thread pool and return immediately.
static void
waitForCallback(const std::atomic<bool> &done, int timeout_ms = 5000) {
    auto deadline =
        std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!done.load() && std::chrono::steady_clock::now() < deadline)
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
}

// ─── Test fixture ─────────────────────────────────────────────────────────────

class ScalityClientTest : public testing::Test {};

// ─── PUT (WRITE) tests ────────────────────────────────────────────────────────

// Core PUT test: verify method, URL, both RDMA headers, and empty body.
TEST_F(ScalityClientTest, PutSendsCorrectUrlAndHeaders) {
    TcpServer server;
    nixl_b_params_t params =
        makeScalityParams("http://127.0.0.1:" + std::to_string(server.port()));
    ScalityClient client(&params);

    // This would normally be the cuObjPut RDMA descriptor — we pass a fake one
    // to exercise the HTTP layer without real RDMA hardware.
    const std::string rdma_desc  = "fake-rdma-descriptor-for-testing";
    const size_t      offset     = 512;
    const size_t      data_len   = 1024;
    std::vector<char> buf(data_len);

    std::atomic<bool> done{false}, success{false};
    client.putObjectRdmaAsync("mykey",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              data_len,
                              offset,
                              rdma_desc,
                              [&](bool ok) {
                                  success = ok;
                                  done    = true;
                              });

    // Capture the raw HTTP request the server received, then wait for callback
    std::string req = server.capturedRequest();
    waitForCallback(done);

    // Method and path: PUT /v1/{key}
    EXPECT_NE(req.find("PUT /v1/mykey"), std::string::npos)
        << "Expected 'PUT /v1/mykey' in request:\n"
        << req;

    // x-scal-rdma carries the descriptor the server uses to initiate the
    // RDMA pull; it must match exactly what cuObjPut returned.
    EXPECT_NE(req.find("x-scal-rdma: " + rdma_desc), std::string::npos)
        << "x-scal-rdma header missing or wrong in:\n"
        << req;

    // Body must be empty — the actual data moves via RDMA, not the HTTP body
    EXPECT_NE(req.find("Content-Length: 0"), std::string::npos)
        << "Content-Length: 0 missing in:\n"
        << req;

    EXPECT_TRUE(success.load()) << "putObjectRdmaAsync reported failure";
}

// ─── GET (READ) tests ─────────────────────────────────────────────────────────

// Core GET test: verify method, URL, and both RDMA headers
TEST_F(ScalityClientTest, GetSendsCorrectUrlAndHeaders) {
    TcpServer       server;
    nixl_b_params_t params =
        makeScalityParams("http://127.0.0.1:" + std::to_string(server.port()));
    ScalityClient   client(&params);

    const size_t      offset = 128;
    std::vector<char> buf(512);

    std::atomic<bool> done{false}, success{false};
    client.getObjectRdmaAsync("readkey",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              offset,
                              "rdma-get-token",
                              [&](bool ok) {
                                  success = ok;
                                  done    = true;
                              });

    std::string req = server.capturedRequest();
    waitForCallback(done);

    // Must be GET, not PUT
    EXPECT_NE(req.find("GET /v1/readkey"), std::string::npos)
        << "Expected 'GET /v1/readkey' in:\n"
        << req;
    EXPECT_EQ(req.find("PUT"), std::string::npos)
        << "GET request must not contain PUT method in:\n"
        << req;

    EXPECT_NE(req.find("x-scal-rdma: rdma-get-token"), std::string::npos)
        << "x-scal-rdma missing in:\n"
        << req;

    EXPECT_TRUE(success.load()) << "getObjectRdmaAsync reported failure";
}

// ─── Guard-clause tests (client rejects locally, no server needed) ────────────
//
// These tests verify that the client enforces preconditions BEFORE dispatching
// any work to its thread pool or making a network connection.

// A PUT with an empty RDMA descriptor must fail immediately.
// The server cannot do RDMA without a valid descriptor, so sending the request
// would be pointless and the key would be left in an undefined state.
TEST_F(ScalityClientTest, PutRejectsEmptyRdmaDesc) {
    // Port 1 is reserved and won't accept connections — chosen deliberately so
    // that if the client somehow makes a connection attempt, it will fail visibly.
    nixl_b_params_t params = makeScalityParams("http://127.0.0.1:1");
    ScalityClient   client(&params);

    std::vector<char> buf(64);
    std::atomic<bool> done{false}, success{true}; // start true; expect false

    client.putObjectRdmaAsync("k",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              0,
                              /*rdma_desc=*/"",
                              [&](bool ok) {
                                  success = ok;
                                  done    = true;
                              });

    waitForCallback(done);
    EXPECT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(success.load()) << "Expected failure with empty rdma_desc";
}

// A PUT with data_len = 0 must fail immediately (nothing to transfer)
TEST_F(ScalityClientTest, PutRejectsZeroDataLen) {
    nixl_b_params_t params = makeScalityParams("http://127.0.0.1:1");
    ScalityClient   client(&params);

    std::atomic<bool> done{false}, success{true};
    client.putObjectRdmaAsync("k",
                              /*data_ptr=*/0,
                              /*data_len=*/0,
                              0,
                              "some-token",
                              [&](bool ok) {
                                  success = ok;
                                  done    = true;
                              });

    waitForCallback(done);
    EXPECT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(success.load()) << "Expected failure with data_len=0";
}

// A GET with an empty RDMA descriptor must fail immediately
TEST_F(ScalityClientTest, GetRejectsEmptyRdmaDesc) {
    nixl_b_params_t params = makeScalityParams("http://127.0.0.1:1");
    ScalityClient   client(&params);

    std::vector<char> buf(64);
    std::atomic<bool> done{false}, success{true};
    client.getObjectRdmaAsync("k",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              0,
                              /*rdma_desc=*/"",
                              [&](bool ok) {
                                  success = ok;
                                  done    = true;
                              });

    waitForCallback(done);
    EXPECT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(success.load()) << "Expected failure with empty rdma_desc";
}

// ─── Constructor validation tests ────────────────────────────────────────────

// ScalityClient must throw if the required 'endpoint_override' param is absent
TEST_F(ScalityClientTest, ConstructorThrowsOnMissingEndpoint) {
    nixl_b_params_t params = {}; // intentionally empty
    EXPECT_THROW({ ScalityClient c(&params); }, std::invalid_argument);
}

// ScalityClient must throw if custom_params itself is null
TEST_F(ScalityClientTest, ConstructorThrowsOnNullParams) {
    EXPECT_THROW({ ScalityClient c(nullptr); }, std::invalid_argument);
}

} // namespace gtest::obj
