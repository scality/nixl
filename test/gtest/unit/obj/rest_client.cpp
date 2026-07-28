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
 * RestClient HTTP wire-format tests.
 *
 * Confirms that RestClient sends correctly-formed HTTP requests to the
 * connector endpoint without any RDMA hardware, cuObject, or nvidia-fs: each
 * test starts a one-shot loopback TCP server, points a RestClient at it, issues
 * a request with a fake RDMA descriptor, and asserts on the raw HTTP captured.
 */

#include <gtest/gtest.h>
#include <arpa/inet.h>
#include <atomic>
#include <chrono>
#include <cstring>
#include <future>
#include <netinet/in.h>
#include <optional>
#include <string>
#include <sys/socket.h>
#include <thread>
#include <unistd.h>
#include <vector>

#include "nixl_types.h"
#include "rest_accel/scality_ai_connector/client.h"
#include "rest_accel/scality_ai_connector/split.h"

namespace gtest::obj {

// ---------------------------------------------------------------------------
// Request splitting arithmetic
//
// prepXfer cuts one transfer descriptor into ranged requests whose boundaries fall
// on multiples of split_size in the OBJECT's offset space, not relative to the
// descriptor's start. The split happens in the engine, above RestClient, so it is
// not observable from the HTTP harness below; these cover the arithmetic and its
// edge cases directly.
//
// Reconstructs the emitted ranges the same way prepXfer does, so a change to one
// has to be mirrored in the other.
// ---------------------------------------------------------------------------

namespace {

struct Range {
    size_t offset; ///< absolute object offset
    size_t len;
};

/// Replay prepXfer's loop for a descriptor of `total` bytes at object offset `base`.
std::vector<Range>
splitRanges(size_t base, size_t total, size_t split) {
    std::vector<Range> out;
    size_t off = 0;
    do {
        size_t len = total - off;
        if (split != 0) {
            const size_t to_boundary = split - ((base + off) % split);
            if (to_boundary < len) {
                len = to_boundary;
            }
        }
        out.push_back({base + off, len});
        off += len;
    } while (off < total);
    return out;
}

} // namespace

TEST(ObjRequestSplitTest, AlignedBaseTilesEvenly) {
    // base already on a boundary: no short head, exactly total/split requests.
    EXPECT_EQ(objRequestCount(0, 128u << 20, 16u << 20), 8u);
    const auto r = splitRanges(0, 128u << 20, 16u << 20);
    ASSERT_EQ(r.size(), 8u);
    for (const auto &x : r) {
        EXPECT_EQ(x.len, 16u << 20);
        EXPECT_EQ(x.offset % (16u << 20), 0u);
    }
}

TEST(ObjRequestSplitTest, UnalignedBaseGetsShortHeadThenAligns) {
    // This is the case that matters: a safetensors tensor starts at an arbitrary
    // offset, and every request after the head must land on a boundary.
    const size_t split = 8u << 20;
    const size_t base = (8u << 20) + 1234; // one boundary in, plus a skew
    const auto r = splitRanges(base, 32u << 20, split);

    EXPECT_EQ(objRequestCount(base, 32u << 20, split), r.size());
    ASSERT_GE(r.size(), 2u);
    EXPECT_EQ(r.front().len, split - 1234) << "head should reach the next boundary";
    for (size_t i = 1; i < r.size(); ++i) {
        EXPECT_EQ(r[i].offset % split, 0u)
            << "request " << i << " at offset " << r[i].offset << " is not aligned";
    }
}

TEST(ObjRequestSplitTest, BelowSplitSizeIsOneRequest) {
    EXPECT_EQ(objRequestCount(0, 1, 16u << 20), 1u);
    EXPECT_EQ(objRequestCount(0, 16u << 20, 16u << 20), 1u);
}

TEST(ObjRequestSplitTest, ZeroLengthStillYieldsOneRequest) {
    // Pre-split behaviour was one request per descriptor regardless of length;
    // an empty descriptor must not silently vanish.
    EXPECT_EQ(objRequestCount(0, 0, 16u << 20), 1u);
    EXPECT_EQ(objRequestCount(12345, 0, 0), 1u);
    EXPECT_EQ(splitRanges(12345, 0, 8u << 20).size(), 1u);
}

TEST(ObjRequestSplitTest, SplitSizeZeroDisablesSplitting) {
    EXPECT_EQ(objRequestCount(0, 1u << 30, 0), 1u);
    EXPECT_EQ(splitRanges(999, 1u << 30, 0).size(), 1u);
}

TEST(ObjRequestSplitTest, AlignmentCostsAtMostOneExtraRequest) {
    // The head request is the only price of aligning; anything more would mean the
    // arithmetic is fragmenting the descriptor.
    const size_t split = 8u << 20;
    for (size_t skew : {size_t{0}, size_t{1}, size_t{4095}, split - 1}) {
        const size_t aligned = objRequestCount(skew, 64u << 20, split);
        const size_t relative = (64u << 20) / split;
        EXPECT_LE(aligned, relative + 1) << "skew " << skew << " fragmented the split";
    }
}

TEST(ObjRequestSplitTest, TilingIsContiguousAndExact) {
    // Ranges must tile the descriptor exactly once: no gaps, no overlaps, nothing
    // past the end, and the count must match what objRequestCount predicts.
    for (size_t base : {size_t{0}, size_t{1}, size_t{7}, size_t{16}, size_t{1000003}}) {
        for (size_t total :
             {size_t{0}, size_t{1}, size_t{15}, size_t{16}, size_t{17}, size_t{1024}}) {
            for (size_t split : {size_t{1}, size_t{4}, size_t{16}, size_t{4096}}) {
                const auto r = splitRanges(base, total, split);
                EXPECT_EQ(r.size(), objRequestCount(base, total, split))
                    << "count mismatch base=" << base << " total=" << total
                    << " split=" << split;
                size_t expected = base;
                size_t covered = 0;
                for (const auto &x : r) {
                    EXPECT_EQ(x.offset, expected) << "gap/overlap base=" << base
                                                  << " total=" << total
                                                  << " split=" << split;
                    covered += x.len;
                    expected = x.offset + x.len;
                }
                EXPECT_EQ(covered, total) << "incomplete coverage base=" << base
                                          << " total=" << total << " split=" << split;
            }
        }
    }
}

// A throwaway single-request HTTP server: binds to localhost:0, reads one full
// HTTP request, replies "200 OK", and hands the raw request text back.
class TcpServer {
public:
    explicit TcpServer(int status_code = 200, std::string reason = "OK")
        : status_code_(status_code),
          reason_(std::move(reason)) {
        listen_fd_ = socket(AF_INET, SOCK_STREAM, 0);
        EXPECT_GE(listen_fd_, 0) << "socket() failed: " << strerror(errno);

        int opt = 1;
        setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &opt, sizeof(opt));

        struct sockaddr_in addr{};

        addr.sin_family = AF_INET;
        addr.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
        addr.sin_port = 0; // let the OS assign a free port

        EXPECT_EQ(bind(listen_fd_, reinterpret_cast<struct sockaddr *>(&addr), sizeof(addr)), 0)
            << "bind() failed: " << strerror(errno);
        EXPECT_EQ(listen(listen_fd_, 1), 0) << "listen() failed: " << strerror(errno);

        socklen_t len = sizeof(addr);
        getsockname(listen_fd_, reinterpret_cast<struct sockaddr *>(&addr), &len);
        port_ = ntohs(addr.sin_port);

        future_ = promise_.get_future();
        thread_ = std::thread(&TcpServer::acceptAndRead, this);
    }

    ~TcpServer() {
        if (listen_fd_ >= 0) {
            close(listen_fd_);
        }
        if (thread_.joinable()) {
            thread_.join();
        }
    }

    int
    port() const {
        return port_;
    }

    // Block until one full HTTP request is captured (or timeout); "" on timeout.
    std::string
    capturedRequest(int timeout_sec = 5) {
        if (future_.wait_for(std::chrono::seconds(timeout_sec)) == std::future_status::timeout) {
            return "";
        }
        return future_.get();
    }

private:
    void
    acceptAndRead() {
        int client_fd = accept(listen_fd_, nullptr, nullptr);
        if (client_fd < 0) {
            promise_.set_value("");
            return;
        }

        std::string request;
        char buf[4096];
        while (request.find("\r\n\r\n") == std::string::npos) {
            ssize_t n = recv(client_fd, buf, sizeof(buf), 0);
            if (n <= 0) {
                break;
            }
            request.append(buf, static_cast<std::string::size_type>(n));
        }

        std::string response = "HTTP/1.1 " + std::to_string(status_code_) + " " + reason_ +
            "\r\nContent-Length: 0\r\n\r\n";
        send(client_fd, response.data(), response.size(), 0);
        close(client_fd);

        promise_.set_value(request);
    }

    int status_code_;
    std::string reason_;
    int listen_fd_ = -1;
    int port_ = 0;
    std::thread thread_;
    std::promise<std::string> promise_;
    std::future<std::string> future_;
};

static nixl_b_params_t
makeRestParams(const std::string &endpoint) {
    return {{"endpoint_override", endpoint}};
}

// The async calls dispatch to an internal thread pool and return immediately;
// spin-wait until the callback fires (or timeout).
static void
waitForCallback(const std::atomic<bool> &done, int timeout_ms = 5000) {
    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms);
    while (!done.load() && std::chrono::steady_clock::now() < deadline) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
}

class RestClientTest : public testing::Test {};

TEST_F(RestClientTest, PutSendsCorrectUrlAndHeaders) {
    TcpServer server;
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:" + std::to_string(server.port()));
    RestClient client(&params);

    const std::string rdma_desc = "fake-rdma-descriptor-for-testing";
    const size_t offset = 512;
    const size_t data_len = 1024;
    std::vector<char> buf(data_len);

    std::atomic<bool> done{false}, success{false};
    client.putObjectRdmaAsync("mykey",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              data_len,
                              offset,
                              rdma_desc,
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    std::string req = server.capturedRequest();
    waitForCallback(done);

    EXPECT_NE(req.find("PUT /mykey"), std::string::npos) << "Expected 'PUT /mykey' in request:\n"
                                                         << req;
    EXPECT_NE(req.find("x-scal-rdma: " + rdma_desc), std::string::npos)
        << "x-scal-rdma header missing or wrong in:\n"
        << req;
    // Body must be empty; the data moves via RDMA, not the HTTP body.
    EXPECT_NE(req.find("Content-Length: 0"), std::string::npos) << "Content-Length: 0 missing in:\n"
                                                                << req;
    EXPECT_TRUE(success.load()) << "putObjectRdmaAsync reported failure";
}

TEST_F(RestClientTest, GetSendsCorrectUrlAndHeaders) {
    TcpServer server;
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:" + std::to_string(server.port()));
    RestClient client(&params);

    const size_t offset = 128;
    std::vector<char> buf(512);

    std::atomic<bool> done{false}, success{false};
    client.getObjectRdmaAsync("readkey",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              offset,
                              "rdma-get-token",
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    std::string req = server.capturedRequest();
    waitForCallback(done);

    EXPECT_NE(req.find("GET /readkey"), std::string::npos) << "Expected 'GET /readkey' in:\n"
                                                           << req;
    EXPECT_EQ(req.find("PUT"), std::string::npos) << "GET request must not contain PUT method in:\n"
                                                  << req;
    EXPECT_NE(req.find("x-scal-rdma: rdma-get-token"), std::string::npos)
        << "x-scal-rdma missing in:\n"
        << req;
    EXPECT_NE(req.find("Range: bytes=128-639"), std::string::npos) << "Range missing in:\n" << req;
    EXPECT_TRUE(success.load()) << "getObjectRdmaAsync reported failure";
}

// A sized read at offset 0 must still be ranged, else the server fetches the
// whole object into a buffer registered for only data_len bytes.
TEST_F(RestClientTest, GetAtOffsetZeroSendsRangeHeader) {
    TcpServer server;
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:" + std::to_string(server.port()));
    RestClient client(&params);

    std::vector<char> buf(8);

    std::atomic<bool> done{false}, success{false};
    client.getObjectRdmaAsync("headerprobe",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              /*offset=*/0,
                              "rdma-get-token",
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    std::string req = server.capturedRequest();
    waitForCallback(done);

    EXPECT_NE(req.find("GET /headerprobe"), std::string::npos)
        << "Expected 'GET /headerprobe' in:\n"
        << req;
    EXPECT_NE(req.find("Range: bytes=0-7"), std::string::npos)
        << "offset-0 read must be ranged in:\n"
        << req;
    EXPECT_TRUE(success.load()) << "getObjectRdmaAsync reported failure";
}

TEST_F(RestClientTest, PutRejectsEmptyRdmaDesc) {
    // Port 1 won't accept connections, so a stray connection attempt fails loudly.
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:1");
    RestClient client(&params);

    std::vector<char> buf(64);
    std::atomic<bool> done{false}, success{true};

    client.putObjectRdmaAsync("k",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              0,
                              /*rdma_desc=*/"",
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    waitForCallback(done);
    EXPECT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(success.load()) << "Expected failure with empty rdma_desc";
}

TEST_F(RestClientTest, PutRejectsZeroDataLen) {
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:1");
    RestClient client(&params);

    std::atomic<bool> done{false}, success{true};
    client.putObjectRdmaAsync("k",
                              /*data_ptr=*/0,
                              /*data_len=*/0,
                              0,
                              "some-token",
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    waitForCallback(done);
    EXPECT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(success.load()) << "Expected failure with data_len=0";
}

TEST_F(RestClientTest, GetRejectsEmptyRdmaDesc) {
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:1");
    RestClient client(&params);

    std::vector<char> buf(64);
    std::atomic<bool> done{false}, success{true};
    client.getObjectRdmaAsync("k",
                              reinterpret_cast<uintptr_t>(buf.data()),
                              buf.size(),
                              0,
                              /*rdma_desc=*/"",
                              [&](bool ok) {
                                  success = ok;
                                  done = true;
                              });

    waitForCallback(done);
    EXPECT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(success.load()) << "Expected failure with empty rdma_desc";
}

TEST_F(RestClientTest, ConstructorThrowsOnMissingEndpoint) {
    nixl_b_params_t params = {}; // intentionally empty
    EXPECT_THROW({ RestClient c(&params); }, std::invalid_argument);
}

TEST_F(RestClientTest, ConstructorThrowsOnNullParams) {
    EXPECT_THROW({ RestClient c(nullptr); }, std::invalid_argument);
}

// ─── Existence check (HTTP HEAD) tests ────────────────────────────────────────

TEST_F(RestClientTest, CheckExistsReturnsTrueOn200) {
    TcpServer server(200, "OK");
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:" + std::to_string(server.port()));
    RestClient client(&params);

    std::atomic<bool> done{false};
    std::optional<bool> result;
    client.checkObjectExistsAsync("mykey", [&](std::optional<bool> exists) {
        result = exists;
        done = true;
    });

    std::string req = server.capturedRequest();
    waitForCallback(done);

    EXPECT_NE(req.find("HEAD /mykey"), std::string::npos) << "Expected 'HEAD /mykey' in:\n" << req;
    ASSERT_TRUE(result.has_value()) << "Expected a definite result, got error";
    EXPECT_TRUE(*result) << "Expected exists=true for HTTP 200";
}

TEST_F(RestClientTest, CheckExistsReturnsFalseOn404) {
    TcpServer server(404, "Not Found");
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:" + std::to_string(server.port()));
    RestClient client(&params);

    std::atomic<bool> done{false};
    std::optional<bool> result;
    client.checkObjectExistsAsync("missing", [&](std::optional<bool> exists) {
        result = exists;
        done = true;
    });

    server.capturedRequest();
    waitForCallback(done);

    ASSERT_TRUE(result.has_value()) << "Expected a definite result, got error";
    EXPECT_FALSE(*result) << "Expected exists=false for HTTP 404";
}

TEST_F(RestClientTest, CheckExistsReturnsErrorOn500) {
    TcpServer server(500, "Internal Server Error");
    nixl_b_params_t params = makeRestParams("http://127.0.0.1:" + std::to_string(server.port()));
    RestClient client(&params);

    std::atomic<bool> done{false};
    std::optional<bool> result{false}; // sentinel; a 500 must reset this to nullopt
    client.checkObjectExistsAsync("boom", [&](std::optional<bool> exists) {
        result = exists;
        done = true;
    });

    server.capturedRequest();
    waitForCallback(done);

    EXPECT_TRUE(done.load()) << "Callback was never invoked";
    EXPECT_FALSE(result.has_value()) << "Expected error (nullopt) for HTTP 500";
}

} // namespace gtest::obj
