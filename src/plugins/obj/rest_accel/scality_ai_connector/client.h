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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_CLIENT_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_CLIENT_H

#include <asio/thread_pool.hpp>
#include <asio/post.hpp>
#include <curl/curl.h>
#include <atomic>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <string_view>
#include <thread>
#include <unordered_set>
#include <vector>
#include "obj_backend.h"
#include "nixl_types.h"

/**
 * Interface for Scality AI Connector clients that support RDMA operations.
 */
class iRestClient {
public:
    virtual ~iRestClient() = default;

    /**
     * Asynchronously put an object using RDMA.
     * @param key The object key
     * @param data_ptr Pointer to the data to upload
     * @param data_len Length of the data in bytes
     * @param offset Offset within the object
     * @param rdma_desc RDMA descriptor for the transfer
     * @param callback Callback function to handle the result
     */
    virtual void
    putObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       put_object_callback_t callback) = 0;

    /**
     * Asynchronously get an object using RDMA.
     * @param key The object key
     * @param data_ptr Pointer to the buffer to store the downloaded data
     * @param data_len Maximum length of data to read
     * @param offset Offset within the object to start reading from
     * @param rdma_desc RDMA descriptor for the transfer
     * @param callback Callback function to handle the result
     */
    virtual void
    getObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       get_object_callback_t callback) = 0;

    /**
     * Asynchronously get a byte range into host memory over plain HTTP.
     *
     * No RDMA, no registration, no descriptor: the response body is written straight
     * into the caller's buffer. For small reads that is strictly cheaper than the
     * RDMA path, whose cost is dominated by pinning rather than by bytes -- an
     * 8-byte read still needs one ibv_reg_mr per rail. Intended for metadata
     * (safetensors headers, indexes), not for bulk data.
     *
     * A response shorter than data_len succeeds: a range reaching past the end of
     * the object is answered as a complete 206 with fewer bytes. A body cut short
     * against its own Content-Length fails, as does one exceeding data_len.
     *
     * @param key The object key
     * @param dst Host buffer receiving the bytes; must hold data_len
     * @param data_len Bytes requested, and the capacity of dst
     * @param offset Offset within the object to start reading from
     * @param callback Receives true if the range was delivered
     */
    virtual void
    getObjectBodyAsync(std::string_view key,
                       void *dst,
                       size_t data_len,
                       size_t offset,
                       get_object_callback_t callback) = 0;

    /**
     * Asynchronously check whether an object exists (HTTP HEAD).
     * @param key The object key
     * @param callback Receives true (exists), false (404), or std::nullopt (error)
     */
    virtual void
    checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) = 0;
};

/// Upper bound on the default cap for concurrently-running requests.
///
/// Not used as-is: every running request holds a connection and therefore a file
/// descriptor, so the default is also clamped to a share of RLIMIT_NOFILE (see
/// defaultMaxInflight). This value alone assumed the rest of the process would
/// stay under the other half of a 1024 limit, which a caller with many threads
/// and registrations does not -- one was measured at 639 on its own.
constexpr std::size_t kDefaultMaxInflight = 512;

/**
 * Scality AI Connector HTTP client with RDMA support.
 * Uses libcurl to perform HTTP PUT/GET requests, passing the RDMA
 * descriptor via the x-scal-rdma custom header.
 *
 * URL format: {endpoint}/{key}
 */
class RestClient : public iRestClient {
public:
    /**
     * Constructor.
     * @param custom_params Backend init params; must contain "endpoint_override".
     *                      Optional "num_threads" sizes the callback worker pool
     *                      (default: max(2, hardware_concurrency / 4)).
     *                      Optional "max_inflight" caps concurrently-running
     *                      requests ("0" = unlimited). Left unset it defaults to
     *                      min(kDefaultMaxInflight, RLIMIT_NOFILE / 4), since each
     *                      running request holds a descriptor. An explicit value is
     *                      honoured as given and only warned about if it will not
     *                      fit.
     */
    explicit RestClient(nixl_b_params_t *custom_params);

    ~RestClient() override;

    void
    putObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       put_object_callback_t callback) override;

    void
    getObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       get_object_callback_t callback) override;

    void
    getObjectBodyAsync(std::string_view key,
                       void *dst,
                       size_t data_len,
                       size_t offset,
                       get_object_callback_t callback) override;

    void
    checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) override;

private:
    /// Per-request state for an in-flight curl_multi transfer. Defined in client.cpp.
    struct RequestCtx;

    /// Base endpoint, e.g. "http://10.0.0.1:81"
    std::string endpoint_;

    // A single poller thread owns the curl multi handle and drives all transfers;
    // completed callbacks are offloaded to this pool (sized by numThreads_) so a
    // slow callback can't stall the event loop. curl multi handles are not
    // thread-safe: every curl_multi_* call runs on the poller thread, except
    // curl_multi_wakeup() which producers use to nudge it.
    std::size_t numThreads_;
    asio::thread_pool pool_; // callback worker pool

    // Cap on requests running at once. A caller submitting one descriptor per
    // tensor can queue thousands in a single batch, and every running request
    // holds its own connection (and fd), so an uncapped multi handle can exhaust
    // RLIMIT_NOFILE or the endpoint's connection budget. Excess requests wait in
    // pending_ and start as slots free, which keeps the fabric saturated without
    // the fd blow-up. 0 disables the cap.
    std::size_t maxInflight_;

    // Idle easy handles kept for reuse. curl_easy_cleanup() drops the handle's
    // live connection, so a handle-per-request pattern forces a fresh TCP
    // connect every time and exhausts the client's ephemeral port range under
    // load. curl_easy_reset() clears the options but keeps the connection, so
    // recycling handles restores keepalive. Capped so a large submitted batch
    // cannot leave thousands of idle connections parked here.
    std::mutex easyMtx_;
    std::vector<CURL *> easyCache_;
    std::size_t easyCacheCap_;

    CURLM *multi_ = nullptr;
    std::thread poller_;
    std::mutex queueMtx_;
    std::queue<std::unique_ptr<RequestCtx>> incoming_;
    std::atomic<bool> stop_{false};
    /// Handles currently added to multi_. Poller-thread access only (no lock).
    std::unordered_set<RequestCtx *> inflight_;
    /// Built requests waiting for a free in-flight slot. Poller-thread only.
    std::deque<std::unique_ptr<RequestCtx>> pending_;
    // High-water marks, logged at teardown. Without them a throughput sweep
    // cannot tell whether max_inflight was the binding constraint or the producer
    // simply never supplied enough work. Poller-thread only, so no locking.
    std::size_t peakInflight_ = 0;
    std::size_t peakPending_ = 0;
    // Connection-reuse accounting, logged at teardown: new_connections well
    // below requests means the handle cache is doing its job. Poller-thread only.
    std::size_t totalRequests_ = 0;
    std::size_t newConnects_ = 0;
    /// Connection-level failures re-attempted rather than reported to the caller.
    std::size_t totalRetries_ = 0;
    /// Set once the descriptor-exhaustion diagnostic has been emitted, so a retry
    /// storm reports it once rather than per request. Poller-thread only.
    bool fdExhaustionLogged_ = false;
    /// Set when a connect failure was traced to the process being out of file
    /// descriptors. Unlike a refused connection that may clear on its own, this
    /// cannot resolve while the caller keeps its descriptors, so retrying is
    /// pointless and the queued requests cannot succeed either. Latching it
    /// suppresses retries and fails the backlog at once. Poller-thread only.
    bool fdExhausted_ = false;
    /// Requests failed by the backlog drain, reported once at teardown rather
    /// than one log line each. Poller-thread only.
    std::size_t fdAbandoned_ = 0;

    /**
     * Build the full URL for a given key.
     * @param key Object key
     * @return Full URL string
     */
    std::string
    buildUrl(std::string_view key) const;

    /**
     * Build a PUT/GET RDMA request context and hand it to the poller. PUT and GET
     * differ only in the curl method options, so they share this path.
     * @param op_name Operation label for logging (string literal)
     * @param key Object key
     * @param rdma_desc RDMA descriptor for the x-scal-rdma header
     * @param is_upload true for PUT (upload), false for GET
     * @param callback Result callback (invoked on the worker pool)
     */
    void
    submitRdmaRequest(const char *op_name,
                      std::string_view key,
                      std::string_view rdma_desc,
                      bool is_upload,
                      std::function<void(bool)> callback,
                      size_t data_len = 0,
                      size_t offset = 0);

    /// Apply URL + method-specific curl options to a request's easy handle.
    static void
    buildEasy(RequestCtx *ctx);

    /// Take an idle handle from easyCache_, or create one. Any thread.
    CURL *
    acquireEasy();

    /// Reset a finished handle and return it to easyCache_, cleaning it up if the
    /// cache is full. The handle must already be removed from multi_. Any thread.
    void
    releaseEasy(CURL *easy);

    /// Push a fully-built request onto the queue and wake the poller.
    void
    enqueue(std::unique_ptr<RequestCtx> ctx);

    /// Move pending requests into multi_ while in-flight slots are available.
    /// Poller-thread only.
    void
    startPending();

    /// Body of the poller thread: drain queue, perform, reap, poll.
    void
    pollerLoop();

    /// Reap finished transfers (poller thread only) and dispatch their callbacks.
    void
    reapCompletions();

    /// Map a finished transfer to its callback, dispatch it on the worker pool,
    /// and free the context. Poller-thread only.
    void
    finishRequest(RequestCtx *ctx, CURLcode res, long http_code);
};

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_CLIENT_H
