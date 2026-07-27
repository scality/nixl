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

#include "client.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <asio/post.hpp>
#include <curl/curl.h>
#include <dirent.h>
#include <sys/resource.h>
#include <algorithm>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <unordered_set>
#include <utility>

namespace {

size_t
captureBody(void *ptr, size_t size, size_t nmemb, void *userdata) {
    auto *body = static_cast<std::string *>(userdata);
    body->append(static_cast<char *>(ptr), size * nmemb);
    return size * nmemb;
}

std::once_flag curl_init_flag;

/// Extra attempts allowed after a connection-level failure.
constexpr int kMaxRetries = 3;

/// True if libcurl produced no HTTP response at all, so the endpoint either never
/// saw the request or never answered it. A request that did get a response is left
/// alone: a 4xx/5xx is the server's answer, not a lost request.
bool
isTransient(CURLcode res) {
    switch (res) {
    case CURLE_COULDNT_CONNECT:
    case CURLE_COULDNT_RESOLVE_HOST:
    case CURLE_OPERATION_TIMEDOUT:
    case CURLE_SEND_ERROR:
    case CURLE_RECV_ERROR:
    case CURLE_GOT_NOTHING:
        return true;
    default:
        return false;
    }
}

std::size_t
parseNumThreads(nixl_b_params_t *params) {
    if (!params || params->count("num_threads") == 0) {
        return std::max(2u, std::thread::hardware_concurrency() / 4);
    }
    // A zero pool would accept callbacks but never run them, hanging every
    // transfer; reject non-positive / malformed values instead.
    const std::string &value = params->at("num_threads");
    std::size_t consumed = 0;
    const std::size_t parsed = std::stoul(value, &consumed);
    if (consumed != value.size() || parsed == 0) {
        throw std::invalid_argument("RestClient: num_threads must be a positive integer");
    }
    return parsed;
}

std::size_t
parseMaxInflight(nixl_b_params_t *params) {
    if (!params || params->count("max_inflight") == 0) {
        return kDefaultMaxInflight;
    }
    // 0 is meaningful here (uncapped), so unlike num_threads only malformed
    // values are rejected.
    const std::string &value = params->at("max_inflight");
    std::size_t consumed = 0;
    const std::size_t parsed = std::stoul(value, &consumed);
    if (consumed != value.size()) {
        throw std::invalid_argument("RestClient: max_inflight must be a non-negative integer");
    }
    return parsed;
}

/// The errno libcurl saw from the failing syscall, rendered for a log line, or
/// "0" if it recorded none.
///
/// Worth having because CURLcode alone cannot distinguish two very different
/// faults: a refused connection and a process out of file descriptors both
/// surface as CURLE_COULDNT_CONNECT, and they want opposite fixes (lower
/// max_inflight versus raise RLIMIT_NOFILE). EMFILE is 24, ECONNREFUSED 111.
/// strerror is not thread-safe, but every caller is the poller thread.
std::string
osErrnoText(CURL *easy) {
    long err = 0;
    if (easy) {
        curl_easy_getinfo(easy, CURLINFO_OS_ERRNO, &err);
    }
    if (err == 0) {
        return "0";
    }
    return absl::StrFormat("%ld (%s)", err, std::strerror(static_cast<int>(err)));
}

/// libcurl's description of the failure, or the generic CURLcode text when the
/// error buffer was left empty.
std::string
curlErrorText(CURLcode res, const char *error_buf) {
    if (error_buf && error_buf[0] != '\0') {
        return error_buf;
    }
    return curl_easy_strerror(res);
}

/// File descriptors the process currently holds, or 0 if /proc is unavailable.
/// Reading /proc/self/fd holds one descriptor of its own, which is discounted
/// along with "." and "..".
std::size_t
openFdCount() {
    DIR *dir = opendir("/proc/self/fd");
    if (!dir) {
        return 0;
    }
    std::size_t count = 0;
    while (readdir(dir) != nullptr) {
        count++;
    }
    closedir(dir);
    return count > 3 ? count - 3 : 0;
}

} // namespace

enum class restMethod { PUT, GET, HEAD };

// Per-request state for an in-flight curl_multi transfer. Owns its easy handle,
// header list, and response buffer for the full transfer lifetime; the user
// callback (exactly one of the two, by method) is moved out on completion.
struct RestClient::RequestCtx {
    RestClient *client = nullptr;
    CURL *easy = nullptr;
    struct curl_slist *headers = nullptr;
    std::string url;
    std::string response_body;
    const char *op_name = "";
    restMethod method = restMethod::GET;
    int attempts = 0; // retries already spent on a connection-level failure
    std::function<void(bool)> bool_cb; // Put/Get
    std::function<void(std::optional<bool>)> check_cb; // Head
    /// libcurl's own description of the failure. Needed because CURLINFO_OS_ERRNO
    /// is only populated on some paths -- a connect that failed on socket() with
    /// EMFILE and one the peer refused can both arrive with errno unset -- whereas
    /// this buffer carries the reason as text either way. Lives here rather than on
    /// the handle so it survives the handle being recycled.
    char error_buf[CURL_ERROR_SIZE] = {};

    ~RequestCtx() {
        if (headers) {
            curl_slist_free_all(headers);
        }
        if (easy) {
            // Must already be removed from the multi handle by the poller.
            client->releaseEasy(easy);
        }
    }
};

// Apply URL + method-specific options to a clean easy handle (freshly created, or
// reset on its way back into easyCache_). Every option a request depends on must
// be set here, since a recycled handle carries none of its predecessor's. Wire
// format is kept byte-identical to the previous synchronous implementation.
void
RestClient::buildEasy(RequestCtx *ctx) {
    CURL *curl = ctx->easy;
    // Re-set every time: a recycled handle had its options cleared by
    // curl_easy_reset, and a retried request must not report the previous
    // attempt's message.
    ctx->error_buf[0] = '\0';
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, ctx->error_buf);
    curl_easy_setopt(curl, CURLOPT_URL, ctx->url.c_str());
    switch (ctx->method) {
    case restMethod::PUT:
        curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
        curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t)0);
        break;
    case restMethod::GET:
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
        break;
    case restMethod::HEAD:
        curl_easy_setopt(curl, CURLOPT_NOBODY, 1L); // HTTP HEAD
        break;
    }
    if (ctx->headers) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, ctx->headers);
    }
    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, captureBody);
    curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx->response_body);
    curl_easy_setopt(curl, CURLOPT_PRIVATE, ctx);
    // Bound the control-plane request so a stalled endpoint can't leave a
    // transfer in-flight forever (the body is empty; bulk data is on RDMA).
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 5000L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, 30000L);
}

CURL *
RestClient::acquireEasy() {
    {
        const std::lock_guard<std::mutex> lk(easyMtx_);
        if (!easyCache_.empty()) {
            CURL *easy = easyCache_.back();
            easyCache_.pop_back();
            return easy;
        }
    }
    return curl_easy_init();
}

void
RestClient::releaseEasy(CURL *easy) {
    // Reset before caching: it clears every option, including the pointers into
    // the RequestCtx being destroyed, while keeping the handle's live connection.
    curl_easy_reset(easy);
    {
        const std::lock_guard<std::mutex> lk(easyMtx_);
        if (easyCache_.size() < easyCacheCap_) {
            easyCache_.push_back(easy);
            return;
        }
    }
    curl_easy_cleanup(easy);
}

// Map a finished transfer to its callback, dispatch it on the worker pool, and
// free the context. Runs on the poller thread; the posted closure captures only
// the moved callback plus the plain result, so the pool never touches curl state.
// Callbacks are expected not to throw; the dispatch guards against it defensively
// so a misbehaving callback cannot take down a pool worker.
void
RestClient::finishRequest(RequestCtx *ctx, CURLcode res, long http_code) {
    // Read while the handle still holds this transfer's stats; releaseEasy() resets
    // it. 0 means the connection was reused, 1 means a fresh TCP connect.
    if (ctx->easy) {
        long num_connects = 0;
        if (curl_easy_getinfo(ctx->easy, CURLINFO_NUM_CONNECTS, &num_connects) == CURLE_OK) {
            newConnects_ += static_cast<std::size_t>(num_connects);
        }
        ++totalRequests_;
    }

    if (ctx->method == restMethod::HEAD) {
        std::optional<bool> result;
        if (res != CURLE_OK) {
            NIXL_ERROR << absl::StrFormat(
                "checkObjectExistsAsync: curl_code=%d (%s) os_errno=%s for HEAD %s",
                static_cast<int>(res),
                curlErrorText(res, ctx->error_buf),
                osErrnoText(ctx->easy),
                ctx->url);
            result = std::nullopt;
        } else if (http_code >= 200 && http_code < 300) {
            result = true;
        } else if (http_code == 404) {
            result = false;
        } else {
            NIXL_ERROR << absl::StrFormat(
                "checkObjectExistsAsync: HTTP %ld for HEAD %s", http_code, ctx->url);
            result = std::nullopt;
        }
        auto cb = std::move(ctx->check_cb);
        asio::post(pool_, [cb = std::move(cb), result]() {
            try {
                if (cb) {
                    cb(result);
                }
            }
            catch (...) {
            }
        });
    } else {
        bool success = (res == CURLE_OK) && (http_code >= 200 && http_code < 300);
        if (!success) {
            NIXL_ERROR << absl::StrFormat(
                "%s: failed url=%s curl_code=%d (%s) os_errno=%s http_code=%ld body=%s",
                ctx->op_name,
                ctx->url,
                static_cast<int>(res),
                curlErrorText(res, ctx->error_buf),
                osErrnoText(ctx->easy),
                http_code,
                ctx->response_body.empty() ? "<empty>" : ctx->response_body);
        } else {
            NIXL_DEBUG << absl::StrFormat(
                "%s: success url=%s http_code=%ld", ctx->op_name, ctx->url, http_code);
        }
        auto cb = std::move(ctx->bool_cb);
        asio::post(pool_, [cb = std::move(cb), success]() {
            try {
                if (cb) {
                    cb(success);
                }
            }
            catch (...) {
            }
        });
    }
    delete ctx;
}

RestClient::RestClient(nixl_b_params_t *custom_params)
    : numThreads_(parseNumThreads(custom_params)),
      pool_(numThreads_),
      maxInflight_(parseMaxInflight(custom_params)),
      // At most maxInflight_ handles can be transferring at once, so a larger
      // cache would only hold connections nothing is waiting to use.
      easyCacheCap_(maxInflight_ == 0 ? kDefaultMaxInflight : maxInflight_) {
    std::call_once(curl_init_flag, []() { curl_global_init(CURL_GLOBAL_DEFAULT); });
    if (!custom_params) {
        throw std::invalid_argument("RestClient: custom_params is null");
    }

    auto ep_it = custom_params->find("endpoint_override");
    if (ep_it == custom_params->end() || ep_it->second.empty()) {
        throw std::invalid_argument("RestClient: 'endpoint_override' parameter is required");
    }
    endpoint_ = ep_it->second;

    // Create the multi handle and start the poller only after validation, so a
    // throwing constructor leaves no thread or handle behind.
    multi_ = curl_multi_init();
    if (!multi_) {
        throw std::runtime_error("RestClient: curl_multi_init failed");
    }
    // Left unset, libcurl sizes the connection cache from the number of easy
    // handles currently added to the multi. Handles are removed as soon as they
    // complete, so that count tracks in-flight requests, not the connections
    // worth keeping warm, and a new transfer evicts an idle keep-alive socket to
    // make room. The eviction is a client-side close, which is what burns
    // ephemeral ports. Size the cache to the in-flight cap instead.
    curl_multi_setopt(multi_, CURLMOPT_MAXCONNECTS, static_cast<long>(easyCacheCap_));
    poller_ = std::thread(&RestClient::pollerLoop, this);

    NIXL_INFO << absl::StrFormat(
        "RestClient initialized: endpoint=%s, callback_threads=%zu, max_inflight=%s "
        "(curl_multi poller)",
        endpoint_,
        numThreads_,
        maxInflight_ == 0 ? std::string("unlimited") : std::to_string(maxInflight_));

    // Every running request holds a connection, so max_inflight is a file
    // descriptor requirement as much as a concurrency setting. Report the budget:
    // when it is short, requests fail at connect with EMFILE, which looks exactly
    // like an endpoint refusing them and sends debugging in the wrong direction.
    //
    // The open count is only a floor. This runs at backend-creation time, before
    // the caller registers memory or opens its fabric contexts, so the eventual
    // baseline is higher -- on one 128-thread benchmark it reached 639.
    rlimit lim{};
    const bool have_limit = getrlimit(RLIMIT_NOFILE, &lim) == 0;
    const std::size_t open_now = openFdCount();
    if (have_limit && lim.rlim_cur != RLIM_INFINITY) {
        NIXL_INFO << absl::StrFormat(
            "RestClient fd budget: open=%zu (so far), soft_limit=%llu, "
            "max_inflight=%zu, headroom=%lld",
            open_now,
            static_cast<unsigned long long>(lim.rlim_cur),
            maxInflight_,
            static_cast<long long>(lim.rlim_cur) - static_cast<long long>(open_now) -
                static_cast<long long>(maxInflight_));
        if (maxInflight_ != 0 && open_now + maxInflight_ > lim.rlim_cur) {
            NIXL_WARN << absl::StrFormat(
                "RestClient: max_inflight=%zu plus %zu descriptors already open "
                "exceeds RLIMIT_NOFILE=%llu. Requests will fail at connect with "
                "EMFILE and be reported as curl_code=7, indistinguishable from a "
                "refused connection. Lower max_inflight or raise 'ulimit -n'.",
                maxInflight_,
                open_now,
                static_cast<unsigned long long>(lim.rlim_cur));
        }
    } else {
        NIXL_INFO << absl::StrFormat(
            "RestClient fd budget: open=%zu (so far), soft_limit=unlimited, "
            "max_inflight=%zu",
            open_now,
            maxInflight_);
    }
}

RestClient::~RestClient() {
    stop_.store(true);
    if (multi_) {
        curl_multi_wakeup(multi_); // break the poller out of curl_multi_poll
    }
    if (poller_.joinable()) {
        poller_.join(); // poller fails any outstanding requests before returning
    }
    pool_.join(); // drain queued callbacks
    // The poller returned every handle to the cache; close them before the multi
    // handle that owns their connections goes away.
    for (CURL *easy : easyCache_) {
        curl_easy_cleanup(easy);
    }
    easyCache_.clear();
    if (multi_) {
        curl_multi_cleanup(multi_);
    }
    // At INFO because it answers the first question of any throughput tuning:
    // whether the endpoint was kept busy. peak_pending > 0 means requests were
    // still waiting after a fill pass, so max_inflight was the binding constraint;
    // peak_inflight well below the cap means the caller never submitted enough to
    // saturate anything.
    NIXL_INFO << "RestClient concurrency: peak_inflight=" << peakInflight_
              << ", peak_pending=" << peakPending_ << ", max_inflight="
              << (maxInflight_ == 0 ? std::string("unlimited") : std::to_string(maxInflight_));
    NIXL_INFO << "RestClient connections: requests=" << totalRequests_
              << ", new_connections=" << newConnects_ << ", retries=" << totalRetries_
              << " (new_connections close to requests means keepalive is not working and "
                 "the endpoint will exhaust ephemeral ports)";
}

std::string
RestClient::buildUrl(std::string_view key) const {
    return absl::StrFormat("%s/%s", endpoint_, key);
}

void
RestClient::enqueue(std::unique_ptr<RequestCtx> ctx) {
    {
        const std::lock_guard<std::mutex> lk(queueMtx_);
        incoming_.push(std::move(ctx));
    }
    curl_multi_wakeup(multi_); // thread-safe; nudges the poller to drain the queue
}

void
RestClient::reapCompletions() {
    CURLMsg *msg = nullptr;
    int in_queue = 0;
    while ((msg = curl_multi_info_read(multi_, &in_queue)) != nullptr) {
        if (msg->msg != CURLMSG_DONE) {
            continue;
        }
        CURL *easy = msg->easy_handle;
        CURLcode res = msg->data.result;
        RequestCtx *ctx = nullptr;
        curl_easy_getinfo(easy, CURLINFO_PRIVATE, &ctx);
        long http_code = 0;
        curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &http_code);

        curl_multi_remove_handle(multi_, easy);
        inflight_.erase(ctx);

        // Report the descriptor budget on the first connect failure of any kind.
        // Not gated on the errno being EMFILE: libcurl leaves CURLINFO_OS_ERRNO
        // unset on several paths, so a genuine descriptor exhaustion can arrive
        // with errno 0 and would never be recognised. The count itself is the
        // evidence -- close to the limit means local, far from it means the
        // endpoint -- and now is the only moment it means anything: at
        // construction the caller has not yet registered memory or opened its
        // fabric contexts, and by teardown every connection is closed. Sampled
        // once so a retry storm cannot turn it into a log flood.
        if (res == CURLE_COULDNT_CONNECT && !fdExhaustionLogged_) {
            fdExhaustionLogged_ = true;
            rlimit lim{};
            const bool have_limit = getrlimit(RLIMIT_NOFILE, &lim) == 0;
            const std::size_t open_now = openFdCount();
            NIXL_ERROR << absl::StrFormat(
                "RestClient: first connect failure. fds open=%zu, soft_limit=%s, "
                "max_inflight=%zu, curl says '%s'. Every running request holds a "
                "connection, so open close to the limit means the cap does not fit "
                "in the descriptors left after the caller's own use -- lower "
                "max_inflight or raise 'ulimit -n'. Open well below the limit points "
                "at the endpoint instead. Further connect failures are not logged.",
                open_now,
                (have_limit && lim.rlim_cur != RLIM_INFINITY) ?
                    std::to_string(lim.rlim_cur) :
                    std::string("unlimited"),
                maxInflight_,
                curlErrorText(res, ctx->error_buf));
        }

        // Retry a lost request rather than failing the caller's whole transfer for
        // one blip. It goes to the back of pending_, so the requests already queued
        // are attempted before it comes round again, and the easy handle keeps every
        // option it was built with: only the partial response body has to go.
        //
        // GET and HEAD only. A PUT whose response was lost may well have been
        // applied, and the endpoint versions objects and answers a second write of
        // the same key with 409 "cannot overwrite", so retrying turns a transfer
        // that had succeeded into a failed one.
        if (ctx->method != restMethod::PUT && isTransient(res) && ctx->attempts < kMaxRetries) {
            ctx->attempts++;
            ctx->response_body.clear();
            totalRetries_++;
            // buildEasy is not called again, so clear this here or the next attempt
            // reports whatever the last one failed with.
            ctx->error_buf[0] = '\0';
            NIXL_WARN << absl::StrFormat("%s: curl_code=%d (%s) os_errno=%s url=%s, retry %d/%d",
                                         ctx->op_name,
                                         static_cast<int>(res),
                                         curlErrorText(res, ctx->error_buf),
                                         osErrnoText(ctx->easy),
                                         ctx->url,
                                         ctx->attempts,
                                         kMaxRetries);
            pending_.emplace_back(ctx);
            continue;
        }
        finishRequest(ctx, res, http_code);
    }
}

void
RestClient::startPending() {
    while (!pending_.empty() && (maxInflight_ == 0 || inflight_.size() < maxInflight_)) {
        std::unique_ptr<RequestCtx> ctx = std::move(pending_.front());
        pending_.pop_front();
        RequestCtx *raw = ctx.get();
        CURLMcode mc = curl_multi_add_handle(multi_, raw->easy);
        if (mc != CURLM_OK) {
            NIXL_ERROR << absl::StrFormat(
                "%s: curl_multi_add_handle failed: %s", raw->op_name, curl_multi_strerror(mc));
            finishRequest(ctx.release(), CURLE_FAILED_INIT, 0);
            continue;
        }
        ctx.release(); // ownership tracked via CURLOPT_PRIVATE until completion
        inflight_.insert(raw);
        peakInflight_ = std::max(peakInflight_, inflight_.size());
    }
    // Measured on the way out, so it counts only what the in-flight cap actually
    // held back. Sampling before the loop instead counts a burst of arrivals that
    // this very call is about to start, which reads as cap pressure that never
    // happened.
    peakPending_ = std::max(peakPending_, pending_.size());
}

void
RestClient::pollerLoop() {
    for (;;) {
        const bool stopping = stop_.load();

        // 1. Drain the producer queue into pending_. On shutdown, fail queued
        //    requests instead of starting them.
        std::queue<std::unique_ptr<RequestCtx>> batch;
        {
            const std::lock_guard<std::mutex> lk(queueMtx_);
            std::swap(batch, incoming_);
        }
        while (!batch.empty()) {
            std::unique_ptr<RequestCtx> ctx = std::move(batch.front());
            batch.pop();
            if (stopping) {
                finishRequest(ctx.release(), CURLE_ABORTED_BY_CALLBACK, 0);
                continue;
            }
            pending_.push_back(std::move(ctx));
        }

        // 2. Fill the in-flight slots from pending_.
        startPending();

        // 3. Advance all in-flight transfers (non-blocking).
        int running = 0;
        curl_multi_perform(multi_, &running);

        // 4. Hand finished transfers' callbacks to the worker pool.
        reapCompletions();

        // 5. Completions above freed slots; refill before sleeping so a capped
        //    queue does not stall waiting for the next socket event.
        if (!stopping) {
            startPending();
        }

        // 6. On shutdown, abort everything queued or in flight so every callback
        //    fires exactly once.
        if (stopping) {
            for (RequestCtx *ctx : inflight_) {
                curl_multi_remove_handle(multi_, ctx->easy);
                finishRequest(ctx, CURLE_ABORTED_BY_CALLBACK, 0);
            }
            inflight_.clear();
            while (!pending_.empty()) {
                std::unique_ptr<RequestCtx> ctx = std::move(pending_.front());
                pending_.pop_front();
                finishRequest(ctx.release(), CURLE_ABORTED_BY_CALLBACK, 0);
            }
            break;
        }

        // 7. Block until socket activity, the 1s backstop, or curl_multi_wakeup().
        //    Requests still in pending_ imply the in-flight set is full, so a
        //    completion (socket event) is what frees the next slot.
        int numfds = 0;
        curl_multi_poll(multi_, nullptr, 0, 1000, &numfds);
    }
}

void
RestClient::submitRdmaRequest(const char *op_name,
                              std::string_view key,
                              std::string_view rdma_desc,
                              bool is_upload,
                              std::function<void(bool)> callback,
                              size_t data_len,
                              size_t offset) {
    auto ctx = std::make_unique<RequestCtx>();
    ctx->client = this;
    ctx->op_name = op_name;
    ctx->method = is_upload ? restMethod::PUT : restMethod::GET;
    ctx->url = buildUrl(key);
    ctx->bool_cb = std::move(callback);

    ctx->easy = acquireEasy();
    if (!ctx->easy) {
        NIXL_ERROR << absl::StrFormat("%s: curl_easy_init failed", op_name);
        if (ctx->bool_cb) {
            ctx->bool_cb(false);
        }
        return;
    }

    std::string rdma_header = absl::StrFormat("x-scal-rdma: %s", rdma_desc);
    ctx->headers = curl_slist_append(ctx->headers, rdma_header.c_str());
    if (is_upload) {
        // Content-Length: 0; data is transferred via RDMA, not the HTTP body.
        ctx->headers = curl_slist_append(ctx->headers, "Content-Length: 0");
    } else if (data_len > 0) {
        // Convey the object byte-range to sproxyd so biziod RDMA-writes
        // object[offset : offset+data_len] into the local buffer. Always sent, offset 0
        // included: the caller registered exactly data_len bytes, so an unranged GET
        // would have the server fetch the whole object (an 8-byte safetensors header
        // probe would pull a multi-GB shard). Same as the s3_accel clients.
        std::string range_header =
            absl::StrFormat("Range: bytes=%zu-%zu", offset, offset + data_len - 1);
        ctx->headers = curl_slist_append(ctx->headers, range_header.c_str());
    }

    buildEasy(ctx.get());
    enqueue(std::move(ctx));
}

void
RestClient::putObjectRdmaAsync(std::string_view key,
                               uintptr_t data_ptr,
                               size_t data_len,
                               size_t offset,
                               std::string_view rdma_desc,
                               put_object_callback_t callback) {
    // The RDMA descriptor is sensitive transfer-capability metadata; log only
    // its length, never the raw token.
    NIXL_DEBUG << absl::StrFormat(
        "putObjectRdmaAsync: key=%s, data_ptr=%p, data_len=%zu, offset=%zu, rdma_desc_len=%zu",
        key,
        reinterpret_cast<void *>(data_ptr),
        data_len,
        offset,
        rdma_desc.size());

    if (data_len == 0) {
        NIXL_ERROR << "putObjectRdmaAsync: data_len is 0, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    if (rdma_desc.empty()) {
        NIXL_ERROR << "putObjectRdmaAsync: rdma_desc is empty, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    submitRdmaRequest(
        "putObjectRdmaAsync", key, rdma_desc, /*is_upload=*/true, std::move(callback));
}

void
RestClient::getObjectRdmaAsync(std::string_view key,
                               uintptr_t data_ptr,
                               size_t data_len,
                               size_t offset,
                               std::string_view rdma_desc,
                               get_object_callback_t callback) {
    // Log only the descriptor length, never the raw RDMA token (sensitive).
    NIXL_DEBUG << absl::StrFormat(
        "getObjectRdmaAsync: key=%s, data_ptr=%p, data_len=%zu, offset=%zu, rdma_desc_len=%zu",
        key,
        reinterpret_cast<void *>(data_ptr),
        data_len,
        offset,
        rdma_desc.size());

    if (data_len == 0) {
        NIXL_ERROR << "getObjectRdmaAsync: data_len is 0, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    if ((data_len > 0) && (offset > (SIZE_MAX - (data_len - 1)))) {
        NIXL_ERROR << "getObjectRdmaAsync: offset + data_len would overflow, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    if (rdma_desc.empty()) {
        NIXL_ERROR << "getObjectRdmaAsync: rdma_desc is empty, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    submitRdmaRequest(
        "getObjectRdmaAsync", key, rdma_desc, /*is_upload=*/false, std::move(callback),
        data_len, offset);
}

void
RestClient::checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) {
    auto ctx = std::make_unique<RequestCtx>();
    ctx->client = this;
    ctx->op_name = "checkObjectExistsAsync";
    ctx->method = restMethod::HEAD;
    ctx->url = buildUrl(key);
    ctx->check_cb = std::move(callback);

    ctx->easy = acquireEasy();
    if (!ctx->easy) {
        NIXL_ERROR << "checkObjectExistsAsync: curl_easy_init failed";
        if (ctx->check_cb) {
            ctx->check_cb(std::nullopt);
        }
        return;
    }

    buildEasy(ctx.get());
    enqueue(std::move(ctx));
}
