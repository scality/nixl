/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "rest_client.h"
#include "common/config_traits.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <asio/post.hpp>
#include <curl/curl.h>
#include <dirent.h>
#include <sys/resource.h>
#include <algorithm>
#include <cerrno>
#include <cstring>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <thread>
#include <utility>

enum class rest_method { PUT, GET, HEAD, POST };

namespace {

/// A response body is only error text for the failure log, so only its start is kept.
constexpr size_t max_captured_body = 4096;

size_t
captureBody(void *ptr, size_t size, size_t nmemb, void *userdata) {
    auto *body = static_cast<std::string *>(userdata);
    const size_t n = size * nmemb;
    if (body->size() < max_captured_body) {
        body->append(static_cast<char *>(ptr), std::min(n, max_captured_body - body->size()));
    }
    return n;
}

/// A .batch_exists answer holds a "NNN\n" line per key. One byte more is kept so
/// that a longer body fails to parse instead of being read as a complete answer.
constexpr size_t max_captured_exists_answer = max_keys_per_exists_batch * 4 + 1;

size_t
captureExistsAnswer(void *ptr, size_t size, size_t nmemb, void *userdata) {
    auto *body = static_cast<std::string *>(userdata);
    const size_t n = size * nmemb;
    if (body->size() < max_captured_exists_answer) {
        body->append(static_cast<char *>(ptr),
                     std::min(n, max_captured_exists_answer - body->size()));
    }
    return n;
}

/// True if a key can go in a .batch_exists text body: one key per line, and
/// sproxyd refuses the whole batch for a key starting with '.'.
bool
fitsExistsBatch(std::string_view key) {
    constexpr std::string_view line_breaks("\r\n\0", 3);
    return !key.empty() && key.front() != '.' && key.find_first_of(line_breaks) == key.npos;
}

/// The answer per key of a .batch_exists text answer: 200 exists, 404 does not,
/// any other status is an error. nullopt if it doesn't hold one line per key.
std::optional<std::vector<std::optional<bool>>>
parseExistsAnswer(std::string_view body, size_t count) {
    std::vector<std::optional<bool>> exists;
    exists.reserve(count);
    while (!body.empty()) {
        const size_t eol = body.find('\n');
        if (eol == body.npos) {
            return std::nullopt;
        }
        const std::string_view line = body.substr(0, eol);
        body.remove_prefix(eol + 1);
        if (line == "200") {
            exists.emplace_back(true);
        } else if (line == "404") {
            exists.emplace_back(false);
        } else if (line.size() == 3 &&
                   std::all_of(line.begin(), line.end(), [](char c) { return c >= '0' && c <= '9'; })) {
            exists.emplace_back(std::nullopt);
        } else {
            return std::nullopt;
        }
    }
    if (exists.size() != count) {
        return std::nullopt;
    }
    return exists;
}

/// Destination of a plain-HTTP body read: the caller's buffer and its capacity.
struct httpBodySink {
    char *dst = nullptr;
    size_t cap = 0;
    size_t len = 0; ///< bytes written so far
    bool overflow = false;
    /// The range starts past offset 0, so only a 206 carries the requested bytes.
    bool ranged = false;
    /// Used to read the response status: error bodies are not written to dst.
    CURL *easy = nullptr;
    /// Where an error response's body goes instead of the caller's buffer.
    std::string *errorBody = nullptr;
};

/// Write callback for a body read, bounded by the caller's capacity: a server
/// that ignores the Range header would otherwise overrun the buffer. Returning
/// short aborts the transfer with CURLE_WRITE_ERROR.
size_t
writeToSink(void *ptr, size_t size, size_t nmemb, void *userdata) {
    auto *sink = static_cast<httpBodySink *>(userdata);
    const size_t n = size * nmemb;

    // An error response carries a message, not data: keep it for the failure log
    // instead of writing it into the caller's buffer.
    long http_code = 0;
    if (sink->easy != nullptr &&
        curl_easy_getinfo(sink->easy, CURLINFO_RESPONSE_CODE, &http_code) == CURLE_OK &&
        http_code != 0 && (http_code < 200 || http_code >= 300)) {
        if (sink->errorBody != nullptr) {
            captureBody(ptr, size, nmemb, sink->errorBody);
        }
        return n;
    }

    // A ranged read answered with the object from offset 0: abort instead of writing it.
    if (sink->ranged && http_code != 0 && http_code != 206) {
        return 0;
    }

    if (n > sink->cap - sink->len) {
        sink->overflow = true;
        return 0;
    }
    std::memcpy(sink->dst + sink->len, ptr, n);
    sink->len += n;
    return n;
}

/// The object key as a URL path: bytes outside the RFC 3986 unreserved set are
/// percent-encoded, '/' is kept.
std::string
encodeKey(std::string_view key) {
    std::string out;
    out.reserve(key.size());
    for (const unsigned char c : key) {
        const bool keep = (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') ||
            (c >= '0' && c <= '9') || c == '-' || c == '_' || c == '.' || c == '~' || c == '/';
        if (keep) {
            out += static_cast<char>(c);
        } else {
            out += absl::StrFormat("%%%02X", c);
        }
    }
    return out;
}

std::once_flag curl_init_flag;

/// A customParams integer, or nullopt when it is not one: unlike std::stoul,
/// NIXL's converter rejects a sign, spaces and trailing text.
std::optional<std::size_t>
parseSize(const std::string &value) {
    try {
        return nixl::config::configTraits<std::size_t>::convert(value);
    }
    catch (const std::runtime_error &) {
        return std::nullopt;
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
    const std::optional<std::size_t> parsed = parseSize(value);
    if (!parsed || *parsed == 0) {
        throw std::invalid_argument("restClient: num_threads must be a positive integer");
    }
    return *parsed;
}

std::size_t
parseRequestTimeoutMs(nixl_b_params_t *params) {
    if (!params || params->count("request_timeout_ms") == 0) {
        return default_request_timeout_ms;
    }
    const std::string &value = params->at("request_timeout_ms");
    const std::optional<std::size_t> parsed = parseSize(value);
    if (!parsed || *parsed == 0) {
        throw std::invalid_argument("restClient: request_timeout_ms must be a positive integer");
    }
    return *parsed;
}

/// True if the caller set max_inflight rather than leaving it to us.
bool
hasMaxInflight(nixl_b_params_t *params) {
    return params && params->count("max_inflight") != 0;
}

/// Share of RLIMIT_NOFILE the default cap may claim. The rest is left to the
/// caller, whose threads, registrations and fabric contexts are opened after this
/// client is built and cannot be counted here.
constexpr std::size_t fd_share_divisor = 4;

/// The cap of each class when the caller did not choose one: default_max_inflight,
/// scaled down to the descriptor limit actually in force, shared by the classes.
std::size_t
defaultMaxInflight() {
    rlimit lim{};
    std::size_t total = default_max_inflight;
    if (getrlimit(RLIMIT_NOFILE, &lim) == 0 && lim.rlim_cur != RLIM_INFINITY) {
        total = std::min(total, static_cast<std::size_t>(lim.rlim_cur) / fd_share_divisor);
    }
    return std::max<std::size_t>(1, total / n_request_classes);
}

request_class
requestClassOf(rest_method method) {
    switch (method) {
    case rest_method::PUT:
        return request_class::WRITE;
    case rest_method::GET:
        return request_class::READ;
    case rest_method::HEAD:
    case rest_method::POST:
        break;
    }
    return request_class::CONTROL;
}

const char *
requestClassName(std::size_t cls) {
    switch (static_cast<request_class>(cls)) {
    case request_class::CONTROL:
        return "control";
    case request_class::READ:
        return "read";
    case request_class::WRITE:
        return "write";
    }
    return "?";
}

std::size_t
parseMaxInflight(nixl_b_params_t *params) {
    if (!hasMaxInflight(params)) {
        return defaultMaxInflight();
    }
    // 0 means uncapped. An explicit value is honoured as given; the constructor
    // warns when it does not fit the descriptor limit.
    const std::string &value = params->at("max_inflight");
    const std::optional<std::size_t> parsed = parseSize(value);
    if (!parsed) {
        throw std::invalid_argument("restClient: max_inflight must be a non-negative integer");
    }
    return *parsed;
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

/// The errno libcurl saw from the failing syscall, or "0" if it recorded none.
/// It tells a refused connection (ECONNREFUSED) from descriptor exhaustion
/// (EMFILE), which both surface as CURLE_COULDNT_CONNECT. strerror is not
/// thread-safe, but every caller is the poller thread.
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

/// File descriptors the process currently holds, or nullopt with errno set if the
/// count could not be taken. "." and ".." and our own directory handle are
/// discounted.
std::optional<std::size_t>
openFdCount() {
    DIR *dir = opendir("/proc/self/fd");
    if (!dir) {
        return std::nullopt;
    }
    std::size_t count = 0;
    while (readdir(dir) != nullptr) {
        count++;
    }
    closedir(dir);
    return count > 3 ? count - 3 : 0;
}

/// The descriptor count rendered for a log line. Counting needs a descriptor of
/// its own, so when that fails with EMFILE, say so: it is itself the answer.
std::string
openFdCountText() {
    const int saved = errno;
    errno = 0;
    const std::optional<std::size_t> count = openFdCount();
    const int err = errno;
    errno = saved;
    if (count) {
        return std::to_string(*count);
    }
    if (err == EMFILE || err == ENFILE) {
        return absl::StrFormat("unreadable: %s; taking the count hit the limit too, "
                               "which is itself the answer",
                               std::strerror(err));
    }
    return absl::StrFormat("unreadable (%s)", std::strerror(err));
}

/// True if the process has no descriptor to spare: the count could not even be
/// taken, or it leaves no room for another connection.
bool
descriptorsExhausted() {
    rlimit lim{};
    const bool have_limit = getrlimit(RLIMIT_NOFILE, &lim) == 0;
    const std::optional<std::size_t> open_now = openFdCount();
    return !open_now.has_value() ||
        (have_limit && lim.rlim_cur != RLIM_INFINITY && *open_now + 1 >= lim.rlim_cur);
}

} // namespace

// Per-request state for an in-flight curl_multi transfer. Owns its easy handle,
// header list, and response buffer for the full transfer lifetime; the user
// callback (exactly one of the two, by method) is moved out on completion.
struct restClient::requestCtx {
    restClient *client = nullptr;
    CURL *easy = nullptr;
    struct curl_slist *headers = nullptr;
    std::string url;
    std::string responseBody;
    const char *opName = "";
    rest_method method = rest_method::GET;
    std::function<void(bool)> boolCb; // Put/Get
    std::function<void(std::optional<bool>)> checkCb; // Head
    check_objects_callback_t batchCb; // Post .batch_exists
    /// The keys of a .batch_exists, kept to check them one by one if it is refused.
    std::vector<std::string> batchKeys;
    std::string requestBody;
    /// Set only for a plain-HTTP body read. dst == nullptr means the body is error
    /// text bound for responseBody, as it is for every RDMA and HEAD request.
    httpBodySink sink;
    /// libcurl's own description of the failure, which carries the reason as
    /// text even on paths where CURLINFO_OS_ERRNO stays unset.
    char errorBuf[CURL_ERROR_SIZE] = {};
    /// A 416 (range past the end of the object) means nothing to read: success.
    bool pastEndOk = false;
    /// Builds the x-scal-rdma descriptor when the request starts; unset for HEAD
    /// and body reads.
    rdma_descriptor_fn_t rdmaDesc;

    ~requestCtx() {
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
// be set here, since a recycled handle carries none of its predecessor's.
void
restClient::buildEasy(requestCtx *ctx) const {
    CURL *curl = ctx->easy;
    ctx->errorBuf[0] = '\0';
    curl_easy_setopt(curl, CURLOPT_ERRORBUFFER, ctx->errorBuf);
    curl_easy_setopt(curl, CURLOPT_URL, ctx->url.c_str());
    // Keep '.' and '..' segments: they are part of the key.
    curl_easy_setopt(curl, CURLOPT_PATH_AS_IS, 1L);
    switch (ctx->method) {
    case rest_method::PUT:
        curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
        curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t)0);
        break;
    case rest_method::GET:
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
        break;
    case rest_method::HEAD:
        curl_easy_setopt(curl, CURLOPT_NOBODY, 1L);
        break;
    case rest_method::POST:
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, ctx->requestBody.data());
        curl_easy_setopt(
            curl, CURLOPT_POSTFIELDSIZE_LARGE, static_cast<curl_off_t>(ctx->requestBody.size()));
        break;
    }
    if (ctx->headers) {
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, ctx->headers);
    }
    if (ctx->method == rest_method::POST) {
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, captureExistsAnswer);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx->responseBody);
    } else if (ctx->sink.dst != nullptr) {
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeToSink);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx->sink);
    } else {
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, captureBody);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &ctx->responseBody);
    }
    curl_easy_setopt(curl, CURLOPT_PRIVATE, ctx);
    // No signals from libcurl: this process is multi-threaded.
    curl_easy_setopt(curl, CURLOPT_NOSIGNAL, 1L);
    curl_easy_setopt(curl, CURLOPT_CONNECTTIMEOUT_MS, 1000L);
    curl_easy_setopt(curl, CURLOPT_TIMEOUT_MS, static_cast<long>(requestTimeoutMs_));
}

CURL *
restClient::acquireEasy() {
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
restClient::releaseEasy(CURL *easy) {
    // Reset before caching: it clears every option, including the pointers into
    // the requestCtx being destroyed, while keeping the handle's live connection.
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

// The posted closure captures only the callback and the result, so the pool never
// touches curl state.
void
restClient::finishRequest(requestCtx *ctx, CURLcode res, long http_code) {
    // Read while the handle still holds this transfer's stats; releaseEasy() resets
    // it. 0 means the connection was reused, 1 means a fresh TCP connect.
    if (ctx->easy) {
        long num_connects = 0;
        if (curl_easy_getinfo(ctx->easy, CURLINFO_NUM_CONNECTS, &num_connects) == CURLE_OK) {
            newConnects_ += static_cast<std::size_t>(num_connects);
        }
        ++totalRequests_;
    }

    if (ctx->method == rest_method::POST) {
        finishExistsBatch(ctx, res, http_code);
    } else if (ctx->method == rest_method::HEAD) {
        std::optional<bool> result;
        if (res != CURLE_OK) {
            NIXL_ERROR << absl::StrFormat(
                "checkObjectExistsAsync: curl_code=%d (%s) os_errno=%s for HEAD %s",
                static_cast<int>(res),
                curlErrorText(res, ctx->errorBuf),
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
        auto cb = std::move(ctx->checkCb);
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
        if (!success && ctx->pastEndOk && res == CURLE_OK && http_code == 416) {
            NIXL_TRACE << absl::StrFormat("%s: nothing past the end of %s", ctx->opName, ctx->url);
            success = true;
        }
        // A body shorter than requested is a success: a range past the end of the
        // object is answered with fewer bytes. A body cut short against its own
        // Content-Length has already failed as CURLE_PARTIAL_FILE.
        if (ctx->sink.dst != nullptr) {
            if (ctx->sink.ranged && http_code >= 200 && http_code < 300 && http_code != 206) {
                // A 200 here is the object from offset 0: the Range was ignored.
                NIXL_ERROR << absl::StrFormat(
                    "%s: got HTTP %ld instead of 206 for a range of %s; the endpoint "
                    "appears to have ignored the Range header",
                    ctx->opName,
                    http_code,
                    ctx->url);
                success = false;
            } else if (ctx->sink.overflow) {
                NIXL_ERROR << absl::StrFormat(
                    "%s: response exceeds the %zu-byte request for %s (http_code=%ld); the "
                    "endpoint appears to have ignored the Range header",
                    ctx->opName,
                    ctx->sink.cap,
                    ctx->url,
                    http_code);
                success = false;
            } else if (success && ctx->sink.len < ctx->sink.cap) {
                NIXL_TRACE << absl::StrFormat(
                    "%s: object ended after %zu of %zu requested bytes for %s",
                    ctx->opName,
                    ctx->sink.len,
                    ctx->sink.cap,
                    ctx->url);
            }
        }
        // Once descriptor exhaustion is established, further connect failures are
        // counted and reported at teardown instead of logged one by one.
        const bool redundant = fdExhausted_ && res == CURLE_COULDNT_CONNECT;
        if (!success && !redundant) {
            NIXL_ERROR << absl::StrFormat(
                "%s: failed url=%s curl_code=%d (%s) os_errno=%s http_code=%ld body=%s",
                ctx->opName,
                ctx->url,
                static_cast<int>(res),
                curlErrorText(res, ctx->errorBuf),
                osErrnoText(ctx->easy),
                http_code,
                ctx->responseBody.empty() ? "<empty>" : ctx->responseBody);
        } else {
            NIXL_TRACE << absl::StrFormat(
                "%s: success url=%s http_code=%ld", ctx->opName, ctx->url, http_code);
        }
        auto cb = std::move(ctx->boolCb);
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

void
restClient::finishExistsBatch(requestCtx *ctx, CURLcode res, long http_code) {
    const bool answered = res == CURLE_OK;
    // A refused batch is checked again with one HEAD per key. Not while stopping:
    // the poller would exit before it starts them.
    if (answered && http_code == 400 && !stop_.load()) {
        if (ctx->responseBody.find("unsupported operation") != std::string::npos) {
            NIXL_INFO << absl::StrFormat(
                "checkObjectsExistAsync: %s doesn't support .batch_exists; checking each key "
                "with HEAD from now on",
                endpoint_);
            batchExistsUnsupported_.store(true);
        } else {
            NIXL_ERROR << absl::StrFormat(
                "checkObjectsExistAsync: %s refused a batch of %zu keys, checking them with "
                "HEAD: %s",
                ctx->url,
                ctx->batchKeys.size(),
                ctx->responseBody);
        }
        iRestClient::checkObjectsExistAsync(std::move(ctx->batchKeys), std::move(ctx->batchCb));
        return;
    }

    std::optional<std::vector<std::optional<bool>>> exists;
    if (!answered) {
        NIXL_ERROR << absl::StrFormat(
            "checkObjectsExistAsync: curl_code=%d (%s) os_errno=%s for POST %s",
            static_cast<int>(res),
            curlErrorText(res, ctx->errorBuf),
            osErrnoText(ctx->easy),
            ctx->url);
    } else if (http_code != 200) {
        NIXL_ERROR << absl::StrFormat("checkObjectsExistAsync: HTTP %ld for POST %s: %s",
                                      http_code,
                                      ctx->url,
                                      ctx->responseBody);
    } else {
        exists = parseExistsAnswer(ctx->responseBody, ctx->batchKeys.size());
        if (!exists) {
            NIXL_ERROR << absl::StrFormat(
                "checkObjectsExistAsync: POST %s did not answer one status per key for %zu "
                "keys: %s",
                ctx->url,
                ctx->batchKeys.size(),
                ctx->responseBody);
        }
    }
    if (!exists) {
        exists.emplace(ctx->batchKeys.size(), std::nullopt);
    }
    asio::post(pool_, [cb = std::move(ctx->batchCb), exists = std::move(*exists)]() mutable {
        try {
            if (cb) {
                cb(std::move(exists));
            }
        }
        catch (...) {
        }
    });
}

restClient::restClient(nixl_b_params_t *custom_params)
    : numThreads_(parseNumThreads(custom_params)),
      pool_(numThreads_),
      requestTimeoutMs_(parseRequestTimeoutMs(custom_params)),
      maxInflight_(parseMaxInflight(custom_params)),
      // At most maxInflight_ handles per class can be transferring at once, so a
      // larger cache would only hold connections nothing is waiting to use.
      easyCacheCap_(maxInflight_ == 0 ? default_max_inflight :
                                        n_request_classes * maxInflight_) {
    std::call_once(curl_init_flag, []() { curl_global_init(CURL_GLOBAL_DEFAULT); });
    if (!custom_params) {
        throw std::invalid_argument("restClient: custom_params is null");
    }

    auto ep_it = custom_params->find("endpoint_override");
    if (ep_it == custom_params->end() || ep_it->second.empty()) {
        throw std::invalid_argument("restClient: 'endpoint_override' parameter is required");
    }
    endpoint_ = ep_it->second;

    // Create the multi handle and start the poller only after validation, so a
    // throwing constructor leaves no thread or handle behind.
    multi_ = curl_multi_init();
    if (!multi_) {
        throw std::runtime_error("restClient: curl_multi_init failed");
    }
    // Left unset, libcurl sizes the connection cache to the handles currently
    // added, so each new transfer evicts an idle keep-alive connection and burns
    // an ephemeral port. Size it to the in-flight cap instead.
    curl_multi_setopt(multi_, CURLMOPT_MAXCONNECTS, static_cast<long>(easyCacheCap_));
    poller_ = std::thread(&restClient::pollerLoop, this);

    std::string cap_source = "configured";
    if (!hasMaxInflight(custom_params)) {
        cap_source = n_request_classes * maxInflight_ < default_max_inflight ?
            absl::StrFormat("default: 1/%zu of RLIMIT_NOFILE", fd_share_divisor) :
            "default";
    }
    NIXL_INFO << absl::StrFormat(
        "restClient initialized: endpoint=%s, callback_threads=%zu, request_timeout_ms=%zu, "
        "max_inflight=%s per class (%s, curl_multi poller)",
        endpoint_,
        numThreads_,
        requestTimeoutMs_,
        maxInflight_ == 0 ? std::string("unlimited") : std::to_string(maxInflight_),
        cap_source);

    // Descriptors open so far are only a floor: the caller registers memory and
    // opens its fabric contexts after this.
    const std::size_t max_connections = n_request_classes * maxInflight_;
    rlimit lim{};
    const bool have_limit = getrlimit(RLIMIT_NOFILE, &lim) == 0;
    const std::size_t open_now = openFdCount().value_or(0);
    if (have_limit && lim.rlim_cur != RLIM_INFINITY) {
        NIXL_INFO << absl::StrFormat("restClient fd budget: open=%zu (so far), soft_limit=%llu, "
                                     "max_connections=%zu, headroom=%lld",
                                     open_now,
                                     static_cast<unsigned long long>(lim.rlim_cur),
                                     max_connections,
                                     static_cast<long long>(lim.rlim_cur) -
                                         static_cast<long long>(open_now) -
                                         static_cast<long long>(max_connections));
        if (maxInflight_ != 0 && open_now + max_connections > lim.rlim_cur) {
            NIXL_WARN << absl::StrFormat(
                "restClient: %zu connections (max_inflight=%zu for each of %zu "
                "request classes) plus %zu descriptors already open exceeds "
                "RLIMIT_NOFILE=%llu. Requests will fail at connect with EMFILE and "
                "be reported as curl_code=7, indistinguishable from a refused "
                "connection. Lower max_inflight or raise 'ulimit -n'.",
                max_connections,
                maxInflight_,
                n_request_classes,
                open_now,
                static_cast<unsigned long long>(lim.rlim_cur));
        }
    } else {
        NIXL_INFO << absl::StrFormat(
            "restClient fd budget: open=%zu (so far), soft_limit=unlimited, "
            "max_connections=%zu",
            open_now,
            max_connections);
    }
}

restClient::~restClient() {
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
    for (std::size_t cls = 0; cls < n_request_classes; ++cls) {
        NIXL_INFO << "restClient concurrency (" << requestClassName(cls)
                  << "): peak_inflight=" << queues_[cls].peakInflight
                  << ", peak_pending=" << queues_[cls].peakPending << ", max_inflight="
                  << (maxInflight_ == 0 ? std::string("unlimited") :
                                          std::to_string(maxInflight_));
    }
    NIXL_INFO << "restClient connections: requests=" << totalRequests_
              << ", new_connections=" << newConnects_
              << " (new_connections close to requests means keepalive is not working and "
                 "the endpoint will exhaust ephemeral ports)";
    if (fdExhaustionSeen_) {
        NIXL_ERROR << "restClient: ran out of file descriptors during the run; " << fdAbandoned_
                   << " queued request(s) were failed without being attempted, and "
                      "connect failures after the first were not logged individually. "
                      "See the earlier fd budget line.";
    }
}

std::string
restClient::buildUrl(std::string_view key) const {
    return absl::StrFormat("%s/%s", endpoint_, encodeKey(key));
}

void
restClient::enqueue(std::unique_ptr<requestCtx> ctx) {
    {
        const std::lock_guard<std::mutex> lk(queueMtx_);
        incoming_.push(std::move(ctx));
    }
    curl_multi_wakeup(multi_); // thread-safe; nudges the poller to drain the queue
}

void
restClient::reapCompletions() {
    CURLMsg *msg = nullptr;
    int in_queue = 0;
    while ((msg = curl_multi_info_read(multi_, &in_queue)) != nullptr) {
        if (msg->msg != CURLMSG_DONE) {
            continue;
        }
        CURL *easy = msg->easy_handle;
        CURLcode res = msg->data.result;
        requestCtx *ctx = nullptr;
        curl_easy_getinfo(easy, CURLINFO_PRIVATE, &ctx);
        long http_code = 0;
        curl_easy_getinfo(easy, CURLINFO_RESPONSE_CODE, &http_code);

        curl_multi_remove_handle(multi_, easy);
        inflight_.erase(ctx);
        queueOf(ctx).inflight--;

        // Every connect failure re-checks whether the process is out of
        // descriptors: libcurl often leaves the errno unset, so the count is what
        // tells local exhaustion from a refusing endpoint. The diagnostic is logged
        // on the first failure only. A request that got a response had a
        // descriptor, so it clears the state.
        if (res == CURLE_COULDNT_CONNECT) {
            fdExhausted_ = descriptorsExhausted();
            fdExhaustionSeen_ = fdExhaustionSeen_ || fdExhausted_;
        } else if (http_code != 0) {
            fdExhausted_ = false;
        }
        if (res == CURLE_COULDNT_CONNECT && !fdExhaustionLogged_) {
            fdExhaustionLogged_ = true;
            rlimit lim{};
            const bool have_limit = getrlimit(RLIMIT_NOFILE, &lim) == 0;
            NIXL_ERROR << absl::StrFormat(
                "restClient: first connect failure. fds open=%s, soft_limit=%s, "
                "max_inflight=%zu per class, curl says '%s'. Every running request holds a "
                "connection, so open close to the limit means the cap does not fit "
                "in the descriptors left after the caller's own use: lower "
                "max_inflight or raise 'ulimit -n'. Open well below the limit points "
                "at the endpoint instead. Further connect failures are not logged.",
                openFdCountText(),
                (have_limit && lim.rlim_cur != RLIM_INFINITY) ? std::to_string(lim.rlim_cur) :
                                                                std::string("unlimited"),
                maxInflight_,
                curlErrorText(res, ctx->errorBuf));
        }

        finishRequest(ctx, res, http_code);
    }
}

restClient::classQueue &
restClient::queueOf(const requestCtx *ctx) {
    return queues_[static_cast<std::size_t>(requestClassOf(ctx->method))];
}

std::size_t
restClient::failPending(CURLcode res) {
    std::size_t n = 0;
    for (classQueue &q : queues_) {
        while (!q.pending.empty()) {
            std::unique_ptr<requestCtx> ctx = std::move(q.pending.front());
            q.pending.pop_front();
            finishRequest(ctx.release(), res, 0);
            n++;
        }
    }
    return n;
}

void
restClient::startPending() {
    for (classQueue &q : queues_) {
        while (!q.pending.empty() && (maxInflight_ == 0 || q.inflight < maxInflight_)) {
            std::unique_ptr<requestCtx> ctx = std::move(q.pending.front());
            q.pending.pop_front();
            requestCtx *raw = ctx.get();
            // Built only now: a request waiting for a slot holds no access to the
            // buffer, and at most maxInflight_ accesses per class are open at once.
            if (raw->rdmaDesc) {
                const std::string desc = raw->rdmaDesc();
                if (desc.empty()) {
                    NIXL_ERROR << absl::StrFormat(
                        "%s: no RDMA descriptor for url=%s; not sent", raw->opName, raw->url);
                    finishRequest(ctx.release(), CURLE_FAILED_INIT, 0);
                    continue;
                }
                raw->headers = curl_slist_append(
                    raw->headers, absl::StrFormat("x-scal-rdma: %s", desc).c_str());
                curl_easy_setopt(raw->easy, CURLOPT_HTTPHEADER, raw->headers);
            }
            CURLMcode mc = curl_multi_add_handle(multi_, raw->easy);
            if (mc != CURLM_OK) {
                NIXL_ERROR << absl::StrFormat(
                    "%s: curl_multi_add_handle failed: %s", raw->opName, curl_multi_strerror(mc));
                finishRequest(ctx.release(), CURLE_FAILED_INIT, 0);
                continue;
            }
            ctx.release(); // ownership tracked via CURLOPT_PRIVATE until completion
            inflight_.insert(raw);
            q.inflight++;
            q.peakInflight = std::max(q.peakInflight, q.inflight);
        }
        // Sampled after starting, so it counts only what the cap held back.
        q.peakPending = std::max(q.peakPending, q.pending.size());
    }
}

void
restClient::pollerLoop() {
    for (;;) {
        const bool stopping = stop_.load();

        // 1. Drain the producer queue into the class queues. On shutdown, fail
        //    queued requests instead of starting them.
        std::queue<std::unique_ptr<requestCtx>> batch;
        {
            const std::lock_guard<std::mutex> lk(queueMtx_);
            std::swap(batch, incoming_);
        }
        while (!batch.empty()) {
            std::unique_ptr<requestCtx> ctx = std::move(batch.front());
            batch.pop();
            if (stopping) {
                finishRequest(ctx.release(), CURLE_ABORTED_BY_CALLBACK, 0);
                continue;
            }
            classQueue &q = queueOf(ctx.get());
            q.pending.push_back(std::move(ctx));
        }

        // 2. Fill the in-flight slots from the class queues.
        startPending();

        // 3. Advance all in-flight transfers (non-blocking).
        int running = 0;
        curl_multi_perform(multi_, &running);

        // 4. Hand finished transfers' callbacks to the worker pool.
        reapCompletions();

        // 5. Out of descriptors: nothing queued can connect, so fail the backlog
        //    now, unless descriptors were freed since the last check. Counted and
        //    reported at teardown.
        const bool any_pending =
            std::any_of(queues_.begin(), queues_.end(), [](const classQueue &q) {
                return !q.pending.empty();
            });
        if (fdExhausted_ && any_pending) {
            fdExhausted_ = descriptorsExhausted();
        }
        if (fdExhausted_ && any_pending) {
            fdAbandoned_ += failPending(CURLE_COULDNT_CONNECT);
        }

        // 6. Completions above freed slots; refill before sleeping so a capped
        //    queue does not stall waiting for the next socket event.
        if (!stopping) {
            startPending();
        }

        // 7. On shutdown, abort everything queued or in flight so every callback
        //    fires exactly once.
        if (stopping) {
            for (requestCtx *ctx : inflight_) {
                curl_multi_remove_handle(multi_, ctx->easy);
                finishRequest(ctx, CURLE_ABORTED_BY_CALLBACK, 0);
            }
            inflight_.clear();
            for (classQueue &q : queues_) {
                q.inflight = 0;
            }
            failPending(CURLE_ABORTED_BY_CALLBACK);
            break;
        }

        // 8. Block until socket activity, the 1s backstop, or curl_multi_wakeup().
        int numfds = 0;
        curl_multi_poll(multi_, nullptr, 0, 1000, &numfds);
    }
}

void
restClient::submitRdmaRequest(const char *op_name,
                              std::string_view key,
                              rdma_descriptor_fn_t rdma_desc,
                              bool is_upload,
                              std::function<void(bool)> callback,
                              size_t data_len,
                              size_t offset,
                              bool past_end_ok) {
    auto ctx = std::make_unique<requestCtx>();
    ctx->client = this;
    ctx->opName = op_name;
    ctx->pastEndOk = past_end_ok;
    ctx->method = is_upload ? rest_method::PUT : rest_method::GET;
    ctx->url = buildUrl(key);
    ctx->boolCb = std::move(callback);
    ctx->rdmaDesc = std::move(rdma_desc);

    ctx->easy = acquireEasy();
    if (!ctx->easy) {
        NIXL_ERROR << absl::StrFormat("%s: curl_easy_init failed", op_name);
        if (ctx->boolCb) {
            ctx->boolCb(false);
        }
        return;
    }

    if (is_upload) {
        // Content-Length: 0; data is transferred via RDMA, not the HTTP body.
        ctx->headers = curl_slist_append(ctx->headers, "Content-Length: 0");
    } else if (data_len > 0) {
        // Always sent, offset 0 included: the caller registered exactly data_len
        // bytes, and an unranged GET would have the server send the whole object.
        std::string range_header =
            absl::StrFormat("Range: bytes=%zu-%zu", offset, offset + data_len - 1);
        ctx->headers = curl_slist_append(ctx->headers, range_header.c_str());
    }

    buildEasy(ctx.get());
    enqueue(std::move(ctx));
}

void
restClient::putObjectRdmaAsync(std::string_view key,
                               uintptr_t data_ptr,
                               size_t data_len,
                               size_t offset,
                               rdma_descriptor_fn_t rdma_desc,
                               put_object_callback_t callback) {
    NIXL_TRACE << absl::StrFormat(
        "putObjectRdmaAsync: key=%s, data_ptr=%p, data_len=%zu, offset=%zu",
        key,
        reinterpret_cast<void *>(data_ptr),
        data_len,
        offset);

    if (data_len == 0) {
        NIXL_ERROR << "putObjectRdmaAsync: data_len is 0, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    if (!rdma_desc) {
        NIXL_ERROR << "putObjectRdmaAsync: no rdma_desc, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    submitRdmaRequest(
        "putObjectRdmaAsync", key, std::move(rdma_desc), /*is_upload=*/true, std::move(callback));
}

void
restClient::getObjectRdmaAsync(std::string_view key,
                               uintptr_t data_ptr,
                               size_t data_len,
                               size_t offset,
                               rdma_descriptor_fn_t rdma_desc,
                               bool past_end_ok,
                               get_object_callback_t callback) {
    NIXL_TRACE << absl::StrFormat(
        "getObjectRdmaAsync: key=%s, data_ptr=%p, data_len=%zu, offset=%zu",
        key,
        reinterpret_cast<void *>(data_ptr),
        data_len,
        offset);

    if (data_len == 0) {
        NIXL_ERROR << "getObjectRdmaAsync: data_len is 0, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    if (offset > (SIZE_MAX - (data_len - 1))) {
        NIXL_ERROR << "getObjectRdmaAsync: offset + data_len would overflow, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    if (!rdma_desc) {
        NIXL_ERROR << "getObjectRdmaAsync: no rdma_desc, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    submitRdmaRequest("getObjectRdmaAsync",
                      key,
                      std::move(rdma_desc),
                      /*is_upload=*/false,
                      std::move(callback),
                      data_len,
                      offset,
                      past_end_ok);
}

void
restClient::getObjectBodyAsync(std::string_view key,
                               void *dst,
                               size_t data_len,
                               size_t offset,
                               get_object_callback_t callback) {
    NIXL_TRACE << absl::StrFormat(
        "getObjectBodyAsync: key=%s, dst=%p, data_len=%zu, offset=%zu", key, dst, data_len, offset);

    if (dst == nullptr || data_len == 0) {
        NIXL_ERROR << absl::StrFormat(
            "getObjectBodyAsync: dst=%p data_len=%zu, returning failure", dst, data_len);
        if (callback) {
            callback(false);
        }
        return;
    }

    if (offset > (SIZE_MAX - (data_len - 1))) {
        NIXL_ERROR << "getObjectBodyAsync: offset + data_len would overflow, returning failure";
        if (callback) {
            callback(false);
        }
        return;
    }

    auto ctx = std::make_unique<requestCtx>();
    ctx->client = this;
    ctx->opName = "getObjectBodyAsync";
    ctx->method = rest_method::GET;
    ctx->url = buildUrl(key);
    ctx->boolCb = std::move(callback);
    ctx->sink.dst = static_cast<char *>(dst);
    ctx->sink.cap = data_len;
    ctx->sink.ranged = offset > 0;
    ctx->sink.errorBody = &ctx->responseBody;

    ctx->easy = acquireEasy();
    if (!ctx->easy) {
        NIXL_ERROR << "getObjectBodyAsync: curl_easy_init failed";
        if (ctx->boolCb) {
            ctx->boolCb(false);
        }
        return;
    }
    ctx->sink.easy = ctx->easy;

    // No x-scal-rdma header: the endpoint then answers with the bytes in the body.
    std::string range_header =
        absl::StrFormat("Range: bytes=%zu-%zu", offset, offset + data_len - 1);
    ctx->headers = curl_slist_append(ctx->headers, range_header.c_str());

    buildEasy(ctx.get());
    enqueue(std::move(ctx));
}

void
restClient::checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) {
    auto ctx = std::make_unique<requestCtx>();
    ctx->client = this;
    ctx->opName = "checkObjectExistsAsync";
    ctx->method = rest_method::HEAD;
    ctx->url = buildUrl(key);
    ctx->checkCb = std::move(callback);

    ctx->easy = acquireEasy();
    if (!ctx->easy) {
        NIXL_ERROR << "checkObjectExistsAsync: curl_easy_init failed";
        if (ctx->checkCb) {
            ctx->checkCb(std::nullopt);
        }
        return;
    }

    buildEasy(ctx.get());
    enqueue(std::move(ctx));
}

void
iRestClient::checkObjectsExistAsync(std::vector<std::string> keys,
                                    check_objects_callback_t callback) {
    if (keys.empty()) {
        if (callback) {
            callback({});
        }
        return;
    }

    struct checks {
        std::mutex mu;
        std::vector<std::optional<bool>> exists;
        size_t left;
        check_objects_callback_t callback;
    };

    auto state = std::make_shared<checks>();
    state->exists.assign(keys.size(), std::nullopt);
    state->left = keys.size();
    state->callback = std::move(callback);

    for (size_t i = 0; i < keys.size(); ++i) {
        checkObjectExistsAsync(keys[i], [state, i](std::optional<bool> exists) {
            {
                const std::lock_guard<std::mutex> lk(state->mu);
                state->exists[i] = exists;
                if (--state->left > 0) {
                    return;
                }
            }
            if (state->callback) {
                state->callback(std::move(state->exists));
            }
        });
    }
}

void
restClient::checkObjectsExistAsync(std::vector<std::string> keys,
                                   check_objects_callback_t callback) {
    const bool fits = !keys.empty() && keys.size() <= max_keys_per_exists_batch &&
        std::all_of(keys.begin(), keys.end(), fitsExistsBatch);
    if (!fits || batchExistsUnsupported_.load()) {
        iRestClient::checkObjectsExistAsync(std::move(keys), std::move(callback));
        return;
    }

    auto ctx = std::make_unique<requestCtx>();
    ctx->client = this;
    ctx->opName = "checkObjectsExistAsync";
    ctx->method = rest_method::POST;
    ctx->url = absl::StrFormat("%s/.batch_exists", endpoint_);
    for (const std::string &key : keys) {
        ctx->requestBody += key;
        ctx->requestBody += '\n';
    }
    ctx->batchKeys = std::move(keys);
    ctx->batchCb = std::move(callback);

    ctx->easy = acquireEasy();
    if (!ctx->easy) {
        NIXL_ERROR << "checkObjectsExistAsync: curl_easy_init failed";
        if (ctx->batchCb) {
            ctx->batchCb(std::vector<std::optional<bool>>(ctx->batchKeys.size()));
        }
        return;
    }

    ctx->headers = curl_slist_append(ctx->headers, "Content-Type: text/plain");
    ctx->headers = curl_slist_append(ctx->headers, "Accept: text/plain");
    // Send the body at once: waiting for a 100 Continue costs a round trip.
    ctx->headers = curl_slist_append(ctx->headers, "Expect:");

    buildEasy(ctx.get());
    enqueue(std::move(ctx));
}
