#include "client.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <asio/post.hpp>
#include <curl/curl.h>
#include <mutex>
#include <stdexcept>

static size_t captureBody(void *ptr, size_t size, size_t nmemb, void *userdata) {
    auto *body = static_cast<std::string *>(userdata);
    body->append(static_cast<char *>(ptr), size * nmemb);
    return size * nmemb;
}

// RAII wrapper for thread_local CURL handle — cleaned up when the thread exits.
// curl_easy_reset() preserves the connection cache while clearing options.
struct thread_local_curl_t {
    CURL *handle;
    thread_local_curl_t() : handle(curl_easy_init()) {}
    ~thread_local_curl_t() { if (handle) curl_easy_cleanup(handle); }
};

static std::once_flag curl_init_flag;

static std::size_t
parseNumThreads(nixl_b_params_t *params) {
    constexpr std::size_t kDefault = 4;
    if (!params) return kDefault;
    auto it = params->find("num_threads");
    if (it == params->end() || it->second.empty()) return kDefault;
    try {
        auto val = std::stoul(it->second);
        return val > 0 ? val : kDefault;
    } catch (...) {
        return kDefault;
    }
}

ScalityClient::ScalityClient(nixl_b_params_t *custom_params)
    : numThreads_(parseNumThreads(custom_params)),
      pool_(numThreads_) {
    std::call_once(curl_init_flag, []() {
        curl_global_init(CURL_GLOBAL_DEFAULT);
    });
    if (!custom_params) {
        throw std::invalid_argument("ScalityClient: custom_params is null");
    }

    auto ep_it = custom_params->find("endpoint_override");
    if (ep_it == custom_params->end() || ep_it->second.empty()) {
        throw std::invalid_argument(
            "ScalityClient: 'endpoint_override' parameter is required");
    }
    endpoint_ = ep_it->second;

    NIXL_INFO << absl::StrFormat(
        "ScalityClient initialized: endpoint=%s, num_threads=%zu",
        endpoint_, numThreads_);
}

std::string
ScalityClient::buildUrl(std::string_view key) const {
    return absl::StrFormat("%s/v1/%s", endpoint_, std::string(key));
}

void
ScalityClient::putObjectRdmaAsync(std::string_view key,
                                  uintptr_t data_ptr,
                                  size_t data_len,
                                  size_t offset,
                                  std::string_view rdma_desc,
                                  put_object_callback_t callback) {
    NIXL_DEBUG << absl::StrFormat(
        "putObjectRdmaAsync: key=%s, data_ptr=%p, data_len=%zu, offset=%zu, rdma_desc=%s",
        std::string(key).c_str(),
        reinterpret_cast<void *>(data_ptr),
        data_len,
        offset,
        rdma_desc.empty() ? "<empty>" : std::string(rdma_desc).c_str());

    if (data_len == 0) {
        NIXL_ERROR << "putObjectRdmaAsync: data_len is 0, returning failure";
        callback(false);
        return;
    }

    if (rdma_desc.empty()) {
        NIXL_ERROR << "putObjectRdmaAsync: rdma_desc is empty, returning failure";
        callback(false);
        return;
    }

    std::string url = buildUrl(key);
    std::string rdma_header =
        absl::StrFormat("x-scal-rdma: %s", std::string(rdma_desc));

    asio::post(pool_, [url, rdma_header, callback]() {
        thread_local thread_local_curl_t tls_curl;
        CURL *curl = tls_curl.handle;
        if (!curl) {
            NIXL_ERROR << "putObjectRdmaAsync: curl handle unavailable";
            callback(false);
            return;
        }
        curl_easy_reset(curl);

        struct curl_slist *headers = nullptr;
        headers = curl_slist_append(headers, rdma_header.c_str());
        // Content-Length: 0 — data is transferred via RDMA, not the HTTP body
        headers = curl_slist_append(headers, "Content-Length: 0");

        std::string response_body;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_UPLOAD, 1L);
        curl_easy_setopt(curl, CURLOPT_INFILESIZE_LARGE, (curl_off_t)0);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, captureBody);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);

        CURLcode res = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_slist_free_all(headers);

        bool success = (res == CURLE_OK) && (http_code >= 200 && http_code < 300);
        if (!success) {
            NIXL_ERROR << absl::StrFormat(
                "putObjectRdmaAsync: failed url=%s curl_code=%d http_code=%ld body=%s",
                url,
                static_cast<int>(res),
                http_code,
                response_body.empty() ? "<empty>" : response_body);
        } else {
            NIXL_DEBUG << absl::StrFormat(
                "putObjectRdmaAsync: success url=%s http_code=%ld", url, http_code);
        }
        callback(success);
    });
}

void
ScalityClient::getObjectRdmaAsync(std::string_view key,
                                  uintptr_t data_ptr,
                                  size_t data_len,
                                  size_t offset,
                                  std::string_view rdma_desc,
                                  get_object_callback_t callback) {
    NIXL_DEBUG << absl::StrFormat(
        "getObjectRdmaAsync: key=%s, data_ptr=%p, data_len=%zu, offset=%zu, rdma_desc=%s",
        std::string(key).c_str(),
        reinterpret_cast<void *>(data_ptr),
        data_len,
        offset,
        rdma_desc.empty() ? "<empty>" : std::string(rdma_desc).c_str());

    if (data_len == 0) {
        NIXL_ERROR << "getObjectRdmaAsync: data_len is 0, returning failure";
        callback(false);
        return;
    }

    if ((data_len > 0) && (offset > (SIZE_MAX - (data_len - 1)))) {
        NIXL_ERROR << "getObjectRdmaAsync: offset + data_len would overflow, returning failure";
        callback(false);
        return;
    }

    if (rdma_desc.empty()) {
        NIXL_ERROR << "getObjectRdmaAsync: rdma_desc is empty, returning failure";
        callback(false);
        return;
    }

    std::string url = buildUrl(key);
    std::string rdma_header =
        absl::StrFormat("x-scal-rdma: %s", std::string(rdma_desc));

    asio::post(pool_, [url, rdma_header, callback]() {
        thread_local thread_local_curl_t tls_curl;
        CURL *curl = tls_curl.handle;
        if (!curl) {
            NIXL_ERROR << "getObjectRdmaAsync: curl handle unavailable";
            callback(false);
            return;
        }
        curl_easy_reset(curl);

        struct curl_slist *headers = nullptr;
        headers = curl_slist_append(headers, rdma_header.c_str());

        std::string response_body;
        curl_easy_setopt(curl, CURLOPT_URL, url.c_str());
        curl_easy_setopt(curl, CURLOPT_HTTPGET, 1L);
        curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, captureBody);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &response_body);

        CURLcode res = curl_easy_perform(curl);
        long http_code = 0;
        curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);

        curl_slist_free_all(headers);

        bool success = (res == CURLE_OK) && (http_code >= 200 && http_code < 300);
        if (!success) {
            NIXL_ERROR << absl::StrFormat(
                "getObjectRdmaAsync: failed url=%s curl_code=%d http_code=%ld body=%s",
                url,
                static_cast<int>(res),
                http_code,
                response_body.empty() ? "<empty>" : response_body);
        } else {
            NIXL_DEBUG << absl::StrFormat(
                "getObjectRdmaAsync: success url=%s http_code=%ld", url, http_code);
        }
        callback(success);
    });
}
