#ifndef NIXL_OBJ_PLUGIN_S3_SCALITY_AI_CONNECTOR_CLIENT_H
#define NIXL_OBJ_PLUGIN_S3_SCALITY_AI_CONNECTOR_CLIENT_H

#include <asio/thread_pool.hpp>
#include <asio/post.hpp>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <string_view>
#include "obj_backend.h"
#include "nixl_types.h"

/**
 * Interface for Scality AI Connector clients that support RDMA operations.
 */
class iScalityRdmaClient {
public:
    virtual ~iScalityRdmaClient() = default;

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
};

/**
 * Scality AI Connector HTTP client with RDMA support.
 * Uses libcurl to perform HTTP PUT/GET requests, passing the RDMA
 * descriptor via the x-scal-rdma custom header.
 *
 * URL format: {endpoint}/v1/{key}
 */
class ScalityClient : public iScalityRdmaClient {
public:
    /**
     * Constructor.
     * @param custom_params Backend init params; must contain "endpoint_override".
     *                      Optional "num_threads" controls the HTTP worker pool size
     *                      (default: 4).
     */
    explicit ScalityClient(nixl_b_params_t *custom_params);

    virtual ~ScalityClient() = default;

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

private:
    /// Base endpoint, e.g. "http://10.0.0.1:81"
    std::string endpoint_;

    // Thread pool for dispatching async HTTP requests via libcurl.
    //
    // Why not std::async?
    //   std::async returns a [[nodiscard]] std::future. If that future is
    //   discarded (not stored), its destructor fires immediately at the end of
    //   the expression and *blocks* until the task completes — making the call
    //   synchronous. Storing the future instead would require the caller to
    //   manage its lifetime, which conflicts with the fire-and-forget callback
    //   contract of this interface.
    //
    // Why not the AWS SDK executor (asioThreadPoolExecutor) used by the Dell path?
    //   The Dell client delegates async dispatch to the AWS SDK through
    //   PutObjectAsync/GetObjectAsync, which internally use the executor passed
    //   at S3Client construction time. The Scality AI Connector does not use the
    //   AWS SDK at all (it speaks a plain REST dialect with no AWS authentication),
    //   so there is no SDK-managed executor available. We own the thread pool
    //   directly here.
    std::size_t numThreads_;
    asio::thread_pool pool_;

    /**
     * Build the full URL for a given key.
     * @param key Object key
     * @return Full URL string
     */
    std::string
    buildUrl(std::string_view key) const;
};

#endif // NIXL_OBJ_PLUGIN_S3_SCALITY_AI_CONNECTOR_CLIENT_H
