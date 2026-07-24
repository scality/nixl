/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_ACCEL_RDMA_TOKEN_CLIENT_H
#define NIXL_OBJ_PLUGIN_S3_ACCEL_RDMA_TOKEN_CLIENT_H

#include <cuobjclient.h>

#include <memory>

/**
 * Abstract RDMA token client interface.
 *
 * The seam that lets the cuObject-accelerated engine pick a token producer at
 * runtime while leaving registerMem/prepXfer/postXfer untouched.  The method
 * signatures deliberately mirror cuObjClient so a backend is a drop-in.
 *
 * Implementations:
 *  - S3CuObjTokenClient: thin wrapper around NVIDIA cuObjClient (DC transport),
 *    the default.
 */
class iS3RdmaTokenClient {
public:
    virtual ~iS3RdmaTokenClient() = default;

    /** Return true when the transport layer is ready for transfers. */
    virtual bool
    isConnected() const = 0;

    /**
     * Register a memory region and obtain an RDMA descriptor (token).
     * @param dev_id CUDA device ordinal for VRAM buffers (affinity hint), or -1
     *               for host memory / no affinity.  Implementations may ignore it.
     * @return CU_OBJ_SUCCESS on success.
     */
    virtual cuObjErr_t
    cuMemObjGetDescriptor(void *ptr, size_t size, int dev_id = -1) = 0;

    /**
     * Deregister a previously registered memory region.
     * @return CU_OBJ_SUCCESS on success.
     */
    virtual cuObjErr_t
    cuMemObjPutDescriptor(void *ptr) = 0;

    /**
     * Initiate a GET (read) transfer; produces the RDMA token into ctx->rdma_desc
     * so the remote side can perform the data transfer.
     */
    virtual ssize_t
    cuObjGet(void *ctx, void *ptr, size_t size, loff_t offset, loff_t buf_offset = 0) = 0;

    /**
     * Initiate a PUT (write) transfer; produces the RDMA token into ctx->rdma_desc
     * so the remote side can perform the data transfer.
     */
    virtual ssize_t
    cuObjPut(void *ctx, void *ptr, size_t size, loff_t offset, loff_t buf_offset = 0) = 0;

    /**
     * Extract the user context pointer from cuObjClient's opaque handle.
     */
    static void *
    getCtx(const void *handle) {
        if (!handle) {
            return nullptr;
        }
        return cuObjClient::getCtx(handle);
    }
};

#endif // NIXL_OBJ_PLUGIN_S3_ACCEL_RDMA_TOKEN_CLIENT_H
