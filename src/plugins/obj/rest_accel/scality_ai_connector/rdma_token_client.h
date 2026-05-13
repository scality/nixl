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

#ifndef NIXL_OBJ_PLUGIN_S3_SCALITY_AI_CONNECTOR_RDMA_TOKEN_CLIENT_H
#define NIXL_OBJ_PLUGIN_S3_SCALITY_AI_CONNECTOR_RDMA_TOKEN_CLIENT_H

#ifdef HAVE_CUOBJ_CLIENT
#include <cuobjclient.h>
#else

#include <cstddef>
#include <cstdint>
#include <sys/types.h>

// Standalone type definitions when cuobjclient is not available

enum cuObjErr_t {
    CU_OBJ_SUCCESS = 0,
    CU_OBJ_FAIL = -1,
};

struct cufileRDMAInfo_t {
    int version;
    int desc_len;
    const char *desc_str;
};

typedef ssize_t (*cuObjGetOp_t)(const void *handle,
                                char *buf,
                                size_t size,
                                loff_t offset,
                                const cufileRDMAInfo_t *infop);

typedef ssize_t (*cuObjPutOp_t)(const void *handle,
                                const char *buf,
                                size_t size,
                                loff_t offset,
                                const cufileRDMAInfo_t *infop);

struct CUObjIOOps {
    cuObjGetOp_t get;
    cuObjPutOp_t put;
};

using CUObjOps_t = CUObjIOOps;

// 4 GiB — matches cuobjclient limit
#define CUOBJ_MAX_MEMORY_REG_SIZE (4ULL * 1024 * 1024 * 1024)

#endif // HAVE_CUOBJ_CLIENT

#include <memory>

/**
 * Handle wrapper used by the ibverbs RC path to pass context through callbacks.
 * The real cuObjClient uses an opaque internal handle; this struct mirrors that
 * contract for the RC code path.  A magic field lets getCtx() distinguish
 * MockHandle from cuObjClient's internal handle at runtime (needed when both
 * transports are compiled in).
 */
struct MockHandle {
    static constexpr uint64_t MOCK_MAGIC = 0x4D4F434B48444C45ULL; // "MOCKHDLE"
    uint64_t magic = MOCK_MAGIC;
    void *ctx;
};

/**
 * Abstract RDMA token client interface.
 *
 * Two implementations exist:
 *  - CuObjRdmaTokenClient  — thin wrapper around NVIDIA cuObjClient (DC transport)
 *  - IbverbsRcRdmaTokenClient — libibverbs RC implementation (works with Soft-RoCE)
 */
class iRdmaTokenClient {
public:
    virtual ~iRdmaTokenClient() = default;

    /** Return true when the transport layer is ready for transfers. */
    virtual bool isConnected() const = 0;

    /**
     * Register a memory region and obtain an RDMA descriptor (token).
     * @return CU_OBJ_SUCCESS on success.
     */
    virtual cuObjErr_t cuMemObjGetDescriptor(void *ptr, size_t size) = 0;

    /**
     * Deregister a previously registered memory region.
     * @return CU_OBJ_SUCCESS on success.
     */
    virtual cuObjErr_t cuMemObjPutDescriptor(void *ptr) = 0;

    /**
     * Initiate a GET (read) transfer — calls the user-supplied get callback
     * with the RDMA token so that the remote side can perform the data transfer.
     */
    virtual ssize_t cuObjGet(void *ctx, void *ptr, size_t size, loff_t offset,
                             loff_t buf_offset = 0) = 0;

    /**
     * Initiate a PUT (write) transfer — calls the user-supplied put callback
     * with the RDMA token so that the remote side can perform the data transfer.
     */
    virtual ssize_t cuObjPut(void *ctx, void *ptr, size_t size, loff_t offset,
                             loff_t buf_offset = 0) = 0;

    /**
     * Extract the user context pointer from an opaque handle.
     *
     * When HAVE_CUOBJ_CLIENT is defined the real cuObjClient::getCtx is used;
     * otherwise we dereference a MockHandle.
     */
    static void *getCtx(const void *handle) {
        if (!handle) return nullptr;
        // Check if this is a MockHandle (RC path) by inspecting the magic field.
        const MockHandle *mh = static_cast<const MockHandle *>(handle);
        if (mh->magic == MockHandle::MOCK_MAGIC) {
            return mh->ctx;
        }
#ifdef HAVE_CUOBJ_CLIENT
        // Not a MockHandle — must be cuObjClient's internal handle (DC path).
        return cuObjClient::getCtx(handle);
#else
        return nullptr;
#endif
    }
};

#endif // NIXL_OBJ_PLUGIN_S3_SCALITY_AI_CONNECTOR_RDMA_TOKEN_CLIENT_H
