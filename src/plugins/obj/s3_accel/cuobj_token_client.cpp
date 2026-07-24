/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "cuobj_token_client.h"
#include "rdma_ctx.h"
#include "common/nixl_log.h"

namespace {

// cuObject invokes these on cuObjGet/cuObjPut, handing back the RDMA descriptor
// in infop->desc_str.  Copy it into the per-transfer rdma_ctx_t the engine owns.
ssize_t
objectGet(const void *handle,
          char *buf,
          size_t size,
          loff_t offset,
          const cufileRDMAInfo_t *infop) {
    if (infop == nullptr || infop->desc_str == nullptr) {
        NIXL_ERROR << "objectGet: infop or infop->desc_str is null";
        return -EINVAL;
    }

    void *ctx = iS3RdmaTokenClient::getCtx(handle);
    if (ctx == nullptr) {
        NIXL_ERROR << "objectGet: context is null";
        return -EINVAL;
    }
    NIXL_DEBUG << "objectGet: handle=" << handle << ", buf=" << static_cast<const void *>(buf)
               << ", size=" << size << ", offset=" << offset << ", infop=" << infop;
    rdma_ctx_t *rctx = static_cast<rdma_ctx_t *>(ctx);
    rctx->rdma_desc = infop->desc_str;
    return 0;
}

ssize_t
objectPut(const void *handle,
          const char *buf,
          size_t size,
          loff_t offset,
          const cufileRDMAInfo_t *infop) {
    if (infop == nullptr || infop->desc_str == nullptr) {
        NIXL_ERROR << "objectPut: infop or infop->desc_str is null";
        return -EINVAL;
    }

    void *ctx = iS3RdmaTokenClient::getCtx(handle);
    if (ctx == nullptr) {
        NIXL_ERROR << "objectPut: context is null";
        return -EINVAL;
    }
    NIXL_DEBUG << "objectPut: handle=" << handle << ", buf=" << static_cast<const void *>(buf)
               << ", size=" << size << ", offset=" << offset << ", infop=" << infop;
    rdma_ctx_t *rctx = static_cast<rdma_ctx_t *>(ctx);
    rctx->rdma_desc = infop->desc_str;
    return 0;
}

CUObjIOOps obs_ops = {.get = objectGet, .put = objectPut};

} // namespace

S3CuObjTokenClient::S3CuObjTokenClient()
    : inner_(std::make_shared<cuObjClient>(obs_ops, CUOBJ_PROTO_RDMA_DC_V1)) {
    if (!inner_->isConnected()) {
        NIXL_ERROR << "CUObjClient failed to connect.";
        return;
    }
    NIXL_INFO << "S3CuObjTokenClient initialized (DC transport)";
}

bool
S3CuObjTokenClient::isConnected() const {
    return inner_->isConnected();
}

cuObjErr_t
S3CuObjTokenClient::cuMemObjGetDescriptor(void *ptr, size_t size, int /*dev_id*/) {
    // cuObject selects the NIC internally; the affinity hint does not apply here.
    return inner_->cuMemObjGetDescriptor(ptr, size);
}

cuObjErr_t
S3CuObjTokenClient::cuMemObjPutDescriptor(void *ptr) {
    return inner_->cuMemObjPutDescriptor(ptr);
}

ssize_t
S3CuObjTokenClient::cuObjGet(void *ctx, void *ptr, size_t size, loff_t offset, loff_t buf_offset) {
    return inner_->cuObjGet(ctx, ptr, size, offset, buf_offset);
}

ssize_t
S3CuObjTokenClient::cuObjPut(void *ctx, void *ptr, size_t size, loff_t offset, loff_t buf_offset) {
    return inner_->cuObjPut(ctx, ptr, size, offset, buf_offset);
}
