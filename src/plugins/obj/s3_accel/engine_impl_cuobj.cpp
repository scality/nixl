/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl_cuobj.h"
#include "s3_accel/rdma_interface.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <algorithm>
#include <chrono>
#include <future>
#include <memory>
#include <vector>

namespace {

typedef struct rdma_ctx {
    std::string rdma_desc;
} rdma_ctx_t;

bool
isValidPrepXferParams(const nixl_xfer_op_t &operation,
                      const nixl_meta_dlist_t &local,
                      const nixl_meta_dlist_t &remote,
                      const std::string &remote_agent,
                      const std::string &local_agent) {
    if (operation != NIXL_WRITE && operation != NIXL_READ) {
        NIXL_ERROR << absl::StrFormat("Error: Invalid operation type: %d", operation);
        return false;
    }

    if (remote_agent != local_agent) {
        NIXL_WARN << absl::StrFormat(
            "Warning: Remote agent doesn't match the requesting agent (%s). Got %s",
            local_agent,
            remote_agent);
    }

    if ((local.getType() != DRAM_SEG) && (local.getType() != VRAM_SEG)) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Local memory type must be VRAM_SEG or DRAM_SEG, got %d", local.getType());
        return false;
    }

    if (remote.getType() != OBJ_SEG) {
        NIXL_ERROR << absl::StrFormat("Error: Remote memory type must be OBJ_SEG, got %d",
                                      remote.getType());
        return false;
    }

    if (local.descCount() != remote.descCount()) {
        NIXL_ERROR << absl::StrFormat(
            "Error: Local and remote descriptor counts must match. Got %d local, %d remote",
            local.descCount(),
            remote.descCount());
        return false;
    }

    return true;
}

class obsObjTransferRequestH {
public:
    uintptr_t addr;
    size_t size;
    size_t offset;
    std::string rdma_desc;
    std::string obj_key;
    rdma_ctx_t ctx;

    obsObjTransferRequestH() : addr(0), size(0), offset(0), rdma_desc(""), obj_key("") {}

    obsObjTransferRequestH(uintptr_t a, size_t s, size_t off)
        : addr(a), size(s), offset(off), rdma_desc(""), obj_key("") {}

    ~obsObjTransferRequestH() = default;
};

class nixlObsObjBackendReqH : public nixlBackendReqH {
public:
    std::vector<obsObjTransferRequestH> reqs_;
    std::vector<std::future<nixl_status_t>> statusFutures_;

    nixlObsObjBackendReqH() = default;
    ~nixlObsObjBackendReqH() = default;

    nixl_status_t
    getOverallStatus() {
        bool has_pending = false;
        auto it = statusFutures_.begin();
        while (it != statusFutures_.end()) {
            if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                auto current_status = it->get();
                if (current_status != NIXL_SUCCESS) {
                    statusFutures_.clear();
                    return current_status;
                }
                it = statusFutures_.erase(it);
            } else {
                ++it;
                has_pending = true;
            }
        }
        if (has_pending) {
            return NIXL_IN_PROG;
        }
        return NIXL_SUCCESS;
    }
};

class nixlObsObjMetadata : public nixlBackendMD {
public:
    nixlObsObjMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, std::string obj_key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(obj_key),
          localAddr(0) {}

    nixlObsObjMetadata(nixl_mem_t nixl_mem, uintptr_t addr)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(0),
          objKey(""),
          localAddr(addr) {}

    ~nixlObsObjMetadata() = default;

    nixl_mem_t nixlMem;
    uint64_t devId;
    std::string objKey;
    uintptr_t localAddr;
};

static ssize_t
objectGet(const void *handle,
          char *buf,
          size_t size,
          loff_t offset,
          const cufileRDMAInfo_t *infop) {
    if (infop == nullptr || infop->desc_str == nullptr) {
        NIXL_ERROR << "objectGet: infop or infop->desc_str is null";
        return -EINVAL;
    }

    void *ctx = cuObjClient::getCtx(handle);
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

static ssize_t
objectPut(const void *handle,
          const char *buf,
          size_t size,
          loff_t offset,
          const cufileRDMAInfo_t *infop) {
    if (infop == nullptr || infop->desc_str == nullptr) {
        NIXL_ERROR << "objectPut: infop or infop->desc_str is null";
        return -EINVAL;
    }

    void *ctx = cuObjClient::getCtx(handle);
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

S3CuObjEngineImpl::S3CuObjEngineImpl(const nixlBackendInitParams *init_params)
    : S3AccelObjEngineImpl(init_params, NoClientTag{}) {
    cuClient_ = std::make_shared<cuObjClient>(obs_ops, CUOBJ_PROTO_RDMA_DC_V1);
    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "CUObjClient failed to connect.";
        return;
    }
}

nixl_status_t
S3CuObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                               const nixl_mem_t &nixl_mem,
                               nixlBackendMD *&out) {
    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "CUObjClient is not connected.";
        return NIXL_ERR_BACKEND;
    }

    auto supported_mems = {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end()) {
        return NIXL_ERR_NOT_SUPPORTED;
    }

    if (nixl_mem == OBJ_SEG) {
        std::unique_ptr<nixlObsObjMetadata> obj_md = std::make_unique<nixlObsObjMetadata>(
            nixl_mem, mem.devId, mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo);
        devIdToObjKey_[mem.devId] = obj_md->objKey;
        out = obj_md.release();
    } else if ((nixl_mem == DRAM_SEG) || (nixl_mem == VRAM_SEG)) {
        if (mem.len > CUOBJ_MAX_MEMORY_REG_SIZE) {
            NIXL_ERROR << "Memory size too large for cuObject registration: " << mem.len;
            return NIXL_ERR_NOT_SUPPORTED;
        }

        NIXL_DEBUG << "registerMem: addr=" << mem.addr << ", len=" << mem.len
                   << ", nixl_mem=" << nixl_mem;
        std::unique_ptr<nixlObsObjMetadata> mem_md =
            std::make_unique<nixlObsObjMetadata>(nixl_mem, mem.addr);

        cuObjErr_t cuda_status = cuClient_->cuMemObjGetDescriptor((void *)(mem.addr), mem.len);
        if (cuda_status != CU_OBJ_SUCCESS) {
            NIXL_ERROR << "cuMemObjGetDescriptor failed with status: " << cuda_status;
            return NIXL_ERR_BACKEND;
        }
        out = mem_md.release();
    }

    return NIXL_SUCCESS;
}

nixl_status_t
S3CuObjEngineImpl::deregisterMem(nixlBackendMD *meta) {
    nixlObsObjMetadata *md = static_cast<nixlObsObjMetadata *>(meta);
    if (md) {
        if (md->nixlMem == OBJ_SEG) {
            std::unique_ptr<nixlObsObjMetadata> obj_md_ptr(md);
            devIdToObjKey_.erase(obj_md_ptr->devId);
        } else if ((md->nixlMem == DRAM_SEG) || (md->nixlMem == VRAM_SEG)) {
            std::unique_ptr<nixlObsObjMetadata> mem_md_ptr(md);
            cuObjErr_t cuda_status =
                cuClient_->cuMemObjPutDescriptor((void *)(mem_md_ptr->localAddr));
            if (cuda_status != CU_OBJ_SUCCESS) {
                NIXL_ERROR << "cuMemObjPutDescriptor failed with status: " << cuda_status;
                mem_md_ptr.release();
                return NIXL_ERR_BACKEND;
            }
        }
    }
    return NIXL_SUCCESS;
}

nixl_status_t
S3CuObjEngineImpl::prepXfer(const nixl_xfer_op_t &operation,
                            const nixl_meta_dlist_t &local,
                            const nixl_meta_dlist_t &remote,
                            const std::string &remote_agent,
                            const std::string &local_agent,
                            nixlBackendReqH *&handle,
                            const nixl_opt_b_args_t *opt_args) const {
    if (!cuClient_->isConnected()) {
        NIXL_ERROR << "CUObjClient is not connected.";
        return NIXL_ERR_BACKEND;
    }

    if (!isValidPrepXferParams(operation, local, remote, remote_agent, local_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto req_h = std::make_unique<nixlObsObjBackendReqH>();

    for (int i = 0; i < local.descCount(); ++i) {
        obsObjTransferRequestH req(local[i].addr, local[i].len, remote[i].addr);

        auto obj_key_search = devIdToObjKey_.find(remote[i].devId);
        if (obj_key_search == devIdToObjKey_.end()) {
            NIXL_ERROR << "The object segment key " << remote[i].devId
                       << " is not registered with the backend";
            return NIXL_ERR_INVALID_PARAM;
        }
        req.obj_key = obj_key_search->second;

        if (operation == NIXL_WRITE) {
            ssize_t cuda_status =
                cuClient_->cuObjPut(&req.ctx, (void *)req.addr, req.size, req.offset);
            if (cuda_status < 0) {
                NIXL_ERROR << "cuObjPut failed with status: " << cuda_status;
                return NIXL_ERR_BACKEND;
            }
        } else if (operation == NIXL_READ) {
            ssize_t cuda_status =
                cuClient_->cuObjGet(&req.ctx, (void *)req.addr, req.size, req.offset);
            if (cuda_status < 0) {
                NIXL_ERROR << "cuObjGet failed with status: " << cuda_status;
                return NIXL_ERR_BACKEND;
            }
        }
        req.rdma_desc = req.ctx.rdma_desc;

        req_h->reqs_.push_back(req);
    }

    handle = req_h.release();
    return NIXL_SUCCESS;
}

nixl_status_t
S3CuObjEngineImpl::postXfer(const nixl_xfer_op_t &operation,
                            const nixl_meta_dlist_t &local,
                            const nixl_meta_dlist_t &remote,
                            const std::string &remote_agent,
                            nixlBackendReqH *&handle,
                            const nixl_opt_b_args_t *opt_args) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlObsObjBackendReqH *req_h = static_cast<nixlObsObjBackendReqH *>(handle);

    // getClient() dispatches to the vendor subclass, which provides the RDMA-capable client.
    auto rdmaClient = dynamic_cast<iS3RdmaClient *>(getClient());
    if (!rdmaClient) {
        NIXL_ERROR << "cuObject RDMA operations require an iS3RdmaClient";
        for (size_t i = 0; i < req_h->reqs_.size(); ++i) {
            auto status_promise = std::make_shared<std::promise<nixl_status_t>>();
            req_h->statusFutures_.push_back(status_promise->get_future());
            status_promise->set_value(NIXL_ERR_BACKEND);
        }
        return NIXL_IN_PROG;
    }

    for (const auto &req : req_h->reqs_) {
        auto status_promise = std::make_shared<std::promise<nixl_status_t>>();
        req_h->statusFutures_.push_back(status_promise->get_future());

        // S3 client signals completion via callback; NIXL polls the handle.
        // Use future/promise to bridge the two interfaces.
        if (operation == NIXL_WRITE) {
            rdmaClient->putObjectRdmaAsync(req.obj_key,
                                           req.addr,
                                           req.size,
                                           req.offset,
                                           req.rdma_desc,
                                           [status_promise](bool success) {
                                               status_promise->set_value(
                                                   success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
                                           });
        } else {
            rdmaClient->getObjectRdmaAsync(req.obj_key,
                                           req.addr,
                                           req.size,
                                           req.offset,
                                           req.rdma_desc,
                                           [status_promise](bool success) {
                                               status_promise->set_value(
                                                   success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
                                           });
        }
    }

    return NIXL_IN_PROG;
}

nixl_status_t
S3CuObjEngineImpl::checkXfer(nixlBackendReqH *handle) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlObsObjBackendReqH *req_h = static_cast<nixlObsObjBackendReqH *>(handle);
    return req_h->getOverallStatus();
}

nixl_status_t
S3CuObjEngineImpl::releaseReqH(nixlBackendReqH *handle) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlObsObjBackendReqH *req_h = static_cast<nixlObsObjBackendReqH *>(handle);
    delete req_h;
    return NIXL_SUCCESS;
}
