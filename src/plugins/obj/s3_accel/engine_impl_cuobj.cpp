/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl_cuobj.h"
#include "s3_accel/rdma_interface.h"
#include "s3_accel/rdma_ctx.h"
#include "s3_accel/cuobj_token_client.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <future>
#include <memory>
#include <optional>
#include <vector>

#ifdef S3_ACCEL_HAVE_IBVERBS_DC
#include "s3_accel/cufile_nics.h"
#include "s3_accel/ibverbs_dc_token_client.h"
#include <cuda_runtime.h>
#include <fstream>
#include <sstream>
#endif

namespace {

#ifdef S3_ACCEL_HAVE_IBVERBS_DC
/**
 * Scoped CUDA device switch. Registering a VRAM buffer requires the buffer's
 * GPU to be the current device; restore the previous device on scope exit.
 */
class CudaDeviceGuard {
public:
    explicit CudaDeviceGuard(int dev) {
        if (cudaGetDevice(&prevDev_) != cudaSuccess) {
            return;
        }
        if (dev >= 0 && dev != prevDev_ && cudaSetDevice(dev) == cudaSuccess) {
            restore_ = true;
        }
    }
    ~CudaDeviceGuard() {
        if (restore_) {
            cudaSetDevice(prevDev_);
        }
    }
    CudaDeviceGuard(const CudaDeviceGuard &) = delete;
    CudaDeviceGuard &
    operator=(const CudaDeviceGuard &) = delete;

private:
    int prevDev_ = 0;
    bool restore_ = false;
};

/// Look up a key in customParams, returning a default when absent/empty.
std::string
paramOr(const nixl_b_params_t *params, const std::string &key, const std::string &def) {
    if (params == nullptr) {
        return def;
    }
    auto it = params->find(key);
    return (it == params->end() || it->second.empty()) ? def : it->second;
}

/// Split a comma-separated list, trimming surrounding whitespace; skips empties.
std::vector<std::string>
splitCsv(const std::string &s) {
    std::vector<std::string> out;
    size_t start = 0;
    while (start <= s.size()) {
        size_t comma = s.find(',', start);
        size_t end = (comma == std::string::npos) ? s.size() : comma;
        size_t a = s.find_first_not_of(" \t", start);
        if (a != std::string::npos && a < end) {
            size_t b = s.find_last_not_of(" \t", end - 1);
            out.push_back(s.substr(a, b - a + 1));
        }
        if (comma == std::string::npos) {
            break;
        }
        start = comma + 1;
    }
    return out;
}

/// Read cufile.json content (CUFILE_ENV_PATH_JSON, else /etc/cufile.json).
/// Returns "" if unreadable. Used only to resolve the DC key.
std::string
readCufileJson() {
    const char *env = std::getenv("CUFILE_ENV_PATH_JSON");
    const std::string path = env ? env : "/etc/cufile.json";
    std::ifstream f(path);
    if (!f) {
        return std::string();
    }
    std::stringstream ss;
    ss << f.rdbuf();
    return ss.str();
}
#endif // S3_ACCEL_HAVE_IBVERBS_DC

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

    // DRAM/VRAM: keep the GPU ordinal for VRAM NIC affinity and the CUDA device
    // guard used when (de)registering the buffer with the ibverbs DC client.
    nixlObsObjMetadata(nixl_mem_t nixl_mem, uintptr_t addr, uint64_t dev_id)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(""),
          localAddr(addr) {}

    ~nixlObsObjMetadata() = default;

    nixl_mem_t nixlMem;
    uint64_t devId;
    std::string objKey;
    uintptr_t localAddr;
};

} // namespace

S3CuObjEngineImpl::S3CuObjEngineImpl(const nixlBackendInitParams *init_params)
    : S3AccelObjEngineImpl(init_params, NoClientTag{}) {
    tokenClient_ = std::make_shared<S3CuObjTokenClient>();
    if (!tokenClient_->isConnected()) {
        NIXL_ERROR << "S3 RDMA token client failed to connect.";
        return;
    }

#ifdef S3_ACCEL_HAVE_IBVERBS_DC
    // Multi-rail: when a NIC list is resolvable, DRAM/VRAM route to the ibverbs
    // DC client (built lazily in ensureHostClient), which spreads registrations
    // across the NICs, instead of cuObject, which pins host memory to a single
    // NIC. The list comes from the rdma_nics param, else cufile.json's
    // rdma_dev_addr_list (mirroring the Scality AI Connector). No list at all
    // (param unset and cufile absent/empty) keeps cuObject as the default.
    const nixl_b_params_t *params = init_params ? init_params->customParams : nullptr;
    const std::string cufile = readCufileJson();

    const std::string nics_param = paramOr(params, "rdma_nics", "");
    if (!nics_param.empty()) {
        rdmaNics_ = splitCsv(nics_param);
    } else {
        rdmaNics_ = parseRdmaDevAddrList(cufile);
        if (!rdmaNics_.empty()) {
            NIXL_INFO << "s3_accel: resolved " << rdmaNics_.size()
                      << " DRAM/VRAM NIC(s) from cufile.json rdma_dev_addr_list";
        }
    }

    if (!rdmaNics_.empty()) {
        std::string dckey = paramOr(params, "rdma_dc_key", "");
        if (dckey.empty()) {
            dckey = parseRdmaDcKey(cufile); // "" if absent/commented
        }
        if (!dckey.empty()) {
            dcKey_ = std::strtoull(dckey.c_str(), nullptr, 0);
        }
        NIXL_INFO << "s3_accel: ibverbs DC multi-rail enabled (" << rdmaNics_.size()
                  << " NIC(s))";
    }
#endif
}

nixl_status_t
S3CuObjEngineImpl::ensureHostClient() {
    std::lock_guard<std::mutex> lk(hostClientMu_);
    if (hostClient_) {
        return NIXL_SUCCESS;
    }
#ifdef S3_ACCEL_HAVE_IBVERBS_DC
    if (rdmaNics_.empty()) {
        NIXL_ERROR << "ibverbs DC transport requires a NIC list: set the 'rdma_nics' "
                      "parameter (IPv4 addresses or device names like mlx5_1)";
        return NIXL_ERR_BACKEND;
    }
    auto client = std::make_shared<S3IbverbsDcTokenClient>(rdmaNics_, dcKey_);
    if (!client->isConnected()) {
        NIXL_ERROR << "Failed to initialize the ibverbs DC client for DRAM/VRAM transfers";
        return NIXL_ERR_BACKEND;
    }
    hostClient_ = std::move(client);
    NIXL_INFO << "s3_accel: DRAM/VRAM transfers using ibverbs DC across " << rdmaNics_.size()
              << " NIC(s)";
    return NIXL_SUCCESS;
#else
    return NIXL_ERR_BACKEND;
#endif
}

nixl_status_t
S3CuObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                               const nixl_mem_t &nixl_mem,
                               nixlBackendMD *&out) {
    if (!tokenClient_->isConnected()) {
        NIXL_ERROR << "S3 RDMA token client is not connected.";
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

#ifdef S3_ACCEL_HAVE_IBVERBS_DC
        // With a resolved NIC list, DRAM/VRAM route to the ibverbs DC client;
        // build it on first use. No NIC list -> clientFor() stays on cuObject.
        if (!rdmaNics_.empty()) {
            nixl_status_t st = ensureHostClient();
            if (st != NIXL_SUCCESS) {
                return st;
            }
        }
#endif

        NIXL_DEBUG << "registerMem: addr=" << mem.addr << ", len=" << mem.len
                   << ", nixl_mem=" << nixl_mem;
        std::unique_ptr<nixlObsObjMetadata> mem_md =
            std::make_unique<nixlObsObjMetadata>(nixl_mem, mem.addr, mem.devId);

        // VRAM passes the GPU ordinal as the DC NIC-affinity hint (ignored by
        // cuObject); DRAM passes -1. cuObject picks the NIC internally either way.
        int affinity_dev = -1;
#ifdef S3_ACCEL_HAVE_IBVERBS_DC
        std::optional<CudaDeviceGuard> dev_guard;
        if (nixl_mem == VRAM_SEG && hostClient_) {
            dev_guard.emplace((int)mem.devId);
            affinity_dev = (int)mem.devId;
        }
#endif
        cuObjErr_t cuda_status =
            clientFor(nixl_mem)->cuMemObjGetDescriptor((void *)(mem.addr), mem.len, affinity_dev);
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
#ifdef S3_ACCEL_HAVE_IBVERBS_DC
            std::optional<CudaDeviceGuard> dev_guard;
            if (mem_md_ptr->nixlMem == VRAM_SEG && hostClient_) {
                dev_guard.emplace((int)mem_md_ptr->devId);
            }
#endif
            cuObjErr_t cuda_status =
                clientFor(mem_md_ptr->nixlMem)->cuMemObjPutDescriptor((void *)(mem_md_ptr->localAddr));
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
    if (!isValidPrepXferParams(operation, local, remote, remote_agent, local_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    // DRAM/VRAM route to the ibverbs DC client when multi-rail is opted in,
    // otherwise (and for the default) to cuObject. The buffers were registered
    // with this same client in registerMem.
    const std::shared_ptr<iS3RdmaTokenClient> &tokenClient = clientFor(local.getType());
    if (!tokenClient->isConnected()) {
        NIXL_ERROR << "S3 RDMA token client is not connected.";
        return NIXL_ERR_BACKEND;
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
                tokenClient->cuObjPut(&req.ctx, (void *)req.addr, req.size, req.offset);
            if (cuda_status < 0) {
                NIXL_ERROR << "cuObjPut failed with status: " << cuda_status;
                return NIXL_ERR_BACKEND;
            }
        } else if (operation == NIXL_READ) {
            ssize_t cuda_status =
                tokenClient->cuObjGet(&req.ctx, (void *)req.addr, req.size, req.offset);
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
