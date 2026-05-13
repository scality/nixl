#include "engine_impl.h"
#include "client.h"
#include "common/nixl_log.h"
#include <absl/strings/str_format.h>
#ifdef HAVE_CUOBJ_CLIENT
#include "cuobj_rdma_token_client.h"
#endif
#ifdef USE_IBVERBS_RC_RDMA
#include "ibverbs_rc_rdma_token_client.h"
#endif
#include <cstdlib>
#include <memory>
#include <future>
#include <vector>
#include <chrono>
#include <algorithm>

#include "obj_engine_registry.h"

namespace {

objAccelEngineRegistrar reg_scality(
    "scality_ai_connector",
    [](const nixlBackendInitParams *p) { return std::make_unique<ScalityObjEngineImpl>(p); },
    [](const nixlBackendInitParams *p, std::shared_ptr<iS3Client>, std::shared_ptr<iS3Client>) {
        return std::make_unique<ScalityObjEngineImpl>(p);
    });

/**
 * RDMA context structure for cuObject operations.
 */
typedef struct rdma_ctx {
    /// RDMA descriptor string
    std::string rdma_desc;
} rdma_ctx_t;

/**
 * Helper: look up a key in customParams with a default value.
 */
std::string
getParamOrDefault(nixl_b_params_t *params, const std::string &key,
                  const std::string &defaultVal) {
    if (!params) return defaultVal;
    auto it = params->find(key);
    if (it != params->end()) return it->second;
    return defaultVal;
}

/**
 * Validate parameters for prepXfer operation.
 */
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

    if (remote_agent != local_agent)
        NIXL_WARN << absl::StrFormat(
            "Warning: Remote agent doesn't match the requesting agent (%s). Got %s",
            local_agent,
            remote_agent);

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

/**
 * Transfer request handle for Scality AI Connector RDMA operations.
 */
class scalityObjTransferRequestH {
public:
    uintptr_t addr;
    size_t size;
    size_t offset;
    std::string rdma_desc;
    std::string obj_key;
    rdma_ctx_t ctx;

    scalityObjTransferRequestH() : addr(0), size(0), offset(0), rdma_desc(""), obj_key("") {}

    scalityObjTransferRequestH(uintptr_t a, size_t s, size_t off)
        : addr(a),
          size(s),
          offset(off),
          rdma_desc(""),
          obj_key("") {}

    ~scalityObjTransferRequestH() = default;
};

/**
 * Backend request handle for Scality AI Connector RDMA operations.
 * Manages multiple transfer requests and their completion futures.
 */
class nixlScalityObjBackendReqH : public nixlBackendReqH {
public:
    std::vector<scalityObjTransferRequestH> reqs_;
    std::vector<std::future<nixl_status_t>> statusFutures_;

    nixlScalityObjBackendReqH() = default;
    ~nixlScalityObjBackendReqH() = default;

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
        return has_pending ? NIXL_IN_PROG : NIXL_SUCCESS;
    }
};

/**
 * Metadata for Scality AI Connector RDMA operations.
 */
class nixlScalityObjMetadata : public nixlBackendMD {
public:
    nixlScalityObjMetadata(nixl_mem_t nixl_mem, uint64_t dev_id, std::string obj_key)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(dev_id),
          objKey(obj_key),
          localAddr(0) {}

    nixlScalityObjMetadata(nixl_mem_t nixl_mem, uintptr_t addr)
        : nixlBackendMD(true),
          nixlMem(nixl_mem),
          devId(0),
          objKey(""),
          localAddr(addr) {}

    ~nixlScalityObjMetadata() = default;

    nixl_mem_t nixlMem;
    uint64_t devId;
    std::string objKey;
    uintptr_t localAddr;
};

/**
 * cuObject get callback — extracts RDMA descriptor from cuFile RDMA info.
 */
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

    void *ctx = iRdmaTokenClient::getCtx(handle);
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

/**
 * cuObject put callback — extracts RDMA descriptor from cuFile RDMA info.
 */
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

    void *ctx = iRdmaTokenClient::getCtx(handle);
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

/// cuObject I/O operations for Scality AI Connector
CUObjIOOps scality_ops = {.get = objectGet, .put = objectPut};

/**
 * Create the RDMA token client based on the transport parameter.
 * Returns {client, useDcTransport}.
 */
std::pair<std::shared_ptr<iRdmaTokenClient>, bool>
createRdmaClient(nixl_b_params_t *customParams) {
    std::string transport = getParamOrDefault(customParams, "rdma_transport", "auto");
    std::string advertiseIp = getParamOrDefault(customParams, "rdma_advertise_ip", "");

    if (transport == "dc") {
#if defined(HAVE_CUOBJ_CLIENT)
        auto client = std::make_shared<CuObjRdmaTokenClient>(scality_ops);
        return {client, true};
#else
        NIXL_ERROR << "DC transport requested but HAVE_CUOBJ_CLIENT not available";
        return {nullptr, false};
#endif
    }

    if (transport == "rc") {
#if defined(USE_IBVERBS_RC_RDMA)
        auto client = std::make_shared<IbverbsRcRdmaTokenClient>(scality_ops, advertiseIp);
        return {client, false};
#else
        NIXL_ERROR << "RC transport requested but USE_IBVERBS_RC_RDMA not available";
        return {nullptr, false};
#endif
    }

    // "auto" — try DC first, fallback to RC
#if defined(HAVE_CUOBJ_CLIENT)
    auto dc = std::make_shared<CuObjRdmaTokenClient>(scality_ops);
    if (dc->isConnected()) {
        NIXL_INFO << "Auto-selected DC transport";
        return {dc, true};
    }
    NIXL_WARN << "DC transport not connected, falling back to RC";
#endif

#if defined(USE_IBVERBS_RC_RDMA)
    auto rc = std::make_shared<IbverbsRcRdmaTokenClient>(scality_ops, advertiseIp);
    NIXL_INFO << "Auto-selected RC transport";
    return {rc, false};
#endif

    NIXL_ERROR << "No RDMA transport available (need HAVE_CUOBJ_CLIENT or USE_IBVERBS_RC_RDMA)";
    return {nullptr, false};
}

} // namespace

ScalityObjEngineImpl::ScalityObjEngineImpl(const nixlBackendInitParams *init_params)
    : ScalityObjEngineImpl(init_params, nullptr) {
}

ScalityObjEngineImpl::ScalityObjEngineImpl(const nixlBackendInitParams *init_params,
                                           std::shared_ptr<iScalityRdmaClient> connector_client) {
    if (connector_client) {
        connectorClient_ = connector_client;
    } else {
        connectorClient_ = std::make_shared<ScalityClient>(init_params->customParams);
    }

    NIXL_INFO << "Object storage backend initialized with Scality AI Connector RDMA client";

    auto [client, isDc] = createRdmaClient(init_params->customParams);
    cuClient_ = client;
    useDcTransport_ = isDc;

    if (!cuClient_ || !cuClient_->isConnected()) {
        NIXL_ERROR << "RDMA token client failed to connect.";
        return;
    }
}

nixl_status_t
ScalityObjEngineImpl::registerMem(const nixlBlobDesc &mem,
                                  const nixl_mem_t &nixl_mem,
                                  nixlBackendMD *&out) {
    if (!cuClient_ || !cuClient_->isConnected()) {
        NIXL_ERROR << "RDMA token client is not connected.";
        return NIXL_ERR_BACKEND;
    }

    auto supported_mems = {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    if (std::find(supported_mems.begin(), supported_mems.end(), nixl_mem) == supported_mems.end())
        return NIXL_ERR_NOT_SUPPORTED;

    if (nixl_mem == OBJ_SEG) {
        std::unique_ptr<nixlScalityObjMetadata> obj_md =
            std::make_unique<nixlScalityObjMetadata>(
                nixl_mem,
                mem.devId,
                mem.metaInfo.empty() ? std::to_string(mem.devId) : mem.metaInfo);
        devIdToObjKey_[mem.devId] = obj_md->objKey;
        out = obj_md.release();
    } else if ((nixl_mem == DRAM_SEG) || (nixl_mem == VRAM_SEG)) {
        if (mem.len > CUOBJ_MAX_MEMORY_REG_SIZE) {
            NIXL_ERROR << "Memory size too large for cuObject registration: " << mem.len;
            return NIXL_ERR_NOT_SUPPORTED;
        }

        NIXL_DEBUG << absl::StrFormat(
            "registerMem: addr=0x%016x, len=%zu, nixl_mem=%d",
            mem.addr, mem.len, nixl_mem);
        std::unique_ptr<nixlScalityObjMetadata> mem_md =
            std::make_unique<nixlScalityObjMetadata>(nixl_mem, mem.addr);

        cuObjErr_t cuda_status = cuClient_->cuMemObjGetDescriptor((void *)(mem.addr), mem.len);
        if (cuda_status != CU_OBJ_SUCCESS) {
            NIXL_ERROR << "cuMemObjGetDescriptor failed with status: " << cuda_status;
            const char *cfg = std::getenv("CUFILE_ENV_PATH_JSON");
            NIXL_ERROR << "Hint: check rdma_dev_addr_list in "
                       << (cfg ? cfg : "/etc/cufile.json");
            return NIXL_ERR_BACKEND;
        }
        out = mem_md.release();
    }

    return NIXL_SUCCESS;
}

nixl_status_t
ScalityObjEngineImpl::deregisterMem(nixlBackendMD *meta) {
    nixlScalityObjMetadata *md = static_cast<nixlScalityObjMetadata *>(meta);
    if (md) {
        if (md->nixlMem == OBJ_SEG) {
            std::unique_ptr<nixlScalityObjMetadata> obj_md_ptr =
                std::unique_ptr<nixlScalityObjMetadata>(md);
            devIdToObjKey_.erase(obj_md_ptr->devId);
        } else if ((md->nixlMem == DRAM_SEG) || (md->nixlMem == VRAM_SEG)) {
            std::unique_ptr<nixlScalityObjMetadata> mem_md_ptr =
                std::unique_ptr<nixlScalityObjMetadata>(md);
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
ScalityObjEngineImpl::prepXfer(const nixl_xfer_op_t &operation,
                               const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent,
                               const std::string &local_agent,
                               nixlBackendReqH *&handle,
                               const nixl_opt_b_args_t *opt_args) const {
    if (!cuClient_ || !cuClient_->isConnected()) {
        NIXL_ERROR << "RDMA token client is not connected.";
        return NIXL_ERR_BACKEND;
    }

    if (!isValidPrepXferParams(operation, local, remote, remote_agent, local_agent)) {
        return NIXL_ERR_INVALID_PARAM;
    }

    auto req_h = std::make_unique<nixlScalityObjBackendReqH>();

    for (int i = 0; i < local.descCount(); ++i) {
        scalityObjTransferRequestH req(local[i].addr, local[i].len, remote[i].addr);

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
        NIXL_DEBUG << absl::StrFormat(
            "prepXfer: addr=0x%016x, size=%zu, offset=%zu, rdma_desc=%s",
            req.addr, req.size, req.offset, req.rdma_desc);

        req_h->reqs_.push_back(req);
    }

    handle = req_h.release();
    return NIXL_SUCCESS;
}

nixl_status_t
ScalityObjEngineImpl::postXfer(const nixl_xfer_op_t &operation,
                               const nixl_meta_dlist_t &local,
                               const nixl_meta_dlist_t &remote,
                               const std::string &remote_agent,
                               nixlBackendReqH *&handle,
                               const nixl_opt_b_args_t *opt_args) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }

    nixlScalityObjBackendReqH *req_h = static_cast<nixlScalityObjBackendReqH *>(handle);

    for (const auto &req : req_h->reqs_) {
        auto status_promise = std::make_shared<std::promise<nixl_status_t>>();
        req_h->statusFutures_.push_back(status_promise->get_future());

        if (operation == NIXL_WRITE) {
            connectorClient_->putObjectRdmaAsync(req.obj_key,
                                               req.addr,
                                               req.size,
                                               req.offset,
                                               req.rdma_desc,
                                               [status_promise](bool success) {
                                                   status_promise->set_value(
                                                       success ? NIXL_SUCCESS : NIXL_ERR_BACKEND);
                                               });
        } else {
            connectorClient_->getObjectRdmaAsync(req.obj_key,
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
ScalityObjEngineImpl::checkXfer(nixlBackendReqH *handle) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlScalityObjBackendReqH *req_h = static_cast<nixlScalityObjBackendReqH *>(handle);
    return req_h->getOverallStatus();
}

nixl_status_t
ScalityObjEngineImpl::releaseReqH(nixlBackendReqH *handle) const {
    if (handle == nullptr) {
        NIXL_ERROR << "transfer request handle is null";
        return NIXL_ERR_INVALID_PARAM;
    }
    nixlScalityObjBackendReqH *req_h = static_cast<nixlScalityObjBackendReqH *>(handle);
    delete req_h;
    return NIXL_SUCCESS;
}

nixl_status_t
ScalityObjEngineImpl::queryMem(const nixl_reg_dlist_t &descs,
                               std::vector<nixl_query_resp_t> &resp) const {
    (void)descs;
    (void)resp;
    return NIXL_ERR_NOT_SUPPORTED;
}
