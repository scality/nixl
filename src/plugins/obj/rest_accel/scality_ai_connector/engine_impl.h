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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_ENGINE_IMPL_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_ENGINE_IMPL_H

#include "obj_backend.h"
#include "rest_accel/scality_ai_connector/client.h"
#include "rest_accel/scality_ai_connector/rdma_token_client.h"
#include "rest_accel/scality_ai_connector/split.h"
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

/**
 * Scality AI Connector RDMA Engine Implementation.
 * Provides RDMA-accelerated object storage operations using Scality's
 * AI Connector HTTP service. Implements nixlObjEngineImpl directly and
 * uses the cuObject API for GPU-direct storage operations.
 *
 * The RDMA descriptor obtained from cuObject is transmitted via the
 * x-scal-rdma custom HTTP header, enabling the connector to perform the data
 * transfer via RDMA instead of the HTTP body.
 */
class ScalityObjEngineImpl : public nixlObjEngineImpl {
public:
    /**
     * Constructor that initializes the Scality AI Connector engine.
     * Creates both the HTTP client and cuObject client for RDMA operations.
     *
     * @param init_params Backend initialization parameters
     */
    explicit ScalityObjEngineImpl(const nixlBackendInitParams *init_params);

    /**
     * Constructor that accepts an injected client (for testing).
     *
     * @param init_params Backend initialization parameters
     * @param connector_client Pre-configured client (can be mock for testing)
     */
    ScalityObjEngineImpl(const nixlBackendInitParams *init_params,
                         std::shared_ptr<iRestClient> connector_client);

    nixl_mem_list_t
    getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

    nixl_status_t
    queryMem(const nixl_reg_dlist_t &descs, std::vector<nixl_query_resp_t> &resp) const override;

    nixl_status_t
    prepXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             const std::string &local_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args) const override;

    nixl_status_t
    postXfer(const nixl_xfer_op_t &operation,
             const nixl_meta_dlist_t &local,
             const nixl_meta_dlist_t &remote,
             const std::string &remote_agent,
             nixlBackendReqH *&handle,
             const nixl_opt_b_args_t *opt_args = nullptr) const override;

    nixl_status_t
    checkXfer(nixlBackendReqH *handle) const override;

    nixl_status_t
    releaseReqH(nixlBackendReqH *handle) const override;

private:
    /// Token client for a given segment type: DRAM and VRAM both use the
    /// ibverbs DC client (multi-NIC, GPU/NIC-affinity aware); anything else falls
    /// back to cuClient_. hostClient_ is built lazily on the first DRAM/VRAM
    /// registration, so this returns the ibverbs client only once it exists.
    ///
    /// prepXfer only ever asks about the local segment, which is DRAM or VRAM, so
    /// in practice the fallback is not taken and cuClient_ is never built.
    const std::shared_ptr<iRdmaTokenClient> &
    clientFor(const nixl_mem_t &nixl_mem) const {
        if ((nixl_mem == DRAM_SEG || nixl_mem == VRAM_SEG) && hostClient_) {
            return hostClient_;
        }
        return ensureCuClient();
    }

    /// Build the cuObject DC client on first use and return it.
    ///
    /// Deferred because construction costs around 1.4 seconds and nothing on the
    /// DRAM/VRAM data path needs it. A connect failure is not fatal: the caller
    /// checks isConnected() and fails that one transfer.
    const std::shared_ptr<iRdmaTokenClient> &
    ensureCuClient() const;

    /// Build the ibverbs DC client (once) from the resolved NIC list. Returns
    /// NIXL_ERR_BACKEND if no NICs were resolved or the client fails to connect.
    nixl_status_t
    ensureHostClient();

    /// Maps device IDs to object keys
    std::unordered_map<uint64_t, std::string> devIdToObjKey_;
    /// RDMA token client (DC via cuObjClient); retained for OBJ only, no longer
    /// on the DRAM/VRAM data path (kept until a follow-up removal cleanup).
    /// Mutable so clientFor(), which is const, can build it on first use.
    mutable std::shared_ptr<iRdmaTokenClient> cuClient_;
    mutable std::mutex cuClientMu_;
    /// libibverbs DC token client for DRAM/VRAM multi-NIC spreading (lazy).
    std::shared_ptr<iRdmaTokenClient> hostClient_;
    std::mutex hostClientMu_;
    /// RDMA NIC specifiers (IPv4 or device names) for DRAM/VRAM, resolved at
    /// construction from customParams 'rdma_nics' or cufile.json.
    std::vector<std::string> rdmaNics_;
    /// DC access key for the ibverbs DC client.
    uint64_t dcKey_ = 0xffeeddccULL;
    /// Bytes per object request. A transfer descriptor of any size is cut into
    /// requests of at most this, so callers hand down whole tensors and the
    /// backend decides the wire granularity. 0 disables splitting (one request
    /// per descriptor, whatever its size). Resolved from customParams
    /// 'split_size' at construction.
    size_t splitSize_ = kDefaultSplitSize;
    /// Whether host (DRAM) transfers use RDMA. True by default. Set false via the
    /// 'dram_rdma' parameter and DRAM pins nothing: registerMem creates no MR and
    /// needs no NIC list, and transfers read the object over plain HTTP straight
    /// into the caller's buffer.
    ///
    /// Registration cost tracks the number of memory regions, not their size: an
    /// ibv_reg_mr costs about 1 ms and its ibv_dereg_mr about 0.65 ms whatever the
    /// length, and a buffer is registered on every rail. An 8-byte metadata read
    /// therefore costs several ms of pinning to move 8 bytes, which HTTP does not.
    ///
    /// Scoped to this backend instance and to DRAM, so VRAM transfers are
    /// unaffected -- but it is not per-buffer, since the backend's registerMem
    /// takes no per-call parameters. A caller that also moves bulk data through
    /// DRAM on the same backend would get HTTP for that too.
    bool dramRdma_ = true;
    /// Scality AI Connector HTTP client with RDMA support
    std::shared_ptr<iRestClient> connectorClient_;
};

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_ENGINE_IMPL_H
