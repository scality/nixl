/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_ACCEL_ENGINE_IMPL_CUOBJ_H
#define NIXL_OBJ_PLUGIN_S3_ACCEL_ENGINE_IMPL_CUOBJ_H

#include "s3_accel/engine_impl.h"
#include "s3_accel/rdma_token_client.h"

#include <cstdint>
#include <mutex>
#include <string>
#include <vector>

/**
 * Abstract base class for cuObject-accelerated S3 engines.
 *
 * Implements the cuObject pinning/unpinning lifecycle, cuObjPut/cuObjGet call
 * sequence in prepXfer, and the future/promise bridge in postXfer that adapts
 * S3 callbacks to NIXL's polling interface.  Vendor subclasses must supply the
 * RDMA-capable S3 client by overriding getClient().
 */
class S3CuObjEngineImpl : public S3AccelObjEngineImpl {
public:
    nixl_mem_list_t
    getSupportedMems() const override {
        return {OBJ_SEG, DRAM_SEG, VRAM_SEG};
    }

    nixl_status_t
    registerMem(const nixlBlobDesc &mem, const nixl_mem_t &nixl_mem, nixlBackendMD *&out) override;

    nixl_status_t
    deregisterMem(nixlBackendMD *meta) override;

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

protected:
    explicit S3CuObjEngineImpl(const nixlBackendInitParams *init_params);

    // Subclasses must provide the RDMA-capable S3 client.
    iS3Client *
    getClient() const override = 0;

    // Default token producer (cuObject DC): the OBJ path and, unless multi-rail
    // is opted in, the DRAM/VRAM path too.
    std::shared_ptr<iS3RdmaTokenClient> tokenClient_;

private:
    /// Token client for a segment type: DRAM/VRAM use the ibverbs DC client once
    /// it has been built (whenever a NIC list is resolvable); everything else,
    /// and the no-NIC-list default, uses cuObject.
    const std::shared_ptr<iS3RdmaTokenClient> &
    clientFor(const nixl_mem_t &nixl_mem) const {
        return ((nixl_mem == DRAM_SEG || nixl_mem == VRAM_SEG) && hostClient_) ? hostClient_
                                                                               : tokenClient_;
    }

    /// Build the ibverbs DC client once from the resolved NIC list. Returns
    /// NIXL_ERR_BACKEND if no NICs were resolved or the client fails to connect.
    nixl_status_t
    ensureHostClient();

    /// libibverbs DC token client for DRAM/VRAM multi-NIC spreading (lazy,
    /// opt-in). Null unless rdma_nics was set and the client was built.
    std::shared_ptr<iS3RdmaTokenClient> hostClient_;
    std::mutex hostClientMu_;
    /// RDMA NIC specifiers (IPv4 or device names like mlx5_1) for the DC client,
    /// resolved from the rdma_nics param, else cufile.json rdma_dev_addr_list.
    /// Empty means no NIC list was found, so DRAM/VRAM stay on cuObject.
    std::vector<std::string> rdmaNics_;
    /// DC access key for the ibverbs DC client (rdma_dc_key / cufile.json).
    uint64_t dcKey_ = 0xffeeddccULL;
};

#endif // NIXL_OBJ_PLUGIN_S3_ACCEL_ENGINE_IMPL_CUOBJ_H
