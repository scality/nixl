/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_ACCEL_ENGINE_IMPL_CUOBJ_H
#define NIXL_OBJ_PLUGIN_S3_ACCEL_ENGINE_IMPL_CUOBJ_H

#include "s3_accel/engine_impl.h"
#include <cuobjclient.h>

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

    std::shared_ptr<cuObjClient> cuClient_;
};

#endif // NIXL_OBJ_PLUGIN_S3_ACCEL_ENGINE_IMPL_CUOBJ_H
