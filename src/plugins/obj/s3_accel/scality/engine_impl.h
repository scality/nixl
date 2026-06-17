/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_SCALITY_ENGINE_IMPL_H
#define NIXL_OBJ_PLUGIN_S3_SCALITY_ENGINE_IMPL_H

#include "s3_accel/engine_impl_cuobj.h"

/**
 * S3 Scality RING engine.  Thin vendor shell over S3CuObjEngineImpl.
 * Supplies the RDMA-capable awsS3ScalityClient and enforces the
 * resp_checksum=required default needed because Scality RDMA GET responses
 * carry an empty HTTP body (data delivered out-of-band over RoCEv2).
 */
class S3ScalityObjEngineImpl : public S3CuObjEngineImpl {
public:
    explicit S3ScalityObjEngineImpl(const nixlBackendInitParams *init_params);
    S3ScalityObjEngineImpl(const nixlBackendInitParams *init_params,
                           std::shared_ptr<iS3Client> s3_client);

protected:
    iS3Client *
    getClient() const override;

private:
    std::shared_ptr<iS3Client> s3Client_;
};

#endif // NIXL_OBJ_PLUGIN_S3_SCALITY_ENGINE_IMPL_H
