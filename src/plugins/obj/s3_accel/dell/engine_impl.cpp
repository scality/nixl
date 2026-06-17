/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#include "engine_impl.h"
#include "client.h"
#include "common/nixl_log.h"

#include "obj_engine_registry.h"

namespace {

objAccelEngineRegistrar reg_dell(
    "dell",
    [](const nixlBackendInitParams *p) { return std::make_unique<S3DellObsObjEngineImpl>(p); },
    [](const nixlBackendInitParams *p, std::shared_ptr<iS3Client> s3, std::shared_ptr<iS3Client>) {
        return std::make_unique<S3DellObsObjEngineImpl>(p, std::move(s3));
    });

} // namespace

static void
applyRespChecksumDefault(const nixlBackendInitParams *init_params,
                         nixl_b_params_t &local_params,
                         nixl_b_params_t *&params_to_use) {
    // RDMA GET responses have an empty HTTP body (data delivered out-of-band over RoCEv2),
    // so SDK body-checksum validation always fails when the server returns checksum headers.
    // emplace() preserves any user-provided override.
    if (!init_params->customParams) {
        local_params["resp_checksum"] = "required";
        params_to_use = &local_params;
    } else {
        init_params->customParams->emplace("resp_checksum", "required");
        params_to_use = init_params->customParams;
    }
}

S3DellObsObjEngineImpl::S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params)
    : S3CuObjEngineImpl(init_params) {
    nixl_b_params_t local_params;
    nixl_b_params_t *params_to_use = nullptr;
    applyRespChecksumDefault(init_params, local_params, params_to_use);
    s3Client_ = std::make_shared<awsS3DellObsClient>(params_to_use, executor_);
    NIXL_INFO << "Object storage backend initialized with S3 Dell ObjectScale client";
}

S3DellObsObjEngineImpl::S3DellObsObjEngineImpl(const nixlBackendInitParams *init_params,
                                               std::shared_ptr<iS3Client> s3_client)
    : S3CuObjEngineImpl(init_params) {
    nixl_b_params_t local_params;
    nixl_b_params_t *params_to_use = nullptr;
    applyRespChecksumDefault(init_params, local_params, params_to_use);
    s3Client_ = s3_client ? s3_client
                          : std::make_shared<awsS3DellObsClient>(params_to_use, executor_);
    NIXL_INFO << "Object storage backend initialized with S3 Dell ObjectScale client";
}

iS3Client *
S3DellObsObjEngineImpl::getClient() const {
    return s3Client_.get();
}
