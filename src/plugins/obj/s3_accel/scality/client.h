/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_SCALITY_CLIENT_H
#define NIXL_OBJ_PLUGIN_S3_SCALITY_CLIENT_H

#include <memory>
#include <string_view>
#include <cstdint>
#include <aws/s3/S3Client.h>
#include <aws/core/utils/memory/stl/AWSString.h>
#include "s3_accel/client.h"
#include "s3_accel/rdma_interface.h"
#include "nixl_types.h"

/**
 * S3 Accelerated Object Client for use with Scality RING.  Inherits from the
 * accelerated S3 client and adds RDMA-backed put/get operations using the
 * cuObject API with the "x-amz-rdma-token" header.
 */
class awsS3ScalityClient : public awsS3AccelClient, public iS3RdmaClient {
public:
    /**
     * Constructor that creates an AWS S3 client for use with Scality RING from
     * custom parameters.
     * @param custom_params Custom parameters containing S3 configuration
     * @param executor Optional executor for async operations
     */
    awsS3ScalityClient(nixl_b_params_t *custom_params,
                       std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr);

    virtual ~awsS3ScalityClient() = default;

    void
    putObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       put_object_callback_t callback) override;

    void
    getObjectRdmaAsync(std::string_view key,
                       uintptr_t data_ptr,
                       size_t data_len,
                       size_t offset,
                       std::string_view rdma_desc,
                       get_object_callback_t callback) override;
};

#endif // NIXL_OBJ_PLUGIN_S3_SCALITY_CLIENT_H
