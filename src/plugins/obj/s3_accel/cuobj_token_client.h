/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_ACCEL_CUOBJ_TOKEN_CLIENT_H
#define NIXL_OBJ_PLUGIN_S3_ACCEL_CUOBJ_TOKEN_CLIENT_H

#include "rdma_token_client.h"

/**
 * Default token producer: a thin wrapper around NVIDIA cuObjClient (DC
 * transport).  Owns the cuObject GET/PUT callbacks that copy the returned
 * descriptor string into the per-transfer rdma_ctx_t.  Behaviour is identical
 * to the previously hard-wired cuObjClient path.
 */
class S3CuObjTokenClient : public iS3RdmaTokenClient {
public:
    S3CuObjTokenClient();
    ~S3CuObjTokenClient() override = default;

    bool
    isConnected() const override;
    cuObjErr_t
    cuMemObjGetDescriptor(void *ptr, size_t size, int dev_id = -1) override;
    cuObjErr_t
    cuMemObjPutDescriptor(void *ptr) override;
    ssize_t
    cuObjGet(void *ctx, void *ptr, size_t size, loff_t offset, loff_t buf_offset = 0) override;
    ssize_t
    cuObjPut(void *ctx, void *ptr, size_t size, loff_t offset, loff_t buf_offset = 0) override;

private:
    std::shared_ptr<cuObjClient> inner_;
};

#endif // NIXL_OBJ_PLUGIN_S3_ACCEL_CUOBJ_TOKEN_CLIENT_H
