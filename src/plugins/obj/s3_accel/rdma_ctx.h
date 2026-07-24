/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_OBJ_PLUGIN_S3_ACCEL_RDMA_CTX_H
#define NIXL_OBJ_PLUGIN_S3_ACCEL_RDMA_CTX_H

#include <string>

/**
 * Per-transfer context passed (as void*) to iRdmaTokenClient::cuObjGet/cuObjPut.
 * The client fills rdma_desc with the RDMA descriptor (token) for the buffer:
 * either via the cuObject callback (CuObjRdmaTokenClient) or directly (a future
 * in-process ibverbs client).  Shared so every implementation agrees on the type.
 */
typedef struct rdma_ctx {
    /// RDMA descriptor string
    std::string rdma_desc;
} rdma_ctx_t;

#endif // NIXL_OBJ_PLUGIN_S3_ACCEL_RDMA_CTX_H
