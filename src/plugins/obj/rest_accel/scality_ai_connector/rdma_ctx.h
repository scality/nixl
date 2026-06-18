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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_RDMA_CTX_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_RDMA_CTX_H

#include <string>

/**
 * Per-transfer context passed (as void*) to iRdmaTokenClient::cuObjGet/cuObjPut.
 * The client fills rdma_desc with the RDMA descriptor (token) for the buffer:
 * either via the cuObject callback (CuObjRdmaTokenClient) or directly
 * (IbverbsDcRdmaTokenClient). Shared so both implementations agree on the type.
 */
typedef struct rdma_ctx {
    /// RDMA descriptor string
    std::string rdma_desc;
} rdma_ctx_t;

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_RDMA_CTX_H
