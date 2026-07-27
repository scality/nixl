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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_SPLIT_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_SPLIT_H

#include <cstddef>

/**
 * Request-splitting policy for the Scality AI Connector.
 *
 * A caller hands down one transfer descriptor per logical object range, whatever
 * its size, and the backend decides the wire granularity: prepXfer cuts each
 * descriptor into requests of at most split_size bytes. The DC token client uses
 * the same value to align the pieces of a fanned-out registration, so a request
 * never straddles two memory regions.
 *
 * Kept dependency-free (no cuObject, no NIXL types) so both the engine and its
 * tests can include it.
 */

/// Default bytes per object request. Overridable via the 'split_size' backend
/// parameter; 0 there disables splitting.
///
/// 8 MiB rather than 16: it doubles the requests a transfer of a given size is
/// cut into, which both raises the concurrency the endpoint sees and halves the
/// transfer size at which a registration's chunks reach every RDMA NIC. An
/// 8-GPU VRAM READ at this granularity reached 360 of 400 Gb/s.
constexpr size_t kDefaultSplitSize = 8 * 1024 * 1024;

/// How many ranged requests a descriptor of `total` bytes becomes.
///
/// Always at least 1: a zero-length descriptor still yields one empty request,
/// matching the behaviour before splitting existed. `split_size` 0 disables
/// splitting, giving one request whatever the size. Request r then covers
/// [r * split_size, min((r + 1) * split_size, total)).
constexpr size_t
objRequestCount(size_t total, size_t split_size) {
    if (split_size == 0 || total == 0) {
        return 1;
    }
    return (total + split_size - 1) / split_size;
}

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_SPLIT_H
