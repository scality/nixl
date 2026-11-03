/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#ifndef NIXL_SRC_PLUGINS_REST_RDMA_DESCRIPTOR_PROVIDER_H
#define NIXL_SRC_PLUGINS_REST_RDMA_DESCRIPTOR_PROVIDER_H

#include <cstddef>
#include <cstdint>
#include <string>

#include "nixl_types.h"

/** The server's access to a registered range, for one request. */
struct rdmaAccess {
    /// Sent to the server; empty when the access could not be opened.
    std::string descriptor;
    /// Identifies the access to closeAccess().
    uint64_t id = 0;
};

/**
 * Data-plane side of the Scality AI Connector: registers local buffers for RDMA
 * and gives the server access to a registered byte range, described by an RDMA
 * descriptor. The RDMA descriptor travels to the server in an HTTP header; the
 * server then performs the one-sided RDMA read or write on that range itself.
 */
class iRdmaDescriptorProvider {
public:
    virtual ~iRdmaDescriptorProvider() = default;

    /** True once the transport is ready to register memory. */
    virtual bool
    isConnected() const = 0;

    /**
     * Register [ptr, ptr + size) for remote RDMA access.
     * @param dev_id CUDA device ordinal of a VRAM buffer, or -1 for host memory.
     *               Implementations may use it as a NIC-affinity hint.
     */
    virtual nixl_status_t
    registerMemory(void *ptr, size_t size, int dev_id) = 0;

    /** Release the registration made by registerMemory() at ptr. */
    virtual nixl_status_t
    deregisterMemory(void *ptr) = 0;

    /** True when [ptr, ptr + size) lies within a single registration. */
    virtual bool
    isRegistered(void *ptr, size_t size) const = 0;

    /**
     * Give the server access to [ptr, ptr + size), which must lie within a single
     * registration, for one request. Every access opened must be closed with
     * closeAccess() once its request has finished.
     * @return the access; its descriptor is empty when it could not be opened.
     */
    virtual rdmaAccess
    openAccess(void *ptr, size_t size) = 0;

    /**
     * Take back an access given by openAccess(). On success the server can no
     * longer read or write the range through it, even an RDMA it performs late.
     * On failure it may still be able to: the range must not be reused.
     */
    virtual nixl_status_t
    closeAccess(uint64_t id) = 0;

    /**
     * Make the RDMA writes the server performed into the memory of GPU dev_id
     * visible to that GPU. Called once a READ into its VRAM has completed and
     * before the transfer reports success, so the caller's next kernel never
     * reads stale data. A provider whose writes need no flush keeps the default.
     */
    virtual nixl_status_t
    flushRdmaWrites(int dev_id) {
        return NIXL_SUCCESS;
    }
};

#endif // NIXL_SRC_PLUGINS_REST_RDMA_DESCRIPTOR_PROVIDER_H
