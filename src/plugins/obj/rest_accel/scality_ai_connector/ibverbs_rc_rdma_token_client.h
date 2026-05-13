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

#ifndef NIXL_OBJ_PLUGIN_S3_SCALITY_AI_CONNECTOR_IBVERBS_RC_RDMA_TOKEN_CLIENT_H
#define NIXL_OBJ_PLUGIN_S3_SCALITY_AI_CONNECTOR_IBVERBS_RC_RDMA_TOKEN_CLIENT_H

#if defined(USE_IBVERBS_RC_RDMA)

#include "rdma_token_client.h"

#include <atomic>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

#include <rdma/rdma_cma.h>
#include <infiniband/verbs.h>

/**
 * Per-registration RDMA resources.
 * One RdmaMemRegion is created for each cuMemObjGetDescriptor() call and torn
 * down by cuMemObjPutDescriptor().
 */
struct RdmaMemRegion {
    rdma_event_channel *cm_channel = nullptr;
    rdma_cm_id *listen_id = nullptr;
    ibv_pd *pd = nullptr;
    ibv_mr *mr = nullptr;
    uintptr_t base_addr = 0;
    uint32_t rkey = 0;
    uint16_t port = 0;
    /// Live QPs accepted on this listener (kept alive for connection reuse).
    std::vector<rdma_cm_id *> active_conns;
};

/**
 * libibverbs RC-based RDMA token client.
 *
 * Acts as an RDMA *target*: for each registered memory region it creates an
 * RDMA CM listener, registers the buffer with ibv_reg_mr, and advertises a
 * token string "IP:PORT:ADDR:SIZE:RKEY" that the server uses to connect
 * back and perform RDMA READ/WRITE operations.
 *
 * A single background thread handles all CM events (connect requests,
 * established connections, disconnects) via epoll across all active listeners.
 */
class IbverbsRcRdmaTokenClient : public iRdmaTokenClient {
public:
    /**
     * @param ops        User-supplied I/O callbacks (get/put).
     * @param advertiseIp  Override IP for the RDMA token.  Empty string means
     *                     auto-detect from the first RDMA device's parent netdev.
     */
    IbverbsRcRdmaTokenClient(CUObjOps_t &ops, const std::string &advertiseIp = "");
    ~IbverbsRcRdmaTokenClient() override;

    bool isConnected() const override;
    cuObjErr_t cuMemObjGetDescriptor(void *ptr, size_t size) override;
    cuObjErr_t cuMemObjPutDescriptor(void *ptr) override;
    ssize_t cuObjGet(void *ctx, void *ptr, size_t size, loff_t offset,
                     loff_t buf_offset = 0) override;
    ssize_t cuObjPut(void *ctx, void *ptr, size_t size, loff_t offset,
                     loff_t buf_offset = 0) override;

    /** Return total number of live RDMA QPs across all registered regions. */
    size_t getActiveConnectionCount() const;

private:
    /** Resolve the advertise IP from the first RDMA device's parent netdev. */
    static std::string detectAdvertiseIp();

    /** Background thread: epoll_wait on all CM channel fds, handle events. */
    void acceptLoop();

    /** Process a single RDMA CM event for the given region. */
    void handleCmEvent(rdma_cm_event *event, RdmaMemRegion &region);

    /** Find the registered region containing the given address. Caller must hold mu_. */
    const RdmaMemRegion *findRegion(uintptr_t addr) const;

    CUObjOps_t userOps_;
    std::string advertiseIp_;

    mutable std::mutex mu_;
    std::map<void *, RdmaMemRegion> regions_;

    std::thread acceptThread_;
    std::atomic<bool> running_{false};
    int epollFd_ = -1;
    bool connected_ = false;
};

#endif // USE_IBVERBS_RC_RDMA
#endif // NIXL_OBJ_PLUGIN_S3_SCALITY_AI_CONNECTOR_IBVERBS_RC_RDMA_TOKEN_CLIENT_H
