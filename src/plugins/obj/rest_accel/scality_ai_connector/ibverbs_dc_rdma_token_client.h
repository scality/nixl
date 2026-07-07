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

#ifndef NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_IBVERBS_DC_RDMA_TOKEN_CLIENT_H
#define NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_IBVERBS_DC_RDMA_TOKEN_CLIENT_H

#include <cstdint>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

#include "dc_descriptor.h"
#include "rdma_token_client.h"

/**
 * In-process libibverbs DC (Dynamically Connected) RDMA token client.
 *
 * Bypasses cuObject to spread host-memory (DRAM) transfers across multiple NICs:
 * cuObject pins every host registration to a single NIC, whereas this client
 * opens one DC target context per NIC and round-robins memory regions across
 * them, building the DC descriptor itself with the chosen NIC's GID. The Scality
 * server accepts the same wire format produced by the bench rdma-helper.
 *
 * The DC target is passive: the server (DCI side) initiates and performs all
 * one-sided RDMA, so this client runs no background threads, CM listener, or
 * CQ polling. Ported from bench/cmd/rdma-helper/rdma_dc.c.
 */
class IbverbsDcRdmaTokenClient : public iRdmaTokenClient {
public:
    /**
     * @param nic_ips  IPv4 addresses selecting the RDMA devices (one DC context
     *                 per IP); memory regions are round-robined across them.
     * @param dc_key   DC access key the server's DCI side must present.
     */
    IbverbsDcRdmaTokenClient(const std::vector<std::string> &nic_ips, uint64_t dc_key);
    ~IbverbsDcRdmaTokenClient() override;

    IbverbsDcRdmaTokenClient(const IbverbsDcRdmaTokenClient &) = delete;
    IbverbsDcRdmaTokenClient &
    operator=(const IbverbsDcRdmaTokenClient &) = delete;

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
    /// Per-NIC DC target context, created once at construction.
    struct NicCtx {
        ibv_context *ctx = nullptr;
        ibv_pd *pd = nullptr;
        ibv_cq *cq = nullptr;
        ibv_srq *srq = nullptr;
        int num_lag_ports = 1;
        ibv_qp *dct_qps[kDcMaxLagPorts] = {};
        uint32_t dctns[kDcMaxLagPorts] = {};
        uint16_t lid = 0;
        uint8_t gid[16] = {};
        int gid_index = 0;
        uint64_t lag_seq = 0; ///< round-robin cursor over this NIC's LAG ports
        std::string dev_name;  ///< ib device name (e.g. mlx5_1), for logging.
        std::string pci_path;  ///< canonical /sys/devices/... PCIe path of the NIC.
        int numa_node = -1;    ///< NUMA node of the NIC, or -1 if unknown.
    };

    /// One registered memory region: its MR plus the NIC/DCTN it was bound to.
    struct Region {
        ibv_mr *mr = nullptr;
        size_t len = 0;
        int nic_idx = 0;
        uint32_t dctn = 0;
        int dev_id = -1; ///< GPU ordinal (VRAM) or -1 (host), for the assignment dump.
    };

    /// Build (and INIT->RTR) a NIC's DC context for the given IP. Returns false on error.
    bool
    setupNic(const std::string &ip, uint64_t dc_key, NicCtx &out);
    /// Detect bond width for an IB device via sysfs; clamps to [1, kDcMaxLagPorts].
    static int
    numLagPorts(const char *dev_name);
    /// Build the descriptor for the region covering ptr; empty string if not found.
    std::string
    descriptorFor(void *ptr, size_t size);
    /// One-shot NIXL_INFO dump of the GPU->NIC buffer assignment. Caller holds mu_.
    void
    logAssignment();

    /// Pick the NIC index to register a buffer on. dev_id < 0 (host memory) uses
    /// the global round-robin; dev_id >= 0 (VRAM) round-robins within the GPU's
    /// PCIe-affine NIC set (see affineNicsFor). Caller must hold mu_.
    int
    selectNicFor(int dev_id);
    /// Resolve (and cache) the set of NIC indices PCIe-closest to a GPU:
    /// longest common PCIe-path prefix, else same NUMA node, else all NICs.
    /// Caller must hold mu_.
    const std::vector<int> &
    affineNicsFor(int dev_id);

    std::vector<NicCtx> nics_;
    bool connected_ = false;
    uint64_t reg_counter_ = 0; ///< round-robin cursor over NICs (host memory)
    /// dev_id -> PCIe-affine NIC indices, resolved once per GPU.
    std::map<int, std::vector<int>> gpu_affine_nics_;
    /// dev_id -> round-robin cursor within that GPU's affine NIC set.
    std::map<int, uint64_t> gpu_reg_cursor_;
    /// Guard so the GPU->NIC assignment is dumped once, at the first transfer.
    bool assignment_logged_ = false;
    /// RoCE service level for the DCT AV. Under `trust pcp` this selects the
    /// egress priority (SL -> PCP), so 3 targets the lossless PFC lane by
    /// default. Overridable via UCX_IB_SL (0-15), shared with the UCX backend.
    uint8_t sl_ = 3;
    /// RoCE traffic class (ToS byte; RoCEv2 DSCP = tc >> 2) for the DCT AV.
    /// Only relevant under `trust dscp`. 0 = default lane; overridable via
    /// UCX_IB_TRAFFIC_CLASS, shared with the UCX backend.
    uint8_t traffic_class_ = 0;

    mutable std::mutex mu_;
    /// base address -> region, ordered so a sub-address can be found by range.
    std::map<uintptr_t, Region> regions_;
};

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_IBVERBS_DC_RDMA_TOKEN_CLIENT_H
