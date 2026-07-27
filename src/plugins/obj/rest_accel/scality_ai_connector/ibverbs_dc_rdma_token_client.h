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
     * @param nic_ips     IPv4 addresses selecting the RDMA devices (one DC context
     *                    per IP); memory regions are round-robined across them.
     * @param dc_key      DC access key the server's DCI side must present.
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
    /// A caller registration may fan out into several adjacent Regions on
    /// different NICs; every piece carries the caller's base address in
    /// parent_base so release can find its siblings.
    struct Region {
        ibv_mr *mr = nullptr;
        size_t len = 0;
        int nic_idx = 0;
        uint32_t dctn = 0;
        int dev_id = -1; ///< GPU ordinal (VRAM) or -1 (host), for the assignment dump.
        uintptr_t parent_base = 0; ///< Address the caller registered; equals the
                                   ///< piece base for a single-piece registration.
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

    /// Pick the NIC index to register a buffer on. dev_id < 0 (host memory)
    /// balances across all NICs; dev_id >= 0 (VRAM) prefers the GPU's affine
    /// rails but can overflow onto the others (see affineNicsFor). Pure: the
    /// load is bumped by registerPiece. Caller must hold mu_.
    int
    selectNicFor(int dev_id);
    /// Least-loaded NIC among candidates (fewest live registrations; ties break
    /// to the lowest index). Pure. Caller must hold mu_.
    int
    leastLoadedNic(const std::vector<int> &candidates);
    /// Register [ptr, ptr+size) as a single MR on the selected NIC, recording it
    /// under parent_base. Returns false and registers nothing on failure.
    /// Caller must hold mu_.
    bool
    registerPiece(void *ptr, size_t size, int dev_id, int nic_idx, uintptr_t parent_base);
    /// Deregister every piece belonging to the registration at parent_base and
    /// drop its NIC load. Returns the number of pieces released (0 if unknown).
    /// Caller must hold mu_.
    size_t
    releaseRegistration(uintptr_t parent_base);
    /// Resolve (and cache) the set of NIC indices affine to a GPU. Prefers NICs
    /// that share a PCIe switch with the GPU (PXB/PIX-local); when no NIC is
    /// switch-local (the RDMA NICs sit on a shared bridge equidistant from every
    /// GPU in the node), keeps all same-NUMA NICs so registrations spread across
    /// both rails. Falls back to all NICs when no topology info is available.
    /// Caller must hold mu_.
    const std::vector<int> &
    affineNicsFor(int dev_id);

    std::vector<NicCtx> nics_;
    bool connected_ = false;
    /// All NIC indices [0, nics_.size()); candidate set for host-memory balancing.
    std::vector<int> all_nics_;
    /// Live registration count per NIC, used to balance new registrations across
    /// the affine set (a per-GPU cursor can't spread when each GPU registers once).
    std::vector<uint64_t> nic_load_;
    /// dev_id -> affine NIC indices, resolved once per GPU.
    std::map<int, std::vector<int>> gpu_affine_nics_;
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
