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
 * opens one DC target context per NIC and registers every buffer on all of them,
 * building the DC descriptor itself with the chosen NIC's GID. The Scality
 * server accepts the same wire format produced by the bench rdma-helper.
 *
 * A descriptor carries the request's own address and length, and an rkey covers
 * any sub-range of its MR, so the NIC a request travels on is decided when its
 * descriptor is built rather than fixed at registration. That keeps one MR per
 * (buffer, NIC) instead of one per request-sized offset, and lets rail selection
 * follow the live load: see pickRail and kNonAffineHandicap.
 *
 * The DC target is passive: the server (DCI side) initiates and performs all
 * one-sided RDMA, so this client runs no background threads, CM listener, or
 * CQ polling. Ported from bench/cmd/rdma-helper/rdma_dc.c.
 */
class IbverbsDcRdmaTokenClient : public iRdmaTokenClient {
public:
    /**
     * @param nic_ips     IPv4 addresses selecting the RDMA devices (one DC context
     *                    per IP); every buffer is registered on all of them.
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

    /// One rail of a registered buffer: an MR covering the WHOLE buffer on that
    /// NIC. An rkey is valid for any sub-range of its MR, so a request at any
    /// offset can be described from any rail the buffer is registered on.
    struct RailMr {
        ibv_mr *mr = nullptr;
        int nic_idx = 0;
    };

    /// A caller registration, held on every NIC that accepted it. The rail a
    /// request travels on is chosen when its descriptor is built, not here, so
    /// the choice can follow live load instead of being fixed at registration.
    struct Buffer {
        size_t len = 0;
        int dev_id = -1; ///< GPU ordinal (VRAM) or -1 (host memory).
        std::vector<RailMr> rails;
    };

    /// Build (and INIT->RTR) a NIC's DC context for the given IP. Returns false on error.
    bool
    setupNic(const std::string &ip, uint64_t dc_key, NicCtx &out);
    /// Detect bond width for an IB device via sysfs; clamps to [1, kDcMaxLagPorts].
    static int
    numLagPorts(const char *dev_name);
    /// Build the descriptor for the buffer covering ptr; empty string if none does.
    std::string
    descriptorFor(void *ptr, size_t size);
    /// One-shot NIXL_INFO dump of the registered MR layout. Caller holds mu_.
    void
    logLayout();
    /// NIXL_INFO summary of how requests were spread over the rails. `detailed`
    /// adds the per-GPU breakdown under the one-line summary.
    void
    logRequestSpread(bool detailed);

    /// Index into buf.rails for the next request against buf. Prefers the GPU's
    /// affine rails, handing the rest kNonAffineHandicap so they stay unused
    /// while the affine rails keep up and absorb the excess once they do not.
    /// Caller must hold mu_.
    size_t
    pickRail(const Buffer &buf);
    /// Register [ptr, ptr+size) on one NIC. Returns nullptr on failure.
    ibv_mr *
    registerRail(void *ptr, size_t size, int nic_idx);
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
    /// Requests handed out per NIC. Drives rail selection: a NIC that has taken
    /// fewer requests is the cheaper next hop, so an idle rail is picked up
    /// automatically and a busy one sheds to its neighbours.
    std::vector<uint64_t> nic_issued_;
    /// Requests that left the owning GPU's NUMA node. The headline number: it
    /// should sit near 0 whenever every NUMA node has GPUs driving it, and rise
    /// only when one node's rails have no local work to keep them busy.
    uint64_t cross_numa_ = 0;
    /// Requests since the last spread summary.
    uint64_t since_spread_log_ = 0;
    /// dev_id -> requests sent down each NIC, for the teardown dump.
    std::map<int, std::vector<uint64_t>> gpu_nic_requests_;
    /// dev_id -> affine NIC indices, resolved once per GPU.
    std::map<int, std::vector<int>> gpu_affine_nics_;
    /// dev_id -> NUMA node, cached alongside gpu_affine_nics_ for logging.
    std::map<int, int> gpu_numa_;
    /// Guard so the MR layout is dumped once, at the first transfer.
    bool layout_logged_ = false;
    /// RoCE service level for the DCT AV. Under `trust pcp` this selects the
    /// egress priority (SL -> PCP), so 3 targets the lossless PFC lane by
    /// default. Overridable via UCX_IB_SL (0-15), shared with the UCX backend.
    uint8_t sl_ = 3;
    /// RoCE traffic class (ToS byte; RoCEv2 DSCP = tc >> 2) for the DCT AV.
    /// Only relevant under `trust dscp`. 0 = default lane; overridable via
    /// UCX_IB_TRAFFIC_CLASS, shared with the UCX backend.
    uint8_t traffic_class_ = 0;
    // Registration cost, reported with the rail summary. Pinning GPU pages for RDMA
    // is not free and a caller that registers per transfer pays it on the critical
    // path, so this says whether the MR count is worth attacking: a loader doing one
    // transfer per tensor group registers rails-per-group times, where registering
    // a reused pool once would cost rails-per-pool. Guarded by mu_.
    std::size_t reg_calls_ = 0;
    std::size_t reg_rails_ = 0;
    std::size_t reg_us_ = 0;
    std::size_t dereg_us_ = 0;
    mutable std::mutex mu_;
    /// base address -> buffer, ordered so a sub-address can be found by range.
    std::map<uintptr_t, Buffer> buffers_;
};

#endif // NIXL_SRC_PLUGINS_OBJ_REST_ACCEL_SCALITY_AI_CONNECTOR_IBVERBS_DC_RDMA_TOKEN_CLIENT_H
