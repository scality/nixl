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

#include "ibverbs_dc_rdma_token_client.h"

#include <arpa/inet.h>
#include <dirent.h>
#include <limits.h>
#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "common/nixl_log.h"
#include "rdma_ctx.h"

namespace {

constexpr int kDcPort = 1;

/// Handicap (in live registrations) applied to a NIC that is NOT in a GPU's
/// affine set when picking a rail for a VRAM registration. Affine rails are
/// preferred while loads are close, but once they run this many registrations
/// ahead the overflow spills onto the remaining NICs, so no NIC stays idle
/// under load. 0 would make selection pure global least-loaded (affinity
/// ignored); larger values prefer the affine rails more strongly.
constexpr uint64_t kNonAffineHandicap = 2;

/// Canonicalize a sysfs symlink to its /sys/devices/... target. Empty on failure.
std::string
canonicalPath(const std::string &path) {
    char resolved[PATH_MAX];
    if (realpath(path.c_str(), resolved) == nullptr) {
        return "";
    }
    return std::string(resolved);
}

/// Read a single integer from a sysfs file; returns fallback on any failure.
int
readIntFile(const std::string &path, int fallback = -1) {
    std::ifstream f(path);
    int v = fallback;
    if (f && (f >> v)) {
        return v;
    }
    return fallback;
}

/// Number of shared leading '/'-separated components between two paths. Used as
/// a PCIe-proximity score: a longer shared prefix means a closer shared bridge.
size_t
commonPathPrefix(const std::string &a, const std::string &b) {
    if (a.empty() || b.empty()) {
        return 0;
    }
    std::stringstream sa(a), sb(b);
    std::string ca, cb;
    size_t n = 0;
    while (std::getline(sa, ca, '/') && std::getline(sb, cb, '/')) {
        if (ca != cb) {
            break;
        }
        n++;
    }
    return n;
}

/// PCI bus id ("0000:1b:00.0", lower-case) for a CUDA device. Empty on error.
/// Stable identity for the physical GPU under CUDA_VISIBLE_DEVICES, which
/// renumbers the runtime-visible ordinals; maps directly to `nvidia-smi topo`.
std::string
gpuBusId(int dev_id) {
    char busid[32] = {};
    if (cudaDeviceGetPCIBusId(busid, sizeof(busid), dev_id) != cudaSuccess) {
        return "";
    }
    // cudaDeviceGetPCIBusId yields upper-case "0000:1B:00.0"; sysfs uses lower-case.
    for (char *p = busid; *p; ++p) {
        *p = static_cast<char>(std::tolower(static_cast<unsigned char>(*p)));
    }
    return busid;
}

/// Canonical /sys PCIe path for a CUDA device, via its PCI bus id. Empty on error.
std::string
gpuPciPath(int dev_id) {
    const std::string busid = gpuBusId(dev_id);
    if (busid.empty()) {
        return "";
    }
    return canonicalPath(std::string("/sys/bus/pci/devices/") + busid);
}

/// Map a runtime-visible GPU ordinal back to the id the user set: the entry at
/// position dev_id in CUDA_VISIBLE_DEVICES (which renumbers visible devices to
/// 0..N). Falls back to dev_id when the variable is unset or the entry is not a
/// plain integer (e.g. a UUID or MIG handle).
int
visibleToUserGpu(int dev_id) {
    const char *cvd = std::getenv("CUDA_VISIBLE_DEVICES");
    if (!cvd || dev_id < 0) {
        return dev_id;
    }
    std::stringstream ss(cvd);
    std::string tok;
    for (int i = 0; std::getline(ss, tok, ','); ++i) {
        if (i != dev_id) {
            continue;
        }
        char *end = nullptr;
        long v = std::strtol(tok.c_str(), &end, 10);
        if (end != tok.c_str() && *end == '\0' && v >= 0) {
            return static_cast<int>(v);
        }
        break; // non-integer entry; can't map to a number
    }
    return dev_id;
}

/// RoCE GID type via sysfs: 2 (RoCEv2), 1 (RoCEv1), 0 (IB), -1 on error.
int
gidTypeSysfs(const char *dev_name, int gid_index) {
    char path[256];
    snprintf(path,
             sizeof(path),
             "/sys/class/infiniband/%s/ports/1/gid_attrs/types/%d",
             dev_name,
             gid_index);
    FILE *f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    char buf[32];
    int got = (fgets(buf, sizeof(buf), f) != nullptr);
    fclose(f);
    if (!got) {
        return -1;
    }
    if (strstr(buf, "RoCE v2")) {
        return 2;
    }
    if (strstr(buf, "v1")) {
        return 1;
    }
    return 0;
}

/// True if the GID is an IPv4-mapped address (::ffff:a.b.c.d) — the routable
/// RoCEv2 GID the server expects, as opposed to a link-local fe80:: GID.
bool
isIpv4MappedGid(const uint8_t raw[16]) {
    static const uint8_t prefix[12] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    return memcmp(raw, prefix, 12) == 0;
}

/**
 * Open the RDMA device for a NIC specifier, which is either an IPv4 address or an
 * RDMA device name (e.g. "mlx5_1"):
 *  - IPv4: match the device/GID whose IPv4-mapped GID (::ffff:<addr>) equals it.
 *  - device name: open that device and select its best RoCEv2 GID.
 * Among candidates, prefer the highest GID type (RoCEv2 > RoCEv1 > IB).
 * Ported from / extends bench rdma_dc.c open_device_for_addr.
 */
ibv_context *
openDeviceForNic(const std::string &nic, int *out_gid_index) {
    *out_gid_index = 0;

    int num_devices = 0;
    ibv_device **dev_list = ibv_get_device_list(&num_devices);
    if (!dev_list || num_devices == 0) {
        NIXL_ERROR << "ibverbs_dc: no RDMA devices found";
        if (dev_list) {
            ibv_free_device_list(dev_list);
        }
        return nullptr;
    }

    in_addr v4{};
    const bool is_ip = (inet_aton(nic.c_str(), &v4) != 0);
    uint8_t target_gid[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0, 0, 0, 0};
    if (is_ip) {
        memcpy(target_gid + 12, &v4, 4);
    }

    int best_dev = -1, best_gidx = 0, best_score = -1;
    for (int d = 0; d < num_devices; d++) {
        const char *dev_name = ibv_get_device_name(dev_list[d]);
        // For a device-name specifier, only consider that device.
        if (!is_ip && nic != dev_name) {
            continue;
        }
        ibv_context *ctx = ibv_open_device(dev_list[d]);
        if (!ctx) {
            continue;
        }
        ibv_port_attr port_attr{};
        if (ibv_query_port(ctx, kDcPort, &port_attr) != 0) {
            ibv_close_device(ctx);
            continue;
        }
        for (int g = 0; g < port_attr.gid_tbl_len; g++) {
            ibv_gid gid;
            if (ibv_query_gid(ctx, kDcPort, g, &gid) != 0) {
                continue;
            }
            // IPv4 specifier: the GID must match. Device-name specifier: take the
            // device's best GID (skip the all-zero/unset entries).
            if (is_ip) {
                if (memcmp(gid.raw, target_gid, 16) != 0) {
                    continue;
                }
            } else {
                ibv_gid zero{};
                if (memcmp(gid.raw, zero.raw, 16) == 0) {
                    continue;
                }
            }
            // Prefer RoCEv2, then the routable IPv4-mapped GID over a link-local
            // fe80:: one (both are RoCEv2; the server can only route the former).
            int gtype = gidTypeSysfs(dev_name, g);
            int score = gtype * 2 + (isIpv4MappedGid(gid.raw) ? 1 : 0);
            if (score > best_score) {
                best_dev = d;
                best_gidx = g;
                best_score = score;
            }
        }
        ibv_close_device(ctx);
    }

    ibv_context *result = nullptr;
    if (best_dev >= 0) {
        result = ibv_open_device(dev_list[best_dev]);
        if (result) {
            *out_gid_index = best_gidx;
            NIXL_INFO << "ibverbs_dc: NIC " << nic << " -> "
                      << ibv_get_device_name(dev_list[best_dev]) << " gid_index=" << best_gidx
                      << " (score " << best_score << ")";
        }
    } else {
        NIXL_ERROR << "ibverbs_dc: no RDMA device found for NIC '" << nic
                   << "' (expected an IPv4 address or device name like mlx5_1)";
    }

    ibv_free_device_list(dev_list);
    return result;
}

} // namespace

int
IbverbsDcRdmaTokenClient::numLagPorts(const char *dev_name) {
    // /sys/class/infiniband/<dev>/device/net/<iface>
    char net_dir[256];
    snprintf(net_dir, sizeof(net_dir), "/sys/class/infiniband/%s/device/net", dev_name);
    DIR *d = opendir(net_dir);
    if (!d) {
        return 1;
    }
    std::string iface;
    for (dirent *ent = readdir(d); ent != nullptr; ent = readdir(d)) {
        if (ent->d_name[0] == '.') {
            continue;
        }
        iface = ent->d_name;
        break;
    }
    closedir(d);
    if (iface.empty()) {
        return 1;
    }

    // /sys/class/net/<iface>/master -> bond
    char master_link[256];
    snprintf(master_link, sizeof(master_link), "/sys/class/net/%s/master", iface.c_str());
    char master_target[256];
    ssize_t len = readlink(master_link, master_target, sizeof(master_target) - 1);
    if (len < 0) {
        return 1; // not a bond slave
    }
    master_target[len] = '\0';
    char *bond = strrchr(master_target, '/');
    bond = bond ? bond + 1 : master_target;

    // /sys/class/net/<bond>/bonding/slaves
    char slaves_path[512];
    snprintf(slaves_path, sizeof(slaves_path), "/sys/class/net/%s/bonding/slaves", bond);
    FILE *f = fopen(slaves_path, "r");
    if (!f) {
        return 1;
    }
    char buf[256];
    int got = (fgets(buf, sizeof(buf), f) != nullptr);
    fclose(f);
    if (!got) {
        return 1;
    }

    int n = countBondSlaves(buf);
    if (n < 1 || n > kDcMaxLagPorts) {
        NIXL_WARN << "ibverbs_dc: bond '" << bond << "' has " << n
                  << " slave(s), out of range [1," << kDcMaxLagPorts << "], using 1";
        return 1;
    }
    NIXL_INFO << "ibverbs_dc: bond '" << bond << "' has " << n << " slave(s) (iface " << iface
              << ")";
    return n;
}

bool
IbverbsDcRdmaTokenClient::setupNic(const std::string &ip, uint64_t dc_key, NicCtx &nic) {
    nic.ctx = openDeviceForNic(ip, &nic.gid_index);
    if (!nic.ctx) {
        return false;
    }

    nic.num_lag_ports = numLagPorts(ibv_get_device_name(nic.ctx->device));
    NIXL_INFO << "ibverbs_dc: NIC " << ip << " LAG ports: " << nic.num_lag_ports;

    // Record PCIe topology for GPU/NIC affinity (used only as an advisory hint;
    // failures degrade to no-affinity, never block).
    nic.dev_name = ibv_get_device_name(nic.ctx->device);
    const std::string ib_dev = "/sys/class/infiniband/" + nic.dev_name + "/device";
    nic.pci_path = canonicalPath(ib_dev);
    nic.numa_node = readIntFile(ib_dev + "/numa_node");
    NIXL_INFO << "ibverbs_dc: NIC " << ip << " (" << nic.dev_name << ") pci_path='" << nic.pci_path
              << "' numa_node=" << nic.numa_node;

    // Device ceilings that bound RDMA READ concurrency: max_qp_rd_atom caps the
    // DCT responder resources (max_dest_rd_atomic) this target can grant to a
    // remote reader (biziod). If the DCT ends up below this ceiling, biziod's
    // read pipeline is throttled regardless of its own max_rd_atomic.
    ibv_device_attr dev_attr{};
    if (ibv_query_device(nic.ctx, &dev_attr) == 0) {
        NIXL_INFO << "ibverbs_dc: NIC " << ip << " device caps: max_qp_rd_atom="
                  << dev_attr.max_qp_rd_atom
                  << " max_qp_init_rd_atom=" << dev_attr.max_qp_init_rd_atom;
    } else {
        NIXL_WARN << "ibverbs_dc: ibv_query_device failed for " << ip;
    }

    nic.pd = ibv_alloc_pd(nic.ctx);
    if (!nic.pd) {
        NIXL_ERROR << "ibverbs_dc: ibv_alloc_pd failed for " << ip;
        return false;
    }

    nic.cq = ibv_create_cq(nic.ctx, 2 * nic.num_lag_ports, nullptr, nullptr, 0);
    if (!nic.cq) {
        NIXL_ERROR << "ibverbs_dc: ibv_create_cq failed for " << ip;
        return false;
    }

    ibv_srq_init_attr srq_attr{};
    srq_attr.attr.max_wr = 1;
    srq_attr.attr.max_sge = 1;
    nic.srq = ibv_create_srq(nic.pd, &srq_attr);
    if (!nic.srq) {
        NIXL_ERROR << "ibverbs_dc: ibv_create_srq failed for " << ip;
        return false;
    }

    for (int p = 0; p < nic.num_lag_ports; p++) {
        ibv_qp_init_attr_ex qp_attr{};
        qp_attr.qp_type = IBV_QPT_DRIVER;
        qp_attr.send_cq = nic.cq;
        qp_attr.recv_cq = nic.cq;
        qp_attr.srq = nic.srq;
        qp_attr.pd = nic.pd;
        qp_attr.comp_mask = IBV_QP_INIT_ATTR_PD;
        qp_attr.cap.max_send_wr = 1;
        qp_attr.cap.max_recv_wr = 1;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;

        mlx5dv_qp_init_attr mlx5_attr{};
        mlx5_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_DC;
        mlx5_attr.dc_init_attr.dc_type = MLX5DV_DCTYPE_DCT;
        mlx5_attr.dc_init_attr.dct_access_key = dc_key;

        nic.dct_qps[p] = mlx5dv_create_qp(nic.ctx, &qp_attr, &mlx5_attr);
        if (!nic.dct_qps[p]) {
            NIXL_ERROR << "ibverbs_dc: mlx5dv_create_qp (DCT) failed for " << ip << " lag_port "
                       << (p + 1);
            return false;
        }

        ibv_qp_attr mod{};
        mod.qp_state = IBV_QPS_INIT;
        mod.port_num = kDcPort;
        mod.pkey_index = 0;
        mod.qp_access_flags = IBV_ACCESS_REMOTE_WRITE | IBV_ACCESS_REMOTE_READ;
        if (ibv_modify_qp(nic.dct_qps[p],
                          &mod,
                          IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
            NIXL_ERROR << "ibverbs_dc: DCT QP -> INIT failed for " << ip;
            return false;
        }

        memset(&mod, 0, sizeof(mod));
        mod.qp_state = IBV_QPS_RTR;
        mod.path_mtu = IBV_MTU_4096;
        mod.ah_attr.sl = sl_;
        mod.ah_attr.port_num = kDcPort;
        mod.ah_attr.is_global = 1;
        mod.ah_attr.grh.sgid_index = nic.gid_index;
        mod.ah_attr.grh.hop_limit = 64;
        mod.ah_attr.grh.traffic_class = traffic_class_;
        mod.min_rnr_timer = 12;
        if (ibv_modify_qp(nic.dct_qps[p],
                          &mod,
                          IBV_QP_STATE | IBV_QP_PATH_MTU | IBV_QP_AV | IBV_QP_MIN_RNR_TIMER)) {
            NIXL_ERROR << "ibverbs_dc: DCT QP -> RTR failed for " << ip;
            return false;
        }

        nic.dctns[p] = nic.dct_qps[p]->qp_num;
        NIXL_INFO << "ibverbs_dc: NIC " << ip << " DCT QP[" << p << "] dctn=" << nic.dctns[p];

        // Report the negotiated responder capacity: how many concurrent incoming
        // RDMA READs this DCT will service. The RTR modify above does not set
        // IBV_QP_MAX_DEST_RD_ATOMIC, so this is the driver default and is the
        // suspected ceiling on biziod's read pipeline throughput.
        ibv_qp_attr qa{};
        ibv_qp_init_attr qia{};
        if (ibv_query_qp(nic.dct_qps[p], &qa, IBV_QP_MAX_DEST_RD_ATOMIC, &qia) == 0) {
            NIXL_INFO << "ibverbs_dc: NIC " << ip << " DCT QP[" << p
                      << "] max_dest_rd_atomic=" << (int)qa.max_dest_rd_atomic;
        } else {
            NIXL_WARN << "ibverbs_dc: ibv_query_qp(DCT) failed for " << ip;
        }
    }

    ibv_port_attr port_attr{};
    if (ibv_query_port(nic.ctx, kDcPort, &port_attr)) {
        NIXL_ERROR << "ibverbs_dc: ibv_query_port failed for " << ip;
        return false;
    }
    nic.lid = port_attr.lid; // 0 on RoCEv2

    ibv_gid gid;
    if (ibv_query_gid(nic.ctx, kDcPort, nic.gid_index, &gid)) {
        NIXL_ERROR << "ibverbs_dc: ibv_query_gid failed for " << ip;
        return false;
    }
    memcpy(nic.gid, gid.raw, 16);
    return true;
}

IbverbsDcRdmaTokenClient::IbverbsDcRdmaTokenClient(const std::vector<std::string> &nic_ips,
                                                   uint64_t dc_key,
                                                   size_t split_size)
    : split_size_(split_size) {
    if (nic_ips.empty()) {
        NIXL_ERROR << "ibverbs_dc: no NIC addresses provided (rdma_nics is empty)";
        return;
    }
    // Reuse UCX's well-known knobs so a single setting steers RoCE priority for
    // both the UCX backend and this (raw-ibverbs) connector on a NIXL host.
    // Reuse UCX's well-known knobs so a single setting steers RoCE priority for
    // both the UCX backend and this (raw-ibverbs) connector on a NIXL host.
    // Non-numeric values (e.g. UCX_IB_SL=auto) are left to the default.
    if (const char *sl = std::getenv("UCX_IB_SL")) {
        char *end = nullptr;
        long parsed = std::strtol(sl, &end, 0);
        if (end != sl && *end == '\0' && parsed >= 0 && parsed <= 15) {
            sl_ = static_cast<uint8_t>(parsed);
        } else {
            NIXL_WARN << "ibverbs_dc: ignoring non-numeric/out-of-range UCX_IB_SL: " << sl;
        }
    }
    if (const char *tos = std::getenv("UCX_IB_TRAFFIC_CLASS")) {
        char *end = nullptr;
        long parsed = std::strtol(tos, &end, 0);
        if (end != tos && *end == '\0' && parsed >= 0 && parsed <= 255) {
            traffic_class_ = static_cast<uint8_t>(parsed);
        } else {
            NIXL_WARN << "ibverbs_dc: ignoring non-numeric/out-of-range UCX_IB_TRAFFIC_CLASS: "
                      << tos;
        }
    }
    NIXL_INFO << "ibverbs_dc: RoCE sl=" << (int)sl_ << " traffic_class=" << (int)traffic_class_
              << " (DSCP " << (traffic_class_ >> 2) << ")";
    nics_.resize(nic_ips.size());
    for (size_t i = 0; i < nic_ips.size(); i++) {
        if (!setupNic(nic_ips[i], dc_key, nics_[i])) {
            NIXL_ERROR << "ibverbs_dc: failed to set up NIC " << nic_ips[i];
            return;
        }
    }
    nic_load_.assign(nics_.size(), 0);
    all_nics_.resize(nics_.size());
    for (size_t i = 0; i < nics_.size(); i++) {
        all_nics_[i] = static_cast<int>(i);
    }
    connected_ = true;
    NIXL_INFO << "ibverbs_dc: DC transport ready across " << nics_.size() << " NIC(s), key=0x"
              << std::hex << dc_key << std::dec;
}

IbverbsDcRdmaTokenClient::~IbverbsDcRdmaTokenClient() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        for (auto &kv : regions_) {
            if (kv.second.mr) {
                ibv_dereg_mr(kv.second.mr);
            }
        }
        regions_.clear();
    }
    for (auto &nic : nics_) {
        for (int p = 0; p < kDcMaxLagPorts; p++) {
            if (nic.dct_qps[p]) {
                ibv_destroy_qp(nic.dct_qps[p]);
            }
        }
        if (nic.srq) {
            ibv_destroy_srq(nic.srq);
        }
        if (nic.cq) {
            ibv_destroy_cq(nic.cq);
        }
        if (nic.pd) {
            ibv_dealloc_pd(nic.pd);
        }
        if (nic.ctx) {
            ibv_close_device(nic.ctx);
        }
    }
    nics_.clear();
}

bool
IbverbsDcRdmaTokenClient::isConnected() const {
    return connected_;
}

const std::vector<int> &
IbverbsDcRdmaTokenClient::affineNicsFor(int dev_id) {
    auto it = gpu_affine_nics_.find(dev_id);
    if (it != gpu_affine_nics_.end()) {
        return it->second;
    }

    const std::string gpu_path = gpuPciPath(dev_id);
    const int gpu_numa = gpu_path.empty() ? -1 : readIntFile(gpu_path + "/numa_node");

    // Candidate rails: same-NUMA NICs (avoids the cross-socket SYS path). If NUMA
    // is unknown, every NIC is a candidate.
    std::vector<int> cand;
    if (gpu_numa >= 0) {
        for (size_t i = 0; i < nics_.size(); ++i) {
            if (nics_[i].numa_node == gpu_numa) {
                cand.push_back(static_cast<int>(i));
            }
        }
    }
    if (cand.empty()) {
        cand = all_nics_;
    }

    // Narrow to PCIe switch-local NICs, but ONLY when the locality is real. A NIC
    // is switch-local to the GPU when the GPU's PCIe path reaches into the NIC's
    // switch, i.e. the GPU shares MORE of the path with that NIC than the NIC's
    // same-switch siblings share with each other. On hardware where the RDMA NICs
    // sit on a shared bridge equidistant (NODE) from every GPU in the node, no NIC
    // is switch-local, so all same-NUMA rails stay in the set and registrations
    // spread across both. On hardware where each GPU has a dedicated (PXB/PIX)
    // NIC, that one rail wins and pinning is preserved.
    std::vector<int> affine;
    if (!gpu_path.empty() && cand.size() > 1) {
        int best = cand.front();
        size_t best_prefix = commonPathPrefix(gpu_path, nics_[best].pci_path);
        for (int idx : cand) {
            size_t p = commonPathPrefix(gpu_path, nics_[idx].pci_path);
            if (p > best_prefix) {
                best_prefix = p;
                best = idx;
            }
        }
        // How deep the best NIC's switch neighborhood extends among the candidates.
        size_t sibling_prefix = 0;
        for (int idx : cand) {
            if (idx == best) {
                continue;
            }
            sibling_prefix = std::max(
                sibling_prefix, commonPathPrefix(nics_[best].pci_path, nics_[idx].pci_path));
        }
        // GPU reaches past where the siblings diverge -> genuinely switch-local;
        // keep every candidate at that deepest tier and drop the rest.
        if (best_prefix > sibling_prefix) {
            for (int idx : cand) {
                if (commonPathPrefix(gpu_path, nics_[idx].pci_path) == best_prefix) {
                    affine.push_back(idx);
                }
            }
        }
    }

    // Not switch-local (equidistant NODE-level rails) or NUMA-only: use them all.
    if (affine.empty()) {
        affine = cand;
    }

    if (gpu_path.empty()) {
        NIXL_WARN << "ibverbs_dc: no PCIe path for GPU " << dev_id << "; using all NICs";
    } else {
        std::string names;
        for (int i : affine) {
            names += (names.empty() ? "" : ",") + nics_[i].dev_name;
        }
        NIXL_INFO << "ibverbs_dc: GPU " << visibleToUserGpu(dev_id) << " (numa " << gpu_numa
                  << ") -> affine NIC(s): " << names;
    }

    auto res = gpu_affine_nics_.emplace(dev_id, std::move(affine));
    return res.first->second;
}

int
IbverbsDcRdmaTokenClient::leastLoadedNic(const std::vector<int> &candidates) {
    int best = candidates.front();
    for (int idx : candidates) {
        if (nic_load_[idx] < nic_load_[best]) {
            best = idx;
        }
    }
    return best;
}

int
IbverbsDcRdmaTokenClient::selectNicFor(int dev_id) {
    // Host memory (dev_id < 0): balance across all NICs.
    if (dev_id < 0) {
        return leastLoadedNic(all_nics_);
    }
    // VRAM: prefer the GPU's affine ("best") rails, but never restrict to them.
    // Rank every NIC by live load, handing the non-affine rails a fixed handicap
    // (kNonAffineHandicap) so affine rails win while loads are close yet the rest
    // still absorb the overflow once the affine rails run ahead. A GPU whose
    // affine set is a 2-NIC NUMA node therefore still reaches the other NUMA's
    // NICs under load instead of leaving them idle. Least-loaded (not a per-GPU
    // cursor) so multiple GPUs spread even when each registers one buffer.
    const std::vector<int> &affine = affineNicsFor(dev_id);
    auto cost = [&](int idx) -> uint64_t {
        const bool is_affine =
            std::find(affine.begin(), affine.end(), idx) != affine.end();
        return nic_load_[idx] + (is_affine ? 0 : kNonAffineHandicap);
    };
    int best = all_nics_.front();
    uint64_t best_cost = cost(best);
    for (int idx : all_nics_) {
        const uint64_t c = cost(idx);
        if (c < best_cost) {
            best_cost = c;
            best = idx;
        }
    }
    return best;
}

std::vector<int>
IbverbsDcRdmaTokenClient::fanRailsFor(int dev_id) {
    // mu_ held by caller. Affine rails first, then the rest, so a buffer with fewer
    // chunks than rails favours the GPU's own while a bigger one still reaches every
    // rail. Host memory (dev_id < 0) is affine to all NICs already.
    const std::vector<int> &affine = affineNicsFor(dev_id);
    std::vector<int> rails = affine;
    for (int idx : all_nics_) {
        if (std::find(affine.begin(), affine.end(), idx) == affine.end()) {
            rails.push_back(idx);
        }
    }
    return rails;
}

bool
IbverbsDcRdmaTokenClient::registerPiece(
    void *ptr, size_t size, int dev_id, int nic_idx, uintptr_t parent_base) {
    // mu_ held by caller.
    NicCtx &nic = nics_[nic_idx];

    /* Relaxed ordering lets the NIC pipeline the PCIe writes into GPU BAR instead
     * of serializing them.  Without it, GET writes into VRAM collapse under
     * multi-GPU concentration on a single dual-port card: the card cannot drain
     * the strict-ordered writes fast enough, asserts PFC pause, and throughput
     * falls below even a single GPU.  ib_write_bw (perftest) sets this by
     * default, which is why it scales on the same hardware.  It is an optional
     * access flag: drivers that don't support it silently ignore it. */
    unsigned int access = IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                          IBV_ACCESS_REMOTE_READ | IBV_ACCESS_RELAXED_ORDERING;
    ibv_mr *mr = ibv_reg_mr(nic.pd, ptr, size, access);
    if (!mr) {
        NIXL_ERROR << "ibverbs_dc: ibv_reg_mr failed (ptr=" << ptr << ", size=" << size << ")";
        return false;
    }

    Region r;
    r.mr = mr;
    r.len = size;
    r.nic_idx = nic_idx;
    r.dev_id = dev_id;
    r.dctn = nic.dctns[nic.lag_seq++ % nic.num_lag_ports];
    r.parent_base = parent_base;
    regions_[reinterpret_cast<uintptr_t>(ptr)] = r;
    // Load accounting lives here, not in the selectors, so it stays correct
    // whether the NIC was chosen by selectNicFor or handed in by a fan-out.
    nic_load_[nic_idx]++;
    return true;
}

cuObjErr_t
IbverbsDcRdmaTokenClient::cuMemObjGetDescriptor(void *ptr, size_t size, int dev_id) {
    if (!connected_) {
        return CU_OBJ_FAIL;
    }
    if (size > UINT32_MAX) {
        NIXL_ERROR << "ibverbs_dc: registration size " << size
                   << " exceeds the 32-bit DC descriptor SIZE field";
        return CU_OBJ_FAIL;
    }

    std::lock_guard<std::mutex> lk(mu_);
    const uintptr_t base = reinterpret_cast<uintptr_t>(ptr);

    // Spread the buffer over the rails one request-sized chunk at a time, cycling
    // rails as it goes.
    //
    // Splitting it into a few contiguous pieces instead (half the buffer per rail)
    // does not work: callers transfer from the buffer base, so a transfer shorter
    // than the buffer lands entirely in the first piece and the rails holding the
    // rest stay idle. Measured that way, two of four rails sat at exactly 0 Gb/s
    // for every block size below the piece length. Interleaving at the request
    // granularity means a transfer spanning N chunks touches min(N, rails) rails,
    // so a single buffer can fill the node instead of being capped at one rail's
    // line rate. Costs one MR per chunk.
    const std::vector<int> rails = fanRailsFor(dev_id);
    const size_t chunks = split_size_ > 0 ? (size + split_size_ - 1) / split_size_ : 1;
    if (chunks < 2 || rails.size() < 2) {
        // Nothing to interleave. dev_id >= 0 (VRAM) selects a PCIe-affine NIC for
        // the GPU; dev_id < 0 (host memory) keeps the global round-robin.
        if (!registerPiece(ptr, size, dev_id, selectNicFor(dev_id), base)) {
            return CU_OBJ_FAIL;
        }
        return CU_OBJ_SUCCESS;
    }

    // Advance the starting rail per registration so buffers with fewer chunks than
    // there are rails still spread across them rather than all starting on rail 0.
    const size_t rotor = fan_rotor_++;
    for (size_t i = 0; i < chunks; ++i) {
        const size_t off = i * split_size_;
        const size_t len = std::min(split_size_, size - off);
        const int nic = rails[(rotor + i) % rails.size()];
        if (!registerPiece(reinterpret_cast<void *>(base + off), len, dev_id, nic, base)) {
            // Unwind the chunks already registered so the caller sees all-or-nothing.
            releaseRegistration(base);
            return CU_OBJ_FAIL;
        }
    }

    NIXL_DEBUG << "ibverbs_dc: interleaved registration 0x" << std::hex << base << std::dec << " ("
               << size << " bytes) over " << rails.size() << " rail(s) in " << chunks
               << " chunk(s) of " << split_size_ << ", dev_id=" << dev_id;
    return CU_OBJ_SUCCESS;
}

size_t
IbverbsDcRdmaTokenClient::releaseRegistration(uintptr_t parent_base) {
    // mu_ held by caller. Pieces of one registration are contiguous from
    // parent_base, so walk forward while they still belong to it.
    size_t released = 0;
    auto it = regions_.lower_bound(parent_base);
    while (it != regions_.end() && it->second.parent_base == parent_base) {
        if (it->second.mr) {
            ibv_dereg_mr(it->second.mr);
        }
        if (nic_load_[it->second.nic_idx] > 0) {
            nic_load_[it->second.nic_idx]--;
        }
        it = regions_.erase(it);
        released++;
    }
    return released;
}

cuObjErr_t
IbverbsDcRdmaTokenClient::cuMemObjPutDescriptor(void *ptr) {
    std::lock_guard<std::mutex> lk(mu_);
    if (releaseRegistration(reinterpret_cast<uintptr_t>(ptr)) == 0) {
        NIXL_ERROR << "ibverbs_dc: cuMemObjPutDescriptor: ptr " << ptr << " not registered";
        return CU_OBJ_FAIL;
    }
    return CU_OBJ_SUCCESS;
}

void
IbverbsDcRdmaTokenClient::logAssignment() {
    // mu_ held by caller. Summarize registered regions as GPU -> per-NIC counts.
    std::map<int, std::vector<size_t>> per_gpu; // dev_id -> count per NIC index
    std::vector<size_t> per_nic(nics_.size(), 0);
    for (const auto &kv : regions_) {
        const Region &r = kv.second;
        per_nic[r.nic_idx]++;
        auto &v = per_gpu[r.dev_id];
        if (v.empty()) {
            v.assign(nics_.size(), 0);
        }
        v[r.nic_idx]++;
    }

    NIXL_INFO << "ibverbs_dc: GPU->NIC buffer assignment (" << regions_.size()
              << " registered regions):";
    for (const auto &g : per_gpu) {
        std::string line;
        for (size_t i = 0; i < nics_.size(); ++i) {
            if (g.second[i] == 0) {
                continue;
            }
            line += (line.empty() ? "" : " ") + nics_[i].dev_name + ":" +
                    std::to_string(g.second[i]);
        }
        if (g.first < 0) {
            NIXL_INFO << "ibverbs_dc:   host memory -> " << line;
        } else {
            NIXL_INFO << "ibverbs_dc:   GPU " << visibleToUserGpu(g.first) << " -> " << line;
        }
    }
    std::string totals;
    for (size_t i = 0; i < nics_.size(); ++i) {
        totals += (totals.empty() ? "" : " ") + nics_[i].dev_name + ":" +
                  std::to_string(per_nic[i]);
    }
    NIXL_INFO << "ibverbs_dc:   per-NIC totals: " << totals;
}

std::string
IbverbsDcRdmaTokenClient::descriptorFor(void *ptr, size_t size) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard<std::mutex> lk(mu_);
    // Registrations are complete by the first transfer; dump the layout once.
    if (!assignment_logged_) {
        logAssignment();
        assignment_logged_ = true;
    }
    // Find the region [base, base+len) that contains addr.
    auto it = regions_.upper_bound(addr);
    if (it == regions_.begin()) {
        return std::string();
    }
    --it;
    const Region &r = it->second;
    if (addr < it->first || addr + size > it->first + r.len) {
        return std::string();
    }
    const NicCtx &nic = nics_[r.nic_idx];
    return formatDcDescriptor(static_cast<uint64_t>(addr),
                              static_cast<uint32_t>(size),
                              r.mr->rkey,
                              nic.lid,
                              r.dctn,
                              nic.gid);
}

ssize_t
IbverbsDcRdmaTokenClient::cuObjGet(void *ctx, void *ptr, size_t size, loff_t, loff_t) {
    std::string desc = descriptorFor(ptr, size);
    if (desc.empty()) {
        NIXL_ERROR << "ibverbs_dc: cuObjGet: no registration covers ptr " << ptr << " size "
                   << size;
        return -1;
    }
    static_cast<rdma_ctx_t *>(ctx)->rdma_desc = std::move(desc);
    return static_cast<ssize_t>(size);
}

ssize_t
IbverbsDcRdmaTokenClient::cuObjPut(void *ctx, void *ptr, size_t size, loff_t, loff_t) {
    std::string desc = descriptorFor(ptr, size);
    if (desc.empty()) {
        NIXL_ERROR << "ibverbs_dc: cuObjPut: no registration covers ptr " << ptr << " size "
                   << size;
        return -1;
    }
    static_cast<rdma_ctx_t *>(ctx)->rdma_desc = std::move(desc);
    return static_cast<ssize_t>(size);
}
