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
#include <unistd.h>

#include <cstring>

#include "common/nixl_log.h"
#include "rdma_ctx.h"

namespace {

constexpr int kDcPort = 1;

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
        mod.ah_attr.port_num = kDcPort;
        mod.ah_attr.is_global = 1;
        mod.ah_attr.grh.sgid_index = nic.gid_index;
        mod.ah_attr.grh.hop_limit = 64;
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
                                                   uint64_t dc_key) {
    if (nic_ips.empty()) {
        NIXL_ERROR << "ibverbs_dc: no NIC addresses provided (rdma_nics is empty)";
        return;
    }
    nics_.resize(nic_ips.size());
    for (size_t i = 0; i < nic_ips.size(); i++) {
        if (!setupNic(nic_ips[i], dc_key, nics_[i])) {
            NIXL_ERROR << "ibverbs_dc: failed to set up NIC " << nic_ips[i];
            return;
        }
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

cuObjErr_t
IbverbsDcRdmaTokenClient::cuMemObjGetDescriptor(void *ptr, size_t size) {
    if (!connected_) {
        return CU_OBJ_FAIL;
    }
    if (size > UINT32_MAX) {
        NIXL_ERROR << "ibverbs_dc: registration size " << size
                   << " exceeds the 32-bit DC descriptor SIZE field";
        return CU_OBJ_FAIL;
    }

    std::lock_guard<std::mutex> lk(mu_);
    int nic_idx = static_cast<int>(reg_counter_++ % nics_.size());
    NicCtx &nic = nics_[nic_idx];

    ibv_mr *mr = ibv_reg_mr(nic.pd,
                            ptr,
                            size,
                            IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE |
                                IBV_ACCESS_REMOTE_READ);
    if (!mr) {
        NIXL_ERROR << "ibverbs_dc: ibv_reg_mr failed (ptr=" << ptr << ", size=" << size << ")";
        return CU_OBJ_FAIL;
    }

    Region r;
    r.mr = mr;
    r.len = size;
    r.nic_idx = nic_idx;
    r.dctn = nic.dctns[nic.lag_seq++ % nic.num_lag_ports];
    regions_[reinterpret_cast<uintptr_t>(ptr)] = r;
    return CU_OBJ_SUCCESS;
}

cuObjErr_t
IbverbsDcRdmaTokenClient::cuMemObjPutDescriptor(void *ptr) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = regions_.find(reinterpret_cast<uintptr_t>(ptr));
    if (it == regions_.end()) {
        NIXL_ERROR << "ibverbs_dc: cuMemObjPutDescriptor: ptr " << ptr << " not registered";
        return CU_OBJ_FAIL;
    }
    if (it->second.mr) {
        ibv_dereg_mr(it->second.mr);
    }
    regions_.erase(it);
    return CU_OBJ_SUCCESS;
}

std::string
IbverbsDcRdmaTokenClient::descriptorFor(void *ptr, size_t size) {
    uintptr_t addr = reinterpret_cast<uintptr_t>(ptr);
    std::lock_guard<std::mutex> lk(mu_);
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
