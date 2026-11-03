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

#include "dc_request_keys.h"

#include <algorithm>
#include <cerrno>
#include <cstring>
#include <random>

#include "common/nixl_log.h"

namespace {

/// Bytes one entry of a key covers. rdma-core refuses longer entries in a window
/// bind; an mkey entry is kept to the same limit.
constexpr uint64_t key_entry_max = 1ull << 30;
/// Entries of a key: enough for the 4 GiB - 1 bytes a DC descriptor can describe.
constexpr uint16_t key_entries = 4;
/// Keys one NIC may hold. One request uses one key while it runs; the rest are
/// ready or resting.
constexpr size_t keys_max = 1u << 20;
/// Longest wait for a UMR or an invalidation, which the NIC completes in about
/// a microsecond.
constexpr std::chrono::seconds completion_timeout{1};

} // namespace

dcRequestKeys::dcRequestKeys(ibv_context *ctx,
                             ibv_pd *pd,
                             uint8_t port,
                             int gid_index,
                             ibv_mtu mtu,
                             size_t initial_keys)
    : ctx_(ctx),
      pd_(pd),
      port_(port),
      gidIndex_(gid_index),
      mtu_(mtu) {
    cq_ = ibv_create_cq(ctx_, 4, nullptr, nullptr, 0);
    if (!cq_) {
        error_ = std::string("ibv_create_cq: ") + strerror(errno);
        return;
    }
    if (!createQp()) {
        return;
    }
    // The first one also checks that the driver can change a key's tag, which
    // every open relies on.
    std::lock_guard<std::mutex> lk(mu_);
    ok_ = createKey();
    for (size_t i = 1; ok_ && i < initial_keys; ++i) {
        if (!createKey()) {
            // Requests create what is missing.
            NIXL_WARN << "ibverbs_dc: created " << keys_.size() << " of " << initial_keys
                      << " request keys at setup";
            break;
        }
    }
}

dcRequestKeys::~dcRequestKeys() {
    if (opens_ > 0) {
        NIXL_INFO << "ibverbs_dc: " << ibv_get_device_name(ctx_->device) << ": " << opens_
                  << " request key open(s) on " << keys_.size() << " key(s), " << resting_.size()
                  << " resting at teardown";
    }
    for (key &k : keys_) {
        if (k.mkey) {
            mlx5dv_destroy_mkey(k.mkey);
        }
    }
    destroyQp();
    if (cq_) {
        ibv_destroy_cq(cq_);
    }
}

bool
dcRequestKeys::createQp() {
    ibv_qp_init_attr_ex attr{};
    attr.qp_type = IBV_QPT_RC;
    attr.send_cq = cq_;
    attr.recv_cq = cq_;
    attr.pd = pd_;
    attr.comp_mask = IBV_QP_INIT_ATTR_PD | IBV_QP_INIT_ATTR_SEND_OPS_FLAGS;
    attr.send_ops_flags = IBV_QP_EX_WITH_LOCAL_INV;
    attr.cap.max_send_wr = 4;
    attr.cap.max_recv_wr = 1;
    attr.cap.max_send_sge = 1;
    attr.cap.max_recv_sge = 1;
    // The entries of a configured key travel inline in the work request.
    attr.cap.max_inline_data = 128;

    mlx5dv_qp_init_attr dv_attr{};
    dv_attr.comp_mask = MLX5DV_QP_INIT_ATTR_MASK_SEND_OPS_FLAGS;
    dv_attr.send_ops_flags = MLX5DV_QP_EX_WITH_MKEY_CONFIGURE;

    qp_ = mlx5dv_create_qp(ctx_, &attr, &dv_attr);
    if (!qp_) {
        error_ = std::string("mlx5dv_create_qp (RC): ") + strerror(errno);
        return false;
    }

    ibv_gid gid;
    if (ibv_query_gid(ctx_, port_, gidIndex_, &gid)) {
        error_ = "ibv_query_gid failed";
        return false;
    }

    ibv_qp_attr mod{};
    mod.qp_state = IBV_QPS_INIT;
    mod.port_num = port_;
    if (ibv_modify_qp(
            qp_, &mod, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS)) {
        error_ = std::string("RC QP -> INIT: ") + strerror(errno);
        return false;
    }

    // Connected to itself: no peer to agree with, and it never sends anything.
    memset(&mod, 0, sizeof(mod));
    mod.qp_state = IBV_QPS_RTR;
    mod.path_mtu = mtu_;
    mod.dest_qp_num = qp_->qp_num;
    mod.max_dest_rd_atomic = 1;
    mod.min_rnr_timer = 12;
    mod.ah_attr.port_num = port_;
    mod.ah_attr.is_global = 1;
    mod.ah_attr.grh.dgid = gid;
    mod.ah_attr.grh.sgid_index = gidIndex_;
    mod.ah_attr.grh.hop_limit = 64;
    if (ibv_modify_qp(qp_,
                      &mod,
                      IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                          IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER)) {
        error_ = std::string("RC QP -> RTR: ") + strerror(errno);
        return false;
    }

    memset(&mod, 0, sizeof(mod));
    mod.qp_state = IBV_QPS_RTS;
    mod.timeout = 14;
    mod.retry_cnt = 7;
    mod.rnr_retry = 7;
    mod.max_rd_atomic = 1;
    if (ibv_modify_qp(qp_,
                      &mod,
                      IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY |
                          IBV_QP_SQ_PSN | IBV_QP_MAX_QP_RD_ATOMIC)) {
        error_ = std::string("RC QP -> RTS: ") + strerror(errno);
        return false;
    }
    return true;
}

void
dcRequestKeys::destroyQp() {
    if (qp_) {
        ibv_destroy_qp(qp_);
        qp_ = nullptr;
    }
}

void
dcRequestKeys::resetQp() {
    destroyQp();
    ibv_wc wc;
    while (ibv_poll_cq(cq_, 1, &wc) > 0) {}
    if (!createQp()) {
        ok_ = false;
        NIXL_ERROR << "ibverbs_dc: cannot recreate the queue pair of the request keys (" << error_
                   << "); no RDMA request can run on this NIC";
    }
}

bool
dcRequestKeys::waitCompletion(const char *what) {
    const time_point_t give_up = std::chrono::steady_clock::now() + completion_timeout;
    ibv_wc wc;
    int n;
    while ((n = ibv_poll_cq(cq_, 1, &wc)) == 0) {
        if (std::chrono::steady_clock::now() > give_up) {
            NIXL_ERROR << "ibverbs_dc: " << what << " of a request key did not complete";
            return false;
        }
    }
    if (n < 0) {
        NIXL_ERROR << "ibverbs_dc: polling the completion of a request key failed";
        return false;
    }
    if (wc.status != IBV_WC_SUCCESS) {
        NIXL_ERROR << "ibverbs_dc: " << what
                   << " of a request key failed: " << ibv_wc_status_str(wc.status)
                   << " (vendor error 0x" << std::hex << wc.vendor_err << std::dec << ")";
        return false;
    }
    return true;
}

bool
dcRequestKeys::createKey() {
    if (keys_.size() >= keys_max) {
        NIXL_ERROR << "ibverbs_dc: " << keys_max << " request keys in use on one NIC";
        return false;
    }
    mlx5dv_mkey_init_attr attr{};
    attr.pd = pd_;
    attr.create_flags =
        MLX5DV_MKEY_INIT_ATTR_FLAGS_INDIRECT | MLX5DV_MKEY_INIT_ATTR_FLAGS_UPDATE_TAG;
    attr.max_entries = key_entries;
    mlx5dv_mkey *mkey = mlx5dv_create_mkey(&attr);
    if (!mkey) {
        error_ = std::string("mlx5dv_create_mkey: ") + strerror(errno) +
            (errno == EOPNOTSUPP ? " (the mlx5 driver cannot change a key's tag)" : "");
        NIXL_ERROR << "ibverbs_dc: cannot create a request key: " << error_;
        return false;
    }

    key k;
    k.mkey = mkey;
    k.index = mkey->lkey >> 8;
    // A random first tag: an rkey a server still holds from an earlier process
    // is unlikely to match.
    static thread_local std::mt19937 rng{std::random_device{}()};
    k.tag = static_cast<uint8_t>(rng());

    const uint32_t slot = static_cast<uint32_t>(keys_.size());
    auto retired = retired_.find(k.index);
    if (retired != retired_.end() && retired->second > std::chrono::steady_clock::now()) {
        k.restUntil = retired->second;
        keys_.push_back(k);
        resting_.push_back(slot);
    } else {
        keys_.push_back(k);
        ready_.push_back(slot);
    }
    if (retired != retired_.end()) {
        retired_.erase(retired);
    }
    return true;
}

bool
dcRequestKeys::destroyKey(uint32_t slot) {
    key &k = keys_[slot];
    if (mlx5dv_destroy_mkey(k.mkey) != 0) {
        NIXL_ERROR << "ibverbs_dc: cannot destroy request key 0x" << std::hex << k.index << std::dec
                   << "; its last rkey may stay usable";
        return false;
    }
    k.mkey = nullptr;
    retired_[k.index] = std::chrono::steady_clock::now() + keyReuseGuard;
    return true;
}

bool
dcRequestKeys::configure(key &k, const ibv_mr *mr, uint64_t addr, uint64_t len) {
    ibv_sge entries[key_entries];
    uint16_t n = 0;
    for (uint64_t off = 0; off < len; off += key_entry_max) {
        entries[n].addr = addr + off;
        entries[n].length = static_cast<uint32_t>(std::min(key_entry_max, len - off));
        entries[n].lkey = mr->lkey;
        n++;
    }

    k.mkey->lkey = (k.index << 8) | k.tag;
    k.mkey->rkey = k.mkey->lkey;

    ibv_qp_ex *qpx = ibv_qp_to_qp_ex(qp_);
    mlx5dv_qp_ex *dvqp = mlx5dv_qp_ex_from_ibv_qp_ex(qpx);
    mlx5dv_mkey_conf_attr conf{};
    ibv_wr_start(qpx);
    qpx->wr_flags = IBV_SEND_SIGNALED | IBV_SEND_INLINE;
    mlx5dv_wr_mkey_configure(dvqp, k.mkey, 2 /* setters */, &conf);
    mlx5dv_wr_set_mkey_access_flags(
        dvqp, IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ | IBV_ACCESS_REMOTE_WRITE);
    mlx5dv_wr_set_mkey_layout_list(dvqp, n, entries);
    const int err = ibv_wr_complete(qpx);
    if (err) {
        NIXL_ERROR << "ibverbs_dc: posting the UMR of a request key failed: " << strerror(err);
        return false;
    }
    return waitCompletion("the UMR");
}

bool
dcRequestKeys::invalidate(key &k) {
    ibv_qp_ex *qpx = ibv_qp_to_qp_ex(qp_);
    ibv_wr_start(qpx);
    qpx->wr_flags = IBV_SEND_SIGNALED;
    ibv_wr_local_inv(qpx, k.mkey->rkey);
    const int err = ibv_wr_complete(qpx);
    if (err) {
        NIXL_ERROR << "ibverbs_dc: posting the invalidation of a request key failed: "
                   << strerror(err);
        return false;
    }
    return waitCompletion("the invalidation");
}

bool
dcRequestKeys::open(const ibv_mr *mr, uint64_t addr, uint64_t len, uint32_t &slot, uint32_t &rkey) {
    if (len == 0 || len > key_entries * key_entry_max) {
        NIXL_ERROR << "ibverbs_dc: a request key cannot cover " << len << " bytes";
        return false;
    }
    std::lock_guard<std::mutex> lk(mu_);
    if (!ok_) {
        return false;
    }
    const time_point_t now = std::chrono::steady_clock::now();
    while (!resting_.empty() && keys_[resting_.front()].restUntil <= now) {
        ready_.push_back(resting_.front());
        resting_.pop_front();
    }
    // A new key may come back resting when the firmware reuses a destroyed one's
    // index: create until one is ready.
    while (ready_.empty()) {
        if (!createKey()) {
            return false;
        }
    }
    slot = ready_.front();
    ready_.pop_front();
    key &k = keys_[slot];

    if (!configure(k, mr, addr, len)) {
        // The key may be configured or not: destroy it rather than guess.
        resetQp();
        destroyKey(slot);
        return false;
    }
    rkey = k.mkey->rkey;
    opens_++;
    return true;
}

bool
dcRequestKeys::close(uint32_t slot) {
    std::lock_guard<std::mutex> lk(mu_);
    key &k = keys_[slot];
    if (!invalidate(k)) {
        // Destroying the key also makes its rkey unusable.
        resetQp();
        return destroyKey(slot);
    }
    k.tag++;
    if (++k.opens == 256) {
        k.opens = 0;
        k.restUntil = std::chrono::steady_clock::now() + keyReuseGuard;
        resting_.push_back(slot);
    } else {
        ready_.push_back(slot);
    }
    return true;
}
