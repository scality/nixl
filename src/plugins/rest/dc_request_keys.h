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

#ifndef NIXL_SRC_PLUGINS_REST_DC_REQUEST_KEYS_H
#define NIXL_SRC_PLUGINS_REST_DC_REQUEST_KEYS_H

#include <chrono>
#include <cstdint>
#include <deque>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#include <infiniband/mlx5dv.h>
#include <infiniband/verbs.h>

/**
 * Remote keys that are valid for one request each, on one NIC.
 *
 * The server reads or writes a buffer through the remote key (rkey) of the
 * RDMA descriptor it receives. The rkey of the buffer's memory region stays
 * valid as long as the buffer is registered: a server that performs its RDMA
 * late, after the request failed or timed out, would still reach the buffer,
 * which the caller may by then hold other data in. So each request gets its own
 * key, and the key is closed before the request reports its outcome. A late
 * RDMA then fails on the server and never touches the buffer.
 *
 * A key is an mlx5 indirect memory key (mkey) that points at the request's
 * range of the memory region. Opening it sets that range with a UMR work
 * request (mlx5dv_wr_mkey_configure); closing it is a local invalidation. Both
 * are local to the NIC and take about a microsecond each.
 *
 * Workarounds, and why memory windows are not used:
 *  - A type 2 memory window only accepts access through the queue pair that
 *    bound it. The server reaches the buffer through the DC target (DCT), which
 *    cannot bind anything: it has no send queue.
 *  - A type 1 window would work, but rdma-core turns every window bind posted on
 *    a DCI into a type 2 bind, and Linux refuses IBV_ACCESS_MW_BIND on a DMA-BUF
 *    registration, which VRAM uses. The refusal is on purpose (commit
 *    09d09e5d79f3): an exporter revokes a DMA-BUF by destroying its region,
 *    which the NIC refuses while windows are bound to it.
 *  - The mkey is created by a firmware command (DEVX), not through the kernel's
 *    memory registration, so it works on a DMA-BUF region. The kernel does not
 *    know the mkey points at the region: every key must be closed before its
 *    region is deregistered.
 *  - rdma-core configures mkeys only from an RC queue pair: on a DCI it allows
 *    raw work requests and memcpy only. So each NIC has an RC queue pair
 *    connected to itself that is used for nothing else. It sends nothing on the
 *    network. Once rdma-core configures mkeys from a DCI, it can go.
 *
 * The address space of a key starts at 0 (rdma-core gives a configured mkey no
 * start address), so the descriptor carries address 0 for the first byte of
 * the request.
 *
 * Each time a key opens, its 8-bit tag changes, and with it the rkey: a server
 * holding an older rkey of the same key is refused. A tag comes back after 256
 * opens, so a key that has used all 256 rests for keyReuseGuard before it
 * opens again.
 */
class dcRequestKeys {
public:
    /// Longest a server may hold an rkey and still be refused once its key closed.
    static constexpr std::chrono::seconds keyReuseGuard{60};

    /**
     * @param ctx       device of the NIC.
     * @param pd        protection domain of the memory regions keys point at.
     * @param port      port the self-connected queue pair uses.
     * @param gid_index its GID; the queue pair connects to itself through it.
     * @param mtu       its path MTU.
     * @param initial_keys keys created now, at least 1: creating one takes a
     *                  firmware command of about 350 us, which a request that
     *                  finds a key ready does not wait for.
     * Check ok() before use.
     */
    dcRequestKeys(ibv_context *ctx,
                  ibv_pd *pd,
                  uint8_t port,
                  int gid_index,
                  ibv_mtu mtu,
                  size_t initial_keys);
    ~dcRequestKeys();

    dcRequestKeys(const dcRequestKeys &) = delete;
    dcRequestKeys &
    operator=(const dcRequestKeys &) = delete;

    /// Whether keys can be opened; why not otherwise, in error().
    bool
    ok() const {
        return ok_;
    }

    const std::string &
    error() const {
        return error_;
    }

    /**
     * Open a key for [addr, addr + len) of mr.
     * @param slot set to what close() takes.
     * @param rkey set to the key's rkey; byte 0 of the range is at address 0.
     * @return false when no key could be opened.
     */
    bool
    open(const ibv_mr *mr, uint64_t addr, uint64_t len, uint32_t &slot, uint32_t &rkey);

    /**
     * Close a key opened by open().
     * @return true once no server can use its rkey; false when it may still
     *         be usable.
     */
    bool
    close(uint32_t slot);

private:
    using time_point_t = std::chrono::steady_clock::time_point;

    struct key {
        mlx5dv_mkey *mkey = nullptr; ///< null once destroyed
        uint32_t index = 0; ///< mkey index: the rkey without its tag
        uint8_t tag = 0; ///< tag of the next open
        unsigned opens = 0; ///< opens since the key last rested
        time_point_t restUntil{};
    };

    bool
    createQp();
    void
    destroyQp();
    /// Replace a queue pair that a failed work request left in the error state.
    void
    resetQp();
    /// Wait for the completion of the one work request in flight.
    bool
    waitCompletion(const char *what);
    /// A new key, ready or resting. Returns false when none can be created.
    bool
    createKey();
    /// Destroy a key whose state is unknown. Its index may come back from the
    /// firmware for a new key, which then rests as if the old one had.
    bool
    destroyKey(uint32_t slot);
    bool
    configure(key &k, const ibv_mr *mr, uint64_t addr, uint64_t len);
    bool
    invalidate(key &k);

    ibv_context *ctx_;
    ibv_pd *pd_;
    uint8_t port_;
    int gidIndex_;
    ibv_mtu mtu_;
    ibv_cq *cq_ = nullptr;
    ibv_qp *qp_ = nullptr;
    bool ok_ = false;
    std::string error_;

    std::mutex mu_;
    std::vector<key> keys_;
    /// Keys ready to open, oldest first.
    std::deque<uint32_t> ready_;
    /// Keys resting after 256 opens, in the order they become ready.
    std::deque<uint32_t> resting_;
    /// Index of a destroyed key -> when a new key with that index may open.
    std::map<uint32_t, time_point_t> retired_;
    /// Keys opened since construction, for the teardown summary.
    uint64_t opens_ = 0;
};

#endif // NIXL_SRC_PLUGINS_REST_DC_REQUEST_KEYS_H
