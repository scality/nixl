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

#ifndef NIXL_OBJ_PLUGIN_S3_ACCEL_DC_DESCRIPTOR_H
#define NIXL_OBJ_PLUGIN_S3_ACCEL_DC_DESCRIPTOR_H

#include <cinttypes>
#include <cstdint>
#include <cstdio>
#include <string>

/**
 * Pure helpers for the DC RDMA descriptor wire format, kept free of libibverbs
 * so they can be unit-tested on any host. The format matches the Scality server
 * parser (and the bench rdma-helper):
 *   ADDR(16hex):SIZE(8hex):RKEY(8hex):LID(4hex):DCTN(6hex):GID_PRESENT(1):GID(32hex)
 */

/// Maximum DCT QPs per NIC (one per physical LAG/bond port).
static constexpr int kDcMaxLagPorts = 4;

/**
 * Format a DC descriptor string. addr is the buffer virtual address the server
 * will RDMA against, size the transfer length (32-bit field), rkey the MR remote
 * key, lid the local id (0 on RoCEv2), dctn the DC target QP number, and gid the
 * 16-byte RoCEv2 GID. GID_PRESENT is always 1 (RoCEv2).
 */
inline std::string
formatDcDescriptor(uint64_t addr,
                   uint32_t size,
                   uint32_t rkey,
                   uint16_t lid,
                   uint32_t dctn,
                   const uint8_t gid[16]) {
    char gid_hex[33];
    for (int i = 0; i < 16; i++) {
        snprintf(gid_hex + 2 * i, 3, "%02x", gid[i]);
    }
    gid_hex[32] = '\0';

    char buf[128];
    snprintf(buf,
             sizeof(buf),
             "%016" PRIx64 ":%08" PRIx32 ":%08" PRIx32 ":%04" PRIx16 ":%06" PRIx32 ":1:%s",
             addr,
             size,
             rkey,
             lid,
             dctn,
             gid_hex);
    return std::string(buf);
}

/**
 * Count the space/newline-separated entries in a bonding "slaves" line (the
 * contents of /sys/class/net/<bond>/bonding/slaves). Returns the slave count,
 * or 0 for an empty line. Callers clamp to [1, kDcMaxLagPorts].
 */
inline int
countBondSlaves(const char *slaves_line) {
    if (slaves_line == nullptr) {
        return 0;
    }
    int n = 0;
    for (const char *p = slaves_line; *p;) {
        while (*p == ' ' || *p == '\n' || *p == '\r' || *p == '\t') {
            p++;
        }
        if (*p) {
            n++;
            while (*p && *p != ' ' && *p != '\n' && *p != '\r' && *p != '\t') {
                p++;
            }
        }
    }
    return n;
}

#endif // NIXL_OBJ_PLUGIN_S3_ACCEL_DC_DESCRIPTOR_H
