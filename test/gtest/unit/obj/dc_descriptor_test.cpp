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

/**
 * Unit tests for the Scality AI Connector DC descriptor wire format and bond
 * parsing. These helpers are libibverbs-free, so the test runs on any host.
 */
#include <gtest/gtest.h>

#include "rest_accel/scality_ai_connector/cufile_nics.h"
#include "rest_accel/scality_ai_connector/dc_descriptor.h"

namespace gtest::obj {

/** Golden DC descriptor string: ADDR:SIZE:RKEY:LID:DCTN:1:GID. */
TEST(DcDescriptorTest, FormatMatchesWireFormat) {
    const uint8_t gid[16] = {
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 0x0a, 0x0a, 0x30, 0xd0};
    std::string d = formatDcDescriptor(0x7f1234567890ULL, 0x100000U, 0xaabbccddU, 0, 0x42U, gid);
    EXPECT_EQ(d, "00007f1234567890:00100000:aabbccdd:0000:000042:1:00000000000000000000ffff0a0a30d0");
}

/** Fields are zero-padded to their fixed widths. */
TEST(DcDescriptorTest, FieldsAreZeroPadded) {
    const uint8_t gid[16] = {};
    std::string d = formatDcDescriptor(0, 0, 0, 0, 0, gid);
    EXPECT_EQ(d,
              "0000000000000000:00000000:00000000:0000:000000:1:"
              "00000000000000000000000000000000");
}

/** Bond slave count handles the usual sysfs line shapes. */
TEST(DcDescriptorTest, CountBondSlaves) {
    EXPECT_EQ(countBondSlaves("mlx5_0 mlx5_1\n"), 2);
    EXPECT_EQ(countBondSlaves("enp1s0f0 enp1s0f1 enp2s0f0 enp2s0f1\n"), 4);
    EXPECT_EQ(countBondSlaves("  single  \n"), 1);
    EXPECT_EQ(countBondSlaves(""), 0);
    EXPECT_EQ(countBondSlaves("\n"), 0);
    EXPECT_EQ(countBondSlaves(nullptr), 0);
}

/** cufile.json rdma_dev_addr_list extraction, tolerating // comments. */
TEST(DcDescriptorTest, ParseRdmaDevAddrList) {
    const std::string json = R"({
        "properties": {
            // client-side rdma list
            "rdma_dev_addr_list": [ "mlx5_1", "mlx5_2", "mlx5_7", "mlx5_8" ],
            "rdma_load_balancing_policy": "RoundRobin"
        },
        "fs": {
            "lustre": {
                //"rdma_dev_addr_list" : ["10.0.0.1"]
            }
        }
    })";
    std::vector<std::string> nics = parseRdmaDevAddrList(json);
    ASSERT_EQ(nics.size(), 4u);
    EXPECT_EQ(nics[0], "mlx5_1");
    EXPECT_EQ(nics[3], "mlx5_8");
}

/** IPv4 entries and an absent key. */
TEST(DcDescriptorTest, ParseRdmaDevAddrListIpsAndMissing) {
    EXPECT_EQ(parseRdmaDevAddrList("{ \"foo\": 1 }").size(), 0u);
    std::vector<std::string> nics =
        parseRdmaDevAddrList("\"rdma_dev_addr_list\": [\"10.10.40.208\", \"10.10.48.208\"]");
    ASSERT_EQ(nics.size(), 2u);
    EXPECT_EQ(nics[0], "10.10.40.208");
    EXPECT_EQ(nics[1], "10.10.48.208");
}

/** rdma_dc_key extraction: present, commented (absent), and missing. */
TEST(DcDescriptorTest, ParseRdmaDcKey) {
    EXPECT_EQ(parseRdmaDcKey("\"rdma_dc_key\": \"0x12345678\","), "0x12345678");
    // Commented out (the cufile.json default) -> treated as absent.
    EXPECT_EQ(parseRdmaDcKey("    //\"rdma_dc_key\": \"0xffeeddcc\""), "");
    EXPECT_EQ(parseRdmaDcKey("{ \"properties\": {} }"), "");
}

} // namespace gtest::obj
