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

#if defined(USE_IBVERBS_RC_RDMA)

#include "ibverbs_rc_rdma_token_client.h"
#include "common/nixl_log.h"

#include <absl/strings/ascii.h>
#include <absl/strings/str_format.h>
#include <absl/strings/str_join.h>
#include <absl/strings/str_split.h>

#include <algorithm>
#include <arpa/inet.h>
#include <cerrno>
#include <fcntl.h>
#include <cstring>
#include <dirent.h>
#include <fstream>
#include <ifaddrs.h>
#include <net/if.h>
#include <sys/epoll.h>
#include <unistd.h>

// ---------------------------------------------------------------------------
// Construction / destruction
// ---------------------------------------------------------------------------

IbverbsRcRdmaTokenClient::IbverbsRcRdmaTokenClient(CUObjOps_t &ops,
                                                     const std::string &advertiseIp)
    : userOps_(ops) {
    if (!advertiseIp.empty()) {
        advertiseIps_ = absl::StrSplit(advertiseIp, ',', absl::SkipWhitespace());
        for (auto &ip : advertiseIps_) {
            ip = std::string(absl::StripAsciiWhitespace(ip));
        }
    } else {
        advertiseIps_ = detectAdvertiseIps();
    }

    if (advertiseIps_.empty()) {
        NIXL_ERROR << "IbverbsRcRdmaTokenClient: could not determine advertise IP";
        return;
    }

    epollFd_ = epoll_create1(EPOLL_CLOEXEC);
    if (epollFd_ < 0) {
        NIXL_ERROR << "epoll_create1 failed: " << strerror(errno);
        return;
    }

    running_ = true;
    acceptThread_ = std::thread(&IbverbsRcRdmaTokenClient::acceptLoop, this);
    connected_ = true;

    NIXL_INFO << "IbverbsRcRdmaTokenClient initialized (RC transport), advertiseIps="
              << absl::StrJoin(advertiseIps_, ",");
}

IbverbsRcRdmaTokenClient::~IbverbsRcRdmaTokenClient() {
    NIXL_DEBUG << "IbverbsRcRdmaTokenClient destructor: stopping accept loop";
    running_ = false;

    if (epollFd_ >= 0) {
        close(epollFd_);
        epollFd_ = -1;
    }

    if (acceptThread_.joinable()) {
        NIXL_DEBUG << "IbverbsRcRdmaTokenClient destructor: joining accept thread";
        acceptThread_.join();
        NIXL_DEBUG << "IbverbsRcRdmaTokenClient destructor: accept thread joined";
    }

    // Tear down any remaining regions
    std::lock_guard<std::mutex> lk(mu_);
    for (auto &[ptr, reg] : regions_) {
        for (auto *id : reg.active_conns) {
            rdma_disconnect(id);
            if (id->qp) rdma_destroy_qp(id);
            rdma_destroy_id(id);
        }
        reg.active_conns.clear();
        if (reg.mr) ibv_dereg_mr(reg.mr);
        if (reg.pd) ibv_dealloc_pd(reg.pd);
        if (reg.listen_id) rdma_destroy_id(reg.listen_id);
        if (reg.cm_channel) rdma_destroy_event_channel(reg.cm_channel);
    }
    regions_.clear();
}

// ---------------------------------------------------------------------------
// IP auto-detection
// ---------------------------------------------------------------------------

std::vector<std::string>
IbverbsRcRdmaTokenClient::detectAdvertiseIps() {
    // Walk /sys/class/infiniband/<dev>/ports/1/gid_attrs/ndevs/0 to find the
    // parent netdev of every ACTIVE device, then resolve their IPv4 addresses
    // via getifaddrs().
    const char *sysfs_base = "/sys/class/infiniband";
    DIR *dir = opendir(sysfs_base);
    if (!dir) {
        NIXL_WARN << "Cannot open " << sysfs_base;
        return {};
    }

    std::vector<std::string> netdevs;
    struct dirent *ent;
    while ((ent = readdir(dir)) != nullptr) {
        if (ent->d_name[0] == '.') continue;

        // Skip devices whose port 1 is not ACTIVE
        std::string statePath =
            std::string(sysfs_base) + "/" + ent->d_name + "/ports/1/state";
        std::ifstream stateFile(statePath);
        if (stateFile.good()) {
            std::string state;
            std::getline(stateFile, state);
            if (state.find("ACTIVE") == std::string::npos) {
                NIXL_DEBUG << "Skipping RDMA device " << ent->d_name
                           << " (port state: " << state << ")";
                continue;
            }
        }

        std::string netdev;
        // Try the "parent" symlink first (typical for RXE / SIW)
        std::string parentPath =
            std::string(sysfs_base) + "/" + ent->d_name + "/parent";
        std::ifstream parentFile(parentPath);
        if (parentFile.good()) {
            std::getline(parentFile, netdev);
        }
        // Fallback: gid_attrs ndev
        if (netdev.empty()) {
            std::string ndevPath =
                std::string(sysfs_base) + "/" + ent->d_name + "/ports/1/gid_attrs/ndevs/0";
            std::ifstream ndevFile(ndevPath);
            if (ndevFile.good()) {
                std::getline(ndevFile, netdev);
            }
        }
        if (!netdev.empty() &&
            std::find(netdevs.begin(), netdevs.end(), netdev) == netdevs.end()) {
            NIXL_DEBUG << "Detected RDMA parent netdev: " << netdev << " (device "
                       << ent->d_name << ")";
            netdevs.push_back(netdev);
        }
    }
    closedir(dir);

    if (netdevs.empty()) {
        NIXL_WARN << "Could not find parent netdev for any RDMA device";
        return {};
    }

    // Resolve each netdev → IPv4 address
    struct ifaddrs *ifas = nullptr;
    if (getifaddrs(&ifas) != 0) {
        NIXL_WARN << "getifaddrs failed: " << strerror(errno);
        return {};
    }

    std::vector<std::string> ips;
    for (const auto &netdev : netdevs) {
        std::string ip;
        for (struct ifaddrs *ifa = ifas; ifa; ifa = ifa->ifa_next) {
            if (!ifa->ifa_addr || ifa->ifa_addr->sa_family != AF_INET) continue;
            if (netdev != ifa->ifa_name) continue;
            char buf[INET_ADDRSTRLEN];
            auto *sin = reinterpret_cast<struct sockaddr_in *>(ifa->ifa_addr);
            inet_ntop(AF_INET, &sin->sin_addr, buf, sizeof(buf));
            ip = buf;
            break;
        }
        if (ip.empty()) {
            NIXL_WARN << "No IPv4 address found for netdev " << netdev;
        } else if (std::find(ips.begin(), ips.end(), ip) == ips.end()) {
            NIXL_DEBUG << "Resolved advertise IP: " << ip << " (netdev " << netdev << ")";
            ips.push_back(ip);
        }
    }
    freeifaddrs(ifas);

    return ips;
}

// ---------------------------------------------------------------------------
// isConnected
// ---------------------------------------------------------------------------

bool
IbverbsRcRdmaTokenClient::isConnected() const {
    return connected_;
}

// ---------------------------------------------------------------------------
// cuMemObjGetDescriptor — register a buffer and create a listener
// ---------------------------------------------------------------------------

cuObjErr_t
IbverbsRcRdmaTokenClient::cuMemObjGetDescriptor(void *ptr, size_t size) {
    RdmaMemRegion reg;

    // 1. Create CM channel + id (non-blocking so rdma_get_cm_event won't hang)
    reg.cm_channel = rdma_create_event_channel();
    if (!reg.cm_channel) {
        NIXL_ERROR << "rdma_create_event_channel failed: " << strerror(errno);
        return CU_OBJ_FAIL;
    }
    {
        int flags = fcntl(reg.cm_channel->fd, F_GETFL);
        fcntl(reg.cm_channel->fd, F_SETFL, flags | O_NONBLOCK);
    }

    if (rdma_create_id(reg.cm_channel, &reg.listen_id, nullptr, RDMA_PS_TCP) != 0) {
        NIXL_ERROR << "rdma_create_id failed: " << strerror(errno);
        rdma_destroy_event_channel(reg.cm_channel);
        return CU_OBJ_FAIL;
    }

    // 2. Bind to the advertise IP:0 so RDMA CM resolves the correct device.
    //    INADDR_ANY does not work with RXE on veth interfaces.
    //    Registrations are round-robined across the configured advertise IPs
    //    so multi-rail setups spread buffers (and thus traffic) over all NICs.
    reg.advertise_ip = advertiseIps_[nextIpIdx_.fetch_add(1) % advertiseIps_.size()];
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = 0;
    if (inet_pton(AF_INET, reg.advertise_ip.c_str(), &addr.sin_addr) != 1) {
        NIXL_ERROR << "Invalid advertise IP: " << reg.advertise_ip;
        rdma_destroy_id(reg.listen_id);
        rdma_destroy_event_channel(reg.cm_channel);
        return CU_OBJ_FAIL;
    }

    if (rdma_bind_addr(reg.listen_id, reinterpret_cast<struct sockaddr *>(&addr)) != 0) {
        NIXL_ERROR << "rdma_bind_addr failed: " << strerror(errno);
        rdma_destroy_id(reg.listen_id);
        rdma_destroy_event_channel(reg.cm_channel);
        return CU_OBJ_FAIL;
    }

    // 3. Listen
    if (rdma_listen(reg.listen_id, 128) != 0) {
        NIXL_ERROR << "rdma_listen failed: " << strerror(errno);
        rdma_destroy_id(reg.listen_id);
        rdma_destroy_event_channel(reg.cm_channel);
        return CU_OBJ_FAIL;
    }

    // 4. Allocate PD + register MR
    if (!reg.listen_id->verbs) {
        NIXL_ERROR << "No RDMA device found after rdma_bind_addr "
                   << "(is an RXE/Soft-RoCE device configured?)";
        rdma_destroy_id(reg.listen_id);
        rdma_destroy_event_channel(reg.cm_channel);
        return CU_OBJ_FAIL;
    }

    reg.pd = ibv_alloc_pd(reg.listen_id->verbs);
    if (!reg.pd) {
        NIXL_ERROR << "ibv_alloc_pd failed: " << strerror(errno);
        rdma_destroy_id(reg.listen_id);
        rdma_destroy_event_channel(reg.cm_channel);
        return CU_OBJ_FAIL;
    }

    reg.mr = ibv_reg_mr(reg.pd, ptr, size,
                         IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_READ |
                         IBV_ACCESS_REMOTE_WRITE);
    if (!reg.mr) {
        NIXL_ERROR << "ibv_reg_mr failed: " << strerror(errno);
        ibv_dealloc_pd(reg.pd);
        rdma_destroy_id(reg.listen_id);
        rdma_destroy_event_channel(reg.cm_channel);
        return CU_OBJ_FAIL;
    }

    // 5. Store base registration info (per-transfer tokens generated in cuObjGet/Put)
    auto *local_addr = rdma_get_local_addr(reg.listen_id);
    reg.port = ntohs(reinterpret_cast<const struct sockaddr_in *>(local_addr)->sin_port);
    reg.base_addr = reinterpret_cast<uintptr_t>(reg.mr->addr);
    reg.rkey = reg.mr->rkey;

    NIXL_DEBUG << "Registered RDMA region: ptr=" << ptr << " size=" << size
               << " ip=" << reg.advertise_ip << " port=" << reg.port << " rkey=0x" << std::hex
               << reg.rkey;

    // 7. Add CM channel fd to epoll
    int fd = reg.cm_channel->fd;
    struct epoll_event ev{};
    ev.events = EPOLLIN;
    ev.data.fd = fd;
    if (epoll_ctl(epollFd_, EPOLL_CTL_ADD, fd, &ev) != 0) {
        NIXL_ERROR << "epoll_ctl ADD failed: " << strerror(errno);
        ibv_dereg_mr(reg.mr);
        ibv_dealloc_pd(reg.pd);
        rdma_destroy_id(reg.listen_id);
        rdma_destroy_event_channel(reg.cm_channel);
        return CU_OBJ_FAIL;
    }

    // 8. Store region
    std::lock_guard<std::mutex> lk(mu_);
    regions_[ptr] = std::move(reg);
    return CU_OBJ_SUCCESS;
}

// ---------------------------------------------------------------------------
// cuMemObjPutDescriptor — tear down a registered buffer
// ---------------------------------------------------------------------------

cuObjErr_t
IbverbsRcRdmaTokenClient::cuMemObjPutDescriptor(void *ptr) {
    std::lock_guard<std::mutex> lk(mu_);
    auto it = regions_.find(ptr);
    if (it == regions_.end()) {
        NIXL_ERROR << "cuMemObjPutDescriptor: ptr not found in regions map";
        return CU_OBJ_FAIL;
    }

    RdmaMemRegion &reg = it->second;

    // Disconnect and destroy all active QPs on this region
    for (auto *id : reg.active_conns) {
        rdma_disconnect(id);
        if (id->qp) rdma_destroy_qp(id);
        rdma_destroy_id(id);
    }
    reg.active_conns.clear();

    // Remove from epoll
    if (epollFd_ >= 0 && reg.cm_channel) {
        epoll_ctl(epollFd_, EPOLL_CTL_DEL, reg.cm_channel->fd, nullptr);
    }

    if (reg.mr) ibv_dereg_mr(reg.mr);
    if (reg.pd) ibv_dealloc_pd(reg.pd);
    if (reg.listen_id) rdma_destroy_id(reg.listen_id);
    if (reg.cm_channel) rdma_destroy_event_channel(reg.cm_channel);

    regions_.erase(it);
    NIXL_DEBUG << "Deregistered RDMA region: ptr=" << ptr;
    return CU_OBJ_SUCCESS;
}

// ---------------------------------------------------------------------------
// findRegion — locate the registered MR containing a given address
// ---------------------------------------------------------------------------

const RdmaMemRegion *
IbverbsRcRdmaTokenClient::findRegion(uintptr_t addr) const {
    for (const auto &[ptr, reg] : regions_) {
        uintptr_t start = reg.base_addr;
        uintptr_t end = start + reg.mr->length;
        if (addr >= start && addr < end) {
            return &reg;
        }
    }
    return nullptr;
}

// ---------------------------------------------------------------------------
// cuObjGet / cuObjPut — invoke user callbacks with the RDMA token
// ---------------------------------------------------------------------------

ssize_t
IbverbsRcRdmaTokenClient::cuObjGet(void *ctx, void *ptr, size_t size, loff_t offset,
                                    loff_t buf_offset) {
    std::string token;
    {
        std::lock_guard<std::mutex> lk(mu_);
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr) + static_cast<uintptr_t>(buf_offset);
        const RdmaMemRegion *reg = findRegion(addr);
        if (!reg) {
            NIXL_ERROR << "cuObjGet: ptr not within any registered region";
            return -EINVAL;
        }

        // Per-transfer token: IP:PORT:0xADDR:0xSIZE:0xRKEY
        token = absl::StrFormat("%s:%d:0x%016x:0x%08x:0x%08x",
                                reg->advertise_ip, reg->port,
                                addr,
                                static_cast<uint32_t>(size),
                                reg->rkey);
    }

    cufileRDMAInfo_t rdmaInfo{};
    rdmaInfo.version = 1;
    rdmaInfo.desc_len = static_cast<int>(token.size());
    rdmaInfo.desc_str = token.c_str();

    MockHandle mh{MockHandle::MOCK_MAGIC, ctx};
    return userOps_.get(&mh,
                        static_cast<char *>(ptr) + buf_offset,
                        size, offset, &rdmaInfo);
}

ssize_t
IbverbsRcRdmaTokenClient::cuObjPut(void *ctx, void *ptr, size_t size, loff_t offset,
                                    loff_t buf_offset) {
    std::string token;
    {
        std::lock_guard<std::mutex> lk(mu_);
        uintptr_t addr = reinterpret_cast<uintptr_t>(ptr) + static_cast<uintptr_t>(buf_offset);
        const RdmaMemRegion *reg = findRegion(addr);
        if (!reg) {
            NIXL_ERROR << "cuObjPut: ptr not within any registered region";
            return -EINVAL;
        }

        // Per-transfer token: IP:PORT:0xADDR:0xSIZE:0xRKEY
        token = absl::StrFormat("%s:%d:0x%016x:0x%08x:0x%08x",
                                reg->advertise_ip, reg->port,
                                addr,
                                static_cast<uint32_t>(size),
                                reg->rkey);
    }

    cufileRDMAInfo_t rdmaInfo{};
    rdmaInfo.desc_str = token.c_str();

    MockHandle mh{MockHandle::MOCK_MAGIC, ctx};
    return userOps_.put(&mh,
                        static_cast<const char *>(ptr) + buf_offset,
                        size, offset, &rdmaInfo);
}

// ---------------------------------------------------------------------------
// Background accept thread
// ---------------------------------------------------------------------------

void
IbverbsRcRdmaTokenClient::acceptLoop() {
    constexpr int MAX_EVENTS = 16;
    struct epoll_event events[MAX_EVENTS];

    while (running_.load()) {
        int fd = epollFd_;
        if (fd < 0) break;
        int nfds = epoll_wait(fd, events, MAX_EVENTS, 200 /* ms */);
        if (nfds < 0) {
            if (errno == EINTR || errno == EBADF) continue;
            if (!running_.load()) break;
            NIXL_ERROR << "epoll_wait failed: " << strerror(errno);
            break;
        }
        if (!running_.load()) break;

        for (int i = 0; i < nfds; ++i) {
            // Find the region for this fd
            rdma_event_channel *ch = nullptr;
            RdmaMemRegion *region = nullptr;
            {
                std::lock_guard<std::mutex> lk(mu_);
                for (auto &[ptr, reg] : regions_) {
                    if (reg.cm_channel && reg.cm_channel->fd == events[i].data.fd) {
                        ch = reg.cm_channel;
                        region = &reg;
                        break;
                    }
                }
            }
            if (!ch || !region) continue;

            rdma_cm_event *event = nullptr;
            while (rdma_get_cm_event(ch, &event) == 0) {
                // Copy event data before ack
                rdma_cm_event event_copy = *event;
                rdma_ack_cm_event(event);
                handleCmEvent(&event_copy, *region);
            }
        }
    }
}

void
IbverbsRcRdmaTokenClient::handleCmEvent(rdma_cm_event *event, RdmaMemRegion &region) {
    switch (event->event) {
    case RDMA_CM_EVENT_CONNECT_REQUEST: {
        rdma_cm_id *new_id = event->id;

        if (!region.pd) {
            NIXL_ERROR << "No PD provided for incoming connection";
            rdma_reject(new_id, nullptr, 0);
            return;
        }

        // Create RC QP
        ibv_qp_init_attr qp_attr{};
        qp_attr.qp_type = IBV_QPT_RC;
        qp_attr.cap.max_send_wr = 16;
        qp_attr.cap.max_recv_wr = 16;
        qp_attr.cap.max_send_sge = 1;
        qp_attr.cap.max_recv_sge = 1;

        if (rdma_create_qp(new_id, region.pd, &qp_attr) != 0) {
            NIXL_ERROR << "rdma_create_qp failed: " << strerror(errno);
            rdma_reject(new_id, nullptr, 0);
            return;
        }

        rdma_conn_param conn_param{};
        conn_param.responder_resources = 16;
        conn_param.initiator_depth = 16;

        if (rdma_accept(new_id, &conn_param) != 0) {
            NIXL_ERROR << "rdma_accept failed: " << strerror(errno);
            rdma_destroy_qp(new_id);
            rdma_destroy_id(new_id);
            return;
        }

        // Track the accepted connection in the owning region
        {
            std::lock_guard<std::mutex> lk(mu_);
            region.active_conns.push_back(new_id);
        }
        NIXL_DEBUG << "Accepted RDMA RC connection";
        break;
    }
    case RDMA_CM_EVENT_ESTABLISHED:
        NIXL_DEBUG << "RDMA RC connection established";
        break;
    case RDMA_CM_EVENT_DISCONNECTED: {
        rdma_cm_id *disc_id = event->id;

        // Remove from the owning region's active connections
        {
            std::lock_guard<std::mutex> lk(mu_);
            auto &conns = region.active_conns;
            auto it = std::find(conns.begin(), conns.end(), disc_id);
            if (it != conns.end()) {
                conns.erase(it);
            }
        }

        if (disc_id->qp) {
            rdma_destroy_qp(disc_id);
        }
        rdma_destroy_id(disc_id);
        NIXL_DEBUG << "RDMA RC connection disconnected and cleaned up";
        break;
    }
    default:
        NIXL_DEBUG << "Unhandled CM event: " << rdma_event_str(event->event);
        break;
    }
}

size_t
IbverbsRcRdmaTokenClient::getActiveConnectionCount() const {
    std::lock_guard<std::mutex> lk(mu_);
    size_t count = 0;
    for (const auto &[ptr, reg] : regions_) {
        count += reg.active_conns.size();
    }
    return count;
}

#endif // USE_IBVERBS_RC_RDMA
