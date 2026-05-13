/*
 * SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef NIXL_TEST_GTEST_UNIT_OBJ_MOCK_S3_CLIENT_H
#define NIXL_TEST_GTEST_UNIT_OBJ_MOCK_S3_CLIENT_H

#include "obj_backend.h"
#include "obj_executor.h"
#include <chrono>
#include <functional>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace gtest::obj {

class mockS3Client : public iS3Client {
private:
    bool simulateSuccess_ = true;
    bool simulateError_ = false;
    std::shared_ptr<asioThreadPoolExecutor> executor_;
    std::vector<std::function<void()>> pendingCallbacks_;
    std::set<std::string> checkedKeys_;
    std::map<std::string, bool> keyOutcomes_;
    std::map<std::string, std::chrono::milliseconds> keyDelays_;
    std::set<std::string> keyErrors_;

public:
    mockS3Client() = default;

    mockS3Client([[maybe_unused]] nixl_b_params_t *custom_params,
                 std::shared_ptr<Aws::Utils::Threading::Executor> executor = nullptr) {
        if (executor) {
            executor_ = std::dynamic_pointer_cast<asioThreadPoolExecutor>(executor);
        }
    }

    void
    setSimulateSuccess(bool success) {
        simulateSuccess_ = success;
    }

    void
    setSimulateError(bool error) {
        simulateError_ = error;
    }

    void
    setKeyOutcome(const std::string &key, bool success) {
        keyOutcomes_[key] = success;
    }

    void
    setKeyDelay(const std::string &key, std::chrono::milliseconds delay) {
        keyDelays_[key] = delay;
    }

    void
    setKeyError(const std::string &key) {
        keyErrors_.insert(key);
    }

    void
    setExecutor(std::shared_ptr<Aws::Utils::Threading::Executor> executor) override {
        executor_ = std::dynamic_pointer_cast<asioThreadPoolExecutor>(executor);
    }

    void
    putObjectAsync([[maybe_unused]] std::string_view key,
                   [[maybe_unused]] uintptr_t data_ptr,
                   [[maybe_unused]] size_t data_len,
                   [[maybe_unused]] size_t offset,
                   put_object_callback_t callback) override {
        pendingCallbacks_.push_back([callback, this]() { callback(simulateSuccess_); });
    }

    void
    getObjectAsync([[maybe_unused]] std::string_view key,
                   uintptr_t data_ptr,
                   size_t data_len,
                   size_t offset,
                   get_object_callback_t callback) override {
        pendingCallbacks_.push_back([callback, data_ptr, data_len, offset, this]() {
            if (simulateSuccess_ && data_ptr && data_len > 0) {
                char *buffer = reinterpret_cast<char *>(data_ptr);
                for (size_t i = 0; i < data_len; ++i) {
                    buffer[i] = static_cast<char>('A' + ((i + offset) % 26));
                }
            }
            callback(simulateSuccess_);
        });
    }

    void
    checkObjectExistsAsync(std::string_view key, check_object_callback_t callback) override {
        std::string key_str(key);
        checkedKeys_.insert(key_str);

        bool simulate_error = keyErrors_.count(key_str) > 0 || simulateError_;

        auto outcome_it = keyOutcomes_.find(key_str);
        bool success = (outcome_it != keyOutcomes_.end()) ? outcome_it->second : simulateSuccess_;

        std::chrono::milliseconds delay{0};
        auto delay_it = keyDelays_.find(key_str);
        if (delay_it != keyDelays_.end()) {
            delay = delay_it->second;
        }

        if (executor_) {
            executor_->Submit([callback, success, simulate_error, delay]() {
                if (delay.count() > 0) {
                    std::this_thread::sleep_for(delay);
                }
                callback(simulate_error ? std::optional<bool>(std::nullopt) :
                                          std::optional<bool>(success));
            });
        } else {
            if (delay.count() > 0) {
                std::this_thread::sleep_for(delay);
            }
            callback(simulate_error ? std::optional<bool>(std::nullopt) :
                                      std::optional<bool>(success));
        }
    }

    void
    execAsync() {
        if (!executor_) throw std::runtime_error("mockS3Client::execAsync: executor not set");
        for (auto &callback : pendingCallbacks_) {
            executor_->Submit([callback]() { callback(); });
        }
        pendingCallbacks_.clear();
        executor_->waitUntilIdle();
    }

    size_t
    getPendingCount() const {
        return pendingCallbacks_.size();
    }

    const std::set<std::string> &
    getCheckedKeys() const {
        return checkedKeys_;
    }

    bool
    hasExecutor() const {
        return executor_ != nullptr;
    }

protected:
    std::vector<std::function<void()>> &
    getPendingCallbacks() {
        return pendingCallbacks_;
    }

    bool
    getSimulateSuccess() const {
        return simulateSuccess_;
    }
};

} // namespace gtest::obj

#endif // NIXL_TEST_GTEST_UNIT_OBJ_MOCK_S3_CLIENT_H
