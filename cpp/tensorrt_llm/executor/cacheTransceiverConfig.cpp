/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include "tensorrt_llm/common/logger.h"
#include "tensorrt_llm/executor/executor.h"

namespace tensorrt_llm::executor
{

CacheTransceiverConfig::CacheTransceiverConfig(
    std::optional<BackendType> backendType, std::optional<size_t> maxNumTokens,
    std::optional<std::chrono::milliseconds> kvTransferTimeoutMs)
    : mBackendType(backendType)
    , mMaxTokensInBuffer(maxNumTokens)
    , mKvTransferTimeoutMs(kvTransferTimeoutMs)
{
    if (kvTransferTimeoutMs.has_value()) {
        TLLM_LOG_INFO("CacheTransceiverConfig initialized with KV transfer timeout: %ld ms", 
                      kvTransferTimeoutMs.value().count());
    } else {
        // On gen server, this is getting printed? (actually not sure it's gen...)
        // See logs:
        // [TensorRT-LLM][INFO] CacheTransceiverConfig initialized without KV transfer timeout        
        // [DEBUG] Setting KV cache transfer timeout to 5000ms 
        // The above is coming from llm_args.py ... (??? what's happening here?)
        // Actually, the same is happening on ctx side to, see below:
        // [TensorRT-LLM][INFO] CacheTransceiverConfig initialized without KV transfer timeout
        // [DEBUG] Setting KV cache transfer timeout to 10000ms
        // This happens at L257 in the logs.
        // This is also hit twice at L265/266:
        // [TensorRT-LLM][INFO] CacheTransceiverConfig initialized without KV transfer timeout
        // [TensorRT-LLM][INFO] CacheTransceiverConfig initialized without KV transfer timeout
        TLLM_LOG_INFO("CacheTransceiverConfig initialized without KV transfer timeout");
    }
}

bool CacheTransceiverConfig::operator==(CacheTransceiverConfig const& other) const
{
    return mMaxTokensInBuffer == other.mMaxTokensInBuffer && mBackendType == other.mBackendType
        && mKvTransferTimeoutMs == other.mKvTransferTimeoutMs;
}

void CacheTransceiverConfig::setBackendType(std::optional<BackendType> backendType)
{
    mBackendType = backendType;
}

void CacheTransceiverConfig::setMaxTokensInBuffer(std::optional<size_t> maxTokensInBuffer)
{
    mMaxTokensInBuffer = maxTokensInBuffer;
}

std::optional<CacheTransceiverConfig::BackendType> CacheTransceiverConfig::getBackendType() const
{
    return mBackendType;
}

std::optional<size_t> CacheTransceiverConfig::getMaxTokensInBuffer() const
{
    return mMaxTokensInBuffer;
}

void CacheTransceiverConfig::setKvTransferTimeoutMs(std::optional<std::chrono::milliseconds> kvTransferTimeoutMs)
{
    mKvTransferTimeoutMs = kvTransferTimeoutMs;
}

std::optional<std::chrono::milliseconds> CacheTransceiverConfig::getKvTransferTimeoutMs() const
{
    return mKvTransferTimeoutMs;
}

} // namespace tensorrt_llm::executor
