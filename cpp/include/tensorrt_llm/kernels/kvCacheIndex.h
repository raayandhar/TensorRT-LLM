/*
 * Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include "tensorrt_llm/common/assert.h"

#include <cstdint>
#include <cuda_runtime.h>

namespace tensorrt_llm::kernels
{

class KVCacheIndex
{
public:
    using UnderlyingType = std::int32_t;

    enum class PoolType : UnderlyingType
    {
        kPrimary = 0,   // Fast GPU memory
        kSecondary = 1, // DRAM / Disk
        kTertiary = 2,  // Disk
        // We could make a reserved pool type in the future?
        // We should never do GPU <-> Disk <-> DRAM. That's my working assumption, because I don't know why we would
        // ever do that.
    };

    // Pool type encoding uses top 2 bits now instead 1 bit.
    // The more I work on this the less I like it, or at least the less I like the way I'm doing it *_*
    // Have to be careful here because changes here can fail silently and be impossible to debug.
    static constexpr UnderlyingType kPoolTypeBits = 2;
    static constexpr UnderlyingType kPoolTypeShift = 8 * sizeof(UnderlyingType) - kPoolTypeBits;
    static constexpr UnderlyingType kPoolTypeMask = ((1 << kPoolTypeBits) - 1) << kPoolTypeShift;
    static constexpr UnderlyingType kIndexMask = ~kPoolTypeMask;

    // This might be the easiest for now, basically keep the old stuff but maybe should be changed.
    // I don't like the way I'm doing it :((
    static constexpr UnderlyingType kSecondaryPoolFlag = static_cast<UnderlyingType>(PoolType::kSecondary)
        << kPoolTypeShift;

    explicit KVCacheIndex(UnderlyingType value, PoolType poolType = PoolType::kPrimary)
        : value{(value & kIndexMask) | (static_cast<UnderlyingType>(poolType) << kPoolTypeShift)}
    {
        TLLM_CHECK_DEBUG(value >= 0);
        TLLM_CHECK_DEBUG((value & kPoolTypeMask) == 0); // Ensure no pool bits in input value
    }

    // Going to keep this for now.
    explicit KVCacheIndex(UnderlyingType value, bool isSecondary)
        : KVCacheIndex(value, isSecondary ? PoolType::kSecondary : PoolType::kPrimary)
    {
    }

    __host__ __device__ [[nodiscard]] UnderlyingType get() const
    {
        return value & kIndexMask;
    }

    __host__ __device__ [[nodiscard]] PoolType getPoolType() const
    {
        return static_cast<PoolType>((value & kPoolTypeMask) >> kPoolTypeShift);
    }

    __host__ __device__ [[nodiscard]] bool isPrimary() const
    {
        return getPoolType() == PoolType::kPrimary;
    }

    __host__ __device__ [[nodiscard]] bool isSecondary() const
    {
        return getPoolType() == PoolType::kSecondary;
    }

    __host__ __device__ [[nodiscard]] bool isTertiary() const
    {
        return getPoolType() == PoolType::kTertiary;
    }

private:
    UnderlyingType value;
};

} // namespace tensorrt_llm::kernels
