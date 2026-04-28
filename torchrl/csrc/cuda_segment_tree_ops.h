// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace torchrl {

enum class CudaSegmentTreeOp : int64_t { Sum = 0, Min = 1 };

template <typename T>
void CudaSegmentTreeUpdate(torch::Tensor values, int64_t capacity,
                           torch::Tensor index, torch::Tensor value,
                           CudaSegmentTreeOp op);

template <typename T>
torch::Tensor CudaSegmentTreeQuery(torch::Tensor values, int64_t capacity,
                                   torch::Tensor l, torch::Tensor r,
                                   CudaSegmentTreeOp op);

template <typename T>
torch::Tensor CudaSumSegmentTreeScanLowerBound(torch::Tensor values,
                                               int64_t size, int64_t capacity,
                                               torch::Tensor value);

}  // namespace torchrl
