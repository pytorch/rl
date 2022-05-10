// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <torch/torch.h>

#include <cstdint>

namespace torchrl {
namespace utils {

template <typename T>
struct TorchDataType;

template <>
struct TorchDataType<int64_t> {
  static constexpr torch::ScalarType value = torch::kInt64;
};

template <>
struct TorchDataType<float> {
  static constexpr torch::ScalarType value = torch::kFloat;
};

template <>
struct TorchDataType<double> {
  static constexpr torch::ScalarType value = torch::kDouble;
};

}  // namespace utils
}  // namespace torchrl
