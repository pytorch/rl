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
}  // namespace rloptim
