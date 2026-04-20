// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <pybind11/numpy.h>

#include <algorithm>
#include <cstdint>
#include <vector>

namespace py = pybind11;

namespace torchrl {
namespace utils {

template <typename T>
std::vector<int64_t> NumpyArrayShape(const py::array_t<T>& arr) {
  const int64_t ndim = arr.ndim();
  std::vector<int64_t> shape(ndim);
  for (int64_t i = 0; i < ndim; ++i) {
    shape[i] = static_cast<int64_t>(arr.shape(i));
  }
  return shape;
}

template <typename T_SRC, typename T_DST = T_SRC>
py::array_t<T_DST> NumpyEmptyLike(const py::array_t<T_SRC>& src) {
  py::array_t<T_DST> dst(src.size());
  const std::vector<int64_t> shape = NumpyArrayShape(src);
  dst.resize(shape);
  return dst;
}

}  // namespace utils
}  // namespace torchrl
