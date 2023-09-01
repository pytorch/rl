// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <torch/extension.h>
#include <torch/torch.h>

#include <memory>

#include "segment_tree.h"
#include "utils.h"

namespace py = pybind11;

PYBIND11_MODULE(_torchrl, m) {
  torchrl::DefineSumSegmentTree<float>("Fp32", m);
  torchrl::DefineSumSegmentTree<double>("Fp64", m);

  torchrl::DefineMinSegmentTree<float>("Fp32", m);
  torchrl::DefineMinSegmentTree<double>("Fp64", m);

  m.def("safetanh", &safetanh, "Safe Tanh");
  m.def("safeatanh", &safeatanh, "Safe Inverse Tanh");
}
