// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include <torch/torch.h>

#include <iostream>

torch::Tensor safetanh(torch::Tensor input) {
  return torch::clamp(torch::tanh(input), -0.999999, 0.999999);
}
