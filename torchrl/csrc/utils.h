// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include <torch/torch.h>

#include <iostream>

torch::Tensor safetanh(torch::Tensor input) {
  auto out = torch::tanh(input);
  auto data = out.data_ptr<float>();
  for (int64_t i = 0; i < out.numel(); ++i) {
    data[i] = std::clamp(data[i], -0.999999f, 0.999999f);
  }
  return out;
}
