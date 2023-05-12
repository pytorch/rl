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

torch::Tensor simplefilter(torch::Tensor input, float decay) {
  input = input.contiguous();
  torch::Tensor result = torch::empty_like(input);
  int rows = input.size(-1);

  torch::Tensor val = torch::zeros_like(input.select(-1, 0));
  for (int i = rows - 1; i >= 0; i--) {
    auto row = input.index({torch::indexing::Ellipsis, i});
    val = row + decay * val;
//    auto indices = torch::tensor({i}, torch::kLong);
    result.index_put_({torch::indexing::Ellipsis, i}, val);
  }
  return result;
}
