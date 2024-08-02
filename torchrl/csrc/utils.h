// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// utils.h

#pragma once

#include <torch/extension.h>
#include <torch/torch.h>

torch::Tensor safetanh(torch::Tensor input, float eps = 1e-6);
torch::Tensor safeatanh(torch::Tensor input, float eps = 1e-6);

class SafeTanh : public torch::autograd::Function<SafeTanh> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor input, float eps);
  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs);
};

class SafeInvTanh : public torch::autograd::Function<SafeInvTanh> {
 public:
  static torch::Tensor forward(torch::autograd::AutogradContext* ctx,
                               torch::Tensor input, float eps);
  static torch::autograd::tensor_list backward(
      torch::autograd::AutogradContext* ctx,
      torch::autograd::tensor_list grad_outputs);
};
