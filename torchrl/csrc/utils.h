// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include <torch/torch.h>

#include <iostream>

using namespace torch::autograd;

class SafeTanh : public Function<SafeTanh> {
 public:
  static torch::Tensor forward(AutogradContext* ctx, torch::Tensor input,
                               float eps = 1e-6) {
    auto out = torch::tanh(input);
    auto lim = 1.0 - eps;
    out = out.clamp(-lim, lim);
    ctx->save_for_backward({out});
    return out;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto out = saved[0];
    auto go = grad_outputs[0];
    auto grad = go * (1 - out * out);
    return {grad, torch::Tensor()};
  }
};

torch::Tensor safetanh(torch::Tensor input, float eps = 1e-6) {
  return SafeTanh::apply(input, eps);
}

class SafeInvTanh : public Function<SafeInvTanh> {
 public:
  static torch::Tensor forward(AutogradContext* ctx, torch::Tensor input,
                               float eps = 1e-6) {
    auto lim = 1.0 - eps;
    auto intermediate = input.clamp(-lim, lim);
    ctx->save_for_backward({intermediate});
    auto out = torch::atanh(intermediate);
    return out;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto input = saved[0];
    auto go = grad_outputs[0];
    auto grad = go / (1 - input * input);
    return {grad, torch::Tensor()};
  }
};

torch::Tensor safeatanh(torch::Tensor input, float eps = 1e-6) {
  return SafeInvTanh::apply(input, eps);
}
