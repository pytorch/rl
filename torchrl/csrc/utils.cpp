// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.
// utils.h
#include "utils.h"

#include <iostream>
torch::Tensor safetanh(torch::Tensor input, float eps) {
  return SafeTanh::apply(input, eps);
}
torch::Tensor safeatanh(torch::Tensor input, float eps) {
  return SafeInvTanh::apply(input, eps);
}
torch::Tensor SafeTanh::forward(torch::autograd::AutogradContext* ctx,
                                torch::Tensor input, float eps) {
  auto out = torch::tanh(input);
  auto lim = 1.0 - eps;
  out = out.clamp(-lim, lim);
  ctx->save_for_backward({out});
  return out;
}
torch::autograd::tensor_list SafeTanh::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto out = saved[0];
  auto go = grad_outputs[0];
  auto grad = go * (1 - out * out);
  return {grad, torch::Tensor()};
}
torch::Tensor SafeInvTanh::forward(torch::autograd::AutogradContext* ctx,
                                   torch::Tensor input, float eps) {
  auto lim = 1.0 - eps;
  auto intermediate = input.clamp(-lim, lim);
  ctx->save_for_backward({intermediate});
  auto out = torch::atanh(intermediate);
  return out;
}
torch::autograd::tensor_list SafeInvTanh::backward(
    torch::autograd::AutogradContext* ctx,
    torch::autograd::tensor_list grad_outputs) {
  auto saved = ctx->get_saved_variables();
  auto input = saved[0];
  auto go = grad_outputs[0];
  auto grad = go / (1 - input * input);
  return {grad, torch::Tensor()};
}
