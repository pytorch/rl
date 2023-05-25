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
  static torch::Tensor forward(AutogradContext* ctx, torch::Tensor input) {
    auto out = torch::tanh(input);
    ctx->save_for_backward({out});
    out = out.clamp(-0.999999, 0.999999);
    return out;
  }

  static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto saved = ctx->get_saved_variables();
    auto out = saved[0];
    auto go = grad_outputs[0];
    auto grad = go * (1 - out * out);
    return {grad};
  }
};

torch::Tensor safetanh(torch::Tensor input) { return SafeTanh::apply(input); }
