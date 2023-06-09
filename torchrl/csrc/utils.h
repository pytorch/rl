// Copyright (c) Meta Platforms, Inc. and affiliates.
//
// This source code is licensed under the MIT license found in the
// LICENSE file in the root directory of this source tree.

#include <torch/extension.h>
#include <torch/torch.h>

#include <iostream>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

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

py::object unravel_keys(const py::object& key, bool make_tuple = false) {
    if (py::isinstance<py::str>(key)) {
        if (make_tuple) {
            return py::make_tuple(key);
        }
        return key;
    }
    if (py::isinstance<py::tuple>(key)) {
        py::list newkey;
        for (const auto& subkey : key) {
            if (py::isinstance<py::str>(subkey)) {
                newkey.append(subkey);
            } else {
                auto _key = unravel_keys(subkey.cast<py::object>());
                for (const auto& k : _key) {
                    newkey.append(k);
                }
            }
        }
        return py::tuple(newkey);
    } else {
        throw std::runtime_error("key should be a Sequence<NestedKey>");
    }
}
