# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn


class BatchRenorm(nn.Module):
    """
    BatchRenorm Module (https://arxiv.org/abs/1702.03275).

    BatchRenorm is an enhanced version of the standard BatchNorm. Unlike BatchNorm,
    BatchRenorm utilizes running statistics to normalize batches after an initial warmup phase.
    This approach reduces the impact of "outlier" batches that may occur during extended training periods,
    making BatchRenorm more robust for long training runs.

    During the warmup phase, BatchRenorm functions identically to a BatchNorm layer.

    Args:
        num_features (int): Number of features in the input tensor.

    Keyword Args:
        momentum (float, optional): Momentum factor for computing the running mean and variance. Default is 0.01.
        eps (float, optional): Small value added to the variance to avoid division by zero. Default is 1e-5.
        max_r (float, optional): Maximum value for the scaling factor r. Default is 3.0.
        max_d (float, optional): Maximum value for the bias factor d. Default is 5.0.
        warmup_steps (int, optional): Number of warm-up steps for the running mean and variance. Default is 10000.
    """

    def __init__(
        self,
        num_features,
        momentum=0.01,
        eps=1e-5,
        max_r=3.0,
        max_d=5.0,
        warmup_steps=10000,
    ):
        super(BatchRenorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.max_r = max_r
        self.max_d = max_d
        self.warmup_steps = warmup_steps

        self.register_buffer(
            "running_mean", torch.zeros(num_features, dtype=torch.float32)
        )
        self.register_buffer(
            "running_var", torch.ones(num_features, dtype=torch.float32)
        )
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.int64))
        self.weight = nn.Parameter(torch.ones(num_features, dtype=torch.float32))
        self.bias = nn.Parameter(torch.zeros(num_features, dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        assert x.dim() >= 2
        view_dims = [1, x.shape[1]] + [1] * (x.dim() - 2)
        # _v = lambda v: v.view(view_dims)

        def _v(v):
            return v.view(view_dims)

        running_std = (self.running_var + self.eps).sqrt_()

        if self.training:
            reduce_dims = [i for i in range(x.dim()) if i != 1]
            b_mean = x.mean(reduce_dims)
            b_var = x.var(reduce_dims, unbiased=False)
            b_std = (b_var + self.eps).sqrt_()

            r = torch.clamp((b_std.detach() / running_std), 1 / self.max_r, self.max_r)
            d = torch.clamp(
                (b_mean.detach() - self.running_mean) / running_std,
                -self.max_d,
                self.max_d,
            )

            # Compute warmup factor (0 during warmup, 1 after warmup)
            warmup_factor = torch.clamp(
                self.num_batches_tracked / self.warmup_steps, 0.0, 1.0
            )
            r = 1.0 + (r - 1.0) * warmup_factor
            d = d * warmup_factor

            x = (x - _v(b_mean)) / _v(b_std) * _v(r) + _v(d)

            unbiased_var = b_var.detach() * x.shape[1] / (x.shape[1] - 1)
            self.running_var += self.momentum * (unbiased_var - self.running_var)
            self.running_mean += self.momentum * (b_mean.detach() - self.running_mean)
            self.num_batches_tracked += 1
        else:
            x = (x - _v(self.running_mean)) / _v(running_std)

        x = _v(self.weight) * x + _v(self.bias)
        return x
