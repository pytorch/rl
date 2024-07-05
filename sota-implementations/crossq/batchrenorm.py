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
        r_max (float, optional): Maximum value for the scaling factor r. Default is 3.0.
        d_max (float, optional): Maximum value for the bias factor d. Default is 5.0.
        warmup_steps (int, optional): Number of warm-up steps for the running mean and variance. Default is 5000.
    """

    def __init__(
        self,
        num_features,
        momentum=0.99,
        epsilon=1e-5,
        max_r=3.0,
        max_d=5.0,
        warmup_steps=10000,
    ):
        super(BatchRenorm, self).__init__()
        self.num_features = num_features
        self.epsilon = epsilon
        self.momentum = momentum
        self.max_r = max_r
        self.max_d = max_d
        self.warmup_steps = warmup_steps

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))

        self.register_buffer(
            "num_batches_tracked", torch.tensor(0, dtype=torch.float32)
        )

    def forward(self, x):
        if x.dim() not in [2, 3]:
            raise ValueError("BatchRenorm expects 2D or 3D inputs")

        if x.dim() == 3:
            batch_size, seq_len, _ = x.size()
            x = x.reshape(batch_size * seq_len, self.num_features)

        if self.training:
            self.num_batches_tracked += 1

            batch_mean = x.mean(dim=0)
            batch_var = x.var(dim=0, unbiased=False)

            # Compute r and d factors
            r = torch.clamp(
                (batch_var.sqrt() / (self.running_var.sqrt() + self.epsilon)),
                1 / self.max_r,
                self.max_r,
            )
            d = torch.clamp(
                (
                    (batch_mean - self.running_mean)
                    / (self.running_var.sqrt() + self.epsilon)
                ),
                -self.max_d,
                self.max_d,
            )

            # Compute warmup factor (0 during warmup, 1 after warmup)
            warmup_factor = torch.clamp(
                self.num_batches_tracked / self.warmup_steps, 0.0, 1.0
            )

            # Interpolate between batch norm and renorm based on warmup factor
            effective_r = 1.0 + (r - 1.0) * warmup_factor
            effective_d = d * warmup_factor

            x_hat = (x - batch_mean[None, :]) * effective_r[None, :] + effective_d[
                None, :
            ]
            x_hat = x_hat / (batch_var[None, :] + self.epsilon).sqrt()

            # Update running statistics using Flax-style momentum
            self.running_mean = (
                self.momentum * self.running_mean + (1 - self.momentum) * batch_mean
            )
            self.running_var = (
                self.momentum * self.running_var + (1 - self.momentum) * batch_var
            )

        else:
            x_hat = (x - self.running_mean[None, :]) / (
                self.running_var[None, :] + self.epsilon
            ).sqrt()

        output = self.weight[None, :] * x_hat + self.bias[None, :]

        if x.dim() == 3:
            output = output.reshape(batch_size, seq_len, self.num_features)

        return output
