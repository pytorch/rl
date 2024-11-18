# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import torch
import torch.nn as nn


class BatchRenorm1d(nn.Module):
    """BatchRenorm Module (https://arxiv.org/abs/1702.03275).

    The code is adapted from https://github.com/google-research/corenet

    BatchRenorm is an enhanced version of the standard BatchNorm. Unlike BatchNorm,
    it utilizes running statistics to normalize batches after an initial warmup phase.
    This approach reduces the impact of "outlier" batches that may occur during
    extended training periods, making BatchRenorm more robust for long training runs.

    During the warmup phase, BatchRenorm functions identically to a BatchNorm layer.

    Args:
        num_features (int): Number of features in the input tensor.

    Keyword Args:
        momentum (:obj:`float`, optional): Momentum factor for computing the running mean and variance.
            Defaults to ``0.01``.
        eps (:obj:`float`, optional): Small value added to the variance to avoid division by zero.
            Defaults to ``1e-5``.
        max_r (:obj:`float`, optional): Maximum value for the scaling factor r.
            Defaults to ``3.0``.
        max_d (:obj:`float`, optional): Maximum value for the bias factor d.
            Defaults to ``5.0``.
        warmup_steps (int, optional): Number of warm-up steps for the running mean and variance.
            Defaults to ``10000``.
        smooth (bool, optional): if ``True``, the behavior smoothly transitions from regular
            batch-norm (when ``iter=0``) to batch-renorm (when ``iter=warmup_steps``).
            Otherwise, the behavior will transition from batch-norm to batch-renorm when
            ``iter=warmup_steps``. Defaults to ``False``.
    """

    def __init__(
        self,
        num_features: int,
        *,
        momentum: float = 0.01,
        eps: float = 1e-5,
        max_r: float = 3.0,
        max_d: float = 5.0,
        warmup_steps: int = 10000,
        smooth: bool = False,
    ):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.max_r = max_r
        self.max_d = max_d
        self.warmup_steps = warmup_steps
        self.smooth = smooth

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
        if not x.dim() >= 2:
            raise ValueError(
                f"The {type(self).__name__} expects a 2D (or more) tensor, got {x.dim()}."
            )

        view_dims = [1, x.shape[1]] + [1] * (x.dim() - 2)

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
            if self.warmup_steps > 0:
                if self.smooth:
                    warmup_factor = self.num_batches_tracked / self.warmup_steps
                else:
                    warmup_factor = self.num_batches_tracked // self.warmup_steps
                r = 1.0 + (r - 1.0) * warmup_factor
                d = d * warmup_factor

            x = (x - _v(b_mean)) / _v(b_std) * _v(r) + _v(d)

            unbiased_var = b_var.detach() * x.shape[0] / (x.shape[0] - 1)
            self.running_var += self.momentum * (unbiased_var - self.running_var)
            self.running_mean += self.momentum * (b_mean.detach() - self.running_mean)
            self.num_batches_tracked += 1
            self.num_batches_tracked.clamp_max(self.warmup_steps)
        else:
            x = (x - _v(self.running_mean)) / _v(running_std)

        x = _v(self.weight) * x + _v(self.bias)
        return x
