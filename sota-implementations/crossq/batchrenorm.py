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
        eps (float, optional): Small value added to the variance to avoid division by zero. Default is 1e-5.
        momentum (float, optional): Momentum factor for computing the running mean and variance. Default is 0.01.
        r_max (float, optional): Maximum value for the scaling factor r. Default is 3.0.
        d_max (float, optional): Maximum value for the bias factor d. Default is 5.0.
        warmup_steps (int, optional): Number of warm-up steps for the running mean and variance. Default is 5000.
    """

    def __init__(
        self,
        num_features,
        eps=0.01,
        momentum=0.99,
        r_max=3.0,
        d_max=5.0,
        warmup_steps=100000,
    ):

        super(BatchRenorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.r_max = r_max
        self.d_max = d_max
        self.warmup_steps = warmup_steps
        self.step_count = 0

        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))

    def forward(self, x):
        self.step_count += 1

        # Compute the dimensions for mean and variance calculation
        dims = [i for i in range(x.dim()) if i != 1]
        expand_dims = [1 if i != 1 else -1 for i in range(x.dim())]

        # Compute batch statistics
        batch_mean = x.mean(dims, keepdim=True)
        batch_var = x.var(dims, unbiased=False, keepdim=True)

        if self.training:
            if self.step_count <= self.warmup_steps:
                # Use classical BatchNorm during warmup
                x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
            else:
                # Use Batch Renormalization
                with torch.no_grad():
                    r = torch.clamp(
                        batch_var / self.running_var.view(*expand_dims),
                        1.0 / self.r_max,
                        self.r_max,
                    )
                    d = torch.clamp(
                        (batch_mean - self.running_mean.view(*expand_dims))
                        / torch.sqrt(self.running_var.view(*expand_dims) + self.eps),
                        -self.d_max,
                        self.d_max,
                    )

                x_hat = (x - batch_mean) / torch.sqrt(batch_var + self.eps)
                x_hat = x_hat * r + d

            # Update running statistics
            self.running_mean.mul_(1 - self.momentum).add_(
                batch_mean.squeeze().detach() * self.momentum
            )
            self.running_var.mul_(1 - self.momentum).add_(
                batch_var.squeeze().detach() * self.momentum
            )
        else:
            # Use running statistics during inference
            x_hat = (x - self.running_mean.view(*expand_dims)) / torch.sqrt(
                self.running_var.view(*expand_dims) + self.eps
            )

        return self.gamma.view(*expand_dims) * x_hat + self.beta.view(*expand_dims)


import torch.nn as nn


class AdaptiveBatchRenorm(nn.Module):
    def __init__(
        self,
        num_features,
        epsilon=1e-5,
        momentum=0.99,
        max_r=3.0,
        max_d=5.0,
        warmup_steps=10000,
    ):
        super(AdaptiveBatchRenorm, self).__init__()
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
            raise ValueError("AdaptiveBatchRenorm expects 2D or 3D inputs")

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
