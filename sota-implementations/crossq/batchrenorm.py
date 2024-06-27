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
