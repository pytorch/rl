import torch
import torch.nn as nn
import torch.distributions as d

class LogLikelihood(nn.Module):
    """
    Log Likelihood
    Computes the log likelihood of the observation given the target
    """

    def __init__(self, reduction: str = "mean"):
        super().__init__()
        self.reduction = reduction

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_likelihood =  -d.Normal(input, 1).log_prob(target)

        if self.reduction == "mean":
            return log_likelihood.mean()
        elif self.reduction == "sum":
            return log_likelihood.sum()
        elif self.reduction == "none":
            return log_likelihood
        else:
            raise ValueError(f"Unknown reduction {self.reduction}")