# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Sequence, Union

import torch
from torch import nn
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

__all__ = ["NoisyLinear", "NoisyLazyLinear", "reset_noise"]

from torchrl.data.utils import DEVICE_TYPING
from torchrl.modules.utils import inv_softplus


class NoisyLinear(nn.Linear):
    """
    Noisy Linear Layer, as presented in "Noisy Networks for Exploration", https://arxiv.org/abs/1706.10295v3

    A Noisy Linear Layer is a linear layer with parametric noise added to the weights. This induced stochasticity can
    be used in RL networks for the agent's policy to aid efficient exploration. The parameters of the noise are learned
    with gradient descent along with any other remaining network weights. Factorized Gaussian
    noise is the type of noise usually employed.


    Args:
        in_features (int): input features dimension
        out_features (int): out features dimension
        bias (bool): if True, a bias term will be added to the matrix multiplication: Ax + b.
            default: True
        device (str, int or torch.device, optional): device of the layer.
            default: "cpu"
        dtype (torch.dtype, optional): dtype of the parameters.
            default: None
        std_init (scalar): initial value of the Gaussian standard deviation before optimization.
            default: 1.0
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[torch.dtype] = None,
        std_init: float = 0.1,
    ):
        nn.Module.__init__(self)
        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.std_init = std_init

        self.weight_mu = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.weight_sigma = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
                requires_grad=True,
            )
        )
        self.register_buffer(
            "weight_epsilon",
            torch.empty(out_features, in_features, device=device, dtype=dtype),
        )
        if bias:
            self.bias_mu = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.bias_sigma = nn.Parameter(
                torch.empty(
                    out_features,
                    device=device,
                    dtype=dtype,
                    requires_grad=True,
                )
            )
            self.register_buffer(
                "bias_epsilon",
                torch.empty(out_features, device=device, dtype=dtype),
            )
        else:
            self.bias_mu = None
        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self) -> None:
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / math.sqrt(self.in_features))
        if self.bias_mu is not None:
            self.bias_mu.data.uniform_(-mu_range, mu_range)
            self.bias_sigma.data.fill_(self.std_init / math.sqrt(self.out_features))

    def reset_noise(self) -> None:
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.outer(epsilon_in))
        if self.bias_mu is not None:
            self.bias_epsilon.copy_(epsilon_out)

    def _scale_noise(self, size: Union[int, torch.Size, Sequence]) -> torch.Tensor:
        if isinstance(size, int):
            size = (size,)
        x = torch.randn(*size, device=self.weight_mu.device)
        return x.sign().mul_(x.abs().sqrt_())

    @property
    def weight(self) -> torch.Tensor:
        if self.training:
            return self.weight_mu + self.weight_sigma * self.weight_epsilon
        else:
            return self.weight_mu

    @property
    def bias(self) -> Optional[torch.Tensor]:
        if self.bias_mu is not None:
            if self.training:
                return self.bias_mu + self.bias_sigma * self.bias_epsilon
            else:
                return self.bias_mu
        else:
            return None


class NoisyLazyLinear(LazyModuleMixin, NoisyLinear):
    """
    Noisy Lazy Linear Layer.

    This class makes the Noisy Linear layer lazy, in that the in_feature argument does not need to be passed at
    initialization (but is inferred after the first call to the layer).

    For more context on noisy layers, see the NoisyLinear class.

    Args:
        out_features (int): out features dimension
        bias (bool): if True, a bias term will be added to the matrix multiplication: Ax + b.
            default: True
        device (str, int or torch.device, optional): device of the layer.
            default: "cpu"
        dtype (torch.dtype, optional): dtype of the parameters.
            default: None
        std_init (scalar): initial value of the Gaussian standard deviation before optimization.
            default: 1.0
    """

    def __init__(
        self,
        out_features: int,
        bias: bool = True,
        device: Optional[DEVICE_TYPING] = None,
        dtype: Optional[torch.dtype] = None,
        std_init: float = 0.1,
    ):
        super().__init__(0, 0, False)
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = UninitializedParameter(
            device=device, dtype=dtype
        )
        self.weight_sigma = UninitializedParameter(
            device=device, dtype=dtype
        )
        self.register_buffer(
            "weight_epsilon",
            UninitializedBuffer(device=device, dtype=dtype)
          
        )
        if bias:
            self.bias_mu = UninitializedParameter(
                device=device, dtype=dtype
            )
            self.bias_sigma = UninitializedParameter(
                device=device, dtype=dtype
            )
            self.register_buffer(
                "bias_epsilon",
                UninitializedBuffer(device=device, dtype=dtype)
              
            )
        else:
            self.bias_mu = None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_parameters()

    def reset_noise(self) -> None:
        if not self.has_uninitialized_params() and self.in_features != 0:
            super().reset_noise()

    def initialize_parameters(
        self, input: torch.Tensor
    ) -> None:[override]
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.weight_mu.materialize(
                    (self.out_features, self.in_features)
                )
                self.weight_sigma.materialize(
                    (self.out_features, self.in_features)
                )
                self.weight_epsilon.materialize(
                    (self.out_features, self.in_features)
                )
                if self.bias_mu is not None:
                    self.bias_mu.materialize((self.out_features,))
                    self.bias_sigma.materialize((self.out_features,))
                    self.bias_epsilon.materialize((self.out_features,))
                self.reset_parameters()
                self.reset_noise()

    @property
    def weight(self) -> torch.Tensor:
        if not self.has_uninitialized_params() and self.in_features != 0:
            return super().weight

    @property
    def bias(self) -> torch.Tensor:
        if not self.has_uninitialized_params() and self.in_features != 0:
            return super().bias


def reset_noise(layer: nn.Module) -> None:
    if hasattr(layer, "reset_noise"):
        layer.reset_noise()


class gSDEWrapper(nn.Module):
    """A gSDE exploration wrapper as presented in "Smooth Exploration for
    Robotic Reinforcement Learning" by Antonin Raffin, Jens Kober,
    Freek Stulp (https://arxiv.org/abs/2005.05719)

    gSDEWrapper encapsulates nn.Module that outputs the average of a
    normal distribution and adds a state-dependent exploration noise to it.
    It outputs the mean, scale (standard deviation) of the normal
    distribution as well as the chosen action.

    For now, only vector states are considered, but the distribution can
    read other inputs (e.g. hidden states etc.)

    When used, the gSDEWrapper should also be accompanied by a few
    configuration changes: the exploration mode of the policy should be set
    to "net_output", meaning that the action from the ProbabilisticTDModule
    will be retrieved directly from the network output and not simulated
    from the constructed distribution. Second, the noise input should be
    created through a `torchrl.envs.transforms.gSDENoise` instance,
    which will reset this noise parameter each time the environment is reset.
    Finally, a regular normal distribution should be used to sample the
    actions, the `ProbabilisticTDModule` should be created
    in safe mode (in order for the action to be clipped in the desired
    range) and its input keys should include `"_eps_gSDE"` which is the
    default gSDE noise key:

        >>> actor = ProbabilisticActor(
        ...     wrapped_module,
        ...     in_keys=["observation", "_eps_gSDE"]
        ...     spec,
        ...     distribution_class=IndependentNormal,
        ...     safe=True)

    Args:
        policy_model (nn.Module): a model that reads observations and
            outputs a distribution average.
        action_dim (int): the dimension of the action.
        state_dim (int): the state dimension.
        sigma_init (float): the initial value of the standard deviation. The
            softplus non-linearity is used to map the log_sigma parameter to a
            positive value.

    Examples:
        >>> batch, state_dim, action_dim = 3, 7, 5
        >>> model = nn.Linear(state_dim, action_dim)
        >>> wrapped_model = gSDEWrapper(model, action_dim=action_dim,
        ...     state_dim=state_dim)
        >>> state = torch.randn(batch, state_dim)
        >>> eps_gSDE = torch.randn(batch, action_dim, state_dim)
        >>> # the module takes inputs (state, *additional_vectors, noise_param)
        >>> mu, sigma, action = wrapped_model(state, eps_gSDE)
        >>> print(mu.shape, sigma.shape, action.shape)
        torch.Size([3, 5]) torch.Size([3, 5]) torch.Size([3, 5])
    """

    def __init__(
        self,
        policy_model: nn.Module,
        action_dim: int,
        state_dim: int,
        sigma_init: float = None,
    ) -> None:
        super().__init__()
        self.policy_model = policy_model
        self.action_dim = action_dim
        self.state_dim = state_dim
        if sigma_init is None:
            sigma_init = inv_softplus(math.sqrt(1 / state_dim))
        self.register_parameter(
            "log_sigma",
            nn.Parameter(torch.zeros((action_dim, state_dim), requires_grad=True)),
        )
        self.register_buffer("sigma_init", torch.tensor(sigma_init))

    def forward(self, state, *tensors):
        *tensors, gSDE_noise = tensors
        sigma = torch.nn.functional.softplus(self.log_sigma + self.sigma_init)
        if gSDE_noise is None:
            gSDE_noise = torch.randn_like(sigma)
        gSDE_noise = sigma * gSDE_noise
        eps = (gSDE_noise @ state.unsqueeze(-1)).squeeze(-1)
        mu = self.policy_model(state, *tensors)
        action = mu + eps
        sigma = (sigma * state.unsqueeze(-2)).pow(2).sum(-1).clamp_min(1e-5).sqrt()
        if not torch.isfinite(sigma).all():
            print("inf sigma")
        return mu, sigma, action
