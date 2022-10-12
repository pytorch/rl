# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math
from typing import Optional, Sequence, Union

import torch
from torch import nn, distributions as d
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

__all__ = ["NoisyLinear", "NoisyLazyLinear", "reset_noise"]

from torchrl._utils import prod
from torchrl.data.utils import DEVICE_TYPING, DEVICE_TYPING_ARGS
from torchrl.envs.utils import exploration_mode
from torchrl.modules.distributions.utils import _cast_transform_device
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
        device (DEVICE_TYPING, optional): device of the layer.
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
        device (DEVICE_TYPING, optional): device of the layer.
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
        super().__init__(0, 0, False, device=device)
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = UninitializedParameter(device=device, dtype=dtype)
        self.weight_sigma = UninitializedParameter(device=device, dtype=dtype)
        self.register_buffer(
            "weight_epsilon", UninitializedBuffer(device=device, dtype=dtype)
        )
        if bias:
            self.bias_mu = UninitializedParameter(device=device, dtype=dtype)
            self.bias_sigma = UninitializedParameter(device=device, dtype=dtype)
            self.register_buffer(
                "bias_epsilon", UninitializedBuffer(device=device, dtype=dtype)
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

    def initialize_parameters(self, input: torch.Tensor) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.in_features = input.shape[-1]
                self.weight_mu.materialize((self.out_features, self.in_features))
                self.weight_sigma.materialize((self.out_features, self.in_features))
                self.weight_epsilon.materialize((self.out_features, self.in_features))
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


class gSDEModule(nn.Module):
    """A gSDE exploration module as presented in "Smooth Exploration for
    Robotic Reinforcement Learning" by Antonin Raffin, Jens Kober,
    Freek Stulp (https://arxiv.org/abs/2005.05719)

    gSDEModule adds a state-dependent exploration noise to an input action.
    It also outputs the mean, scale (standard deviation) of the normal
    distribution, as well as the Gaussian noise used.

    The noise input should be reset through a `torchrl.envs.transforms.gSDENoise`
    instance: each time the environment is reset, the input noise will be set to
    zero by the environment transform, indicating to gSDEModule that it has to be resampled.
    This scheme allows us to have the environemt tell the module to resample a
    noise only the latter knows the shape of.

    A variable transform function can also be provided to map the noicy action
    to the desired space (e.g. a SafeTanhTransform or similar).

    Args:
        policy_model (nn.Module): a model that reads observations and
            outputs a distribution average.
        action_dim (int): the dimension of the action.
        state_dim (int): the state dimension.
        sigma_init (float): the initial value of the standard deviation. The
            softplus non-linearity is used to map the log_sigma parameter to a
            positive value.
        scale_min (float, optional): min value of the scale.
        scale_max (float, optional): max value of the scale.
        transform (torch.distribution.Transform, optional): a transform to apply
            to the sampled action.
        device (DEVICE_TYPING, optional): device to create the model on.

    Examples:
        >>> from torchrl.modules import TensorDictModule, TensorDictSequential, ProbabilisticActor, TanhNormal
        >>> from torchrl.data import TensorDict
        >>> batch, state_dim, action_dim = 3, 7, 5
        >>> model = nn.Linear(state_dim, action_dim)
        >>> deterministic_policy = TensorDictModule(model, in_keys=["obs"], out_keys=["action"])
        >>> stochatstic_part = TensorDictModule(
        ...     gSDEModule(action_dim, state_dim),
        ...     in_keys=["action", "obs", "_eps_gSDE"],
        ...     out_keys=["loc", "scale", "action", "_eps_gSDE"])
        >>> stochatstic_part = ProbabilisticActor(stochatstic_part,
        ...      dist_param_keys=["loc", "scale"],
        ...      distribution_class=TanhNormal)
        >>> stochatstic_policy = TensorDictSequential(deterministic_policy, stochatstic_part)
        >>> tensordict = TensorDict({'obs': torch.randn(state_dim), '_epx_gSDE': torch.zeros(1)}, [])
        >>> _ = stochatstic_policy(tensordict)
        >>> print(tensordict)
        TensorDict(
            fields={
                obs: Tensor(torch.Size([7]), dtype=torch.float32),
                _epx_gSDE: Tensor(torch.Size([1]), dtype=torch.float32),
                action: Tensor(torch.Size([5]), dtype=torch.float32),
                loc: Tensor(torch.Size([5]), dtype=torch.float32),
                scale: Tensor(torch.Size([5]), dtype=torch.float32),
                _eps_gSDE: Tensor(torch.Size([5, 7]), dtype=torch.float32)},
            batch_size=torch.Size([]),
            device=cpu,
            is_shared=False)
        >>> action_first_call = tensordict.get("action").clone()
        >>> dist, *_ = stochatstic_policy.get_dist(tensordict)
        >>> print(dist)
        TanhNormal(loc: torch.Size([5]), scale: torch.Size([5]))
        >>> _ = stochatstic_policy(tensordict)
        >>> action_second_call = tensordict.get("action").clone()
        >>> assert (action_second_call == action_first_call).all()  # actions are the same
        >>> assert (action_first_call != dist.base_dist.base_dist.loc).all()  # actions are truly stochastic
    """

    def __init__(
        self,
        action_dim: int,
        state_dim: int,
        sigma_init: float = None,
        scale_min: float = 0.01,
        scale_max: float = 10.0,
        learn_sigma: bool = True,
        transform: Optional[d.Transform] = None,
        device: Optional[DEVICE_TYPING] = None,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.scale_min = scale_min
        self.scale_max = scale_max
        self.transform = transform
        self.learn_sigma = learn_sigma
        if learn_sigma:
            if sigma_init is None:
                sigma_init = inv_softplus(math.sqrt((1.0 - scale_min) / state_dim))
            self.register_parameter(
                "log_sigma",
                nn.Parameter(
                    torch.zeros(
                        (action_dim, state_dim), requires_grad=True, device=device
                    )
                ),
            )
        else:
            if sigma_init is None:
                sigma_init = math.sqrt((1.0 - scale_min) / state_dim)
            self.register_buffer(
                "_sigma",
                torch.full((action_dim, state_dim), sigma_init, device=device),
            )

        if sigma_init != 0.0:
            self.register_buffer("sigma_init", torch.tensor(sigma_init, device=device))

    @property
    def sigma(self):
        if self.learn_sigma:
            sigma = torch.nn.functional.softplus(self.log_sigma)
            return sigma.clamp_min(self.scale_min)
        else:
            return self._sigma.clamp_min(self.scale_min)

    def forward(self, mu, state, _eps_gSDE):
        sigma = self.sigma.clamp_max(self.scale_max)
        _err_explo = f"gSDE behaviour for exploration mode {exploration_mode()} is not defined. Choose from 'random' or 'mode'."

        if state.shape[:-1] != mu.shape[:-1]:
            _err_msg = f"mu and state are expected to have matching batch size, got shapes {mu.shape} and {state.shape}"
            raise RuntimeError(_err_msg)
        if _eps_gSDE is not None and (
            _eps_gSDE.shape[: state.ndimension() - 1] != state.shape[:-1]
        ):
            _err_msg = f"noise and state are expected to have matching batch size, got shapes {_eps_gSDE.shape} and {state.shape}"
            raise RuntimeError(_err_msg)

        if _eps_gSDE is None and exploration_mode() == "mode":
            # noise is irrelevant in with no exploration
            _eps_gSDE = torch.zeros(
                *state.shape[:-1], *sigma.shape, device=sigma.device, dtype=sigma.dtype
            )
        elif (_eps_gSDE is None and exploration_mode() == "random") or (
            _eps_gSDE is not None
            and _eps_gSDE.numel() == prod(state.shape[:-1])
            and (_eps_gSDE == 0).all()
        ):
            _eps_gSDE = torch.randn(
                *state.shape[:-1], *sigma.shape, device=sigma.device, dtype=sigma.dtype
            )
        elif _eps_gSDE is None:
            raise RuntimeError(_err_explo)

        gSDE_noise = sigma * _eps_gSDE
        eps = (gSDE_noise @ state.unsqueeze(-1)).squeeze(-1)

        if exploration_mode() in ("random",):
            action = mu + eps
        elif exploration_mode() in ("mode",):
            action = mu
        else:
            raise RuntimeError(_err_explo)

        sigma = (sigma * state.unsqueeze(-2)).pow(2).sum(-1).clamp_min(1e-5).sqrt()
        if not torch.isfinite(sigma).all():
            print("inf sigma")

        if self.transform is not None:
            action = self.transform(action)
        return mu, sigma, action, _eps_gSDE

    def to(self, device_or_dtype: Union[torch.dtype, DEVICE_TYPING]):
        if isinstance(device_or_dtype, DEVICE_TYPING_ARGS):
            self.transform = _cast_transform_device(self.transform, device_or_dtype)
        return super().to(device_or_dtype)


class LazygSDEModule(LazyModuleMixin, gSDEModule):
    """Lazy gSDE Module.
    This module behaves exactly as gSDEModule except that it does not require the
    user to specify the action and state dimension.
    If the input state is multi-dimensional (i.e. more than one state is provided), the
    sigma value is initialized such that the resulting variance will match `sigma_init`
    (or 1 if no `sigma_init` value is provided).

    """

    cls_to_become = gSDEModule
    log_sigma: UninitializedParameter
    _sigma: UninitializedBuffer
    sigma_init: UninitializedBuffer

    def __init__(
        self,
        sigma_init: float = None,
        scale_min: float = 0.01,
        scale_max: float = 10.0,
        learn_sigma: bool = True,
        transform: Optional[d.Transform] = None,
        device: Optional[DEVICE_TYPING] = None,
    ) -> None:
        super().__init__(
            0,
            0,
            sigma_init=0.0,
            scale_min=scale_min,
            scale_max=scale_max,
            learn_sigma=learn_sigma,
            transform=transform,
            device=device,
        )
        factory_kwargs = {
            "device": device,
            "dtype": torch.get_default_dtype(),
        }
        self._sigma_init = sigma_init
        self.sigma_init = UninitializedBuffer(**factory_kwargs)
        if learn_sigma:
            self.log_sigma = UninitializedParameter(**factory_kwargs)
        else:
            self._sigma = UninitializedBuffer(**factory_kwargs)

    def reset_parameters(self) -> None:
        pass

    def initialize_parameters(
        self, mu: torch.Tensor, state: torch.Tensor, _eps_gSDE: torch.Tensor
    ) -> None:
        if self.has_uninitialized_params():
            action_dim = mu.shape[-1]
            state_dim = state.shape[-1]
            with torch.no_grad():
                if state.ndimension() > 2:
                    state = state.flatten(0, -2).squeeze(0)
                if state.ndimension() == 1:
                    state_flatten_var = torch.ones(1, device=mu.device)
                else:
                    state_flatten_var = state.pow(2).mean(dim=0).reciprocal()

                self.sigma_init.materialize(state_flatten_var.shape)
                if self.learn_sigma:
                    if self._sigma_init is None:
                        state_flatten_var.clamp_min_(self.scale_min)
                        self.sigma_init.data.copy_(
                            inv_softplus((state_flatten_var / state_dim).sqrt())
                        )
                    else:
                        self.sigma_init.data.copy_(
                            inv_softplus(
                                self._sigma_init
                                * (state_flatten_var / state_dim).sqrt()
                            )
                        )

                    self.log_sigma.materialize((action_dim, state_dim))
                    self.log_sigma.data.copy_(self.sigma_init.expand_as(self.log_sigma))

                else:
                    if self._sigma_init is None:
                        self.sigma_init.data.copy_(
                            (state_flatten_var / state_dim).sqrt()
                        )
                    else:
                        self.sigma_init.data.copy_(
                            (state_flatten_var / state_dim).sqrt() * self._sigma_init
                        )
                    self._sigma.materialize((action_dim, state_dim))
                    self._sigma.data.copy_(self.sigma_init.expand_as(self._sigma))
