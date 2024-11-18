# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import functools
import math
import warnings
from typing import List, Optional, Sequence, Union

import torch

from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey
from torch import distributions as d, nn
from torch.nn import functional as F
from torch.nn.modules.dropout import _DropoutNd
from torch.nn.modules.lazy import LazyModuleMixin
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter
from torchrl._utils import prod
from torchrl.data.tensor_specs import Unbounded
from torchrl.data.utils import DEVICE_TYPING, DEVICE_TYPING_ARGS
from torchrl.envs.utils import exploration_type, ExplorationType
from torchrl.modules.distributions.utils import _cast_transform_device
from torchrl.modules.utils import inv_softplus


class NoisyLinear(nn.Linear):
    """Noisy Linear Layer.

    Presented in "Noisy Networks for Exploration", https://arxiv.org/abs/1706.10295v3

    A Noisy Linear Layer is a linear layer with parametric noise added to the weights. This induced stochasticity can
    be used in RL networks for the agent's policy to aid efficient exploration. The parameters of the noise are learned
    with gradient descent along with any other remaining network weights. Factorized Gaussian
    noise is the type of noise usually employed.


    Args:
        in_features (int): input features dimension
        out_features (int): out features dimension
        bias (bool, optional): if ``True``, a bias term will be added to the matrix multiplication: Ax + b.
            Defaults to ``True``
        device (DEVICE_TYPING, optional): device of the layer.
            Defaults to ``"cpu"``
        dtype (torch.dtype, optional): dtype of the parameters.
            Defaults to ``None`` (default pytorch dtype)
        std_init (scalar, optional): initial value of the Gaussian standard deviation before optimization.
            Defaults to ``0.1``

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
    """Noisy Lazy Linear Layer.

    This class makes the Noisy Linear layer lazy, in that the in_feature argument does not need to be passed at
    initialization (but is inferred after the first call to the layer).

    For more context on noisy layers, see the NoisyLinear class.

    Args:
        out_features (int): out features dimension
        bias (bool, optional): if ``True``, a bias term will be added to the matrix multiplication: Ax + b.
            Defaults to ``True``.
        device (DEVICE_TYPING, optional): device of the layer.
            Defaults to ``"cpu"``.
        dtype (torch.dtype, optional): dtype of the parameters.
            Defaults to the default PyTorch dtype.
        std_init (scalar): initial value of the Gaussian standard deviation before optimization.
            Defaults to 0.1

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
    """Resets the noise of noisy layers."""
    if hasattr(layer, "reset_noise"):
        layer.reset_noise()


class gSDEModule(nn.Module):
    """A gSDE exploration module.

     Presented in "Smooth Exploration for Robotic Reinforcement Learning" by Antonin Raffin, Jens Kober, Freek Stulp (https://arxiv.org/abs/2005.05719)

    gSDEModule adds a state-dependent exploration noise to an input action.
    It also outputs the mean, scale (standard deviation) of the normal
    distribution, as well as the Gaussian noise used.

    The noise input should be reset through a :obj:`torchrl.envs.transforms.gSDENoise`
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
        sigma_init (:obj:`float`, optional): the initial value of the standard deviation. The
            softplus non-linearity is used to map the log_sigma parameter to a
            positive value. Defaults to ``1.0``.
        scale_min (:obj:`float`, optional): min value of the scale. Defaults to ``0.01``.
        scale_max (:obj:`float`, optional): max value of the scale. Defaults to ``10.0``.
        learn_sigma (bool, optional): if ``True``, the value of the ``sigma``
            variable will be included in the module parameters, making it learnable.
            Defaults to ``True``.
        transform (torch.distribution.Transform, optional): a transform to apply
            to the sampled action. Defaults to ``None`` (no transform).
        device (torch.device, optional): device to create the model on.
            Defaults to ``"cpu"``.

    Examples:
        >>> from tensordict import TensorDict
        >>> from torchrl.modules import ProbabilisticActor, TanhNormal
        >>> from tensordict.nn import TensorDictModule, ProbabilisticTensorDictSequential
        >>> batch, state_dim, action_dim = 3, 7, 5
        >>> model = nn.Linear(state_dim, action_dim)
        >>> deterministic_policy = TensorDictModule(model, in_keys=["obs"], out_keys=["action"])
        >>> stochastic_part = TensorDictModule(
        ...     gSDEModule(action_dim, state_dim),
        ...     in_keys=["action", "obs", "_eps_gSDE"],
        ...     out_keys=["loc", "scale", "action", "_eps_gSDE"])
        >>> stochastic_part = ProbabilisticActor(stochastic_part,
        ...      in_keys=["loc", "scale"],
        ...      distribution_class=TanhNormal)
        >>> stochastic_policy = ProbabilisticTensorDictSequential(deterministic_policy, *stochastic_part)
        >>> tensordict = TensorDict({'obs': torch.randn(state_dim), '_epx_gSDE': torch.zeros(1)}, [])
        >>> _ = stochastic_policy(tensordict)
        >>> print(tensordict)
        TensorDict(
            fields={
                _eps_gSDE: Tensor(shape=torch.Size([5, 7]), device=cpu, dtype=torch.float32, is_shared=False),
                _epx_gSDE: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
                action: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.float32, is_shared=False),
                loc: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.float32, is_shared=False),
                obs: Tensor(shape=torch.Size([7]), device=cpu, dtype=torch.float32, is_shared=False),
                scale: Tensor(shape=torch.Size([5]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> action_first_call = tensordict.get("action").clone()
        >>> dist = stochastic_policy.get_dist(tensordict)
        >>> print(dist)
        TanhNormal(loc: torch.Size([5]), scale: torch.Size([5]))
        >>> _ = stochastic_policy(tensordict)
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
            self.register_buffer(
                "sigma_init", torch.as_tensor(sigma_init, device=device)
            )

    @property
    def sigma(self):
        if self.learn_sigma:
            sigma = torch.nn.functional.softplus(self.log_sigma)
            return sigma.clamp_min(self.scale_min)
        else:
            return self._sigma.clamp_min(self.scale_min)

    def forward(self, mu, state, _eps_gSDE):
        sigma = self.sigma.clamp_max(self.scale_max)
        _err_explo = f"gSDE behavior for exploration mode {exploration_type()} is not defined. Choose from 'random' or 'mode'."

        if state.shape[:-1] != mu.shape[:-1]:
            _err_msg = f"mu and state are expected to have matching batch size, got shapes {mu.shape} and {state.shape}"
            raise RuntimeError(_err_msg)
        if _eps_gSDE is not None and (
            _eps_gSDE.shape[: state.ndimension() - 1] != state.shape[:-1]
        ):
            _err_msg = f"noise and state are expected to have matching batch size, got shapes {_eps_gSDE.shape} and {state.shape}"
            raise RuntimeError(_err_msg)

        if _eps_gSDE is None and exploration_type() != ExplorationType.RANDOM:
            # noise is irrelevant in with no exploration
            _eps_gSDE = torch.zeros(
                *state.shape[:-1], *sigma.shape, device=sigma.device, dtype=sigma.dtype
            )
        elif (_eps_gSDE is None and exploration_type() == ExplorationType.RANDOM) or (
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

        if exploration_type() in (ExplorationType.RANDOM,):
            action = mu + eps
        elif exploration_type() in (
            ExplorationType.MODE,
            ExplorationType.MEAN,
            ExplorationType.DETERMINISTIC,
        ):
            action = mu
        else:
            raise RuntimeError(_err_explo)

        sigma = (sigma * state.unsqueeze(-2)).pow(2).sum(-1).clamp_min(1e-5).sqrt()
        if not torch.isfinite(sigma).all():
            warnings.warn("inf sigma")

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
    sigma value is initialized such that the resulting variance will match ``sigma_init``
    (or 1 if no ``sigma_init`` value is provided).

    Args:
        sigma_init (:obj:`float`, optional): the initial value of the standard deviation. The
            softplus non-linearity is used to map the log_sigma parameter to a
            positive value. Defaults to ``None`` (learned).
        scale_min (:obj:`float`, optional): min value of the scale. Defaults to ``0.01``.
        scale_max (:obj:`float`, optional): max value of the scale. Defaults to ``10.0``.
        learn_sigma (bool, optional): if ``True``, the value of the ``sigma``
            variable will be included in the module parameters, making it learnable.
            Defaults to ``True``.
        transform (torch.distribution.Transform, optional): a transform to apply
            to the sampled action. Defaults to ``None`` (no transform).
        device (torch.device, optional): device to create the model on.
            Defaults to ``"cpu"``.

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


class ConsistentDropout(_DropoutNd):
    """Implements a :class:`~torch.nn.Dropout` variant with consistent dropout.

    This method is proposed in `"Consistent Dropout for Policy Gradient Reinforcement Learning" (Hausknecht & Wagener, 2022)
    <https://arxiv.org/abs/2202.11818>`_.

    This :class:`~torch.nn.Dropout` variant attempts to increase training stability and
    reduce update variance by caching the dropout masks used during rollout
    and reusing them during the update phase.

    The class you are looking at is independent of the rest of TorchRL's API and does not require tensordict to be run.
    :class:`~torchrl.modules.ConsistentDropoutModule` is a wrapper around ``ConsistentDropout`` that capitalizes on the extensibility
    of ``TensorDict``s by storing generated dropout masks in the transition ``TensorDict`` themselves.
    See this class for a detailed explanation as well as usage examples.

    There is otherwise little conceptual deviance from the PyTorch
    :class:`~torch.nn.Dropout` implementation.

    ..note:: TorchRL's data collectors perform rollouts in :meth:`~torch.no_grad` mode but not in `eval` mode,
        so the dropout masks will be applied unless the policy passed to the collector is in eval mode.

    .. note:: Unlike other exploration modules, :class:`~torchrl.modules.ConsistentDropoutModule`
      uses the ``train``/``eval`` mode to comply with the regular `Dropout` API in PyTorch.
      The :func:`~torchrl.envs.utils.set_exploration_type` context manager will have no effect on
      this module.

    Args:
       p (:obj:`float`, optional): Dropout probability. Defaults to ``0.5``.

    .. seealso::

      - :class:`~torchrl.collectors.SyncDataCollector`:
        :meth:`~torchrl.collectors.SyncDataCollector.rollout()` and :meth:`~torchrl.collectors.SyncDataCollector.iterator()`
      - :class:`~torchrl.collectors.MultiSyncDataCollector`:
        Uses :meth:`~torchrl.collectors.collectors._main_async_collector` (:class:`~torchrl.collectors.SyncDataCollector`)
        under the hood
      - :class:`~torchrl.collectors.MultiaSyncDataCollector`, :class:`~torchrl.collectors.aSyncDataCollector`: Ditto.

    """

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """During training (rollouts & updates), this call masks a tensor full of ones before multiplying with the input tensor.

        During evaluation, this call results in a no-op and only the input is returned.

        Args:
            x (torch.Tensor): the input tensor.
            mask (torch.Tensor, optional): the optional mask for the dropout.

        Returns: a tensor and a corresponding mask in train mode, and only a tensor in eval mode.
        """
        if self.training:
            if mask is None:
                mask = self.make_mask(input=x)
            return x * mask, mask

        return x

    def make_mask(self, *, input=None, shape=None):
        if input is not None:
            return F.dropout(
                torch.ones_like(input), self.p, self.training, inplace=False
            )
        elif shape is not None:
            return F.dropout(torch.ones(shape), self.p, self.training, inplace=False)
        else:
            raise RuntimeError("input or shape must be passed to make_mask.")


class ConsistentDropoutModule(TensorDictModuleBase):
    """A TensorDictModule wrapper for :class:`~ConsistentDropout`.

    Args:
        p (:obj:`float`, optional): Dropout probability. Default: ``0.5``.
        in_keys (NestedKey or list of NestedKeys): keys to be read
            from input tensordict and passed to this module.
        out_keys (NestedKey or iterable of NestedKeys): keys to be written to the input tensordict.
            Defaults to ``in_keys`` values.

    Keyword Args:
        input_shape (tuple, optional): the shape of the input (non-batchted), used to generate the
            tensordict primers with :meth:`~.make_tensordict_primer`.
        input_dtype (torch.dtype, optional): the dtype of the input for the primer. If none is pased,
            ``torch.get_default_dtype`` is assumed.

    .. note:: To use this class within a policy, one needs the mask to be reset at reset time.
      This can be achieved through a :class:`~torchrl.envs.TensorDictPrimer` transform that can be obtained
      with :meth:`~.make_tensordict_primer`. See this method for more information.

    Examples:
        >>> from tensordict import TensorDict
        >>> module = ConsistentDropoutModule(p = 0.1)
        >>> td = TensorDict({"x": torch.randn(3, 4)}, [3])
        >>> module(td)
        TensorDict(
            fields={
                mask_6127171760: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.bool, is_shared=False),
                x: Tensor(shape=torch.Size([3, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
    """

    def __init__(
        self,
        p: float,
        in_keys: NestedKey | List[NestedKey],
        out_keys: NestedKey | List[NestedKey] | None = None,
        input_shape: torch.Size = None,
        input_dtype: torch.dtype | None = None,
    ):
        if isinstance(in_keys, NestedKey):
            in_keys = [in_keys, f"mask_{id(self)}"]
        if out_keys is None:
            out_keys = list(in_keys)
        if isinstance(out_keys, NestedKey):
            out_keys = [out_keys, f"mask_{id(self)}"]
        if len(in_keys) != 2 or len(out_keys) != 2:
            raise ValueError(
                "in_keys and out_keys length must be 2 for consistent dropout."
            )
        self.in_keys = in_keys
        self.out_keys = out_keys
        self.input_shape = input_shape
        self.input_dtype = input_dtype
        super().__init__()

        if not 0 <= p < 1:
            raise ValueError(f"p must be in [0,1), got p={p: 4.4f}.")

        self.consistent_dropout = ConsistentDropout(p)

    def forward(self, tensordict):
        x = tensordict.get(self.in_keys[0])
        mask = tensordict.get(self.in_keys[1], default=None)
        if self.consistent_dropout.training:
            x, mask = self.consistent_dropout(x, mask=mask)
            tensordict.set(self.out_keys[0], x)
            tensordict.set(self.out_keys[1], mask)
        else:
            x = self.consistent_dropout(x, mask=mask)
            tensordict.set(self.out_keys[0], x)

        return tensordict

    def make_tensordict_primer(self):
        """Makes a tensordict primer for the environment to generate random masks during reset calls.

        .. seealso:: :func:`torchrl.modules.utils.get_primers_from_module` for a method to generate all primers for a given
        module.

        Examples:
            >>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
            >>> from torchrl.envs import GymEnv, StepCounter, SerialEnv
            >>> m = Seq(
            ...     Mod(torch.nn.Linear(7, 4), in_keys=["observation"], out_keys=["intermediate"]),
            ...     ConsistentDropoutModule(
            ...         p=0.5,
            ...         input_shape=(2, 4),
            ...         in_keys="intermediate",
            ...     ),
            ...     Mod(torch.nn.Linear(4, 7), in_keys=["intermediate"], out_keys=["action"]),
            ... )
            >>> primer = get_primers_from_module(m)
            >>> env0 = GymEnv("Pendulum-v1").append_transform(StepCounter(5))
            >>> env1 = GymEnv("Pendulum-v1").append_transform(StepCounter(6))
            >>> env = SerialEnv(2, [lambda env=env0: env, lambda env=env1: env])
            >>> env = env.append_transform(primer)
            >>> r = env.rollout(10, m, break_when_any_done=False)
            >>> mask = [k for k in r.keys() if k.startswith("mask")][0]
            >>> assert (r[mask][0, :5] != r[mask][0, 5:6]).any()
            >>> assert (r[mask][0, :4] == r[mask][0, 4:5]).all()

        """
        from torchrl.envs.transforms.transforms import TensorDictPrimer

        shape = self.input_shape
        dtype = self.input_dtype
        if dtype is None:
            dtype = torch.get_default_dtype()
        if shape is None:
            raise RuntimeError(
                "Cannot infer the shape of the input automatically. "
                "Please pass the shape of the tensor to `ConstistentDropoutModule` during construction "
                "with the `input_shape` kwarg."
            )
        return TensorDictPrimer(
            primers={self.in_keys[1]: Unbounded(dtype=dtype, shape=shape)},
            default_value=functools.partial(
                self.consistent_dropout.make_mask, shape=shape
            ),
        )
