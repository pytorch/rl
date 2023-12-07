# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Optional, Union

import numpy as np
import torch

from tensordict.nn import (
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictModuleWrapper,
)
from tensordict.tensordict import TensorDictBase
from tensordict.utils import expand_as_right, expand_right, NestedKey

from torchrl.data.tensor_specs import CompositeSpec, TensorSpec
from torchrl.envs.utils import exploration_type, ExplorationType
from torchrl.modules.tensordict_module.common import _forward_hook_safe_action

__all__ = [
    "EGreedyWrapper",
    "EGreedyModule",
    "AdditiveGaussianWrapper",
    "OrnsteinUhlenbeckProcessWrapper",
]


class EGreedyModule(TensorDictModuleBase):
    """Epsilon-Greedy exploration module.

    This module randomly updates the action(s) in a tensordict given an epsilon greedy exploration strategy.
    At each call, random draws (one per action) are executed given a certain probability threshold. If successful,
    the corresponding actions are being replaced by random samples drawn from the action spec provided.
    Others are left unchanged.

    Args:
        spec (TensorSpec): the spec used for sampling actions.
        eps_init (scalar, optional): initial epsilon value.
            default: 1.0
        eps_end (scalar, optional): final epsilon value.
            default: 0.1
        annealing_num_steps (int, optional): number of steps it will take for epsilon to reach
            the ``eps_end`` value. Defaults to `1000`.

    Keyword Args:
        action_key (NestedKey, optional): the key where the action can be found in the input tensordict.
            Default is ``"action"``.
        action_mask_key (NestedKey, optional): the key where the action mask can be found in the input tensordict.
            Default is ``None`` (corresponding to no mask).

    .. note::
        It is crucial to incorporate a call to :meth:`~.step` in the training loop
        to update the exploration factor.
        Since it is not easy to capture this omission no warning or exception
        will be raised if this is ommitted!

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictSequential
        >>> from torchrl.modules import EGreedyModule, Actor
        >>> from torchrl.data import BoundedTensorSpec
        >>> torch.manual_seed(0)
        >>> spec = BoundedTensorSpec(-1, 1, torch.Size([4]))
        >>> module = torch.nn.Linear(4, 4, bias=False)
        >>> policy = Actor(spec=spec, module=module)
        >>> explorative_policy = TensorDictSequential(policy,  EGreedyModule(eps_init=0.2))
        >>> td = TensorDict({"observation": torch.zeros(10, 4)}, batch_size=[10])
        >>> print(explorative_policy(td).get("action"))
        tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.9055, -0.9277, -0.6295, -0.2532],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000]], grad_fn=<AddBackward0>)

    """

    def __init__(
        self,
        spec: TensorSpec,
        eps_init: float = 1.0,
        eps_end: float = 0.1,
        annealing_num_steps: int = 1000,
        *,
        action_key: Optional[NestedKey] = "action",
        action_mask_key: Optional[NestedKey] = None,
    ):
        if not isinstance(eps_init, float):
            warnings.warn("eps_init should be a float.")
        if eps_end > eps_init:
            raise RuntimeError("eps should decrease over time or be constant")
        self.action_key = action_key
        self.action_mask_key = action_mask_key
        in_keys = [self.action_key]
        if self.action_mask_key is not None:
            in_keys.append(self.action_mask_key)
        self.in_keys = in_keys
        self.out_keys = [self.action_key]

        super().__init__()

        self.register_buffer("eps_init", torch.tensor([eps_init]))
        self.register_buffer("eps_end", torch.tensor([eps_end]))
        self.annealing_num_steps = annealing_num_steps
        self.register_buffer("eps", torch.tensor([eps_init], dtype=torch.float32))

        if spec is not None:
            if not isinstance(spec, CompositeSpec) and len(self.out_keys) >= 1:
                spec = CompositeSpec({action_key: spec}, shape=spec.shape[:-1])
        self._spec = spec

    @property
    def spec(self):
        return self._spec

    def step(self, frames: int = 1) -> None:
        """A step of epsilon decay.

        After `self.annealing_num_steps` calls to this method, calls result in no-op.

        Args:
            frames (int, optional): number of frames since last step. Defaults to ``1``.

        """
        for _ in range(frames):
            self.eps.data[0] = max(
                self.eps_end.item(),
                (
                    self.eps - (self.eps_init - self.eps_end) / self.annealing_num_steps
                ).item(),
            )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            if isinstance(self.action_key, tuple) and len(self.action_key) > 1:
                action_tensordict = tensordict.get(self.action_key[:-1])
                action_key = self.action_key[-1]
            else:
                action_tensordict = tensordict
                action_key = self.action_key

            out = action_tensordict.get(action_key)
            eps = self.eps.item()
            cond = (
                torch.rand(action_tensordict.shape, device=action_tensordict.device)
                < eps
            ).to(out.dtype)
            cond = expand_as_right(cond, out)
            spec = self.spec
            if spec is not None:
                if isinstance(spec, CompositeSpec):
                    spec = spec[self.action_key]
                if spec.shape != out.shape:
                    # In batched envs if the spec is passed unbatched, the rand() will not
                    # cover all batched dims
                    if (
                        not len(spec.shape)
                        or out.shape[-len(spec.shape) :] == spec.shape
                    ):
                        spec = spec.expand(out.shape)
                    else:
                        raise ValueError(
                            "Action spec shape does not match the action shape"
                        )
                if self.action_mask_key is not None:
                    action_mask = tensordict.get(self.action_mask_key, None)
                    if action_mask is None:
                        raise KeyError(
                            f"Action mask key {self.action_mask_key} not found in {tensordict}."
                        )
                    spec.update_mask(action_mask)
                out = cond * spec.rand().to(out.device) + (1 - cond) * out
            else:
                raise RuntimeError("spec must be provided to the exploration wrapper.")
            action_tensordict.set(action_key, out)
        return tensordict


class EGreedyWrapper(TensorDictModuleWrapper):
    """[Deprecated] Epsilon-Greedy PO wrapper.

    Args:
        policy (TensorDictModule): a deterministic policy.

    Keyword Args:
        eps_init (scalar, optional): initial epsilon value.
            default: 1.0
        eps_end (scalar, optional): final epsilon value.
            default: 0.1
        annealing_num_steps (int, optional): number of steps it will take for epsilon to reach the eps_end value
        action_key (NestedKey, optional): the key where the action can be found in the input tensordict.
            Default is ``"action"``.
        action_mask_key (NestedKey, optional): the key where the action mask can be found in the input tensordict.
            Default is ``None`` (corresponding to no mask).
        spec (TensorSpec, optional): if provided, the sampled action will be
            taken from this action space. If not provided,
            the exploration wrapper will attempt to recover it from the policy.

    .. note::
        Once a module has been wrapped in :class:`EGreedyWrapper`, it is
        crucial to incorporate a call to :meth:`~.step` in the training loop
        to update the exploration factor.
        Since it is not easy to capture this omission no warning or exception
        will be raised if this is ommitted!

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.modules import EGreedyWrapper, Actor
        >>> from torchrl.data import BoundedTensorSpec
        >>> torch.manual_seed(0)
        >>> spec = BoundedTensorSpec(-1, 1, torch.Size([4]))
        >>> module = torch.nn.Linear(4, 4, bias=False)
        >>> policy = Actor(spec=spec, module=module)
        >>> explorative_policy = EGreedyWrapper(policy, eps_init=0.2)
        >>> td = TensorDict({"observation": torch.zeros(10, 4)}, batch_size=[10])
        >>> print(explorative_policy(td).get("action"))
        tensor([[ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.9055, -0.9277, -0.6295, -0.2532],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000,  0.0000]], grad_fn=<AddBackward0>)

    """

    def __init__(
        self,
        policy: TensorDictModule,
        *,
        eps_init: float = 1.0,
        eps_end: float = 0.1,
        annealing_num_steps: int = 1000,
        action_key: Optional[NestedKey] = "action",
        action_mask_key: Optional[NestedKey] = None,
        spec: Optional[TensorSpec] = None,
    ):
        warnings.warn(
            "EGreedyWrapper is deprecated and it will be removed in v0.3. "
            "Please use torchrl.modules.EGreedyModule instead.",
            category=DeprecationWarning,
        )

        super().__init__(policy)
        self.register_buffer("eps_init", torch.tensor([eps_init]))
        self.register_buffer("eps_end", torch.tensor([eps_end]))
        if self.eps_end > self.eps_init:
            raise RuntimeError("eps should decrease over time or be constant")
        self.annealing_num_steps = annealing_num_steps
        self.register_buffer("eps", torch.tensor([eps_init], dtype=torch.float32))
        self.action_key = action_key
        self.action_mask_key = action_mask_key
        if spec is not None:
            if not isinstance(spec, CompositeSpec) and len(self.out_keys) >= 1:
                spec = CompositeSpec({action_key: spec}, shape=spec.shape[:-1])
            self._spec = spec
        elif hasattr(self.td_module, "_spec"):
            self._spec = self.td_module._spec.clone()
            if action_key not in self._spec.keys():
                self._spec[action_key] = None
        elif hasattr(self.td_module, "spec"):
            self._spec = self.td_module.spec.clone()
            if action_key not in self._spec.keys():
                self._spec[action_key] = None
        else:
            self._spec = spec

    @property
    def spec(self):
        return self._spec

    def step(self, frames: int = 1) -> None:
        """A step of epsilon decay.

        After self.annealing_num_steps, this function is a no-op.

        Args:
            frames (int): number of frames since last step.

        """
        for _ in range(frames):
            self.eps.data[0] = max(
                self.eps_end.item(),
                (
                    self.eps - (self.eps_init - self.eps_end) / self.annealing_num_steps
                ).item(),
            )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = self.td_module.forward(tensordict)
        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            if isinstance(self.action_key, tuple) and len(self.action_key) > 1:
                action_tensordict = tensordict.get(self.action_key[:-1])
                action_key = self.action_key[-1]
            else:
                action_tensordict = tensordict
                action_key = self.action_key

            out = action_tensordict.get(action_key)
            eps = self.eps.item()
            cond = (
                torch.rand(action_tensordict.shape, device=action_tensordict.device)
                < eps
            ).to(out.dtype)
            cond = expand_as_right(cond, out)
            spec = self.spec
            if spec is not None:
                if isinstance(spec, CompositeSpec):
                    spec = spec[self.action_key]
                if spec.shape != out.shape:
                    # In batched envs if the spec is passed unbatched, the rand() will not
                    # cover all batched dims
                    if (
                        not len(spec.shape)
                        or out.shape[-len(spec.shape) :] == spec.shape
                    ):
                        spec = spec.expand(out.shape)
                    else:
                        raise ValueError(
                            "Action spec shape does not match the action shape"
                        )
                if self.action_mask_key is not None:
                    action_mask = tensordict.get(self.action_mask_key, None)
                    if action_mask is None:
                        raise KeyError(
                            f"Action mask key {self.action_mask_key} not found in {tensordict}."
                        )
                    spec.update_mask(action_mask)
                out = cond * spec.rand().to(out.device) + (1 - cond) * out
            else:
                raise RuntimeError(
                    "spec must be provided by the policy or directly to the exploration wrapper."
                )
            action_tensordict.set(action_key, out)
        return tensordict


class AdditiveGaussianWrapper(TensorDictModuleWrapper):
    """Additive Gaussian PO wrapper.

    Args:
        policy (TensorDictModule): a policy.

    Keyword Args:
        sigma_init (scalar, optional): initial epsilon value.
            default: 1.0
        sigma_end (scalar, optional): final epsilon value.
            default: 0.1
        annealing_num_steps (int, optional): number of steps it will take for
            sigma to reach the :obj:`sigma_end` value.
        mean (float, optional): mean of each output element’s normal distribution.
        std (float, optional): standard deviation of each output element’s normal distribution.
        action_key (NestedKey, optional): if the policy module has more than one output key,
            its output spec will be of type CompositeSpec. One needs to know where to
            find the action spec.
            Default is "action".
        spec (TensorSpec, optional): if provided, the sampled action will be
            projected onto the valid action space once explored. If not provided,
            the exploration wrapper will attempt to recover it from the policy.
        safe (boolean, optional): if False, the TensorSpec can be None. If it
            is set to False but the spec is passed, the projection will still
            happen.
            Default is True.

    .. note::
        Once an environment has been wrapped in :class:`AdditiveGaussianWrapper`, it is
        crucial to incorporate a call to :meth:`~.step` in the training loop
        to update the exploration factor.
        Since it is not easy to capture this omission no warning or exception
        will be raised if this is ommitted!


    """

    def __init__(
        self,
        policy: TensorDictModule,
        *,
        sigma_init: float = 1.0,
        sigma_end: float = 0.1,
        annealing_num_steps: int = 1000,
        mean: float = 0.0,
        std: float = 1.0,
        action_key: Optional[NestedKey] = "action",
        spec: Optional[TensorSpec] = None,
        safe: Optional[bool] = True,
    ):
        super().__init__(policy)
        if sigma_end > sigma_init:
            raise RuntimeError("sigma should decrease over time or be constant")
        self.register_buffer("sigma_init", torch.tensor([sigma_init]))
        self.register_buffer("sigma_end", torch.tensor([sigma_end]))
        self.annealing_num_steps = annealing_num_steps
        self.register_buffer("mean", torch.tensor([mean]))
        self.register_buffer("std", torch.tensor([std]))
        self.register_buffer("sigma", torch.tensor([sigma_init], dtype=torch.float32))
        self.action_key = action_key
        self.out_keys = list(self.td_module.out_keys)
        if action_key not in self.out_keys:
            raise RuntimeError(
                f"The action key {action_key} was not found in the td_module out_keys {self.td_module.out_keys}."
            )
        if spec is not None:
            if not isinstance(spec, CompositeSpec) and len(self.out_keys) >= 1:
                spec = CompositeSpec({action_key: spec}, shape=spec.shape[:-1])
            self._spec = spec
        elif hasattr(self.td_module, "_spec"):
            self._spec = self.td_module._spec.clone()
            if action_key not in self._spec.keys(True, True):
                self._spec[action_key] = None
        elif hasattr(self.td_module, "spec"):
            self._spec = self.td_module.spec.clone()
            if action_key not in self._spec.keys(True, True):
                self._spec[action_key] = None
        else:
            self._spec = CompositeSpec({key: None for key in policy.out_keys})

        self.safe = safe
        if self.safe:
            self.register_forward_hook(_forward_hook_safe_action)

    @property
    def spec(self):
        return self._spec

    def step(self, frames: int = 1) -> None:
        """A step of sigma decay.

        After self.annealing_num_steps, this function is a no-op.

        Args:
            frames (int): number of frames since last step.

        """
        for _ in range(frames):
            self.sigma.data[0] = max(
                self.sigma_end.item(),
                (
                    self.sigma
                    - (self.sigma_init - self.sigma_end) / self.annealing_num_steps
                ).item(),
            )

    def _add_noise(self, action: torch.Tensor) -> torch.Tensor:
        sigma = self.sigma.item()
        noise = torch.normal(
            mean=torch.ones(action.shape) * self.mean.item(),
            std=torch.ones(action.shape) * self.std.item(),
        ).to(action.device)
        action = action + noise * sigma
        spec = self.spec
        spec = spec[self.action_key]
        if spec is not None:
            action = spec.project(action)
        elif self.safe:
            raise RuntimeError(
                "the action spec must be provided to AdditiveGaussianWrapper unless "
                "the `safe` keyword argument is turned off at initialization."
            )
        return action

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = self.td_module.forward(tensordict)
        if exploration_type() is ExplorationType.RANDOM or exploration_type() is None:
            out = tensordict.get(self.action_key)
            out = self._add_noise(out)
            tensordict.set(self.action_key, out)
        return tensordict


class OrnsteinUhlenbeckProcessWrapper(TensorDictModuleWrapper):
    r"""Ornstein-Uhlenbeck exploration policy wrapper.

    Presented in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING", https://arxiv.org/pdf/1509.02971.pdf.

    The OU exploration is to be used with continuous control policies and introduces a auto-correlated exploration
    noise. This enables a sort of 'structured' exploration.

    Noise equation:

    .. math::
        noise_t = noise_{t-1} + \theta * (mu - noise_{t-1}) * dt + \sigma_t * \sqrt{dt} * W

    Sigma equation:

    .. math::
        \sigma_t = max(\sigma^{min, (-(\sigma_{t-1} - \sigma^{min}) / (n^{\text{steps annealing}}) * n^{\text{steps}} + \sigma))

    To keep track of the steps and noise from sample to sample, an :obj:`"ou_prev_noise{id}"` and :obj:`"ou_steps{id}"` keys
    will be written in the input/output tensordict. It is expected that the tensordict will be zeroed at reset,
    indicating that a new trajectory is being collected. If not, and is the same tensordict is used for consecutive
    trajectories, the step count will keep on increasing across rollouts. Note that the collector classes take care of
    zeroing the tensordict at reset time.

    .. note::
        Once an environment has been wrapped in :class:`OrnsteinUhlenbeckProcessWrapper`, it is
        crucial to incorporate a call to :meth:`~.step` in the training loop
        to update the exploration factor.
        Since it is not easy to capture this omission no warning or exception
        will be raised if this is ommitted!

    Args:
        policy (TensorDictModule): a policy

    Keyword Args:
        eps_init (scalar): initial epsilon value, determining the amount of noise to be added.
            default: 1.0
        eps_end (scalar): final epsilon value, determining the amount of noise to be added.
            default: 0.1
        annealing_num_steps (int): number of steps it will take for epsilon to reach the eps_end value.
            default: 1000
        theta (scalar): theta factor in the noise equation
            default: 0.15
        mu (scalar): OU average (mu in the noise equation).
            default: 0.0
        sigma (scalar): sigma value in the sigma equation.
            default: 0.2
        dt (scalar): dt in the noise equation.
            default: 0.01
        x0 (Tensor, ndarray, optional): initial value of the process.
            default: 0.0
        sigma_min (number, optional): sigma_min in the sigma equation.
            default: None
        n_steps_annealing (int): number of steps for the sigma annealing.
            default: 1000
        action_key (NestedKey, optional): key of the action to be modified.
            default: "action"
        is_init_key (NestedKey, optional): key where to find the is_init flag used to reset the noise steps.
            default: "is_init"
        spec (TensorSpec, optional): if provided, the sampled action will be
            projected onto the valid action space once explored. If not provided,
            the exploration wrapper will attempt to recover it from the policy.
        safe (bool): if ``True``, actions that are out of bounds given the action specs will be projected in the space
            given the :obj:`TensorSpec.project` heuristic.
            default: True

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import BoundedTensorSpec
        >>> from torchrl.modules import OrnsteinUhlenbeckProcessWrapper, Actor
        >>> torch.manual_seed(0)
        >>> spec = BoundedTensorSpec(-1, 1, torch.Size([4]))
        >>> module = torch.nn.Linear(4, 4, bias=False)
        >>> policy = Actor(module=module, spec=spec)
        >>> explorative_policy = OrnsteinUhlenbeckProcessWrapper(policy)
        >>> td = TensorDict({"observation": torch.zeros(10, 4)}, batch_size=[10])
        >>> print(explorative_policy(td))
        TensorDict(
            fields={
                _ou_prev_noise: Tensor(torch.Size([10, 4]), dtype=torch.float32),
                _ou_steps: Tensor(torch.Size([10, 1]), dtype=torch.int64),
                action: Tensor(torch.Size([10, 4]), dtype=torch.float32),
                observation: Tensor(torch.Size([10, 4]), dtype=torch.float32)},
            batch_size=torch.Size([10]),
            device=None,
            is_shared=False)
    """

    def __init__(
        self,
        policy: TensorDictModule,
        *,
        eps_init: float = 1.0,
        eps_end: float = 0.1,
        annealing_num_steps: int = 1000,
        theta: float = 0.15,
        mu: float = 0.0,
        sigma: float = 0.2,
        dt: float = 1e-2,
        x0: Optional[Union[torch.Tensor, np.ndarray]] = None,
        sigma_min: Optional[float] = None,
        n_steps_annealing: int = 1000,
        action_key: Optional[NestedKey] = "action",
        is_init_key: Optional[NestedKey] = "is_init",
        spec: TensorSpec = None,
        safe: bool = True,
        key: Optional[NestedKey] = None,
    ):
        if key is not None:
            action_key = key
            warnings.warn(
                f"the 'key' keyword argument of {type(self)} has been renamed 'action_key'. The 'key' entry will be deprecated soon."
            )
        super().__init__(policy)
        self.ou = _OrnsteinUhlenbeckProcess(
            theta=theta,
            mu=mu,
            sigma=sigma,
            dt=dt,
            x0=x0,
            sigma_min=sigma_min,
            n_steps_annealing=n_steps_annealing,
            key=action_key,
        )
        self.register_buffer("eps_init", torch.tensor([eps_init]))
        self.register_buffer("eps_end", torch.tensor([eps_end]))
        if self.eps_end > self.eps_init:
            raise ValueError(
                "eps should decrease over time or be constant, "
                f"got eps_init={eps_init} and eps_end={eps_end}"
            )
        self.annealing_num_steps = annealing_num_steps
        self.register_buffer("eps", torch.tensor([eps_init], dtype=torch.float32))
        self.out_keys = list(self.td_module.out_keys) + self.ou.out_keys
        self.is_init_key = is_init_key
        noise_key = self.ou.noise_key
        steps_key = self.ou.steps_key

        if spec is not None:
            if not isinstance(spec, CompositeSpec) and len(self.out_keys) >= 1:
                spec = CompositeSpec({action_key: spec}, shape=spec.shape[:-1])
            self._spec = spec
        elif hasattr(self.td_module, "_spec"):
            self._spec = self.td_module._spec.clone()
            if action_key not in self._spec.keys(True, True):
                self._spec[action_key] = None
        elif hasattr(self.td_module, "spec"):
            self._spec = self.td_module.spec.clone()
            if action_key not in self._spec.keys(True, True):
                self._spec[action_key] = None
        else:
            self._spec = CompositeSpec({key: None for key in policy.out_keys})
        ou_specs = {
            noise_key: None,
            steps_key: None,
        }
        self._spec.update(ou_specs)
        if len(set(self.out_keys)) != len(self.out_keys):
            raise RuntimeError(f"Got multiple identical output keys: {self.out_keys}")
        self.safe = safe
        if self.safe:
            self.register_forward_hook(_forward_hook_safe_action)

    @property
    def spec(self):
        return self._spec

    def step(self, frames: int = 1) -> None:
        """Updates the eps noise factor.

        Args:
            frames (int): number of frames of the current batch (corresponding to the number of updates to be made).

        """
        for _ in range(frames):
            if self.annealing_num_steps > 0:
                self.eps.data[0] = max(
                    self.eps_end.item(),
                    (
                        self.eps
                        - (self.eps_init - self.eps_end) / self.annealing_num_steps
                    ).item(),
                )
            else:
                raise ValueError(
                    f"{self.__class__.__name__}.step() called when "
                    f"self.annealing_num_steps={self.annealing_num_steps}. Expected a strictly positive "
                    f"number of frames."
                )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        tensordict = super().forward(tensordict)
        if exploration_type() == ExplorationType.RANDOM or exploration_type() is None:
            is_init = tensordict.get(self.is_init_key, None)
            if is_init is None:
                warnings.warn(
                    f"The tensordict passed to {self.__class__.__name__} appears to be "
                    f"missing the '{self.is_init_key}' entry. This entry is used to "
                    f"reset the noise at the beginning of a trajectory, without it "
                    f"the behaviour of this exploration method is undefined. "
                    f"This is allowed for BC compatibility purposes but it will be deprecated soon! "
                    f"To create a '{self.is_init_key}' entry, simply append an torchrl.envs.InitTracker "
                    f"transform to your environment with `env = TransformedEnv(env, InitTracker())`."
                )
            tensordict = self.ou.add_sample(
                tensordict, self.eps.item(), is_init=is_init
            )
        return tensordict


# Based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
class _OrnsteinUhlenbeckProcess:
    def __init__(
        self,
        theta: float,
        mu: float = 0.0,
        sigma: float = 0.2,
        dt: float = 1e-2,
        x0: Optional[Union[torch.Tensor, np.ndarray]] = None,
        sigma_min: Optional[float] = None,
        n_steps_annealing: int = 1000,
        key: Optional[NestedKey] = "action",
        is_init_key: Optional[NestedKey] = "is_init",
    ):
        self.mu = mu
        self.sigma = sigma

        if sigma_min is not None:
            self.m = -float(sigma - sigma_min) / float(n_steps_annealing)
            self.c = sigma
            self.sigma_min = sigma_min
        else:
            self.m = 0.0
            self.c = sigma
            self.sigma_min = sigma

        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0 if x0 is not None else 0.0
        self.key = key
        self.is_init_key = is_init_key
        self._noise_key = "_ou_prev_noise"
        self._steps_key = "_ou_steps"
        self.out_keys = [self.noise_key, self.steps_key]

    @property
    def noise_key(self):
        return self._noise_key  # + str(id(self))

    @property
    def steps_key(self):
        return self._steps_key  # + str(id(self))

    def _make_noise_pair(
        self,
        action_tensordict: TensorDictBase,
        tensordict: TensorDictBase,
        is_init: torch.Tensor,
    ):
        if self.steps_key not in tensordict.keys():
            noise = torch.zeros(
                tensordict.get(self.key).shape, device=tensordict.device
            )
            steps = torch.zeros(
                action_tensordict.batch_size, dtype=torch.long, device=tensordict.device
            )
            tensordict.set(self.noise_key, noise)
            tensordict.set(self.steps_key, steps)
        else:
            noise = tensordict.get(self.noise_key)
            steps = tensordict.get(self.steps_key)
        if is_init is not None:
            noise[is_init] = 0
            steps[is_init] = 0
        return noise, steps

    def add_sample(
        self,
        tensordict: TensorDictBase,
        eps: float = 1.0,
        is_init: Optional[torch.Tensor] = None,
    ) -> TensorDictBase:

        # Get the nested tensordict where the action lives
        if isinstance(self.key, tuple) and len(self.key) > 1:
            action_tensordict = tensordict.get(self.key[:-1])
        else:
            action_tensordict = tensordict

        if is_init is None:
            is_init = tensordict.get(self.is_init_key, None)
        if (
            is_init is not None
        ):  # is_init has the shape of done_spec, let's bring it to the action_tensordict shape
            if is_init.ndim > 1 and is_init.shape[-1] == 1:
                is_init = is_init.squeeze(-1)  # Squeeze dangling dim
            if (
                action_tensordict.ndim >= is_init.ndim
            ):  # if is_init has less dimensions than action_tensordict we expand it
                is_init = expand_right(is_init, action_tensordict.shape)
            else:
                is_init = is_init.sum(
                    tuple(range(action_tensordict.batch_dims, is_init.ndim)),
                    dtype=torch.bool,
                )  # otherwise we reduce it to that batch_size
            if is_init.shape != action_tensordict.shape:
                raise ValueError(
                    f"'{self.is_init_key}' shape not compatible with action tensordict shape, "
                    f"got {tensordict.get(self.is_init_key).shape} and {action_tensordict.shape}"
                )

        prev_noise, n_steps = self._make_noise_pair(
            action_tensordict, tensordict, is_init
        )

        prev_noise = prev_noise + self.x0
        noise = (
            prev_noise
            + self.theta * (self.mu - prev_noise) * self.dt
            + self.current_sigma(expand_as_right(n_steps, prev_noise))
            * np.sqrt(self.dt)
            * torch.randn_like(prev_noise)
        )
        tensordict.set_(self.noise_key, noise - self.x0)
        tensordict.set_(self.key, tensordict.get(self.key) + eps * noise)
        tensordict.set_(self.steps_key, n_steps + 1)
        return tensordict

    def current_sigma(self, n_steps: torch.Tensor) -> torch.Tensor:
        sigma = (self.m * n_steps + self.c).clamp_min(self.sigma_min)
        return sigma
