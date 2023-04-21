# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
from typing import Optional, Union

import numpy as np
import torch
from tensordict.nn import TensorDictModule, TensorDictModuleWrapper
from tensordict.tensordict import TensorDictBase
from tensordict.utils import expand_as_right

from torchrl.data.tensor_specs import (
    CompositeSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.envs.utils import exploration_type, ExplorationType
from torchrl.modules.tensordict_module.common import _forward_hook_safe_action

__all__ = [
    "EGreedyWrapper",
    "AdditiveGaussianWrapper",
    "OrnsteinUhlenbeckProcessWrapper",
]


class EGreedyWrapper(TensorDictModuleWrapper):
    """Epsilon-Greedy PO wrapper.

    Args:
        policy (TensorDictModule): a deterministic policy.
        eps_init (scalar, optional): initial epsilon value.
            default: 1.0
        eps_end (scalar, optional): final epsilon value.
            default: 0.1
        annealing_num_steps (int, optional): number of steps it will take for epsilon to reach the eps_end value
        action_key (str, optional): if the policy module has more than one output key,
            its output spec will be of type CompositeSpec. One needs to know where to
            find the action spec.
            Default is "action".
        spec (TensorSpec, optional): if provided, the sampled action will be
            projected onto the valid action space once explored. If not provided,
            the exploration wrapper will attempt to recover it from the policy.

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
        eps_init: float = 1.0,
        eps_end: float = 0.1,
        annealing_num_steps: int = 1000,
        action_key: str = "action",
        spec: Optional[TensorSpec] = None,
    ):
        super().__init__(policy)
        self.register_buffer("eps_init", torch.tensor([eps_init]))
        self.register_buffer("eps_end", torch.tensor([eps_end]))
        if self.eps_end > self.eps_init:
            raise RuntimeError("eps should decrease over time or be constant")
        self.annealing_num_steps = annealing_num_steps
        self.register_buffer("eps", torch.tensor([eps_init]))
        self.action_key = action_key
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
            self._spec = CompositeSpec({key: None for key in policy.out_keys})

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
            out = tensordict.get(self.action_key)
            eps = self.eps.item()
            cond = (torch.rand(tensordict.shape, device=tensordict.device) < eps).to(
                out.dtype
            )
            cond = expand_as_right(cond, out)
            spec = self.spec
            if spec is not None:
                if isinstance(spec, CompositeSpec):
                    spec = spec[self.action_key]
                out = cond * spec.rand().to(out.device) + (1 - cond) * out
            else:
                raise RuntimeError(
                    "spec must be provided by the policy or directly to the exploration wrapper."
                )
            tensordict.set(self.td_module.out_keys[0], out)
        return tensordict


class AdditiveGaussianWrapper(TensorDictModuleWrapper):
    """Additive Gaussian PO wrapper.

    Args:
        policy (TensorDictModule): a policy.
        sigma_init (scalar, optional): initial epsilon value.
            default: 1.0
        sigma_end (scalar, optional): final epsilon value.
            default: 0.1
        annealing_num_steps (int, optional): number of steps it will take for
            sigma to reach the :obj:`sigma_end` value.
        mean (float, optional): mean of each output element’s normal distribution.
        std (float, optional): standard deviation of each output element’s normal distribution.
        action_key (str, optional): if the policy module has more than one output key,
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
        action_key: str = "action",
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
        self.register_buffer("sigma", torch.tensor([sigma_init]))
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
            if action_key not in self._spec.keys():
                self._spec[action_key] = None
        elif hasattr(self.td_module, "spec"):
            self._spec = self.td_module.spec.clone()
            if action_key not in self._spec.keys():
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
    """Ornstein-Uhlenbeck exploration policy wrapper.

    Presented in "CONTINUOUS CONTROL WITH DEEP REINFORCEMENT LEARNING", https://arxiv.org/pdf/1509.02971.pdf.

    The OU exploration is to be used with continuous control policies and introduces a auto-correlated exploration
    noise. This enables a sort of 'structured' exploration.

        Noise equation:
            noise = prev_noise + theta * (mu - prev_noise) * dt + current_sigma * sqrt(dt) * W
        Sigma equation:
            current_sigma = (-(sigma - sigma_min) / (n_steps_annealing) * n_steps + sigma).clamp_min(sigma_min)

    To keep track of the steps and noise from sample to sample, an :obj:`"ou_prev_noise{id}"` and :obj:`"ou_steps{id}"` keys
    will be written in the input/output tensordict. It is expected that the tensordict will be zeroed at reset,
    indicating that a new trajectory is being collected. If not, and is the same tensordict is used for consecutive
    trajectories, the step count will keep on increasing across rollouts. Note that the collector classes take care of
    zeroing the tensordict at reset time.

    Args:
        policy (TensorDictModule): a policy
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
        action_key (str): key of the action to be modified.
            default: "action"
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
        action_key: str = "action",
        spec: TensorSpec = None,
        safe: bool = True,
        key: str = None,
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
        self.register_buffer("eps", torch.tensor([eps_init]))
        self.out_keys = list(self.td_module.out_keys) + self.ou.out_keys
        noise_key = self.ou.noise_key
        steps_key = self.ou.steps_key

        ou_specs = {
            noise_key: None,
            steps_key: UnboundedContinuousTensorSpec(
                shape=(*self.td_module._spec.shape, 1),
                device=self.td_module._spec.device,
                dtype=torch.int64,
            ),
        }
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
            self._spec = CompositeSpec({key: None for key in policy.out_keys})
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
            if "is_init" not in tensordict.keys():
                warnings.warn(
                    f"The tensordict passed to {self.__class__.__name__} appears to be "
                    f"missing the 'is_init' entry. This entry is used to "
                    f"reset the noise at the beginning of a trajectory, without it "
                    f"the behaviour of this exploration method is undefined. "
                    f"This is allowed for BC compatibility purposes but it will be deprecated soon! "
                    f"To create a 'step_count' entry, simply append an torchrl.envs.InitTracker "
                    f"transform to your environment with `env = TransformedEnv(env, InitTracker())`."
                )
                tensordict.set(
                    "is_init", torch.zeros(*tensordict.shape, 1, dtype=torch.bool)
                )
            tensordict = self.ou.add_sample(tensordict, self.eps.item())
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
        key: str = "action",
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
        self._noise_key = "_ou_prev_noise"
        self._steps_key = "_ou_steps"
        self.out_keys = [self.noise_key, self.steps_key]

    @property
    def noise_key(self):
        return self._noise_key  # + str(id(self))

    @property
    def steps_key(self):
        return self._steps_key  # + str(id(self))

    def _make_noise_pair(self, tensordict: TensorDictBase, is_init=None) -> None:
        if is_init is not None:
            tensordict = tensordict.get_sub_tensordict(is_init.view(tensordict.shape))
        tensordict.set(
            self.noise_key,
            torch.zeros(tensordict.get(self.key).shape, device=tensordict.device),
            inplace=is_init is not None,
        )
        tensordict.set(
            self.steps_key,
            torch.zeros(
                torch.Size([*tensordict.batch_size, 1]),
                dtype=torch.long,
                device=tensordict.device,
            ),
            inplace=is_init is not None,
        )

    def add_sample(
        self, tensordict: TensorDictBase, eps: float = 1.0
    ) -> TensorDictBase:

        if self.noise_key not in tensordict.keys():
            self._make_noise_pair(tensordict)
        is_init = tensordict.get("is_init", None)
        if is_init is not None and is_init.any():
            self._make_noise_pair(tensordict, is_init.view(tensordict.shape))

        prev_noise = tensordict.get(self.noise_key)
        prev_noise = prev_noise + self.x0

        n_steps = tensordict.get(self.steps_key)

        noise = (
            prev_noise
            + self.theta * (self.mu - prev_noise) * self.dt
            + self.current_sigma(n_steps)
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
