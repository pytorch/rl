# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
from collections.abc import Sequence
from copy import copy
from enum import IntEnum
from textwrap import indent
from typing import Any, TYPE_CHECKING

import torch

from tensordict import TensorDictBase
from tensordict.utils import NestedKey, unravel_key
from torch import nn

from torchrl.data.tensor_specs import (
    Bounded,
    Categorical,
    Composite,
    ContinuousBox,
    MultiCategorical,
    MultiOneHot,
    OneHot,
    TensorSpec,
    Unbounded,
)

if TYPE_CHECKING:
    pass

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

from torchrl.envs.transforms._base import FORWARD_NOT_IMPLEMENTED, Transform

__all__ = [
    "ActionDiscretizer",
    "ActionMask",
    "ActionScaling",
    "DiscreteActionProjection",
    "MultiAction",
]


class DiscreteActionProjection(Transform):
    """Projects discrete actions from a high dimensional space to a low dimensional space.

    Given a discrete action (from 1 to N) encoded as a one-hot vector and a
    maximum action index num_actions (with num_actions < N), transforms the action such that
    action_out is at most num_actions.

    If the input action is > num_actions, it is being replaced by a random value
    between 0 and num_actions-1. Otherwise the same action is kept.
    This is intended to be used with policies applied over multiple discrete
    control environments with different action space.

    A call to DiscreteActionProjection.forward (eg from a replay buffer or in a
    sequence of nn.Modules) will call the transform num_actions_effective -> max_actions
    on the :obj:`"in_keys"`, whereas a call to _call will be ignored. Indeed,
    transformed envs are instructed to update the input keys only for the inner
    base_env, but the original input keys will remain unchanged.

    Args:
        num_actions_effective (int): max number of action considered.
        max_actions (int): maximum number of actions that this module can read.
        action_key (NestedKey, optional): key name of the action. Defaults to "action".
        include_forward (bool, optional): if ``True``, a call to forward will also
            map the action from one domain to the other when the module is called
            by a replay buffer or an nn.Module chain. Defaults to `True`.

    Examples:
        >>> torch.manual_seed(0)
        >>> N = 3
        >>> M = 2
        >>> action = torch.zeros(N, dtype=torch.long)
        >>> action[-1] = 1
        >>> td = TensorDict({"action": action}, [])
        >>> transform = DiscreteActionProjection(num_actions_effective=M, max_actions=N)
        >>> _ = transform.inv(td)
        >>> print(td.get("action"))
        tensor([1])
    """

    def __init__(
        self,
        num_actions_effective: int,
        max_actions: int,
        action_key: NestedKey = "action",
        include_forward: bool = True,
    ):
        in_keys_inv = [action_key]
        if include_forward:
            in_keys = in_keys_inv
        else:
            in_keys = []
        if in_keys_inv is None:
            in_keys_inv = []
        super().__init__(
            in_keys=in_keys,
            out_keys=copy(in_keys),
            in_keys_inv=in_keys_inv,
            out_keys_inv=copy(in_keys_inv),
        )
        self.num_actions_effective = num_actions_effective
        self.max_actions = max_actions
        if max_actions < num_actions_effective:
            raise RuntimeError(
                "The `max_actions` int must be greater or equal to `num_actions_effective`."
            )

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        # We don't do anything here because the action is modified by the inv
        # method but we don't need to map it back as it won't be updated in the original
        # tensordict
        return next_tensordict

    def _apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        # We still need to code the forward transform for replay buffers and models
        action = action.argmax(-1)  # bool to int
        action = nn.functional.one_hot(action, self.max_actions)
        return action

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        if action.shape[-1] != self.max_actions:
            raise RuntimeError(
                f"action.shape[-1]={action.shape[-1]} must match self.max_actions={self.max_actions}."
            )
        action = action.long().argmax(-1)  # bool to int
        idx = action >= self.num_actions_effective
        if idx.any():
            action[idx] = torch.randint(self.num_actions_effective, (idx.sum(),))
        action = nn.functional.one_hot(action, self.num_actions_effective)
        return action

    def transform_input_spec(self, input_spec: Composite):
        input_spec = input_spec.clone()
        for key in input_spec["full_action_spec"].keys(True, True):
            key = ("full_action_spec", key)
            break
        else:
            raise KeyError("key not found in action_spec.")
        input_spec[key] = OneHot(
            self.max_actions,
            shape=(*input_spec[key].shape[:-1], self.max_actions),
            device=input_spec.device,
            dtype=input_spec[key].dtype,
        )
        return input_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(num_actions_effective={self.num_actions_effective}, max_actions={self.max_actions}, "
            f"in_keys_inv={self.in_keys_inv})"
        )


class ActionMask(Transform):
    """An adaptive action masker.

    This transform is useful to ensure that randomly generated actions
    respect legal actions, by masking the action specs.
    It reads the mask from the input tensordict after the step is executed,
    and adapts the mask of the finite action spec.

    .. note:: This transform will fail when used without an environment.

    .. note:: **MultiDiscrete action spaces with 2D masks (e.g., board games)**

        When wrapping a Gym environment with a ``MultiDiscrete`` action space
        (e.g., ``MultiDiscrete([5, 5])``) and an ``action_mask`` observation whose
        shape matches the ``nvec`` (e.g., shape ``(5, 5)``), the :class:`~torchrl.envs.GymWrapper`
        automatically converts the action space to a flattened ``Categorical(n=25)``
        or ``OneHot(n=25)``. This allows the mask to represent all possible action
        combinations (25 in this example) rather than independent sub-actions.

        This is particularly useful for grid-based games where the mask indicates
        which (row, column) positions are valid moves.

    Args:
        action_key (NestedKey, optional): the key where the action tensor can be found.
            Defaults to ``"action"``.
        mask_key (NestedKey, optional): the key where the action mask can be found.
            Defaults to ``"action_mask"``.

    Examples:
        >>> import torch
        >>> from torchrl.data.tensor_specs import Categorical, Binary, Unbounded, Composite
        >>> from torchrl.envs.transforms import ActionMask, TransformedEnv
        >>> from torchrl.envs.common import EnvBase
        >>> class MaskedEnv(EnvBase):
        ...     def __init__(self, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...         self.action_spec = Categorical(4)
        ...         self.state_spec = Composite(action_mask=Binary(4, dtype=torch.bool))
        ...         self.observation_spec = Composite(obs=Unbounded(3))
        ...         self.reward_spec = Unbounded(1)
        ...
        ...     def _reset(self, tensordict=None):
        ...         td = self.observation_spec.rand()
        ...         td.update(torch.ones_like(self.state_spec.rand()))
        ...         return td
        ...
        ...     def _step(self, data):
        ...         td = self.observation_spec.rand()
        ...         mask = data.get("action_mask")
        ...         action = data.get("action")
        ...         mask = mask.scatter(-1, action.unsqueeze(-1), 0)
        ...
        ...         td.set("action_mask", mask)
        ...         td.set("reward", self.reward_spec.rand())
        ...         td.set("done", ~mask.any().view(1))
        ...         return td
        ...
        ...     def _set_seed(self, seed) -> None:
        ...         pass
        ...
        >>> torch.manual_seed(0)
        >>> base_env = MaskedEnv()
        >>> env = TransformedEnv(base_env, ActionMask())
        >>> r = env.rollout(10)
        >>> r["action_mask"]
        tensor([[ True,  True,  True,  True],
                [ True,  True, False,  True],
                [ True,  True, False, False],
                [ True, False, False, False]])

    """

    ACCEPTED_SPECS = (
        OneHot,
        Categorical,
        MultiOneHot,
        MultiCategorical,
    )
    SPEC_TYPE_ERROR = "The action spec must be one of {}. Got {} instead."

    def __init__(
        self, action_key: NestedKey = "action", mask_key: NestedKey = "action_mask"
    ):
        if not isinstance(action_key, (tuple, str)):
            raise ValueError(
                f"The action key must be a nested key. Got {type(action_key)} instead."
            )
        if not isinstance(mask_key, (tuple, str)):
            raise ValueError(
                f"The mask key must be a nested key. Got {type(mask_key)} instead."
            )
        super().__init__(
            in_keys=[action_key, mask_key], out_keys=[], in_keys_inv=[], out_keys_inv=[]
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError(FORWARD_NOT_IMPLEMENTED.format(type(self)))

    @property
    def action_spec(self) -> TensorSpec:
        action_spec = self.container.full_action_spec[self.in_keys[0]]
        if not isinstance(action_spec, self.ACCEPTED_SPECS):
            raise ValueError(
                self.SPEC_TYPE_ERROR.format(self.ACCEPTED_SPECS, type(action_spec))
            )
        return action_spec

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if self.parent is None:
            raise RuntimeError(
                f"{type(self)}.parent cannot be None: make sure this transform is executed within an environment."
            )

        mask = next_tensordict.get(self.in_keys[1])
        self.action_spec.update_mask(mask.to(self.action_spec.device))

        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return self._call(tensordict_reset)


class ActionDiscretizer(Transform):
    """A transform to discretize a continuous action space.

    This transform makes it possible to use an algorithm designed for discrete
    action spaces such as DQN over environments with a continuous action space.

    Args:
        num_intervals (int or torch.Tensor): the number of discrete values
            for each element of the action space. If a single integer is provided,
            all action items are sliced with the same number of elements.
            If a tensor is provided, it must have the same number of elements
            as the action space (ie, the length of the ``num_intervals`` tensor
            must match the last dimension of the action space).
        action_key (NestedKey, optional): the action key to use. Points to
            the action of the parent env (the floating point action).
            Defaults to ``"action"``.
        out_action_key (NestedKey, optional): the key where the discrete
            action should be written. If ``None`` is provided, it defaults to
            the value of ``action_key``. If both keys do not match, the
            continuous action_spec is moved from the ``full_action_spec``
            environment attribute to the ``full_state_spec`` container,
            as only the discrete action should be sampled for an action to
            be taken. Providing ``out_action_key`` can ensure that the
            floating point action is available to be recorded.
        sampling (ActionDiscretizer.SamplingStrategy, optinoal): an element
            of the ``ActionDiscretizer.SamplingStrategy`` ``IntEnum`` object
            (``MEDIAN``, ``LOW``, ``HIGH`` or ``RANDOM``). Indicates how the
            continuous action should be sampled in the provided interval.
        categorical (bool, optional): if ``False``, one-hot encoding is used.
            Defaults to ``True``.

    Examples:
        >>> from torchrl.envs import GymEnv, check_env_specs
        >>> import torch
        >>> base_env = GymEnv("HalfCheetah-v4")
        >>> num_intervals = torch.arange(5, 11)
        >>> categorical = True
        >>> sampling = ActionDiscretizer.SamplingStrategy.MEDIAN
        >>> t = ActionDiscretizer(
        ...     num_intervals=num_intervals,
        ...     categorical=categorical,
        ...     sampling=sampling,
        ...     out_action_key="action_disc",
        ... )
        >>> env = base_env.append_transform(t)
        TransformedEnv(
            env=GymEnv(env=HalfCheetah-v4, batch_size=torch.Size([]), device=cpu),
            transform=ActionDiscretizer(
                num_intervals=tensor([ 5,  6,  7,  8,  9, 10]),
                action_key=action,
                out_action_key=action_disc,,
                sampling=0,
                categorical=True))
        >>> check_env_specs(env)
        >>> # Produce a rollout
        >>> r = env.rollout(4)
        >>> print(r)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([4, 6]), device=cpu, dtype=torch.float32, is_shared=False),
                action_disc: Tensor(shape=torch.Size([4, 6]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([4, 17]), device=cpu, dtype=torch.float64, is_shared=False),
                        reward: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([4]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([4, 17]), device=cpu, dtype=torch.float64, is_shared=False),
                terminated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([4]),
            device=cpu,
            is_shared=False)
        >>> assert r["action"].dtype == torch.float
        >>> assert r["action_disc"].dtype == torch.int64
        >>> assert (r["action"] < base_env.action_spec.high).all()
        >>> assert (r["action"] > base_env.action_spec.low).all()

    .. note:: Custom Sampling Strategies

        To implement a custom sampling strategy beyond the built-in options
        (``MEDIAN``, ``LOW``, ``HIGH``, ``RANDOM``), subclass ``ActionDiscretizer``
        and override the :meth:`~ActionDiscretizer.custom_arange` method. This
        method computes the normalized interval positions (values in ``[0, 1)``)
        that determine where each discrete action maps within the continuous
        action interval.

        Example:
            >>> class LogSpacedActionDiscretizer(ActionDiscretizer):
            ...     def custom_arange(self, nint, device):
            ...         # Use logarithmic spacing instead of linear
            ...         return torch.logspace(-2, 0, nint, device=device) - 0.01

    """

    class SamplingStrategy(IntEnum):
        """The sampling strategies for ActionDiscretizer."""

        MEDIAN = 0
        LOW = 1
        HIGH = 2
        RANDOM = 3

    def __init__(
        self,
        num_intervals: int | torch.Tensor,
        action_key: NestedKey = "action",
        out_action_key: NestedKey = None,
        sampling=None,
        categorical: bool = True,
    ):
        if out_action_key is None:
            out_action_key = action_key
        super().__init__(in_keys_inv=[action_key], out_keys_inv=[out_action_key])
        self.action_key = action_key
        self.out_action_key = out_action_key
        if not isinstance(num_intervals, torch.Tensor):
            self.num_intervals = num_intervals
        else:
            self.register_buffer("num_intervals", num_intervals)
        if sampling is None:
            sampling = self.SamplingStrategy.MEDIAN
        self.sampling = sampling
        self.categorical = categorical

    def __repr__(self) -> str:
        def _indent(s):
            return indent(s, 4 * " ")

        num_intervals = f"num_intervals={self.num_intervals}"
        action_key = f"action_key={self.action_key}"
        out_action_key = f"out_action_key={self.out_action_key}"
        sampling = f"sampling={self.sampling}"
        categorical = f"categorical={self.categorical}"
        return (
            f"{type(self).__name__}(\n{_indent(num_intervals)},\n{_indent(action_key)},"
            f"\n{_indent(out_action_key)},\n{_indent(sampling)},\n{_indent(categorical)})"
        )

    def custom_arange(self, nint, device):
        """Compute the normalized interval positions for discretization.

        This method generates values in the range [0, 1) that determine where
        each discrete action maps within the continuous action interval.

        Override this method in a subclass to implement custom sampling
        strategies beyond the built-in ``MEDIAN``, ``LOW``, ``HIGH``, and
        ``RANDOM`` strategies.

        Args:
            nint (int): the number of intervals (discrete actions) for this
                action dimension.
            device (torch.device): the device on which to create the tensor.

        Returns:
            torch.Tensor: a 1D tensor of shape ``(nint,)`` with values in
                ``[0, 1)`` representing the normalized positions within each
                interval.

        Example:
            >>> class CustomActionDiscretizer(ActionDiscretizer):
            ...     def custom_arange(self, nint, device):
            ...         # Custom sampling: use logarithmic spacing
            ...         return torch.logspace(-2, 0, nint, device=device) - 0.01

        """
        result = torch.arange(
            start=0.0,
            end=1.0,
            step=1 / nint,
            dtype=self.dtype,
            device=device,
        )
        result_ = result
        if self.sampling in (
            self.SamplingStrategy.HIGH,
            self.SamplingStrategy.MEDIAN,
        ):
            result_ = (1 - result).flip(0)
        if self.sampling == self.SamplingStrategy.MEDIAN:
            result = (result + result_) / 2
        else:
            result = result_
        return result

    def transform_input_spec(self, input_spec):
        try:
            action_spec = self.parent.full_action_spec_unbatched[self.in_keys_inv[0]]
            if not isinstance(action_spec, Bounded):
                raise TypeError(
                    f"action spec type {type(action_spec)} is not supported. The action spec type must be Bounded."
                )

            n_act = action_spec.shape
            if not n_act:
                n_act = ()
                empty_shape = True
            else:
                n_act = (n_act[-1],)
                empty_shape = False
            self.n_act = n_act

            self.dtype = action_spec.dtype
            interval = action_spec.high - action_spec.low

            num_intervals = self.num_intervals

            if not empty_shape:
                interval = interval.unsqueeze(-1)
            elif isinstance(num_intervals, torch.Tensor):
                num_intervals = int(num_intervals.squeeze())
                self.num_intervals = torch.as_tensor(num_intervals)

            if isinstance(num_intervals, int):
                arange = (
                    self.custom_arange(num_intervals, action_spec.device).expand(
                        (*n_act, num_intervals)
                    )
                    * interval
                )
                low = action_spec.low
                if not empty_shape:
                    low = low.unsqueeze(-1)
                self.register_buffer("intervals", low + arange)
            else:
                arange = [
                    self.custom_arange(_num_intervals, action_spec.device) * interval
                    for _num_intervals, interval in zip(
                        num_intervals.tolist(), interval.unbind(-2)
                    )
                ]
                self.intervals = [
                    low + arange
                    for low, arange in zip(
                        action_spec.low.unsqueeze(-1).unbind(-2), arange
                    )
                ]

            if not isinstance(num_intervals, torch.Tensor):
                nvec = torch.as_tensor(num_intervals, device=action_spec.device)
            else:
                nvec = num_intervals
            if nvec.ndim > 1:
                raise RuntimeError(f"Cannot use num_intervals with shape {nvec.shape}")
            if nvec.ndim == 0 or nvec.numel() == 1:
                if not empty_shape:
                    nvec = nvec.expand(action_spec.shape[-1])
                else:
                    nvec = nvec.squeeze()
            self.register_buffer("nvec", nvec)
            if self.sampling == self.SamplingStrategy.RANDOM:
                # compute jitters
                self.jitters = interval.squeeze(-1) / nvec
            shape = (
                action_spec.shape
                if self.categorical
                else (*action_spec.shape[:-1], nvec.sum())
            )

            if not empty_shape:
                cls = (
                    functools.partial(MultiCategorical, remove_singleton=False)
                    if self.categorical
                    else MultiOneHot
                )
                action_spec = cls(nvec=nvec, shape=shape, device=action_spec.device)

            else:
                cls = Categorical if self.categorical else OneHot
                action_spec = cls(n=int(nvec), shape=shape, device=action_spec.device)

            batch_size = self.parent.batch_size
            if batch_size:
                action_spec = action_spec.expand(batch_size + action_spec.shape)
            input_spec["full_action_spec", self.out_keys_inv[0]] = action_spec

            if self.out_keys_inv[0] != self.in_keys_inv[0]:
                input_spec["full_state_spec", self.in_keys_inv[0]] = input_spec[
                    "full_action_spec", self.in_keys_inv[0]
                ].clone()
                del input_spec["full_action_spec", self.in_keys_inv[0]]
            return input_spec
        except AttributeError as err:
            # To avoid silent AttributeErrors
            raise RuntimeError(str(err))

    def _init(self):
        # We just need to access the action spec for everything to be initialized
        try:
            _ = self.container.full_action_spec
        except AttributeError:
            raise RuntimeError(
                f"Cannot execute transform {type(self).__name__} without a parent env."
            )

    def inv(self, tensordict):
        if self.out_keys_inv[0] == self.in_keys_inv[0]:
            return super().inv(tensordict)
        # We re-write this because we don't want to clone the TD here
        return self._inv_call(tensordict)

    def _inv_call(self, tensordict):
        # action is categorical, map it to desired dtype
        intervals = getattr(self, "intervals", None)
        if intervals is None:
            self._init()
            return self._inv_call(tensordict)
        action = tensordict.get(self.out_keys_inv[0])
        if self.categorical:
            action = action.unsqueeze(-1)
            if isinstance(intervals, torch.Tensor):
                shape = action.shape[: -intervals.ndim]
                intervals = intervals.expand(shape + intervals.shape)
                action = intervals.gather(index=action, dim=-1).squeeze(-1)
            else:
                action = torch.stack(
                    [
                        interval.gather(index=action, dim=-1).squeeze(-1)
                        for interval, action in zip(intervals, action.unbind(-2))
                    ],
                    -1,
                )
        else:
            nvec = self.nvec
            empty_shape = not nvec.ndim
            if not empty_shape:
                nvec = nvec.tolist()
                if isinstance(intervals, torch.Tensor):
                    shape = action.shape[: (-intervals.ndim + 1)]
                    intervals = intervals.expand(shape + intervals.shape)
                    intervals = intervals.unbind(-2)
                action = action.split(nvec, dim=-1)
                action = torch.stack(
                    [
                        intervals[action].view(action.shape[:-1])
                        for (intervals, action) in zip(intervals, action)
                    ],
                    -1,
                )
            else:
                shape = action.shape[: -intervals.ndim]
                intervals = intervals.expand(shape + intervals.shape)
                action = intervals[action].squeeze(-1)

        if self.sampling == self.SamplingStrategy.RANDOM:
            action = action + self.jitters * torch.rand_like(self.jitters)
        return tensordict.set(self.in_keys_inv[0], action)


class MultiAction(Transform):
    """A transform to execute multiple actions in the parent environment.

    This transform unbinds the actions along a specific dimension and passes each action independently.
    The returned transform can be either a stack of the observations gathered during the steps or only the
    last observation (and similarly for the rewards, see args below).

    By default, the actions must be stacked along the first dimension after the root tensordict batch-dims, i.e.

        >>> td = policy(td)
        >>> actions = td.select(*env.action_keys)
        >>> # Adapt the batch-size
        >>> actions = actions.auto_batch_size_(td.ndim + 1)
        >>> # Step-wise actions
        >>> actions = actions.unbind(-1)

    If a `"done"` entry is encountered, the next steps are skipped for the env that has reached that state.

    .. note:: If a transform is appended before the MultiAction, it will be called multiple times. If it is appended
        after, it will be called once per macro-step.

    Keyword Args:
        dim (int, optional): the stack dimension with respect to the tensordict ``ndim`` attribute.
            Must be greater than 0. Defaults to ``1`` (the first dimension after the batch-dims).
        stack_rewards (bool, optional): if ``True``, each step's reward will be stack in the output tensordict.
            If ``False``, only the last reward will be returned. The reward spec is adapted accordingly. The
            stack dimension is the same as the action stack dimension. Defaults to ``True``.
        stack_observations (bool, optional): if ``True``, each step's observation will be stack in the output tensordict.
            If ``False``, only the last observation will be returned. The observation spec is adapted accordingly. The
            stack dimension is the same as the action stack dimension. Defaults to ``False``.

    """

    def __init__(
        self,
        *,
        dim: int = 1,
        stack_rewards: bool = True,
        stack_observations: bool = False,
    ):
        super().__init__()
        self.stack_rewards = stack_rewards
        self.stack_observations = stack_observations
        self.dim = dim

    def _stack_tds(self, td_list, next_tensordict, keys):
        td = torch.stack(td_list + [next_tensordict.select(*keys)], -1)
        if self.dim != 1:
            d = td.ndim - 1
            td.auto_batch_size_(d + self.dim)
            td = td.transpose(d, d + self.dim)
        return td

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        # Collect the stacks if needed
        if self.stack_rewards:
            reward_td = self.rewards
            reward_td = self._stack_tds(
                reward_td, next_tensordict, self.parent.reward_keys
            )
            next_tensordict.update(reward_td)
        if self.stack_observations:
            obs_td = self.obs
            obs_td = self._stack_tds(
                obs_td, next_tensordict, self.parent.observation_keys
            )
            next_tensordict.update(obs_td)
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return tensordict_reset

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Get the actions
        parent = self.parent
        action_keys = parent.action_keys
        actions = tensordict.select(*action_keys)
        actions = actions.auto_batch_size_(batch_dims=tensordict.ndim + self.dim)
        actions = actions.unbind(-1)
        td = tensordict
        idx = None
        global_idx = None
        reset = False
        if self.stack_rewards:
            self.rewards = rewards = []
        if self.stack_observations:
            self.obs = obs = []
        for a in actions[:-1]:
            if global_idx is not None:
                a = a[global_idx]
            td = td.replace(a)
            td = parent.step(td)

            # Save rewards and done states
            if self.stack_rewards:
                reward_td = td["next"].select(*self.parent.reward_keys)
                if global_idx is not None:
                    reward_td_expand = reward_td.new_zeros(
                        global_idx.shape + reward_td.shape[global_idx.ndim :]
                    )
                    reward_td_expand[global_idx] = reward_td
                else:
                    reward_td_expand = reward_td

                rewards.append(reward_td_expand)
            if self.stack_observations:
                obs_td = td["next"].select(*self.parent.observation_keys)
                # obs_td = td.select("next", *self.parent.observation_keys).set("next", obs_td)
                if global_idx is not None:
                    obs_td = torch.where(global_idx, obs_td, 0)
                obs.append(obs_td)

            td = parent.step_mdp(td)
            if self.stack_rewards:
                td.update(reward_td)

            any_done = parent.any_done(td)
            if any_done:
                # Intersect the resets to avoid making any step after reset has been called
                reset = reset | td.pop("_reset").view(td.shape)
                if reset.all():
                    # Skip step for all
                    td["_step"] = ~reset
                    break
                elif parent.batch_locked:
                    td["_step"] = ~reset
                else:
                    # we can simply index the tensordict
                    idx = ~reset.view(td.shape)
                    if global_idx is None:
                        global_idx = idx.clone()
                        td_out = td
                    else:
                        td_out[global_idx] = td
                        global_idx = torch.masked_scatter(global_idx, global_idx, idx)
                    td = td[idx]
                    reset = reset[idx]  # Should be all False

        if global_idx is None:
            td_out = td.replace(actions[-1])
            if (self.stack_rewards or self.stack_observations) and not td_out.get(
                "_step", torch.ones((), dtype=torch.bool)
            ).any():
                td_out = self._step(None, td_out)
        else:
            td_out[global_idx] = td.replace(actions[-1][global_idx])
            if self.stack_rewards or self.stack_observations:
                td_out = self._step(None, td_out)
                if self.stack_rewards:
                    self.rewards = list(
                        torch.stack(self.rewards, -1)[global_idx].unbind(-1)
                    )
                if self.stack_observations:
                    self.obs = list(torch.stack(self.obs, -1)[global_idx].unbind(-1))

            td_out["_step"] = global_idx

        return td_out

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        try:
            action_spec = input_spec["full_action_spec"]
        except KeyError:
            raise KeyError(
                f"{type(self).__name__} requires an action spec to be present."
            )
        for _ in range(self.dim):
            action_spec = action_spec.unsqueeze(input_spec.ndim)
        # Make the dim dynamic
        action_spec = action_spec.expand(
            tuple(
                d if i != (input_spec.ndim + self.dim - 1) else -1
                for i, d in enumerate(action_spec.shape)
            )
        )
        input_spec["full_action_spec"] = action_spec
        return input_spec

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        if "full_reward_spec" in output_spec.keys():
            output_spec["full_reward_spec"] = self._transform_reward_spec(
                output_spec["full_reward_spec"], output_spec.ndim
            )
        if "full_observation_spec" in output_spec.keys():
            output_spec["full_observation_spec"] = self._transform_observation_spec(
                output_spec["full_observation_spec"], output_spec.ndim
            )
        return output_spec

    def _transform_reward_spec(self, reward_spec: TensorSpec, ndim) -> TensorSpec:
        if not self.stack_rewards:
            return reward_spec
        for _ in range(self.dim):
            reward_spec = reward_spec.unsqueeze(ndim)
        # Make the dim dynamic
        reward_spec = reward_spec.expand(
            tuple(
                d if i != (ndim + self.dim - 1) else -1
                for i, d in enumerate(reward_spec.shape)
            )
        )
        return reward_spec

    def _transform_observation_spec(
        self, observation_spec: TensorSpec, ndim
    ) -> TensorSpec:
        if not self.stack_observations:
            return observation_spec
        for _ in range(self.dim):
            observation_spec = observation_spec.unsqueeze(ndim)
        # Make the dim dynamic
        observation_spec = observation_spec.expand(
            tuple(
                d if i != (ndim + self.dim - 1) else -1
                for i, d in enumerate(observation_spec.shape)
            )
        )
        return observation_spec


class ActionScaling(Transform):
    r"""Affine-scale a continuous action using the bounds of the action spec.

    Given a bounded action spec with bounds ``[low, high]``, this transform exposes
    a normalized action space to the policy and rescales actions back to the
    original env range before they are passed to the environment.

    The ``loc`` and ``scale`` are derived from the spec:

    .. math::

        loc = \frac{high + low}{2}, \quad scale = \frac{high - low}{2}.

    When ``standard_normal=True`` (default) the normalized action space is
    ``[-1, 1]`` and the inverse mapping (policy action -> env action) is

    .. math::

        a_{env} = a_{norm} \cdot scale + loc.

    The forward mapping (env action -> normalized action, used by replay buffer
    transforms) is the inverse:

    .. math::

        a_{norm} = (a_{env} - loc) / scale.

    When ``standard_normal=False`` the normalized space is ``[0, 1]`` and the
    mapping is rescaled accordingly so that ``0`` maps to ``low`` and ``1`` to
    ``high``.

    Args:
        in_keys_inv (sequence of NestedKey, optional): keys read during the
            ``inv`` direction (policy -> env). Defaults to ``["action"]``. A
            single key per :class:`ActionScaling` instance is supported; compose
            several instances to scale several actions.
        out_keys_inv (sequence of NestedKey, optional): keys written during the
            ``inv`` direction. Defaults to ``in_keys_inv``.
        in_keys (sequence of NestedKey, optional): keys read during the forward
            direction (env action -> normalized action, used by replay buffers
            and inside :class:`~torch.nn.Module` chains). Defaults to
            ``in_keys_inv``.
        out_keys (sequence of NestedKey, optional): keys written during the
            forward direction. Defaults to ``in_keys``.

    Keyword Args:
        loc (torch.Tensor or float, optional): explicit location of the affine
            transform. If both ``loc`` and ``scale`` are provided the values are
            used as-is and no derivation from the spec is performed (useful when
            no parent environment is available, e.g. inside a replay buffer).
            Defaults to ``None``.
        scale (torch.Tensor or float, optional): explicit scale of the affine
            transform. Must be provided together with ``loc``.
            Defaults to ``None``.
        standard_normal (bool, optional): if ``True`` (default), the normalized
            action space is ``[-1, 1]``. If ``False``, the normalized action
            space is ``[0, 1]``.

    Raises:
        RuntimeError: if the action spec is unbounded or partially unbounded
            (any bound is non-finite).

    Examples:
        >>> import torch
        >>> from torchrl.data.tensor_specs import Bounded
        >>> from torchrl.envs.transforms import ActionScaling, TransformedEnv
        >>> from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv
        >>> base_env = ContinuousActionVecMockEnv(
        ...     action_spec=Bounded(low=-2.0, high=4.0, shape=(7,))
        ... )
        >>> env = TransformedEnv(base_env, ActionScaling())
        >>> env.action_spec.space.low
        tensor([-1., -1., -1., -1., -1., -1., -1.])
        >>> env.action_spec.space.high
        tensor([1., 1., 1., 1., 1., 1., 1.])
    """

    invertible = True

    def __init__(
        self,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        *,
        loc: torch.Tensor | float | None = None,
        scale: torch.Tensor | float | None = None,
        standard_normal: bool = True,
    ):
        if in_keys_inv is None:
            in_keys_inv = ["action"]
        if not isinstance(in_keys_inv, (list, tuple)):
            in_keys_inv = [in_keys_inv]
        if len(in_keys_inv) != 1:
            raise ValueError(
                "ActionScaling only supports a single action key per instance. "
                "Compose several ActionScaling transforms to scale multiple actions."
            )
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)
        if in_keys is None:
            in_keys = copy(in_keys_inv)
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )
        self.standard_normal = bool(standard_normal)

        if (loc is None) != (scale is None):
            raise ValueError(
                "loc and scale must either both be provided or both be None."
            )
        self._explicit = loc is not None
        if loc is not None:
            loc = torch.as_tensor(loc)
            scale = torch.as_tensor(scale)
            if not loc.dtype.is_floating_point:
                loc = loc.to(torch.get_default_dtype())
            if not scale.dtype.is_floating_point:
                scale = scale.to(torch.get_default_dtype())
            if (scale == 0).any():
                raise ValueError(
                    "scale must not contain zero entries (would cause division by zero)."
                )
            self.register_buffer("loc", loc)
            self.register_buffer("scale", scale)
        else:
            self.register_buffer("loc", nn.UninitializedBuffer())
            self.register_buffer("scale", nn.UninitializedBuffer())

    @property
    def initialized(self) -> bool:
        return not isinstance(self.loc, nn.UninitializedBuffer)

    def _ensure_initialized(self) -> None:
        # Lazily populate ``loc`` and ``scale`` from the parent env's action
        # spec at the insertion point of this transform. ``self.parent`` is
        # rebuilt with all transforms up to (but not including) ``self``, so
        # its action spec is exactly the env-scale spec we need to read.
        if self.initialized:
            return
        parent = self.parent
        if parent is None:
            raise RuntimeError(
                "ActionScaling has not been initialized: pass explicit ``loc`` "
                "and ``scale`` to the constructor, or attach this transform to "
                "a TransformedEnv whose action spec is bounded so that the "
                "values can be derived automatically."
            )
        in_key = unravel_key(self.in_keys_inv[0])
        full_action_spec = parent.full_action_spec
        if in_key not in full_action_spec.keys(True, True):
            raise RuntimeError(
                f"ActionScaling could not find key {in_key!r} in the parent "
                f"environment's action spec. Available keys: "
                f"{list(full_action_spec.keys(True, True))}."
            )
        self._init_from_spec(full_action_spec[in_key])

    def _init_from_spec(self, leaf_spec: TensorSpec) -> None:
        low, high = self._validate_bounded(leaf_spec)
        dtype = low.dtype if low.dtype.is_floating_point else torch.get_default_dtype()
        loc = ((high + low) / 2).to(dtype)
        scale = ((high - low) / 2).to(dtype)
        self._materialize_loc_scale(loc, scale)

    def _materialize_loc_scale(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        if isinstance(self.loc, nn.UninitializedBuffer):
            self.loc.materialize(shape=loc.shape, dtype=loc.dtype)
            self.scale.materialize(shape=scale.shape, dtype=scale.dtype)
        self.loc.data.copy_(loc)
        self.scale.data.copy_(scale)

    @staticmethod
    def _validate_bounded(action_spec: TensorSpec) -> tuple[torch.Tensor, torch.Tensor]:
        space = getattr(action_spec, "space", None)
        if not isinstance(space, ContinuousBox):
            raise RuntimeError(
                f"ActionScaling requires a bounded continuous action spec, got "
                f"{type(action_spec).__name__} with space "
                f"{type(space).__name__ if space is not None else None}. "
                "Unbounded or discrete action specs are not supported."
            )
        # ``Unbounded`` specs use a ``ContinuousBox`` whose low/high are set to
        # ``finfo.min`` and ``finfo.max`` respectively, so checking the spec type
        # is more reliable than ``torch.isfinite``.
        if isinstance(action_spec, Unbounded):
            raise RuntimeError(
                "ActionScaling cannot be used with an Unbounded action spec. "
                "The action spec must be fully bounded for spec-based normalization."
            )
        low = space.low
        high = space.high
        # Partially unbounded: one side is finite but the other matches the
        # ``finfo`` extreme used internally by ``Unbounded``.
        dtype = low.dtype
        if dtype.is_floating_point:
            extreme_low = torch.finfo(dtype).min
            extreme_high = torch.finfo(dtype).max
            if (low == extreme_low).any() or (high == extreme_high).any():
                raise RuntimeError(
                    "ActionScaling requires fully bounded actions: at least one "
                    "entry of the action spec is unbounded (low equals finfo.min or "
                    "high equals finfo.max)."
                )
        if not torch.isfinite(low).all() or not torch.isfinite(high).all():
            raise RuntimeError(
                "ActionScaling requires fully bounded actions: every entry of the "
                "action spec must have a finite lower and upper bound. Got "
                "non-finite values in low or high."
            )
        if (high <= low).any():
            raise RuntimeError(
                "ActionScaling requires high > low for every entry of the action "
                "spec. Got entries with high <= low."
            )
        return low, high

    def _transform_leaf(self, leaf_spec: TensorSpec) -> TensorSpec:
        # Validate the leaf action spec, lazily populate ``loc``/``scale`` from
        # the bounds and return a new bounded spec in normalized space.
        if not self._explicit and not self.initialized:
            self._init_from_spec(leaf_spec)
        else:
            # Still validate so that downstream users get a consistent error
            # message for unbounded / partially-unbounded specs.
            self._validate_bounded(leaf_spec)
        dtype = (
            leaf_spec.dtype
            if leaf_spec.dtype.is_floating_point
            else torch.get_default_dtype()
        )
        low = leaf_spec.space.low.to(dtype)
        high = leaf_spec.space.high.to(dtype)
        if self.standard_normal:
            new_low = torch.full_like(low, -1.0)
            new_high = torch.full_like(high, 1.0)
        else:
            new_low = torch.zeros_like(low)
            new_high = torch.ones_like(high)
        return Bounded(
            low=new_low,
            high=new_high,
            shape=leaf_spec.shape,
            device=leaf_spec.device,
            dtype=leaf_spec.dtype,
        )

    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        # We iterate the action spec manually rather than relying on
        # ``@_apply_to_composite_inv``: when ``transform_input_spec`` calls this
        # method with the action sub-composite, the decorator only inspects
        # top-level keys (``input_spec.keys(False, True)``) and silently skips
        # nested action keys. Iterating ourselves makes dict-structured action
        # spaces (e.g. ``("agent", "action")``) work like flat ones.
        if not isinstance(action_spec, Composite):
            return self._transform_leaf(action_spec)
        action_spec = action_spec.clone()
        in_key = unravel_key(self.in_keys_inv[0])
        out_key = unravel_key(self.out_keys_inv[0])
        if in_key in action_spec.keys(True, True):
            leaf = action_spec[in_key].clone()
            action_spec[out_key] = self._transform_leaf(leaf)
            if in_key != out_key:
                del action_spec[in_key]
        return action_spec

    def _loc_scale(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        # Only move the buffers when the device actually differs: an
        # unconditional ``.to()`` inserts a copy node in the compile graph,
        # whereas the device comparison is resolved at trace time (device is
        # static metadata, not data), so the common same-device path stays
        # copy-free and compile-friendly.
        loc, scale = self.loc, self.scale
        if loc.device != device:
            loc = loc.to(device)
            scale = scale.to(device)
        return loc, scale

    def _apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        self._ensure_initialized()
        loc, scale = self._loc_scale(action.device)
        normalized = (action - loc) / scale
        if not self.standard_normal:
            normalized = (normalized + 1) / 2
        return normalized

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        self._ensure_initialized()
        loc, scale = self._loc_scale(action.device)
        if not self.standard_normal:
            action = action * 2 - 1
        return action * scale + loc

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        # The action only flows through the inv direction during env stepping;
        # the ``next_tensordict`` returned by the base env does not contain it.
        # Overriding ``_call`` as a no-op avoids the default loop raising a
        # ``KeyError`` for the missing action key. The forward direction
        # (env action -> normalized) is still wired through ``forward`` /
        # ``_apply_transform`` for replay buffers and ``nn.Module`` chains.
        return next_tensordict

    def __repr__(self) -> str:
        loc = self.loc if self.initialized else "<uninitialized>"
        scale = self.scale if self.initialized else "<uninitialized>"
        return (
            f"{self.__class__.__name__}("
            f"loc={loc}, scale={scale}, standard_normal={self.standard_normal}, "
            f"in_keys_inv={self.in_keys_inv})"
        )
