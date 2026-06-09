# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import functools
import math
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
    from torchrl.data.vla import RobotDatasetMetadata

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

from torchrl.data.vla.schema import (
    ACTION_CHUNK_KEY,
    ACTION_IS_PAD_KEY,
    ACTION_KEY,
    ACTION_TOKENS_KEY,
)
from torchrl.data.vla.tokenizers import ActionTokenizerBase
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms._base import FORWARD_NOT_IMPLEMENTED, Transform

__all__ = [
    "ActionChunkTransform",
    "ActionDiscretizer",
    "ActionMask",
    "ActionNormalize",
    "ActionScaling",
    "ActionTokenizerTransform",
    "DiscreteActionProjection",
    "FlattenAction",
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

    .. seealso:: :class:`~torchrl.envs.transforms.ActionNormalize` -- the
        forward-only, dataset-statistics-driven action normalizer for the
        offline / replay-buffer training path (with an explicit ``denormalize``
        for execution), as opposed to this bidirectional, spec-coupled env-side
        scaler.
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


class FlattenAction(Transform):
    """Flatten adjacent dimensions of an action.

    Mirrors :class:`~torchrl.envs.transforms.FlattenObservation`, but applies
    to actions: the policy sees a flattened action space and the original
    multi-dimensional shape is restored on the inv direction before the action
    is passed to the base environment.

    On the inv direction (policy -> env), a 1-D ``flattened`` action is
    unflattened to the original ``(dim_first, ..., dim_last)`` span of the env
    action. On the forward direction (env action -> flattened, used inside
    replay buffers and :class:`~torch.nn.Module` chains), the adjacent dims
    ``[first_dim, last_dim]`` are flattened.

    Args:
        first_dim (int): first dimension to flatten. Must be negative unless
            ``allow_positive_dim`` is ``True``.
        last_dim (int): last dimension to flatten (inclusive). Must be negative
            unless ``allow_positive_dim`` is ``True``.
        in_keys_inv (sequence of NestedKey, optional): keys read during the
            ``inv`` direction (policy -> env). Defaults to ``["action"]``.
            Multiple keys are supported - the same flatten span is applied to
            each one, which is useful for dict-structured action spaces.
        out_keys_inv (sequence of NestedKey, optional): keys written during the
            ``inv`` direction. Defaults to ``in_keys_inv``.
        in_keys (sequence of NestedKey, optional): keys read during the forward
            direction (env action -> flattened). Defaults to ``in_keys_inv``.
        out_keys (sequence of NestedKey, optional): keys written during the
            forward direction. Defaults to ``in_keys``.
        allow_positive_dim (bool, optional): if ``True``, positive dimensions
            are accepted. Defaults to ``False`` so that the same transform
            works regardless of the parent environment's batch size.

    Keyword Args:
        action_shape (sequence of int, optional): explicit pre-flatten shape
            of the dimensions ``[first_dim, last_dim]``. Useful when the
            transform is used outside a :class:`TransformedEnv` (e.g. inside
            a replay buffer) and the original action shape cannot be derived
            from a parent env. The same span is applied to every entry of
            ``in_keys_inv``. Defaults to ``None``, in which case the shape is
            derived lazily from the parent env's action spec.

    Examples:
        >>> import torch
        >>> from torchrl.data.tensor_specs import Bounded
        >>> from torchrl.envs.transforms import FlattenAction, TransformedEnv
        >>> from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv
        >>> base_env = ContinuousActionVecMockEnv(
        ...     action_spec=Bounded(low=-1.0, high=1.0, shape=(3, 5))
        ... )
        >>> env = TransformedEnv(base_env, FlattenAction(first_dim=-2, last_dim=-1))
        >>> env.action_spec.shape
        torch.Size([15])
    """

    invertible = True

    def __init__(
        self,
        first_dim: int = -2,
        last_dim: int = -1,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        allow_positive_dim: bool = False,
        *,
        action_shape: Sequence[int] | None = None,
    ):
        if in_keys_inv is None:
            in_keys_inv = ["action"]
        if not isinstance(in_keys_inv, (list, tuple)):
            in_keys_inv = [in_keys_inv]
        if out_keys_inv is None:
            out_keys_inv = copy(list(in_keys_inv))
        if in_keys is None:
            in_keys = copy(list(in_keys_inv))
        if out_keys is None:
            out_keys = copy(list(in_keys))
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )
        if not allow_positive_dim and first_dim >= 0:
            raise ValueError(
                "first_dim should be smaller than 0 to accommodate for "
                "envs of different batch_sizes. Set allow_positive_dim=True "
                "to allow positive dimensions."
            )
        if not allow_positive_dim and last_dim >= 0:
            raise ValueError(
                "last_dim should be smaller than 0 to accommodate for "
                "envs of different batch_sizes. Set allow_positive_dim=True "
                "to allow positive dimensions."
            )
        if first_dim > last_dim:
            raise ValueError(
                f"first_dim ({first_dim}) must be <= last_dim ({last_dim})."
            )
        self._first_dim = first_dim
        self._last_dim = last_dim
        self.allow_positive_dim = bool(allow_positive_dim)
        # Per-action-key original (pre-flatten) span, populated from the spec
        # or seeded from the ``action_shape`` constructor kwarg.
        self._unflatten_shapes: dict[NestedKey, tuple[int, ...]] = {}
        if action_shape is not None:
            action_shape = tuple(int(s) for s in action_shape)
            for in_key in self.in_keys_inv:
                self._unflatten_shapes[unravel_key(in_key)] = action_shape

    @property
    def first_dim(self) -> int:
        if self._first_dim >= 0 and self.parent is not None:
            return len(self.parent.batch_size) + self._first_dim
        return self._first_dim

    @property
    def last_dim(self) -> int:
        if self._last_dim >= 0 and self.parent is not None:
            return len(self.parent.batch_size) + self._last_dim
        return self._last_dim

    @property
    def _flat_merged_dim(self) -> int:
        # Index of the merged dim in the post-flatten tensor. The flat tensor
        # has ``(last - first)`` fewer dims than the original. For positive
        # ``first_dim``, the merged dim sits at exactly ``first_dim``. For
        # negative ``first_dim``, ``last_dim`` (also negative) already points
        # at the merged dim in the new, shorter tensor, because
        # ``last_dim - first_dim`` dims were collapsed strictly to its left.
        if self._first_dim >= 0:
            return self.first_dim
        return self.last_dim

    def _apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        # env-scale action -> flattened
        return torch.flatten(action, self.first_dim, self.last_dim)

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        # flattened action -> env-scale (unflatten)
        self._ensure_unflatten_shapes()
        # ``_inv_apply_transform`` only receives a tensor, with no information
        # about which ``in_keys_inv`` entry it came from. For multi-key
        # transforms we cannot disambiguate, so we route those through
        # ``_inv_call`` (which knows the key) and raise here. Single-key
        # instances are unambiguous and remain supported.
        if len(self.in_keys_inv) != 1:
            raise RuntimeError(
                f"FlattenAction._inv_apply_transform cannot disambiguate "
                f"between {len(self.in_keys_inv)} action keys. Use "
                f"``FlattenAction.inv(td)`` / ``_inv_call(td)`` instead, which "
                f"know which key each tensor belongs to."
            )
        in_key = unravel_key(self.in_keys_inv[0])
        shape = self._unflatten_shapes.get(in_key)
        if shape is None:
            raise RuntimeError(
                f"FlattenAction has no stored unflatten shape for key "
                f"{in_key!r}. Pass ``action_shape`` to the constructor or "
                f"attach the transform to a TransformedEnv with a bounded "
                f"action spec for this key."
            )
        return torch.unflatten(action, self._flat_merged_dim, shape)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Route each action key to its own unflatten span using the per-key
        # state computed at ``transform_action_spec`` time.
        if not self.in_keys_inv:
            return tensordict
        self._ensure_unflatten_shapes()
        flat_dim = self._flat_merged_dim
        for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
            in_key_u = unravel_key(in_key)
            out_key_u = unravel_key(out_key)
            data = tensordict.get(out_key_u, default=None)
            if data is None:
                if not self.missing_tolerance:
                    raise KeyError(
                        f"'{out_key_u}' not found in tensordict {tensordict}"
                    )
                continue
            shape = self._unflatten_shapes.get(in_key_u)
            if shape is None:
                raise RuntimeError(
                    f"FlattenAction has no stored unflatten shape for key "
                    f"{in_key_u!r}. Pass ``action_shape`` to the constructor "
                    f"or attach the transform to a TransformedEnv with a "
                    f"bounded action spec for this key."
                )
            tensordict.set(in_key_u, torch.unflatten(data, flat_dim, shape))
        return tensordict

    def _ensure_unflatten_shapes(self) -> None:
        # Lazily populate ``_unflatten_shapes`` from the parent env's action
        # spec at the insertion point of this transform. ``self.parent`` is
        # rebuilt with all transforms up to (but not including) ``self``, so
        # its action spec is exactly the env-scale spec we need to read. If
        # ``action_shape`` was provided at construction time this is a no-op.
        if self._unflatten_shapes:
            return
        parent = self.parent
        if parent is None:
            return
        full_action_spec = parent.full_action_spec
        for in_key in self.in_keys_inv:
            in_key = unravel_key(in_key)
            if in_key in full_action_spec.keys(True, True):
                self._unflatten_shapes[in_key] = self._span_from_spec(
                    full_action_spec[in_key]
                )

    def _span_from_spec(self, leaf_spec: TensorSpec) -> tuple[int, ...]:
        ndim = len(leaf_spec.shape)
        first = self._first_dim
        last = self._last_dim
        if first < 0:
            first = ndim + first
        if last < 0:
            last = ndim + last
        if first < 0 or last >= ndim or first > last:
            raise RuntimeError(
                f"FlattenAction(first_dim={self._first_dim}, last_dim={self._last_dim}) "
                f"is not compatible with an action of shape {tuple(leaf_spec.shape)}."
            )
        return tuple(int(s) for s in leaf_spec.shape[first : last + 1])

    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        # Manual iteration so that nested action keys are supported - the
        # generic ``_apply_to_composite_inv`` decorator only iterates top-level
        # keys when it is called with the full action spec directly.
        if not isinstance(action_spec, Composite):
            self._unflatten_shapes[
                unravel_key(self.in_keys_inv[0])
            ] = self._span_from_spec(action_spec)
            return self._flatten_leaf(action_spec)
        action_spec = action_spec.clone()
        for in_key, out_key in zip(self.in_keys_inv, self.out_keys_inv):
            in_key_u = unravel_key(in_key)
            out_key_u = unravel_key(out_key)
            if in_key_u in action_spec.keys(True, True):
                leaf = action_spec[in_key_u].clone()
                self._unflatten_shapes[in_key_u] = self._span_from_spec(leaf)
                action_spec[out_key_u] = self._flatten_leaf(leaf)
                if in_key_u != out_key_u:
                    del action_spec[in_key_u]
        return action_spec

    def _flatten_leaf(self, leaf_spec: TensorSpec) -> TensorSpec:
        space = getattr(leaf_spec, "space", None)
        if isinstance(space, ContinuousBox):
            new_low = torch.flatten(space.low, self.first_dim, self.last_dim)
            new_high = torch.flatten(space.high, self.first_dim, self.last_dim)
            return Bounded(
                low=new_low,
                high=new_high,
                shape=new_low.shape,
                device=leaf_spec.device,
                dtype=leaf_spec.dtype,
            )
        shape = list(leaf_spec.shape)
        ndim = len(shape)
        first = self.first_dim if self.first_dim >= 0 else ndim + self.first_dim
        last = self.last_dim if self.last_dim >= 0 else ndim + self.last_dim
        flat = math.prod(shape[first : last + 1])
        new_shape = torch.Size((*shape[:first], flat, *shape[last + 1 :]))
        leaf_spec = leaf_spec.clone()
        leaf_spec.shape = new_shape
        return leaf_spec

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        # The action does not appear in ``next_tensordict`` during env
        # stepping, so we leave it untouched here. The default ``_call`` loop
        # would otherwise raise ``KeyError`` for the missing action key. The
        # ``forward`` path used by replay buffers is still wired through
        # ``_apply_transform``.
        return next_tensordict

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"first_dim={int(self.first_dim)}, last_dim={int(self.last_dim)}, "
            f"in_keys_inv={self.in_keys_inv}, out_keys_inv={self.out_keys_inv})"
        )


class ActionChunkTransform(Transform):
    """Build fixed-length action chunks from a trajectory window.

    Action *chunking* is the defining trait of modern VLA policies (ACT,
    OpenVLA-OFT, pi0, SmolVLA): instead of predicting a single action, the
    policy predicts a short horizon ``H`` of future actions. This transform
    turns a per-step action tensor ``[*B, T, action_dim]`` into the
    corresponding training target ``action_chunk`` of shape
    ``[*B, T, H, action_dim]`` -- for each time step ``t`` it gathers the
    actions ``a[t], a[t+1], ..., a[t+H-1]`` -- together with a boolean
    ``action_is_pad`` mask ``[*B, T, H]`` marking the steps that ran past the
    end of the window (and were filled by repeating the last available action).

    It is a replay-buffer / offline transform that operates on **time-structured**
    data: the action tensor must be shaped ``[*B, T, action_dim]`` and each row
    along ``time_dim`` must be a single contiguous trajectory window. Chunks are
    built independently per row and never cross a row boundary; the downstream
    chunked behavior-cloning loss masks the padded steps out using
    ``action_is_pad``.

    .. note::
        A :class:`~torchrl.data.SliceSampler` returns a *flat* ``[B * T, ...]``
        batch -- reshape it to ``[num_slices, slice_len, ...]`` before applying
        this transform, otherwise chunks would span across trajectory boundaries.
        Datasets that store one trajectory window per item (e.g.
        :class:`~torchrl.data.datasets.OpenXExperienceReplay`) already yield
        time-structured ``[batch, T, ...]`` samples and can use this transform
        directly. This transform cannot be used as a
        :class:`~torchrl.envs.TransformedEnv` transform.

    Args:
        chunk_size (int): the horizon ``H`` of the action chunk.

    Keyword Args:
        action_key (NestedKey): the per-step action to read.
            Defaults to ``"action"``.
        chunk_key (NestedKey): where to write the action chunk.
            Defaults to ``"action_chunk"``.
        pad_key (NestedKey): where to write the padding mask.
            Defaults to ``"action_is_pad"``.
        time_dim (int): the time dimension of the action tensor (the action
            dimension must come right after it). Defaults to ``-2``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.transforms import ActionChunkTransform
        >>> t = ActionChunkTransform(chunk_size=3)
        >>> td = TensorDict(
        ...     {"action": torch.arange(4).view(1, 4, 1).float()}, batch_size=[1, 4]
        ... )
        >>> td = t(td)
        >>> td["action_chunk"].shape
        torch.Size([1, 4, 3, 1])
        >>> td["action_chunk"][0, :, :, 0]
        tensor([[0., 1., 2.],
                [1., 2., 3.],
                [2., 3., 3.],
                [3., 3., 3.]])
        >>> td["action_is_pad"][0]
        tensor([[False, False, False],
                [False, False, False],
                [False, False,  True],
                [False,  True,  True]])
    """

    ENV_ERR = (
        "ActionChunkTransform is a replay-buffer / offline transform and cannot "
        "be used as an environment transform."
    )

    def __init__(
        self,
        chunk_size: int,
        *,
        action_key: NestedKey = ACTION_KEY,
        chunk_key: NestedKey = ACTION_CHUNK_KEY,
        pad_key: NestedKey = ACTION_IS_PAD_KEY,
        time_dim: int = -2,
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        self.chunk_size = int(chunk_size)
        self.time_dim = int(time_dim)
        super().__init__(in_keys=[action_key], out_keys=[chunk_key, pad_key])

    @property
    def action_key(self) -> NestedKey:
        return self.in_keys[0]

    @property
    def chunk_key(self) -> NestedKey:
        return self.out_keys[0]

    @property
    def pad_key(self) -> NestedKey:
        return self.out_keys[1]

    def _build_chunk(self, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        H = self.chunk_size
        dim = self.time_dim if self.time_dim >= 0 else action.dim() + self.time_dim
        if dim != action.dim() - 2:
            raise ValueError(
                f"{type(self).__name__} expects the action dimension to immediately "
                f"follow the time dimension (action shaped [..., T, action_dim]); got "
                f"action.shape={tuple(action.shape)} with time_dim={self.time_dim}."
            )
        T = action.shape[dim]
        device = action.device
        # idx[t, h] = t + h, clamped to the last valid step; is_pad marks h that
        # ran past the end of the window.
        idx = torch.arange(T, device=device).unsqueeze(-1) + torch.arange(
            H, device=device
        ).unsqueeze(0)
        is_pad = idx >= T
        idx = idx.clamp_max(T - 1).reshape(-1)
        chunk = action.index_select(dim, idx).unflatten(dim, (T, H))
        is_pad = is_pad.expand(chunk.shape[:-1]).contiguous()
        return chunk, is_pad

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key, default=None)
        if action is None:
            if self.missing_tolerance:
                return tensordict
            raise KeyError(
                f"{type(self).__name__}: '{self.action_key}' not found in tensordict "
                f"{tensordict}."
            )
        chunk, is_pad = self._build_chunk(action)
        tensordict.set(self.chunk_key, chunk)
        tensordict.set(self.pad_key, is_pad)
        return tensordict

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        raise ValueError(self.ENV_ERR)

    def set_container(self, container) -> None:
        if (
            isinstance(container, EnvBase)
            or getattr(container, "parent", None) is not None
        ):
            raise ValueError(self.ENV_ERR)
        super().set_container(container)


class ActionNormalize(Transform):
    """Affine action normalization for VLA training and execution.

    The action-space analogue of :class:`~torchrl.envs.transforms.ObservationNorm`.
    On the forward (data) path it normalizes expert actions, so the policy is
    trained to predict normalized actions::

        normalized = (action - loc) / scale

    A policy operating in this normalized space is mapped back to raw actions
    with :meth:`denormalize` (the inverse affine map
    ``action = normalized * scale + loc``) before they are sent to a simulator
    or robot.

    ``loc`` and ``scale`` are per-dimension statistics. Use :meth:`from_stats`
    (mean/std or min/max) or :meth:`from_metadata` to build them from a
    :class:`~torchrl.data.vla.RobotDatasetMetadata`.

    .. note::
        This is a replay-buffer / offline transform: only the forward direction
        is wired, so it normalizes on ``sample`` and is a no-op on ``extend``,
        and it cannot be used as a :class:`~torchrl.envs.TransformedEnv`
        transform. Use :meth:`denormalize` to map a policy's predicted action
        back to the raw action space for execution. For env-side action
        normalization derived from a bounded action spec, use
        :class:`~torchrl.envs.transforms.ActionScaling` instead.

    Args:
        loc (float or torch.Tensor): per-dimension location (e.g. action mean).
        scale (float or torch.Tensor): per-dimension scale (e.g. action std).

    Keyword Args:
        action_key (NestedKey): the action to normalize. Defaults to ``"action"``.
        out_key (NestedKey, optional): where to write the normalized action.
            Defaults to ``action_key`` (in place).
        eps (float): floor applied to ``scale`` to avoid division by zero.
            Defaults to ``1e-6``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.transforms import ActionNormalize
        >>> t = ActionNormalize(loc=torch.tensor([1.0, 2.0]), scale=torch.tensor([2.0, 4.0]))
        >>> td = t(TensorDict({"action": torch.tensor([[3.0, 6.0]])}, batch_size=[1]))
        >>> td["action"]
        tensor([[1., 1.]])
        >>> t.denormalize(td["action"])
        tensor([[3., 6.]])

    .. seealso::
        :class:`~torchrl.envs.transforms.ObservationNorm` (the observation-side
        affine normalizer) and :class:`~torchrl.envs.transforms.ActionScaling`
        (the bidirectional, spec-coupled env-side action scaler). Use
        ``ActionNormalize`` for the forward-only, dataset-statistics-driven
        training path with an explicit :meth:`denormalize` for execution; use
        ``ActionScaling`` to expose a normalized action space to a policy in a
        :class:`~torchrl.envs.TransformedEnv`.
    """

    ENV_ERR = (
        "ActionNormalize is a replay-buffer / offline transform and cannot be "
        "used as an environment transform. Use ActionNormalize.denormalize() on "
        "the execution path, or ActionScaling for env-side action normalization."
    )

    def __init__(
        self,
        loc: float | torch.Tensor,
        scale: float | torch.Tensor,
        *,
        action_key: NestedKey = ACTION_KEY,
        out_key: NestedKey | None = None,
        eps: float = 1e-6,
    ) -> None:
        loc = torch.as_tensor(loc, dtype=torch.float32)
        scale = torch.as_tensor(scale, dtype=torch.float32).clamp_min(eps)
        if loc.shape != scale.shape:
            raise ValueError(
                f"loc and scale must have the same shape, got {tuple(loc.shape)} "
                f"and {tuple(scale.shape)}."
            )
        out_key = action_key if out_key is None else out_key
        super().__init__(in_keys=[action_key], out_keys=[out_key])
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    @property
    def action_key(self) -> NestedKey:
        return self.in_keys[0]

    def _check_dim(self, action: torch.Tensor) -> None:
        # Guard the silent-broadcast hazard: a per-dimension loc/scale must
        # match the action's trailing dim. A scalar (or shape-[1]) loc/scale
        # broadcasts freely and is left alone.
        if self.loc.numel() > 1 and action.shape[-1] != self.loc.shape[-1]:
            raise ValueError(
                f"action last dim {action.shape[-1]} does not match the "
                f"loc/scale dimension {self.loc.shape[-1]}."
            )

    def _apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        self._check_dim(action)
        return (action - self.loc) / self.scale

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        self._check_dim(action)
        return action * self.scale + self.loc

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        raise ValueError(self.ENV_ERR)

    def set_container(self, container) -> None:
        if (
            isinstance(container, EnvBase)
            or getattr(container, "parent", None) is not None
        ):
            raise ValueError(self.ENV_ERR)
        super().set_container(container)

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        """Map a raw action to the normalized space ``(action - loc) / scale``."""
        return self._apply_transform(action)

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        """Map a normalized action back to raw actions ``action * scale + loc``."""
        return self._inv_apply_transform(action)

    @classmethod
    def from_stats(
        cls,
        *,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        low: torch.Tensor | None = None,
        high: torch.Tensor | None = None,
        **kwargs,
    ) -> ActionNormalize:
        """Build from mean/std (zero-mean, unit-std) or min/max (maps to ``[-1, 1]``).

        Provide exactly one complete pair: ``mean`` and ``std``, or ``low`` and
        ``high``.
        """
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be provided together.")
        if (low is None) != (high is None):
            raise ValueError("low and high must be provided together.")
        if (mean is not None) == (low is not None):
            raise ValueError("Provide exactly one of (mean, std) or (low, high).")
        if mean is not None:
            loc = torch.as_tensor(mean, dtype=torch.float32)
            scale = torch.as_tensor(std, dtype=torch.float32)
        else:
            low = torch.as_tensor(low, dtype=torch.float32)
            high = torch.as_tensor(high, dtype=torch.float32)
            loc = (low + high) / 2
            scale = (high - low) / 2
        return cls(loc, scale, **kwargs)

    @classmethod
    def from_metadata(cls, metadata: RobotDatasetMetadata, **kwargs) -> ActionNormalize:
        """Build from the action statistics of a :class:`~torchrl.data.vla.RobotDatasetMetadata`."""
        kwargs.setdefault("action_key", metadata.action_key)
        if metadata.action_mean is not None and metadata.action_std is not None:
            return cls.from_stats(
                mean=metadata.action_mean, std=metadata.action_std, **kwargs
            )
        if metadata.action_low is not None and metadata.action_high is not None:
            return cls.from_stats(
                low=metadata.action_low, high=metadata.action_high, **kwargs
            )
        raise ValueError(
            f"metadata {metadata.dataset_id!r} has no action normalization statistics "
            "(set action_mean/action_std or action_low/action_high)."
        )


class ActionTokenizerTransform(Transform):
    """Encode continuous actions into discrete tokens with an action tokenizer.

    A replay-buffer / offline transform that wraps an
    :class:`~torchrl.data.vla.ActionTokenizerBase`: on the forward path it
    encodes the continuous action (or action chunk) at ``in_key`` into discrete
    token ids at ``out_key`` -- the training target / input for an
    autoregressive (RT-2 / OpenVLA-style) token VLA. Decoding tokens back to
    continuous actions is done with the wrapped tokenizer's ``decode`` (the
    transform itself is forward-only).

    Args:
        tokenizer (ActionTokenizerBase): the tokenizer to apply.

    Keyword Args:
        in_key (NestedKey): the continuous action to encode.
            Defaults to ``"action"``.
        out_key (NestedKey): where to write the token ids.
            Defaults to ``"action_tokens"``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data.vla import UniformActionTokenizer
        >>> from torchrl.envs.transforms import ActionTokenizerTransform
        >>> tok = UniformActionTokenizer(256, low=-1.0, high=1.0)
        >>> t = ActionTokenizerTransform(tok)
        >>> td = t(TensorDict({"action": torch.tensor([[-1.0, 0.0, 1.0]])}, batch_size=[1]))
        >>> td["action_tokens"]
        tensor([[  0, 128, 255]])
    """

    ENV_ERR = (
        "ActionTokenizerTransform is a replay-buffer / offline transform and "
        "cannot be used as an environment transform."
    )

    def __init__(
        self,
        tokenizer: ActionTokenizerBase,
        *,
        in_key: NestedKey = ACTION_KEY,
        out_key: NestedKey = ACTION_TOKENS_KEY,
    ) -> None:
        if not isinstance(tokenizer, ActionTokenizerBase):
            raise TypeError(
                f"tokenizer must be an ActionTokenizerBase, got {type(tokenizer)}."
            )
        super().__init__(in_keys=[in_key], out_keys=[out_key])
        self.tokenizer = tokenizer

    @property
    def in_key(self) -> NestedKey:
        return self.in_keys[0]

    @property
    def out_key(self) -> NestedKey:
        return self.out_keys[0]

    def _apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        return self.tokenizer.encode(action)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        raise ValueError(self.ENV_ERR)

    def set_container(self, container) -> None:
        if (
            isinstance(container, EnvBase)
            or getattr(container, "parent", None) is not None
        ):
            raise ValueError(self.ENV_ERR)
        super().set_container(container)
