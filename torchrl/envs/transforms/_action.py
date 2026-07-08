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
from typing import Any, Literal, TYPE_CHECKING

import torch

from tensordict import TensorDict, TensorDictBase
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
from torchrl.envs.transforms._base import Compose, FORWARD_NOT_IMPLEMENTED, Transform
from torchrl.envs.transforms._observation import CatFrames, UnsqueezeTransform

__all__ = [
    "ActionChunkTransform",
    "ActionDiscretizer",
    "ActionMask",
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

    .. seealso:: :class:`~torchrl.envs.transforms.ActionTokenizerTransform` -- a
        bidirectional action <-> token codec built around an explicit
        :class:`~torchrl.data.vla.ActionTokenizerBase`. Prefer
        ``ActionDiscretizer`` when the binning should be derived from the env's
        bounded ``action_spec`` (with configurable in-bin sampling); prefer
        ``ActionTokenizerTransform`` when the binning is owned by a tokenizer
        that must be shared between offline encoding (replay buffer) and online
        decoding (env), e.g. for an autoregressive token VLA policy.
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

    .. note:: Extra entries written by the policy alongside the actions (e.g. the action tokens and
        log-probabilities of a token-head policy) are left untouched on the root tensordict and therefore
        ride along on the outer (macro-step) transition: each outer step of a rollout carries the policy
        outputs of the chunk decided at that step.

    .. note:: When a done state fires inside the chunk (with ``stack_rewards=True``), the reward stack of
        that outer step holds the executed steps' rewards followed by a single zero-filled slot for the
        skipped remainder of the chunk. Its length therefore differs from a full chunk's, and stacking
        such outer steps in a rollout yields a lazy stack with ragged reward entries. If the per-chunk
        reward is computed from the outer transition anyway (e.g. with
        :class:`~torchrl.envs.transforms.SuccessReward` appended after this transform), pass
        ``stack_rewards=False`` to keep the outer transition dense and uniform.

    .. note:: Skipping the remaining steps after a done state relies on the ``"_step"`` partial-step
        entry. Single (unbatched) environments and batched environments
        (:class:`~torchrl.envs.SerialEnv` / :class:`~torchrl.envs.ParallelEnv`) handle it natively; for a
        batch-locked vectorized environment, the base environment's ``_step`` is trusted to honor the
        mask itself (see :meth:`~torchrl.envs.EnvBase.step`) and environments that ignore it will keep
        stepping every sub-environment until the end of the chunk.

    Keyword Args:
        dim (int, optional): the stack dimension with respect to the tensordict ``ndim`` attribute.
            Must be greater than 0. Defaults to ``1`` (the first dimension after the batch-dims).
        stack_rewards (bool, optional): if ``True``, each step's reward will be stack in the output tensordict.
            If ``False``, only the last reward will be returned. The reward spec is adapted accordingly. The
            stack dimension is the same as the action stack dimension. Defaults to ``True``.
        stack_observations (bool, optional): if ``True``, each step's observation will be stack in the output tensordict.
            If ``False``, only the last observation will be returned. The observation spec is adapted accordingly. The
            stack dimension is the same as the action stack dimension. Defaults to ``False``.
        action_key (NestedKey, optional): the one-step action key consumed by
            the base environment. Defaults to the parent environment action key.
        chunk_key (NestedKey, optional): the policy-facing key that holds the
            stacked actions. Defaults to ``action_key`` for backward
            compatibility. Set this to values such as
            ``("vla_action", "chunk")`` when a chunk policy should act through
            :class:`MultiAction` without re-keying its output. See also
            :meth:`from_vla`.

    .. seealso:: :class:`~torchrl.envs.transforms.ActionChunkTransform` -- when
        the stacked actions are a chunk policy's *prediction* (overlapping
        per-step training targets) rather than a macro action to replay
        verbatim. The chunk transform builds the training targets on the data
        path and, attached to an env, executes only the first action of each
        predicted chunk (re-planning at every step) instead of stepping the
        base env once per action.
    """

    def __init__(
        self,
        *,
        dim: int = 1,
        stack_rewards: bool = True,
        stack_observations: bool = False,
        action_key: NestedKey | None = None,
        chunk_key: NestedKey | None = None,
    ):
        if action_key is None and chunk_key is not None:
            action_key = "action"
        if action_key is not None and chunk_key is None:
            chunk_key = action_key
        in_keys_inv = None if action_key is None else [action_key]
        out_keys_inv = None if chunk_key is None else [chunk_key]
        super().__init__(in_keys_inv=in_keys_inv, out_keys_inv=out_keys_inv)
        self.stack_rewards = stack_rewards
        self.stack_observations = stack_observations
        self.dim = dim

    @classmethod
    def from_vla(cls, *, action_key: NestedKey = ACTION_KEY, **kwargs) -> MultiAction:
        """Build a :class:`MultiAction` that consumes the default VLA chunk key.

        Args:
            action_key (NestedKey): the one-step action key consumed by the base
                environment. Defaults to ``"action"``.

        Keyword Args:
            Additional :class:`MultiAction` keyword arguments.

        Examples:
            >>> from torchrl.envs.transforms import MultiAction
            >>> transform = MultiAction.from_vla(stack_rewards=False)
            >>> transform.out_keys_inv
            [('vla_action', 'chunk')]
        """
        return cls(action_key=action_key, chunk_key=ACTION_CHUNK_KEY, **kwargs)

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
        action_keys = self.in_keys_inv or parent.action_keys
        chunk_keys = self.out_keys_inv or action_keys
        if len(action_keys) != len(chunk_keys):
            raise ValueError(
                "action_key and chunk_key lists must have the same length, got "
                f"{len(action_keys)} and {len(chunk_keys)}."
            )
        actions = tensordict.empty()
        for action_key, chunk_key in zip(action_keys, chunk_keys):
            action = tensordict.get(chunk_key, None)
            if action is None:
                raise KeyError(
                    f"{type(self).__name__} expected stacked actions at key "
                    f"{chunk_key!r} before env.step, but the key was missing. "
                    "For VLA policies, use MultiAction.from_vla() or pass "
                    "chunk_key=('vla_action', 'chunk'). Available keys are "
                    f"{list(tensordict.keys(True, True))}."
                )
            actions.set(action_key, action)
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
                        # td_out's root reward/observation tensors alias the
                        # entries just appended to the stacks: de-alias them so
                        # the masked writes into td_out below do not corrupt
                        # the stacked history
                        keys = []
                        if self.stack_rewards:
                            keys += list(self.parent.reward_keys)
                        if self.stack_observations:
                            keys += list(self.parent.observation_keys)
                        for key in keys:
                            td_out.set(key, td_out.get(key).clone())
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
                if self.stack_rewards:
                    # the final outer step is skipped for every env (done fired
                    # inside the chunk): its slot in the reward stack would
                    # otherwise carry the stale reward of the last executed
                    # step - zero it, matching the zero-fill of the other
                    # skipped slots
                    for key in self.parent.reward_keys:
                        td_out.set(key, torch.zeros_like(td_out.get(key)))
                td_out = self._step(None, td_out)
        else:
            td_out[global_idx] = td.replace(actions[-1][global_idx])
            if self.stack_rewards:
                # zero the trailing reward slot of the envs that finished
                # early: their final outer step is skipped, so it would
                # otherwise carry the stale reward of their last executed step
                for key in self.parent.reward_keys:
                    reward = td_out.get(key).clone()
                    reward[~global_idx] = 0
                    td_out.set(key, reward)
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
        action_keys = self.in_keys_inv or list(action_spec.keys(True, True))
        chunk_keys = self.out_keys_inv or action_keys
        if len(action_keys) != len(chunk_keys):
            raise ValueError(
                "action_key and chunk_key lists must have the same length, got "
                f"{len(action_keys)} and {len(chunk_keys)}."
            )
        for action_key, chunk_key in zip(action_keys, chunk_keys):
            leaf_spec = action_spec[action_key]
            for _ in range(self.dim):
                leaf_spec = leaf_spec.unsqueeze(input_spec.ndim)
            # Make the dim dynamic
            leaf_spec = leaf_spec.expand(
                tuple(
                    d if i != (input_spec.ndim + self.dim - 1) else -1
                    for i, d in enumerate(leaf_spec.shape)
                )
            )
            action_spec[chunk_key] = leaf_spec
            if chunk_key != action_key:
                del action_spec[action_key]
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
            several instances to scale several actions. Pass an empty list for
            a forward-only transform (normalize raw dataset actions on the
            replay-buffer sample path while leaving ``extend`` and the env-side
            action interface untouched); this requires explicit ``loc`` and
            ``scale``.
        out_keys_inv (sequence of NestedKey, optional): keys written during the
            ``inv`` direction. Defaults to ``in_keys_inv``.
        in_keys (sequence of NestedKey, optional): keys read during the forward
            direction (env action -> normalized action, used by replay buffers
            and inside :class:`~torch.nn.Module` chains). Defaults to
            ``in_keys_inv``, or ``["action"]`` when ``in_keys_inv=[]``
            (forward-only mode).
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
        RuntimeError: if ``loc`` and ``scale`` are derived from the spec (no
            explicit values passed) and the action spec is unbounded or
            partially unbounded (any bound is non-finite). With explicit
            ``loc``/``scale``, a bounded spec is mapped through the affine
            transform and an unbounded (or partially unbounded) spec is
            advertised as ``Unbounded`` instead of raising.

    With explicit ``loc`` and ``scale`` the transform is fully spec-independent
    -- the standard workflow when training on dataset action statistics, e.g.
    for VLA policies. Use :meth:`from_stats` (``mean``/``std`` or
    ``low``/``high``) or :meth:`from_metadata` to build such an instance from
    dataset statistics. Attached to an environment, it denormalizes the
    policy's actions on the inverse path: a bounded action spec is mapped
    through the affine transform (and an unbounded action spec stays
    unbounded), so the advertised normalized space reflects the actual
    statistics rather than being assumed ``[-1, 1]``. Appended to a replay
    buffer, it normalizes actions on the ``sample`` path; beware that
    ``ReplayBuffer.extend`` applies the *inverse* transform, so when raw
    (env-scale) data is written through ``extend``, use a forward-only
    instance (``in_keys_inv=[]``) to leave the stored data untouched -- the
    default bidirectional keys suit the env side and pre-populated dataset
    storages.

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
        >>> # dataset-statistics-driven normalization (no env required): the
        >>> # forward pass maps raw actions to the normalized space
        >>> from tensordict import TensorDict
        >>> t = ActionScaling.from_stats(
        ...     mean=torch.tensor([1.0, 2.0]), std=torch.tensor([2.0, 4.0])
        ... )
        >>> td = TensorDict({"action": torch.tensor([[3.0, 6.0]])}, batch_size=[1])
        >>> t(td)["action"]
        tensor([[1., 1.]])
        >>> # on a replay buffer, a forward-only instance (in_keys_inv=[])
        >>> # normalizes on sample and leaves data written through extend
        >>> # untouched (extend applies the inverse pass)
        >>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
        >>> t = ActionScaling.from_stats(
        ...     mean=torch.tensor([1.0, 2.0]),
        ...     std=torch.tensor([2.0, 4.0]),
        ...     in_keys_inv=[],
        ... )
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyTensorStorage(10), transform=t, batch_size=2
        ... )
        >>> raw = TensorDict(
        ...     {"action": torch.tensor([[3.0, 6.0]]).expand(10, 2)}, batch_size=[10]
        ... )
        >>> indices = rb.extend(raw)  # stored as-is
        >>> rb.sample()["action"]  # normalized with the dataset statistics
        tensor([[1., 1.],
                [1., 1.]])
        >>> # the same affine map is exposed on raw tensors for execution-time
        >>> # use, e.g. mapping a policy's normalized prediction to the robot
        >>> t.denormalize(torch.tensor([[1.0, 1.0]]))
        tensor([[3., 6.]])
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
        if len(in_keys_inv) > 1:
            raise ValueError(
                "ActionScaling only supports a single action key per instance. "
                "Compose several ActionScaling transforms to scale multiple actions."
            )
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)
        if in_keys is None:
            # Forward-only mode (``in_keys_inv=[]``) still normalizes "action"
            # on the forward (sample) path by default.
            in_keys = copy(in_keys_inv) if in_keys_inv else ["action"]
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
        if not in_keys_inv and not self._explicit:
            raise ValueError(
                "in_keys_inv=[] (forward-only mode) requires explicit loc and "
                "scale: without an inverse action key there is no action spec "
                "to derive them from."
            )
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

    @staticmethod
    def _is_finitely_bounded(leaf_spec: TensorSpec) -> bool:
        # ``Unbounded`` (and partially-unbounded ``Bounded``) specs encode the
        # open sides with ``finfo`` extremes; mapping those through the affine
        # would overflow, so they are treated as unbounded instead.
        if isinstance(leaf_spec, Unbounded):
            return False
        low, high = leaf_spec.space.low, leaf_spec.space.high
        if low.dtype.is_floating_point:
            extreme_low = torch.finfo(low.dtype).min
            extreme_high = torch.finfo(high.dtype).max
            if (low == extreme_low).any() or (high == extreme_high).any():
                return False
        return bool(torch.isfinite(low).all() and torch.isfinite(high).all())

    def _transform_leaf(self, leaf_spec: TensorSpec) -> TensorSpec:
        dtype = (
            leaf_spec.dtype
            if leaf_spec.dtype.is_floating_point
            else torch.get_default_dtype()
        )
        if self._explicit:
            # Explicit loc/scale: no bounds are required from the spec. A
            # bounded spec is mapped through the forward affine (monotonic,
            # scale > 0); an unbounded (or partially unbounded) spec stays
            # unbounded, since the affine image of an unbounded space is
            # unbounded.
            space = getattr(leaf_spec, "space", None)
            if not isinstance(space, ContinuousBox):
                raise RuntimeError(
                    f"ActionScaling requires a continuous action spec, got "
                    f"{type(leaf_spec).__name__}. Discrete action specs are "
                    "not supported."
                )
            if not self._is_finitely_bounded(leaf_spec):
                return Unbounded(
                    shape=leaf_spec.shape,
                    device=leaf_spec.device,
                    dtype=leaf_spec.dtype,
                )
            loc, scale = self._loc_scale(space.low.device)
            new_low = (space.low.to(dtype) - loc) / scale
            new_high = (space.high.to(dtype) - loc) / scale
            if not self.standard_normal:
                new_low = (new_low + 1) / 2
                new_high = (new_high + 1) / 2
        else:
            # Spec-derived loc/scale: bounds are mandatory and define the
            # normalized space exactly ([-1, 1] or [0, 1]).
            if not self.initialized:
                self._init_from_spec(leaf_spec)
            else:
                self._validate_bounded(leaf_spec)
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
        if not self.in_keys_inv:
            # Forward-only mode: the env-side action interface is untouched.
            return action_spec
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

    def _check_dim(self, action: torch.Tensor) -> None:
        # Guard the silent-broadcast hazard: a per-dimension loc/scale must
        # match the action's trailing dim. A scalar (or shape-[1]) loc/scale
        # broadcasts freely and is left alone.
        if self.loc.numel() > 1 and (
            action.ndim == 0 or action.shape[-1] != self.loc.shape[-1]
        ):
            raise ValueError(
                f"action shape {tuple(action.shape)} does not match the "
                f"loc/scale dimension {self.loc.shape[-1]} on its last dim."
            )

    def _apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        self._ensure_initialized()
        self._check_dim(action)
        loc, scale = self._loc_scale(action.device)
        normalized = (action - loc) / scale
        if not self.standard_normal:
            normalized = (normalized + 1) / 2
        return normalized

    def _inv_apply_transform(self, action: torch.Tensor) -> torch.Tensor:
        self._ensure_initialized()
        self._check_dim(action)
        loc, scale = self._loc_scale(action.device)
        if not self.standard_normal:
            action = action * 2 - 1
        return action * scale + loc

    def normalize(self, action: torch.Tensor) -> torch.Tensor:
        """Map an env-scale action to the normalized space (the forward map)."""
        return self._apply_transform(action)

    def denormalize(self, action: torch.Tensor) -> torch.Tensor:
        """Map a normalized action back to the env scale (the inverse map)."""
        return self._inv_apply_transform(action)

    @classmethod
    def from_stats(
        cls,
        *,
        mean: torch.Tensor | None = None,
        std: torch.Tensor | None = None,
        low: torch.Tensor | None = None,
        high: torch.Tensor | None = None,
        eps: float = 1e-6,
        **kwargs,
    ) -> ActionScaling:
        """Build an :class:`ActionScaling` from dataset action statistics.

        Provide exactly one complete pair: ``mean`` and ``std`` (zero-mean,
        unit-std normalized space) or ``low`` and ``high`` (maps the range to
        ``[-1, 1]``).

        Keyword Args:
            mean (torch.Tensor, optional): per-dimension action mean.
            std (torch.Tensor, optional): per-dimension action std.
            low (torch.Tensor, optional): per-dimension action minimum.
            high (torch.Tensor, optional): per-dimension action maximum.
            eps (float, optional): floor applied to the scale to avoid division
                by zero on constant action dimensions. Defaults to ``1e-6``.
            **kwargs: forwarded to the constructor (e.g. ``in_keys_inv``,
                ``standard_normal``).
        """
        if (mean is None) != (std is None):
            raise ValueError("mean and std must be provided together.")
        if (low is None) != (high is None):
            raise ValueError("low and high must be provided together.")
        if (mean is not None) == (low is not None):
            raise ValueError("Provide exactly one of (mean, std) or (low, high).")
        if mean is not None:
            loc = torch.as_tensor(mean, dtype=torch.get_default_dtype())
            scale = torch.as_tensor(std, dtype=torch.get_default_dtype())
        else:
            low = torch.as_tensor(low, dtype=torch.get_default_dtype())
            high = torch.as_tensor(high, dtype=torch.get_default_dtype())
            loc = (low + high) / 2
            scale = (high - low) / 2
        if loc.shape != scale.shape:
            raise ValueError(
                f"loc and scale must have the same shape, got {tuple(loc.shape)} "
                f"and {tuple(scale.shape)}."
            )
        return cls(loc=loc, scale=scale.clamp_min(eps), **kwargs)

    @classmethod
    def from_metadata(cls, metadata: RobotDatasetMetadata, **kwargs) -> ActionScaling:
        """Build from the action statistics of a :class:`~torchrl.data.vla.RobotDatasetMetadata`.

        Uses ``action_mean``/``action_std`` when available, falling back to
        ``action_low``/``action_high``. The action key defaults to the
        metadata's ``action_key``.
        """
        kwargs.setdefault("in_keys_inv", [metadata.action_key])
        if not kwargs["in_keys_inv"]:
            # Forward-only mode: keep normalizing the metadata's action key on
            # the sample path rather than falling back to the generic "action".
            kwargs.setdefault("in_keys", [metadata.action_key])
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


class ActionChunkTransform(Compose):
    """Build fixed-length action chunks from a trajectory window.

    Action *chunking* is the defining trait of modern VLA policies (ACT,
    OpenVLA-OFT, pi0, SmolVLA): instead of predicting a single action, the
    policy predicts a short horizon ``H`` of future actions. This transform
    turns a per-step action tensor ``[*B, T, action_dim]`` into the
    corresponding training target ``("vla_action", "chunk")`` of shape
    ``[*B, T, H, action_dim]`` -- for each time step ``t`` it gathers the
    actions ``a[t], a[t+1], ..., a[t+H-1]`` -- together with a boolean
    ``action_is_pad`` mask ``[*B, T, H]`` marking the steps that ran past the
    end of the window (and were filled by repeating the last available action).

    Internally this is a recipe over the generic transforms (the same pattern
    as :class:`~torchrl.envs.transforms.R3MTransform`): an
    :class:`~torchrl.envs.transforms.UnsqueezeTransform` opens the chunk dim
    and a forward-looking :class:`~torchrl.envs.transforms.CatFrames`
    (``future=True, padding="same", mask_key=...``) does the windowing, so
    chunking shares one sliding-window implementation with frame stacking.

    .. versionchanged:: 0.14
        ``ActionChunkTransform`` is now a :class:`~torchrl.envs.transforms.Compose`
        recipe over :class:`~torchrl.envs.transforms.CatFrames`. The output is
        unchanged, and additionally chunks become *boundary-aware* when the
        sampled data carries its done state (see ``done_key``).

    .. note:: **How to read "many actions in one tensor".** The ``H`` actions
        of a chunk are *predictions* -- overlapping, stride-1 training targets
        (each dataset step ``t`` gets its own window ``a[t..t+H-1]``, so a
        given action appears in up to ``H`` different chunks) -- not a macro
        action to be replayed verbatim. This transform is a pure *data*
        transform (it builds training targets) and never touches the
        environment; how many of the ``H`` predicted actions actually get
        executed per policy call is a separate, execution-time choice:

        - :class:`~torchrl.envs.transforms.MultiAction` executes every action
          in the tensor by stepping the base env once per action with a
          single policy call per chunk (one outer step = ``H`` base steps,
          rewards stacked or aggregated);
        - :class:`~torchrl.modules.tensordict_module.MultiStepActorWrapper`
          keeps the env timing unchanged: it caches the predicted actions and
          emits one per step, skipping the actor call while the cache lasts
          -- open-loop by default, receding horizon with
          ``replan_interval < n_steps``, closed loop with
          ``replan_interval=1``.

    The forward (data) path operates on **time-structured** data: the action
    tensor must be shaped ``[*B, T, action_dim]`` and each row along
    ``time_dim`` must be a single contiguous trajectory window. Chunks are
    built independently per row and never cross a row boundary; the downstream
    chunked behavior-cloning loss masks the padded steps out using
    ``action_is_pad``. When the input additionally carries its done state at
    ``("next", done_key)``, chunks are also cut at the trajectory boundaries
    *inside* a row: the steps past a done are padded (repeating the last
    in-trajectory action) and flagged in ``action_is_pad``, exactly like the
    end of the window.

    .. note::
        A :class:`~torchrl.data.SliceSampler` returns a *flat* ``[B * T, ...]``
        batch -- reshape it to ``[num_slices, slice_len, ...]`` before applying
        this transform, otherwise chunks would span across trajectory boundaries.
        Datasets that store one trajectory window per item (e.g.
        :class:`~torchrl.data.datasets.OpenXExperienceReplay`) already yield
        time-structured ``[batch, T, ...]`` samples and can use this transform
        directly. When this transform is appended to a replay buffer, the
        chunks are built on the ``sample`` path only; ``extend`` leaves the
        stored (raw, per-step) data untouched.

    Args:
        chunk_size (int): the horizon ``H`` of the action chunk.

    Keyword Args:
        action_key (NestedKey): the per-step action to read.
            Defaults to ``"action"``.
        chunk_key (NestedKey): where to write the action chunk.
            Defaults to ``("vla_action", "chunk")``.
        pad_key (NestedKey): where to write the padding mask.
            Defaults to ``"action_is_pad"``.
        time_dim (int): the time dimension of the action tensor (the action
            dimension must come right after it). Defaults to ``-2``.
        done_key (NestedKey or None): the leaf done key: when the input
            tensordict has a ``("next", done_key)`` entry (shaped like the
            action without its trailing ``action_dim``, with or without a
            trailing singleton), chunks do not cross the trajectory boundaries
            it marks. When the entry is absent, each row is treated as a
            single contiguous trajectory (the pre-0.14 behavior). Pass
            ``None`` to ignore the done state altogether.
            Defaults to ``"done"``.

            .. versionadded:: 0.14

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.transforms import ActionChunkTransform
        >>> # for each step t the chunk gathers a[t], a[t+1], a[t+2], repeating
        >>> # the last action past the end of the window (masked by action_is_pad)
        >>> t = ActionChunkTransform(chunk_size=3)
        >>> td = TensorDict(
        ...     {"action": torch.arange(4).view(1, 4, 1).float()}, batch_size=[1, 4]
        ... )
        >>> td = t(td)
        >>> td["vla_action", "chunk"][0, :, :, 0]
        tensor([[0., 1., 2.],
                [1., 2., 3.],
                [2., 3., 3.],
                [3., 3., 3.]])
        >>> td["action_is_pad"][0]
        tensor([[False, False, False],
                [False, False, False],
                [False, False,  True],
                [False,  True,  True]])
        >>> # when the window carries its done state, chunks are also cut at
        >>> # the trajectory boundary inside the window (here after step 1)
        >>> td = TensorDict(
        ...     {
        ...         "action": torch.arange(4).view(1, 4, 1).float(),
        ...         ("next", "done"): torch.tensor(
        ...             [False, True, False, False]
        ...         ).view(1, 4, 1),
        ...     },
        ...     batch_size=[1, 4],
        ... )
        >>> t(td)["vla_action", "chunk"][0, :, :, 0]
        tensor([[0., 1., 1.],
                [1., 1., 1.],
                [2., 3., 3.],
                [3., 3., 3.]])
        >>> # on a replay buffer: extend with raw [T, action_dim] trajectory
        >>> # windows (stored as-is), the chunks are built on the sample path
        >>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyTensorStorage(8),
        ...     transform=ActionChunkTransform(chunk_size=3),
        ...     batch_size=2,
        ... )
        >>> windows = TensorDict(
        ...     {"action": torch.randn(8, 4, 1)}, batch_size=[8]
        ... )  # 8 trajectory windows of T=4 steps each
        >>> indices = rb.extend(windows)
        >>> rb.sample()["vla_action", "chunk"].shape  # [batch, T, chunk_size, action_dim]
        torch.Size([2, 4, 3, 1])
    """

    def __init__(
        self,
        chunk_size: int,
        *,
        action_key: NestedKey = ACTION_KEY,
        chunk_key: NestedKey = ACTION_CHUNK_KEY,
        pad_key: NestedKey = ACTION_IS_PAD_KEY,
        time_dim: int = -2,
        done_key: NestedKey | None = "done",
    ) -> None:
        if chunk_size < 1:
            raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")
        # The recipe: open a singleton chunk dim on a copy of the action, then
        # concatenate the N upcoming actions along it. ``padding="same"``
        # repeats the last in-trajectory action past the boundaries and
        # ``mask_key`` flags those fabricated slots.
        super().__init__(
            UnsqueezeTransform(dim=-2, in_keys=[action_key], out_keys=[chunk_key]),
            CatFrames(
                N=int(chunk_size),
                dim=-2,
                in_keys=[chunk_key],
                out_keys=[chunk_key],
                padding="same",
                future=True,
                mask_key=pad_key,
                done_key=done_key,
            ),
        )
        self.chunk_size = int(chunk_size)
        self.time_dim = int(time_dim)
        self._action_key = action_key
        self._chunk_key = chunk_key
        self._pad_key = pad_key
        self._done_key = done_key

    @property
    def action_key(self) -> NestedKey:
        return self._action_key

    @property
    def chunk_key(self) -> NestedKey:
        return self._chunk_key

    @property
    def pad_key(self) -> NestedKey:
        return self._pad_key

    @property
    def done_key(self) -> NestedKey | None:
        return self._done_key

    def _maybe_get_done(
        self, tensordict: TensorDictBase, action: torch.Tensor, dim: int
    ) -> torch.Tensor | None:
        if self._done_key is None:
            return None
        done = tensordict.get(("next", self._done_key), default=None)
        if done is None:
            return None
        lead = action.shape[: dim + 1]
        if done.shape == lead:
            done = done.unsqueeze(-1)
        elif done.shape != torch.Size((*lead, 1)):
            raise ValueError(
                f"{type(self).__name__}: the ('next', {self._done_key!r}) entry "
                f"of shape {tuple(done.shape)} does not line up with the action "
                f"of shape {tuple(action.shape)}: expected {(*lead, 1)} or "
                f"{tuple(lead)}. Pass done_key=None to chunk without "
                "trajectory-boundary information."
            )
        return done

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = tensordict.get(self.action_key, default=None)
        if action is None:
            if self.missing_tolerance:
                return tensordict
            raise KeyError(
                f"{type(self).__name__}: '{self.action_key}' not found in tensordict "
                f"{tensordict}."
            )
        dim = self.time_dim if self.time_dim >= 0 else action.dim() + self.time_dim
        if dim != action.dim() - 2 or dim < 0:
            raise ValueError(
                f"{type(self).__name__} expects the action dimension to immediately "
                f"follow the time dimension (action shaped [..., T, action_dim]); got "
                f"action.shape={tuple(action.shape)} with time_dim={self.time_dim}."
            )
        # CatFrames' offline path keys the windowing on the *tensordict* batch
        # dims (time last), while the chunk convention is keyed on the action's
        # shape ([*B, T, action_dim]) -- the sampled tensordict may well be
        # flat. Bridge the two by windowing a minimal time-structured view of
        # the action (and of the done state, when available).
        inner = TensorDict(batch_size=action.shape[: dim + 1])
        inner.set(self.action_key, action)
        done = self._maybe_get_done(tensordict, action, dim)
        if done is not None:
            inner.set(("next", self._done_key), done)
        inner = inner.refine_names(*[None] * dim, "time")
        inner = super().forward(inner)
        tensordict.set(self.chunk_key, inner.get(self.chunk_key))
        tensordict.set(self.pad_key, inner.get(self.pad_key))
        return tensordict

    def clone(self) -> Self:
        # Compose.clone returns a plain Compose, which would drop the
        # env-path overrides below; rebuild the recipe instead.
        return type(self)(
            self.chunk_size,
            action_key=self._action_key,
            chunk_key=self._chunk_key,
            pad_key=self._pad_key,
            time_dim=self.time_dim,
            done_key=self._done_key,
        )

    # Attached to an environment, the transform is a documented no-op: chunk
    # execution belongs to MultiStepActorWrapper / MultiAction, and the inner
    # CatFrames is offline-only (future=True). The Compose machinery is
    # bypassed on every env-facing path.
    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        return next_tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return tensordict_reset

    def _reset_on_native_autoreset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        return tensordict_reset

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        return input_spec

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        return output_spec


class ActionTokenizerTransform(Transform):
    """Encode and decode actions with an :class:`~torchrl.data.vla.ActionTokenizerBase`.

    A bidirectional action <-> token codec wrapping an action tokenizer (the
    bins live in the tokenizer; no environment is needed to construct it).
    Like any TorchRL transform it plugs onto a replay buffer or an environment
    interchangeably:

    - **forward encode mode** (``mode="encode"``, the default): maps the
      continuous action (or action chunk) at ``in_key`` to discrete token ids at
      ``out_key`` -- e.g. building the token training target for an
      autoregressive (RT-2 / OpenVLA-style) token VLA on the replay-buffer
      sample path.
    - **inverse encode mode**: maps token ids at ``out_key`` back to a
      continuous action at ``in_key`` -- e.g. decoding the tokens a token-head
      policy emits, on the environment action-input path, before the base env
      consumes them.
    - **forward decode mode** (``mode="decode"``): maps token ids at
      ``out_key`` to continuous actions at ``in_key``. This is useful on the
      policy side, for instance as a module after a token VLA policy in a
      :class:`~tensordict.nn.TensorDictSequential`, so CPU environments can
      receive decoded actions without owning tokenizer buffers.

    On a replay buffer the inverse is a no-op when the token entry is absent,
    so extending with raw (untokenized) data is safe; attached to an
    environment, missing tokens on the step path raise instead.

    When attached to an environment in encode mode, the policy-facing action
    spec is rewritten to a :class:`~torchrl.data.Categorical` over the
    tokenizer's vocabulary, so the env advertises the token interface the
    policy is expected to produce (the decoded continuous action is consumed by
    the base env internally). Using the same tokenizer instance on the replay
    buffer (encode) and on the env (decode through the inverse path) guarantees
    that training targets and execution share the exact same binning.

    Args:
        tokenizer (ActionTokenizerBase): the tokenizer to apply.

    Keyword Args:
        in_key (NestedKey): the continuous action. Defaults to ``"action"``.
        out_key (NestedKey): the discrete token ids. Defaults to
            ``("vla_action", "tokens")``. Pass ``"action_tokens"`` for the
            flat compatibility key.
        mode (str, optional): ``"encode"`` makes :meth:`forward` encode
            actions into tokens and :meth:`inv` decode tokens into actions.
            ``"decode"`` swaps these directions. Defaults to ``"encode"``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data.vla import UniformActionTokenizer
        >>> from torchrl.envs.transforms import ActionTokenizerTransform
        >>> tok = UniformActionTokenizer(256, low=-1.0, high=1.0)
        >>> t = ActionTokenizerTransform(tok)
        >>> td = t(TensorDict({"action": torch.tensor([[-1.0, 0.0, 1.0]])}, batch_size=[1]))
        >>> td["vla_action", "tokens"]
        tensor([[  0, 128, 255]])
        >>> # the inverse decodes tokens back to a continuous action
        >>> back = t.inv(TensorDict({("vla_action", "tokens"): td["vla_action", "tokens"]}, batch_size=[1]))
        >>> back["action"].shape
        torch.Size([1, 3])
        >>> # policy-side decode: token policy -> decoded continuous action
        >>> decode = ActionTokenizerTransform(tok, mode="decode")
        >>> policy_td = TensorDict({("vla_action", "tokens"): td["vla_action", "tokens"]}, batch_size=[1])
        >>> decode(policy_td)["action"].shape
        torch.Size([1, 3])
        >>> # on a replay buffer: raw actions written through extend are stored
        >>> # as-is and tokenized on the sample path
        >>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyTensorStorage(8),
        ...     transform=ActionTokenizerTransform(tok),
        ...     batch_size=2,
        ... )
        >>> indices = rb.extend(
        ...     TensorDict({"action": torch.rand(8, 3) * 2 - 1}, batch_size=[8])
        ... )
        >>> rb.sample()["vla_action", "tokens"].shape
        torch.Size([2, 3])
        >>> # on an environment: the policy-facing action spec becomes the token
        >>> # interface, and emitted tokens are decoded before the base env
        >>> # consumes them
        >>> from torchrl.envs import GymEnv, TransformedEnv
        >>> tok_env = UniformActionTokenizer(256, low=-2.0, high=2.0)  # Pendulum bounds
        >>> env = TransformedEnv(GymEnv("Pendulum-v1"), ActionTokenizerTransform(tok_env))
        >>> env.full_action_spec["vla_action", "tokens"].shape
        torch.Size([1])
        >>> env.rollout(2)["vla_action", "tokens"].dtype
        torch.int64

    .. seealso:: :class:`~torchrl.envs.transforms.ActionDiscretizer` -- the
        env-only discretizer that derives its bins from the environment's
        bounded ``action_spec`` (with configurable in-bin sampling strategies)
        so a discrete-action policy can act on a continuous env. Use
        ``ActionDiscretizer`` when the binning should follow the env spec; use
        ``ActionTokenizerTransform`` when the binning is owned by a tokenizer
        (dataset statistics, FAST/DCT-style codecs) that must be shared between
        offline encoding and online decoding.
    """

    def __init__(
        self,
        tokenizer: ActionTokenizerBase,
        *,
        in_key: NestedKey = ACTION_KEY,
        out_key: NestedKey = ACTION_TOKENS_KEY,
        mode: Literal["encode", "decode"] = "encode",
    ) -> None:
        if not isinstance(tokenizer, ActionTokenizerBase):
            raise TypeError(
                f"tokenizer must be an ActionTokenizerBase, got {type(tokenizer)}."
            )
        if mode not in ("encode", "decode"):
            raise ValueError(f"mode must be either 'encode' or 'decode', got {mode!r}.")
        action_key = unravel_key(in_key)
        token_key = unravel_key(out_key)
        # ``forward`` is fully overridden (encode in_key -> out_key on the data
        # path), so no forward keys are declared: the token entry only exists
        # on the data path, never in the env's output specs. The inverse
        # direction reads ``out_keys_inv`` (the tokens) and writes
        # ``in_keys_inv`` (the action passed to the base env).
        if mode == "encode":
            in_keys = []
            out_keys = []
            in_keys_inv = [action_key]
            out_keys_inv = [token_key]
        else:
            in_keys = [token_key]
            out_keys = [action_key]
            in_keys_inv = [token_key]
            out_keys_inv = [action_key]
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )
        self.tokenizer = tokenizer
        self.mode = mode
        self._action_key = action_key
        self._token_key = token_key

    @property
    def in_key(self) -> NestedKey:
        return self._action_key

    @property
    def out_key(self) -> NestedKey:
        return self._token_key

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.mode == "encode":
            source_key = self.in_key
            dest_key = self.out_key
            transform = self.tokenizer.encode
        else:
            source_key = self.out_key
            dest_key = self.in_key
            transform = self.tokenizer.decode
        value = tensordict.get(source_key, default=None)
        if value is None:
            if self.missing_tolerance:
                return tensordict
            raise KeyError(
                f"{type(self).__name__}: '{source_key}' not found in tensordict "
                f"{tensordict}."
            )
        tensordict.set(dest_key, transform(value))
        return tensordict

    def _inv_apply_transform(self, tokens: torch.Tensor) -> torch.Tensor:
        if self.mode == "encode":
            return self.tokenizer.decode(tokens)
        return self.tokenizer.encode(tokens)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        # This transform acts on data/replay-buffer samples through ``forward``
        # and on env actions through ``inv``. When used as a policy-side decode
        # module, ``in_keys``/``out_keys`` are populated for
        # TensorDictSequential introspection, but an attached env should not try
        # to decode observations on reset/step.
        return next_tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Without a parent env (replay-buffer ``extend``), raw (untokenized)
        # data passes through untouched. Attached to an env, missing tokens on
        # the step path are an error (the rewritten action spec advertises
        # them; a policy writing the raw action key by mistake should not
        # silently bypass the decode). The env reset path never reaches the
        # inverse: ``enable_inv_on_reset`` defaults to False.
        expected_key = self.out_key if self.mode == "encode" else self.in_key
        if self.parent is None and tensordict.get(expected_key, default=None) is None:
            return tensordict
        return super()._inv_call(tensordict)

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        if self.mode == "decode":
            return input_spec
        # Expose the token interface to the policy: replace the base env's
        # continuous action spec with a Categorical over the tokenizer
        # vocabulary. The continuous spec is removed rather than moved to the
        # state spec: the decoded action is written on the inverse path's
        # working copy and consumed by the base env, so it never surfaces in
        # the outer tensordict (which keeps single and batched envs
        # consistent).
        action_key = unravel_key(self.in_key)
        token_key = unravel_key(self.out_key)
        try:
            leaf_spec = input_spec["full_action_spec", action_key]
        except KeyError:
            raise RuntimeError(
                f"{type(self).__name__} could not find key {action_key!r} in "
                f"the parent environment's action spec. Available keys: "
                f"{list(input_spec['full_action_spec'].keys(True, True))}."
            )
        # the incoming spec is batched and may carry dynamic (-1) dims (e.g.
        # MultiAction's chunk dim, which full_action_spec_unbatched would
        # mangle); dynamic dims cannot be passed to the constructor, so build
        # with placeholder dims and expand
        shape = leaf_spec.shape
        concrete = torch.Size([1 if dim < 0 else dim for dim in shape])
        token_spec = Categorical(
            n=self.tokenizer.vocab_size,
            shape=concrete,
            device=leaf_spec.device,
            dtype=torch.long,
        )
        if concrete != shape:
            token_spec = token_spec.expand(shape)
        input_spec["full_action_spec", token_key] = token_spec
        if token_key != action_key:
            del input_spec["full_action_spec", action_key]
        return input_spec

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        if self.mode == "decode":
            return output_spec
        return super().transform_output_spec(output_spec)
