# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Sequence
from copy import copy
from typing import Any, TYPE_CHECKING

import torch

from tensordict import TensorDict, TensorDictBase, unravel_key
from tensordict.utils import (
    _unravel_key_to_tuple,
    _zip_strict,
    expand_as_right,
    NestedKey,
)
from torch import Tensor

from torchrl._utils import _replace_last

from torchrl.data.tensor_specs import (
    Binary,
    Bounded,
    BoundedContinuous,
    Composite,
    TensorSpec,
    Unbounded,
    UnboundedContinuous,
)
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms.utils import _get_reset, _set_missing_tolerance

if TYPE_CHECKING:
    pass

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

from torchrl.envs.transforms._base import (
    _apply_to_composite,
    FORWARD_NOT_IMPLEMENTED,
    Transform,
)

__all__ = [
    "BinarizeReward",
    "LineariseRewards",
    "Reward2GoTransform",
    "RewardClipping",
    "RewardSum",
    "SignTransform",
    "TargetReturn",
]


class TargetReturn(Transform):
    """Sets a target return for the agent to achieve in the environment.

    In goal-conditioned RL, the :class:`~.TargetReturn` is defined as the
    expected cumulative reward obtained from the current state to the goal state
    or the end of the episode. It is used as input for the policy to guide its behavior.
    For a trained policy typically the maximum return in the environment is
    chosen as the target return.
    However, as it is used as input to the policy module, it should be scaled
    accordingly.
    With the :class:`~.TargetReturn` transform, the tensordict can be updated
    to include the user-specified target return.
    The ``mode`` parameter can be used to specify
    whether the target return gets updated at every step by subtracting the
    reward achieved at each step or remains constant.

    Args:
        target_return (:obj:`float`): target return to be achieved by the agent.
        mode (str): mode to be used to update the target return. Can be either "reduce" or "constant". Default: "reduce".
        in_keys (sequence of NestedKey, optional): keys pointing to the reward
            entries. Defaults to the reward keys of the parent env.
        out_keys (sequence of NestedKey, optional): keys pointing to the
            target keys. Defaults to a copy of in_keys where the last element
            has been substituted by ``"target_return"``, and raises an exception
            if these keys aren't unique.
        reset_key (NestedKey, optional): the reset key to be used as partial
            reset indicator. Must be unique. If not provided, defaults to the
            only reset key of the parent environment (if it has only one)
            and raises an exception otherwise.

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> env = TransformedEnv(
        ...     GymEnv("CartPole-v1"),
        ...     TargetReturn(10.0, mode="reduce"))
        >>> env.set_seed(0)
        >>> torch.manual_seed(0)
        >>> env.rollout(20)['target_return'].squeeze()
        tensor([10.,  9.,  8.,  7.,  6.,  5.,  4.,  3.,  2.,  1.,  0., -1., -2., -3.])

    """

    MODES = ["reduce", "constant"]
    MODE_ERR = "Mode can only be 'reduce' or 'constant'."

    def __init__(
        self,
        target_return: float,
        mode: str = "reduce",
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        reset_key: NestedKey | None = None,
    ):
        if mode not in self.MODES:
            raise ValueError(self.MODE_ERR)

        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.target_return = target_return
        self.mode = mode
        self.reset_key = reset_key

    @property
    def reset_key(self) -> NestedKey:
        reset_key = getattr(self, "_reset_key", None)
        if reset_key is not None:
            return reset_key
        reset_keys = self.parent.reset_keys
        if len(reset_keys) > 1:
            raise RuntimeError(
                f"Got more than one reset key in env {self.container}, cannot infer which one to use. Consider providing the reset key in the {type(self)} constructor."
            )
        reset_key = reset_keys[0]
        self._reset_key = reset_key
        return reset_key

    @reset_key.setter
    def reset_key(self, value):
        self._reset_key = value

    @property
    def in_keys(self) -> Sequence[NestedKey]:
        in_keys = self.__dict__.get("_in_keys", None)
        if in_keys is None:
            in_keys = self.parent.reward_keys
            self._in_keys = in_keys
        return in_keys

    @in_keys.setter
    def in_keys(self, value):
        self._in_keys = value

    @property
    def out_keys(self) -> Sequence[NestedKey]:
        out_keys = self.__dict__.get("_out_keys", None)
        if out_keys is None:
            out_keys = [
                _replace_last(in_key, "target_return") for in_key in self.in_keys
            ]
            if len(set(out_keys)) < len(out_keys):
                raise ValueError(
                    "Could not infer the target_return because multiple rewards are located at the same level."
                )
            self._out_keys = out_keys
        return out_keys

    @out_keys.setter
    def out_keys(self, value):
        self._out_keys = value

    def _reset(self, tensordict: TensorDict, tensordict_reset: TensorDictBase):
        _reset = _get_reset(self.reset_key, tensordict)
        for out_key in self.out_keys:
            target_return = tensordict.get(out_key, None)
            if target_return is None:
                target_return = torch.full(
                    size=(*tensordict.batch_size, 1),
                    fill_value=self.target_return,
                    dtype=torch.float32,
                    device=tensordict.device,
                )
            else:
                target_return = torch.where(
                    expand_as_right(~_reset, target_return),
                    target_return,
                    self.target_return,
                )
            tensordict_reset.set(
                out_key,
                target_return,
            )
        return tensordict_reset

    def _call(self, next_tensordict: TensorDict) -> TensorDict:
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            val_in = next_tensordict.get(in_key, None)
            val_out = next_tensordict.get(out_key, None)
            if val_in is not None:
                target_return = self._apply_transform(
                    val_in,
                    val_out,
                )
                next_tensordict.set(out_key, target_return)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {next_tensordict}")
        return next_tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        for out_key in self.out_keys:
            next_tensordict.set(out_key, tensordict.get(out_key))
        return super()._step(tensordict, next_tensordict)

    def _apply_transform(
        self, reward: torch.Tensor, target_return: torch.Tensor
    ) -> torch.Tensor:
        if target_return.shape != reward.shape:
            raise ValueError(
                f"The shape of the reward ({reward.shape}) and target return ({target_return.shape}) must match."
            )
        if self.mode == "reduce":
            target_return = target_return - reward
            return target_return
        elif self.mode == "constant":
            target_return = target_return
            return target_return
        else:
            raise ValueError(f"Unknown mode: {self.mode}")

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            FORWARD_NOT_IMPLEMENTED.format(self.__class__.__name__)
        )

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            if in_key in self.parent.full_observation_spec.keys(True):
                target = self.parent.full_observation_spec[in_key]
            elif in_key in self.parent.full_reward_spec.keys(True):
                target = self.parent.full_reward_spec[in_key]
            elif in_key in self.parent.full_done_spec.keys(True):
                # we account for this for completeness but it should never be the case
                target = self.parent.full_done_spec[in_key]
            else:
                raise RuntimeError(f"in_key {in_key} not found in output_spec.")
            target_return_spec = Unbounded(
                shape=target.shape,
                dtype=target.dtype,
                device=target.device,
            )
            # because all reward keys are discarded from the data during calls
            # to step_mdp, we must put this in observation_spec
            observation_spec[out_key] = target_return_spec
        return observation_spec

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        # we must add the target return to the input spec
        input_spec["full_state_spec"] = self.transform_observation_spec(
            input_spec["full_state_spec"]
        )
        return input_spec


class RewardClipping(Transform):
    """Clips the reward between `clamp_min` and `clamp_max`.

    Args:
        clip_min (scalar): minimum value of the resulting reward.
        clip_max (scalar): maximum value of the resulting reward.

    """

    def __init__(
        self,
        clamp_min: float | None = None,
        clamp_max: float | None = None,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = ["reward"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        clamp_min_tensor = (
            clamp_min if isinstance(clamp_min, Tensor) else torch.as_tensor(clamp_min)
        )
        clamp_max_tensor = (
            clamp_max if isinstance(clamp_max, Tensor) else torch.as_tensor(clamp_max)
        )
        self.register_buffer("clamp_min", clamp_min_tensor)
        self.register_buffer("clamp_max", clamp_max_tensor)

    def _apply_transform(self, reward: torch.Tensor) -> torch.Tensor:
        if self.clamp_max is not None and self.clamp_min is not None:
            reward = reward.clamp(self.clamp_min, self.clamp_max)
        elif self.clamp_min is not None:
            reward = reward.clamp_min(self.clamp_min)
        elif self.clamp_max is not None:
            reward = reward.clamp_max(self.clamp_max)
        return reward

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if isinstance(reward_spec, Unbounded):
            return Bounded(
                self.clamp_min,
                self.clamp_max,
                shape=reward_spec.shape,
                device=reward_spec.device,
                dtype=reward_spec.dtype,
            )
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.transform_reward_spec not "
                f"implemented for tensor spec of type"
                f" {type(reward_spec).__name__}"
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"clamp_min={float(self.clamp_min):4.4f}, clamp_max"
            f"={float(self.clamp_max):4.4f}, keys={self.in_keys})"
        )


class BinarizeReward(Transform):
    """Maps the reward to a binary value (0 or 1) if the reward is null or non-null, respectively.

    Args:
        in_keys (List[NestedKey]): input keys
        out_keys (List[NestedKey], optional): output keys. Defaults to value
            of ``in_keys``.
        dtype (torch.dtype, optional): the dtype of the binerized reward.
            Defaults to ``torch.int8``.
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = ["reward"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _apply_transform(self, reward: torch.Tensor) -> torch.Tensor:
        if not reward.shape or reward.shape[-1] != 1:
            raise RuntimeError(
                f"Reward shape last dimension must be singleton, got reward of shape {reward.shape}"
            )
        return (reward > 0.0).to(torch.int8)

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        return Binary(
            n=1,
            device=reward_spec.device,
            shape=reward_spec.shape,
        )


class RewardSum(Transform):
    """Tracks episode cumulative rewards.

    This transform accepts a list of tensordict reward keys (i.e. 'in_keys') and tracks their cumulative
    value along the time dimension for each episode.

    When called, the transform writes a new tensordict entry for each ``in_key`` named
    ``episode_{in_key}`` where the cumulative values are written.

    Args:
        in_keys (list of NestedKeys, optional): Input reward keys.
            All 'in_keys' should be part of the environment reward_spec.
            If no ``in_keys`` are specified, this transform assumes ``"reward"`` to be the input key.
            However, multiple rewards (e.g. ``"reward1"`` and ``"reward2""``) can also be specified.
        out_keys (list of NestedKeys, optional): The output sum keys, should be one per each input key.
        reset_keys (list of NestedKeys, optional): the list of reset_keys to be
            used, if the parent environment cannot be found. If provided, this
            value will prevail over the environment ``reset_keys``.

    Keyword Args:
        reward_spec (bool, optional): if ``True``, the new reward entry will be registered in the
            reward specs. Defaults to ``False`` (registered in ``observation_specs``).

    Examples:
        >>> from torchrl.envs.transforms import RewardSum, TransformedEnv
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(GymEnv("CartPole-v1"), RewardSum())
        >>> env.set_seed(0)
        >>> torch.manual_seed(0)
        >>> td = env.reset()
        >>> print(td["episode_reward"])
        tensor([0.])
        >>> td = env.rollout(3)
        >>> print(td["next", "episode_reward"])
        tensor([[1.],
                [2.],
                [3.]])
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        reset_keys: Sequence[NestedKey] | None = None,
        *,
        reward_spec: bool = False,
    ):
        """Initialises the transform. Filters out non-reward input keys and defines output keys."""
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self._reset_keys = reset_keys
        self._keys_checked = False
        self.reward_spec = reward_spec

    @property
    def in_keys(self) -> Sequence[NestedKey]:
        in_keys = self.__dict__.get("_in_keys", None)
        if in_keys in (None, []):
            # retrieve rewards from parent env
            parent = self.parent
            if parent is None:
                in_keys = ["reward"]
            else:
                in_keys = copy(parent.reward_keys)
            self._in_keys = in_keys
        return in_keys

    @in_keys.setter
    def in_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._in_keys = value

    @property
    def out_keys(self) -> Sequence[NestedKey]:
        out_keys = self.__dict__.get("_out_keys", None)
        if out_keys in (None, []):
            out_keys = [
                _replace_last(in_key, f"episode_{_unravel_key_to_tuple(in_key)[-1]}")
                for in_key in self.in_keys
            ]
            self._out_keys = out_keys
        return out_keys

    @out_keys.setter
    def out_keys(self, value):
        # we must access the private attribute because this check occurs before
        # the parent env is defined
        if value is not None and len(self._in_keys) != len(value):
            raise ValueError(
                "RewardSum expects the same number of input and output keys"
            )
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._out_keys = value

    @property
    def reset_keys(self) -> Sequence[NestedKey]:
        reset_keys = self.__dict__.get("_reset_keys", None)
        if reset_keys is None:
            parent = self.parent
            if parent is None:
                raise TypeError(
                    "reset_keys not provided but parent env not found. "
                    "Make sure that the reset_keys are provided during "
                    "construction if the transform does not have a container env."
                )
            # let's try to match the reset keys with the in_keys.
            # We take the filtered reset keys, which are the only keys that really
            # matter when calling reset, and check that they match the in_keys root.
            reset_keys = parent._filtered_reset_keys
            if len(reset_keys) == 1:
                reset_keys = list(reset_keys) * len(self.in_keys)

            def _check_match(reset_keys, in_keys):
                # if this is called, the length of reset_keys and in_keys must match
                for reset_key, in_key in _zip_strict(reset_keys, in_keys):
                    # having _reset at the root and the reward_key ("agent", "reward") is allowed
                    # but having ("agent", "_reset") and "reward" isn't
                    if isinstance(reset_key, tuple) and isinstance(in_key, str):
                        return False
                    if (
                        isinstance(reset_key, tuple)
                        and isinstance(in_key, tuple)
                        and in_key[: (len(reset_key) - 1)] != reset_key[:-1]
                    ):
                        return False
                return True

            if not _check_match(reset_keys, self.in_keys):
                raise ValueError(
                    f"Could not match the env reset_keys {reset_keys} with the {type(self)} in_keys {self.in_keys}. "
                    f"Please provide the reset_keys manually. Reset entries can be "
                    f"non-unique and must be right-expandable to the shape of "
                    f"the input entries."
                )
            reset_keys = copy(reset_keys)
            self._reset_keys = reset_keys

        if not self._keys_checked and len(reset_keys) != len(self.in_keys):
            raise ValueError(
                f"Could not match the env reset_keys {reset_keys} with the in_keys {self.in_keys}. "
                "Please make sure that these have the same length."
            )
        self._keys_checked = True

        return reset_keys

    @reset_keys.setter
    def reset_keys(self, value):
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(val) for val in value]
        self._reset_keys = value

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Resets episode rewards."""
        for in_key, reset_key, out_key in _zip_strict(
            self.in_keys, self.reset_keys, self.out_keys
        ):
            _reset = _get_reset(reset_key, tensordict)
            value = tensordict.get(out_key, default=None)
            if value is None:
                value = self.parent.full_reward_spec[in_key].zero()
            else:
                value = torch.where(expand_as_right(~_reset, value), value, 0.0)
            tensordict_reset.set(out_key, value)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """Updates the episode rewards with the step rewards."""
        # Update episode rewards
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            if in_key in next_tensordict.keys(include_nested=True):
                reward = next_tensordict.get(in_key)
                prev_reward = tensordict.get(out_key, 0.0)
                next_tensordict.set(out_key, prev_reward + reward)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")
        return next_tensordict

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        state_spec = input_spec["full_state_spec"]
        if state_spec is None:
            state_spec = Composite(shape=input_spec.shape, device=input_spec.device)
        state_spec.update(self._generate_episode_reward_spec())
        input_spec["full_state_spec"] = state_spec
        return input_spec

    def _generate_episode_reward_spec(self) -> Composite:
        episode_reward_spec = Composite()
        reward_spec = self.parent.full_reward_spec
        reward_spec_keys = self.parent.reward_keys
        # Define episode specs for all out_keys
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            if (
                in_key in reward_spec_keys
            ):  # if this out_key has a corresponding key in reward_spec
                out_key = _unravel_key_to_tuple(out_key)
                temp_episode_reward_spec = episode_reward_spec
                temp_rew_spec = reward_spec
                for sub_key in out_key[:-1]:
                    if (
                        not isinstance(temp_rew_spec, Composite)
                        or sub_key not in temp_rew_spec.keys()
                    ):
                        break
                    if sub_key not in temp_episode_reward_spec.keys():
                        temp_episode_reward_spec[sub_key] = temp_rew_spec[
                            sub_key
                        ].empty()
                    temp_rew_spec = temp_rew_spec[sub_key]
                    temp_episode_reward_spec = temp_episode_reward_spec[sub_key]
                episode_reward_spec[out_key] = reward_spec[in_key].clone()
            else:
                raise ValueError(
                    f"The in_key: {in_key} is not present in the reward spec {reward_spec}."
                )
        return episode_reward_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        """Transforms the observation spec, adding the new keys generated by RewardSum."""
        if self.reward_spec:
            return observation_spec
        if not isinstance(observation_spec, Composite):
            observation_spec = Composite(
                observation=observation_spec, shape=self.parent.batch_size
            )
        observation_spec.update(self._generate_episode_reward_spec())
        return observation_spec

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if not self.reward_spec:
            return reward_spec
        reward_spec.update(self._generate_episode_reward_spec())
        return reward_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        time_dim = [i for i, name in enumerate(tensordict.names) if name == "time"]
        if not time_dim:
            raise ValueError(
                "At least one dimension of the tensordict must be named 'time' in offline mode"
            )
        time_dim = time_dim[0] - 1
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            reward = tensordict[in_key]
            cumsum = reward.cumsum(time_dim)
            tensordict.set(out_key, cumsum)
        return tensordict


class Reward2GoTransform(Transform):
    """Calculates the reward to go based on the episode reward and a discount factor.

    As the :class:`~.Reward2GoTransform` is only an inverse transform the ``in_keys`` will be directly used for the ``in_keys_inv``.
    The reward-to-go can be only calculated once the episode is finished. Therefore, the transform should be applied to the replay buffer
    and not to the collector or within an environment.

    Args:
        gamma (:obj:`float` or torch.Tensor): the discount factor. Defaults to 1.0.
        in_keys (sequence of NestedKey): the entries to rename. Defaults to
            ``("next", "reward")`` if none is provided.
        out_keys (sequence of NestedKey): the entries to rename. Defaults to
            the values of ``in_keys`` if none is provided.
        done_key (NestedKey): the done entry. Defaults to ``"done"``.
        truncated_key (NestedKey): the truncated entry. Defaults to ``"truncated"``.
            If no truncated entry is found, only the ``"done"`` will be used.

    Examples:
        >>> # Using this transform as part of a replay buffer
        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage
        >>> torch.manual_seed(0)
        >>> r2g = Reward2GoTransform(gamma=0.99, out_keys=["reward_to_go"])
        >>> rb = ReplayBuffer(storage=LazyTensorStorage(100), transform=r2g)
        >>> batch, timesteps = 4, 5
        >>> done = torch.zeros(batch, timesteps, 1, dtype=torch.bool)
        >>> for i in range(batch):
        ...     while not done[i].any():
        ...         done[i] = done[i].bernoulli_(0.1)
        >>> reward = torch.ones(batch, timesteps, 1)
        >>> td = TensorDict(
        ...     {"next": {"done": done, "reward": reward}},
        ...     [batch, timesteps],
        ... )
        >>> rb.extend(td)
        >>> sample = rb.sample(1)
        >>> print(sample["next", "reward"])
        tensor([[[1.],
                 [1.],
                 [1.],
                 [1.],
                 [1.]]])
        >>> print(sample["reward_to_go"])
        tensor([[[4.9010],
                 [3.9404],
                 [2.9701],
                 [1.9900],
                 [1.0000]]])

    One can also use this transform directly with a collector: make sure to
    append the `inv` method of the transform.

    Examples:
        >>> from torchrl.modules import RandomPolicy        >>>         >>>         >>> from torchrl.collectors import Collector
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> t = Reward2GoTransform(gamma=0.99, out_keys=["reward_to_go"])
        >>> env = GymEnv("Pendulum-v1")
        >>> collector = Collector(
        ...     env,
        ...     RandomPolicy(env.action_spec),
        ...     frames_per_batch=200,
        ...     total_frames=-1,
        ...     postproc=t.inv
        ... )
        >>> for data in collector:
        ...     break
        >>> print(data)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                collector: TensorDict(
                    fields={
                        traj_ids: Tensor(shape=torch.Size([200]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                reward: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                reward_to_go: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([200]),
            device=cpu,
            is_shared=False)

    Using this transform as part of an env will raise an exception

    Examples:
        >>> t = Reward2GoTransform(gamma=0.99)
        >>> TransformedEnv(GymEnv("Pendulum-v1"), t)  # crashes

    .. note:: In settings where multiple done entries are present, one should build
        a single :class:`~Reward2GoTransform` for each done-reward pair.

    """

    ENV_ERR = (
        "The Reward2GoTransform is only an inverse transform and can "
        "only be applied to the replay buffer."
    )

    def __init__(
        self,
        gamma: float | torch.Tensor | None = 1.0,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        done_key: NestedKey | None = "done",
    ):
        if in_keys is None:
            in_keys = [("next", "reward")]
        if out_keys is None:
            out_keys = copy(in_keys)
        # out_keys = ["reward_to_go"]
        super().__init__(
            in_keys=in_keys,
            in_keys_inv=in_keys,
            out_keys_inv=out_keys,
        )
        self.done_key = done_key

        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma)

        self.register_buffer("gamma", gamma)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.parent is not None:
            raise ValueError(self.ENV_ERR)
        done = tensordict.get(("next", self.done_key))

        if not done.any(-2).all():
            raise RuntimeError(
                "No episode ends found to calculate the reward to go. Make sure that the number of frames_per_batch is larger than number of steps per episode."
            )
        found = False
        for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
            if in_key in tensordict.keys(include_nested=True):
                found = True
                item = self._inv_apply_transform(tensordict.get(in_key), done)
                tensordict.set(out_key, item)
        if not found:
            raise KeyError(f"Could not find any of the input keys {self.in_keys}.")
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        raise ValueError(self.ENV_ERR)

    def _inv_apply_transform(
        self, reward: torch.Tensor, done: torch.Tensor
    ) -> torch.Tensor:
        from torchrl.objectives.value.functional import reward2go

        return reward2go(reward, done, self.gamma)

    def set_container(self, container):
        if isinstance(container, EnvBase) or container.parent is not None:
            raise ValueError(self.ENV_ERR)


class SignTransform(Transform):
    """A transform to compute the signs of TensorDict values.

    This transform reads the tensors in ``in_keys`` and ``in_keys_inv``, computes the
    signs of their elements and writes the resulting sign tensors to ``out_keys`` and
    ``out_keys_inv`` respectively.

    Args:
        in_keys (list of NestedKeys): input entries (read)
        out_keys (list of NestedKeys): input entries (write)
        in_keys_inv (list of NestedKeys): input entries (read) during ``inv`` calls.
        out_keys_inv (list of NestedKeys): input entries (write) during ``inv`` calls.

    Examples:
        >>> from torchrl.envs import GymEnv, TransformedEnv, SignTransform
        >>> base_env = GymEnv("Pendulum-v1")
        >>> env = TransformedEnv(base_env, SignTransform(in_keys=['observation']))
        >>> r = env.rollout(100)
        >>> obs = r["observation"]
        >>> assert (torch.logical_or(torch.logical_or(obs == -1, obs == 1), obs == 0.0)).all()
    """

    def __init__(
        self,
        in_keys=None,
        out_keys=None,
        in_keys_inv=None,
        out_keys_inv=None,
    ):
        if in_keys is None:
            in_keys = []
        if out_keys is None:
            out_keys = copy(in_keys)
        if in_keys_inv is None:
            in_keys_inv = []
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        return obs.sign()

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        return state.sign()

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return Bounded(
            shape=observation_spec.shape,
            device=observation_spec.device,
            dtype=observation_spec.dtype,
            high=1.0,
            low=-1.0,
        )

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        for key in self.in_keys:
            if key in self.parent.reward_keys:
                spec = self.parent.output_spec["full_reward_spec"][key]
                self.parent.output_spec["full_reward_spec"][key] = Bounded(
                    shape=spec.shape,
                    device=spec.device,
                    dtype=spec.dtype,
                    high=1.0,
                    low=-1.0,
                )
        return self.parent.output_spec["full_reward_spec"]

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class LineariseRewards(Transform):
    """Transforms a multi-objective reward signal to a single-objective one via a weighted sum.

    Args:
        in_keys (List[NestedKey]): The keys under which the multi-objective rewards are found.
        out_keys (List[NestedKey], optional): The keys under which single-objective rewards should be written. Defaults to :attr:`in_keys`.
        weights (List[float], Tensor, optional): Dictates how to weight each reward when summing them. Defaults to `[1.0, 1.0, ...]`.

    .. warning::
        If a sequence of `in_keys` of length strictly greater than one is passed (e.g. one group for each agent in a
        multi-agent set-up), the same weights will be applied for each entry. If you need to aggregate rewards
        differently for each group, use several :class:`~torchrl.envs.LineariseRewards` in a row.

    Example:
        >>> import mo_gymnasium as mo_gym
        >>> from torchrl.envs import MOGymWrapper
        >>> mo_env = MOGymWrapper(mo_gym.make("deep-sea-treasure-v0"))
        >>> mo_env.reward_spec
        BoundedContinuous(
            shape=torch.Size([2]),
            space=ContinuousBox(
            low=Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, contiguous=True),
            high=Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, contiguous=True)),
            ...)
        >>> so_env = TransformedEnv(mo_env, LineariseRewards(in_keys=("reward",)))
        >>> so_env.reward_spec
        BoundedContinuous(
            shape=torch.Size([1]),
            space=ContinuousBox(
                low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
                high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
            ...)
        >>> td = so_env.rollout(5)
        >>> td["next", "reward"].shape
        torch.Size([5, 1])
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey],
        out_keys: Sequence[NestedKey] | None = None,
        *,
        weights: Sequence[float] | Tensor | None = None,
    ) -> None:
        out_keys = in_keys if out_keys is None else out_keys
        super().__init__(in_keys=in_keys, out_keys=out_keys)

        if weights is not None:
            weights = weights if isinstance(weights, Tensor) else torch.tensor(weights)

            # This transform should only receive vectorial weights (all batch dimensions will be aggregated similarly).
            if weights.ndim >= 2:
                raise ValueError(
                    f"Expected weights to be a unidimensional tensor. Got {weights.ndim} dimension."
                )

            self.register_buffer("weights", weights)
        else:
            self.weights = None

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if not reward_spec.domain == "continuous":
            raise NotImplementedError(
                "Aggregation of rewards that take discrete values is not supported."
            )

        *batch_size, num_rewards = reward_spec.shape
        weights = (
            torch.ones(num_rewards, device=reward_spec.device, dtype=reward_spec.dtype)
            if self.weights is None
            else self.weights
        )

        num_weights = torch.numel(weights)
        if num_weights != num_rewards:
            raise ValueError(
                "The number of rewards and weights should match. "
                f"Got: {num_rewards} and {num_weights}"
            )

        if isinstance(reward_spec, UnboundedContinuous):
            reward_spec.shape = torch.Size([*batch_size, 1])
            return reward_spec

        weights_pos = weights.clamp(min=0)
        weights_neg = weights.clamp(max=0)

        low_pos = (weights_pos * reward_spec.space.low).sum(dim=-1, keepdim=True)
        low_neg = (weights_neg * reward_spec.space.high).sum(dim=-1, keepdim=True)

        high_pos = (weights_pos * reward_spec.space.high).sum(dim=-1, keepdim=True)
        high_neg = (weights_neg * reward_spec.space.low).sum(dim=-1, keepdim=True)

        return BoundedContinuous(
            low=low_pos + low_neg,
            high=high_pos + high_neg,
            device=reward_spec.device,
            dtype=reward_spec.dtype,
        )

    def _apply_transform(self, reward: Tensor) -> TensorDictBase:
        if self.weights is None:
            return reward.sum(dim=-1, keepdim=True)

        *batch_size, num_rewards = reward.shape
        num_weights = torch.numel(self.weights)
        if num_weights != num_rewards:
            raise ValueError(
                "The number of rewards and weights should match. "
                f"Got: {num_rewards} and {num_weights}."
            )

        return (self.weights * reward).sum(dim=-1, keepdim=True)
