# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from copy import copy
from typing import Any, TYPE_CHECKING

from tensordict import (
    is_tensor_collection,
    NonTensorData,
    TensorDictBase,
    unravel_key,
    unravel_key_list,
)
from tensordict.utils import _zip_strict, NestedKey

from torchrl.data.tensor_specs import Composite, TensorSpec
from torchrl.envs.transforms.utils import _set_missing_tolerance

if TYPE_CHECKING:
    pass

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

from torchrl.envs.transforms._base import Transform

__all__ = [
    "ExcludeTransform",
    "FlattenTensorDict",
    "RemoveEmptySpecs",
    "RenameTransform",
    "SelectTransform",
]


class ExcludeTransform(Transform):
    """Excludes keys from the data.

    Args:
        *excluded_keys (iterable of NestedKey): The name of the keys to exclude. If the key is
            not present, it is simply ignored.
        inverse (bool, optional): if ``True``, the exclusion will occur during the ``inv`` call.
            Defaults to ``False``.

    Examples:
        >>> import gymnasium
        >>> from torchrl.envs import GymWrapper
        >>> env = TransformedEnv(
        ...     GymWrapper(gymnasium.make("Pendulum-v1")),
        ...     ExcludeTransform("truncated")
        ... )
        >>> env.rollout(3)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

    """

    def __init__(self, *excluded_keys, inverse: bool = False):
        super().__init__()
        try:
            excluded_keys = unravel_key_list(excluded_keys)
        except TypeError:
            raise TypeError(
                "excluded keys must be a list or tuple of strings or tuples of strings."
            )
        self.excluded_keys = excluded_keys
        self.inverse = inverse

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if not self.inverse:
            return next_tensordict.exclude(*self.excluded_keys)
        return next_tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.inverse:
            return tensordict.exclude(*self.excluded_keys)
        return tensordict

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        if not self.inverse:
            return tensordict_reset.exclude(*self.excluded_keys)
        return tensordict

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        if not self.inverse:
            full_done_spec = output_spec["full_done_spec"]
            full_reward_spec = output_spec["full_reward_spec"]
            full_observation_spec = output_spec["full_observation_spec"]
            for key in self.excluded_keys:
                # done_spec
                if unravel_key(key) in list(full_done_spec.keys(True, True)):
                    del full_done_spec[key]
                    continue
                # reward_spec
                if unravel_key(key) in list(full_reward_spec.keys(True, True)):
                    del full_reward_spec[key]
                    continue
                # observation_spec
                if unravel_key(key) in list(full_observation_spec.keys(True, True)):
                    del full_observation_spec[key]
                    continue
                raise KeyError(f"Key {key} not found in the environment outputs.")
        return output_spec


class SelectTransform(Transform):
    """Select keys from the input tensordict.

    In general, the :obj:`ExcludeTransform` should be preferred: this transforms also
        selects the "action" (or other keys from input_spec), "done" and "reward"
        keys but other may be necessary.

    Args:
        *selected_keys (iterable of NestedKey): The name of the keys to select. If the key is
            not present, it is simply ignored.

    Keyword Args:
        keep_rewards (bool, optional): if ``False``, the reward keys must be provided
            if they should be kept. Defaults to ``True``.
        keep_dones (bool, optional): if ``False``, the done keys must be provided
            if they should be kept. Defaults to ``True``.

    Examples:
        >>> import gymnasium
        >>> from torchrl.envs import GymWrapper
        >>> env = TransformedEnv(
        ...     GymWrapper(gymnasium.make("Pendulum-v1")),
        ...     SelectTransform("observation", "reward", "done", keep_dones=False), # we leave done behind
        ... )
        >>> env.rollout(3)  # the truncated key is now absent
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)

    """

    def __init__(
        self,
        *selected_keys: NestedKey,
        keep_rewards: bool = True,
        keep_dones: bool = True,
    ):
        super().__init__()
        try:
            selected_keys = unravel_key_list(selected_keys)
        except TypeError:
            raise TypeError(
                "selected keys must be a list or tuple of strings or tuples of strings."
            )
        self.selected_keys = selected_keys
        self.keep_done_keys = keep_dones
        self.keep_reward_keys = keep_rewards

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if self.parent is not None:
            input_keys = self.parent.state_spec.keys(True, True)
        else:
            input_keys = []
        if self.keep_reward_keys:
            reward_keys = self.parent.reward_keys if self.parent else ["reward"]
        else:
            reward_keys = []
        if self.keep_done_keys:
            done_keys = self.parent.done_keys if self.parent else ["done"]
        else:
            done_keys = []
        return next_tensordict.select(
            *self.selected_keys, *reward_keys, *done_keys, *input_keys, strict=False
        )

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        if self.parent is not None:
            input_keys = self.parent.state_spec.keys(True, True)
        else:
            input_keys = []
        if self.keep_reward_keys:
            reward_keys = self.parent.reward_keys if self.parent else ["reward"]
        else:
            reward_keys = []
        if self.keep_done_keys:
            done_keys = self.parent.done_keys if self.parent else ["done"]
        else:
            done_keys = []
        return tensordict_reset.select(
            *self.selected_keys, *reward_keys, *done_keys, *input_keys, strict=False
        )

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        full_done_spec = output_spec["full_done_spec"]
        full_reward_spec = output_spec["full_reward_spec"]
        full_observation_spec = output_spec["full_observation_spec"]
        if not self.keep_done_keys:
            for key in list(full_done_spec.keys(True, True)):
                if unravel_key(key) not in self.selected_keys:
                    del full_done_spec[key]

        for key in list(full_observation_spec.keys(True, True)):
            if unravel_key(key) not in self.selected_keys:
                del full_observation_spec[key]

        if not self.keep_reward_keys:
            for key in list(full_reward_spec.keys(True, True)):
                if unravel_key(key) not in self.selected_keys:
                    del full_reward_spec[key]

        return output_spec


class RenameTransform(Transform):
    """A transform to rename entries in the output tensordict (or input tensordict via the inverse keys).

    Args:
        in_keys (sequence of NestedKey): the entries to rename.
        out_keys (sequence of NestedKey): the name of the entries after renaming.
        in_keys_inv (sequence of NestedKey, optional): the entries to rename
            in the input tensordict, which will be passed to :meth:`EnvBase._step`.
        out_keys_inv (sequence of NestedKey, optional): the names of the entries
            in the input tensordict after renaming.
        create_copy (bool, optional): if ``True``, the entries will be copied
            with a different name rather than being renamed. This allows for
            renaming immutable entries such as ``"reward"`` and ``"done"``.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(
        ...     GymEnv("Pendulum-v1"),
        ...     RenameTransform(["observation", ], ["stuff",], create_copy=False),
        ... )
        >>> tensordict = env.rollout(3)
        >>> print(tensordict)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        stuff: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=cpu,
                    is_shared=False),
                stuff: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([3]),
            device=cpu,
            is_shared=False)
        >>> # if the output is also an input, we need to rename if both ways:
        >>> from torchrl.envs.libs.brax import BraxEnv
        >>> env = TransformedEnv(
        ...     BraxEnv("fast"),
        ...     RenameTransform(["state"], ["newname"], ["state"], ["newname"])
        ... )
        >>> _ = env.set_seed(1)
        >>> tensordict = env.rollout(3)
        >>> assert "newname" in tensordict.keys()
        >>> assert "state" not in tensordict.keys()

    """

    def __init__(
        self, in_keys, out_keys, in_keys_inv=None, out_keys_inv=None, create_copy=False
    ):
        if in_keys_inv is None:
            in_keys_inv = []
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)
        self.create_copy = create_copy
        super().__init__(in_keys, out_keys, in_keys_inv, out_keys_inv)
        if len(self.in_keys) != len(self.out_keys):
            raise ValueError(
                f"The number of in_keys ({len(self.in_keys)}) should match the number of out_keys ({len(self.out_keys)})."
            )
        if len(self.in_keys_inv) != len(self.out_keys_inv):
            raise ValueError(
                f"The number of in_keys_inv ({len(self.in_keys_inv)}) should match the number of out_keys_inv ({len(self.out_keys)})."
            )
        if len(set(out_keys).intersection(in_keys)):
            raise ValueError(
                f"Cannot have matching in and out_keys because order is unclear. "
                f"Please use separated transforms. "
                f"Got in_keys={in_keys} and out_keys={out_keys}."
            )

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if self.create_copy:
            out = next_tensordict.select(
                *self.in_keys, strict=not self._missing_tolerance
            )
            for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
                try:
                    out.rename_key_(in_key, out_key)
                except KeyError:
                    if not self._missing_tolerance:
                        raise
            next_tensordict = next_tensordict.update(out)
        else:
            for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
                try:
                    next_tensordict.rename_key_(in_key, out_key)
                except KeyError:
                    if not self._missing_tolerance:
                        raise
        return next_tensordict

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # no in-place modif
        if self.create_copy:
            out = tensordict.select(
                *self.out_keys_inv, strict=not self._missing_tolerance
            )
            for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
                try:
                    out.rename_key_(out_key, in_key)
                except KeyError:
                    if not self._missing_tolerance:
                        raise

            tensordict = tensordict.update(out)
        else:
            for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
                try:
                    tensordict.rename_key_(out_key, in_key)
                except KeyError:
                    if not self._missing_tolerance:
                        raise
        return tensordict

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        for done_key in self.parent.done_keys:
            if done_key in self.in_keys:
                for i, out_key in enumerate(self.out_keys):  # noqa: B007
                    if self.in_keys[i] == done_key:
                        break
                else:
                    # unreachable
                    raise RuntimeError
                output_spec["full_done_spec"][out_key] = output_spec["full_done_spec"][
                    done_key
                ].clone()
                if not self.create_copy:
                    del output_spec["full_done_spec"][done_key]
        for reward_key in self.parent.reward_keys:
            if reward_key in self.in_keys:
                for i, out_key in enumerate(self.out_keys):  # noqa: B007
                    if self.in_keys[i] == reward_key:
                        break
                else:
                    # unreachable
                    raise RuntimeError
                output_spec["full_reward_spec"][out_key] = output_spec[
                    "full_reward_spec"
                ][reward_key].clone()
                if not self.create_copy:
                    del output_spec["full_reward_spec"][reward_key]
        for observation_key in self.parent.full_observation_spec.keys(True):
            if observation_key in self.in_keys:
                for i, out_key in enumerate(self.out_keys):  # noqa: B007
                    if self.in_keys[i] == observation_key:
                        break
                else:
                    # unreachable
                    raise RuntimeError
                output_spec["full_observation_spec"][out_key] = output_spec[
                    "full_observation_spec"
                ][observation_key].clone()
                if not self.create_copy:
                    del output_spec["full_observation_spec"][observation_key]
        return output_spec

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        for action_key in self.parent.action_keys:
            if action_key in self.in_keys_inv:
                for i, out_key in enumerate(self.out_keys_inv):  # noqa: B007
                    if self.in_keys_inv[i] == action_key:
                        break
                else:
                    # unreachable
                    raise RuntimeError
                input_spec["full_action_spec"][out_key] = input_spec[
                    "full_action_spec"
                ][action_key].clone()
        if not self.create_copy:
            for action_key in self.parent.action_keys:
                if action_key in self.in_keys_inv:
                    del input_spec["full_action_spec"][action_key]
        for state_key in self.parent.full_state_spec.keys(True, True):
            if state_key in self.in_keys_inv:
                for i, out_key in enumerate(self.out_keys_inv):  # noqa: B007
                    if self.in_keys_inv[i] == state_key:
                        break
                else:
                    # unreachable
                    raise RuntimeError
                input_spec["full_state_spec"][out_key] = input_spec["full_state_spec"][
                    state_key
                ].clone()
        if not self.create_copy:
            for state_key in self.parent.full_state_spec.keys(True, True):
                if state_key in self.in_keys_inv:
                    del input_spec["full_state_spec"][state_key]
        return input_spec


class RemoveEmptySpecs(Transform):
    """Removes empty specs and content from an environment.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import Unbounded, Composite, \
        ...     Categorical
        >>> from torchrl.envs import EnvBase, TransformedEnv, RemoveEmptySpecs
        >>>
        >>>
        >>> class DummyEnv(EnvBase):
        ...     def __init__(self, *args, **kwargs):
        ...         super().__init__(*args, **kwargs)
        ...         self.observation_spec = Composite(
        ...             observation=UnboundedContinuous((*self.batch_size, 3)),
        ...             other=Composite(
        ...                 another_other=Composite(shape=self.batch_size),
        ...                 shape=self.batch_size,
        ...             ),
        ...             shape=self.batch_size,
        ...         )
        ...         self.action_spec = UnboundedContinuous((*self.batch_size, 3))
        ...         self.done_spec = Categorical(
        ...             2, (*self.batch_size, 1), dtype=torch.bool
        ...         )
        ...         self.full_done_spec["truncated"] = self.full_done_spec[
        ...             "terminated"].clone()
        ...         self.reward_spec = Composite(
        ...             reward=UnboundedContinuous(*self.batch_size, 1),
        ...             other_reward=Composite(shape=self.batch_size),
        ...             shape=self.batch_size
        ...             )
        ...
        ...     def _reset(self, tensordict):
        ...         return self.observation_spec.rand().update(self.full_done_spec.zero())
        ...
        ...     def _step(self, tensordict):
        ...         return TensorDict(
        ...             {},
        ...             batch_size=[]
        ...         ).update(self.observation_spec.rand()).update(
        ...             self.full_done_spec.zero()
        ...             ).update(self.full_reward_spec.rand())
        ...
        ...     def _set_seed(self, seed) -> None:
        ...         pass
        >>>
        >>>
        >>> base_env = DummyEnv()
        >>> print(base_env.rollout(2))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        other: TensorDict(
                            fields={
                                another_other: TensorDict(
                                    fields={
                                    },
                                    batch_size=torch.Size([2]),
                                    device=cpu,
                                    is_shared=False)},
                            batch_size=torch.Size([2]),
                            device=cpu,
                            is_shared=False),
                        other_reward: TensorDict(
                            fields={
                            },
                            batch_size=torch.Size([2]),
                            device=cpu,
                            is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([2]),
            device=cpu,
            is_shared=False)
        >>> check_env_specs(base_env)
        >>> env = TransformedEnv(base_env, RemoveEmptySpecs())
        >>> print(env.rollout(2))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([2]),
            device=cpu,
            is_shared=False)
        check_env_specs(env)
    """

    _has_empty_input = True

    @staticmethod
    def _sorter(key_val):
        key, _ = key_val
        if isinstance(key, str):
            return 0
        return len(key)

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        full_done_spec = output_spec["full_done_spec"]
        full_reward_spec = output_spec["full_reward_spec"]
        full_observation_spec = output_spec["full_observation_spec"]
        # we reverse things to make sure we delete things from the back
        for key, spec in sorted(
            full_done_spec.items(True), key=self._sorter, reverse=True
        ):
            if isinstance(spec, Composite) and spec.is_empty():
                del full_done_spec[key]

        for key, spec in sorted(
            full_observation_spec.items(True), key=self._sorter, reverse=True
        ):
            if isinstance(spec, Composite) and spec.is_empty():
                del full_observation_spec[key]

        for key, spec in sorted(
            full_reward_spec.items(True), key=self._sorter, reverse=True
        ):
            if isinstance(spec, Composite) and spec.is_empty():
                del full_reward_spec[key]
        return output_spec

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        full_action_spec = input_spec["full_action_spec"]
        full_state_spec = input_spec["full_state_spec"]
        # we reverse things to make sure we delete things from the back

        self._has_empty_input = False
        for key, spec in sorted(
            full_action_spec.items(True), key=self._sorter, reverse=True
        ):
            if isinstance(spec, Composite) and spec.is_empty():
                self._has_empty_input = True
                del full_action_spec[key]

        for key, spec in sorted(
            full_state_spec.items(True), key=self._sorter, reverse=True
        ):
            if isinstance(spec, Composite) and spec.is_empty():
                self._has_empty_input = True
                del full_state_spec[key]
        return input_spec

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._has_empty_input:
            input_spec = getattr(self.parent, "input_spec", None)
            if input_spec is None:
                return tensordict

            full_action_spec = input_spec["full_action_spec"]
            full_state_spec = input_spec["full_state_spec"]
            # we reverse things to make sure we delete things from the back

            for key, spec in sorted(
                full_action_spec.items(True), key=self._sorter, reverse=True
            ):
                if (
                    isinstance(spec, Composite)
                    and spec.is_empty()
                    and key not in tensordict.keys(True)
                ):
                    tensordict.create_nested(key)

            for key, spec in sorted(
                full_state_spec.items(True), key=self._sorter, reverse=True
            ):
                if (
                    isinstance(spec, Composite)
                    and spec.is_empty()
                    and key not in tensordict.keys(True)
                ):
                    tensordict.create_nested(key)
        return tensordict

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        for key, value in sorted(
            next_tensordict.items(True), key=self._sorter, reverse=True
        ):
            if (
                is_tensor_collection(value)
                and not isinstance(value, NonTensorData)
                and value.is_empty()
            ):
                del next_tensordict[key]
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Resets a transform if it is stateful."""
        return self._call(tensordict_reset)

    forward = _call


class FlattenTensorDict(Transform):
    """Flattens TensorDict batch dimensions during inverse pass for replay buffer usage.

    This transform is specifically designed for replay buffers where data needs
    to be flattened before being stored. It performs an identity operation during
    the forward pass and flattens the batch dimensions during the inverse pass.

    This is useful when collecting batched data that needs to be stored as
    individual experiences in a replay buffer.

    .. warning::
        This transform is NOT intended for use with environments. If you try to use
        it as an environment transform, it will raise an exception. For reshaping
        environment batch dimensions, use :class:`~torchrl.envs.BatchSizeTransform`
        instead.

    .. note::
        This transform should be applied to replay buffers, not to environments.
        It is designed to be used with :meth:`~torchrl.data.ReplayBuffer.append_transform`.

    Examples:
        Using with a replay buffer:

        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.transforms import FlattenTensorDict
        >>> from torchrl.data import TensorDictReplayBuffer, LazyTensorStorage
        >>>
        >>> # Create a replay buffer with the transform
        >>> transform = FlattenTensorDict()
        >>> rb = TensorDictReplayBuffer(
        ...     storage=LazyTensorStorage(1000),
        ...     transform=transform,
        ...     batch_size=32
        ... )
        >>>
        >>> # Create batched data (e.g., from multiple environments)
        >>> td = TensorDict({
        ...     "observation": torch.randn(4, 2, 3),
        ...     "action": torch.randn(4, 2, 1),
        ...     "reward": torch.randn(4, 2, 1),
        ... }, batch_size=[4, 2])
        >>>
        >>> # When extending the buffer, data gets flattened automatically
        >>> rb.extend(td)  # Data is flattened from [4, 2] to [8] before storage
        >>>
        >>> # When sampling, data comes out in the requested batch size
        >>> sample = rb.sample(4)  # Shape will be [4, ...]

        Direct usage (for testing):

        >>> # Forward pass (identity)
        >>> td_forward = transform(td)
        >>> print(td_forward.batch_size)  # [4, 2]
        >>>
        >>> # Inverse pass (flatten)
        >>> td_inverse = transform.inv(td)
        >>> print(td_inverse.batch_size)  # [8]
    """

    _ENV_ERROR_MSG = (
        "FlattenTensorDict is designed for replay buffers and should not be used "
        "as an environment transform. For reshaping environment batch dimensions, "
        "use BatchSizeTransform instead."
    )

    def __init__(self, inverse: bool = True):
        super().__init__(in_keys=[], out_keys=[])
        self.inverse = inverse

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Forward pass - identity operation."""
        if not self.inverse:
            return tensordict.reshape(-1)
        return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Inverse pass - flatten the tensordict."""
        if self.inverse:
            return tensordict.reshape(-1)
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Forward pass - identity operation."""
        return self._call(tensordict)

    def inv(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Inverse pass - flatten the tensordict."""
        return self._inv_call(tensordict)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Reset pass - identity operation."""
        return self._call(tensordict_reset)

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        """Transform input spec - not supported for environments."""
        raise RuntimeError(self._ENV_ERROR_MSG)

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        """Transform output spec - not supported for environments."""
        raise RuntimeError(self._ENV_ERROR_MSG)

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        """Transform observation spec - not supported for environments."""
        raise RuntimeError(self._ENV_ERROR_MSG)

    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        """Transform action spec - not supported for environments."""
        raise RuntimeError(self._ENV_ERROR_MSG)

    def transform_state_spec(self, state_spec: TensorSpec) -> TensorSpec:
        """Transform state spec - not supported for environments."""
        raise RuntimeError(self._ENV_ERROR_MSG)

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        """Transform reward spec - not supported for environments."""
        raise RuntimeError(self._ENV_ERROR_MSG)

    def transform_done_spec(self, done_spec: TensorSpec) -> TensorSpec:
        """Transform done spec - not supported for environments."""
        raise RuntimeError(self._ENV_ERROR_MSG)
