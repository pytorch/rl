# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from collections.abc import Callable, Sequence
from copy import copy
from typing import Any, TYPE_CHECKING

import numpy as np

import torch

from tensordict import TensorDict, TensorDictBase, unravel_key
from tensordict.utils import _zip_strict, expand_as_right, NestedKey

from torchrl.data.tensor_specs import TensorSpec
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms.utils import (
    _get_reset,
    _set_missing_tolerance,
    check_finite,
)
from torchrl.envs.utils import _terminated_or_truncated

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
    TransformedEnv,
)

__all__ = [
    "ConditionalPolicySwitch",
    "ConditionalSkip",
    "FiniteTensorDictCheck",
    "PinMemoryTransform",
    "RandomCropTensorDict",
    "TimeMaxPool",
    "VecGymEnvTransform",
]


class FiniteTensorDictCheck(Transform):
    """This transform will check that all the items of the tensordict are finite, and raise an exception if they are not."""

    def __init__(self):
        super().__init__(in_keys=[])

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        next_tensordict.apply(check_finite, filter_empty=True)
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    forward = _call


class PinMemoryTransform(Transform):
    """Calls pin_memory on the tensordict to facilitate writing on CUDA devices."""

    def __init__(self):
        super().__init__()

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        return next_tensordict.pin_memory()

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class TimeMaxPool(Transform):
    """Take the maximum value in each position over the last T observations.

    This transform take the maximum value in each position for all in_keys tensors over the last T time steps.

    Args:
        in_keys (sequence of NestedKey, optional): input keys on which the max pool will be applied. Defaults to "observation" if left empty.
        out_keys (sequence of NestedKey, optional): output keys where the output will be written. Defaults to `in_keys` if left empty.
        T (int, optional): Number of time steps over which to apply max pooling.
        reset_key (NestedKey, optional): the reset key to be used as partial
            reset indicator. Must be unique. If not provided, defaults to the
            only reset key of the parent environment (if it has only one)
            and raises an exception otherwise.

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> base_env = GymEnv("Pendulum-v1")
        >>> env = TransformedEnv(base_env, TimeMaxPool(in_keys=["observation"], T=10))
        >>> torch.manual_seed(0)
        >>> env.set_seed(0)
        >>> rollout = env.rollout(10)
        >>> print(rollout["observation"])  # values should be increasing up until the 10th step
        tensor([[ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0000,  0.0000],
                [ 0.0000,  0.0216,  0.0000],
                [ 0.0000,  0.1149,  0.0000],
                [ 0.0000,  0.1990,  0.0000],
                [ 0.0000,  0.2749,  0.0000],
                [ 0.0000,  0.3281,  0.0000],
                [-0.9290,  0.3702, -0.8978]])

    .. note:: :class:`~TimeMaxPool` currently only supports ``done`` signal at the root.
        Nested ``done``, such as those found in MARL settings, are currently not supported.
        If this feature is needed, please raise an issue on TorchRL repo.

    """

    invertible = False

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        T: int = 1,
        reset_key: NestedKey | None = None,
    ):
        if in_keys is None:
            in_keys = ["observation"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if T < 1:
            raise ValueError(
                "TimeMaxPoolTransform T parameter should have a value greater or equal to one."
            )
        if len(self.in_keys) != len(self.out_keys):
            raise ValueError(
                "TimeMaxPoolTransform in_keys and out_keys don't have the same number of elements"
            )
        self.buffer_size = T
        for in_key in self.in_keys:
            buffer_name = self._buffer_name(in_key)
            setattr(
                self,
                buffer_name,
                torch.nn.parameter.UninitializedBuffer(
                    device=torch.device("cpu"), dtype=torch.get_default_dtype()
                ),
            )
        self.reset_key = reset_key

    @staticmethod
    def _buffer_name(in_key):
        in_key_str = "_".join(in_key) if isinstance(in_key, tuple) else in_key
        buffer_name = f"_maxpool_buffer_{in_key_str}"
        return buffer_name

    @property
    def reset_key(self) -> NestedKey:
        reset_key = self.__dict__.get("_reset_key", None)
        if reset_key is None:
            reset_keys = self.parent.reset_keys
            if len(reset_keys) > 1:
                raise RuntimeError(
                    f"Got more than one reset key in env {self.container}, cannot infer which one to use. Consider providing the reset key in the {type(self)} constructor."
                )
            reset_key = self._reset_key = reset_keys[0]
        return reset_key

    @reset_key.setter
    def reset_key(self, value):
        self._reset_key = value

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:

        _reset = _get_reset(self.reset_key, tensordict)
        for in_key in self.in_keys:
            buffer_name = self._buffer_name(in_key)
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                continue
            if not _reset.all():
                _reset_exp = _reset.expand(buffer.shape[0], *_reset.shape)
                buffer[_reset_exp] = 0.0
            else:
                buffer.fill_(0.0)
        with _set_missing_tolerance(self, True):
            for in_key in self.in_keys:
                val_reset = tensordict_reset.get(in_key, None)
                val_prev = tensordict.get(in_key, None)
                # if an in_key is missing, we try to copy it from the previous step
                if val_reset is None and val_prev is not None:
                    tensordict_reset.set(in_key, val_prev)
                elif val_prev is None and val_reset is None:
                    raise KeyError(f"Could not find {in_key} in the reset data.")
            return self._call(tensordict_reset, _reset=_reset)

    def _make_missing_buffer(self, tensordict, in_key, buffer_name):
        buffer = getattr(self, buffer_name)
        data = tensordict.get(in_key)
        size = list(data.shape)
        size.insert(0, self.buffer_size)
        buffer.materialize(size)
        buffer = buffer.to(dtype=data.dtype, device=data.device).zero_()
        setattr(self, buffer_name, buffer)
        return buffer

    def _call(self, next_tensordict: TensorDictBase, _reset=None) -> TensorDictBase:
        """Update the episode tensordict with max pooled keys."""
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            # Lazy init of buffers
            buffer_name = self._buffer_name(in_key)
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                buffer = self._make_missing_buffer(next_tensordict, in_key, buffer_name)
            if _reset is not None:
                # we must use only the reset data
                buffer[:, _reset] = torch.roll(buffer[:, _reset], shifts=1, dims=0)
                # add new obs
                data = next_tensordict.get(in_key)
                buffer[0, _reset] = data[_reset]
                # apply max pooling
                pooled_tensor, _ = buffer[:, _reset].max(dim=0)
                pooled_tensor = torch.zeros_like(data).masked_scatter_(
                    expand_as_right(_reset, data), pooled_tensor
                )
                # add to tensordict
                next_tensordict.set(out_key, pooled_tensor)
                continue
            # shift obs 1 position to the right
            buffer.copy_(torch.roll(buffer, shifts=1, dims=0))
            # add new obs
            buffer[0].copy_(next_tensordict.get(in_key))
            # apply max pooling
            pooled_tensor, _ = buffer.max(dim=0)
            # add to tensordict
            next_tensordict.set(out_key, pooled_tensor)

        return next_tensordict

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            "TimeMaxPool cannot be called independently, only its step and reset methods "
            "are functional. The reason for this is that it is hard to consider using "
            "TimeMaxPool with non-sequential data, such as those collected by a replay buffer "
            "or a dataset. If you need TimeMaxPool to work on a batch of sequential data "
            "(ie as LSTM would work over a whole sequence of data), file an issue on "
            "TorchRL requesting that feature."
        )


class RandomCropTensorDict(Transform):
    """A trajectory sub-sampler for ReplayBuffer and modules.

    Gathers a sub-sequence of a defined length along the last dimension of the input
    tensordict.
    This can be used to get cropped trajectories from trajectories sampled
    from a ReplayBuffer.

    This transform is primarily designed to be used with replay buffers and modules.
    Currently, it cannot be used as an environment transform.
    Do not hesitate to request for this behavior through an issue if this is
    desired.

    Args:
        sub_seq_len (int): the length of the sub-trajectory to sample
        sample_dim (int, optional): the dimension along which the cropping
            should occur. Negative dimensions should be preferred to make
            the transform robust to tensordicts of varying batch dimensions.
            Defaults to -1 (the default time dimension in TorchRL).
        mask_key (NestedKey): If provided, this represents the mask key to be looked
            for when doing the sampling. If provided, it only valid elements will
            be returned. It is assumed that the mask is a boolean tensor with
            first True values and then False values, not mixed together.
            :class:`RandomCropTensorDict` will NOT check that this is respected
            hence any error caused by an improper mask risks to go unnoticed.
            Defaults: None (no mask key).
    """

    def __init__(
        self,
        sub_seq_len: int,
        sample_dim: int = -1,
        mask_key: NestedKey | None = None,
    ):
        self.sub_seq_len = sub_seq_len
        if sample_dim > 0:
            warnings.warn(
                "A positive shape has been passed to the RandomCropTensorDict "
                "constructor. This may have unexpected behaviors when the "
                "passed tensordicts have inconsistent batch dimensions. "
                "For context, by convention, TorchRL concatenates time steps "
                "along the last dimension of the tensordict."
            )
        self.sample_dim = sample_dim
        self.mask_key = mask_key
        super().__init__()

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        shape = tensordict.shape
        dim = self.sample_dim
        # shape must have at least one dimension
        if not len(shape):
            raise RuntimeError(
                "Cannot sub-sample from a tensordict with an empty shape."
            )
        if shape[dim] < self.sub_seq_len:
            raise RuntimeError(
                f"Cannot sample trajectories of length {self.sub_seq_len} along"
                f" dimension {dim} given a tensordict of shape "
                f"{tensordict.shape}. Consider reducing the sub_seq_len "
                f"parameter or increase sample length."
            )
        max_idx_0 = shape[dim] - self.sub_seq_len
        idx_shape = list(tensordict.shape)
        idx_shape[dim] = 1
        device = tensordict.device
        if device is None:
            device = torch.device("cpu")
        if self.mask_key is None or self.mask_key not in tensordict.keys(
            isinstance(self.mask_key, tuple)
        ):
            idx_0 = torch.randint(max_idx_0, idx_shape, device=device)
        else:
            # get the traj length for each entry
            mask = tensordict.get(self.mask_key)
            if mask.shape != tensordict.shape:
                raise ValueError(
                    "Expected a mask of the same shape as the tensordict. Got "
                    f"mask.shape={mask.shape} and tensordict.shape="
                    f"{tensordict.shape} instead."
                )
            traj_lengths = mask.cumsum(self.sample_dim).max(self.sample_dim, True)[0]
            if (traj_lengths < self.sub_seq_len).any():
                raise RuntimeError(
                    f"Cannot sample trajectories of length {self.sub_seq_len} when the minimum "
                    f"trajectory length is {traj_lengths.min()}."
                )
            # take a random number between 0 and traj_lengths - self.sub_seq_len
            idx_0 = (
                torch.rand(idx_shape, device=device) * (traj_lengths - self.sub_seq_len)
            ).to(torch.long)
        arange = torch.arange(self.sub_seq_len, device=idx_0.device)
        arange_shape = [1 for _ in range(tensordict.ndimension())]
        arange_shape[dim] = len(arange)
        arange = arange.view(arange_shape)
        idx = idx_0 + arange
        return tensordict.gather(dim=self.sample_dim, index=idx)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self.forward(tensordict_reset)
        return tensordict_reset


class VecGymEnvTransform(Transform):
    """A transform for GymWrapper subclasses that handles the auto-reset in a consistent way.

    Gym, gymnasium and SB3 provide vectorized (read, parallel or batched) environments
    that are automatically reset. When this occurs, the actual observation resulting
    from the action is saved within a key in the info.
    The class :class:`torchrl.envs.libs.gym.terminal_obs_reader` reads that observation
    and stores it in a ``"final"`` key within the output tensordict.
    In turn, this transform reads that final data, swaps it with the observation
    written in its place that results from the actual reset, and saves the
    reset output in a private container. The resulting data truly reflects
    the output of the step.

    This class works from gym 0.13 till the most recent gymnasium version.

    .. note:: Gym versions < 0.22 did not return the final observations. For these,
        we simply fill the next observations with NaN (because it is lost) and
        do the swap at the next step.

    Then, when calling `env.reset`, the saved data is written back where it belongs
    (and the `reset` is a no-op).

    This transform is automatically appended to the gym env whenever the wrapper
    is created with an async env.

    Args:
        final_name (str, optional): the name of the final observation in the dict.
            Defaults to `"final"`.
        missing_obs_value (Any, optional): default value to use as placeholder for missing
            last observations. Defaults to `np.nan`.

    .. note:: In general, this class should not be handled directly. It is
        created whenever a vectorized environment is placed within a :class:`GymWrapper`.

    """

    def __init__(self, final_name: str = "final", missing_obs_value: Any = np.nan):
        self.final_name = final_name
        super().__init__()
        self._memo = {}
        if not isinstance(missing_obs_value, torch.Tensor):
            missing_obs_value = torch.tensor(missing_obs_value)
        self.missing_obs_value = missing_obs_value

    def set_container(self, container: Transform | EnvBase) -> None:
        out = super().set_container(container)
        self._done_keys = None
        self._obs_keys = None
        return out

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        # save the final info
        done = False
        for done_key in self.done_keys:
            # we assume dones can be broadcast
            done = done | next_tensordict.get(done_key)
        if done is False:
            raise RuntimeError(
                f"Could not find any done signal in tensordict:\n{tensordict}"
            )
        self._memo["done"] = done
        final = next_tensordict.pop(self.final_name, None)
        # if anything's done, we need to swap the final obs
        if done.any():
            done = done.squeeze(-1)
            if final is not None:
                saved_next = next_tensordict.select(*final.keys(True, True)).clone()
                next_tensordict[done] = final[done]
            else:
                saved_next = next_tensordict.select(*self.obs_keys).clone()
                for obs_key in self.obs_keys:
                    next_tensordict[obs_key][done] = self.missing_obs_value

            self._memo["saved_next"] = saved_next
        else:
            self._memo["saved_next"] = None
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        done = self._memo.get("done", None)
        reset = tensordict.get("_reset", done)
        if done is not None:
            done = done.view_as(reset)
        if (
            reset is not done
            and (reset != done).any()
            # it can happen that all are reset, in which case
            # it's fine (doesn't need to match done)
            and not reset.all()
        ):
            raise RuntimeError(
                "Cannot partially reset a gym(nasium) async env with a "
                "reset mask that does not match the done mask. "
                f"Got reset={reset}\nand done={done}"
            )
        # if not reset.any(), we don't need to do anything.
        # if reset.all(), we don't either (bc GymWrapper will call a plain reset).
        if reset is not None and reset.any():
            if reset.all():
                # We're fine: this means that a full reset was passed and the
                # env was manually reset
                tensordict_reset.pop(self.final_name, None)
                return tensordict_reset
            saved_next = self._memo["saved_next"]
            if saved_next is None:
                raise RuntimeError(
                    "Did not find a saved tensordict while the reset mask was "
                    f"not empty: reset={reset}. Done was {done}."
                )
            # reset = reset.view(tensordict.shape)
            # we have a data container from the previous call to step
            # that contains part of the observation we need.
            # We can safely place them back in the reset result tensordict:
            # in env.rollout(), the result of reset() is assumed to be just
            # the td from previous step with updated values from reset.
            # In our case, it will always be the case that all these values
            # are properly set.
            # collectors even take care of doing an extra masking so it's even
            # safer.
            tensordict_reset.update(saved_next)
            for done_key in self.done_keys:
                # Make sure that all done are False
                done = tensordict.get(done_key, None)
                if done is not None:
                    done = done.clone().fill_(0)
                else:
                    done = torch.zeros(
                        (*tensordict.batch_size, 1),
                        device=tensordict.device,
                        dtype=torch.bool,
                    )
                tensordict.set(done_key, done)
        tensordict_reset.pop(self.final_name, None)
        return tensordict_reset

    @property
    def done_keys(self) -> list[NestedKey]:
        keys = self.__dict__.get("_done_keys", None)
        if keys is None:
            keys = self.parent.done_keys
            # we just want the "done" key
            _done_keys = []
            for key in keys:
                if not isinstance(key, tuple):
                    key = (key,)
                if key[-1] == "done":
                    _done_keys.append(unravel_key(key))
            if not len(_done_keys):
                raise RuntimeError("Could not find a 'done' key in the env specs.")
            self._done_keys = _done_keys
        return keys

    @property
    def obs_keys(self) -> list[NestedKey]:
        keys = self.__dict__.get("_obs_keys", None)
        if keys is None:
            keys = list(self.parent.observation_spec.keys(True, True))
            self._obs_keys = keys
        return keys

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if self.final_name in observation_spec.keys(True):
            del observation_spec[self.final_name]
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError(FORWARD_NOT_IMPLEMENTED.format(type(self)))


class _InvertTransform(Transform):
    _MISSING_TRANSFORM_ERROR = (
        "There is not generic rule to invert a spec transform. "
        "Please file an issue on github to get help."
    )

    def __init__(self, transform: Transform):
        super().__init__()
        self.transform = transform

    @property
    def in_keys(self) -> Sequence[NestedKey]:
        return self.transform.in_keys_inv

    @in_keys.setter
    def in_keys(self, value):
        if value is not None:
            raise RuntimeError("Cannot set non-null value in in_keys.")

    @property
    def in_keys_inv(self) -> Sequence[NestedKey]:
        return self.transform.in_keys

    @in_keys_inv.setter
    def in_keys_inv(self, value):
        if value is not None:
            raise RuntimeError("Cannot set non-null value in in_keys_inv.")

    @property
    def out_keys(self) -> Sequence[NestedKey]:
        return self.transform.out_keys_inv

    @out_keys.setter
    def out_keys(self, value):
        if value is not None:
            raise RuntimeError("Cannot set non-null value in out_keys.")

    @property
    def out_keys_inv(self) -> Sequence[NestedKey]:
        return self.transform.out_keys

    @out_keys_inv.setter
    def out_keys_inv(self, value):
        if value is not None:
            raise RuntimeError("Cannot set non-null value in out_keys_inv.")

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.transform.inv(tensordict)

    def inv(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.transform.forward(tensordict)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        return self.transform._inv_call(next_tensordict)

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return self.transform._call(tensordict)

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        raise RuntimeError(self._MISSING_TRANSFORM_ERROR)

    def transform_state_spec(self, state_spec: TensorSpec) -> TensorSpec:
        raise RuntimeError(self._MISSING_TRANSFORM_ERROR)

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        raise RuntimeError(self._MISSING_TRANSFORM_ERROR)

    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        raise RuntimeError(self._MISSING_TRANSFORM_ERROR)

    def transform_done_spec(self, done_spec: TensorSpec) -> TensorSpec:
        raise RuntimeError(self._MISSING_TRANSFORM_ERROR)


class ConditionalSkip(Transform):
    """A transform that skips steps in the env if certain conditions are met.

    This transform writes the result of `cond(tensordict)` in the `"_step"` entry of the
    tensordict passed as input to the `TransformedEnv.base_env._step` method.
    If the `base_env` is not batch-locked (generally speaking, it is stateless), the tensordict is
    reduced to its element that need to go through the step.
    If it is batch-locked (generally speaking, it is stateful), the step is skipped altogether if no
    value in `"_step"` is ``True``. Otherwise, it is trusted that the environment will account for the
    `"_step"` signal accordingly.

    .. note:: The skip will affect transforms that modify the environment output too, i.e., any transform
        that is to be executed on the tensordict returned by :meth:`~torchrl.envs.EnvBase.step` will be
        skipped if the condition is met. To palliate this effect if it is not desirable, one can wrap
        the transformed env in another transformed env, since the skip only affects the first-degree parent
        of the ``ConditionalSkip`` transform. See example below.

    Args:
        cond (Callable[[TensorDictBase], bool | torch.Tensor]): a callable for the tensordict input
            that checks whether the next env step must be skipped (`True` = skipped, `False` = execute
            env.step).

    Examples:
        >>> import torch
        >>>
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.envs.transforms.transforms import ConditionalSkip, StepCounter, TransformedEnv, Compose
        >>>
        >>> torch.manual_seed(0)
        >>>
        >>> base_env = TransformedEnv(
        ...     GymEnv("Pendulum-v1"),
        ...     StepCounter(step_count_key="inner_count"),
        ... )
        >>> middle_env = TransformedEnv(
        ...     base_env,
        ...     Compose(
        ...         StepCounter(step_count_key="middle_count"),
        ...         ConditionalSkip(cond=lambda td: td["step_count"] % 2 == 1),
        ...     ),
        ...     auto_unwrap=False)  # makes sure that transformed envs are properly wrapped
        >>> env = TransformedEnv(
        ...     middle_env,
        ...     StepCounter(step_count_key="step_count"),
        ...     auto_unwrap=False)
        >>> env.set_seed(0)
        >>>
        >>> r = env.rollout(10)
        >>> print(r["observation"])
        tensor([[-0.9670, -0.2546, -0.9669],
                [-0.9802, -0.1981, -1.1601],
                [-0.9802, -0.1981, -1.1601],
                [-0.9926, -0.1214, -1.5556],
                [-0.9926, -0.1214, -1.5556],
                [-0.9994, -0.0335, -1.7622],
                [-0.9994, -0.0335, -1.7622],
                [-0.9984,  0.0561, -1.7933],
                [-0.9984,  0.0561, -1.7933],
                [-0.9895,  0.1445, -1.7779]])
        >>> print(r["inner_count"])
        tensor([[0],
                [1],
                [1],
                [2],
                [2],
                [3],
                [3],
                [4],
                [4],
                [5]])
        >>> print(r["middle_count"])
        tensor([[0],
                [1],
                [1],
                [2],
                [2],
                [3],
                [3],
                [4],
                [4],
                [5]])
        >>> print(r["step_count"])
        tensor([[0],
                [1],
                [2],
                [3],
                [4],
                [5],
                [6],
                [7],
                [8],
                [9]])


    """

    def __init__(self, cond: Callable[[TensorDict], bool | torch.Tensor]):
        super().__init__()
        self.cond = cond

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Run cond
        cond = self.cond(tensordict)
        # Write result in step
        tensordict["_step"] = tensordict.get("_step", True) & ~cond
        if tensordict["_step"].shape != tensordict.batch_size:
            tensordict["_step"] = tensordict["_step"].view(tensordict.batch_size)
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            FORWARD_NOT_IMPLEMENTED.format(self.__class__.__name__)
        )


class ConditionalPolicySwitch(Transform):
    """A transform that conditionally switches between policies based on a specified condition.

    This transform evaluates a condition on the data returned by the environment's `step` method.
    If the condition is met, it applies a specified policy to the data. Otherwise, the data is
    returned unaltered. This is useful for scenarios where different policies need to be applied
    based on certain criteria, such as alternating turns in a game.

    Args:
        policy (Callable[[TensorDictBase], TensorDictBase]):
            The policy to be applied when the condition is met. This should be a callable that
            takes a `TensorDictBase` and returns a `TensorDictBase`.
        condition (Callable[[TensorDictBase], bool]):
            A callable that takes a `TensorDictBase` and returns a boolean or a tensor indicating
            whether the policy should be applied.

    .. warning:: This transform must have a parent environment.

    .. note:: Ideally, it should be the last transform  in the stack. If the policy requires transformed
        data (e.g., images), and this transform  is applied before those transformations, the policy will
        not receive the transformed data.

    Examples:
        >>> import torch
        >>> from tensordict.nn import TensorDictModule as Mod
        >>>
        >>> from torchrl.envs import GymEnv, ConditionalPolicySwitch, Compose, StepCounter
        >>> # Create a CartPole environment. We'll be looking at the obs: if the first element of the obs is greater than
        >>> # 0 (left position) we do a right action (action=0) using the switch policy. Otherwise, we use our main
        >>> # policy which does a left action.
        >>> base_env = GymEnv("CartPole-v1", categorical_action_encoding=True)
        >>>
        >>> policy = Mod(lambda: torch.ones((), dtype=torch.int64), in_keys=[], out_keys=["action"])
        >>> policy_switch = Mod(lambda: torch.zeros((), dtype=torch.int64), in_keys=[], out_keys=["action"])
        >>>
        >>> cond = lambda td: td.get("observation")[..., 0] >= 0
        >>>
        >>> env = base_env.append_transform(
        ...     Compose(
        ...         # We use two step counters to show that one counts the global steps, whereas the other
        ...         # only counts the steps where the main policy is executed
        ...         StepCounter(step_count_key="step_count_total"),
        ...         ConditionalPolicySwitch(condition=cond, policy=policy_switch),
        ...         StepCounter(step_count_key="step_count_main"),
        ...     )
        ... )
        >>>
        >>> env.set_seed(0)
        >>> torch.manual_seed(0)
        >>>
        >>> r = env.rollout(100, policy=policy)
        >>> print("action", r["action"])
        action tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        >>> print("obs", r["observation"])
        obs tensor([[ 0.0322, -0.1540,  0.0111,  0.3190],
                [ 0.0299, -0.1544,  0.0181,  0.3280],
                [ 0.0276, -0.1550,  0.0255,  0.3414],
                [ 0.0253, -0.1558,  0.0334,  0.3596],
                [ 0.0230, -0.1569,  0.0422,  0.3828],
                [ 0.0206, -0.1582,  0.0519,  0.4117],
                [ 0.0181, -0.1598,  0.0629,  0.4469],
                [ 0.0156, -0.1617,  0.0753,  0.4891],
                [ 0.0130, -0.1639,  0.0895,  0.5394],
                [ 0.0104, -0.1665,  0.1058,  0.5987],
                [ 0.0076, -0.1696,  0.1246,  0.6685],
                [ 0.0047, -0.1732,  0.1463,  0.7504],
                [ 0.0016, -0.1774,  0.1715,  0.8459],
                [-0.0020,  0.0150,  0.1884,  0.6117],
                [-0.0017,  0.2071,  0.2006,  0.3838]])
        >>> print("obs'", r["next", "observation"])
        obs' tensor([[ 0.0299, -0.1544,  0.0181,  0.3280],
                [ 0.0276, -0.1550,  0.0255,  0.3414],
                [ 0.0253, -0.1558,  0.0334,  0.3596],
                [ 0.0230, -0.1569,  0.0422,  0.3828],
                [ 0.0206, -0.1582,  0.0519,  0.4117],
                [ 0.0181, -0.1598,  0.0629,  0.4469],
                [ 0.0156, -0.1617,  0.0753,  0.4891],
                [ 0.0130, -0.1639,  0.0895,  0.5394],
                [ 0.0104, -0.1665,  0.1058,  0.5987],
                [ 0.0076, -0.1696,  0.1246,  0.6685],
                [ 0.0047, -0.1732,  0.1463,  0.7504],
                [ 0.0016, -0.1774,  0.1715,  0.8459],
                [-0.0020,  0.0150,  0.1884,  0.6117],
                [-0.0017,  0.2071,  0.2006,  0.3838],
                [ 0.0105,  0.2015,  0.2115,  0.5110]])
        >>> print("total step count", r["step_count_total"].squeeze())
        total step count tensor([ 1,  3,  5,  7,  9, 11, 13, 15, 17, 19, 21, 23, 25, 26, 27])
        >>> print("total step with main policy", r["step_count_main"].squeeze())
        total step with main policy tensor([ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14])

    """

    def __init__(
        self,
        policy: Callable[[TensorDictBase], TensorDictBase],
        condition: Callable[[TensorDictBase], bool],
    ):
        super().__init__([], [])
        self.__dict__["policy"] = policy
        self.condition = condition

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        cond = self.condition(next_tensordict)
        if not isinstance(cond, (bool, torch.Tensor)):
            raise RuntimeError(
                "Calling the condition function should return a boolean or a tensor."
            )
        elif isinstance(cond, (torch.Tensor,)):
            if tuple(cond.shape) not in ((1,), (), tuple(tensordict.shape)):
                raise RuntimeError(
                    "Tensor outputs must have the shape of the tensordict, or contain a single element."
                )
        else:
            cond = torch.tensor(cond, device=tensordict.device)

        if cond.any():
            step = tensordict.get("_step", cond)
            if step.shape != cond.shape:
                step = step.view_as(cond)
            cond = cond & step

            parent: TransformedEnv = self.parent
            any_done, done = self._check_done(next_tensordict)
            next_td_save = None
            if any_done:
                if next_tensordict.numel() == 1 or done.all():
                    return next_tensordict
                if parent.base_env.batch_locked:
                    raise RuntimeError(
                        "Cannot run partial steps in a batched locked environment. "
                        "Hint: Parallel and Serial envs can be unlocked through a keyword argument in "
                        "the constructor."
                    )
                done = done.view(next_tensordict.shape)
                cond = cond & ~done
            if not cond.all():
                if parent.base_env.batch_locked:
                    raise RuntimeError(
                        "Cannot run partial steps in a batched locked environment. "
                        "Hint: Parallel and Serial envs can be unlocked through a keyword argument in "
                        "the constructor."
                    )
                next_td_save = next_tensordict
                next_tensordict = next_tensordict[cond]
                tensordict = tensordict[cond]

            # policy may be expensive or raise an exception when executed with unadequate data so
            # we index the td first
            td = self.policy(
                parent.step_mdp(tensordict.copy().set("next", next_tensordict))
            )
            # Mark the partial steps if needed
            if next_td_save is not None:
                td_new = td.new_zeros(cond.shape)
                # TODO: swap with masked_scatter when avail
                td_new[cond] = td
                td = td_new
                td.set("_step", cond)
            next_tensordict = parent._step(td)
            if next_td_save is not None:
                return torch.where(cond, next_tensordict, next_td_save)
            return next_tensordict
        return next_tensordict

    def _check_done(self, tensordict):
        env = self.parent
        if env._simple_done:
            done = tensordict._get_str("done", default=None)
            if done is not None:
                any_done = done.any()
            else:
                any_done = False
        else:
            any_done = _terminated_or_truncated(
                tensordict,
                full_done_spec=env.output_spec["full_done_spec"],
                key="_reset",
            )
            done = tensordict.pop("_reset")
        return any_done, done

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        cond = self.condition(tensordict_reset)
        # TODO: move to validate
        if not isinstance(cond, (bool, torch.Tensor)):
            raise RuntimeError(
                "Calling the condition function should return a boolean or a tensor."
            )
        elif isinstance(cond, (torch.Tensor,)):
            if tuple(cond.shape) not in ((1,), (), tuple(tensordict.shape)):
                raise RuntimeError(
                    "Tensor outputs must have the shape of the tensordict, or contain a single element."
                )
        else:
            cond = torch.tensor(cond, device=tensordict.device)

        if cond.any():
            reset = tensordict.get("_reset", cond)
            if reset.shape != cond.shape:
                reset = reset.view_as(cond)
            cond = cond & reset

            parent: TransformedEnv = self.parent
            reset_td_save = None
            if not cond.all():
                reset_td_save = tensordict_reset.copy()
                tensordict_reset = tensordict_reset[cond]
                tensordict = tensordict[cond]

            td = self.policy(tensordict_reset)
            # Mark the partial steps if needed
            if reset_td_save is not None:
                td_new = td.new_zeros(cond.shape)
                # TODO: swap with masked_scatter when avail
                td_new[cond] = td
                td = td_new
                td.set("_step", cond)
            tensordict_reset = parent._step(td).exclude(*parent.reward_keys)
            if reset_td_save is not None:
                return torch.where(cond, tensordict_reset, reset_td_save)
            return tensordict_reset

        return tensordict_reset

    def forward(self, tensordict: TensorDictBase) -> Any:
        raise RuntimeError(
            "ConditionalPolicySwitch cannot be called independently, only its step and reset methods are functional."
        )
