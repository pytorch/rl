# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from tensordict import NestedKey, TensorDictBase
from torchrl.data.postprocs.postprocs import _multi_step_func
from torchrl.envs.transforms.transforms import Transform


def _under_next(key: NestedKey) -> tuple:
    """Return the ``("next", *key)`` variant of a nested key."""
    if isinstance(key, tuple):
        return ("next", *key)
    return ("next", key)


class MultiStepTransform(Transform):
    """A MultiStep transformation for ReplayBuffers.

    This transform keeps the previous ``n_steps`` observations in a local buffer.
    The inverse transform (called during :meth:`~torchrl.data.ReplayBuffer.extend`)
    outputs the transformed previous ``n_steps`` with the ``T-n_steps`` current
    frames.

    All entries in the ``"next"`` tensordict that are not part of the ``done_keys``
    or ``reward_keys`` will be mapped to their respective ``t + n_steps - 1``
    correspondent.

    This transform is a more hyperparameter resistant version of
    :class:`~torchrl.data.postprocs.postprocs.MultiStep`:
    the replay buffer transform will make the multi-step transform insensitive
    to the collectors hyperparameters, whereas the post-process
    version will output results that are sensitive to these
    (because collectors have no memory of previous output).

    Args:
        n_steps (int): Number of steps in multi-step. The number of steps can be
            dynamically changed by changing the ``n_steps`` attribute of this
            transform.
        gamma (:obj:`float`): Discount factor.

    Keyword Args:
        reward_keys (list of NestedKey, optional): the reward keys in the input tensordict.
            The reward entries indicated by these keys will be accumulated and discounted
            across ``n_steps`` steps in the future. A corresponding ``<reward_key>_orig``
            entry will be written in the ``"next"`` entry of the output tensordict
            to keep track of the original value of the reward.
            Defaults to ``["reward"]``.
        done_key (NestedKey, optional): the done key in the input tensordict, used to indicate
            an end of trajectory.
            Defaults to ``"done"``.
        done_keys (list of NestedKey, optional): the list of end keys in the input tensordict.
            All the entries indicated by these keys will be left untouched by the transform.
            Defaults to ``["done", "truncated", "terminated"]``.
        mask_key (NestedKey, optional): the mask key in the input tensordict.
            The mask represents the valid frames in the input tensordict and
            should have a shape that allows the input tensordict to be masked
            with.
            Defaults to ``"mask"``.

    Examples:
        >>> from torchrl.envs import GymEnv, TransformedEnv, StepCounter, MultiStepTransform, SerialEnv
        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage
        >>> rb = ReplayBuffer(
        ...     storage=LazyTensorStorage(100, ndim=2),
        ...     transform=MultiStepTransform(n_steps=3, gamma=0.95)
        ... )
        >>> base_env = SerialEnv(2, lambda: GymEnv("CartPole"))
        >>> env = TransformedEnv(base_env, StepCounter())
        >>> _ = env.set_seed(0)
        >>> _ = torch.manual_seed(0)
        >>> tdreset = env.reset()
        >>> for _ in range(100):
        ...     rollout = env.rollout(max_steps=50, break_when_any_done=False,
        ...         tensordict=tdreset, auto_reset=False)
        ...     indices = rb.extend(rollout)
        ...     tdreset = rollout[..., -1]["next"]
        >>> print("step_count", rb[:]["step_count"][:, :5])
        step_count tensor([[[ 9],
                 [10],
                 [11],
                 [12],
                 [13]],
        <BLANKLINE>
                [[12],
                 [13],
                 [14],
                 [15],
                 [16]]])
        >>> # The next step_count is 3 steps in the future
        >>> print("next step_count", rb[:]["next", "step_count"][:, :5])
        next step_count tensor([[[13],
                 [14],
                 [15],
                 [16],
                 [17]],
        <BLANKLINE>
                [[16],
                 [17],
                 [18],
                 [19],
                 [20]]])

    """

    ENV_ERR = (
        "The MultiStepTransform is only an inverse transform and can "
        "be applied exclusively to replay buffers."
    )

    def __init__(
        self,
        n_steps,
        gamma,
        *,
        reward_keys: list[NestedKey] | None = None,
        done_key: NestedKey | None = None,
        done_keys: list[NestedKey] | None = None,
        mask_key: NestedKey | None = None,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.reward_keys = reward_keys
        self.done_key = done_key
        self.done_keys = done_keys
        self.mask_key = mask_key
        self.gamma = gamma
        self._buffer = None
        self._validated = False

    @property
    def n_steps(self):
        """The look ahead window of the transform.

        This value can be dynamically edited during training.
        """
        return self._n_steps

    @n_steps.setter
    def n_steps(self, value):
        if not isinstance(value, int) or not (value >= 1):
            raise ValueError(
                "The value of n_steps must be a strictly positive integer."
            )
        self._n_steps = value

    @property
    def done_key(self):
        return self._done_key

    @done_key.setter
    def done_key(self, value):
        if value is None:
            value = "done"
        self._done_key = value

    @property
    def done_keys(self):
        return self._done_keys

    @done_keys.setter
    def done_keys(self, value):
        if value is None:
            value = ["done", "terminated", "truncated"]
        self._done_keys = value

    @property
    def reward_keys(self):
        return self._reward_keys

    @reward_keys.setter
    def reward_keys(self, value):
        if value is None:
            value = [
                "reward",
            ]
        self._reward_keys = value

    @property
    def mask_key(self):
        return self._mask_key

    @mask_key.setter
    def mask_key(self, value):
        if value is None:
            value = "mask"
        self._mask_key = value

    def _validate(self):
        if self.parent is not None:
            raise ValueError(self.ENV_ERR)
        self._validated = True

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self._validated:
            self._validate()

        total_cat = self._append_tensordict(tensordict)
        if total_cat.shape[-1] > self.n_steps:
            out = _multi_step_func(
                total_cat,
                done_key=self.done_key,
                done_keys=self.done_keys,
                reward_keys=self.reward_keys,
                mask_key=self.mask_key,
                n_steps=self.n_steps,
                gamma=self.gamma,
            )
            return out[..., : -self.n_steps]

    def _append_tensordict(self, data):
        if self._buffer is None:
            total_cat = data
            self._buffer = data[..., -self.n_steps :].copy()
        else:
            total_cat = torch.cat([self._buffer, data], -1)
            self._buffer = total_cat[..., -self.n_steps :].copy()
        return total_cat


class NextStateReconstructor(Transform):
    """Re-hydrate ``("next", obs)`` keys at sampling time by shifting along the batch.

    Pairs with :class:`~torchrl.collectors.SyncDataCollector` configured with
    ``compact_obs=True`` (and the analogous flag on the multi-process collectors):
    the collector drops the observation and state keys from the
    ``("next", ...)`` sub-tensordict before stacking because those values are
    bit-for-bit identical to the root keys at ``t + 1`` within the same
    trajectory; this transform rebuilds them on the consumer side.

    **Core rule.** For each registered root key ``k`` and each position ``i``
    of the flat sampled batch:

    - if position ``i + 1`` is in the batch *and* belongs to the same
      trajectory as position ``i``, write
      ``data[("next", k)][i] = data[k][i + 1]``;
    - otherwise write ``data[("next", k)][i] = fill_value`` (``NaN`` by
      default).

    "Same trajectory" is decided from a trajectory id key in the sample,
    by default ``("collector", "traj_ids")`` — the key that
    :class:`~torchrl.collectors.SyncDataCollector` populates when
    ``track_traj_ids=True`` (the default). The semantics fall out cleanly for
    every common sampler:

    - :class:`~torchrl.data.replay_buffers.samplers.SliceSampler` with
      ``traj_key``: positions inside a slice mirror to the next position;
      slice boundaries differ in trajectory id and become ``NaN``.
    - A full rollout sampled as one contiguous batch: every transition inside
      a trajectory is reconstructed; trajectory ends become ``NaN``.
    - :class:`~torchrl.data.replay_buffers.samplers.RandomSampler` and similar:
      adjacent batch positions almost never share a trajectory id, so the
      result is mostly ``NaN``. This is correct — the next observation is
      genuinely not available in the sampled batch — and it makes the
      mis-use loud rather than silent.

    The trajectory-id check alone is *not* enough: a sampler is allowed to
    place two slices of the *same* trajectory back-to-back in one batch
    (e.g. :class:`~torchrl.data.replay_buffers.samplers.SliceSampler` sampling
    with replacement when there are fewer trajectories than slices). In that
    case the two positions across the splice would share a trajectory id
    without being consecutive in time. The transform therefore also consults
    ``("next", "done")`` (if present): when ``done[i]`` is ``True`` the
    trajectory ended at step ``i``, so position ``i + 1`` is never the next
    step of trajectory ``traj_id[i]`` no matter what.

    An additional, stricter ``step_count_key`` cross-check is available for
    setups where neither ``traj_id`` nor ``done`` are bulletproof — see below.

    Args:
        keys (sequence of NestedKey, optional): the root keys whose
            ``("next", k)`` counterparts should be reconstructed. Defaults to
            ``("observation",)``. For environments with nested observation
            specs, pass the full leaf list, e.g.
            ``[("agents", "pos"), ("agents", "vel")]``.

    Keyword Args:
        traj_key (NestedKey, optional): key carrying the trajectory id used
            to detect boundaries. Defaults to ``("collector", "traj_ids")``.
            Set to ``None`` to skip the trajectory check and treat the entire
            sampled batch as one trajectory (only the very last position is
            then filled with ``fill_value``).
        done_key (NestedKey, optional): key whose ``True`` entries indicate
            that the trajectory terminated at position ``i``, so position
            ``i + 1`` is not the next step. Defaults to ``("next", "done")``.
            Set to ``None`` to disable the check.
        step_count_key (NestedKey, optional): if not ``None``, also require
            ``data[step_count_key][i + 1] == data[step_count_key][i] + 1`` to
            consider position ``i + 1`` as the canonical next step. The
            collector populates ``("collector", "step_count")`` only when a
            :class:`~torchrl.envs.transforms.StepCounter` is in the env
            transform chain. Defaults to ``None``.
        fill_value (float, optional): value written wherever the next
            observation is not available. Defaults to ``float("nan")``. For
            integer-typed observation keys, NaN cannot be represented; pass
            an explicit integer (e.g. ``0``).
        strict (bool, optional): if ``True`` (default) and any configured
            marker key (``traj_key``, ``done_key``, ``step_count_key``) is
            missing from the sampled batch, raise. If ``False``, silently
            drop that check.

    Example:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import ReplayBuffer, LazyTensorStorage
        >>> from torchrl.data.replay_buffers.samplers import SliceSampler
        >>> from torchrl.envs.transforms.rb_transforms import (
        ...     NextStateReconstructor,
        ... )
        >>> rb = ReplayBuffer(
        ...     storage=LazyTensorStorage(100),
        ...     sampler=SliceSampler(
        ...         slice_len=4, traj_key=("collector", "traj_ids"),
        ...     ),
        ...     transform=NextStateReconstructor(),
        ...     batch_size=8,
        ... )
        >>> # populate `rb` with a collector configured with `compact_obs=True`
        >>> # so that ``("next", "observation")`` is absent from storage:
        >>> data = TensorDict({
        ...     "observation": torch.arange(8, dtype=torch.float32).view(8, 1),
        ...     ("next", "reward"): torch.zeros(8, 1),
        ...     ("next", "done"): torch.tensor([[False]] * 7 + [[True]]),
        ...     ("collector", "traj_ids"): torch.tensor([0, 0, 0, 0, 1, 1, 1, 1]),
        ... }, batch_size=[8])
        >>> rb.extend(data)
        >>> sample = rb.sample()  # ('next', 'observation') is reconstructed
    """

    def __init__(
        self,
        keys: Sequence[NestedKey] = ("observation",),
        *,
        traj_key: NestedKey | None = ("collector", "traj_ids"),
        done_key: NestedKey | None = ("next", "done"),
        step_count_key: NestedKey | None = None,
        fill_value: float = float("nan"),
        strict: bool = True,
    ):
        super().__init__()
        self.keys = tuple(keys)
        self.traj_key = traj_key
        self.done_key = done_key
        self.step_count_key = step_count_key
        self.fill_value = fill_value
        self.strict = strict

    @staticmethod
    def _flatten_marker(t: torch.Tensor, B: int) -> torch.Tensor:
        """Reduce a marker tensor of shape ``(B, ...)`` to ``(B,)`` along trailing dims."""
        if t.shape[0] != B:
            raise ValueError(
                f"NextStateReconstructor: marker tensor has leading dim {t.shape[0]} "
                f"but sample batch size is {B}."
            )
        if t.ndim == 1:
            return t
        return t.reshape(B, -1)[:, 0]

    def _fetch_marker(
        self,
        tensordict: TensorDictBase,
        key: NestedKey,
        what: str,
        B: int,
    ) -> torch.Tensor | None:
        if key in tensordict.keys(True, True):
            return self._flatten_marker(tensordict.get(key), B)
        if self.strict:
            raise KeyError(
                f"NextStateReconstructor: {what} {key!r} is not present in the "
                "sampled batch. Pass the corresponding constructor kwarg "
                "explicitly (or `None` to disable), or `strict=False` to drop "
                "the check silently."
            )
        return None

    def _valid_mask(self, tensordict: TensorDictBase, B: int) -> torch.Tensor:
        """Return a ``(B,)`` bool tensor where ``True`` means ``i + 1`` is a usable next step."""
        valid = torch.zeros(B, dtype=torch.bool, device=tensordict.device)
        if B >= 2:
            valid[:-1] = True
        if self.traj_key is not None:
            traj = self._fetch_marker(tensordict, self.traj_key, "trajectory key", B)
            if traj is not None:
                valid[:-1] &= traj[1:] == traj[:-1]
        if self.done_key is not None:
            done = self._fetch_marker(tensordict, self.done_key, "done key", B)
            if done is not None:
                valid[:-1] &= ~done[:-1].to(torch.bool)
        if self.step_count_key is not None:
            sc = self._fetch_marker(
                tensordict, self.step_count_key, "step-count key", B
            )
            if sc is not None:
                valid[:-1] &= sc[1:] == sc[:-1] + 1
        return valid

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if tensordict.batch_dims != 1:
            raise ValueError(
                "NextStateReconstructor expects a flat ``(B,)`` sample. Got "
                f"batch_size={tuple(tensordict.batch_size)}. Reshape or use a "
                "1-d storage / sampler combination."
            )
        B = tensordict.batch_size[0]
        if B < 1:
            return tensordict
        valid = self._valid_mask(tensordict, B)
        invalid = ~valid
        for k in self.keys:
            next_k = _under_next(k)
            root = tensordict.get(k)
            if (
                not root.is_floating_point()
                and isinstance(self.fill_value, float)
                and math.isnan(self.fill_value)
            ):
                raise TypeError(
                    f"NextStateReconstructor: root key {k!r} has non-floating "
                    f"dtype {root.dtype}; pass an explicit integer `fill_value` "
                    "for this key (NaN cannot be represented)."
                )
            next_view = torch.empty_like(root)
            if B >= 2:
                next_view[:-1] = root[1:]
            # Whatever sat at [-1] is overwritten below via the mask
            # (it is always invalid: no i+1 in the batch).
            invalid_expanded = invalid.reshape(B, *([1] * (root.ndim - 1))).expand_as(
                root
            )
            next_view = torch.where(
                invalid_expanded, root.new_full((), self.fill_value), next_view
            )
            tensordict.set(next_k, next_view)
        return tensordict
