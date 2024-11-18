# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import List

import torch

from tensordict import NestedKey, TensorDictBase
from torchrl.data.postprocs.postprocs import _multi_step_func
from torchrl.envs.transforms.transforms import Transform


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
        reward_keys: List[NestedKey] | None = None,
        done_key: NestedKey | None = None,
        done_keys: List[NestedKey] | None = None,
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
        if total_cat.shape[-1] >= self.n_steps:
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
