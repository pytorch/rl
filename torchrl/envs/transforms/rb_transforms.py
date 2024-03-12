# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

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

    This transform is a more hyperparameter resistant version of
    :class:`~torchrl.data.postprocs.postprocs.MultiStep`:
    the replay buffer transform will make the multi-step transform insensitive
    to the collectors hyperparameters, whereas the post-process
    version will output results that are sensitive to these
    (because collectors have no memory of previous output).

    Args:
        n_steps (int): Number of steps in multi-step.
        gamma (float): Discount factor.

    Keyword Args:
        reward_key (NestedKey, optional): the reward key in the input tensordict.
            Defaults to ``"reward"``.
        done_key (NestedKey, optional): the done key in the input tensordict.
            Defaults to ``"done"``.
        terminated_key (NestedKey, optional): the terminated key in the input tensordict.
            Defaults to ``"terminated"``.
        truncated_key (NestedKey, optional): the truncated key in the input tensordict.
            Defaults to ``"truncated"``.
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
        reward_key: NestedKey | None = None,
        done_key: NestedKey | None = None,
        truncated_key: NestedKey | None = None,
        terminated_key: NestedKey | None = None,
        mask_key: NestedKey | None = None,
    ):
        super().__init__()
        self.n_steps = n_steps
        self.reward_key = reward_key
        self.done_key = done_key
        self.terminated_key = terminated_key
        self.truncated_key = truncated_key
        self.mask_key = mask_key
        self.gamma = gamma
        self.buffer = None
        self._validated = False

    @property
    def done_key(self):
        return self._done_key

    @done_key.setter
    def done_key(self, value):
        if value is None:
            value = "done"
        self._done_key = value

    @property
    def terminated_key(self):
        return self._terminated_key

    @terminated_key.setter
    def terminated_key(self, value):
        if value is None:
            value = "terminated"
        self._terminated_key = value

    @property
    def truncated_key(self):
        return self._truncated_key

    @truncated_key.setter
    def truncated_key(self, value):
        if value is None:
            value = "truncated"
        self._truncated_key = value

    @property
    def reward_key(self):
        return self._reward_key

    @reward_key.setter
    def reward_key(self, value):
        if value is None:
            value = "reward"
        self._reward_key = value

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
                reward_key=self.reward_key,
                mask_key=self.mask_key,
                n_steps=self.n_steps,
                gamma=self.gamma,
                terminated_key=self.terminated_key,
                truncated_key=self.truncated_key,
            )
            return out[..., : -self.n_steps]

    def _append_tensordict(self, data):
        if self.buffer is None:
            total_cat = data
            self.buffer = data[..., -self.n_steps :].copy()
        else:
            total_cat = torch.cat([self.buffer, data], -1)
            self.buffer = total_cat[..., -self.n_steps :].copy()
        return total_cat
