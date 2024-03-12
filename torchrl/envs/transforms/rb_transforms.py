# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from tensordict import TensorDictBase, NestedKey
from torchrl.data.postprocs.postprocs import _multi_step_func
from torchrl.envs import Transform
import torch

class MultiStepTransform(Transform):
    def __init__(self, n_steps, gamma, *, reward_key:NestedKey|None=None, done_key:NestedKey|None=None,truncated_key:NestedKey|None=None,terminated_key:NestedKey|None=None, mask_key:NestedKey|None=None):
        super().__init__()
        self.n_steps = n_steps
        self.reward_key = reward_key
        self.done_key = done_key
        self.terminated_key = terminated_key
        self.truncated_key = truncated_key
        self.mask_key = mask_key
        self.gamma = gamma
        self.buffer = None

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

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        print('before')
        print(tensordict.get(("next", "done")).any())
        total_cat = self._append_tensordict(tensordict)
        print('after')
        print(total_cat.get(("next", "done")).any())
        out = _multi_step_func(
            total_cat,
            done_key=self.done_key,
            reward_key=self.reward_key,
            mask_key=self.mask_key,
            n_steps=self.n_steps,
            gamma=self.gamma,
            terminated_key=self.terminated_key,
            truncated_key=self.truncated_key,
            )[..., :-self.n_steps]
        if out.numel():
            return out
        return None

    def _append_tensordict(self, data):
        if self.buffer is None:
            total_cat = data
            self.buffer = data[..., -self.n_steps:].copy()
        else:
            total_cat = torch.cat([self.buffer, data], -1)
            self.buffer = total_cat[..., -self.n_steps:].copy()
        return total_cat
