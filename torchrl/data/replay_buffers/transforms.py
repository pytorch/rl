# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from tensordict import TensorDictBase
from torchrl.envs.transforms.transforms import Transform


class _CallableTransform(Transform):
    # A wrapper around a custom callable to make it possible to transform any data type
    def __init__(self, func):
        super().__init__()
        self.func = func

    def forward(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def _call(self, tensordict):
        raise RuntimeError

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict
