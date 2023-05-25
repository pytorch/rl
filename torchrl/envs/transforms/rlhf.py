# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy

from tensordict.nn import (
    is_functional,
    make_functional,
    repopulate_module,
    TensorDictModuleBase,
)
from torchrl.envs import Transform


class KLRewardTransform(Transform):
    DEFAULT_IN_KEYS = ["reward"]

    def __init__(self, actor: TensorDictModuleBase, in_keys=None, out_keys=None):
        if in_keys is None:
            in_keys = self.DEFAULT_IN_KEYS
        if out_keys is None:
            out_keys = in_keys

        # check that the model has parameters
        params = make_functional(actor, keep_params=False)
        self.reward_model = deepcopy(actor)
        repopulate_module(actor, params)
        # we need to register these params as buffer to have `to` and similar
        # methods work properly
        # self.params = params.clone().detach()
        self._buffers["actor_params"] = params.clone().detach()
