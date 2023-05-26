# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy

from tensordict import TensorDictBase
from tensordict.nn import (
    is_functional,
    make_functional,
    repopulate_module,
    TensorDictModuleBase,
)
from tensordict.utils import is_seq_of_nested_key
from torchrl.envs import Transform


class KLRewardTransform(Transform):
    DEFAULT_IN_KEYS = [("next", "reward"), "sample_log_prob"]

    def __init__(self, actor: TensorDictModuleBase, in_keys=None, out_keys=None):
        if in_keys is None:
            in_keys = self.DEFAULT_IN_KEYS
        if out_keys is None:
            out_keys = in_keys
        if not isinstance(in_keys, list):
            in_keys = [in_keys]
        if not isinstance(out_keys, list):
            out_keys = [out_keys]
        if not is_seq_of_nested_key(in_keys) or not is_seq_of_nested_key(out_keys):
            raise RuntimeError(f"invalid in_keys / out_keys:\nin_keys={in_keys} \nout_keys={out_keys}")
        if len(in_keys) != 2:
            raise RuntimeError("in_keys should have 2 elements: a reward key and a log-prob key.")
        if len(out_keys) != 1:
            raise RuntimeError("Only one out_key is allowed.")
        # for convenience, convert out_keys to tuples
        self.out_keys = [out_key if isinstance(out_key, tuple) else (out_key,) for out_key in self.out_keys]

        # update the in_keys for dispatch etc
        self.in_keys += actor.in_keys

        super().__init__(in_keys=in_keys, out_keys=out_keys)
        # check that the model has parameters
        params = make_functional(actor, keep_params=False, funs_to_decorate=["forward", "get_dist"])
        self.functional_actor = deepcopy(actor)
        repopulate_module(actor, params)
        # we need to register these params as buffer to have `to` and similar
        # methods work properly
        # self.params = params.clone().detach()
        self._buffers["actor_params"] = params.clone().detach()

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # run the actor on the tensordict
        dist = self.functional_actor.get_dist(
            tensordict.clone(False),
            params=self.actor_params
        )
        # get the log_prob given the original model
        log_prob = dist.log_prob(tensordict.get("action"))
        reward_key = self.in_keys[0]
        reward = tensordict.get("next").get(reward_key)
        curr_log_prob = tensordict.get("sample_log_prob")
        tensordict.set(("next", *self.out_keys[0]), )
        return log_prob
