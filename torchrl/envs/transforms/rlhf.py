# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy

import torch

from tensordict import TensorDictBase
from tensordict.nn import (
    is_functional,
    make_functional,
    repopulate_module,
    TensorDictModuleBase,
)
from tensordict.utils import _normalize_key, is_seq_of_nested_key
from torchrl.data import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs import Transform


class KLRewardTransform(Transform):
    DEFAULT_IN_KEYS = ["reward"]

    def __init__(
        self, actor: TensorDictModuleBase, coef=1.0, in_keys=None, out_keys=None
    ):
        if in_keys is None:
            in_keys = self.DEFAULT_IN_KEYS
        if out_keys is None:
            out_keys = in_keys
        if not isinstance(in_keys, list):
            in_keys = [in_keys]
        if not isinstance(out_keys, list):
            out_keys = [out_keys]
        if not is_seq_of_nested_key(in_keys) or not is_seq_of_nested_key(out_keys):
            raise ValueError(
                f"invalid in_keys / out_keys:\nin_keys={in_keys} \nout_keys={out_keys}"
            )
        if len(in_keys) != 1 or len(out_keys) != 1:
            raise ValueError(
                f"Only one in_key/out_key is allowed, got in_keys={in_keys}, out_keys={out_keys}."
            )
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        # for convenience, convert out_keys to tuples
        self.out_keys = [
            out_key if isinstance(out_key, tuple) else (out_key,)
            for out_key in self.out_keys
        ]

        # update the in_keys for dispatch etc
        self.in_keys = self.in_keys + actor.in_keys

        # check that the model has parameters
        params = make_functional(
            actor, keep_params=False, funs_to_decorate=["forward", "get_dist"]
        )
        self.functional_actor = deepcopy(actor)
        repopulate_module(actor, params)
        # we need to register these params as buffer to have `to` and similar
        # methods work properly
        self.frozen_params = params.clone().detach()
        # self._buffers["actor_params"] = params.clone().detach()

        # find the sample log-prob key
        self.sample_log_prob_key = "sample_log_prob"

        def find_sample_log_prob(module):
            if hasattr(module, "SAMPLE_LOG_PROB_KEY"):
                self.sample_log_prob_key = module.SAMPLE_LOG_PROB_KEY

        self.functional_actor.apply(find_sample_log_prob)

        if not isinstance(coef, torch.Tensor):
            coef = torch.tensor(coef)
        self.register_buffer("coef", coef)

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # run the actor on the tensordict
        action = tensordict.get("action", None)
        if action is None:
            # being called after reset or without action, skipping
            if self.out_keys[0] != ("reward",) and self.parent is not None:
                tensordict.set(self.out_keys[0], self.parent.reward_spec.zero())
            return tensordict
        dist = self.functional_actor.get_dist(
            tensordict.clone(False), params=self.frozen_params
        )
        # get the log_prob given the original model
        log_prob = dist.log_prob(action)
        reward_key = self.in_keys[0]
        reward = tensordict.get("next").get(reward_key)
        curr_log_prob = tensordict.get(self.sample_log_prob_key)
        # we use the unbiased consistent estimator of the KL: log_p(x) - log_q(x) when x ~ p(x)
        kl = (curr_log_prob - log_prob).view_as(reward)
        tensordict.set(
            ("next", *self.out_keys[0]), reward + self.coef * kl
        )
        return tensordict

    _step = _call
    forward = _call

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
        output_spec = super().transform_output_spec(output_spec)
        output_spec.unlock_()
        # todo: here we'll need to use the reward_key once it's implemented
        # parent = self.parent
        in_key = _normalize_key(self.in_keys[0])
        out_key = _normalize_key(self.out_keys[0])
        if in_key == "reward" and out_key == "reward":
            reward_spec = UnboundedContinuousTensorSpec(
                device=output_spec.device, shape=output_spec["reward"].shape
            )
            output_spec["reward"] = reward_spec
        elif in_key == "reward":
            reward_spec = UnboundedContinuousTensorSpec(
                device=output_spec.device, shape=output_spec["reward"].shape
            )
            # then we need to populate the output keys
            observation_spec = output_spec["observation"]
            observation_spec[out_key] = reward_spec
        else:
            observation_spec = output_spec["observation"]
            reward_spec = UnboundedContinuousTensorSpec(
                device=output_spec.device, shape=observation_spec[in_key].shape
            )
            # then we need to populate the output keys
            observation_spec[out_key] = reward_spec
        output_spec.lock_()
        return output_spec
