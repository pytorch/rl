# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy

import torch

from tensordict import TensorDictBase, unravel_key
from tensordict.nn import (
    make_functional,
    ProbabilisticTensorDictModule,
    repopulate_module,
)
from tensordict.utils import is_seq_of_nested_key
from torchrl.data.tensor_specs import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.transforms.transforms import Transform


class KLRewardTransform(Transform):
    """A transform to add a KL[pi_current||pi_0] correction term to the reward.

    This transform is used to constrain the policy to remain close to its original
    configuration which limits overfitting when fine-tuning using RLHF.

    Args:
        actor (ProbabilisticTensorDictModule): a probabilistic actor. It must
            have the following features: it must have a set of input (``in_keys``)
            and output keys (``out_keys``). It must have a ``get_dist`` method
            that outputs the distribution of the action.
        coef (float): the coefficient of the KL term. Defaults to ``1.0``.
        in_keys (str or list of str/tuples of str): the input key where the
            reward should be fetched. Defaults to ``"reward"``.
        out_keys (str or list of str/tuples of str): the output key where the
            reward should be written. Defaults to ``"reward"``.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.envs import TransformedEnv
        >>> from tensordict.nn import TensorDictModule as Mod, NormalParamExtractor
        >>> from torchrl.modules import ProbabilisticActor
        >>> from tensordict import TensorDict
        >>> from torchrl.modules.distributions import TanhNormal
        >>> from torch import nn
        >>> base_env = GymEnv("Pendulum-v1")
        >>> n_obs = base_env.observation_spec["observation"].shape[-1]
        >>> n_act = base_env.action_spec.shape[-1]
        >>> module = Mod(
        ...     nn.Sequential(nn.Linear(n_obs, n_act * 2), NormalParamExtractor()),
        ...     in_keys=["observation"],
        ...     out_keys=["loc", "scale"],
        ... )
        >>> actor = ProbabilisticActor(
        ...     module,
        ...     in_keys=["loc", "scale"],
        ...     distribution_class=TanhNormal,
        ...     return_log_prob=True,
        ... )
        >>> transform = KLRewardTransform(actor, out_keys="reward_kl")
        >>> env = TransformedEnv(base_env, transform)
        >>> with torch.no_grad():
        ...     # modify the actor parameters
        ...     _ = TensorDict(dict(actor.named_parameters()), []).apply_(lambda x: x.data.copy_(x.data + 1))
        ...     td = env.rollout(3, actor)
        >>> # check that rewards have been modified
        >>> assert (td.get(("next", "reward")) != td.get(("next", "reward_kl"))).all()

    .. note::
      Because the KL formulat is not always available and the parameters of the
      original distribution may not have been recorded, we use a stochastic estimate
      of the KL divergence.

    """

    DEFAULT_IN_KEYS = ["reward"]

    def __init__(
        self,
        actor: ProbabilisticTensorDictModule,
        coef=1.0,
        in_keys=None,
        out_keys=None,
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
        tensordict.set(("next", *self.out_keys[0]), reward + self.coef * kl)
        return tensordict

    _step = _call
    forward = _call

    def transform_output_spec(self, output_spec: CompositeSpec) -> CompositeSpec:
        output_spec = super().transform_output_spec(output_spec)
        # todo: here we'll need to use the reward_key once it's implemented
        # parent = self.parent
        in_key = unravel_key(self.in_keys[0])
        out_key = unravel_key(self.out_keys[0])

        if in_key == "reward" and out_key == "reward":
            parent = self.parent
            reward_spec = UnboundedContinuousTensorSpec(
                device=output_spec.device,
                shape=output_spec["_reward_spec"][parent.reward_key].shape,
            )
            output_spec["_reward_spec"] = CompositeSpec(
                {parent.reward_key: reward_spec},
                shape=output_spec["_reward_spec"].shape,
            )
        elif in_key == "reward":
            parent = self.parent
            reward_spec = UnboundedContinuousTensorSpec(
                device=output_spec.device,
                shape=output_spec["_reward_spec"][parent.reward_key].shape,
            )
            # then we need to populate the output keys
            observation_spec = output_spec["_observation_spec"]
            observation_spec[out_key] = reward_spec
        else:
            observation_spec = output_spec["_observation_spec"]
            reward_spec = UnboundedContinuousTensorSpec(
                device=output_spec.device, shape=observation_spec[in_key].shape
            )
            # then we need to populate the output keys
            observation_spec[out_key] = reward_spec
        return output_spec
