# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy

import torch

from tensordict import TensorDictBase
from tensordict.nn import (
    make_functional,
    ProbabilisticTensorDictModule,
    repopulate_module,
    TensorDictModule,
    TensorDictModuleBase,
)
from tensordict.utils import _normalize_key, is_seq_of_nested_key, unravel_keys
from torchrl.data.tensor_specs import (
    CompositeSpec,
    TensorSpec,
    UnboundedContinuousTensorSpec,
)
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
        in_key = _normalize_key(self.in_keys[0])
        out_key = _normalize_key(self.out_keys[0])

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


class RewardModel(Transform):
    def __init__(
        self,
        reward_model: TensorDictModuleBase,
        in_keys=None,
        out_keys=None,
        on_done_only=True,
        truncated_key="truncated",
        use_next: bool = True,
        gradient_mode=False,
    ):
        self.on_done_only = on_done_only
        self.truncated_key = truncated_key
        self.use_next = use_next
        self.gradient_mode = gradient_mode
        if not isinstance(reward_model, TensorDictModuleBase):
            if out_keys is None:
                out_keys = ["reward"]
            if in_keys is None:
                raise ValueError(
                    "Reward models that are not TensorDictModuleBase require the in_keys to be specified."
                )
            reward_model = TensorDictModule(reward_model, in_keys, out_keys)
        else:
            if in_keys is None:
                in_keys = reward_model.in_keys
            elif in_keys != reward_model.in_keys:
                raise ValueError(
                    f"Got conflicting in_keys: reward_mode.in_keys={reward_model.out_keys} and in_keys={in_keys}."
                )
            if out_keys is None:
                out_keys = reward_model.out_keys
                if len(out_keys) != 1:
                    raise RuntimeError(
                        "out_keys was not provided but the number of out_keys of the reward model isn't 1."
                    )
            elif len(out_keys) != 1:
                raise ValueError("One and only one reward key should be provided.")
            if out_keys[0] not in reward_model.out_keys:
                raise ValueError(
                    f"Got conflicting out_keys: reward_mode.out_keys={reward_model.out_keys} and out_keys={out_keys}."
                )

        super().__init__(in_keys, out_keys)
        self.reward_model = reward_model

    def _call(self, tensordict: TensorDictBase) -> TensorDictBase:
        with torch.set_grad_enabled(self.gradient_mode):
            tensordict_save = tensordict
            if self.on_done_only:
                done = tensordict.get("done")
                if self.out_keys[0] in tensordict.keys(include_nested=True):
                    tensordict = tensordict.exclude(self.out_keys[0])
                truncated = tensordict.get(self.truncated_key, None)
                if truncated is not None:
                    done = done | truncated
                done = done.squeeze(-1)
                if done.shape != tensordict.shape:
                    raise ValueError(
                        "the done state shape must match the tensordict shape."
                    )
                sub_tensordict = tensordict.get_sub_tensordict(done)
            else:
                sub_tensordict = tensordict
            out = self.reward_model(sub_tensordict)
            if out is not sub_tensordict:
                raise RuntimeError(
                    f"The reward model provided to {type(self)} must modify the tensordict in place."
                )
            tensordict_save.update(tensordict, inplace=True)
        return tensordict_save

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        parent = self.parent
        reward_key = parent.reward_key
        reward_spec = UnboundedContinuousTensorSpec(shape=(*parent.batch_size, 1))
        if unravel_keys(reward_key, make_tuple=True) != unravel_keys(
            self.out_keys[0], make_tuple=True
        ):
            # we must change the reward key of the parent
            reward_key = self.out_keys[0]
        reward_spec = CompositeSpec({reward_key: reward_spec}, shape=parent.batch_size)
        return reward_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        parent = self.parent
        reward_key = parent.reward_key
        if unravel_keys(reward_key, make_tuple=True) != unravel_keys(
            self.out_keys[0], make_tuple=True
        ):
            # we should move the parent reward spec to the obs
            reward_spec = parent.reward_spec.clone()
            observation_spec[reward_key] = reward_spec
            raise Exception(f"{reward_key}, {self.out_keys[0]}")
        return observation_spec

    def _call_at_reset(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.use_next:
            self._call(tensordict.get("next"))
        else:
            self._call(tensordict)
        return tensordict
