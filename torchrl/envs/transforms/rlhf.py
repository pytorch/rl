# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import copy, deepcopy

import torch
from tensordict import TensorDictBase, unravel_key
from tensordict.nn import (
    make_functional,
    ProbabilisticTensorDictModule,
    repopulate_module,
    TensorDictParams,
)
from tensordict.utils import is_seq_of_nested_key
from torch import nn
from torchrl.data.tensor_specs import CompositeSpec, UnboundedContinuousTensorSpec
from torchrl.envs.transforms.transforms import Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance


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
        requires_grad (bool, optional): if ``True``, the frozen parameters will
            consist of differentiable clones of the original params.
            Defaults to ``False``.

    .. note:: If the parameters are not differentiable (default), they will *not*
        follow the module when dtype or device casting operations will be called
        (such as :meth:`~.cuda`, :meth:`~.to` etc.). When ``requires_grad=True``,
        casting operations will work as expected.

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

    .. note:: Because the KL formulat is not always available and the parameters of the
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
        requires_grad=False,
    ):
        if in_keys is None:
            in_keys = self.DEFAULT_IN_KEYS
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if not is_seq_of_nested_key(self.in_keys) or not is_seq_of_nested_key(
            self.out_keys
        ):
            raise ValueError(
                f"invalid in_keys / out_keys:\nin_keys={self.in_keys} \nout_keys={self.out_keys}"
            )
        if len(self.in_keys) != 1 or len(self.out_keys) != 1:
            raise ValueError(
                f"Only one in_key/out_key is allowed, got in_keys={self.in_keys}, out_keys={self.out_keys}."
            )
        # for convenience, convert out_keys to tuples
        self._out_keys = [
            out_key if isinstance(out_key, tuple) else (out_key,)
            for out_key in self._out_keys
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

        def _make_detached_param(x):

            if isinstance(x, nn.Parameter):
                # we need an nn.Parameter since some modules (RNN) require nn.Parameters
                return nn.Parameter(x.data.clone(), requires_grad=requires_grad)
            elif x.requires_grad:
                raise ValueError(
                    "Encountered a value that requires gradients but is not an nn.Parameter instance."
                )
            return x.clone()

        self.frozen_params = params.apply(_make_detached_param)
        if requires_grad:
            # includes the frozen params/buffers in the module parameters/buffers
            self.frozen_params = TensorDictParams(self.frozen_params, no_convert=True)

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

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

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

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        with tensordict.unlock_():
            return self._call(tensordict.set("next", next_tensordict)).pop("next")

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
                shape=output_spec["full_reward_spec"][parent.reward_key].shape,
            )
            output_spec["full_reward_spec"] = CompositeSpec(
                {parent.reward_key: reward_spec},
                shape=output_spec["full_reward_spec"].shape,
            )
        elif in_key == "reward":
            parent = self.parent
            reward_spec = UnboundedContinuousTensorSpec(
                device=output_spec.device,
                shape=output_spec["full_reward_spec"][parent.reward_key].shape,
            )
            # then we need to populate the output keys
            observation_spec = output_spec["full_observation_spec"]
            observation_spec[out_key] = reward_spec
        else:
            observation_spec = output_spec["full_observation_spec"]
            reward_spec = UnboundedContinuousTensorSpec(
                device=output_spec.device, shape=observation_spec[in_key].shape
            )
            # then we need to populate the output keys
            observation_spec[out_key] = reward_spec
        return output_spec
