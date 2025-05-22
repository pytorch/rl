# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from copy import copy

import torch
from tensordict import NestedKey, TensorDictBase, unravel_key
from tensordict.nn import ProbabilisticTensorDictModule
from tensordict.utils import is_seq_of_nested_key
from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase, Transform
from torchrl.envs.transforms.utils import _set_missing_tolerance


class KLRewardTransform(Transform):
    """A transform to add a KL[pi_current||pi_0] correction term to the reward.

    This transform is used to constrain the policy to remain close to its original
    configuration which limits overfitting when fine-tuning using RLHF.

    Args:
        actor (ProbabilisticTensorDictModule): a frozen probabilistic actor. It must
            have the following features: it must have a set of input (``in_keys``)
            and output keys (``out_keys``). It must have a ``get_dist`` method
            that outputs the distribution of the action.
        coef (:obj:`float`): the coefficient of the KL term. Defaults to ``1.0``.
        in_keys (str or list of str/tuples of str): the input key where the
            reward should be fetched. Defaults to ``"reward"``.
        out_keys (str or list of str/tuples of str): the output key where the
            reward should be written. Defaults to ``["reward", "kl_penalty", "ref_log_prob"]``.
        add_to_reward (bool): whether to add the reward term to the reward.
            Defaults to ``True``.

    .. note:: If the parameters are not differentiable (default), they will *not*
        follow the module when dtype or device casting operations will be called
        (such as :meth:`cuda`, :meth:`to` etc.). When ``requires_grad=True``,
        casting operations will work as expected.

    Examples:
        TODO

    .. note:: Because the KL formula is not always available and the parameters of the
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
        log_prob_key: NestedKey = "log_probs",
        action_key: NestedKey | None = None,
        device: torch.device | None = None,
        add_to_reward: bool = True,
    ):
        if in_keys is None:
            in_keys = self.DEFAULT_IN_KEYS
        if out_keys is None:
            out_keys = copy(in_keys)
        if len(out_keys) == len(in_keys):
            out_keys = out_keys + ["kl_penalty", "ref_log_prob"]
        elif len(out_keys) != len(in_keys) + 2:
            raise ValueError(
                "The out_keys must have the same length as the in_keys (plus two additional optional kl entries for logging)."
            )
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if not is_seq_of_nested_key(self.in_keys) or not is_seq_of_nested_key(
            self.out_keys
        ):
            raise ValueError(
                f"invalid in_keys / out_keys:\nin_keys={self.in_keys} \nout_keys={self.out_keys}"
            )
        if len(self.in_keys) != 1 or len(self.out_keys) != 3:
            raise ValueError(
                f"Only one in_key/out_key is allowed, got in_keys={self.in_keys}, out_keys={self.out_keys}."
            )
        self._out_keys = [unravel_key(out_key) for out_key in self._out_keys]

        # update the in_keys for dispatch etc
        self.in_keys = self.in_keys + actor.in_keys
        self.in_keys = [unravel_key(in_key) for in_key in self.in_keys]

        self.add_to_reward = add_to_reward
        # check that the model has parameters
        self.__dict__["actor"] = actor

        # self._buffers["actor_params"] = params.clone().detach()

        self.device = device
        self.action_key = action_key

        # find the sample log-prob key
        self.sample_log_prob_key = log_prob_key

        def find_sample_log_prob(module):
            if hasattr(module, "log_prob_key"):
                self.sample_log_prob_key = module.log_prob_key

        self.actor.apply(find_sample_log_prob)

        if not isinstance(coef, torch.Tensor):
            coef = torch.as_tensor(coef)
        self.register_buffer("coef", coef)

    def set_container(self, container: Transform | EnvBase) -> None:
        result = super().set_container(container)
        if self.action_key is None:
            parent = getattr(self, "parent", None)
            if parent is not None:
                action_keys = parent.action_keys
                if len(action_keys) != 1:
                    raise ValueError(
                        f"More than one action_key found. Please pass the `action_key` argument directly to {type(self).__name__}."
                    )
                action_key = action_keys[0]
                self.action_key = action_key
        return result

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._step(tensordict_reset, tensordict_reset)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        # run the actor on the tensordict
        action_key = self.action_key
        if action_key is None:
            raise ValueError(
                f"action_key is required. Please set a parent for the {type(self).__name__} to recover the action keys automatically, "
                f"or pass the action_key argument directly to {type(self).__name__} constructor."
            )
        action = tensordict.get(action_key, None)
        if action is None:
            if not self.missing_tolerance:
                raise RuntimeError(
                    f"Action with key {action_key} not found data {tensordict}"
                )
            # being called after reset or without action, skipping
            if self.out_keys[0] != "reward" and self.parent is not None:
                next_tensordict.set(self.out_keys[0], self.parent.reward_spec.zero())
            return next_tensordict
        if hasattr(self.actor, "log_prob"):
            if self.device is not None:
                td_device = tensordict.to(self.device)
            else:
                td_device = tensordict
            ref_log_prob = self.actor.log_prob(
                td_device, as_nested_tensor=True, layout=torch.strided
            )
        else:
            ref_log_prob = self.actor(tensordict).get(self.sample_log_prob_key)

        reward_key = self.in_keys[0]
        reward = next_tensordict.get(reward_key)
        curr_log_prob = tensordict.get(
            self.sample_log_prob_key, as_nested_tensor=True, layout=torch.strided
        )
        ref_log_prob = ref_log_prob.to(curr_log_prob.device)
        # We want the log-probs to have a similar dim to the reward
        curr_log_prob = curr_log_prob.unsqueeze(-1)
        ref_log_prob = ref_log_prob.unsqueeze(-1)

        # we use the unbiased consistent estimator of the KL: log_p(x) - log_q(x) when x ~ p(x)
        if not reward.is_nested and ref_log_prob.is_nested:
            reward = torch.nested.nested_tensor(
                [rew.expand(lp.shape) for rew, lp in zip(reward, ref_log_prob)],
                layout=torch.strided,
            )
        if ref_log_prob[0].shape != curr_log_prob[0].shape:
            # Don't check shapes if nested
            raise ValueError(
                f"the log-probability tensor shapes must match, got cur_log_prob.shape={curr_log_prob[0].shape} and log_prob.shape={ref_log_prob[0].shape}. "
                f"One possible reason is that the padding token is identical to the eos token, which means that the eos_token log_prob is truncated from the "
                f"reference model output."
            )
        if reward is not None and reward.ndim != curr_log_prob.ndim:
            raise ValueError(
                "The number of dimensions of reward must be the same as the number of dimensions of the KL "
                f"term. Got ndim={reward.ndim} and {curr_log_prob.ndim} respectively."
            )
        kl = curr_log_prob - ref_log_prob
        if self.add_to_reward:
            if reward is None:
                reward = 0
            next_tensordict.set(self.out_keys[0], reward - self.coef * kl)
        next_tensordict.set(self.out_keys[1], kl)
        next_tensordict.set(self.out_keys[2], ref_log_prob)
        return next_tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        next_td = tensordict.pop("next")
        next_td = self._step(tensordict, next_td)
        return tensordict.set("next", next_td)

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        in_key = unravel_key(self.in_keys[0])
        out_key = unravel_key(self.out_keys[0])

        if "full_observation_spec" in output_spec.keys():
            observation_spec = output_spec["full_observation_spec"]
        else:
            observation_spec = Composite(
                shape=output_spec.shape, device=output_spec.device
            )
            output_spec["full_observation_spec"] = observation_spec

        if in_key == "reward" and out_key == "reward":
            parent = self.parent

            reward_keys = parent.reward_keys
            if len(reward_keys) == 1:
                reward_key = reward_keys[0]
                shape = output_spec["full_reward_spec"][reward_key].shape
                if len(shape) > 2:
                    # For LLMs, the shape of the reward is (batch, -1, 1)
                    shape = (*shape[:-2], -1, 1)
            elif "reward" in reward_keys:
                reward_key = "reward"
                shape = output_spec["full_reward_spec"][reward_key].shape
                if len(shape) > 2:
                    # For LLMs, the shape of the reward is (batch, -1, 1)
                    shape = (*shape[:-2], -1, 1)
            else:
                shape = output_spec.shape
                shape = (*shape, -1, 1)
                reward_key = "reward"
            reward_spec = Unbounded(
                device=output_spec.device,
                shape=shape,
            )
            output_spec["full_reward_spec"] = Composite(
                {reward_key: reward_spec},
                shape=output_spec["full_reward_spec"].shape,
            )
        elif in_key == "reward":
            # TODO: we should at least allow to make this a component of the reward specs, to avoid a call during reset
            parent = self.parent
            reward_spec = output_spec["full_reward_spec"][parent.reward_key]

            shape = reward_spec.shape
            if len(shape) > 2:
                # For LLMs, the shape of the reward is (batch, -1, 1)
                shape = (*shape[:-2], -1, 1)
            reward_spec = reward_spec.clone()
            reward_spec.shape = torch.Size(shape)

            # then we need to populate the output keys
            observation_spec[out_key] = reward_spec
        else:
            observation_spec = output_spec["full_observation_spec"]
            reward_spec = observation_spec[in_key]

            shape = reward_spec.shape
            shape = (*shape[:-2], -1, 1)
            reward_spec = reward_spec.clone()
            reward_spec.shape = torch.Size(shape)

            # then we need to populate the output keys
            observation_spec[out_key] = reward_spec

        observation_spec[self.out_keys[1]] = reward_spec.clone()

        return output_spec
