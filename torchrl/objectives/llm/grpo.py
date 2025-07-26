# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import (
    is_tensor_collection,
    NestedKey,
    TensorClass,
    TensorDict,
    TensorDictBase,
    TensorDictParams,
)
from tensordict.nn import (
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
)
from tensordict.utils import expand_as_right
from torch import distributions as d
from torchrl._utils import logger as torchrl_logger
from torchrl.envs.transforms.transforms import Transform
from torchrl.modules.llm import LLMWrapperBase
from torchrl.objectives.ppo import ClipPPOLoss
from torchrl.objectives.utils import _reduce, _sum_td_features


class GRPOLossOutput(TensorClass["nocast"]):
    """GRPO Loss Output."""

    loss_objective: torch.Tensor
    clip_fraction: torch.Tensor
    kl_approx: torch.Tensor
    ESS: torch.Tensor
    entropy: torch.Tensor | None = None
    loss_entropy: torch.Tensor | None = None
    loss_kl_to_ref: torch.Tensor | None = None
    kl_to_ref: torch.Tensor | None = None
    loss_kl_to_inference: torch.Tensor | None = None
    kl_to_inference: torch.Tensor | None = None


class GRPOLoss(ClipPPOLoss):
    """GRPO loss.

    The clipped importance weighted loss is computed as follows:
        loss = -min( weight * advantage, min(max(weight, 1-eps), 1+eps) * advantage)

    Args:
        actor_network (LLMWrapperBase): policy operator.

    .. note::
        It is critical to keep your model in eval mode during GRPO training to ensure deterministic behavior and correct
        importance sampling. A mismatch between train and eval modes is a common cause of instability or failure to learn
        in RL post-training.

    .. note::
        The Effective Sample Size (ESS) is a key diagnostic metric in GRPO. ESS measures the effective number of samples
        in the batch, computed as the inverse of the sum of the squared importance weights.
        A value of 1 indicates that all importance weights are equal (ideal case). If ESS drops or increases significantly,
        it usually indicates a problem with the model configuration, such as a train/eval mode mismatch or a large policy update.

    .. note::
        The masking_strategy parameter is crucial for LLM training scenarios. It determines which tokens are included
        in the loss computation:
        - "sft": Only response tokens (excludes prompt tokens) - suitable for single-turn conversations
        - "rlhf": Only assistant tokens (excludes user/system tokens) - suitable for multi-turn conversations
        - "generic": All valid tokens (excludes padding tokens) - suitable for generic scenarios

        The masking strategy must match the strategy used for advantage computation to avoid shape mismatches.

    Keyword Args:
        clip_epsilon (scalar, optional): weight clipping threshold in the clipped PPO loss equation.
            default: 0.2
        entropy_bonus (bool, optional): if ``True``, an entropy bonus will be added to the
            loss to favour exploratory policies.
        samples_mc_entropy (int, optional): if the distribution retrieved from the policy
            operator does not have a closed form
            formula for the entropy, a Monte-Carlo estimate will be used.
            ``samples_mc_entropy`` will control how many
            samples will be used to compute this estimate.
            Defaults to ``1``.
        entropy_coeff (scalar, optional): entropy multiplier when computing the total loss.
            Defaults to ``0.01``.
        advantage_key (str, optional): [Deprecated, use set_keys(advantage_key=advantage_key) instead]
            The input tensordict key where the advantage is
            expected to be written. Defaults to ``"advantage"``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.
        clip_value (bool or float, optional): If a ``float`` is provided, it will be used to compute a clipped
            version of the value prediction with respect to the input tensordict value estimate and use it to
            calculate the value loss. The purpose of clipping is to limit the impact of extreme value predictions,
            helping stabilize training and preventing large updates. However, it will have no impact if the value
            estimate was done by the current version of the value estimator. If instead ``True`` is provided, the
            ``clip_epsilon`` parameter will be used as the clipping threshold. If not provided or ``False``, no
            clipping will be performed. Defaults to ``False``.
        kl_to_ref_coeff (float, optional): coefficient for the KL divergence to the reference policy. Defaults to ``None`` (no KL divergence).
        kl_to_inference_coeff (float, optional): coefficient for the KL divergence to the inference policy. Defaults to ``None`` (no KL divergence).
        device (torch.device, optional): device of the buffers. Defaults to ``None``.
        masking_strategy (Literal["sft", "rlhf", "generic"], optional): The masking strategy to use for distribution creation.
            - "sft": Use prompt masking (response tokens only, suitable for single-turn)
            - "rlhf": Use assistant masking (assistant tokens only, suitable for multi-turn)
            - "generic": Use attention masking (all valid tokens)
            Defaults to "sft" since we can't guarantee assistant masks are available.

            .. note:: Parameters and buffers from the policy / critic will not be cast to that device to ensure that
                the storages match the ones that are passed to other components, such as data collectors.
    """

    actor_network: LLMWrapperBase
    critic_network: TensorDictModule
    actor_network_params: TensorDictParams
    critic_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    target_critic_network_params: TensorDictParams

    @dataclass
    class _AcceptedKeys(ClipPPOLoss._AcceptedKeys):
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values
        """

        ref_log_probs: NestedKey = ("next", "ref_log_probs", "full")

    def __init__(
        self,
        actor_network: LLMWrapperBase | None = None,
        *,
        clip_epsilon: float = 0.2,
        entropy_bonus: bool = True,
        samples_mc_entropy: int = 1,
        entropy_coeff: float = 0.01,
        gamma: float | None = None,
        reduction: str | None = None,
        clip_value: bool | float | None = None,
        kl_to_ref_coeff: float | None = None,
        kl_to_inference_coeff: float | None = None,
        device: torch.device | None = None,
        masking_strategy: Literal["sft", "rlhf", "generic"] = "sft",
        **kwargs,
    ):
        # Define clipping of the value loss
        if isinstance(clip_value, bool):
            clip_value = clip_epsilon if clip_value else None

        super().__init__(
            actor_network,
            critic_network=None,
            entropy_bonus=entropy_bonus,
            samples_mc_entropy=samples_mc_entropy,
            entropy_coeff=entropy_coeff,
            gamma=gamma,
            separate_losses=False,
            reduction=reduction,
            clip_value=clip_value,
            functional=False,
            device=device,
            **kwargs,
        )
        # We don't want to use the string action but the tokens
        self._set_in_keys()
        self.masking_strategy = masking_strategy
        # Always use the full tokens for the action
        self.set_keys(sample_log_prob=("log_probs", "full"), action=("tokens", "full"))
        # TODO: make this a buffer
        self.kl_to_ref_coeff = kl_to_ref_coeff
        self.kl_to_inference_coeff = kl_to_inference_coeff

    def _get_cur_log_prob(self, tensordict):
        """Override to use LLM-specific distribution with explicit masking strategy.

        This ensures that the loss is computed with the correct masking strategy,
        and provides helpful error messages when there are shape mismatches.
        """
        if isinstance(
            self.actor_network,
            (ProbabilisticTensorDictSequential, ProbabilisticTensorDictModule),
        ) or hasattr(self.actor_network, "get_dist"):
            # Use the specified masking strategy
            #  dists are always defined over the whole sequence, so we can re-use the mask as the dist will always
            #  be a MaskedCategorical
            # TODO: eventually, we want to always use `get_dist` and just pass the key of the mask
            #  Masks should contain: prompt and response masks, assistant, and attention.
            #  Additionally, we should make sure that the masks are properly updated when log-probs is called (using vllm and transformers)
            #  because in some instances it looks like they can be overwritten with None values.
            if self.masking_strategy == "sft" and hasattr(
                self.actor_network, "_get_sft_dist"
            ):
                dist = self.actor_network._get_sft_dist(tensordict)
            elif self.masking_strategy == "rlhf" and hasattr(
                self.actor_network, "_get_rlhf_dist"
            ):
                dist = self.actor_network._get_rlhf_dist(tensordict)
            elif self.masking_strategy == "generic" and hasattr(
                self.actor_network, "_get_generic_dist"
            ):
                dist = self.actor_network._get_generic_dist(tensordict)
            elif hasattr(self.actor_network, "get_dist"):
                # Fallback to generic distribution method
                dist = self.actor_network.get_dist(
                    tensordict,
                    logits_key="logits",
                )
            else:
                raise NotImplementedError(
                    f"Actor network must have get_dist method or the appropriate method for "
                    f"masking strategy '{self.masking_strategy}'."
                )

            action = tensordict.get(
                self.tensor_keys.action,
                as_padded_tensor=True,
                padding_side="left",
                padding_value=-100,
            )
            log_prob = dist.log_prob(action)
        else:
            raise NotImplementedError(
                "Only probabilistic modules from tensordict.nn are currently supported. "
                "If you need to implement a custom logic to retrieve the log-probs (to compute "
                "the PPO objective) or the distribution (for the PPO entropy), please augment "
                f"the {type(self).__class__} by implementing your own logic in _get_cur_log_prob."
            )
        return log_prob, dist, False

    def forward(self, tensordict: TensorDictBase) -> GRPOLossOutput:
        # Some sanity checks and housekeeping:
        # - We may not have the tokens yet. If not, we will use the tokenizer of the actor to tokenize the text.
        #   We default to history rather than text because the history will account for multiturn, or multimodal inputs.
        if self.tensor_keys.action not in tensordict:
            raise ValueError

        tensordict = tensordict.copy()
        advantage = tensordict.get(
            self.tensor_keys.advantage, None, as_padded_tensor=True
        )
        log_weight, dist, kl_approx = self._log_weight(
            tensordict, adv_shape=advantage.shape[:-1]
        )
        mask = dist.mask
        # ESS for logging
        with torch.no_grad():
            # In theory, ESS should be computed on particles sampled from the same source. Here we sample according
            # to different, unrelated trajectories, which is not standard. Still, it can give an idea of the weights'
            # dispersion.
            lw = log_weight.squeeze(-1)[mask]
            batch = mask.sum()
            ess = (2 * lw.logsumexp(0) - (2 * lw).logsumexp(0)).exp()

        if advantage.ndim != log_weight.ndim:
            raise ValueError(
                f"advantage and log_weight must have the same number of dimensions, got {advantage.ndim=} and {log_weight.ndim=}"
            )
        gain1 = log_weight.exp() * advantage

        log_weight_clip = log_weight.clamp(*self._clip_bounds)
        clip_fraction = (log_weight_clip != log_weight).to(log_weight.dtype).mean()
        ratio = log_weight_clip.exp()
        gain2 = ratio * advantage

        gain = torch.stack([gain1, gain2], -1).min(dim=-1).values
        td_out = TensorDict({"loss_objective": -gain})
        td_out.set("clip_fraction", clip_fraction)
        td_out.set("kl_approx", kl_approx.detach().mean())  # for logging

        if self.entropy_bonus:
            entropy = self._get_entropy(dist, adv_shape=advantage.shape[:-1])
            if is_tensor_collection(entropy):
                # Reports the entropy of each action head.
                td_out.set("composite_entropy", entropy.detach())
                entropy = _sum_td_features(entropy)
            td_out.set("entropy", entropy.detach().mean())  # for logging
            td_out.set("loss_entropy", -self.entropy_coeff * entropy)
        if self._has_critic:
            loss_critic, value_clip_fraction = self.loss_critic(tensordict)
            td_out.set("loss_critic", loss_critic)
            if value_clip_fraction is not None:
                td_out.set("value_clip_fraction", value_clip_fraction)

        td_out.set("ESS", _reduce(ess / batch, self.reduction))
        td_out = td_out.named_apply(
            lambda name, value: _reduce(
                value, reduction=self.reduction, mask=mask
            ).squeeze(-1)
            if name.startswith("loss_")
            else value,
        )
        if self.kl_to_ref_coeff is not None:
            # FIXME: parameterize this
            loss_kl, kl_penalty = self._kl_to_ref(
                tensordict,
                mask=mask,
                dist=dist,
                ref_log_prob=tensordict.get(
                    self.tensor_keys.ref_log_probs,
                    as_padded_tensor=True,
                    padding_side="left",
                    padding_value=0.0,
                ),
            )
            td_out["loss_kl_to_ref"] = loss_kl
            td_out["kl_to_ref"] = kl_penalty.detach()
        if self.kl_to_inference_coeff is not None:
            loss_kl, kl_penalty = self._kl_to_ref(
                tensordict,
                key=self.tensor_keys.sample_log_prob,
                coeff=self.kl_to_inference_coeff,
                mask=mask,
                dist=dist,
            )
            td_out["loss_kl_to_inference"] = loss_kl
            td_out["kl_to_inference"] = kl_penalty.detach()
        del tensordict["_cur_log_prob"]
        return GRPOLossOutput.from_tensordict(td_out)

    def _kl_to_ref(
        self,
        tensordict: TensorDictBase,
        key: NestedKey = ("next", "ref_log_prob"),
        ref_log_prob: torch.Tensor | None = None,
        coeff: float | None = None,
        mask: torch.Tensor | None = None,
        dist: d.Distribution | None = None,
    ):
        if coeff is None:
            coeff = self.kl_to_ref_coeff
        # TODO: customize this
        if ref_log_prob is None:
            ref_log_prob = tensordict.get(
                key,
                as_padded_tensor=True,
                padding_side="left",
                padding_value=0.0,
            )
            if ref_log_prob is None:
                raise KeyError(
                    f"Couldn't find the ref log-prob {key} in the input data ({tensordict.keys(True)=})."
                )
            ref_log_prob = ref_log_prob.squeeze(-1)
        cur_log_prob = tensordict.get("_cur_log_prob")
        # TODO: remove this
        if cur_log_prob.shape != ref_log_prob.shape:
            raise ValueError(
                f"cur_log_prob and ref_log_prob must have the same shape, got {cur_log_prob.shape=} and {ref_log_prob.shape=}"
            )
        if mask is not None:
            ref_log_prob = torch.where(
                expand_as_right(mask, ref_log_prob), ref_log_prob, 0.0
            )
            cur_log_prob = torch.where(
                expand_as_right(mask, cur_log_prob), cur_log_prob, 0.0
            )
        diff = ref_log_prob - cur_log_prob
        kl_penalty = (diff.expm1() - diff).mean()
        return coeff * kl_penalty, kl_penalty

    def _log_weight(
        self, tensordict: TensorDictBase, adv_shape: torch.Size
    ) -> tuple[torch.Tensor, d.Distribution, torch.Tensor]:

        cur_log_prob, dist, is_composite = self._get_cur_log_prob(tensordict)

        prev_log_prob = tensordict.get(
            self.tensor_keys.sample_log_prob,
            as_padded_tensor=True,
            padding_side="left",
            padding_value=0.0,
        )

        if prev_log_prob is None:
            raise KeyError(
                f"Couldn't find the log-prob {self.tensor_keys.sample_log_prob} in the input data."
            )
        if prev_log_prob.requires_grad:
            raise RuntimeError(
                f"tensordict stored {self.tensor_keys.sample_log_prob} requires grad."
            )

        # Check for shape mismatches and provide helpful error messages
        if cur_log_prob.shape != prev_log_prob.shape:
            # Try to provide helpful debugging information
            error_msg = (
                f"Shape mismatch detected in GRPOLoss: current log-prob shape {cur_log_prob.shape} "
                f"!= previous log-prob shape {prev_log_prob.shape}. "
                f"This usually indicates a mismatch between the masking strategy used for "
                f"advantage computation and the masking strategy used for loss computation.\n"
                f"Current masking strategy: '{self.masking_strategy}'\n"
                f"Possible solutions:\n"
                f"1. If using RLHF (multi-turn conversations), set masking_strategy='rlhf'\n"
                f"2. If using SFT (single-turn conversations), set masking_strategy='sft'\n"
                f"3. If using generic scenarios, set masking_strategy='generic'\n"
                f"4. Ensure the advantage was computed with the same masking strategy as the loss"
            )
            raise ValueError(error_msg)

        attention_mask = dist.mask
        cur_log_prob = torch.where(
            expand_as_right(attention_mask, cur_log_prob), cur_log_prob, 0.0
        )
        prev_log_prob = torch.where(
            expand_as_right(attention_mask, prev_log_prob), prev_log_prob, 0.0
        )

        if is_composite:
            raise NotImplementedError
        log_weight = (cur_log_prob - prev_log_prob).unsqueeze(-1)
        if is_tensor_collection(log_weight):
            log_weight = _sum_td_features(log_weight)
            log_weight = log_weight.view(adv_shape).unsqueeze(-1)

        kl_approx = (prev_log_prob - cur_log_prob).unsqueeze(-1)
        if is_tensor_collection(kl_approx):
            kl_approx = _sum_td_features(kl_approx)

        tensordict.set("_cur_log_prob", cur_log_prob)

        return log_weight, dist, kl_approx


class MCAdvantage(Transform):
    """Monte-Carlo advantage computation engine.

    When writing on a replay buffer, this transform keeps track of the existing trajectories with a similar
    initial prompt and holds a queue for that particular prompt in memory.
    When that queue hits a certain length, the advantage is computed by normalizing the rewards across all the
    steps of all the trajectories.

    This transform assumes that :meth:`~torchrl.data.ReplayBuffer.add` and :meth:`~torchrl.data.ReplayBuffer.extend`
    are executed with completed trajectories (i.e., trajectories that end up with a done state). If this is not the
    case, an exception is raised.

    .. warning:: This transform will flatten the input tensordicts and therefore is not compatible yet with replay
        buffers hosting storages of more than one dimension.

    Args:
        grpo_size (int): Number of trajectories to keep in memory for the advantage computation.
        prompt_key (NestedKey): Key to the prompt in the tensordict. Defaults to ("text", "prompt").
        rewards_key (NestedKey): Key to the rewards in the tensordict. Defaults to ("next", "reward").
        advantage_key (NestedKey): Key to the advantage in the tensordict. Defaults to "advantage".
        done_key (NestedKey): Key to the done state in the tensordict. Defaults to ("next", "done").
        verbose (bool): Whether to print verbose information. Defaults to `False`.

    """

    def __init__(
        self,
        grpo_size: int,
        prompt_key: NestedKey = "query",
        rewards_key: NestedKey = ("next", "reward"),
        advantage_key: NestedKey = "advantage",
        done_key: NestedKey = ("next", "done"),
        verbose: bool = False,
    ):
        super().__init__()
        self.in_keys = [prompt_key, rewards_key, done_key]
        self.out_keys = [advantage_key]
        self.prompt_key = prompt_key
        self.rewards_key = rewards_key
        self.advantage_key = advantage_key
        self.done_key = done_key
        self.queues = defaultdict(lambda: deque(maxlen=grpo_size))
        self.grpo_size = grpo_size
        self.verbose = verbose

    def forward(self, tensordict: TensorDictBase) -> GRPOLossOutput:
        return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.verbose:
            torchrl_logger.info(
                f"Invoking MCAdvantage.\nData size: {tensordict.shape}.\nCurrent queue size: {len(self.queues)}.\nTotal queue content: {sum(len(q) for q in self.queues.values())}"
            )
        # Tensordict can be any number of dims, but it must contain entire trajectories
        if tensordict.ndim == 1:
            # Check how many done states we have
            num_done = tensordict[self.done_key].sum()
            if num_done > 1:
                done_idx = tensordict[self.done_key].nonzero(as_tuple=True)[0] + 1
                splits = torch.cat([done_idx.new_zeros((1,)), done_idx], dim=0).diff()
                tensordicts = tensordict.split(splits)
                tensordicts = [self._inv_call(td) for td in tensordicts]
                tensordicts = [td for td in tensordicts if td is not None]
                return torch.cat(tensordicts) if tensordicts else None
            # Then we have a single trajectory
            if not tensordict[-1][self.done_key].all():
                raise RuntimeError("Expected the trajectory to be done.")
            prompt = tensordict[0][self.prompt_key]
            if not isinstance(prompt, str):
                raise TypeError(f"Expected a string as prompt, got {type(prompt)=}")
            self.queues[prompt].append(tensordict)
            if len(self.queues[prompt]) == self.grpo_size:
                if self.verbose:
                    torchrl_logger.info(f"Computing advantage for {prompt=}")
                # Cat is the most robust way to combine the trajs
                tds = torch.cat(list(self.queues[prompt]), -1)
                del self.queues[prompt]
                # Collect rewards
                reward = tds.get(self.rewards_key, as_nested_tensor=True)
                reward_mean = reward.values().mean()
                reward_scale = reward.values().std()
                advantage = (reward - reward_mean) / reward_scale.clamp_min(1e-6)
                if self.verbose:
                    torchrl_logger.info(f"Advantage: {reward_mean=} {reward_scale=}")
                tds.set(self.advantage_key, advantage)
                return tds
            return
        elif tensordict.ndim > 2:
            # keep the time dim at the end
            tensordict = tensordict.flatten(0, -2)
        trajs = tensordict.unbind(0)
        # Iterate over the trajectories
        result = []
        for traj in trajs:
            td_out = self._inv_call(traj)
            if td_out is None:
                continue
            result.append(td_out)
        if result:
            return torch.cat(result, 0)
        return
