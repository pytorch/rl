from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey
from torchrl.objectives.common import LossModule


class COMALoss(LossModule):
    """Counterfactual multi-agent policy-gradient loss.

    The actor is decentralised. The Q-value network is centralised through its
    inputs and must emit one action value per agent action under
    ``("agents", "action_value")``.
    """

    @dataclass
    class _AcceptedKeys:
        action: NestedKey = ("agents", "action")
        action_value: NestedKey = ("agents", "action_value")
        chosen_action_value: NestedKey = ("agents", "chosen_action_value")
        logits: NestedKey = ("agents", "logits")
        sample_log_prob: NestedKey = ("agents", "sample_log_prob")
        advantage_old: NestedKey = ("agents", "advantage_old")
        reward: NestedKey = ("agents", "reward")
        done: NestedKey = ("agents", "done")
        terminated: NestedKey = ("agents", "terminated")
        value_target: NestedKey = "value_target"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys
    out_keys = [
        "loss_actor",
        "loss_qvalue",
        "loss_entropy",
        "entropy",
        "pred_value",
        "target_value",
        "advantage",
    ]

    actor_network: TensorDictModule
    actor_network_params: TensorDictParams
    target_actor_network_params: TensorDictParams
    qvalue_network: TensorDictModule
    qvalue_network_params: TensorDictParams
    target_qvalue_network_params: TensorDictParams

    def __init__(
        self,
        actor_network: TensorDictModule,
        qvalue_network: TensorDictModule,
        *,
        gamma: float = 0.99,
        qvalue_loss_coef: float = 0.5,
        entropy_coef: float = 0.0,
        n_step: int = 1,
        normalize_advantage: bool = False,
        clip_epsilon: float | None = None,
    ) -> None:
        super().__init__()
        if n_step < 1:
            raise ValueError(f"n_step must be >= 1, got {n_step}.")
        if clip_epsilon is not None and not 0.0 < clip_epsilon < 1.0:
            raise ValueError(f"clip_epsilon must be in (0, 1), got {clip_epsilon}.")
        self.convert_to_functional(actor_network, "actor_network")
        self.convert_to_functional(qvalue_network, "qvalue_network", create_target_params=True)
        self.gamma = gamma
        self.qvalue_loss_coef = qvalue_loss_coef
        self.entropy_coef = entropy_coef
        self.n_step = n_step
        self.normalize_advantage = normalize_advantage
        self.clip_epsilon = clip_epsilon

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        td_copy = tensordict.clone(False)

        dist = self.actor_network.get_dist(td_copy)
        with self.qvalue_network_params.to_module(self.qvalue_network):
            self.qvalue_network(td_copy)

        chosen_action_value = self._chosen_action_value(td_copy)
        if self.clip_epsilon is not None:
            # PPO-COMA: the advantage was computed once per batch against the
            # collection policy (pi_old) and frozen; see
            # compute_counterfactual_advantage. Anchoring A and the ratio on
            # the same pi_old is what makes the clipped surrogate coherent.
            if self.tensor_keys.advantage_old not in tensordict.keys(True):
                raise KeyError(
                    "clip_epsilon is set: call compute_counterfactual_advantage on the batch before the update passes."
                )
            advantage = tensordict.get(self.tensor_keys.advantage_old).detach()
        else:
            advantage = self._counterfactual_advantage(td_copy, chosen_action_value).detach()
        if self.normalize_advantage:
            # MAPPO-style per-batch standardisation: same counterfactual
            # advantage, rescaled so the actor step size is batch-invariant.
            advantage = (advantage - advantage.mean()) / advantage.std().clamp_min(1e-6)

        log_prob = dist.log_prob(td_copy.get(self.tensor_keys.action))
        extra_outputs: dict[str, torch.Tensor] = {}
        if self.clip_epsilon is not None:
            old_log_prob = tensordict.get(self.tensor_keys.sample_log_prob).detach()
            ratio = (log_prob - old_log_prob.reshape(log_prob.shape)).exp()
            flat_advantage = advantage.squeeze(-1)
            unclipped = ratio * flat_advantage
            clipped = ratio.clamp(1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * flat_advantage
            loss_actor = -torch.min(unclipped, clipped).mean()
            outside = (ratio < 1.0 - self.clip_epsilon) | (ratio > 1.0 + self.clip_epsilon)
            extra_outputs["ratio_mean"] = ratio.detach().mean()
            extra_outputs["clip_fraction"] = outside.float().mean()
        else:
            loss_actor = -(log_prob * advantage.squeeze(-1)).mean()

        target_value = td_copy.get(self.tensor_keys.value_target)
        loss_qvalue = F.mse_loss(chosen_action_value, target_value) * self.qvalue_loss_coef

        entropy = dist.entropy().mean()
        loss_entropy = -self.entropy_coef * entropy

        return TensorDict(
            {
                "loss_actor": loss_actor,
                "loss_qvalue": loss_qvalue,
                "loss_entropy": loss_entropy,
                "entropy": entropy.detach(),
                "pred_value": chosen_action_value.detach().mean(),
                "target_value": target_value.detach().mean(),
                "advantage": advantage.detach().mean(),
                **extra_outputs,
            },
            batch_size=[],
        )

    def compute_value_target(
        self,
        tensordict: TensorDictBase,
        params: TensorDictParams | None = None,
    ) -> TensorDictBase:
        """Write the n-step Q-value target before flattening rollout data.

        The collector batch is expected to keep time in dimension 1. Targets
        follow the recursion G_k(t) = r(t) + gamma * (1 - done(t)) * G_{k-1}(t+1)
        with G_0 = target-network chosen-action Q-values, applied ``n_step``
        times. ``n_step=1`` is the TD(0) target; ``n_step=10`` matches
        EPyMARL's ``q_nstep: 10`` within episodes. Values past the end of the
        batch are treated as zero, so the last ``n_step`` transitions of a
        batch are biased low unless they end an episode.
        """
        if params is None:
            params = self.target_qvalue_network_params

        td_copy = tensordict.clone(False)
        with params.to_module(self.qvalue_network):
            self.qvalue_network(td_copy)

        chosen_action_value = self._chosen_action_value(td_copy)
        if chosen_action_value.ndim < 2:
            raise ValueError("COMALoss.compute_value_target expects an environment/time rollout batch.")

        reward = tensordict.get(("next",) + tuple(self.tensor_keys.reward))
        done = tensordict.get(("next",) + tuple(self.tensor_keys.done)).to(chosen_action_value.dtype)

        value_target = chosen_action_value
        for _ in range(self.n_step):
            next_value = torch.zeros_like(value_target)
            next_value[:, :-1] = value_target[:, 1:]
            value_target = reward + self.gamma * (1.0 - done) * next_value
        tensordict.set(self.tensor_keys.value_target, value_target.detach())
        return tensordict

    def compute_counterfactual_advantage(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Write the frozen counterfactual advantage of the collection policy.

        PPO-style anchoring: the baseline is computed from the *stored*
        collection-time logits (pi_old) — the actor is deliberately not run
        here — and the result is written once per batch under
        ``advantage_old`` so that every reuse epoch optimises the same fixed
        coefficient, coherent with the importance ratio's anchor.
        """
        td_copy = tensordict.clone(False)
        with self.qvalue_network_params.to_module(self.qvalue_network):
            self.qvalue_network(td_copy)
        chosen_action_value = self._chosen_action_value(td_copy)
        advantage = self._counterfactual_advantage(td_copy, chosen_action_value)
        tensordict.set(self.tensor_keys.advantage_old, advantage.detach())
        return tensordict

    def diagnostics(self, tensordict: TensorDictBase) -> dict[str, torch.Tensor]:
        """Return unreduced COMA quantities for trainer-side observability.

        This deliberately leaves aggregation to a reusable trainer hook: callers
        can log global summaries, per-agent values, or custom histograms without
        changing the loss used for optimization.
        """
        td_copy = tensordict.clone(False)
        with self.qvalue_network_params.to_module(self.qvalue_network):
            self.qvalue_network(td_copy)
        chosen_action_value = self._chosen_action_value(td_copy)
        advantage = self._counterfactual_advantage(td_copy, chosen_action_value)
        target_value = td_copy.get(self.tensor_keys.value_target)

        # Q-contrast measures (flat-critic hypothesis): how differentiated are
        # the Q outputs across own actions, how sensitive are they to the other
        # agents' actions, and how contrasted are the empirical targets by
        # chosen action (the reference the critic should match).
        action_value = td_copy.get(self.tensor_keys.action_value)
        action_value_spread = action_value.std(dim=-1, correction=0)

        permuted = tensordict.clone(False)
        for key in (("agents", "masked_joint_action"), ("agents", "action_without_self")):
            if key in permuted.keys(True):
                values = permuted.get(key)
                index = torch.randperm(values.shape[0], device=values.device)
                permuted.set(key, values[index])
        with self.qvalue_network_params.to_module(self.qvalue_network):
            self.qvalue_network(permuted)
        others_sensitivity = (permuted.get(self.tensor_keys.action_value) - action_value).abs().mean(dim=-1)

        flat_action = tensordict.get(self.tensor_keys.action).to(torch.float).reshape(-1, action_value.shape[-1])
        flat_target = target_value.reshape(-1, 1)
        counts = flat_action.sum(dim=0)
        means = (flat_action * flat_target).sum(dim=0) / counts.clamp_min(1.0)
        taken = counts > 0
        if int(taken.sum()) > 1:
            target_contrast = means[taken].std(correction=0)
        else:
            target_contrast = torch.zeros((), device=action_value.device)

        return {
            "advantage": advantage.detach(),
            "td_error": (target_value - chosen_action_value).detach(),
            "chosen_action_value": chosen_action_value.detach(),
            "target_value": target_value.detach(),
            "action_value_spread": action_value_spread.detach(),
            "others_sensitivity": others_sensitivity.detach(),
            "target_contrast": target_contrast.detach(),
        }

    def _chosen_action_value(self, tensordict: TensorDictBase) -> torch.Tensor:
        action = tensordict.get(self.tensor_keys.action).to(torch.float)
        action_value = tensordict.get(self.tensor_keys.action_value)
        chosen_action_value = (action * action_value).sum(dim=-1, keepdim=True)
        tensordict.set(self.tensor_keys.chosen_action_value, chosen_action_value)
        return chosen_action_value

    def _counterfactual_advantage(
        self,
        tensordict: TensorDictBase,
        chosen_action_value: torch.Tensor,
    ) -> torch.Tensor:
        action_prob = tensordict.get(self.tensor_keys.logits).softmax(dim=-1)
        action_value = tensordict.get(self.tensor_keys.action_value)
        counterfactual_baseline = (action_prob * action_value).sum(dim=-1, keepdim=True)
        return chosen_action_value - counterfactual_baseline


def add_action_without_self(tensordict: TensorDictBase) -> TensorDictBase:
    """Write each agent's other-agent actions under ``action_without_self``."""
    if ("agents", "action") not in tensordict.keys(True):
        return tensordict

    actions = tensordict.get(("agents", "action"))
    n_agents = actions.shape[-2]
    if n_agents == 1:
        tensordict.set(("agents", "action_without_self"), actions.new_empty(*actions.shape[:-2], 1, 0))
        return tensordict

    other_actions = torch.stack(
        [torch.cat([actions[..., :i, :], actions[..., i + 1 :, :]], dim=-2) for i in range(n_agents)],
        dim=-3,
    )
    tensordict.set(("agents", "action_without_self"), other_actions.reshape(*actions.shape[:-2], n_agents, -1))
    return tensordict


def add_joint_observation(tensordict: TensorDictBase) -> TensorDictBase:
    """Write the concatenated team observation under ``joint_observation``.

    Every agent row receives the same concatenation of all agents'
    observations. This mirrors EPyMARL's Gym wrapper, where the global state
    fed to the COMA critic is the concatenation of individual observations.
    """
    observation = tensordict.get(("agents", "observation"))
    n_agents = observation.shape[-2]
    joint = observation.reshape(*observation.shape[:-2], 1, -1).expand(
        *observation.shape[:-2], n_agents, n_agents * observation.shape[-1]
    )
    tensordict.set(("agents", "joint_observation"), joint.clone())
    return tensordict


def add_masked_joint_action(tensordict: TensorDictBase) -> TensorDictBase:
    """Write the joint one-hot action with each agent's own slot zeroed.

    This mirrors EPyMARL's ``COMACritic._build_inputs``: agent ``i`` sees the
    full joint action vector of size ``n_agents * n_actions`` with the block
    corresponding to its own action masked to zero, so its Q-values are not
    conditioned on the action being marginalised by the counterfactual
    baseline.
    """
    actions = tensordict.get(("agents", "action")).to(torch.float)
    n_agents, n_actions = actions.shape[-2], actions.shape[-1]
    joint = actions.reshape(*actions.shape[:-2], 1, -1).expand(*actions.shape[:-2], n_agents, n_agents * n_actions)
    own_block = torch.eye(n_agents, device=actions.device, dtype=actions.dtype).repeat_interleave(n_actions, dim=1)
    tensordict.set(("agents", "masked_joint_action"), joint * (1.0 - own_block))
    return tensordict