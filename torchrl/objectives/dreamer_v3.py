# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 loss modules.

Implements the three loss modules from DreamerV3 (Mastering Diverse Domains in
World Models, Hafner et al. 2023, https://arxiv.org/abs/2301.04104):

- :class:`DreamerV3ModelLoss` — world model (KL balancing + symlog reconstruction)
- :class:`DreamerV3ActorLoss` — actor (REINFORCE + entropy bonus)
- :class:`DreamerV3ValueLoss` — value function (symlog MSE or two-hot CE)

Utility functions :func:`symlog`, :func:`symexp`, :func:`two_hot_encode`,
:func:`two_hot_decode`, and :func:`two_hot_cross_entropy` are also exported for
use in custom models.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import TensorDictModule, TensorDictModuleBase
from tensordict.utils import NestedKey, unravel_key

from torchrl._utils import _maybe_record_function_decorator
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.modules.distributions import HAS_ENTROPY
from torchrl.modules.distributions.utils import rsample_and_log_prob
from torchrl.modules.functional import symexp as _symexp, symlog as symlog
from torchrl.modules.models.model_based_v3 import (  # noqa: F401
    _default_bins,
    _DEFAULT_NUM_BINS,
    _unimix_probs,
    two_hot_cross_entropy as two_hot_cross_entropy,
    two_hot_decode as _two_hot_decode,
    two_hot_encode as _two_hot_encode,
)
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_ERROR,
    dispatch_value_estimator,
    hold_out_net,
    ValueEstimators,
)
from torchrl.objectives.value import ValueEstimatorBase

symexp = _symexp
two_hot_decode = _two_hot_decode
two_hot_encode = _two_hot_encode

# ---------------------------------------------------------------------------
# KL balancing for categorical distributions (DreamerV3 §3)
# ---------------------------------------------------------------------------


def categorical_kl_terms(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    free_nats: float = 1.0,
    unimix: float = 0.01,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return DreamerV3 dynamics and representation KL losses.

    The dynamics term stops gradients through the posterior and the
    representation term stops gradients through the prior. KL divergence is
    summed over the stochastic categoricals before applying the free-nat
    threshold, matching the aggregated one-hot distribution used by the
    reference DreamerV3 implementation.

    Args:
        posterior_logits (torch.Tensor): Posterior logits with shape
            ``[..., num_categoricals, num_classes]``.
        prior_logits (torch.Tensor): Prior logits with the same shape.
        free_nats (float, optional): Minimum aggregated KL in nats. Defaults to
            ``1.0``.
        unimix (float, optional): Fraction of uniform probability mixed into
            each categorical. Defaults to ``0.01``.

    Returns:
        A pair containing the scalar dynamics and representation KL losses.

    Examples:
        >>> import torch
        >>> from torchrl.objectives import categorical_kl_terms
        >>> posterior = torch.randn(2, 4, 8, requires_grad=True)
        >>> prior = torch.randn(2, 4, 8, requires_grad=True)
        >>> dynamics, representation = categorical_kl_terms(posterior, prior)
        >>> dynamics.shape, representation.shape
        (torch.Size([]), torch.Size([]))
    """
    posterior = _unimix_probs(posterior_logits, unimix)
    prior = _unimix_probs(prior_logits, unimix)
    posterior_log = posterior.log()
    prior_log = prior.log()

    dynamics = (posterior.detach() * (posterior_log.detach() - prior_log)).sum((-1, -2))
    representation = (posterior * (posterior_log - prior_log.detach())).sum((-1, -2))
    if free_nats:
        dynamics = dynamics.clamp_min(free_nats)
        representation = representation.clamp_min(free_nats)
    return dynamics.mean(), representation.mean()


def categorical_kl_balanced(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    alpha: float = 0.8,
    free_bits: float = 1.0,
) -> torch.Tensor:
    """KL divergence with balancing between posterior and prior.

    Computes:
        loss = alpha * KL(sg(posterior) || prior)
             + (1 - alpha) * KL(posterior || sg(prior))

    The first term trains only the *prior*; the second trains only the
    *posterior*. ``free_bits`` lower-limits each categorical KL, then the
    function averages over the categoricals and the batch.

    .. note::
        The reference clamps the KL *sum* over the categoricals, not each
        one. Use :func:`categorical_kl_terms` for that behavior.

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        posterior_logits: Shape ``[..., num_categoricals, num_classes]``.
        prior_logits: Shape ``[..., num_categoricals, num_classes]``.
        alpha (float): Balancing weight (0.8 in the paper). Default: 0.8.
        free_bits (float): Minimum per-categorical KL in nats. Default: 1.0.
    Returns:
        Scalar KL loss.

    Examples:
        >>> import torch
        >>> from torchrl.objectives import categorical_kl_balanced
        >>> posterior = torch.randn(4, 8, 16, requires_grad=True)
        >>> prior = torch.randn(4, 8, 16, requires_grad=True)
        >>> kl = categorical_kl_balanced(posterior, prior, alpha=0.8, free_bits=0.1)
        >>> kl.backward()
    """
    posterior = torch.softmax(posterior_logits, dim=-1)
    prior = torch.softmax(prior_logits, dim=-1)

    eps = 1e-8
    posterior = posterior.clamp(min=eps)
    prior = prior.clamp(min=eps)

    post_sg = posterior.detach()
    kl_term1 = (post_sg * (post_sg.log() - prior.log())).sum(-1)

    prior_sg = prior.detach()
    kl_term2 = (posterior * (posterior.log() - prior_sg.log())).sum(-1)

    kl_term1 = kl_term1.clamp_min(free_bits).mean()
    kl_term2 = kl_term2.clamp_min(free_bits).mean()

    return alpha * kl_term1 + (1.0 - alpha) * kl_term2


def _match_trailing_dim(source: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
    """Align ``source`` to the trailing feature dim of ``reference`` for broadcast.

    Multivariate action distributions often emit per-dim log-probs, while the
    discount / advantage tensors carry a singleton trailing dim. This helper
    keeps them compatible by summing over the extra feature dim (multi-D
    log-prob) or unsqueezing when the trailing dim is missing.
    """
    if source.ndim == reference.ndim:
        return source
    if source.ndim == reference.ndim - 1:
        return source.unsqueeze(-1)
    if source.ndim == reference.ndim + 1:
        return source.sum(-1, keepdim=True)
    raise ValueError(
        f"Cannot align source shape {tuple(source.shape)} to "
        f"reference shape {tuple(reference.shape)}"
    )


# ---------------------------------------------------------------------------
# DreamerV3ModelLoss
# ---------------------------------------------------------------------------


class DreamerV3ModelLoss(LossModule):
    """DreamerV3 World Model Loss.

    See :doc:`DreamerV3 in a nutshell </reference/dreamer_v3>` for an overview
    of the world model, RSSM, and training flow.

    Computes three terms:

    1. **KL loss** — balanced KL between prior and posterior categorical
       distributions (see :func:`categorical_kl_balanced`).
    2. **Reconstruction loss** — squared (``"l2"``) or absolute (``"l1"``)
       error between the decoded and the true observations, in symlog space.
    3. **Reward loss** — two-hot cross-entropy or symlog MSE for the predicted
       reward.

    Optionally a **continue loss** (binary cross-entropy) can be enabled
    when the world model outputs a continue predictor.

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        world_model (TensorDictModule): World model that takes a tensordict with
            observations/actions and writes predicted observations, rewards, and
            RSSM prior/posterior logits.
        lambda_kl (float, optional): KL loss weight. Default: 1.0.
        lambda_reco (float, optional): Reconstruction loss weight. Default: 1.0.
        lambda_reward (float, optional): Reward prediction loss weight. Default: 1.0.
        lambda_continue (float, optional): Continue prediction loss weight.
            Default: 0.0 (disabled).
        continue_target_scale (float, optional): Multiplier applied to
            non-terminal continuation targets, for encoding the finite-horizon
            discount in the continuation model. Defaults to 1.0.
        kl_mode ("balanced" or "separate", optional): KL formulation.
            ``"balanced"`` preserves the historical weighted aggregate;
            ``"separate"`` emits the reference dynamics and representation
            losses. Defaults to ``"balanced"``.
        lambda_dynamic (float, optional): Dynamics KL weight in separate mode.
            Defaults to 1.0.
        lambda_representation (float, optional): Representation KL weight in
            separate mode. Defaults to 0.1.
        unimix (float, optional): Uniform mixture used by the categorical KL
            distributions. Defaults to 0.0 for compatibility.
        kl_alpha (float, optional): KL balancing factor (alpha in the paper).
            Default: 0.8.
        free_bits (float, optional): Minimum KL per categorical in nats.
            Default: 1.0.
        reco_loss ("l1" or "l2", optional): Loss type. Default: ``"l2"``.
        reward_two_hot (bool, optional): If ``True``, the reward head is
            expected to output **logits over** ``num_reward_bins`` and the loss
            is two-hot cross-entropy. If ``False``, the reward head outputs a
            **scalar** prediction and the loss is symlog MSE. Default: ``True``.
        num_reward_bins (int, optional): Number of bins for the two-hot reward
            distribution. Default: 255.
        global_average (bool, optional): If ``True``, averages losses over all
            dimensions. Otherwise sums over non-batch/time dims first. Default:
            ``False``.
        detach_output (bool, optional): If ``True``, the returned world model
            output is detached. Set it to ``False`` when a replay value loss
            must train the representation. Default: ``True``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torch import nn
        >>> from torchrl.modules import SymExpTwoHot
        >>> from torchrl.objectives import DreamerV3ModelLoss
        >>> class StubWorldModel(nn.Module):
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.head = nn.LazyLinear(4 * 4)
        ...         self.reward_head = nn.LazyLinear(16)
        ...         self.reward_decoder = SymExpTwoHot(16)
        ...         self.decoder = nn.LazyLinear(3 * 8 * 8)
        ...     def forward(self, td):
        ...         B, T = td.shape
        ...         x = torch.cat([td["state"], td["belief"]], dim=-1)
        ...         logits = self.head(x).view(B, T, 4, 4)
        ...         reco = self.decoder(x).view(B, T, 3, 8, 8)
        ...         reward_logits = self.reward_head(x)
        ...         td.set(("next", "prior_logits"), logits)
        ...         td.set(("next", "posterior_logits"), logits)
        ...         td.set(("next", "reco_pixels"), reco)
        ...         td.set(("next", "reward_logits"), reward_logits)
        ...         td.set(("next", "reward"), self.reward_decoder(reward_logits))
        ...         return td
        >>> wm = StubWorldModel()
        >>> td = TensorDict({
        ...     "state": torch.zeros(2, 3, 16),
        ...     "belief": torch.zeros(2, 3, 8),
        ...     "action": torch.randn(2, 3, 2),
        ...     "next": {
        ...         "pixels": torch.rand(2, 3, 3, 8, 8),
        ...         "reward": torch.randn(2, 3, 1),
        ...         "done": torch.zeros(2, 3, dtype=torch.bool),
        ...     },
        ... }, [2, 3])
        >>> with torch.no_grad():
        ...     wm(td.clone())
        TensorDict(...)
        >>> loss = DreamerV3ModelLoss(wm, num_reward_bins=16, free_bits=0.1)
        >>> loss_td, _ = loss(td)
        >>> sorted(loss_td.keys())
        ['loss_model_kl', 'loss_model_reco', 'loss_model_reward']
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys.

        Attributes:
            reward (NestedKey): Decoded predicted reward. Defaults to
                ``"reward"``.
            reward_logits (NestedKey): Categorical reward logits. Defaults to
                ``"reward_logits"``.
            true_reward (NestedKey): Ground-truth reward (stored temporarily).
                Defaults to ``"true_reward"``.
            prior_logits (NestedKey): Prior categorical logits from the prior
                RSSM. Defaults to ``"prior_logits"``.
            posterior_logits (NestedKey): Posterior categorical logits.
                Defaults to ``"posterior_logits"``.
            pixels (NestedKey): Ground-truth pixel observation.
                Defaults to ``"pixels"``.
            reco_pixels (NestedKey): Predicted pixel observation.
                Defaults to ``"reco_pixels"``.
            continue_pred (NestedKey): Predicted continue logit (optional).
                Defaults to ``"continue_pred"``.
            done (NestedKey): Ground-truth done flag (optional).
                Defaults to ``"done"``.
            terminated (NestedKey): Ground-truth terminal flag (optional).
                Defaults to ``"terminated"``.
        """

        reward: NestedKey = "reward"
        reward_logits: NestedKey = "reward_logits"
        true_reward: NestedKey = "true_reward"
        prior_logits: NestedKey = "prior_logits"
        posterior_logits: NestedKey = "posterior_logits"
        pixels: NestedKey = "pixels"
        reco_pixels: NestedKey = "reco_pixels"
        continue_pred: NestedKey = "continue_pred"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys

    def __init__(
        self,
        world_model: TensorDictModule,
        *,
        lambda_kl: float = 1.0,
        lambda_reco: float = 1.0,
        lambda_reward: float = 1.0,
        lambda_continue: float = 0.0,
        kl_mode: Literal["balanced", "separate"] = "balanced",
        lambda_dynamic: float = 1.0,
        lambda_representation: float = 0.1,
        unimix: float = 0.0,
        continue_target_scale: float = 1.0,
        kl_alpha: float = 0.8,
        free_bits: float = 1.0,
        reco_loss: Literal["l1", "l2"] = "l2",
        reward_two_hot: bool = True,
        num_reward_bins: int = _DEFAULT_NUM_BINS,
        global_average: bool = False,
        detach_output: bool = True,
    ):
        super().__init__()
        self.world_model = world_model
        self.lambda_kl = lambda_kl
        self.lambda_reco = lambda_reco
        self.lambda_reward = lambda_reward
        self.lambda_continue = lambda_continue
        if kl_mode not in ("balanced", "separate"):
            raise ValueError(
                "kl_mode must be 'balanced' or 'separate', got " f"{kl_mode!r}."
            )
        self.kl_mode = kl_mode
        self.lambda_dynamic = lambda_dynamic
        self.lambda_representation = lambda_representation
        self.unimix = unimix
        if not 0 < continue_target_scale <= 1:
            raise ValueError("continue_target_scale must be in (0, 1].")
        self.continue_target_scale = continue_target_scale
        self.kl_alpha = kl_alpha
        self.free_bits = free_bits
        self.reco_loss = reco_loss
        self.reward_two_hot = reward_two_hot
        self.num_reward_bins = num_reward_bins
        self.global_average = global_average
        self.detach_output = detach_output
        self.register_buffer(
            "reward_bins",
            _default_bins(num_reward_bins),
        )

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    @_maybe_record_function_decorator("dreamer_v3/world_model_loss")
    def forward(self, tensordict: TensorDict) -> tuple[TensorDict, TensorDict]:
        tensordict = tensordict.copy()
        tensordict.rename_key_(
            ("next", self.tensor_keys.reward),
            ("next", self.tensor_keys.true_reward),
        )

        tensordict = self.world_model(tensordict)

        # ---- KL loss ----
        prior_logits = tensordict.get(("next", self.tensor_keys.prior_logits))
        posterior_logits = tensordict.get(("next", self.tensor_keys.posterior_logits))
        if self.kl_mode == "separate":
            dynamic_loss, representation_loss = categorical_kl_terms(
                posterior_logits,
                prior_logits,
                free_nats=self.free_bits,
                unimix=self.unimix,
            )
            dynamic_loss = dynamic_loss.unsqueeze(-1)
            representation_loss = representation_loss.unsqueeze(-1)
        else:
            kl_loss = categorical_kl_balanced(
                posterior_logits,
                prior_logits,
                alpha=self.kl_alpha,
                free_bits=self.free_bits,
            ).unsqueeze(-1)

        # ---- Reconstruction loss ----
        pixels = tensordict.get(("next", self.tensor_keys.pixels)).contiguous()
        reco_pixels = tensordict.get(
            ("next", self.tensor_keys.reco_pixels)
        ).contiguous()
        # Apply symlog before computing distance
        if self.reco_loss == "l2":
            reco_loss = (symlog(pixels) - symlog(reco_pixels)).pow(2)
        else:
            reco_loss = (symlog(pixels) - symlog(reco_pixels)).abs()
        if not self.global_average:
            reco_loss = reco_loss.reshape(*tensordict.batch_size, -1).sum(-1)
        reco_loss = reco_loss.mean().unsqueeze(-1)

        # ---- Reward loss ----
        true_reward = tensordict.get(("next", self.tensor_keys.true_reward))
        if self.reward_two_hot:
            reward_logits_key = unravel_key(("next", self.tensor_keys.reward_logits))
            pred_reward = tensordict.get(reward_logits_key, None)
            if pred_reward is None:
                legacy_key = unravel_key(("next", self.tensor_keys.reward))
                pred_reward = tensordict.get(legacy_key)
                warnings.warn(
                    "Storing DreamerV3 categorical reward logits under the decoded "
                    f"reward key {legacy_key!r} is deprecated and will be removed in "
                    "v0.16. Write logits to the configured reward_logits key instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if pred_reward.shape[-1] != self.num_reward_bins:
                raise ValueError(
                    f"reward_two_hot=True expects the reward head to output "
                    f"logits over {self.num_reward_bins} bins, got trailing "
                    f"dim {pred_reward.shape[-1]}."
                )
            reward_loss = two_hot_cross_entropy(
                pred_reward, true_reward, self.reward_bins
            )
        else:
            pred_reward = tensordict.get(("next", self.tensor_keys.reward))
            reward_loss = (symlog(true_reward) - symlog(pred_reward)).pow(2).squeeze(-1)
        reward_loss = reward_loss.mean().unsqueeze(-1)

        td_out = TensorDict(
            loss_model_reco=self.lambda_reco * reco_loss,
            loss_model_reward=self.lambda_reward * reward_loss,
        )
        if self.kl_mode == "separate":
            td_out.set(
                "loss_model_dynamic",
                self.lambda_kl * self.lambda_dynamic * dynamic_loss,
            )
            td_out.set(
                "loss_model_representation",
                self.lambda_kl * self.lambda_representation * representation_loss,
            )
        else:
            td_out.set("loss_model_kl", self.lambda_kl * kl_loss)

        # ---- Optional continue loss ----
        if self.lambda_continue > 0:
            continue_pred = tensordict.get(
                ("next", self.tensor_keys.continue_pred), None
            )
            terminated = tensordict.get(("next", self.tensor_keys.terminated), None)
            if terminated is None:
                terminated = tensordict.get(("next", self.tensor_keys.done), None)
            if continue_pred is not None and terminated is not None:
                continue_target = (~terminated).float() * self.continue_target_scale
                continue_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    continue_pred.squeeze(-1), continue_target.squeeze(-1)
                ).unsqueeze(-1)
                td_out.set("loss_model_continue", self.lambda_continue * continue_loss)

        self._clear_weakrefs(tensordict, td_out)
        return td_out, tensordict.data if self.detach_output else tensordict


# ---------------------------------------------------------------------------
# DreamerV3ActorLoss
# ---------------------------------------------------------------------------


class DreamerV3ActorLoss(LossModule):
    """DreamerV3 Actor Loss.

    See :doc:`DreamerV3 in a nutshell </reference/dreamer_v3>` for an overview
    of latent imagination, actor training, and DreamerV3 nomenclature.

    Rolls out imagined trajectories in latent space using the world model
    environment, then computes:

    .. code-block:: text

        loss_actor = -E[log pi(a_t | z_t) * sg(A_t)] - eta * H[pi(. | z_t)]

    where ``A_t = V_lambda(z_t) - v(z_t)`` is the advantage (lambda return
    minus baseline) and ``eta`` is the entropy bonus weight.

    When the actor is a reparameterizable (continuous) policy the
    reparameterization gradient is used directly instead of REINFORCE.

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        actor_model (TensorDictModule): The actor / policy network.
        value_model (TensorDictModule): The value network.
        model_based_env (DreamerEnv): The imagination environment.
        continuation_model (TensorDictModuleBase, optional): Shared trained
            model that writes continuation probabilities for imagined states.
            Defaults to ``None``.
        imagination_horizon (int, optional): Rollout length inside imagination.
            Default: 15.
        discount_loss (bool, optional): If ``True``, discount the actor loss
            with a cumulative gamma factor. Default: ``True``.
        entropy_bonus (float, optional): Weight for the entropy regularisation
            term ``eta``. Default: ``3e-4``.
        use_reinforce (bool, optional): If ``True``, uses REINFORCE (log-prob
            * stop-gradient advantage). If ``False``, uses the straight
            reparameterization gradient (suitable for continuous Gaussian
            actors). Default: ``False``.
        return_normalization (bool, optional): Normalize detached REINFORCE
            advantages by an EMA return-percentile span. Default: ``True``.
        return_normalization_rate (float, optional): EMA update rate for the
            return statistics. Default: ``0.01``.
        return_normalization_quantiles (tuple of float, optional): Lower and
            upper return quantiles. Default: ``(0.05, 0.95)``.
        return_normalization_min_scale (float, optional): Minimum divisor for
            REINFORCE advantages. Default: ``1.0``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import (
        ...     InteractionType,
        ...     ProbabilisticTensorDictModule,
        ...     ProbabilisticTensorDictSequential,
        ...     TensorDictModule,
        ... )
        >>> from torchrl.data import Unbounded
        >>> from torchrl.envs import TransformedEnv
        >>> from torchrl.envs.model_based.dreamer import DreamerEnv
        >>> from torchrl.envs.transforms import TensorDictPrimer
        >>> from torchrl.modules import MLP, SafeSequential, WorldModelWrapper
        >>> from torchrl.modules.distributions.continuous import TanhNormal
        >>> from torchrl.modules.models.model_based import DreamerActor
        >>> from torchrl.modules.models.model_based_v3 import RSSMPriorV3
        >>> from torchrl.objectives import DreamerV3ActorLoss
        >>> from torchrl.objectives.utils import ValueEstimators
        >>> from torchrl.testing.mocking_classes import ContinuousActionConvMockEnv
        >>> base_env = TransformedEnv(
        ...     ContinuousActionConvMockEnv(pixel_shape=[3, 16, 16]),
        ...     TensorDictPrimer(
        ...         random=False, default_value=0,
        ...         state=Unbounded(16), belief=Unbounded(8),
        ...     ),
        ... )
        >>> action_dim = base_env.action_spec.shape[0]
        >>> rssm_prior = RSSMPriorV3(
        ...     action_shape=base_env.action_spec.shape,
        ...     hidden_dim=8, rnn_hidden_dim=8,
        ...     num_categoricals=4, num_classes=4, action_dim=action_dim,
        ... )
        >>> transition = SafeSequential(
        ...     TensorDictModule(
        ...         rssm_prior,
        ...         in_keys=["state", "belief", "action"],
        ...         out_keys=["_", "state", "belief"],
        ...     ),
        ... )
        >>> reward = TensorDictModule(
        ...     MLP(out_features=1, depth=1, num_cells=8),
        ...     in_keys=["state", "belief"], out_keys=["reward"],
        ... )
        >>> mb_env = DreamerEnv(
        ...     world_model=WorldModelWrapper(transition, reward),
        ...     prior_shape=torch.Size([16]),
        ...     belief_shape=torch.Size([8]),
        ... )
        >>> mb_env.set_specs_from_env(base_env)
        >>> with torch.no_grad():
        ...     _ = mb_env.rollout(3)
        >>> actor_module = DreamerActor(out_features=action_dim, depth=1, num_cells=8)
        >>> actor = ProbabilisticTensorDictSequential(
        ...     TensorDictModule(
        ...         actor_module, in_keys=["state", "belief"], out_keys=["loc", "scale"],
        ...     ),
        ...     ProbabilisticTensorDictModule(
        ...         in_keys=["loc", "scale"], out_keys=["action"],
        ...         default_interaction_type=InteractionType.RANDOM,
        ...         distribution_class=TanhNormal,
        ...     ),
        ... )
        >>> warmup = TensorDict(
        ...     {"state": torch.randn(1, 2, 16), "belief": torch.randn(1, 2, 8)}, [1]
        ... )
        >>> _ = actor(warmup)
        >>> value = TensorDictModule(
        ...     MLP(out_features=1, depth=1, num_cells=8),
        ...     in_keys=["state", "belief"], out_keys=["state_value"],
        ... )
        >>> _ = value(warmup)
        >>> loss = DreamerV3ActorLoss(actor, value, mb_env, imagination_horizon=3)
        >>> loss.make_value_estimator(ValueEstimators.TDLambda)
        >>> td = TensorDict(
        ...     {"state": torch.randn(2, 16), "belief": torch.randn(2, 8)}, [2]
        ... )
        >>> loss_td, _ = loss(td)
        >>> "loss_actor" in loss_td.keys()
        True
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys.

        Attributes:
            state (NestedKey): Stochastic latent state. Defaults to ``"state"``.
            belief (NestedKey): Deterministic GRU hidden state. Defaults to ``"belief"``.
            reward (NestedKey): Imagined reward. Defaults to ``"reward"``.
            value (NestedKey): State value. Defaults to ``"state_value"``.
            action (NestedKey): Imagined action. Defaults to ``"action"``.
            action_log_prob (NestedKey): Log-prob of the taken action.
                Defaults to ``"action_log_prob"``.
            done (NestedKey): Done flag. Defaults to ``"done"``.
            terminated (NestedKey): Terminated flag. Defaults to ``"terminated"``.
            continuation (NestedKey): Predicted continuation probability.
                Defaults to ``"continuation"``.
            discount_weight (NestedKey): Cumulative imagination weight.
                Defaults to ``"discount_weight"``.
        """

        state: NestedKey = "state"
        belief: NestedKey = "belief"
        reward: NestedKey = "reward"
        value: NestedKey = "state_value"
        action: NestedKey = "action"
        action_log_prob: NestedKey = "action_log_prob"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        continuation: NestedKey = "continuation"
        discount_weight: NestedKey = "discount_weight"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys
    default_value_estimator = ValueEstimators.TDLambda

    value_model: TensorDictModule
    actor_model: TensorDictModule

    def __init__(
        self,
        actor_model: TensorDictModule,
        value_model: TensorDictModule,
        model_based_env: DreamerEnv,
        *,
        continuation_model: TensorDictModuleBase | None = None,
        imagination_horizon: int = 15,
        discount_loss: bool = True,
        entropy_bonus: float = 3e-4,
        use_reinforce: bool = False,
        return_normalization: bool = True,
        return_normalization_rate: float = 0.01,
        return_normalization_quantiles: tuple[float, float] = (0.05, 0.95),
        return_normalization_min_scale: float = 1.0,
        gamma: float | None = None,
        lmbda: float | None = None,
    ):
        super().__init__()
        self.actor_model = actor_model
        self.__dict__["value_model"] = value_model
        self.model_based_env = model_based_env
        self.__dict__["continuation_model"] = continuation_model
        self.imagination_horizon = imagination_horizon
        self.discount_loss = discount_loss
        self.entropy_bonus = entropy_bonus
        self.use_reinforce = use_reinforce
        self.return_normalization = return_normalization
        if not 0 <= return_normalization_rate <= 1:
            raise ValueError("return_normalization_rate must be in [0, 1].")
        lower_quantile, upper_quantile = return_normalization_quantiles
        if not 0 <= lower_quantile < upper_quantile <= 1:
            raise ValueError(
                "return_normalization_quantiles must satisfy "
                "0 <= lower < upper <= 1."
            )
        if return_normalization_min_scale <= 0:
            raise ValueError("return_normalization_min_scale must be positive.")
        self.return_normalization_rate = return_normalization_rate
        self.return_normalization_quantiles = return_normalization_quantiles
        self.return_normalization_min_scale = return_normalization_min_scale
        self.register_buffer("return_low", torch.tensor(0.0))
        self.register_buffer("return_high", torch.tensor(0.0))
        if gamma is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)
        if lmbda is not None:
            raise TypeError(_GAMMA_LMBDA_DEPREC_ERROR)

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        if self._value_estimator is not None:
            self._value_estimator.set_keys(value=self._tensor_keys.value)

    @_maybe_record_function_decorator("dreamer_v3/actor_loss")
    def forward(self, tensordict: TensorDict) -> tuple[TensorDict, TensorDict]:
        tensordict = tensordict.select(
            self.tensor_keys.state, self.tensor_keys.belief
        ).data

        with hold_out_net(self.model_based_env), set_exploration_type(
            ExplorationType.RANDOM
        ):
            tensordict = self.model_based_env.reset(tensordict.copy())
            fake_data = self.model_based_env.rollout(
                max_steps=self.imagination_horizon,
                policy=self.actor_model,
                auto_reset=False,
                tensordict=tensordict,
            )
            next_tensordict = step_mdp(fake_data, keep_other=True)
            with hold_out_net(self.value_model):
                next_tensordict = self.value_model(next_tensordict)
        next_value = next_tensordict.get(self.tensor_keys.value)

        reward = fake_data.get(("next", self.tensor_keys.reward))
        continuation = None
        root_continuation = None
        continuation_model = self.__dict__.get("continuation_model")
        if continuation_model is not None:
            # step_mdp shifts by one: only index 0 needs a new forward pass.
            first_td = fake_data[..., :1].select(
                *continuation_model.in_keys, strict=False
            )
            continuation_td = next_tensordict.select(
                *continuation_model.in_keys, strict=False
            )
            with hold_out_net(continuation_model):
                continuation_model(first_td)
                continuation_model(continuation_td)
            continuation = continuation_td.get(self.tensor_keys.continuation)
            root_continuation = torch.cat(
                [
                    first_td.get(self.tensor_keys.continuation),
                    continuation[..., :-1, :],
                ],
                dim=-2,
            )
            fake_data.set(self.tensor_keys.continuation, root_continuation)
            fake_data.set(("next", self.tensor_keys.continuation), continuation)
        lambda_target = self.lambda_target(reward, next_value, continuation)
        fake_data.set("lambda_target", lambda_target)

        if not self.discount_loss:
            discount = torch.ones_like(lambda_target)
        else:
            gamma = self.value_estimator.gamma.to(tensordict.device)
            # w_t uses the root continuations (0..H-1), the returns use 1..H.
            continuations = (
                root_continuation
                if continuation is not None
                else torch.ones_like(lambda_target)
            )
            discount = torch.cat(
                [continuations[..., :1, :], gamma * continuations[..., 1:, :]],
                dim=-2,
            ).cumprod(dim=-2)
        discount = discount.detach()
        fake_data.set(self.tensor_keys.discount_weight, discount)

        if self.use_reinforce or self.entropy_bonus > 0:
            actor_inputs = fake_data.select(
                *self.actor_model.in_keys, strict=False
            ).detach()
            policy_distribution = self.actor_model.get_dist(actor_inputs)

        if self.use_reinforce:
            # REINFORCE: score a stopped action from a stopped imagined state.
            # The rollout action is reparameterized, so its cached log-probability
            # has a pathwise component and is not a score-function estimator.
            action = fake_data.get(self.tensor_keys.action).detach()
            log_prob = policy_distribution.log_prob(action)
            log_prob = _match_trailing_dim(log_prob, lambda_target)
            with hold_out_net(self.value_model):
                baseline_td = fake_data.select(*self.value_model.in_keys, strict=False)
                self.value_model(baseline_td)
            baseline = baseline_td.get(self.tensor_keys.value)
            advantage = (lambda_target - baseline).detach()
            return_scale = self._return_scale(lambda_target)
            advantage = advantage / return_scale
            actor_loss = -(discount * log_prob * advantage).mean()
        else:
            # Reparameterization gradient
            return_scale = torch.ones(
                (), dtype=lambda_target.dtype, device=lambda_target.device
            )
            actor_loss = -(discount * lambda_target).mean()

        if self.entropy_bonus > 0:
            if HAS_ENTROPY.get(type(policy_distribution), False):
                entropy = policy_distribution.entropy()
            else:
                _, entropy_log_prob = rsample_and_log_prob(policy_distribution)
                entropy = -entropy_log_prob
            entropy = _match_trailing_dim(entropy, discount)
            entropy = (discount * entropy).mean()
            actor_loss = actor_loss - self.entropy_bonus * entropy

        loss_tensordict = TensorDict(
            {
                "loss_actor": actor_loss,
                "return_low": self.return_low.detach().clone(),
                "return_high": self.return_high.detach().clone(),
                "return_scale": return_scale.detach().clone(),
                "continuation_mean": (
                    continuation.mean().detach()
                    if continuation is not None
                    else torch.ones((), device=actor_loss.device)
                ),
            },
            [],
        )
        self._clear_weakrefs(tensordict, loss_tensordict)
        return loss_tensordict, fake_data.data

    def _return_scale(self, returns: torch.Tensor) -> torch.Tensor:
        if not self.return_normalization:
            return torch.ones((), dtype=returns.dtype, device=returns.device)
        if self.training:
            quantiles = torch.tensor(
                self.return_normalization_quantiles,
                dtype=self.return_low.dtype,
                device=self.return_low.device,
            )
            current_low, current_high = torch.quantile(
                returns.detach().to(self.return_low), quantiles
            )
            with torch.no_grad():
                self.return_low.lerp_(current_low, self.return_normalization_rate)
                self.return_high.lerp_(current_high, self.return_normalization_rate)
        return (self.return_high - self.return_low).clamp_min(
            self.return_normalization_min_scale
        )

    def lambda_target(
        self,
        reward: torch.Tensor,
        value: torch.Tensor,
        continuation: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if continuation is not None:
            gamma = self.value_estimator.gamma.to(reward)
            lmbda = self.value_estimator.lmbda.to(reward)
            next_return = value[..., -1, :]
            returns = []
            for reward_t, value_t, continuation_t in zip(
                reversed(reward.unbind(-2)),
                reversed(value.unbind(-2)),
                reversed(continuation.unbind(-2)),
            ):
                next_return = reward_t + gamma * continuation_t * (
                    (1 - lmbda) * value_t + lmbda * next_return
                )
                returns.append(next_return)
            return torch.stack(returns[::-1], dim=-2)
        done = torch.zeros(reward.shape, dtype=torch.bool, device=reward.device)
        terminated = torch.zeros(reward.shape, dtype=torch.bool, device=reward.device)
        input_tensordict = TensorDict(
            {
                ("next", self.tensor_keys.reward): reward,
                ("next", self.tensor_keys.value): value,
                ("next", self.tensor_keys.done): done,
                ("next", self.tensor_keys.terminated): terminated,
            },
            [],
        )
        return self.value_estimator.value_estimate(input_tensordict)

    SUPPORTED_VALUE_ESTIMATORS = (
        ValueEstimators.TD0,
        ValueEstimators.TD1,
        ValueEstimators.TDLambda,
    )

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator
        if isinstance(value_type, ValueEstimatorBase) or (
            isinstance(value_type, type) and issubclass(value_type, ValueEstimatorBase)
        ):
            return LossModule.make_value_estimator(self, value_type, **hyperparams)
        if hasattr(self, "lmbda"):
            hyperparams.setdefault("lmbda", self.lmbda)
        if value_type == ValueEstimators.TDLambda:
            hyperparams.setdefault("vectorized", True)
        dispatch_value_estimator(
            self,
            value_type,
            supported=self.SUPPORTED_VALUE_ESTIMATORS,
            tensor_keys={
                "value": self.tensor_keys.value,
                "value_target": "value_target",
            },
            value_network=None,
            **hyperparams,
        )


# ---------------------------------------------------------------------------
# DreamerV3ValueLoss
# ---------------------------------------------------------------------------


def _replay_value_target(
    reward: torch.Tensor,
    done: torch.Tensor,
    terminated: torch.Tensor,
    bootstrap: torch.Tensor,
    horizon: float,
    lmbda: float,
) -> torch.Tensor:
    """Compute the lambda returns along a replay sequence.

    The output has one step less than the input: element ``k`` is the return
    for replay state ``k``, from the rewards and bootstraps at ``k + 1`` on.
    """
    reward = reward.squeeze(-1)
    done = done.squeeze(-1)
    terminated = terminated.squeeze(-1)
    discount = 1 - 1 / horizon
    live = (~terminated[..., 1:]).to(reward.dtype) * discount
    continuation = (~done[..., 1:]).to(reward.dtype) * lmbda
    intermediate = reward[..., 1:] + (1 - continuation) * live * bootstrap[..., 1:]
    next_return = bootstrap[..., -1]
    returns = []
    for time_index in reversed(range(intermediate.shape[-1])):
        next_return = (
            intermediate[..., time_index]
            + live[..., time_index] * continuation[..., time_index] * next_return
        )
        returns.append(next_return)
    return torch.stack(returns[::-1], -1).detach()


class DreamerV3ValueLoss(LossModule):
    """DreamerV3 Value Loss.

    See :doc:`DreamerV3 in a nutshell </reference/dreamer_v3>` for an overview
    of the online critic, slow critic, and their update flow.

    Trains the value network to predict the lambda-target computed by
    :class:`DreamerV3ActorLoss`. Supports two loss modes:

    - ``"symlog_mse"`` (default): ``(symlog(v_pred) - symlog(target))^2``
    - ``"two_hot"``: Two-hot cross-entropy over a fixed bin grid (matches the
      full DreamerV3 distribution-valued critic).

    The discount factor used here must match the one the actor used to compute
    ``lambda_target``. The recommended way to keep them in lock-step is to
    pass the actor loss to the constructor via ``actor_loss=``: the value loss
    will then read ``gamma`` from the actor's value estimator at every forward
    call. The legacy ``gamma=`` kwarg + :meth:`sync_gamma_with_actor_loss`
    pattern is still supported.

    Setting ``slow_critic_regularization`` to a positive value creates a
    checkpointed target critic. Associate a :class:`~torchrl.objectives.SoftUpdate`
    and step it after each critic optimizer step.

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        value_model (TensorDictModule): The value network.
        value_loss ("symlog_mse" or "two_hot", optional): Loss type.
            Default: ``"symlog_mse"``.
        discount_loss (bool, optional): If ``True``, discounts the loss with
            a cumulative gamma factor. Default: ``True``.
        gamma (float, optional): Discount factor used when ``discount_loss=True``.
            Ignored if ``actor_loss`` is provided. Default: ``0.99``.
        num_value_bins (int, optional): Number of bins for ``"two_hot"`` loss.
            Default: 255.
        actor_loss (DreamerV3ActorLoss, optional): If provided, ``gamma`` is
            read from this actor loss's value estimator on every forward call,
            avoiding any chance of a mismatch. Default: ``None``.
        slow_critic_regularization (float, optional): Weight of the auxiliary
            loss that trains the online critic toward decoded target-critic
            predictions. Default: ``0.0``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import MLP
        >>> from torchrl.objectives import DreamerV3ValueLoss
        >>> value_model = TensorDictModule(
        ...     MLP(out_features=1, depth=1, num_cells=8),
        ...     in_keys=["state"],
        ...     out_keys=["state_value"],
        ... )
        >>> td = TensorDict({
        ...     "state": torch.randn(8, 4),
        ...     "lambda_target": torch.randn(8, 1),
        ... }, [8])
        >>> loss = DreamerV3ValueLoss(value_model)
        >>> loss_td, _ = loss(td)
        >>> "loss_value" in loss_td.keys()
        True
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys.

        Attributes:
            value (NestedKey): Decoded predicted value key. Defaults to
                ``"state_value"``.
            value_logits (NestedKey): Categorical value logits key. Defaults to
                ``"state_value_logits"``.
            discount_weight (NestedKey): Optional cumulative imagination
                weight. Defaults to ``"discount_weight"``.
            reward (NestedKey): Replay reward, read under ``"next"``.
                Defaults to ``"reward"``.
            done (NestedKey): Replay episode end, read under ``"next"``.
                Defaults to ``"done"``.
            terminated (NestedKey): Replay terminal, read under ``"next"``.
                Defaults to ``"terminated"``.
            bootstrap (NestedKey): First imagined lambda return of each replay
                state, read at the root. Defaults to ``"bootstrap"``.
        """

        value: NestedKey = "state_value"
        value_logits: NestedKey = "state_value_logits"
        discount_weight: NestedKey = "discount_weight"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        bootstrap: NestedKey = "bootstrap"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys

    value_model: TensorDictModule
    value_model_params: TensorDictParams
    target_value_model_params: TensorDictParams

    def __init__(
        self,
        value_model: TensorDictModule,
        value_loss: Literal["symlog_mse", "two_hot"] = "symlog_mse",
        discount_loss: bool = True,
        gamma: float = 0.99,
        num_value_bins: int = _DEFAULT_NUM_BINS,
        actor_loss: DreamerV3ActorLoss | None = None,
        slow_critic_regularization: float = 0.0,
    ):
        super().__init__()
        if slow_critic_regularization < 0:
            raise ValueError("slow_critic_regularization must be non-negative.")
        self.slow_critic_regularization = slow_critic_regularization
        self.convert_to_functional(
            value_model,
            "value_model",
            create_target_params=bool(slow_critic_regularization),
        )
        self.value_loss = value_loss
        self.gamma = gamma
        self.discount_loss = discount_loss
        if value_loss not in ("symlog_mse", "two_hot"):
            raise ValueError(
                f"value_loss must be 'symlog_mse' or 'two_hot', got '{value_loss}'"
            )
        # Stash without registering as a submodule (avoid double parameter ownership)
        self.__dict__["_actor_loss"] = actor_loss
        self.register_buffer("value_bins", _default_bins(num_value_bins))

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def _resolved_gamma(self) -> float:
        actor_loss = self.__dict__.get("_actor_loss")
        if actor_loss is None:
            return float(self.gamma)
        estimator_gamma = actor_loss.value_estimator.gamma
        if torch.is_tensor(estimator_gamma):
            estimator_gamma = estimator_gamma.item()
        return float(estimator_gamma)

    def sync_gamma_with_actor_loss(self, actor_loss: DreamerV3ActorLoss) -> None:
        """Pull ``gamma`` from an actor loss's value estimator.

        Prefer passing ``actor_loss=`` to the constructor; this method exists
        for backward compatibility with the legacy two-step setup.
        """
        estimator_gamma = actor_loss.value_estimator.gamma
        if torch.is_tensor(estimator_gamma):
            estimator_gamma = estimator_gamma.item()
        self.gamma = float(estimator_gamma)

    def replay_value_loss(
        self,
        tensordict: TensorDictBase,
        *,
        horizon: float = 333.0,
        lmbda: float = 0.95,
    ) -> TensorDictBase:
        """Compute the DreamerV3 critic loss on a replay sequence.

        The return of each replay state uses the reward of the next step and
        bootstraps from ``bootstrap``, the first imagined return of that
        state. The gradient stays on the input features, so the loss can
        train the RSSM representation when the model loss does not detach.

        Args:
            tensordict (TensorDictBase): Posterior replay features, batch size
                ``[B, T]``, with the ``value_model`` input keys and the
                ``reward``, ``done``, ``terminated`` and ``bootstrap`` entries
                that :attr:`tensor_keys` names.
            horizon (float, optional): Discount horizon; the step discount is
                ``1 - 1 / horizon``. Defaults to ``333.0``.
            lmbda (float, optional): Lambda-return coefficient. Default: 0.95.

        Returns:
            A tensordict with the scalar, unweighted ``loss_replay_value``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import TensorDictModule
            >>> from torchrl.modules import MLP
            >>> from torchrl.objectives import DreamerV3ValueLoss
            >>> value_model = TensorDictModule(
            ...     MLP(out_features=1, depth=1, num_cells=8),
            ...     in_keys=["state"],
            ...     out_keys=["state_value"],
            ... )
            >>> loss = DreamerV3ValueLoss(value_model)
            >>> replay = TensorDict({
            ...     "state": torch.randn(2, 5, 4),
            ...     "bootstrap": torch.randn(2, 5),
            ...     "next": {
            ...         "reward": torch.randn(2, 5, 1),
            ...         "done": torch.zeros(2, 5, 1, dtype=torch.bool),
            ...         "terminated": torch.zeros(2, 5, 1, dtype=torch.bool),
            ...     },
            ... }, [2, 5])
            >>> loss_td = loss.replay_value_loss(replay)
            >>> loss_td["loss_replay_value"].shape
            torch.Size([])
        """
        reward = tensordict.get(("next", self.tensor_keys.reward))
        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated))
        bootstrap = tensordict.get(self.tensor_keys.bootstrap)
        target = _replay_value_target(
            reward, done, terminated, bootstrap, horizon, lmbda
        )
        done = done.squeeze(-1)
        # Drop the last step (no next state) and the steps that end an episode.
        weight = ~done[..., :-1]

        value_tensordict = tensordict.select(*self.value_model.in_keys, strict=False)
        with self.value_model_params.to_module(
            self.value_model, preserve_module_state=False
        ):
            self.value_model(value_tensordict)
        prediction = (
            value_tensordict.get(self.tensor_keys.value_logits)[..., :-1, :]
            if self.value_loss == "two_hot"
            else value_tensordict.get(self.tensor_keys.value)[..., :-1, 0]
        )
        loss = self._step_value_loss(prediction, target)

        if self.slow_critic_regularization:
            target_tensordict = tensordict.select(
                *self.value_model.in_keys, strict=False
            )
            with torch.no_grad(), self.target_value_model_params.to_module(
                self.value_model, preserve_module_state=False
            ):
                self.value_model(target_tensordict)
            slow_target = target_tensordict.get(self.tensor_keys.value)[..., :-1, 0]
            loss = loss + self.slow_critic_regularization * self._step_value_loss(
                prediction, slow_target
            )

        return TensorDict(
            {"loss_replay_value": (weight.to(loss.dtype) * loss).mean()}, []
        )

    def _step_value_loss(
        self, prediction: torch.Tensor, target: torch.Tensor
    ) -> torch.Tensor:
        if self.value_loss == "two_hot":
            return two_hot_cross_entropy(prediction, target, self.value_bins)
        return (symlog(prediction) - symlog(target)).square()

    @_maybe_record_function_decorator("dreamer_v3/value_loss")
    def forward(self, fake_data) -> tuple[TensorDict, TensorDict]:
        lambda_target = fake_data.get("lambda_target")

        tensordict_select = fake_data.select(*self.value_model.in_keys, strict=False)
        with self.value_model_params.to_module(
            self.value_model, preserve_module_state=False
        ):
            self.value_model(tensordict_select)
        # lambda_target shape: [N, 1] (flat) or [B, T, 1] (batch x time)
        # Squeeze the trailing 1 for loss computation
        target_sq = lambda_target.squeeze(-1)  # [N] or [B, T]

        provided_discount = fake_data.get(self.tensor_keys.discount_weight, None)
        if provided_discount is not None:
            discount = provided_discount.squeeze(-1)
        elif self.discount_loss and target_sq.ndim >= 2:
            gamma = self._resolved_gamma()
            discount = gamma * torch.ones_like(target_sq)
            discount[..., 0] = 1
            discount = discount.cumprod(dim=-1)
        else:
            discount = torch.ones_like(target_sq)

        if self.value_loss == "two_hot":
            value_pred = tensordict_select.get(self.tensor_keys.value_logits, None)
            if value_pred is None:
                value_pred = tensordict_select.get(self.tensor_keys.value)
                warnings.warn(
                    "Storing DreamerV3 categorical value logits under the decoded "
                    f"value key {unravel_key(self.tensor_keys.value)!r} is deprecated "
                    "and will be removed in v0.16. Write logits to the configured "
                    "value_logits key instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            if value_pred.shape[-1] != self.value_bins.shape[0]:
                raise ValueError(
                    f"value_loss='two_hot' expects the value head to output "
                    f"logits over {self.value_bins.shape[0]} bins, got trailing "
                    f"dim {value_pred.shape[-1]}."
                )
            loss = two_hot_cross_entropy(value_pred, target_sq, self.value_bins)
        else:
            # symlog MSE
            value_pred = tensordict_select.get(self.tensor_keys.value)
            loss = (symlog(value_pred.squeeze(-1)) - symlog(target_sq)).pow(2)

        if self.slow_critic_regularization:
            target_tensordict = fake_data.select(
                *self.value_model.in_keys, strict=False
            )
            with torch.no_grad(), self.target_value_model_params.to_module(
                self.value_model, preserve_module_state=False
            ):
                self.value_model(target_tensordict)
            target_value = target_tensordict.get(self.tensor_keys.value)
            if self.value_loss == "two_hot" and (
                target_value.shape[-1] == self.value_bins.shape[0]
            ):
                target_value = two_hot_decode(target_value, self.value_bins).unsqueeze(
                    -1
                )
            if self.value_loss == "two_hot":
                slow_loss = two_hot_cross_entropy(
                    value_pred, target_value.squeeze(-1), self.value_bins
                )
            else:
                slow_loss = (
                    symlog(value_pred.squeeze(-1)) - symlog(target_value.squeeze(-1))
                ).pow(2)
            loss = loss + self.slow_critic_regularization * slow_loss
        else:
            slow_loss = torch.zeros_like(loss)

        value_loss = (discount * loss).mean()

        loss_tensordict = TensorDict(
            {
                "loss_value": value_loss,
                "value_slow_loss": (discount * slow_loss).mean().detach(),
            }
        )
        self._clear_weakrefs(fake_data, loss_tensordict)
        return loss_tensordict, fake_data.data
