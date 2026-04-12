# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 loss modules.

Implements the three loss modules from DreamerV3 (Mastering Diverse Domains in
World Models, Hafner et al. 2023):

- :class:`DreamerV3ModelLoss` — world model (KL balancing + symlog reconstruction)
- :class:`DreamerV3ActorLoss` — actor (REINFORCE + entropy bonus)
- :class:`DreamerV3ValueLoss` — value function (symlog MSE or two-hot CE)

Utility functions :func:`symlog`, :func:`symexp`, :func:`two_hot_encode` and
:func:`two_hot_decode` are also exported for use in custom models.

Reference: https://arxiv.org/abs/2301.04104
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey

from torchrl._utils import _maybe_record_function_decorator
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type, step_mdp
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import (
    _GAMMA_LMBDA_DEPREC_ERROR,
    default_value_kwargs,
    hold_out_net,
    ValueEstimators,
)
from torchrl.objectives.value import (
    TD0Estimator,
    TD1Estimator,
    TDLambdaEstimator,
    ValueEstimatorBase,
)

# ---------------------------------------------------------------------------
# Symlog / symexp transforms
# ---------------------------------------------------------------------------


def symlog(x: torch.Tensor) -> torch.Tensor:
    """Symmetric logarithm: ``sign(x) * log(|x| + 1)``.

    Used by DreamerV3 to compress the dynamic range of targets and
    predictions before computing reconstruction losses.
    """
    return x.sign() * (x.abs() + 1).log()


def symexp(x: torch.Tensor) -> torch.Tensor:
    """Symmetric exponential: ``sign(x) * (exp(|x|) - 1)``.

    Inverse of :func:`symlog`.
    """
    return x.sign() * (x.abs().exp() - 1)


# ---------------------------------------------------------------------------
# Two-hot encoding (for reward / value distributions)
# ---------------------------------------------------------------------------

# Default 255-bin linspace in symlog space: roughly covers [-20, 20] raw scale
_DEFAULT_NUM_BINS: int = 255
_DEFAULT_BIN_RANGE: float = 20.0


def _default_bins(num_bins: int = _DEFAULT_NUM_BINS, device=None) -> torch.Tensor:
    return torch.linspace(
        -_DEFAULT_BIN_RANGE, _DEFAULT_BIN_RANGE, num_bins, device=device
    )


def two_hot_encode(
    x: torch.Tensor,
    bins: torch.Tensor,
) -> torch.Tensor:
    """Encode a scalar tensor as a two-hot distribution over ``bins``.

    The scalar is split between the two nearest bin centers proportionally so
    that ``E[bins] = x``.

    Args:
        x: Values to encode, shape ``[...]``.
        bins: Sorted bin centers, shape ``[num_bins]``.

    Returns:
        Two-hot vectors, shape ``[..., num_bins]``.
    """
    bins = bins.to(x.device)
    x_clamped = x.clamp(bins[0], bins[-1])

    # Index of the lower bin
    lower_idx = (x_clamped.unsqueeze(-1) >= bins).sum(-1) - 1
    lower_idx = lower_idx.clamp(0, bins.shape[0] - 2)
    upper_idx = lower_idx + 1

    lower_val = bins[lower_idx]
    upper_val = bins[upper_idx]
    span = upper_val - lower_val
    upper_weight = torch.where(
        span > 0, (x_clamped - lower_val) / span, torch.zeros_like(x_clamped)
    )
    lower_weight = 1.0 - upper_weight

    two_hot = torch.zeros(*x.shape, bins.shape[0], device=x.device, dtype=x.dtype)
    two_hot.scatter_(-1, lower_idx.unsqueeze(-1), lower_weight.unsqueeze(-1))
    two_hot.scatter_(-1, upper_idx.unsqueeze(-1), upper_weight.unsqueeze(-1))
    return two_hot


def two_hot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Decode a distribution over ``bins`` to a scalar expectation.

    Args:
        logits: Raw logits, shape ``[..., num_bins]``.
        bins: Sorted bin centers, shape ``[num_bins]``.

    Returns:
        Scalar expected values, shape ``[...]``.
    """
    bins = bins.to(logits.device)
    probs = torch.softmax(logits, dim=-1)
    return (probs * bins).sum(-1)


# ---------------------------------------------------------------------------
# KL balancing for categorical distributions (DreamerV3 §3)
# ---------------------------------------------------------------------------


def categorical_kl_balanced(
    posterior_logits: torch.Tensor,
    prior_logits: torch.Tensor,
    alpha: float = 0.8,
    free_bits: float = 1.0,
) -> torch.Tensor:
    """KL divergence with balancing between posterior and prior.

    Computes:
        loss = α * KL(sg(posterior) ‖ prior) + (1 - α) * KL(posterior ‖ sg(prior))

    The first term trains only the *prior*; the second trains only the
    *posterior*. Free bits are applied per categorical before averaging.

    Args:
        posterior_logits: Shape ``[..., num_categoricals, num_classes]``.
        prior_logits: Shape ``[..., num_categoricals, num_classes]``.
        alpha (float): Balancing weight (0.8 in the paper). Default: 0.8.
        free_bits (float): Minimum KL per categorical in nats. Default: 1.0.

    Returns:
        Scalar KL loss.
    """
    posterior = torch.softmax(posterior_logits, dim=-1)
    prior = torch.softmax(prior_logits, dim=-1)

    # Numerical stability: add small epsilon
    eps = 1e-8
    posterior = posterior.clamp(min=eps)
    prior = prior.clamp(min=eps)

    # KL(sg(posterior) || prior): only prior gets gradients
    post_sg = posterior.detach()
    kl_term1 = (post_sg * (post_sg.log() - prior.log())).sum(-1)  # [..., num_cats]

    # KL(posterior || sg(prior)): only posterior gets gradients
    prior_sg = prior.detach()
    kl_term2 = (posterior * (posterior.log() - prior_sg.log())).sum(
        -1
    )  # [..., num_cats]

    # Free bits per categorical
    kl_term1 = kl_term1.clamp_min(free_bits)
    kl_term2 = kl_term2.clamp_min(free_bits)

    # Average over categoricals and batch
    return (alpha * kl_term1 + (1.0 - alpha) * kl_term2).mean()


# ---------------------------------------------------------------------------
# DreamerV3ModelLoss
# ---------------------------------------------------------------------------


class DreamerV3ModelLoss(LossModule):
    """DreamerV3 World Model Loss.

    Computes three terms:

    1. **KL loss** — balanced KL between prior and posterior categorical
       distributions (see :func:`categorical_kl_balanced`).
    2. **Reconstruction loss** — symlog MSE between predicted and true
       observations.
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
        kl_alpha (float, optional): KL balancing factor (α in the paper).
            Default: 0.8.
        free_bits (float, optional): Minimum KL per categorical in nats.
            Default: 1.0.
        reco_loss (str, optional): Reconstruction loss type (``"l2"`` or
            ``"l1"``). Default: ``"l2"``.
        reward_two_hot (bool, optional): If ``True``, uses two-hot cross-entropy
            for the reward loss; otherwise uses symlog MSE. Default: ``True``.
        num_reward_bins (int, optional): Number of bins for the two-hot reward
            distribution. Default: 255.
        global_average (bool, optional): If ``True``, averages losses over all
            dimensions. Otherwise sums over non-batch/time dims first. Default:
            ``False``.
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys.

        Attributes:
            reward (NestedKey): Predicted reward. Defaults to ``"reward"``.
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
        """

        reward: NestedKey = "reward"
        true_reward: NestedKey = "true_reward"
        prior_logits: NestedKey = "prior_logits"
        posterior_logits: NestedKey = "posterior_logits"
        pixels: NestedKey = "pixels"
        reco_pixels: NestedKey = "reco_pixels"
        continue_pred: NestedKey = "continue_pred"
        done: NestedKey = "done"

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
        kl_alpha: float = 0.8,
        free_bits: float = 1.0,
        reco_loss: str = "l2",
        reward_two_hot: bool = True,
        num_reward_bins: int = _DEFAULT_NUM_BINS,
        global_average: bool = False,
    ):
        super().__init__()
        self.world_model = world_model
        self.lambda_kl = lambda_kl
        self.lambda_reco = lambda_reco
        self.lambda_reward = lambda_reward
        self.lambda_continue = lambda_continue
        self.kl_alpha = kl_alpha
        self.free_bits = free_bits
        self.reco_loss = reco_loss
        self.reward_two_hot = reward_two_hot
        self.global_average = global_average
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
            reco_loss = reco_loss.sum((-3, -2, -1))
        reco_loss = reco_loss.mean().unsqueeze(-1)

        # ---- Reward loss ----
        true_reward = tensordict.get(("next", self.tensor_keys.true_reward))
        pred_reward = tensordict.get(("next", self.tensor_keys.reward))

        if self.reward_two_hot:
            # pred_reward should be logits over reward_bins
            targets = two_hot_encode(symlog(true_reward.squeeze(-1)), self.reward_bins)
            reward_loss = -(targets * torch.log_softmax(pred_reward, dim=-1)).sum(-1)
        else:
            reward_loss = (symlog(true_reward) - symlog(pred_reward)).pow(2).squeeze(-1)
        reward_loss = reward_loss.mean().unsqueeze(-1)

        td_out = TensorDict(
            loss_model_kl=self.lambda_kl * kl_loss,
            loss_model_reco=self.lambda_reco * reco_loss,
            loss_model_reward=self.lambda_reward * reward_loss,
        )

        # ---- Optional continue loss ----
        if self.lambda_continue > 0:
            continue_pred = tensordict.get(
                ("next", self.tensor_keys.continue_pred), None
            )
            done = tensordict.get(("next", self.tensor_keys.done), None)
            if continue_pred is not None and done is not None:
                # continue = 1 - done; BCE with logits
                continue_target = (~done).float()
                continue_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                    continue_pred.squeeze(-1), continue_target.squeeze(-1)
                ).unsqueeze(-1)
                td_out.set("loss_model_continue", self.lambda_continue * continue_loss)

        self._clear_weakrefs(tensordict, td_out)
        return td_out, tensordict.data


# ---------------------------------------------------------------------------
# DreamerV3ActorLoss
# ---------------------------------------------------------------------------


class DreamerV3ActorLoss(LossModule):
    """DreamerV3 Actor Loss.

    Rolls out imagined trajectories in latent space using the world model
    environment, then computes:

    .. code-block:: text

        loss_actor = -E[log π(a_t | z_t) * sg(A_t)] - η * H[π(· | z_t)]

    where ``A_t = V_λ(z_t) - v(z_t)`` is the advantage (normalized lambda
    return minus baseline) and ``η`` is the entropy bonus weight.

    When the actor is a reparameterizable (continuous) policy the
    reparameterization gradient is used directly instead of REINFORCE.

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        actor_model (TensorDictModule): The actor / policy network.
        value_model (TensorDictModule): The value network.
        model_based_env (DreamerEnv): The imagination environment.
        imagination_horizon (int, optional): Rollout length inside imagination.
            Default: 15.
        discount_loss (bool, optional): If ``True``, discount the actor loss
            with a cumulative gamma factor. Default: ``True``.
        entropy_bonus (float, optional): Weight for the entropy regularisation
            term ``η``. Default: ``3e-4``.
        use_reinforce (bool, optional): If ``True``, uses REINFORCE (log-prob
            * stop-gradient advantage). If ``False``, uses the straight
            reparameterization gradient (suitable for continuous Gaussian
            actors). Default: ``False``.
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys.

        Attributes:
            state (NestedKey): Stochastic latent state. Defaults to ``"state"``.
            belief (NestedKey): Deterministic GRU hidden state. Defaults to ``"belief"``.
            reward (NestedKey): Imagined reward. Defaults to ``"reward"``.
            value (NestedKey): State value. Defaults to ``"state_value"``.
            action_log_prob (NestedKey): Log-prob of the taken action.
                Defaults to ``"action_log_prob"``.
            done (NestedKey): Done flag. Defaults to ``"done"``.
            terminated (NestedKey): Terminated flag. Defaults to ``"terminated"``.
        """

        state: NestedKey = "state"
        belief: NestedKey = "belief"
        reward: NestedKey = "reward"
        value: NestedKey = "state_value"
        action_log_prob: NestedKey = "action_log_prob"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"

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
        imagination_horizon: int = 15,
        discount_loss: bool = True,
        entropy_bonus: float = 3e-4,
        use_reinforce: bool = False,
        gamma: int | None = None,
        lmbda: int | None = None,
    ):
        super().__init__()
        self.actor_model = actor_model
        self.__dict__["value_model"] = value_model
        self.model_based_env = model_based_env
        self.imagination_horizon = imagination_horizon
        self.discount_loss = discount_loss
        self.entropy_bonus = entropy_bonus
        self.use_reinforce = use_reinforce
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

        reward = fake_data.get(("next", self.tensor_keys.reward))
        next_value = next_tensordict.get(self.tensor_keys.value)
        lambda_target = self.lambda_target(reward, next_value)
        fake_data.set("lambda_target", lambda_target)

        if self.discount_loss:
            gamma = self.value_estimator.gamma.to(tensordict.device)
            discount = gamma.expand(lambda_target.shape).clone()
            discount[..., 0, :] = 1
            discount = discount.cumprod(dim=-2)
        else:
            discount = torch.ones_like(lambda_target)

        if self.use_reinforce:
            # REINFORCE: log π(a|z) * sg(A_t)
            log_prob = fake_data.get(self.tensor_keys.action_log_prob)
            with hold_out_net(self.value_model):
                baseline_td = fake_data.select(*self.value_model.in_keys, strict=False)
                self.value_model(baseline_td)
            baseline = baseline_td.get(self.tensor_keys.value)
            advantage = (lambda_target - baseline).detach()
            actor_loss = -(discount * log_prob * advantage).sum((-2, -1)).mean()
        else:
            # Reparameterization gradient
            actor_loss = -(discount * lambda_target).sum((-2, -1)).mean()

        # Entropy bonus (if actor provides log_prob)
        log_prob_for_entropy = fake_data.get(self.tensor_keys.action_log_prob, None)
        if log_prob_for_entropy is not None and self.entropy_bonus > 0:
            entropy = -(discount * log_prob_for_entropy).sum((-2, -1)).mean()
            actor_loss = actor_loss - self.entropy_bonus * entropy

        loss_tensordict = TensorDict({"loss_actor": actor_loss}, [])
        self._clear_weakrefs(tensordict, loss_tensordict)
        return loss_tensordict, fake_data.data

    def lambda_target(self, reward: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
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

    def make_value_estimator(self, value_type: ValueEstimators = None, **hyperparams):
        if value_type is None:
            value_type = self.default_value_estimator

        if isinstance(value_type, ValueEstimatorBase) or (
            isinstance(value_type, type) and issubclass(value_type, ValueEstimatorBase)
        ):
            return LossModule.make_value_estimator(self, value_type, **hyperparams)

        self.value_type = value_type
        hp = dict(default_value_kwargs(value_type))
        if hasattr(self, "gamma"):
            hp["gamma"] = self.gamma
        hp.update(hyperparams)
        value_net = None
        if value_type is ValueEstimators.TD1:
            self._value_estimator = TD1Estimator(**hp, value_network=value_net)
        elif value_type is ValueEstimators.TD0:
            self._value_estimator = TD0Estimator(**hp, value_network=value_net)
        elif value_type is ValueEstimators.GAE:
            raise NotImplementedError(
                f"Value type {value_type} is not implemented for {type(self)}."
            )
        elif value_type is ValueEstimators.TDLambda:
            if hasattr(self, "lmbda"):
                hp["lmbda"] = self.lmbda
            self._value_estimator = TDLambdaEstimator(
                **hp, value_network=value_net, vectorized=True
            )
        else:
            raise NotImplementedError(f"Unknown value type {value_type}")

        self._value_estimator.set_keys(
            value=self.tensor_keys.value,
            value_target="value_target",
        )


# ---------------------------------------------------------------------------
# DreamerV3ValueLoss
# ---------------------------------------------------------------------------


class DreamerV3ValueLoss(LossModule):
    """DreamerV3 Value Loss.

    Trains the value network to predict the lambda-target computed by
    :class:`DreamerV3ActorLoss`. Supports two loss modes:

    - ``"symlog_mse"`` (default): ``(symlog(v_pred) - symlog(target))^2``
    - ``"two_hot"``: Two-hot cross-entropy over a fixed bin grid (matches the
      full DreamerV3 distribution-valued critic).

    Reference: https://arxiv.org/abs/2301.04104

    Args:
        value_model (TensorDictModule): The value network.
        value_loss (str, optional): Loss type — ``"symlog_mse"`` or ``"two_hot"``.
            Default: ``"symlog_mse"``.
        discount_loss (bool, optional): If ``True``, discounts the loss with
            a cumulative gamma factor. Default: ``True``.
        gamma (float, optional): Discount factor used when ``discount_loss=True``.
            Default: ``0.99``.
        num_value_bins (int, optional): Number of bins for ``"two_hot"`` loss.
            Default: 255.
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys.

        Attributes:
            value (NestedKey): Predicted value key. Defaults to ``"state_value"``.
        """

        value: NestedKey = "state_value"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys

    value_model: TensorDictModule

    def __init__(
        self,
        value_model: TensorDictModule,
        value_loss: str = "symlog_mse",
        discount_loss: bool = True,
        gamma: float = 0.99,
        num_value_bins: int = _DEFAULT_NUM_BINS,
    ):
        super().__init__()
        self.value_model = value_model
        self.value_loss = value_loss
        self.gamma = gamma
        self.discount_loss = discount_loss
        if value_loss not in ("symlog_mse", "two_hot"):
            raise ValueError(
                f"value_loss must be 'symlog_mse' or 'two_hot', got '{value_loss}'"
            )
        self.register_buffer("value_bins", _default_bins(num_value_bins))

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    @_maybe_record_function_decorator("dreamer_v3/value_loss")
    def forward(self, fake_data) -> tuple[TensorDict, TensorDict]:
        lambda_target = fake_data.get("lambda_target")

        tensordict_select = fake_data.select(*self.value_model.in_keys, strict=False)
        self.value_model(tensordict_select)
        value_pred = tensordict_select.get(self.tensor_keys.value)

        # lambda_target shape: [N, 1] (flat) or [B, T, 1] (batch × time)
        # Squeeze the trailing 1 for loss computation
        target_sq = lambda_target.squeeze(-1)  # [N] or [B, T]

        if self.discount_loss and target_sq.ndim >= 2:
            discount = self.gamma * torch.ones_like(target_sq)
            discount[..., 0] = 1
            discount = discount.cumprod(dim=-1)
        else:
            discount = torch.ones_like(target_sq)

        if self.value_loss == "two_hot":
            # value_pred: logits over value_bins [..., num_bins]
            targets = two_hot_encode(symlog(target_sq), self.value_bins)
            loss = -(targets * torch.log_softmax(value_pred, dim=-1)).sum(-1)
        else:
            # symlog MSE
            loss = (symlog(value_pred.squeeze(-1)) - symlog(target_sq)).pow(2)

        value_loss = (discount * loss).mean()

        loss_tensordict = TensorDict({"loss_value": value_loss})
        self._clear_weakrefs(fake_data, loss_tensordict)
        return loss_tensordict, fake_data
