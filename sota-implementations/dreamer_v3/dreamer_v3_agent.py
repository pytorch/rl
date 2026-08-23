# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""The DreamerV3 networks, acting policy, optimizer and builders."""
from __future__ import annotations

import importlib.util
from collections.abc import Callable, Iterable

import torch
from dreamer_v3_utils import latent_state_dim, POLICY_RNG_STREAM, stream_seed
from omegaconf import DictConfig
from tensordict import TensorDictBase
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictModuleBase,
    TensorDictSequential,
)
from tensordict.utils import NestedKey

from torchrl.data import Unbounded
from torchrl.envs import EnvBase, StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import (
    CatTensors,
    ClipTransform,
    DoubleToFloat,
    InitTracker,
    TensorDictPrimer,
)
from torchrl.modules import DreamerV3MLP, SymExpTwoHot, WorldModelWrapper
from torchrl.modules.distributions.continuous import IndependentNormal
from torchrl.modules.models.model_based_v3 import (
    _dreamer_v3_init,
    RSSMPosteriorV3,
    RSSMPriorV3,
    RSSMRolloutV3,
)
from torchrl.objectives import symexp, symlog

_has_dm_control = importlib.util.find_spec("dm_control") is not None


def _to_float(value: torch.Tensor) -> torch.Tensor:
    return value.float()


def _cast_float(key: NestedKey) -> TensorDictModule:
    return TensorDictModule(_to_float, in_keys=[key], out_keys=[key])


# --- Networks and the acting policy ---


class _DreamerV3Decoder(torch.nn.Module):
    """A shared trunk with one head for each observation event."""

    def __init__(
        self,
        cfg: DictConfig,
        input_dim: int,
        event_dims: tuple[int, ...],
    ):
        super().__init__()
        if not event_dims or any(size <= 0 for size in event_dims):
            raise ValueError(
                f"Decoder event dimensions must be positive: {event_dims}."
            )
        self.backbone = DreamerV3MLP(
            input_dim,
            None,
            depth=cfg.networks.decoder_layers,
            num_cells=cfg.networks.hidden_dim,
            norm_eps=cfg.networks.norm_eps,
        )
        self.output_heads = torch.nn.ModuleList(
            torch.nn.Linear(cfg.networks.hidden_dim, size) for size in event_dims
        )
        self.output_heads.apply(_dreamer_v3_init)

    def forward(self, state: torch.Tensor, belief: torch.Tensor) -> torch.Tensor:
        hidden = self.backbone(state, belief)
        return torch.cat(tuple(head(hidden) for head in self.output_heads), -1)


class _DreamerV3Actor(torch.nn.Module):
    def __init__(self, cfg: DictConfig, action_dim: int):
        super().__init__()
        state_dim = latent_state_dim(cfg)
        self.backbone = DreamerV3MLP(
            state_dim + cfg.networks.rnn_hidden_dim,
            None,
            depth=cfg.networks.actor_layers,
            num_cells=cfg.networks.hidden_dim,
            norm_eps=cfg.networks.norm_eps,
        )
        self.mean_head = torch.nn.Linear(cfg.networks.hidden_dim, action_dim)
        self.std_head = torch.nn.Linear(cfg.networks.hidden_dim, action_dim)
        self.mean_head.apply(_dreamer_v3_init)
        self.std_head.apply(_dreamer_v3_init)
        with torch.no_grad():
            self.mean_head.weight.mul_(0.01)
            self.std_head.weight.mul_(0.01)
        self.action_dim = action_dim
        self.min_std = cfg.networks.policy_min_std
        self.max_std = cfg.networks.policy_max_std

    def forward(
        self, state: torch.Tensor, belief: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Belief before state, unlike the decoder. The order is deliberate.
        hidden = self.backbone(belief, state)
        mean = self.mean_head(hidden)
        std = self.std_head(hidden)
        mean = mean.tanh()
        std = (self.max_std - self.min_std) * torch.sigmoid(std + 2) + self.min_std
        # The Normal parameters stay FP32, also under BF16 autocast.
        return mean.float(), std.float()


class _DreamerV3PolicyFilter(torch.nn.Module):
    def __init__(
        self,
        prior_net: torch.nn.Module,
        posterior_net: torch.nn.Module,
    ):
        super().__init__()
        self.prior_net = prior_net
        self.posterior_net = posterior_net

    def forward(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        previous_action: torch.Tensor,
        encoded_latents: torch.Tensor,
        is_init: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        reset = is_init
        while reset.ndim < state.ndim:
            reset = reset.unsqueeze(-1)
        state = torch.where(reset, 0, state)
        belief = torch.where(reset, 0, belief)
        previous_action = torch.where(reset, 0, previous_action)
        # Advance the recurrence only. The posterior reads the observation.
        belief = self.prior_net._update_belief(state, belief, previous_action)
        _, state = self.posterior_net(belief, encoded_latents)
        # The collector and the replay entries use FP32.
        return state.float(), belief.float()


class _DreamerV3PolicyCarry(torch.nn.Module):
    def forward(
        self,
        state: torch.Tensor,
        belief: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return state.float(), belief.float(), action.float()


class _DreamerV3AutocastPolicy(TensorDictModuleBase):
    """Run the real-environment policy in BF16 on CUDA devices."""

    def __init__(self, module: TensorDictModuleBase, enabled: bool):
        super().__init__()
        self.module = module
        self.enabled = enabled
        self.in_keys = module.in_keys
        self.out_keys = module.out_keys

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        state = tensordict.get("state")
        with torch.autocast(
            device_type=state.device.type,
            dtype=torch.bfloat16,
            enabled=self.enabled and state.device.type == "cuda",
        ):
            return self.module(tensordict)


class DreamerV3BehaviorPolicySync:
    """Give the behavior policy the learner parameters after the next action.

    The snapshot comes from before the first update. Later updates do not
    replace it.
    """

    def __init__(
        self,
        learner_policy: torch.nn.Module,
        behavior_policy: torch.nn.Module,
    ):
        self._learner_policy = learner_policy
        self._behavior_policy = behavior_policy
        learner_parameters = tuple(learner_policy.named_parameters())
        behavior_parameters = tuple(behavior_policy.named_parameters())
        learner_names = tuple(name for name, _ in learner_parameters)
        behavior_names = tuple(name for name, _ in behavior_parameters)
        if learner_names != behavior_names:
            raise RuntimeError(
                "Learner and behavior policies must have identical parameter trees."
            )
        self._learner_parameters = tuple(
            parameter for _, parameter in learner_parameters
        )
        self._behavior_parameters = tuple(
            parameter for _, parameter in behavior_parameters
        )
        if any(
            learner.shape != behavior.shape
            for learner, behavior in zip(
                self._learner_parameters, self._behavior_parameters
            )
        ):
            raise RuntimeError(
                "Learner and behavior policy parameter shapes must be identical."
            )
        self._pending: tuple[torch.Tensor, ...] | None = None

    @property
    def has_pending(self) -> bool:
        return self._pending is not None

    @torch.no_grad()
    def stage_before_training(self) -> None:
        if self._pending is None:
            self._pending = tuple(
                parameter.detach().clone() for parameter in self._learner_parameters
            )

    @torch.no_grad()
    def apply_after_action(self) -> None:
        if self._pending is None:
            return
        for target, source in zip(self._behavior_parameters, self._pending):
            target.copy_(source)
        self._pending = None


class DreamerV3SeededPolicy(TensorDictModuleBase):
    """Give the policy its own random stream, with a new seed for each call."""

    def __init__(self, module: TensorDictModuleBase, seed: int):
        super().__init__()
        self.module = module
        self.seed = seed
        self.counter = 0
        self.in_keys = module.in_keys
        self.out_keys = module.out_keys

    def reset_counter(self) -> None:
        """Restart the counter, because a setup call can move it before step 0."""
        self.counter = 0

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        reference = tensordict.get("state", None)
        if reference is None:
            reference = tensordict.get(self.in_keys[0])
        devices = [reference.device] if reference.device.type == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(stream_seed(self.seed, self.counter, POLICY_RNG_STREAM))
            self.counter += 1
            return self.module(tensordict)


# --- Optimizer ---


class DreamerV3Optimizer(torch.optim.Optimizer):
    """The DreamerV3 optimizer: AGC, RMS scaling, momentum and warmup.

    AGC clips each gradient to the ``agc`` fraction of its parameter norm.
    """

    def __init__(
        self,
        parameters: Iterable[torch.nn.Parameter],
        *,
        lr: float = 4e-5,
        agc: float = 0.3,
        parameter_norm_min: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-20,
        warmup_steps: int = 1000,
    ):
        super().__init__(
            parameters,
            {
                "lr": lr,
                "agc": agc,
                "parameter_norm_min": parameter_norm_min,
                "beta1": beta1,
                "beta2": beta2,
                "eps": eps,
                "warmup_steps": warmup_steps,
                "step": 0,
            },
        )

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            group["step"] += 1
            step = group["step"]
            warmup_steps = group["warmup_steps"]
            schedule_step = step - 1
            warmup = min(1.0, schedule_step / warmup_steps) if warmup_steps else 1.0
            learning_rate = group["lr"] * warmup

            # Group by device and dtype for the multi-tensor kernels.
            buckets: dict[
                tuple[torch.device, torch.dtype], list[torch.nn.Parameter]
            ] = {}
            for parameter in group["params"]:
                if parameter.grad is not None:
                    buckets.setdefault((parameter.device, parameter.dtype), []).append(
                        parameter
                    )

            for parameters in buckets.values():
                gradients = [parameter.grad.float() for parameter in parameters]
                if group["agc"]:
                    gradient_norms = list(torch._foreach_norm(gradients))
                    parameter_norms = list(
                        torch._foreach_norm(
                            [parameter.detach().float() for parameter in parameters]
                        )
                    )
                    torch._foreach_clamp_min_(
                        parameter_norms, group["parameter_norm_min"]
                    )
                    maximum_norms = torch._foreach_mul(parameter_norms, group["agc"])
                    gradient_denominators = torch._foreach_maximum(
                        gradient_norms, maximum_norms
                    )
                    gradient_scales = torch._foreach_div(
                        maximum_norms, gradient_denominators
                    )
                    gradients = list(torch._foreach_mul(gradients, gradient_scales))

                rms = []
                momentum = []
                for parameter in parameters:
                    state = self.state[parameter]
                    if not state:
                        state["rms"] = torch.zeros_like(parameter, dtype=torch.float32)
                        state["momentum"] = torch.zeros_like(
                            parameter, dtype=torch.float32
                        )
                    rms.append(state["rms"])
                    momentum.append(state["momentum"])
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                torch._foreach_mul_(rms, beta2)
                torch._foreach_addcmul_(rms, gradients, gradients, value=1 - beta2)
                rms_hat = torch._foreach_div(rms, 1 - beta2**step)
                rms_denominator = torch._foreach_sqrt(rms_hat)
                torch._foreach_add_(rms_denominator, group["eps"])
                normalized = torch._foreach_div(gradients, rms_denominator)
                torch._foreach_mul_(momentum, beta1)
                torch._foreach_add_(momentum, normalized, alpha=1 - beta1)
                momentum_hat = torch._foreach_div(momentum, 1 - beta1**step)
                if parameters[0].dtype != torch.float32:
                    momentum_hat = [
                        update.to(parameter.dtype)
                        for update, parameter in zip(momentum_hat, parameters)
                    ]
                torch._foreach_add_(parameters, momentum_hat, alpha=-learning_rate)
        return loss


# --- Builders ---


def make_env(cfg: DictConfig, seed: int | None = 0) -> TransformedEnv:
    if cfg.env.backend == "gym":
        base_env = GymEnv(cfg.env.name, device="cpu")
    elif cfg.env.backend == "dm_control":
        if not _has_dm_control:
            raise ImportError(
                "The DMC DreamerV3 preset requires dm_control. Install the "
                "optional dm_control dependencies before running it."
            )
        from torchrl.envs.libs.dm_control import DMControlEnv  # noqa: PLC0415

        # Seed at construction: set_seed() also resets, which moves the stream.
        base_env = DMControlEnv(
            cfg.env.name,
            cfg.env.task,
            device="cpu",
            _seed=seed if cfg.env.use_seed else None,
        )
    else:
        raise ValueError(f"Unknown environment backend {cfg.env.backend!r}.")

    env = TransformedEnv(base_env)
    if cfg.env.backend == "dm_control":
        env.append_transform(
            CatTensors(
                # The encoder reads the keys in sorted order.
                in_keys=sorted(base_env.observation_spec.keys()),
                out_key="observation",
            )
        )
        # The env gets the clipped action. The buffer keeps the raw sample.
        env.append_transform(ClipTransform(in_keys_inv=["action"], low=-1.0, high=1.0))
    env.append_transform(DoubleToFloat())
    env.append_transform(StepCounter(max_steps=cfg.env.max_episode_steps))
    env.append_transform(InitTracker())
    if cfg.env.backend != "dm_control" and cfg.env.use_seed:
        env.set_seed(seed)
    return env


def make_primed_env(
    cfg: DictConfig, seed: int | None, state_dim: int, action_dim: int
) -> TransformedEnv:
    """Build an environment primed with latent, belief and previous action."""
    return TransformedEnv(
        make_env(cfg, seed),
        TensorDictPrimer(
            random=False,
            default_value=0,
            state=Unbounded(state_dim),
            belief=Unbounded(cfg.networks.rnn_hidden_dim),
            previous_action=Unbounded(action_dim),
        ),
    )


def build_world_model(
    *, cfg: DictConfig, obs_dim: int, action_dim: int
) -> tuple[TensorDictSequential, RSSMPriorV3, DreamerV3MLP, SymExpTwoHot, DreamerV3MLP]:
    """Build the world model: encoder, RSSM rollout, decoder and two heads."""
    state_dim = latent_state_dim(cfg)

    encoder = TensorDictSequential(
        TensorDictModule(
            symlog,
            in_keys=[("next", "observation")],
            out_keys=[("next", "symlog_observation")],
        ),
        TensorDictModule(
            DreamerV3MLP(
                in_features=obs_dim,
                # The output is the last hidden activation, with no projection.
                out_features=None,
                depth=cfg.networks.encoder_layers,
                num_cells=cfg.networks.hidden_dim,
                norm_eps=cfg.networks.norm_eps,
            ),
            in_keys=[("next", "symlog_observation")],
            out_keys=[("next", "encoded_latents")],
        ),
    )

    prior_net = RSSMPriorV3(
        action_shape=torch.Size([action_dim]),
        hidden_dim=cfg.networks.hidden_dim,
        rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
        num_categoricals=cfg.networks.num_categoricals,
        num_classes=cfg.networks.num_classes,
        action_dim=action_dim,
        unimix=cfg.networks.unimix,
        recurrent_model=cfg.networks.recurrent_model,
        num_blocks=cfg.networks.num_blocks,
        num_layers=cfg.networks.dynamics_layers,
        prior_num_layers=cfg.networks.prior_layers,
        norm_eps=cfg.networks.norm_eps,
    )
    rssm_prior = TensorDictModule(
        prior_net,
        in_keys=["state", "belief", "action"],
        out_keys=[
            ("next", "prior_logits"),
            ("next", "state"),
            ("next", "belief"),
        ],
    )

    posterior_net = RSSMPosteriorV3(
        hidden_dim=cfg.networks.hidden_dim,
        num_categoricals=cfg.networks.num_categoricals,
        num_classes=cfg.networks.num_classes,
        rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
        obs_embed_dim=cfg.networks.hidden_dim,
        unimix=cfg.networks.unimix,
        use_rms_norm=True,
        num_layers=cfg.networks.posterior_layers,
        norm_eps=cfg.networks.norm_eps,
    )
    rssm_posterior = TensorDictModule(
        posterior_net,
        in_keys=[("next", "belief"), ("next", "encoded_latents")],
        out_keys=[("next", "posterior_logits"), ("next", "state")],
    )

    # Only the reset record of an episode has is_init set, thus a sampled
    # window can cross an episode boundary.
    rollout = RSSMRolloutV3(rssm_prior, rssm_posterior, reset_key="is_init")
    if cfg.optimization.compile_rssm:
        rollout.compile_rollout(
            cfg.optimization.compile_rssm,
            unroll=(
                cfg.optimization.rssm_scan_unroll
                if cfg.optimization.compile_rssm == "scan"
                else 1
            ),
        )

    decoder_event_dims = tuple(cfg.networks.decoder_event_dims or (obs_dim,))
    if sum(decoder_event_dims) != obs_dim:
        raise ValueError(
            "Decoder event dimensions must sum to the flattened observation "
            f"size, got {decoder_event_dims} for {obs_dim}."
        )
    # One head for each event: AGC clips each head separately, thus a merged
    # head trains differently. The FP32 symexp keeps the loss symlog exact.
    decoder = TensorDictSequential(
        TensorDictModule(
            _DreamerV3Decoder(
                cfg,
                state_dim + cfg.networks.rnn_hidden_dim,
                decoder_event_dims,
            ),
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", "reco_symlog_observation")],
        ),
        _cast_float(("next", "reco_symlog_observation")),
        TensorDictModule(
            symexp,
            in_keys=[("next", "reco_symlog_observation")],
            out_keys=[("next", "reco_pixels")],
        ),
    )

    reward_net = DreamerV3MLP(
        in_features=state_dim + cfg.networks.rnn_hidden_dim,
        out_features=cfg.networks.num_reward_bins,
        depth=cfg.networks.reward_layers,
        num_cells=cfg.networks.hidden_dim,
        outscale=0.0,
        norm_eps=cfg.networks.norm_eps,
    )
    reward_decoder = SymExpTwoHot(cfg.networks.num_reward_bins)
    reward_head = TensorDictSequential(
        TensorDictModule(
            reward_net,
            in_keys=[("next", "belief"), ("next", "state")],
            out_keys=[("next", "reward_logits")],
        ),
        TensorDictModule(
            _to_float,
            in_keys=[("next", "reward_logits")],
            out_keys=[("next", "reward_logits")],
        ),
        TensorDictModule(
            reward_decoder,
            in_keys=[("next", "reward_logits")],
            out_keys=[("next", "reward")],
        ),
    )

    continuation_net = DreamerV3MLP(
        in_features=state_dim + cfg.networks.rnn_hidden_dim,
        out_features=1,
        depth=cfg.networks.reward_layers,
        num_cells=cfg.networks.hidden_dim,
        norm_eps=cfg.networks.norm_eps,
    )
    continuation_head = TensorDictSequential(
        TensorDictModule(
            continuation_net,
            in_keys=[("next", "belief"), ("next", "state")],
            out_keys=[("next", "continue_pred")],
        ),
        _cast_float(("next", "continue_pred")),
    )

    world_model = TensorDictSequential(
        encoder, rollout, decoder, reward_head, continuation_head
    )
    return world_model, prior_net, reward_net, reward_decoder, continuation_net


def build_imagination_model(
    *,
    prior_net: RSSMPriorV3,
    reward_net: DreamerV3MLP,
    reward_decoder: SymExpTwoHot,
    compile_prior: bool = False,
) -> WorldModelWrapper:
    """Build the imagination model from the trained world-model modules.

    ``compile_prior`` compiles the prior here only, not in the shared rollout.
    """
    transition_model = TensorDictSequential(
        TensorDictModule(
            torch.compile(prior_net, dynamic=False) if compile_prior else prior_net,
            in_keys=["state", "belief", "action"],
            out_keys=["_", "state", "belief"],
        )
    )
    reward_model = TensorDictSequential(
        TensorDictModule(
            reward_net,
            in_keys=["belief", "state"],
            out_keys=["reward_logits"],
        ),
        _cast_float("reward_logits"),
        TensorDictModule(
            reward_decoder,
            in_keys=["reward_logits"],
            out_keys=["reward"],
        ),
    )
    return WorldModelWrapper(transition_model, reward_model)


def build_continuation_model(*, continuation_net: DreamerV3MLP) -> TensorDictSequential:
    return TensorDictSequential(
        TensorDictModule(
            continuation_net,
            in_keys=["belief", "state"],
            out_keys=["continue_logits"],
        ),
        _cast_float("continue_logits"),
        TensorDictModule(
            torch.nn.Sigmoid(),
            in_keys=["continue_logits"],
            out_keys=["continuation"],
        ),
    )


def build_actor(
    *, cfg: DictConfig, action_dim: int
) -> ProbabilisticTensorDictSequential:
    actor_mlp = _DreamerV3Actor(cfg, action_dim)
    actor_model = ProbabilisticTensorDictSequential(
        TensorDictModule(
            actor_mlp,
            in_keys=["state", "belief"],
            out_keys=["loc", "scale"],
        ),
        ProbabilisticTensorDictModule(
            in_keys=["loc", "scale"],
            out_keys=["action"],
            default_interaction_type=InteractionType.RANDOM,
            distribution_class=IndependentNormal,
            return_log_prob=True,
            log_prob_key="action_log_prob",
        ),
    )
    return actor_model


def build_real_world_actor(
    *,
    world_model: TensorDictSequential,
    actor_model: ProbabilisticTensorDictSequential,
    mixed_precision: bool = False,
) -> TensorDictModuleBase:
    """Build the recurrent policy that acts in the real environment.

    The policy shares the trained encoder, prior, posterior and actor.
    """
    encoder_net = world_model[0][1].module
    rssm_rollout = world_model[1]
    prior_net = rssm_rollout.rssm_prior.module
    posterior_net = rssm_rollout.rssm_posterior.module
    policy = TensorDictSequential(
        TensorDictModule(
            symlog,
            in_keys=["observation"],
            out_keys=["symlog_observation"],
        ),
        TensorDictModule(
            encoder_net,
            in_keys=["symlog_observation"],
            out_keys=["encoded_latents"],
        ),
        TensorDictModule(
            _DreamerV3PolicyFilter(prior_net, posterior_net),
            in_keys=[
                "state",
                "belief",
                "previous_action",
                "encoded_latents",
                "is_init",
            ],
            out_keys=["state", "belief"],
        ),
        actor_model,
        TensorDictModule(
            _DreamerV3PolicyCarry(),
            in_keys=["state", "belief", "action"],
            out_keys=[
                ("next", "state"),
                ("next", "belief"),
                ("next", "previous_action"),
            ],
        ),
    )
    return _DreamerV3AutocastPolicy(policy, enabled=mixed_precision)


def build_value(*, cfg: DictConfig) -> TensorDictSequential:
    state_dim = latent_state_dim(cfg)
    value_model = TensorDictSequential(
        TensorDictModule(
            DreamerV3MLP(
                in_features=state_dim + cfg.networks.rnn_hidden_dim,
                out_features=cfg.networks.num_value_bins,
                depth=cfg.networks.value_layers,
                num_cells=cfg.networks.hidden_dim,
                outscale=0.0,
                norm_eps=cfg.networks.norm_eps,
            ),
            in_keys=["belief", "state"],
            out_keys=["state_value_logits"],
        ),
        _cast_float("state_value_logits"),
        TensorDictModule(
            SymExpTwoHot(cfg.networks.num_value_bins),
            in_keys=["state_value_logits"],
            out_keys=["state_value"],
        ),
    )
    return value_model


def build_mb_env(
    *,
    cfg: DictConfig,
    real_env: EnvBase,
    imagination_model: WorldModelWrapper,
    device: torch.device,
) -> DreamerEnv:
    """Build the imagination environment from the trained world model."""
    state_dim = latent_state_dim(cfg)
    primer_env = TransformedEnv(
        real_env,
        TensorDictPrimer(
            random=False,
            default_value=0,
            state=Unbounded(state_dim),
            belief=Unbounded(cfg.networks.rnn_hidden_dim),
        ),
    )
    mb_env = DreamerEnv(
        world_model=imagination_model,
        prior_shape=torch.Size([state_dim]),
        belief_shape=torch.Size([cfg.networks.rnn_hidden_dim]),
        device=device,
    )
    mb_env.set_specs_from_env(primer_env)
    with torch.no_grad():
        mb_env.rollout(3)
    return mb_env
