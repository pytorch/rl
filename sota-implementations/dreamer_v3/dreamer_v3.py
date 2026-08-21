# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 on Pendulum-v1 — minimal end-to-end training script.

State-based (not pixel-based) to keep the script compact — the 3-D obs is
treated as a flat feature vector with ``global_average=True`` in the model
loss. The wiring is still real:

- collector to replay buffer of sequences
- world model = MLP encoder + RSSMPriorV3 + RSSMPosteriorV3 + MLP decoder + reward head
- RSSM unrolled over each sequence via ``RSSMRolloutV3``
- actor trained via REINFORCE in imagination (``DreamerV3ActorLoss``)
- value trained on the same imagined rollout (``DreamerV3ValueLoss``)
- periodic eval rollouts in the real env, episode reward logged

Plots ``dreamer_v3_pendulum.png`` with two curves: (a) average eval reward,
(b) world-model KL / reconstruction / reward losses.

Usage::

    python sota-implementations/dreamer_v3/dreamer_v3.py \\
        collector.total_frames=5000 logger.eval_every=500
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)

from torchrl._utils import get_available_device, logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, ReplayBuffer, Unbounded
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import (
    CatTensors,
    DoubleToFloat,
    InitTracker,
    TensorDictPrimer,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import DreamerV3MLP, SymExpTwoHot, WorldModelWrapper
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules.models.dreamer_v3 import (
    RSSMPosteriorV3,
    RSSMPriorV3,
    RSSMRolloutV3,
)
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
    symexp,
    symlog,
)
from torchrl.objectives.utils import SoftUpdate, ValueEstimators

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None
_has_dm_control = importlib.util.find_spec("dm_control") is not None


class _DreamerV3Actor(torch.nn.Module):
    def __init__(self, cfg: DictConfig, action_dim: int):
        super().__init__()
        state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
        self.backbone = DreamerV3MLP(
            state_dim + cfg.networks.rnn_hidden_dim,
            2 * action_dim,
            depth=cfg.networks.actor_layers,
            num_cells=cfg.networks.hidden_dim,
            outscale=0.01,
            norm_eps=cfg.networks.norm_eps,
        )
        self.action_dim = action_dim
        self.min_std = cfg.networks.policy_min_std
        self.max_std = cfg.networks.policy_max_std

    def forward(self, state: torch.Tensor, belief: torch.Tensor):
        mean, std = self.backbone(state, belief).split(self.action_dim, -1)
        std = (self.max_std - self.min_std) * torch.sigmoid(std + 2) + self.min_std
        return mean, std


@torch.no_grad()
def adaptive_grad_clip_(parameters, clip: float, minimum: float = 1e-3) -> None:
    """Clip each parameter gradient relative to its parameter norm."""
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad_norm = parameter.grad.norm()
        parameter_norm = parameter.detach().norm().clamp_min(minimum)
        scale = (clip * parameter_norm / grad_norm.clamp_min(1e-12)).clamp_max(1)
        parameter.grad.mul_(scale)


def make_env(cfg: DictConfig, seed: int = 0) -> TransformedEnv:
    if cfg.env.backend == "gym":
        base_env = GymEnv(cfg.env.name, device="cpu")
    elif cfg.env.backend == "dm_control":
        if not _has_dm_control:
            raise ImportError(
                "The DMC DreamerV3 preset requires dm_control. Install the "
                "optional dm_control dependencies before running it."
            )
        from torchrl.envs.libs.dm_control import DMControlEnv

        base_env = DMControlEnv(cfg.env.name, cfg.env.task, device="cpu")
    else:
        raise ValueError(f"Unknown environment backend {cfg.env.backend!r}.")

    env = TransformedEnv(base_env)
    if cfg.env.backend == "dm_control":
        env.append_transform(
            CatTensors(
                in_keys=list(base_env.observation_spec.keys()),
                out_key="observation",
            )
        )
    env.append_transform(DoubleToFloat())
    env.append_transform(StepCounter(max_steps=cfg.env.max_episode_steps))
    env.append_transform(InitTracker())
    env.set_seed(seed)
    return env


def build_world_model(*, cfg: DictConfig, obs_dim: int, action_dim: int):
    """MLP encoder + RSSMRolloutV3 + MLP decoder + reward head.

    Returns a TensorDictSequential whose forward consumes a trajectory batch
    and writes every key DreamerV3ModelLoss expects.
    """
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes

    observation_encoder = DreamerV3MLP(
        in_features=obs_dim,
        out_features=cfg.networks.obs_embed_dim,
        depth=cfg.networks.encoder_layers,
        num_cells=cfg.networks.hidden_dim,
        norm_eps=cfg.networks.norm_eps,
    )
    encoder = TensorDictSequential(
        TensorDictModule(
            symlog,
            in_keys=[("next", "observation")],
            out_keys=[("next", "symlog_observation")],
        ),
        TensorDictModule(
            observation_encoder,
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
        obs_embed_dim=cfg.networks.obs_embed_dim,
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

    rollout = RSSMRolloutV3(rssm_prior, rssm_posterior)

    decoder = TensorDictSequential(
        TensorDictModule(
            DreamerV3MLP(
                in_features=state_dim + cfg.networks.rnn_hidden_dim,
                out_features=obs_dim,
                depth=cfg.networks.decoder_layers,
                num_cells=cfg.networks.hidden_dim,
                norm_eps=cfg.networks.norm_eps,
            ),
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", "reco_symlog_observation")],
        ),
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
            in_keys=[("next", "state"), ("next", "belief")],
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
    continuation_head = TensorDictModule(
        continuation_net,
        in_keys=[("next", "state"), ("next", "belief")],
        out_keys=[("next", "continue_pred")],
    )

    world_model = TensorDictSequential(
        encoder, rollout, decoder, reward_head, continuation_head
    )
    return (
        world_model,
        observation_encoder,
        prior_net,
        posterior_net,
        reward_net,
        reward_decoder,
        continuation_net,
    )


def build_imagination_model(*, prior_net, reward_net, reward_decoder):
    """Build imagination operators backed by the trained world-model heads."""
    transition_model = TensorDictSequential(
        TensorDictModule(
            prior_net,
            in_keys=["state", "belief", "action"],
            out_keys=["_", "state", "belief"],
        )
    )
    reward_model = TensorDictSequential(
        TensorDictModule(
            reward_net,
            in_keys=["state", "belief"],
            out_keys=["reward_logits"],
        ),
        TensorDictModule(
            reward_decoder,
            in_keys=["reward_logits"],
            out_keys=["reward"],
        ),
    )
    return WorldModelWrapper(transition_model, reward_model)


def build_continuation_model(*, continuation_net):
    """Build the imagination continuation predictor from the trained head."""
    return TensorDictSequential(
        TensorDictModule(
            continuation_net,
            in_keys=["state", "belief"],
            out_keys=["continue_logits"],
        ),
        TensorDictModule(
            torch.nn.Sigmoid(),
            in_keys=["continue_logits"],
            out_keys=["continuation"],
        ),
    )


def build_actor(*, cfg: DictConfig, action_dim: int):
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
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
            distribution_class=TanhNormal,
            return_log_prob=True,
            log_prob_key="action_log_prob",
        ),
    )
    with torch.no_grad():
        actor_model(
            TensorDict(
                {
                    "state": torch.randn(1, 2, state_dim),
                    "belief": torch.randn(1, 2, cfg.networks.rnn_hidden_dim),
                },
                [1],
            )
        )
    return actor_model


def build_real_actor(*, observation_encoder, posterior_net, prior_net, actor_model):
    """Build the observation-conditioned recurrent policy used in real envs."""
    return TensorDictSequential(
        TensorDictModule(
            symlog,
            in_keys=["observation"],
            out_keys=["symlog_observation"],
        ),
        TensorDictModule(
            observation_encoder,
            in_keys=["symlog_observation"],
            out_keys=["encoded_latents"],
        ),
        TensorDictModule(
            posterior_net,
            in_keys=["belief", "encoded_latents"],
            out_keys=["_", "state"],
        ),
        actor_model,
        TensorDictModule(
            prior_net,
            in_keys=["state", "belief", "action"],
            out_keys=["_", "_", ("next", "belief")],
        ),
    )


def build_value(*, cfg: DictConfig):
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
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
            in_keys=["state", "belief"],
            out_keys=["state_value_logits"],
        ),
        TensorDictModule(
            SymExpTwoHot(cfg.networks.num_value_bins),
            in_keys=["state_value_logits"],
            out_keys=["state_value"],
        ),
    )
    with torch.no_grad():
        value_model(
            TensorDict(
                {
                    "state": torch.randn(1, 2, state_dim),
                    "belief": torch.randn(1, 2, cfg.networks.rnn_hidden_dim),
                },
                [1],
            )
        )
    return value_model


def build_mb_env(*, cfg: DictConfig, real_env, imagination_model, device: torch.device):
    """Imagination env backed by the trained prior and reward head."""
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
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


@torch.no_grad()
def eval_episode_reward(
    env, actor, num_episodes: int, max_episode_steps: int
) -> torch.Tensor:
    totals = []
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(num_episodes):
            td = env.rollout(
                max_steps=max_episode_steps,
                policy=actor,
                break_when_any_done=True,
                auto_cast_to_device=True,
            )
            totals.append(td.get(("next", "reward")).sum())
    return torch.stack(totals).mean()


@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.env.seed)

    device = (
        torch.device(cfg.optimization.device)
        if cfg.optimization.device
        else get_available_device()
    )
    real_env = make_env(cfg, cfg.env.seed)
    obs_dim = real_env.observation_spec["observation"].shape[0]
    action_dim = real_env.action_spec.shape[0]
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes

    (
        world_model,
        observation_encoder,
        prior_net,
        posterior_net,
        reward_net,
        reward_decoder,
        continuation_net,
    ) = build_world_model(cfg=cfg, obs_dim=obs_dim, action_dim=action_dim)
    world_model = world_model.to(device)
    imagination_model = build_imagination_model(
        prior_net=prior_net,
        reward_net=reward_net,
        reward_decoder=reward_decoder,
    ).to(device)
    continuation_model = build_continuation_model(continuation_net=continuation_net).to(
        device
    )
    actor_model = build_actor(cfg=cfg, action_dim=action_dim).to(device)
    real_actor = build_real_actor(
        observation_encoder=observation_encoder,
        posterior_net=posterior_net,
        prior_net=prior_net,
        actor_model=actor_model,
    )
    value_model = build_value(cfg=cfg).to(device)
    mb_env = build_mb_env(
        cfg=cfg,
        real_env=make_env(cfg, cfg.env.seed + 1),
        imagination_model=imagination_model,
        device=device,
    )

    model_loss = DreamerV3ModelLoss(
        world_model,
        num_reward_bins=cfg.networks.num_reward_bins,
        free_bits=cfg.optimization.free_bits,
        kl_mode="separate",
        lambda_dynamic=cfg.optimization.dynamic_loss_weight,
        lambda_representation=cfg.optimization.representation_loss_weight,
        unimix=cfg.networks.unimix,
        lambda_continue=1.0,
        continue_target_scale=1 - 1 / cfg.optimization.continuation_horizon,
        global_average=False,
    ).to(device)
    model_loss.set_keys(pixels="observation")
    actor_loss = DreamerV3ActorLoss(
        actor_model,
        value_model,
        mb_env,
        continuation_model=continuation_model,
        imagination_horizon=cfg.optimization.imagination_horizon,
        use_reinforce=cfg.optimization.use_reinforce,
        return_normalization_rate=cfg.optimization.return_normalization_rate,
        return_normalization_min_scale=cfg.optimization.return_normalization_min_scale,
    )
    actor_loss.make_value_estimator(
        ValueEstimators.TDLambda,
        gamma=cfg.optimization.gamma,
        lmbda=cfg.optimization.lmbda,
    )
    actor_loss.to(device)
    value_loss = DreamerV3ValueLoss(
        value_model,
        value_loss="two_hot",
        num_value_bins=cfg.networks.num_value_bins,
        actor_loss=actor_loss,
        slow_critic_regularization=cfg.optimization.slow_critic_regularization,
    ).to(device)
    value_target_updater = SoftUpdate(value_loss, tau=cfg.optimization.slow_critic_tau)

    optimizer_kwargs = {
        "lr": cfg.optimization.lr,
        "eps": cfg.optimization.adam_eps,
        "betas": (0.9, 0.999),
    }
    opt_model = torch.optim.Adam(model_loss.parameters(), **optimizer_kwargs)
    opt_actor = torch.optim.Adam(actor_model.parameters(), **optimizer_kwargs)
    opt_value = torch.optim.Adam(value_loss.parameters(), **optimizer_kwargs)
    schedulers = [
        torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1 / cfg.optimization.warmup_steps,
            total_iters=cfg.optimization.warmup_steps,
        )
        for optimizer in (opt_model, opt_actor, opt_value)
    ]

    explore_env = TransformedEnv(
        make_env(cfg, cfg.env.seed + 2),
        TensorDictPrimer(
            random=False,
            default_value=0,
            state=Unbounded(state_dim),
            belief=Unbounded(cfg.networks.rnn_hidden_dim),
        ),
    )

    collector = Collector(
        explore_env,
        real_actor,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        policy_device=device,
        env_device="cpu",
        storing_device="cpu",
        exploration_type=ExplorationType.RANDOM
        if cfg.collector.exploration == "random"
        else ExplorationType.DETERMINISTIC,
    )

    rb = ReplayBuffer(
        storage=LazyTensorStorage(max_size=cfg.replay_buffer.buffer_size),
        sampler=SliceSampler(
            slice_len=cfg.replay_buffer.seq_len,
            traj_key=("collector", "traj_ids"),
        ),
        batch_size=cfg.replay_buffer.batch_size * cfg.replay_buffer.seq_len,
    )

    env_step = 0
    history_steps: list[int] = []
    history_eval: list[torch.Tensor] = []
    loss_history: list[torch.Tensor] = []
    record_loss_history = bool(cfg.logger.output_plot and _has_matplotlib)
    next_eval = 0

    eval_env = TransformedEnv(
        make_env(cfg, cfg.env.seed + 100),
        TensorDictPrimer(
            random=False,
            default_value=0,
            state=Unbounded(state_dim),
            belief=Unbounded(cfg.networks.rnn_hidden_dim),
        ),
    )

    warmup = (
        cfg.replay_buffer.warmup_factor
        * cfg.replay_buffer.batch_size
        * cfg.replay_buffer.seq_len
    )

    updates_per_batch = cfg.optimization.updates_per_batch
    if cfg.optimization.train_ratio is not None:
        updates_per_batch = max(
            1,
            round(
                cfg.collector.frames_per_batch
                * cfg.optimization.train_ratio
                / (cfg.replay_buffer.batch_size * cfg.replay_buffer.seq_len)
            ),
        )

    for data in collector:
        replay_data = data.select(
            "action",
            "is_init",
            ("next", "observation"),
            ("next", "reward"),
            ("next", "terminated"),
            ("collector", "traj_ids"),
        )
        rb.extend(replay_data.reshape(-1))
        env_step += data.numel()

        if len(rb) < warmup:
            continue

        batch_losses = torch.empty(updates_per_batch, 4, device=device)
        for update_index in range(updates_per_batch):
            sample = (
                rb.sample()
                .reshape(cfg.replay_buffer.batch_size, cfg.replay_buffer.seq_len)
                .to(device)
            )

            sample.set(
                "state",
                torch.zeros(
                    cfg.replay_buffer.batch_size,
                    cfg.replay_buffer.seq_len,
                    state_dim,
                    device=sample.device,
                ),
            )
            sample.set(
                "belief",
                torch.zeros(
                    cfg.replay_buffer.batch_size,
                    cfg.replay_buffer.seq_len,
                    cfg.networks.rnn_hidden_dim,
                    device=sample.device,
                ),
            )

            m_td, model_out = model_loss(sample)
            model_kl = m_td["loss_model_dynamic"] + m_td["loss_model_representation"]
            total_m = (
                model_kl
                + m_td["loss_model_reco"]
                + m_td["loss_model_reward"]
                + m_td["loss_model_continue"]
            ).squeeze()
            opt_model.zero_grad(set_to_none=True)
            total_m.backward()
            adaptive_grad_clip_(
                model_loss.parameters(), cfg.optimization.adaptive_grad_clip
            )
            opt_model.step()
            schedulers[0].step()

            # Imagine from posterior states.
            post_state = (
                model_out.get(("next", "state")).detach().reshape(-1, state_dim)
            )
            post_belief = (
                model_out.get(("next", "belief"))
                .detach()
                .reshape(-1, cfg.networks.rnn_hidden_dim)
            )
            actor_input = TensorDict(
                {"state": post_state, "belief": post_belief},
                [post_state.shape[0]],
            )
            a_td, fake_data = actor_loss(actor_input)
            opt_actor.zero_grad(set_to_none=True)
            a_td["loss_actor"].backward()
            adaptive_grad_clip_(
                actor_model.parameters(), cfg.optimization.adaptive_grad_clip
            )
            opt_actor.step()
            schedulers[1].step()

            v_td, _ = value_loss(fake_data.detach())
            opt_value.zero_grad(set_to_none=True)
            v_td["loss_value"].backward()
            adaptive_grad_clip_(
                value_loss.parameters(), cfg.optimization.adaptive_grad_clip
            )
            opt_value.step()
            schedulers[2].step()
            value_target_updater.step()

            batch_losses[update_index] = torch.stack(
                (
                    model_kl.detach().reshape(()),
                    m_td["loss_model_reco"].detach().reshape(()),
                    m_td["loss_model_reward"].detach().reshape(()),
                    a_td["loss_actor"].detach().reshape(()),
                )
            )

        if record_loss_history:
            batch_losses = batch_losses.cpu()
            loss_history.append(batch_losses)

        if env_step >= next_eval:
            latest_losses = batch_losses[-1].cpu()
            r = eval_episode_reward(
                eval_env,
                real_actor,
                cfg.logger.eval_episodes,
                cfg.env.max_episode_steps,
            )
            history_steps.append(env_step)
            history_eval.append(r)
            torchrl_logger.info(
                "[env_step=%5d] eval_reward=%+.2f kl=%.3f reco=%.3f reward=%.3f actor=%.3f",
                env_step,
                r.item(),
                latest_losses[0].item(),
                latest_losses[1].item(),
                latest_losses[2].item(),
                latest_losses[3].item(),
            )
            next_eval = env_step + cfg.logger.eval_every

    if cfg.logger.output_plot and _has_matplotlib:
        import matplotlib.pyplot as plt  # noqa: PLC0415  (optional dep)

        eval_steps = history_steps
        eval_rewards = torch.stack(history_eval).cpu().numpy() if history_eval else []
        loss_curves = torch.cat(loss_history).numpy() if loss_history else None
        kl_vals = loss_curves[:, 0] if loss_curves is not None else []
        reco_vals = loss_curves[:, 1] if loss_curves is not None else []
        reward_vals = loss_curves[:, 2] if loss_curves is not None else []

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        axes[0].plot(eval_steps, eval_rewards, marker="o")
        axes[0].set_title(f"{cfg.env.name} eval reward (real env)")
        axes[0].set_xlabel("env_step")
        axes[0].set_ylabel("avg episode return")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(reco_vals, label="reco", alpha=0.8)
        axes[1].plot(reward_vals, label="reward", alpha=0.8)
        axes[1].plot(kl_vals, label="kl", alpha=0.8)
        axes[1].set_title("World-model losses (update step)")
        axes[1].set_xlabel("update step")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        fig.suptitle(
            f"DreamerV3 on {cfg.env.name} - {cfg.collector.total_frames} env steps, "
            f"{updates_per_batch} updates/batch"
        )
        fig.tight_layout()
        fig.savefig(cfg.logger.output_plot, dpi=120)
        torchrl_logger.info("Saved plot to %s", cfg.logger.output_plot)
    elif cfg.logger.output_plot:
        torchrl_logger.warning(
            "matplotlib is not installed; skipping plot %s", cfg.logger.output_plot
        )

    if cfg.logger.metrics_json:
        metrics_path = Path(cfg.logger.metrics_json)
        metrics_path.parent.mkdir(parents=True, exist_ok=True)
        metrics_path.write_text(
            json.dumps(
                {
                    "backend": cfg.env.backend,
                    "environment": cfg.env.name,
                    "task": cfg.env.task,
                    "seed": cfg.env.seed,
                    "environment_steps": history_steps,
                    "evaluation_returns": [value.item() for value in history_eval],
                },
                indent=2,
            )
            + "\n"
        )
        torchrl_logger.info("Saved evaluation metrics to %s", metrics_path)


if __name__ == "__main__":
    main()
