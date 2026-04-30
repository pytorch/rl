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

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, Unbounded
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import StepCounter, TransformedEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import SafeSequential, WorldModelWrapper
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules.models.model_based import DreamerActor
from torchrl.modules.models.model_based_v3 import (
    RSSMPosteriorV3,
    RSSMPriorV3,
    RSSMRolloutV3,
)
from torchrl.modules.models.models import MLP
from torchrl.objectives import (
    DreamerV3ActorLoss,
    DreamerV3ModelLoss,
    DreamerV3ValueLoss,
)
from torchrl.objectives.utils import ValueEstimators

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None


def make_env(env_name: str, seed: int = 0):
    env = GymEnv(env_name, device="cpu")
    env = TransformedEnv(env, StepCounter())
    env.set_seed(seed)
    return env


def build_world_model(*, cfg: DictConfig, obs_dim: int, action_dim: int):
    """MLP encoder + RSSMRolloutV3 + MLP decoder + reward head.

    Returns a TensorDictSequential whose forward consumes a trajectory batch
    and writes every key DreamerV3ModelLoss expects.
    """
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes

    encoder = TensorDictModule(
        MLP(
            in_features=obs_dim,
            out_features=cfg.networks.obs_embed_dim,
            depth=cfg.networks.depth,
            num_cells=cfg.networks.hidden_dim,
        ),
        in_keys=[("next", "observation")],
        out_keys=[("next", "encoded_latents")],
    )

    prior_net = RSSMPriorV3(
        action_shape=torch.Size([action_dim]),
        hidden_dim=cfg.networks.rnn_hidden_dim,
        rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
        num_categoricals=cfg.networks.num_categoricals,
        num_classes=cfg.networks.num_classes,
        action_dim=action_dim,
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
        hidden_dim=cfg.networks.rnn_hidden_dim,
        num_categoricals=cfg.networks.num_categoricals,
        num_classes=cfg.networks.num_classes,
        rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
        obs_embed_dim=cfg.networks.obs_embed_dim,
    )
    rssm_posterior = TensorDictModule(
        posterior_net,
        in_keys=[("next", "belief"), ("next", "encoded_latents")],
        out_keys=[("next", "posterior_logits"), ("next", "state")],
    )

    rollout = RSSMRolloutV3(rssm_prior, rssm_posterior)

    decoder = TensorDictModule(
        MLP(
            in_features=state_dim,
            out_features=obs_dim,
            depth=cfg.networks.depth,
            num_cells=cfg.networks.hidden_dim,
        ),
        in_keys=[("next", "state")],
        out_keys=[("next", "reco_pixels")],
    )

    reward_head = TensorDictModule(
        MLP(
            in_features=state_dim + cfg.networks.rnn_hidden_dim,
            out_features=cfg.networks.num_reward_bins,
            depth=cfg.networks.depth,
            num_cells=cfg.networks.hidden_dim,
        ),
        in_keys=[("next", "state"), ("next", "belief")],
        out_keys=[("next", "reward")],
    )

    return TensorDictSequential(encoder, rollout, decoder, reward_head)


def build_actor(*, cfg: DictConfig, action_dim: int):
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    actor_mlp = DreamerActor(
        out_features=action_dim,
        depth=cfg.networks.depth,
        num_cells=cfg.networks.hidden_dim,
    )
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


def build_value(*, cfg: DictConfig):
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes
    value_model = TensorDictModule(
        MLP(
            in_features=state_dim + cfg.networks.rnn_hidden_dim,
            out_features=1,
            depth=cfg.networks.depth,
            num_cells=cfg.networks.hidden_dim,
        ),
        in_keys=["state", "belief"],
        out_keys=["state_value"],
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


def build_mb_env(*, cfg: DictConfig, real_env, action_dim: int):
    """Imagination env: DreamerEnv wrapping an independent V3 prior + reward head."""
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
    rssm_prior = RSSMPriorV3(
        action_spec=real_env.action_spec,
        hidden_dim=cfg.networks.rnn_hidden_dim,
        rnn_hidden_dim=cfg.networks.rnn_hidden_dim,
        num_categoricals=cfg.networks.num_categoricals,
        num_classes=cfg.networks.num_classes,
        action_dim=action_dim,
    )
    transition_model = SafeSequential(
        TensorDictModule(
            rssm_prior,
            in_keys=["state", "belief", "action"],
            out_keys=["_", "state", "belief"],
        )
    )
    reward_model = TensorDictModule(
        MLP(
            in_features=state_dim + cfg.networks.rnn_hidden_dim,
            out_features=1,
            depth=cfg.networks.depth,
            num_cells=cfg.networks.hidden_dim,
        ),
        in_keys=["state", "belief"],
        out_keys=["reward"],
    )
    mb_env = DreamerEnv(
        world_model=WorldModelWrapper(transition_model, reward_model),
        prior_shape=torch.Size([state_dim]),
        belief_shape=torch.Size([cfg.networks.rnn_hidden_dim]),
    )
    mb_env.set_specs_from_env(primer_env)
    with torch.no_grad():
        mb_env.rollout(3)
    return mb_env


@torch.no_grad()
def eval_episode_reward(env, actor, num_episodes: int) -> torch.Tensor:
    totals = []
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(num_episodes):
            td = env.rollout(max_steps=200, policy=actor, break_when_any_done=True)
            totals.append(td.get(("next", "reward")).sum())
    return torch.stack(totals).mean()


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.env.seed)

    real_env = make_env(cfg.env.name, cfg.env.seed)
    obs_dim = real_env.observation_spec["observation"].shape[0]
    action_dim = real_env.action_spec.shape[0]
    state_dim = cfg.networks.num_categoricals * cfg.networks.num_classes

    world_model = build_world_model(cfg=cfg, obs_dim=obs_dim, action_dim=action_dim)
    actor_model = build_actor(cfg=cfg, action_dim=action_dim)
    value_model = build_value(cfg=cfg)
    mb_env = build_mb_env(
        cfg=cfg,
        real_env=make_env(cfg.env.name, cfg.env.seed + 1),
        action_dim=action_dim,
    )

    model_loss = DreamerV3ModelLoss(
        world_model,
        num_reward_bins=cfg.networks.num_reward_bins,
        free_bits=cfg.optimization.free_bits,
        kl_alpha=cfg.optimization.kl_alpha,
        global_average=True,  # state-based obs, not (C, H, W) pixels
    )
    model_loss.set_keys(pixels="observation")
    actor_loss = DreamerV3ActorLoss(
        actor_model,
        value_model,
        mb_env,
        imagination_horizon=cfg.optimization.imagination_horizon,
        use_reinforce=cfg.optimization.use_reinforce,
    )
    actor_loss.make_value_estimator(
        ValueEstimators.TDLambda,
        gamma=cfg.optimization.gamma,
        lmbda=cfg.optimization.lmbda,
    )
    value_loss = DreamerV3ValueLoss(
        value_model, value_loss="symlog_mse", actor_loss=actor_loss
    )

    opt_model = torch.optim.Adam(model_loss.parameters(), lr=cfg.optimization.lr)
    opt_actor = torch.optim.Adam(actor_loss.parameters(), lr=cfg.optimization.lr)
    opt_value = torch.optim.Adam(value_loss.parameters(), lr=cfg.optimization.lr)

    explore_env = TransformedEnv(
        make_env(cfg.env.name, cfg.env.seed + 2),
        TensorDictPrimer(
            random=False,
            default_value=0,
            state=Unbounded(state_dim),
            belief=Unbounded(cfg.networks.rnn_hidden_dim),
        ),
    )

    collector = SyncDataCollector(
        explore_env,
        actor_model,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=cfg.env.device,
        exploration_type=ExplorationType.RANDOM
        if cfg.collector.exploration == "random"
        else ExplorationType.MODE,
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
    loss_hist: dict[str, list[torch.Tensor]] = {
        "kl": [],
        "reco": [],
        "reward": [],
        "actor": [],
        "value": [],
    }
    next_eval = 0

    eval_env = TransformedEnv(
        make_env(cfg.env.name, cfg.env.seed + 100),
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

    for data in collector:
        rb.extend(data.reshape(-1))
        env_step += data.numel()

        if len(rb) < warmup:
            continue

        for _ in range(cfg.optimization.updates_per_batch):
            sample = rb.sample().reshape(
                cfg.replay_buffer.batch_size, cfg.replay_buffer.seq_len
            )

            sample.set(
                "state",
                torch.zeros(
                    cfg.replay_buffer.batch_size,
                    cfg.replay_buffer.seq_len,
                    state_dim,
                ),
            )
            sample.set(
                "belief",
                torch.zeros(
                    cfg.replay_buffer.batch_size,
                    cfg.replay_buffer.seq_len,
                    cfg.networks.rnn_hidden_dim,
                ),
            )

            m_td, model_out = model_loss(sample)
            total_m = (
                m_td["loss_model_kl"]
                + m_td["loss_model_reco"]
                + m_td["loss_model_reward"]
            ).squeeze()
            opt_model.zero_grad(set_to_none=True)
            total_m.backward()
            torch.nn.utils.clip_grad_norm_(
                model_loss.parameters(), cfg.optimization.grad_clip
            )
            opt_model.step()

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
            torch.nn.utils.clip_grad_norm_(
                actor_loss.parameters(), cfg.optimization.grad_clip
            )
            opt_actor.step()

            v_td, _ = value_loss(fake_data.detach())
            opt_value.zero_grad(set_to_none=True)
            v_td["loss_value"].backward()
            torch.nn.utils.clip_grad_norm_(
                value_loss.parameters(), cfg.optimization.grad_clip
            )
            opt_value.step()

            loss_hist["kl"].append(m_td["loss_model_kl"].detach())
            loss_hist["reco"].append(m_td["loss_model_reco"].detach())
            loss_hist["reward"].append(m_td["loss_model_reward"].detach())
            loss_hist["actor"].append(a_td["loss_actor"].detach())
            loss_hist["value"].append(v_td["loss_value"].detach())

        if env_step >= next_eval:
            r = eval_episode_reward(eval_env, actor_model, cfg.logger.eval_episodes)
            history_steps.append(env_step)
            history_eval.append(r)
            torchrl_logger.info(
                "[env_step=%5d] eval_reward=%+.2f kl=%.3f reco=%.3f reward=%.3f actor=%.3f",
                env_step,
                r.item(),
                loss_hist["kl"][-1].item(),
                loss_hist["reco"][-1].item(),
                loss_hist["reward"][-1].item(),
                loss_hist["actor"][-1].item(),
            )
            next_eval = env_step + cfg.logger.eval_every

    if cfg.logger.output_plot and _has_matplotlib:
        import matplotlib.pyplot as plt  # noqa: PLC0415  (optional dep)

        eval_steps = history_steps
        eval_rewards = torch.stack(history_eval).cpu().numpy() if history_eval else []
        kl_vals = torch.stack(loss_hist["kl"]).cpu().numpy() if loss_hist["kl"] else []
        reco_vals = (
            torch.stack(loss_hist["reco"]).cpu().numpy() if loss_hist["reco"] else []
        )
        reward_vals = (
            torch.stack(loss_hist["reward"]).cpu().numpy()
            if loss_hist["reward"]
            else []
        )

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
            f"{cfg.optimization.updates_per_batch} updates/batch"
        )
        fig.tight_layout()
        fig.savefig(cfg.logger.output_plot, dpi=120)
        torchrl_logger.info("Saved plot to %s", cfg.logger.output_plot)
    elif cfg.logger.output_plot:
        torchrl_logger.warning(
            "matplotlib is not installed; skipping plot %s", cfg.logger.output_plot
        )


if __name__ == "__main__":
    main()
