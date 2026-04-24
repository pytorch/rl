# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""DreamerV3 on Pendulum-v1 — minimal end-to-end training script.

State-based (not pixel-based) to keep the script compact — the 3-D obs is
treated as a flat feature vector with ``global_average=True`` in the model
loss. The wiring is still real:

- collector → replay buffer of sequences
- world model = MLP encoder + RSSMPriorV3 + RSSMPosteriorV3 + MLP decoder + reward head
- RSSM unrolled over each sequence via ``RSSMRolloutV3``
- actor trained via REINFORCE in imagination (``DreamerV3ActorLoss``)
- value trained on the same imagined rollout (``DreamerV3ValueLoss``)
- periodic eval rollouts in the real env, episode reward logged

Plots ``dreamer_v3_pendulum.png`` with two curves: (a) average eval reward,
(b) world-model KL / reconstruction / reward losses.

Usage::

    python sota-implementations/dreamer_v3/pendulum_train.py \\
        --env-steps 5000 --eval-every 500
"""
from __future__ import annotations

import argparse
import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt
import torch
from tensordict import TensorDict
from tensordict.nn import (
    InteractionType,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictSequential,
    TensorDictModule,
    TensorDictSequential,
)
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, ReplayBuffer, Unbounded
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import ObservationNorm, StepCounter, TransformedEnv
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


def make_env(seed: int = 0):
    env = GymEnv("Pendulum-v1", device="cpu")
    env = TransformedEnv(env, StepCounter())
    env.set_seed(seed)
    return env


def build_world_model(
    *,
    obs_dim: int,
    action_dim: int,
    state_dim: int,
    rnn_hidden_dim: int,
    num_cats: int,
    num_classes: int,
    obs_embed_dim: int,
    num_reward_bins: int,
):
    """MLP encoder + RSSMRolloutV3 + MLP decoder + reward head.

    Produces a single TensorDictModule whose ``forward`` consumes a trajectory
    batch and writes every key ``DreamerV3ModelLoss`` expects.
    """

    # Per-step encoder: obs -> embedding (used on "next" obs)
    encoder = TensorDictModule(
        MLP(
            in_features=obs_dim,
            out_features=obs_embed_dim,
            depth=2,
            num_cells=64,
        ),
        in_keys=[("next", "observation")],
        out_keys=[("next", "encoded_latents")],
    )

    # RSSM prior (per-step)
    prior_net = RSSMPriorV3(
        action_spec=argparse.Namespace(shape=torch.Size([action_dim])),
        hidden_dim=rnn_hidden_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        num_categoricals=num_cats,
        num_classes=num_classes,
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

    # RSSM posterior (per-step)
    posterior_net = RSSMPosteriorV3(
        hidden_dim=rnn_hidden_dim,
        num_categoricals=num_cats,
        num_classes=num_classes,
        rnn_hidden_dim=rnn_hidden_dim,
        obs_embed_dim=obs_embed_dim,
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
            depth=2,
            num_cells=64,
        ),
        in_keys=[("next", "state")],
        out_keys=[("next", "reco_pixels")],
    )

    reward_head = TensorDictModule(
        MLP(
            in_features=state_dim + rnn_hidden_dim,
            out_features=num_reward_bins,
            depth=2,
            num_cells=64,
        ),
        in_keys=[("next", "state"), ("next", "belief")],
        out_keys=[("next", "reward")],
    )

    return TensorDictSequential(encoder, rollout, decoder, reward_head)


def build_actor(*, state_dim: int, rnn_hidden_dim: int, action_dim: int):
    actor_mlp = DreamerActor(
        out_features=action_dim, depth=2, num_cells=64
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
                    "belief": torch.randn(1, 2, rnn_hidden_dim),
                },
                [1],
            )
        )
    return actor_model


def build_value(*, state_dim: int, rnn_hidden_dim: int):
    value_model = TensorDictModule(
        MLP(
            in_features=state_dim + rnn_hidden_dim,
            out_features=1,
            depth=2,
            num_cells=64,
        ),
        in_keys=["state", "belief"],
        out_keys=["state_value"],
    )
    with torch.no_grad():
        value_model(
            TensorDict(
                {
                    "state": torch.randn(1, 2, state_dim),
                    "belief": torch.randn(1, 2, rnn_hidden_dim),
                },
                [1],
            )
        )
    return value_model


def build_mb_env(
    *,
    real_env,
    state_dim: int,
    rnn_hidden_dim: int,
    num_cats: int,
    num_classes: int,
    action_dim: int,
):
    """Imagination env: DreamerEnv wrapping an independent V3 prior + reward head."""
    primer_env = TransformedEnv(
        real_env,
        TensorDictPrimer(
            random=False,
            default_value=0,
            state=Unbounded(state_dim),
            belief=Unbounded(rnn_hidden_dim),
        ),
    )
    rssm_prior = RSSMPriorV3(
        action_spec=real_env.action_spec,
        hidden_dim=rnn_hidden_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        num_categoricals=num_cats,
        num_classes=num_classes,
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
            in_features=state_dim + rnn_hidden_dim,
            out_features=1,
            depth=2,
            num_cells=64,
        ),
        in_keys=["state", "belief"],
        out_keys=["reward"],
    )
    mb_env = DreamerEnv(
        world_model=WorldModelWrapper(transition_model, reward_model),
        prior_shape=torch.Size([state_dim]),
        belief_shape=torch.Size([rnn_hidden_dim]),
    )
    mb_env.set_specs_from_env(primer_env)
    with torch.no_grad():
        mb_env.rollout(3)
    return mb_env


@torch.no_grad()
def eval_episode_reward(env, actor, num_episodes: int = 3) -> float:
    totals = []
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(num_episodes):
            td = env.rollout(max_steps=200, policy=actor, break_when_any_done=True)
            totals.append(td.get(("next", "reward")).sum().item())
    return sum(totals) / len(totals)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-steps", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--frames-per-batch", type=int, default=200)
    parser.add_argument("--updates-per-batch", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument(
        "--output",
        type=str,
        default=os.path.join(
            os.path.dirname(__file__), "dreamer_v3_pendulum.png"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    real_env = make_env(args.seed)
    obs_dim = real_env.observation_spec["observation"].shape[0]
    action_dim = real_env.action_spec.shape[0]

    cfg = dict(
        obs_dim=obs_dim,
        action_dim=action_dim,
        state_dim=16,
        rnn_hidden_dim=32,
        num_cats=4,
        num_classes=4,
        obs_embed_dim=32,
        num_reward_bins=41,
    )

    # ---- build modules ----
    world_model = build_world_model(**cfg)
    actor_model = build_actor(
        state_dim=cfg["state_dim"],
        rnn_hidden_dim=cfg["rnn_hidden_dim"],
        action_dim=cfg["action_dim"],
    )
    value_model = build_value(
        state_dim=cfg["state_dim"], rnn_hidden_dim=cfg["rnn_hidden_dim"]
    )
    mb_env = build_mb_env(
        real_env=make_env(args.seed + 1),
        state_dim=cfg["state_dim"],
        rnn_hidden_dim=cfg["rnn_hidden_dim"],
        num_cats=cfg["num_cats"],
        num_classes=cfg["num_classes"],
        action_dim=cfg["action_dim"],
    )

    model_loss = DreamerV3ModelLoss(
        world_model,
        num_reward_bins=cfg["num_reward_bins"],
        global_average=True,  # obs is 3-D state, not (C,H,W) pixels
    )
    model_loss.set_keys(pixels="observation")
    actor_loss = DreamerV3ActorLoss(
        actor_model, value_model, mb_env,
        imagination_horizon=10,
        use_reinforce=True,
    )
    actor_loss.make_value_estimator(ValueEstimators.TDLambda, gamma=0.99, lmbda=0.95)
    value_loss = DreamerV3ValueLoss(value_model, value_loss="symlog_mse")
    value_loss.sync_gamma_with_actor_loss(actor_loss)

    opt_model = torch.optim.Adam(model_loss.parameters(), lr=args.lr)
    opt_actor = torch.optim.Adam(actor_loss.parameters(), lr=args.lr)
    opt_value = torch.optim.Adam(value_loss.parameters(), lr=args.lr)

    # ---- exploration policy: actor conditioned on state/belief.
    # Use a primed env so the collector hands (state, belief) to the actor. ----
    explore_env = TransformedEnv(
        make_env(args.seed + 2),
        TensorDictPrimer(
            random=False,
            default_value=0,
            state=Unbounded(cfg["state_dim"]),
            belief=Unbounded(cfg["rnn_hidden_dim"]),
        ),
    )

    collector = SyncDataCollector(
        explore_env,
        actor_model,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.env_steps,
        device="cpu",
        exploration_type=ExplorationType.RANDOM,
    )

    # Replay buffer of (seq_len)-length slices.
    rb = ReplayBuffer(
        storage=LazyTensorStorage(max_size=10_000),
        sampler=SliceSampler(slice_len=args.seq_len, traj_key=("collector", "traj_ids")),
        batch_size=args.batch_size * args.seq_len,
    )

    env_step = 0
    history = {"env_step": [], "eval_reward": []}
    loss_hist = {"kl": [], "reco": [], "reward": [], "actor": [], "value": []}
    next_eval = 0

    eval_env = make_env(args.seed + 100)

    for data in collector:
        rb.extend(data.reshape(-1))
        env_step += data.numel()

        if len(rb) < args.batch_size * args.seq_len * 2:
            continue

        for _ in range(args.updates_per_batch):
            sample = rb.sample().reshape(args.batch_size, args.seq_len)

            # Initial state/belief zero at start of each sequence.
            sample.set(
                "state",
                torch.zeros(args.batch_size, args.seq_len, cfg["state_dim"]),
            )
            sample.set(
                "belief",
                torch.zeros(args.batch_size, args.seq_len, cfg["rnn_hidden_dim"]),
            )

            # --- world model update ---
            m_td, model_out = model_loss(sample)
            total_m = (
                m_td["loss_model_kl"]
                + m_td["loss_model_reco"]
                + m_td["loss_model_reward"]
            ).squeeze()
            opt_model.zero_grad(set_to_none=True)
            total_m.backward()
            torch.nn.utils.clip_grad_norm_(model_loss.parameters(), 100.0)
            opt_model.step()

            # --- actor update (imagine from posterior states) ---
            # Use the posterior state/belief produced by the world model as
            # starting points for imagination.
            post_state = model_out.get(("next", "state")).detach().reshape(
                -1, cfg["state_dim"]
            )
            post_belief = model_out.get(("next", "belief")).detach().reshape(
                -1, cfg["rnn_hidden_dim"]
            )
            actor_input = TensorDict(
                {"state": post_state, "belief": post_belief}, [post_state.shape[0]]
            )
            a_td, fake_data = actor_loss(actor_input)
            opt_actor.zero_grad(set_to_none=True)
            a_td["loss_actor"].backward()
            torch.nn.utils.clip_grad_norm_(actor_loss.parameters(), 100.0)
            opt_actor.step()

            # --- value update ---
            v_td, _ = value_loss(fake_data.detach())
            opt_value.zero_grad(set_to_none=True)
            v_td["loss_value"].backward()
            torch.nn.utils.clip_grad_norm_(value_loss.parameters(), 100.0)
            opt_value.step()

            loss_hist["kl"].append(m_td["loss_model_kl"].item())
            loss_hist["reco"].append(m_td["loss_model_reco"].item())
            loss_hist["reward"].append(m_td["loss_model_reward"].item())
            loss_hist["actor"].append(a_td["loss_actor"].item())
            loss_hist["value"].append(v_td["loss_value"].item())

        if env_step >= next_eval:
            # The eval env isn't primed with state/belief; the actor needs them
            # zero-initialised per step, so prime the eval env too.
            primed_eval = TransformedEnv(
                eval_env,
                TensorDictPrimer(
                    random=False,
                    default_value=0,
                    state=Unbounded(cfg["state_dim"]),
                    belief=Unbounded(cfg["rnn_hidden_dim"]),
                ),
            )
            r = eval_episode_reward(primed_eval, actor_model, num_episodes=3)
            history["env_step"].append(env_step)
            history["eval_reward"].append(r)
            print(
                f"[env_step={env_step:5d}] eval_reward={r:+.2f} "
                f"kl={loss_hist['kl'][-1]:.3f} "
                f"reco={loss_hist['reco'][-1]:.3f} "
                f"reward={loss_hist['reward'][-1]:.3f} "
                f"actor={loss_hist['actor'][-1]:.3f}"
            )
            next_eval = env_step + args.eval_every

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(history["env_step"], history["eval_reward"], marker="o")
    axes[0].set_title("Pendulum eval reward (real env)")
    axes[0].set_xlabel("env_step")
    axes[0].set_ylabel("avg episode return")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(loss_hist["reco"], label="reco", alpha=0.8)
    axes[1].plot(loss_hist["reward"], label="reward", alpha=0.8)
    axes[1].plot(loss_hist["kl"], label="kl", alpha=0.8)
    axes[1].set_title("World-model losses (update step)")
    axes[1].set_xlabel("update step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"DreamerV3 on Pendulum-v1 — {args.env_steps} env steps, "
        f"{args.updates_per_batch} updates/batch"
    )
    fig.tight_layout()
    fig.savefig(args.output, dpi=120)
    print(f"\nSaved plot to {args.output}")


if __name__ == "__main__":
    main()
