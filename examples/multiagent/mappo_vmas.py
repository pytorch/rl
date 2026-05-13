# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Minimal MAPPO / IPPO recipe on VMAS using the new
:class:`~torchrl.objectives.multiagent.MAPPOLoss` /
:class:`~torchrl.objectives.multiagent.IPPOLoss` classes.

For the full, hydra-configured, wandb-logged version see
``sota-implementations/multiagent/mappo_ippo.py``. This file is intentionally
short: it's there to show that the new loss classes collapse the boilerplate
that previously required ``ClipPPOLoss`` + manual ``set_keys(done=...,
terminated=...)`` + manual ``make_value_estimator(GAE, ...)`` into a single
construction call.

Usage::

    python examples/multiagent/mappo_vmas.py --algo mappo --frames 200_000
    python examples/multiagent/mappo_vmas.py --algo ippo --frames 200_000

The two should reach similar reward on the easy navigation scenario; MAPPO
typically pulls ahead on harder coordination tasks (Yu et al. 2022).
"""
from __future__ import annotations

import argparse
import time

import torch
from tensordict.nn import TensorDictModule
from tensordict.nn.distributions import NormalParamExtractor
from torch import nn

from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs import RewardSum, TransformedEnv
from torchrl.modules import MultiAgentMLP, ProbabilisticActor, TanhNormal, ValueNorm
from torchrl.objectives import IPPOLoss, MAPPOLoss


def make_actor(env, *, share_params: bool = True) -> ProbabilisticActor:
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    action_dim = env.action_spec.shape[-1]
    backbone = nn.Sequential(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=2 * action_dim,
            n_agents=env.n_agents,
            centralized=False,
            share_params=share_params,
            depth=2,
            num_cells=256,
            activation_class=nn.Tanh,
        ),
        NormalParamExtractor(),
    )
    module = TensorDictModule(
        backbone,
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "loc"), ("agents", "scale")],
    )
    return ProbabilisticActor(
        module=module,
        in_keys=[("agents", "loc"), ("agents", "scale")],
        out_keys=[env.action_key],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.full_action_spec_unbatched[("agents", "action")].space.low,
            "high": env.full_action_spec_unbatched[("agents", "action")].space.high,
        },
        return_log_prob=True,
    )


def make_critic(
    env, *, centralized: bool, share_params: bool = True
) -> TensorDictModule:
    obs_dim = env.observation_spec["agents", "observation"].shape[-1]
    return TensorDictModule(
        MultiAgentMLP(
            n_agent_inputs=obs_dim,
            n_agent_outputs=1,
            n_agents=env.n_agents,
            centralized=centralized,
            share_params=share_params,
            depth=2,
            num_cells=256,
            activation_class=nn.Tanh,
        ),
        in_keys=[("agents", "observation")],
        out_keys=[("agents", "state_value")],
    )


def main(args: argparse.Namespace) -> None:
    try:
        from torchrl.envs.libs.vmas import VmasEnv
    except ImportError as exc:
        raise SystemExit(
            "This example requires VMAS. Install it with `pip install vmas`."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(args.seed)

    n_envs = max(1, args.frames_per_batch // args.max_steps)
    env = TransformedEnv(
        VmasEnv(
            scenario=args.scenario,
            num_envs=n_envs,
            continuous_actions=True,
            max_steps=args.max_steps,
            device=device,
            seed=args.seed,
        ),
        RewardSum(
            in_keys=[("next", "agents", "reward")],
            out_keys=[("agents", "episode_reward")],
        )
        if False
        else RewardSum(in_keys=["reward"], out_keys=["episode_reward"]),
    )

    actor = make_actor(env)
    centralised = args.algo == "mappo"
    critic = make_critic(env, centralized=centralised)

    LossCls = MAPPOLoss if args.algo == "mappo" else IPPOLoss
    value_norm = ValueNorm(shape=1, device=device) if args.algo == "mappo" else None
    loss_module = LossCls(
        actor_network=actor,
        critic_network=critic,
        value_norm=value_norm,
        clip_epsilon=0.2,
        entropy_coeff=0.01,
    )
    loss_module.set_keys(
        value=("agents", "state_value"),
        action=env.action_key,
        reward=env.reward_key,
    )

    collector = SyncDataCollector(
        env,
        actor,
        device=device,
        storing_device=device,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.frames,
    )

    replay_buffer = TensorDictReplayBuffer(
        storage=LazyTensorStorage(args.frames_per_batch, device=device),
        sampler=SamplerWithoutReplacement(),
        batch_size=args.minibatch_size,
    )

    optim = torch.optim.Adam(loss_module.parameters(), lr=args.lr)

    total_frames = 0
    start = time.time()
    for it, td in enumerate(collector):
        with torch.no_grad():
            loss_module.value_estimator(
                td,
                params=loss_module.critic_network_params,
                target_params=loss_module.target_critic_network_params,
            )
        replay_buffer.extend(td.reshape(-1))
        total_frames += td.numel()

        for _ in range(args.epochs):
            for _ in range(args.frames_per_batch // args.minibatch_size):
                subdata = replay_buffer.sample()
                losses = loss_module(subdata)
                loss = (
                    losses["loss_objective"]
                    + losses["loss_critic"]
                    + losses["loss_entropy"]
                )
                optim.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                optim.step()

        collector.update_policy_weights_()
        ep_reward = td.get(("next", "episode_reward")).mean().item()
        print(
            f"[{args.algo}] iter={it:03d} frames={total_frames:>7d} "
            f"reward={ep_reward:+.3f} elapsed={time.time() - start:5.1f}s"
        )

    collector.shutdown()
    if not env.is_closed:
        env.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--algo", choices=("mappo", "ippo"), default="mappo")
    p.add_argument("--scenario", default="navigation")
    p.add_argument("--frames", type=int, default=200_000)
    p.add_argument("--frames_per_batch", type=int, default=6_000)
    p.add_argument("--minibatch_size", type=int, default=400)
    p.add_argument("--max_steps", type=int, default=100)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
