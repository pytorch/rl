# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Diffusion Behavioural Cloning on Pendulum-v1.

This script demonstrates end-to-end training of a :class:`~torchrl.modules.DiffusionActor`
using :class:`~torchrl.objectives.DiffusionBCLoss` on expert demonstrations
collected from a pre-trained policy in the Pendulum-v1 environment.

Steps
-----
1. Collect expert demonstrations with a hand-crafted sinusoidal policy.
2. Train the DiffusionActor on those demonstrations with the BC loss.
3. Evaluate the trained actor in the environment and report mean return.

Usage
-----
    python examples/diffusion_bc_pendulum.py

Dependencies
------------
    pip install torchrl gymnasium
"""
from __future__ import annotations

import torch
from tensordict import TensorDict
from torch.utils.data import DataLoader, TensorDataset

from torchrl.envs import GymEnv
from torchrl.modules import DiffusionActor
from torchrl.objectives import DiffusionBCLoss

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
ENV_NAME = "Pendulum-v1"
OBS_DIM = 3
ACTION_DIM = 1
NUM_DEMO_EPISODES = 50
EPISODE_LEN = 200
NUM_STEPS = 20  # DDPM denoising steps (fewer = faster training)
BATCH_SIZE = 256
LR = 1e-3
TRAIN_EPOCHS = 100
EVAL_EPISODES = 10
SEED = 42


# ---------------------------------------------------------------------------
# Expert policy: sinusoidal torque (good enough for demonstrations)
# ---------------------------------------------------------------------------
def expert_action(obs: torch.Tensor, t: int) -> torch.Tensor:
    """Simple sinusoidal expert for Pendulum."""
    return torch.sin(torch.tensor(t * 0.1)).unsqueeze(-1).expand(*obs.shape[:-1], 1)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------
def collect_demonstrations(env):
    observations, actions = [], []
    for _ in range(NUM_DEMO_EPISODES):
        td = env.reset()
        for t in range(EPISODE_LEN):
            obs = td["observation"]
            act = expert_action(obs, t).clamp(-2.0, 2.0)
            td["action"] = act
            td = env.step(td)
            observations.append(obs)
            actions.append(act)
            td = td["next"]
    return (
        torch.stack(observations),  # (N, obs_dim)
        torch.stack(actions),  # (N, action_dim)
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
def evaluate(actor, env, n_episodes=EVAL_EPISODES):
    returns = []
    for _ in range(n_episodes):
        td = env.reset()
        ep_return = 0.0
        for _ in range(EPISODE_LEN):
            td = actor(td)
            td = env.step(td)
            ep_return += td[("next", "reward")].item()
            td = td["next"]
        returns.append(ep_return)
    return sum(returns) / len(returns)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    torch.manual_seed(SEED)

    env = GymEnv(ENV_NAME)

    print("Collecting expert demonstrations …")
    obs_data, act_data = collect_demonstrations(env)
    print(f"  collected {len(obs_data)} transitions")

    dataset = TensorDataset(obs_data, act_data)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    actor = DiffusionActor(action_dim=ACTION_DIM, obs_dim=OBS_DIM, num_steps=NUM_STEPS)
    loss_fn = DiffusionBCLoss(actor)
    optimizer = torch.optim.Adam(actor.parameters(), lr=LR)

    print("Training …")
    for epoch in range(1, TRAIN_EPOCHS + 1):
        epoch_loss = 0.0
        for obs_batch, act_batch in loader:
            td = TensorDict(
                {"observation": obs_batch, "action": act_batch},
                batch_size=[obs_batch.shape[0]],
            )
            optimizer.zero_grad()
            loss = loss_fn(td)["loss_diffusion_bc"]
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        if epoch % 10 == 0:
            avg = epoch_loss / len(loader)
            print(f"  epoch {epoch:3d}/{TRAIN_EPOCHS}  loss={avg:.4f}")

    print("Evaluating …")
    mean_return = evaluate(actor, env)
    print(f"  mean return over {EVAL_EPISODES} episodes: {mean_return:.2f}")

    env.close()


if __name__ == "__main__":
    main()
