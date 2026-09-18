# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Behavior cloning on CartPole from replay buffer batches sampled in DataLoader workers.

A scripted expert fills a memory-mapped replay buffer, a logits head is trained
with :class:`~torchrl.objectives.BCLoss` on batches that
``torch.utils.data.DataLoader`` workers sample from the buffer, and the greedy
actor built on that head is saved as a checkpoint that ``rlrender`` plays back::

    python examples/replay-buffers/dataloader_behavior_cloning.py \\
        --checkpoint cartpole_bc.pt --plot cartpole_bc.png
    rlrender --ckpt cartpole_bc.pt \\
        --policy examples/replay-buffers/dataloader_behavior_cloning.py:make_render_policy \\
        --env examples/replay-buffers/dataloader_behavior_cloning.py:make_render_env \\
        --from-pixels --max-steps 500 --format gif --out cartpole_bc.gif
"""
from __future__ import annotations

import argparse
import importlib.util

import torch
from tensordict.nn import TensorDictModule
from torch.utils.data import DataLoader

from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.data import LazyMemmapStorage, tensordict_collate, TensorDictReplayBuffer
from torchrl.envs import GymEnv, StepCounter, TransformedEnv
from torchrl.modules import MLP, QValueActor
from torchrl.objectives import BCLoss
from torchrl.render import save_render_checkpoint

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None


class CartPoleExpert(torch.nn.Module):
    """Pushes the cart toward the side the pole is falling to."""

    def forward(self, observation):
        return (observation[..., 2] + 0.5 * observation[..., 3] > 0).long()


def make_env(device="cpu", from_pixels=False):
    env = GymEnv(
        "CartPole-v1",
        device=device,
        from_pixels=from_pixels,
        pixels_only=False,
        categorical_action_encoding=True,
    )
    return TransformedEnv(env, StepCounter())


def make_logits(device="cpu"):
    return MLP(in_features=4, out_features=2, num_cells=[64, 64], device=device)


def make_actor(logits, action_spec):
    return QValueActor(
        logits, in_keys=["observation"], spec=action_spec, action_space="categorical"
    )


def make_render_env(spec):
    return make_env(device=spec.device, from_pixels=spec.from_pixels)


def make_render_policy(spec):
    return make_actor(make_logits(spec.device), make_env().action_spec)


def evaluate(env, actor, episodes):
    with torch.no_grad():
        return torch.stack(
            [env.rollout(500, actor)["next", "reward"].sum() for _ in range(episodes)]
        )


def plot(losses, returns, num_workers, path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(losses.numpy())
    ax.set_xlabel("update")
    ax.set_ylabel("BC loss")
    ax.set_title(
        f"CartPole BC, {num_workers} DataLoader workers, "
        f"eval return {returns.mean():.0f}"
    )
    fig.tight_layout()
    fig.savefig(path)


def main(args):
    torch.manual_seed(args.seed)
    env = make_env()
    expert = TensorDictModule(
        CartPoleExpert(), in_keys=["observation"], out_keys=["action"]
    )
    rb = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(args.frames), batch_size=args.batch_size
    )
    collector = Collector(
        env,
        expert,
        frames_per_batch=1000,
        total_frames=args.frames,
        auto_register_policy_transforms=True,
    )
    for data in collector:
        rb.extend(data)
    collector.shutdown()

    logits = make_logits()
    loss_module = BCLoss(
        TensorDictModule(logits, in_keys=["observation"], out_keys=["action"])
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=args.lr)
    loader = DataLoader(
        rb.as_dataset(num_batches=args.updates),
        batch_size=None,
        num_workers=args.num_workers,
        collate_fn=tensordict_collate,
    )
    losses = []
    for update, batch in enumerate(loader, 1):
        loss = loss_module(batch)["loss_bc"]
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.detach())
        if update % 100 == 0:
            torchrl_logger.info(f"update {update}: loss {loss.item():.4f}")
    losses = torch.stack(losses)

    actor = make_actor(logits, env.action_spec)
    returns = evaluate(env, actor, args.eval_episodes)
    torchrl_logger.info(f"eval return {returns.mean():.1f} +/- {returns.std():.1f}")
    if args.plot and _has_matplotlib:
        plot(losses, returns, args.num_workers, args.plot)
    save_render_checkpoint(
        args.checkpoint,
        actor,
        env_metadata={"env_name": "CartPole-v1"},
        frames=args.frames,
        metrics={"eval_return": returns.mean().item()},
        format="archive",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames", type=int, default=20_000)
    parser.add_argument("--updates", type=int, default=500)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--plot", default=None)
    main(parser.parse_args())
