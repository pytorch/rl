# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Offline-to-online SAC fine-tuning.

Warm-starts SAC on an offline dataset (D4RL/Minari) and fine-tunes it online via
:class:`~torchrl.trainers.algorithms.OfflineToOnlineTrainer`, sampling a mixed
offline/online batch whose offline fraction is annealed to zero over
``--anneal-frames`` collected frames.

Example::

    python train.py --dataset d4rl:halfcheetah-medium-v2 --env HalfCheetah-v4
    python train.py --dataset minari:mujoco/halfcheetah/expert-v0 --total-frames 200000

Requires the dataset backend (``pip install d4rl`` or ``pip install minari``) and
the matching MuJoCo environment.
"""

from __future__ import annotations

import argparse

import torch
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn

from torchrl.collectors import Collector
from torchrl.data import OfflineToOnlineReplayBuffer
from torchrl.data.datasets.utils import load_dataset
from torchrl.envs import DoubleToFloat, GymEnv, TransformedEnv
from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import SACLoss, SoftUpdate
from torchrl.trainers.algorithms.offline_to_online import OfflineToOnlineTrainer


def make_sac_modules(env, num_cells, device):
    obs_dim = env.observation_spec["observation"].shape[-1]
    action_dim = env.action_spec.shape[-1]

    actor_net = nn.Sequential(
        MLP(
            in_features=obs_dim,
            out_features=2 * action_dim,
            num_cells=num_cells,
            device=device,
        ),
        NormalParamExtractor(),
    )
    actor = ProbabilisticActor(
        module=TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        ),
        in_keys=["loc", "scale"],
        spec=env.action_spec,
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec.space.low,
            "high": env.action_spec.space.high,
        },
        return_log_prob=True,
    )
    qvalue = ValueOperator(
        MLP(
            in_features=obs_dim + action_dim,
            out_features=1,
            num_cells=num_cells,
            device=device,
        ),
        in_keys=["observation", "action"],
        out_keys=["state_action_value"],
    )
    return actor, qvalue


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", default="HalfCheetah-v4", help="online gym env id")
    parser.add_argument(
        "--dataset",
        default="d4rl:halfcheetah-medium-v2",
        help="offline dataset id ('d4rl:<id>' or 'minari:<id>')",
    )
    parser.add_argument("--total-frames", type=int, default=1_000_000)
    parser.add_argument("--frames-per-batch", type=int, default=1000)
    parser.add_argument(
        "--anneal-frames",
        type=int,
        default=None,
        help="frames over which the offline fraction decays to 0 (default: half "
        "of --total-frames)",
    )
    parser.add_argument("--offline-fraction", type=float, default=0.5)
    parser.add_argument("--online-capacity", type=int, default=1_000_000)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--utd", type=int, default=64, help="optim steps per batch")
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-cells", type=int, nargs="+", default=[256, 256])
    parser.add_argument("--tau", type=float, default=0.001)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    # Online environment.
    env = TransformedEnv(GymEnv(args.env, device=device), DoubleToFloat())
    env.set_seed(args.seed)

    # SAC agent.
    actor, qvalue = make_sac_modules(env, args.num_cells, device)
    loss = SACLoss(actor_network=actor, qvalue_network=qvalue)
    loss.make_value_estimator(gamma=0.99)
    target_net_updater = SoftUpdate(loss, tau=args.tau)
    optimizer = torch.optim.Adam(loss.parameters(), lr=args.lr)

    # Immutable offline dataset (DoubleToFloat to match the online float32 stream)
    # paired with a growing online buffer.
    offline = load_dataset(args.dataset)
    offline.append_transform(DoubleToFloat())
    replay_buffer = OfflineToOnlineReplayBuffer(
        offline_dataset=offline,
        online_capacity=args.online_capacity,
        offline_fraction=args.offline_fraction,
        batch_size=args.batch_size,
    )

    collector = Collector(
        env,
        actor,
        frames_per_batch=args.frames_per_batch,
        total_frames=args.total_frames,
        init_random_frames=0,  # the offline dataset already warm-starts learning
        device=device,
    )

    anneal_frames = (
        args.anneal_frames if args.anneal_frames is not None else args.total_frames // 2
    )
    trainer = OfflineToOnlineTrainer(
        collector=collector,
        total_frames=args.total_frames,
        frame_skip=1,
        optim_steps_per_batch=args.utd,
        loss_module=loss,
        replay_buffer=replay_buffer,
        anneal_frames=anneal_frames,
        batch_size=args.batch_size,
        optimizer=optimizer,
        target_net_updater=target_net_updater,
        clip_grad_norm=False,
    )
    trainer.train()


if __name__ == "__main__":
    main()
