# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch.nn as nn
import torch.optim
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.replay_buffers import SamplerWithoutReplacement
from torchrl.envs import DoubleToFloat
from torchrl.modules import SafeModule


# ====================================================================
# Offline Replay buffer
# ---------------------------


def make_offline_replay_buffer(rb_cfg):
    data = D4RLExperienceReplay(
        dataset_id=rb_cfg.dataset,
        split_trajs=False,
        batch_size=rb_cfg.batch_size,
        sampler=SamplerWithoutReplacement(drop_last=False),
        prefetch=4,
        direct_download=True,
    )

    data.append_transform(DoubleToFloat())

    return data


def make_gail_discriminator(cfg, train_env, device="cpu"):
    """Make GAIL discriminator."""

    state_dim = train_env.observation_spec["observation"].shape[0]
    action_dim = train_env.action_spec.shape[0]

    hidden_dim = cfg.gail.hidden_dim

    # Define Discriminator Network
    class Discriminator(nn.Module):
        def __init__(self, state_dim, action_dim):
            super().__init__()
            self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
            self.fc2 = nn.Linear(hidden_dim, hidden_dim)
            self.fc3 = nn.Linear(hidden_dim, 1)

        def forward(self, state, action):
            x = torch.cat([state, action], dim=1)
            x = torch.relu(self.fc1(x))
            x = torch.relu(self.fc2(x))
            return torch.sigmoid(self.fc3(x))

    d_module = SafeModule(
        module=Discriminator(state_dim, action_dim),
        in_keys=["observation", "action"],
        out_keys=["d_logits"],
    )
    return d_module.to(device)


def log_metrics(logger, metrics, step):
    if logger is not None:
        for metric_name, metric_value in metrics.items():
            logger.log_scalar(metric_name, metric_value, step)
