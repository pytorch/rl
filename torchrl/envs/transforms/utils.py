# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch


def check_finite(tensor: torch.Tensor):
    """Raise an error if a tensor has non-finite elements."""
    if not tensor.isfinite().all():
        raise ValueError("Encountered a non-finite tensor.")


def compute_reward2go(
    reward: torch.Tensor, episode_ends: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """Compute the discounted cumulative sum of rewards given the episode ends."""
    episode_ends = torch.concat(
        [episode_ends, torch.tensor([reward.shape[0]])]
    )  # add the last episode end
    start_idx = 0
    r2gs = []
    for idx in episode_ends:
        r2gs.append(discounted_cumsum(reward[start_idx:idx], gamma))
        start_idx = idx
    return torch.vstack(r2gs)


def discounted_cumsum(reward: torch.Tensor, gamma: float):
    """Compute the discounted cumulative sum of rewards."""
    discount = torch.pow(gamma, torch.arange(reward.shape[0])).unsqueeze(-1)
    return torch.cumsum(reward * discount, dim=0).flip(0)
