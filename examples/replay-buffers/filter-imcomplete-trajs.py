# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Efficient Trajectory Sampling with CompletedTrajRepertoire

This example demonstrates how to design a custom transform that filters trajectories during sampling,
ensuring that only completed trajectories are present in sampled batches. This can be particularly useful
when dealing with environments where some trajectories might be corrupted or never reach a done state,
which could skew the learning process or lead to biased models. For instance, in robotics or autonomous
driving, a trajectory might be interrupted due to external factors such as hardware failures or human
intervention, resulting in incomplete or inconsistent data. By filtering out these incomplete trajectories,
we can improve the quality of the training data and increase the robustness of our models.
"""

import torch
from tensordict import TensorDictBase
from torchrl.data import LazyTensorStorage, ReplayBuffer
from torchrl.envs import GymEnv, TrajCounter, Transform


class CompletedTrajectoryRepertoire(Transform):
    """
    A transform that keeps track of completed trajectories and filters them out during sampling.
    """

    def __init__(self):
        super().__init__()
        self.completed_trajectories = set()
        self.repertoire_tensor = torch.zeros((), dtype=torch.int64)

    def _update_repertoire(self, tensordict: TensorDictBase) -> None:
        """Updates the repertoire of completed trajectories."""
        done = tensordict["next", "terminated"].squeeze(-1)
        traj = tensordict["next", "traj_count"][done].view(-1)
        if traj.numel():
            self.completed_trajectories = self.completed_trajectories.union(
                traj.tolist()
            )
            self.repertoire_tensor = torch.tensor(
                list(self.completed_trajectories), dtype=torch.int64
            )

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Updates the repertoire of completed trajectories during insertion."""
        self._update_repertoire(tensordict)
        return tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Filters out incomplete trajectories during sampling."""
        traj = tensordict["next", "traj_count"]
        traj = traj.unsqueeze(-1)
        has_traj = (traj == self.repertoire_tensor).any(-1)
        has_traj = has_traj.view(tensordict.shape)
        return tensordict[has_traj]


def main():
    # Create a CartPole environment with trajectory counting
    env = GymEnv("CartPole-v1").append_transform(TrajCounter())

    # Create a replay buffer with the completed trajectory repertoire transform
    buffer = ReplayBuffer(
        storage=LazyTensorStorage(1_000_000), transform=CompletedTrajectoryRepertoire()
    )

    # Roll out the environment for 1000 steps
    while True:
        rollout = env.rollout(1000, break_when_any_done=False)
        if not rollout["next", "done"][-1].item():
            break

    # Extend the replay buffer with the rollout
    buffer.extend(rollout)

    # Get the last trajectory count
    last_traj_count = rollout[-1]["next", "traj_count"].item()
    print(f"Incomplete trajectory: {last_traj_count}")

    # Sample from the replay buffer 10 times
    for _ in range(10):
        sample_traj_counts = buffer.sample(32)["next", "traj_count"].unique()
        print(f"Sampled trajectories: {sample_traj_counts}")
        assert last_traj_count not in sample_traj_counts


if __name__ == "__main__":
    main()
