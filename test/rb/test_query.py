# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for the trajectory query language and Trajectory views."""
from __future__ import annotations

import argparse
import warnings

import pytest
import torch
from tensordict import TensorDict

from torchrl.data import (
    filter_trajectories,
    iter_trajectories,
    LazyTensorStorage,
    TensorDictReplayBuffer,
    traj,
    Trajectory,
)


def _make_data(lengths=(5, 3, 7), reward_scale=1.0, with_ids=True, with_done=True):
    total = sum(lengths)
    ids = torch.cat(
        [torch.full((length,), i, dtype=torch.long) for i, length in enumerate(lengths)]
    )
    done = torch.zeros(total, 1, dtype=torch.bool)
    offset = 0
    for length in lengths:
        offset += length
        done[offset - 1] = True
    rewards = (ids.float() + 1).unsqueeze(-1) * reward_scale
    data = TensorDict(
        {
            "observation": torch.randn(total, 4),
            "state": torch.randn(total, 2),
            "action": torch.randn(total, 2),
            "next": {
                "observation": torch.randn(total, 4),
                "reward": rewards,
                "done": done,
            },
        },
        batch_size=[total],
    )
    if with_ids:
        data["collector", "traj_ids"] = ids
    if not with_done:
        del data["next", "done"]
    return data


class TestTrajectory:
    def test_attribute_access(self):
        data = _make_data(lengths=(5,))
        trajectory = Trajectory(data)
        assert trajectory.observation.shape == (5, 4)
        assert trajectory.state.shape == (5, 2)
        assert trajectory.action.shape == (5, 2)
        assert trajectory.reward.shape == (5, 1)
        assert trajectory.done.shape == (5, 1)
        assert trajectory.length == 5
        assert len(trajectory) == 5

    def test_total_reward(self):
        data = _make_data(lengths=(5,))
        trajectory = Trajectory(data)
        assert torch.isclose(trajectory.total_reward, torch.tensor(5.0))

    def test_missing_key_raises(self):
        trajectory = Trajectory(_make_data(lengths=(5,)))
        with pytest.raises(AttributeError, match="no entry 'pixels'"):
            trajectory.pixels

    def test_batch_dims_check(self):
        data = _make_data(lengths=(4,)).reshape(2, 2)
        with pytest.raises(ValueError, match="single batch dimension"):
            Trajectory(data)

    def test_getitem_passthrough(self):
        data = _make_data(lengths=(5,))
        trajectory = Trajectory(data)
        assert trajectory["observation"].shape == (5, 4)
        assert trajectory[1:3].batch_size == torch.Size([2])


class TestIterTrajectories:
    @pytest.mark.parametrize("with_ids", [True, False])
    def test_split_lengths(self, with_ids):
        data = _make_data(lengths=(5, 3, 7), with_ids=with_ids)
        trajs = list(iter_trajectories(data))
        assert [t.length for t in trajs] == [5, 3, 7]

    def test_split_without_trailing_done(self):
        data = _make_data(lengths=(5, 3, 7), with_ids=False)
        data["next", "done"][-1] = False
        trajs = list(iter_trajectories(data))
        assert [t.length for t in trajs] == [5, 3, 7]

    def test_explicit_trajectory_key(self):
        data = _make_data(lengths=(4, 6), with_done=False)
        data["episode_id"] = data["collector", "traj_ids"]
        del data["collector"]
        trajs = list(iter_trajectories(data, trajectory_key="episode_id"))
        assert [t.length for t in trajs] == [4, 6]

    def test_missing_split_info_raises(self):
        data = _make_data(lengths=(5,), with_ids=False, with_done=False)
        with pytest.raises(KeyError, match="Cannot split data into trajectories"):
            list(iter_trajectories(data))

    def test_end_flag_fallback_warns(self):
        data = _make_data(lengths=(5, 3), with_ids=False)
        with pytest.warns(UserWarning, match="end-of-episode"):
            trajs = list(iter_trajectories(data))
        assert [t.length for t in trajs] == [5, 3]

    def test_id_split_does_not_warn(self):
        data = _make_data(lengths=(5, 3), with_ids=True)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            trajs = list(iter_trajectories(data))
        assert [t.length for t in trajs] == [5, 3]

    def test_truncated_only_split(self):
        data = _make_data(lengths=(5, 3, 7), with_ids=False, with_done=False)
        truncated = torch.zeros(15, 1, dtype=torch.bool)
        truncated[4] = truncated[7] = truncated[14] = True
        data["next", "truncated"] = truncated
        with pytest.warns(UserWarning, match="end-of-episode"):
            trajs = list(iter_trajectories(data))
        assert [t.length for t in trajs] == [5, 3, 7]

    def test_end_flag_union(self):
        # first episode ends by termination, second by truncation only
        data = _make_data(lengths=(5, 3), with_ids=False, with_done=False)
        terminated = torch.zeros(8, 1, dtype=torch.bool)
        terminated[4] = True
        truncated = torch.zeros(8, 1, dtype=torch.bool)
        truncated[7] = True
        data["next", "terminated"] = terminated
        data["next", "truncated"] = truncated
        with pytest.warns(UserWarning, match="end-of-episode"):
            trajs = list(iter_trajectories(data))
        assert [t.length for t in trajs] == [5, 3]

    def test_wrong_key_raises(self):
        data = _make_data(lengths=(5,))
        with pytest.raises(KeyError, match="trajectory_key"):
            list(iter_trajectories(data, trajectory_key="not_a_key"))

    def test_non_tensordict_raises(self):
        with pytest.raises(TypeError, match="TensorDictBase"):
            list(iter_trajectories(torch.randn(10, 3)))


class TestTrajQueryLanguage:
    def test_scalar_reduction_predicate(self):
        data = _make_data(lengths=(5, 3, 7))
        kept = filter_trajectories(data, traj.reward.sum() > 6.0)
        assert [t.length for t in kept] == [7]

    def test_length_predicate(self):
        data = _make_data(lengths=(5, 3, 7))
        kept = filter_trajectories(data, traj.length >= 5)
        assert [t.length for t in kept] == [5, 7]

    def test_combinators(self):
        data = _make_data(lengths=(5, 3, 7))
        kept = filter_trajectories(
            data, (traj.length >= 5) & (traj.reward.mean() > 2.0)
        )
        assert [t.length for t in kept] == [7]
        kept = filter_trajectories(data, (traj.length < 5) | (traj.reward.mean() > 2.0))
        assert [t.length for t in kept] == [3, 7]
        kept = filter_trajectories(data, ~(traj.length >= 5))
        assert [t.length for t in kept] == [3]

    def test_elementwise_any_all(self):
        data = _make_data(lengths=(5, 3, 7))
        kept = filter_trajectories(data, (traj.reward > 2.5).any())
        assert [t.length for t in kept] == [7]
        kept = filter_trajectories(data, (traj.reward >= 1.0).all())
        assert [t.length for t in kept] == [5, 3, 7]

    def test_elementwise_without_reduction_raises(self):
        data = _make_data(lengths=(5, 3))
        with pytest.raises(TypeError, match="any\\(\\) or .all\\(\\)"):
            filter_trajectories(data, traj.reward > 2.5)

    def test_first_last(self):
        data = _make_data(lengths=(5, 3, 7))
        kept = filter_trajectories(data, traj.reward.last() > 2.5)
        assert [t.length for t in kept] == [7]

    def test_nested_key_access(self):
        data = _make_data(lengths=(5, 3, 7))
        kept = filter_trajectories(data, (traj[("collector", "traj_ids")] == 1).all())
        assert [t.length for t in kept] == [3]

    def test_total_reward_shortcut(self):
        data = _make_data(lengths=(5, 3, 7))
        kept = filter_trajectories(data, traj.total_reward > 6.0)
        assert [t.length for t in kept] == [7]

    def test_plain_callable_predicate(self):
        data = _make_data(lengths=(5, 3, 7))
        kept = filter_trajectories(data, lambda t: t.length == 3)
        assert [t.length for t in kept] == [3]

    def test_no_predicate_returns_all(self):
        data = _make_data(lengths=(5, 3, 7))
        assert len(filter_trajectories(data)) == 3


class TestReplayBufferQuery:
    def _make_rb(self, lengths=(5, 3, 7)):
        data = _make_data(lengths=lengths)
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(sum(lengths)), batch_size=4
        )
        rb.extend(data)
        return rb

    def test_query_all(self):
        rb = self._make_rb()
        trajs = rb.query()
        assert [t.length for t in trajs] == [5, 3, 7]

    def test_query_predicate(self):
        rb = self._make_rb()
        kept = rb.query((traj.reward.sum() > 6.0) | (traj.length == 3))
        assert [t.length for t in kept] == [3, 7]

    def test_query_views_stored_data(self):
        rb = self._make_rb()
        kept = rb.query(traj.length == 5)
        assert len(kept) == 1
        assert torch.equal(kept[0].observation, rb[:]["observation"][:5])

    def test_query_after_wraparound(self):
        # Capacity 10; write trajs of lengths (5, 3), then a third of length 4
        # that wraps around and overwrites the head of the first trajectory.
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10), batch_size=4)
        rb.extend(_make_data(lengths=(5, 3)))
        third = _make_data(lengths=(4,))
        third["collector", "traj_ids"] = torch.full((4,), 2, dtype=torch.long)
        rb.extend(third)
        trajs = rb.query()
        # Chronological order: head-truncated traj 0 (2 transitions
        # overwritten), whole traj 1, whole traj 2 (spanning the wrap point).
        assert [t.length for t in trajs] == [3, 3, 4]
        wrapped = rb.query(traj[("collector", "traj_ids")].first() == 2)
        assert len(wrapped) == 1
        assert torch.equal(wrapped[0].reward, third["next", "reward"])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
