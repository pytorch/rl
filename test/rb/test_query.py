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
from tensordict import lazy_stack, TensorDict

from torchrl.data import (
    filter_trajectories,
    iter_trajectories,
    LazyTensorStorage,
    ListStorage,
    ReplayBuffer,
    TensorDictReplayBuffer,
    traj,
    Trajectory,
)
from torchrl.envs.transforms import Transform


class _CountingObsTransform(Transform):
    """Doubles the observation and counts how many times it is applied."""

    def __init__(self):
        super().__init__(in_keys=["observation"], out_keys=["observation"])
        self.calls = 0

    def forward(self, tensordict):
        self.calls += 1
        tensordict["observation"] = tensordict["observation"] * 2
        return tensordict


class _DoubleRewardTransform(Transform):
    """Doubles the post-step reward."""

    def __init__(self):
        super().__init__(in_keys=[("next", "reward")], out_keys=[("next", "reward")])

    def forward(self, tensordict):
        tensordict["next", "reward"] = tensordict["next", "reward"] * 2
        return tensordict


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
        assert trajectory["next", "done"].shape == (5, 1)
        assert trajectory[1:3].batch_size == torch.Size([2])

    def test_slicing_returns_trajectory(self):
        trajectory = Trajectory(_make_data(lengths=(5,)))
        sliced = trajectory[1:4]
        assert isinstance(sliced, Trajectory)
        assert sliced.length == 3
        assert sliced.reward.shape == (3, 1)
        assert isinstance(trajectory[0], Trajectory)

    def test_lazy_stack(self):
        first = Trajectory(_make_data(lengths=(5,)))
        second = Trajectory(_make_data(lengths=(3,)))
        stacked = lazy_stack([first, second])
        assert isinstance(stacked, Trajectory)
        assert stacked.batch_size == torch.Size([2, -1])
        assert isinstance(stacked[1], Trajectory)
        assert stacked[1].length == 3
        assert torch.equal(stacked[0].reward, first.reward)

    def test_tensorclass_ops_preserve_type(self):
        trajectory = Trajectory(_make_data(lengths=(5,)))
        assert isinstance(trajectory.clone(), Trajectory)
        assert isinstance(trajectory.to("cpu"), Trajectory)
        assert trajectory.clone().length == 5


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


class TestRequiredKeys:
    def test_field_reduction(self):
        assert (traj.reward.sum() > 1.0).required_keys() == frozenset({"reward"})
        assert (traj.observation.first() == 0).required_keys() == frozenset(
            {"observation"}
        )

    def test_nested_key(self):
        predicate = (traj[("collector", "traj_ids")] == 1).all()
        assert predicate.required_keys() == frozenset({("collector", "traj_ids")})

    def test_combinators_union(self):
        predicate = (traj.reward.sum() > 1.0) & (traj.length >= 5)
        assert predicate.required_keys() == frozenset({"reward"})
        predicate = (traj.reward.sum() > 1.0) | (traj.action.mean() < 0)
        assert predicate.required_keys() == frozenset({"reward", "action"})
        assert (~(traj.reward.sum() > 1.0)).required_keys() == frozenset({"reward"})

    def test_elementwise_propagates(self):
        assert (traj.reward > 1.0).any().required_keys() == frozenset({"reward"})
        assert (traj.reward > 1.0).all().required_keys() == frozenset({"reward"})

    def test_length_and_total_reward(self):
        assert (traj.length >= 5).required_keys() == frozenset()
        assert (traj.total_reward > 1.0).required_keys() == frozenset({"reward"})


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

    def test_query_result_has_all_keys(self):
        rb = self._make_rb()
        kept = rb.query(traj.reward.sum() > 6.0)
        assert len(kept) == 1
        assert kept[0].observation.shape == (7, 4)
        assert kept[0].state.shape == (7, 2)
        assert torch.equal(kept[0].observation, rb[:]["observation"][8:])

    def test_query_matches_slice_sampler_boundaries(self):
        from torchrl.data.replay_buffers.samplers import SliceSampler

        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10), batch_size=4)
        rb.extend(_make_data(lengths=(5, 3)))
        third = _make_data(lengths=(4,))
        third["collector", "traj_ids"] = torch.full((4,), 2, dtype=torch.long)
        rb.extend(third)
        sampler = SliceSampler(traj_key=("collector", "traj_ids"), num_slices=1)
        _, _, sampler_lengths = sampler._get_stop_and_length(rb._storage)
        query_lengths = [t.length for t in rb.query()]
        assert sorted(sampler_lengths.tolist()) == sorted(query_lengths)

    def test_query_ndim_storage(self):
        ids = torch.tensor([[0, 0, 0, 1, 1, 1], [2, 2, 2, 2, 3, 3]])
        data = TensorDict(
            {
                "observation": torch.randn(2, 6, 4),
                "next": {
                    "reward": (ids.float() + 1).unsqueeze(-1),
                    "done": torch.zeros(2, 6, 1, dtype=torch.bool),
                },
                "collector": {"traj_ids": ids},
            },
            batch_size=[2, 6],
        )
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(12, ndim=2))
        rb.extend(data)
        trajs = rb.query()
        assert [t.length for t in trajs] == [3, 3, 4, 2]
        assert all(isinstance(t, Trajectory) for t in trajs)
        assert trajs[2].reward.shape == (4, 1)
        kept = rb.query(traj.reward.first() == 3.0)
        assert [t.length for t in kept] == [4]

    def test_query_list_storage(self):
        data = _make_data(lengths=(5, 3))
        rb = ReplayBuffer(storage=ListStorage(10), batch_size=2)
        rb.extend(list(data.unbind(0)))
        trajs = rb.query()
        assert [t.length for t in trajs] == [5, 3]
        kept = rb.query(traj.reward.sum() > 5.5)
        assert [t.length for t in kept] == [3]

    def test_query_list_storage_non_td_raises(self):
        rb = ReplayBuffer(storage=ListStorage(10), batch_size=2)
        rb.extend([1, 2, 3])
        with pytest.raises(TypeError, match="tensordict-backed"):
            rb.query()

    def test_query_empty_buffer(self):
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10), batch_size=4)
        assert rb.query() == []

    def test_query_predicate_sees_transformed_values(self):
        data = _make_data(lengths=(5, 3, 7))
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(15),
            batch_size=4,
            transform=_DoubleRewardTransform(),
        )
        rb.extend(data)
        # Raw sums are 5, 6, 21; doubled sums are 10, 12, 42.
        kept = rb.query(traj.reward.sum() > 11.0)
        assert [t.length for t in kept] == [3, 7]
        assert torch.isclose(kept[1].reward.sum(), torch.tensor(42.0))

    def test_query_skips_unneeded_transforms(self):
        transform = _CountingObsTransform()
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(15), batch_size=4, transform=transform
        )
        rb.extend(_make_data(lengths=(5, 3, 7)))
        assert rb.query(traj.reward.sum() > 100.0) == []
        assert transform.calls == 0
        kept = rb.query(traj.reward.sum() > 6.0)
        assert len(kept) == 1
        assert transform.calls == 1
        assert torch.equal(kept[0].observation, rb._storage[:]["observation"][8:] * 2)

    def test_query_opaque_predicate_uses_full_transforms(self):
        transform = _CountingObsTransform()
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(15), batch_size=4, transform=transform
        )
        rb.extend(_make_data(lengths=(5, 3, 7)))
        kept = rb.query(lambda t: bool(t.observation.sum() != 0))
        assert len(kept) == 3
        assert transform.calls == 1


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
