# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for HERReplayBuffer and HindsightStrategy.

These tests previously lived in the monolithic ``test/test_rb.py``.
They moved to a dedicated file when the rb test suite was split into
``test/rb/`` upstream.
"""
from __future__ import annotations

import argparse

import pytest
import torch
from tensordict import TensorDict


class TestHERReplayBuffer:
    """Tests for HERReplayBuffer and HindsightStrategy."""

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _make_goal_env_data(n_steps: int, goal_dim: int = 3, obs_dim: int = 4):
        """Return a TensorDict mimicking a goal-conditioned env rollout."""
        torch.manual_seed(0)
        # All transitions belong to a single episode; last step is done.
        done = torch.zeros(n_steps, 1, dtype=torch.bool)
        done[-1] = True
        terminated = torch.zeros(n_steps, 1, dtype=torch.bool)
        terminated[-1] = True

        desired_goal = torch.randn(n_steps, goal_dim)
        achieved_goal = torch.randn(n_steps, goal_dim)

        return TensorDict(
            {
                "observation": torch.randn(n_steps, obs_dim),
                "desired_goal": desired_goal,
                "achieved_goal": achieved_goal,
                "action": torch.randn(n_steps, 2),
                "next": {
                    "observation": torch.randn(n_steps, obs_dim),
                    "desired_goal": desired_goal,
                    "achieved_goal": achieved_goal,
                    "reward": torch.zeros(n_steps, 1),
                    "done": done,
                    "terminated": terminated,
                },
            },
            batch_size=[n_steps],
        )

    @staticmethod
    def _sparse_reward_fn(td: TensorDict) -> torch.Tensor:
        dist = (td["achieved_goal"] - td["desired_goal"]).norm(dim=-1, keepdim=True)
        return (dist < 0.5).float()

    def _make_rb(self, n_steps=20, **kwargs):
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        rb = HERReplayBuffer(
            reward_fn=self._sparse_reward_fn,
            storage=LazyTensorStorage(1000),
            batch_size=n_steps,
            **kwargs,
        )
        data = self._make_goal_env_data(n_steps)
        rb.extend(data)
        return rb

    # ------------------------------------------------------------------
    # basic API
    # ------------------------------------------------------------------

    def test_import(self):
        from torchrl.data import HERReplayBuffer, HindsightStrategy  # noqa: F401

    def test_invalid_her_ratio(self):
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        with pytest.raises(ValueError, match="her_ratio"):
            HERReplayBuffer(
                reward_fn=self._sparse_reward_fn,
                storage=LazyTensorStorage(100),
                her_ratio=1.5,
            )

    def test_sample_shape(self):
        rb = self._make_rb(n_steps=20, her_ratio=0.8)
        batch = rb.sample()
        assert batch.batch_size == torch.Size([20])

    def test_her_ratio_zero_unchanged(self):
        """her_ratio=0 must return data with the original stored goals."""
        rb = self._make_rb(n_steps=20, her_ratio=0.0)
        batch, info = rb.sample(return_info=True)
        idx = info["index"]
        stored = rb._storage.get(idx)
        torch.testing.assert_close(batch["desired_goal"], stored["desired_goal"])

    # ------------------------------------------------------------------
    # strategy correctness
    # ------------------------------------------------------------------

    @pytest.mark.parametrize("strategy", ["future", "final", "episode", "random"])
    def test_strategies_run(self, strategy):
        """All four strategies must produce a valid batch without error."""
        rb = self._make_rb(n_steps=20, strategy=strategy, her_ratio=0.8)
        batch = rb.sample()
        assert batch.batch_size == torch.Size([20])

    def test_final_strategy_uses_last_achieved(self):
        """FINAL strategy: relabeled goal == achieved_goal of the last step."""
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        n = 10
        data = self._make_goal_env_data(n)
        rb = HERReplayBuffer(
            reward_fn=self._sparse_reward_fn,
            storage=LazyTensorStorage(100),
            batch_size=n,
            her_ratio=1.0,
            strategy="final",
        )
        rb.extend(data)
        batch = rb.sample()

        # All relabeled goals must equal the achieved_goal of the last transition.
        last_achieved = data["next", "achieved_goal"][-1]  # shape [goal_dim]
        for i in range(n):
            torch.testing.assert_close(batch["desired_goal"][i], last_achieved)

    def test_future_goal_not_from_past(self):
        """FUTURE strategy: goal source index must be >= the sampled index."""
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        n = 30
        data = self._make_goal_env_data(n)

        # Tag each achieved_goal with a unique step index so we can trace
        # which step was used as goal source.
        step_ids = torch.arange(n, dtype=torch.float).unsqueeze(1).expand(n, 3)
        data["achieved_goal"] = step_ids.clone()
        data["next", "achieved_goal"] = step_ids.clone()
        # desired_goal starts as all-zeros so we can detect relabeling.
        data["desired_goal"] = torch.zeros(n, 3)
        data["next", "desired_goal"] = torch.zeros(n, 3)

        rb = HERReplayBuffer(
            reward_fn=lambda td: torch.zeros(td.batch_size[0], 1),
            storage=LazyTensorStorage(100),
            batch_size=n,
            her_ratio=1.0,
            strategy="future",
        )
        rb.extend(data)

        # info["index"] gives us the storage indices that were sampled.
        batch, info = rb.sample(return_info=True)
        sampled_idx = info["index"]

        n_her = n  # her_ratio=1.0
        for i in range(n_her):
            sampled_step = sampled_idx[i].item()
            # The relabeled goal is the step ID of the goal source.
            goal_step = batch["desired_goal"][i][0].item()
            assert goal_step >= sampled_step, (
                f"FUTURE goal at storage idx {sampled_step} came from "
                f"earlier step {goal_step}"
            )

    # ------------------------------------------------------------------
    # reward recomputation
    # ------------------------------------------------------------------

    def test_reward_recomputed_for_her_transitions(self):
        """Relabeled transitions must have reward recomputed by reward_fn."""
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        n = 20
        data = self._make_goal_env_data(n)
        # Set all stored rewards to a sentinel value (-99) so we can detect
        # which ones were recomputed.
        sentinel = -99.0
        data["next", "reward"] = torch.full((n, 1), sentinel)

        rb = HERReplayBuffer(
            reward_fn=self._sparse_reward_fn,
            storage=LazyTensorStorage(100),
            batch_size=n,
            her_ratio=0.8,
        )
        rb.extend(data)
        batch = rb.sample()

        n_her = int(n * 0.8)
        # HER slice: reward must not be the sentinel (was recomputed)
        assert not (batch["next", "reward"][:n_her] == sentinel).all()
        # Non-HER slice: reward remains as stored
        assert (batch["next", "reward"][n_her:] == sentinel).all()

    # ------------------------------------------------------------------
    # multi-episode correctness
    # ------------------------------------------------------------------

    def test_multi_episode_final_stays_within_episode(self):
        """FINAL strategy: each relabeled goal must come from the correct episode."""
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        ep_lens = [5, 8, 7]
        n = sum(ep_lens)
        done = torch.zeros(n, 1, dtype=torch.bool)
        terminated = torch.zeros(n, 1, dtype=torch.bool)
        # Mark episode ends
        ends = [4, 12, 19]  # 0-indexed last step of each episode
        for e in ends:
            done[e] = True
            terminated[e] = True

        # Tag achieved_goal with the episode index so we can verify
        episode_ids = torch.zeros(n, dtype=torch.long)
        episode_ids[5:13] = 1
        episode_ids[13:] = 2
        achieved = episode_ids.float().unsqueeze(1).expand(n, 3).clone()

        data = TensorDict(
            {
                "observation": torch.randn(n, 4),
                "desired_goal": torch.zeros(n, 3),
                "achieved_goal": achieved,
                "action": torch.randn(n, 2),
                "next": {
                    "observation": torch.randn(n, 4),
                    "desired_goal": torch.zeros(n, 3),
                    "achieved_goal": achieved,
                    "reward": torch.zeros(n, 1),
                    "done": done,
                    "terminated": terminated,
                },
            },
            batch_size=[n],
        )

        rb = HERReplayBuffer(
            reward_fn=lambda td: torch.zeros(*td.batch_size, 1),
            storage=LazyTensorStorage(100),
            batch_size=n,
            her_ratio=1.0,
            strategy="final",
        )
        rb.extend(data)

        # Run multiple times to average over randomness in index selection
        for _ in range(10):
            batch, info = rb.sample(return_info=True)
            sampled_idx = info["index"]
            for i in range(n):
                sid = sampled_idx[i].item()
                src_ep = int(episode_ids[sid].item())
                relabeled_ep = int(batch["desired_goal"][i][0].item())
                assert src_ep == relabeled_ep, (
                    f"Transition from ep {src_ep} (idx {sid}) got goal "
                    f"from ep {relabeled_ep}"
                )

    # ------------------------------------------------------------------
    # cache invalidation
    # ------------------------------------------------------------------

    def test_cache_rebuilds_after_extend(self):
        """Episode cache must reflect new data after extend."""
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        rb = HERReplayBuffer(
            reward_fn=self._sparse_reward_fn,
            storage=LazyTensorStorage(200),
            batch_size=10,
        )
        data1 = self._make_goal_env_data(10)
        rb.extend(data1)
        # Force a cache build by sampling once
        rb.sample()
        key1 = rb._last_cache_key

        data2 = self._make_goal_env_data(10)
        rb.extend(data2)
        # Sample again — cache must rebuild because storage changed
        rb.sample()
        assert rb._last_cache_key != key1, "Cache key should change after extend"

    def test_cache_rebuilds_after_add(self):
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        rb = HERReplayBuffer(
            reward_fn=self._sparse_reward_fn,
            storage=LazyTensorStorage(200),
            batch_size=5,
        )
        data = self._make_goal_env_data(5)
        rb.extend(data)
        rb.sample()
        key1 = rb._last_cache_key

        single = self._make_goal_env_data(1)
        rb.add(single[0])
        rb.sample()
        assert rb._last_cache_key != key1

    # ------------------------------------------------------------------
    # HindsightStrategy enum
    # ------------------------------------------------------------------

    def test_strategy_accepts_string(self):
        from torchrl.data import HERReplayBuffer, HindsightStrategy, LazyTensorStorage

        rb = HERReplayBuffer(
            reward_fn=self._sparse_reward_fn,
            storage=LazyTensorStorage(100),
            strategy="future",
        )
        assert rb.strategy is HindsightStrategy.FUTURE

    def test_strategy_invalid(self):
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        with pytest.raises(ValueError):
            HERReplayBuffer(
                reward_fn=self._sparse_reward_fn,
                storage=LazyTensorStorage(100),
                strategy="invalid_strategy",
            )

    # ------------------------------------------------------------------
    # EPISODE strategy stays within episode
    # ------------------------------------------------------------------

    def test_episode_strategy_stays_within_episode(self):
        """EPISODE strategy: goal source must lie within the same episode."""
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        n = 30
        data = self._make_goal_env_data(n)
        step_ids = torch.arange(n, dtype=torch.float).unsqueeze(1).expand(n, 3).clone()
        data["achieved_goal"] = step_ids
        data["next", "achieved_goal"] = step_ids
        data["desired_goal"] = torch.zeros(n, 3)

        rb = HERReplayBuffer(
            reward_fn=lambda td: torch.zeros(td.batch_size[0], 1),
            storage=LazyTensorStorage(100),
            batch_size=n,
            her_ratio=1.0,
            strategy="episode",
        )
        rb.extend(data)

        for _ in range(5):
            batch = rb.sample()
            goal_step_ids = batch["desired_goal"][:, 0]
            assert (goal_step_ids >= 0).all()
            assert (goal_step_ids <= n - 1).all()

    # ------------------------------------------------------------------
    # her_ratio=1.0 — full batch relabeled
    # ------------------------------------------------------------------

    def test_her_ratio_one_full_relabel(self):
        """her_ratio=1.0: every transition must be relabeled."""
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        n = 15
        data = self._make_goal_env_data(n)
        sentinel = -99.0
        data["next", "reward"] = torch.full((n, 1), sentinel)

        rb = HERReplayBuffer(
            reward_fn=self._sparse_reward_fn,
            storage=LazyTensorStorage(100),
            batch_size=n,
            her_ratio=1.0,
        )
        rb.extend(data)
        batch = rb.sample()
        assert not (batch["next", "reward"] == sentinel).any()

    # ------------------------------------------------------------------
    # custom reward_key
    # ------------------------------------------------------------------

    def test_custom_reward_key(self):
        """reward_key parameter controls where recomputed reward is written."""
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        n = 10
        done = torch.zeros(n, 1, dtype=torch.bool)
        done[-1] = True
        data = TensorDict(
            {
                "observation": torch.randn(n, 4),
                "desired_goal": torch.randn(n, 3),
                "achieved_goal": torch.randn(n, 3),
                "action": torch.randn(n, 2),
                "shaped_reward": torch.full((n, 1), -99.0),
                "next": {
                    "observation": torch.randn(n, 4),
                    "desired_goal": torch.randn(n, 3),
                    "achieved_goal": torch.randn(n, 3),
                    "done": done,
                },
            },
            batch_size=[n],
        )
        rb = HERReplayBuffer(
            reward_fn=lambda td: torch.ones(td.batch_size[0], 1),
            storage=LazyTensorStorage(100),
            batch_size=n,
            her_ratio=1.0,
            reward_key="shaped_reward",
        )
        rb.extend(data)
        batch = rb.sample()
        assert (batch["shaped_reward"] == 1.0).all()

    # ------------------------------------------------------------------
    # missing key validation
    # ------------------------------------------------------------------

    def test_missing_goal_key_raises(self):
        """Clear KeyError when goal_key is absent from storage."""
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        n = 5
        done = torch.zeros(n, 1, dtype=torch.bool)
        done[-1] = True
        data = TensorDict(
            {
                "observation": torch.randn(n, 4),
                "achieved_goal": torch.randn(n, 3),
                "action": torch.randn(n, 2),
                "next": {"reward": torch.zeros(n, 1), "done": done},
            },
            batch_size=[n],
        )
        rb = HERReplayBuffer(
            reward_fn=self._sparse_reward_fn,
            storage=LazyTensorStorage(100),
            batch_size=n,
        )
        rb.extend(data)
        with pytest.raises(KeyError, match="goal_key"):
            rb.sample()

    # ------------------------------------------------------------------
    # repr
    # ------------------------------------------------------------------

    def test_repr(self):
        rb = self._make_rb(n_steps=10)
        r = repr(rb)
        assert "HERReplayBuffer" in r
        assert "future" in r
        assert "desired_goal" in r

    # ------------------------------------------------------------------
    # state_dict / load_state_dict
    # ------------------------------------------------------------------

    def test_state_dict_round_trip(self):
        """state_dict must preserve the episode-boundary cache."""
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        rb = HERReplayBuffer(
            reward_fn=self._sparse_reward_fn,
            storage=LazyTensorStorage(200),
            batch_size=10,
        )
        rb.extend(self._make_goal_env_data(10))
        rb.sample()
        sd = rb.state_dict()
        assert "_her" in sd
        assert sd["_her"]["episode_ends_cache"] is not None

        rb2 = HERReplayBuffer(
            reward_fn=self._sparse_reward_fn,
            storage=LazyTensorStorage(200),
            batch_size=10,
        )
        rb2.load_state_dict(sd)
        assert rb2._last_cache_key == rb._last_cache_key

    # ------------------------------------------------------------------
    # nested goal keys
    # ------------------------------------------------------------------

    def test_nested_goal_key(self):
        """goal_key and achieved_goal_key can be nested tuples."""
        from torchrl.data import HERReplayBuffer, LazyTensorStorage

        n = 10
        done = torch.zeros(n, 1, dtype=torch.bool)
        done[-1] = True
        data = TensorDict(
            {
                "obs": {
                    "desired_goal": torch.randn(n, 3),
                    "achieved_goal": torch.randn(n, 3),
                    "pixels": torch.randn(n, 4),
                },
                "action": torch.randn(n, 2),
                "next": {
                    "obs": {
                        "desired_goal": torch.randn(n, 3),
                        "achieved_goal": torch.randn(n, 3),
                    },
                    "reward": torch.zeros(n, 1),
                    "done": done,
                },
            },
            batch_size=[n],
        )
        rb = HERReplayBuffer(
            reward_fn=lambda td: torch.zeros(*td.batch_size, 1),
            storage=LazyTensorStorage(100),
            batch_size=n,
            her_ratio=1.0,
            goal_key=("obs", "desired_goal"),
            achieved_goal_key=("obs", "achieved_goal"),
        )
        rb.extend(data)
        # Should not raise
        batch = rb.sample()
        assert batch.batch_size == torch.Size([n])


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
