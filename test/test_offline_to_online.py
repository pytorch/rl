# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch
from tensordict import TensorDict

from torchrl.data import LazyTensorStorage, OfflineToOnlineReplayBuffer, ReplayBuffer
from torchrl.data.datasets.utils import load_dataset
from torchrl.data.replay_buffers.offline_to_online import prefill_replay_buffer


def _make_offline_buffer(n: int = 1000, obs_dim: int = 4, action_dim: int = 2):
    """A plain ReplayBuffer standing in for an offline dataset (no Minari/D4RL needed)."""
    rb = ReplayBuffer(storage=LazyTensorStorage(n))
    rb.extend(
        TensorDict(
            {
                "observation": torch.randn(n, obs_dim),
                "action": torch.randn(n, action_dim),
                ("next", "reward"): torch.randn(n, 1),
            },
            batch_size=[n],
        )
    )
    return rb


def _make_online_data(n: int = 50, obs_dim: int = 4, action_dim: int = 2):
    return TensorDict(
        {
            "observation": torch.randn(n, obs_dim),
            "action": torch.randn(n, action_dim),
            ("next", "reward"): torch.randn(n, 1),
        },
        batch_size=[n],
    )


class TestOfflineToOnlineReplayBuffer:
    def test_construction_with_capacity(self):
        offline = _make_offline_buffer()
        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline,
            online_capacity=500,
            offline_fraction=0.5,
            batch_size=32,
        )
        assert rb.offline_buffer is offline
        assert isinstance(rb.online_buffer, ReplayBuffer)
        assert len(rb.online_buffer) == 0

    def test_construction_with_storage(self):
        offline = _make_offline_buffer()
        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline,
            online_storage=LazyTensorStorage(500),
            batch_size=32,
        )
        assert isinstance(rb.online_buffer, ReplayBuffer)

    def test_construction_requires_exactly_one_online_arg(self):
        offline = _make_offline_buffer()
        with pytest.raises(ValueError, match="not both"):
            OfflineToOnlineReplayBuffer(
                offline_dataset=offline,
                online_capacity=500,
                online_storage=LazyTensorStorage(500),
            )
        with pytest.raises(ValueError, match="one of"):
            OfflineToOnlineReplayBuffer(offline_dataset=offline)

    @pytest.mark.parametrize("fraction", [-0.1, 0.0, 1.0, 1.5])
    def test_invalid_offline_fraction(self, fraction):
        offline = _make_offline_buffer()
        with pytest.raises(ValueError, match="offline_fraction"):
            OfflineToOnlineReplayBuffer(
                offline_dataset=offline,
                online_capacity=500,
                offline_fraction=fraction,
            )

    def test_dataset_kwargs_rejected_for_object(self):
        offline = _make_offline_buffer()
        with pytest.raises(ValueError, match="only forwarded when"):
            OfflineToOnlineReplayBuffer(
                offline_dataset=offline,
                online_capacity=500,
                split_trajs=True,  # stray dataset kwarg
            )

    def test_extend_routes_to_online_only(self):
        offline = _make_offline_buffer(n=1000)
        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline,
            online_capacity=500,
            batch_size=32,
        )
        offline_len_before = len(rb.offline_buffer)
        rb.extend(_make_online_data(50))
        assert len(rb.online_buffer) == 50
        # offline is untouched
        assert len(rb.offline_buffer) == offline_len_before

    def test_sample_falls_back_to_offline_when_online_empty(self):
        offline = _make_offline_buffer()
        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline,
            online_capacity=500,
            batch_size=32,
        )
        batch = rb.sample(32)
        assert batch.batch_size == torch.Size([32])

    def test_sample_returns_flat_batch(self):
        offline = _make_offline_buffer()
        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline,
            online_capacity=500,
            batch_size=32,
        )
        rb.extend(_make_online_data(50))
        batch = rb.sample(64)
        # Flat [64], NOT [2, 32]
        assert batch.batch_size == torch.Size([64])

    def test_sample_uses_default_batch_size(self):
        offline = _make_offline_buffer()
        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline,
            online_capacity=500,
            batch_size=16,
        )
        rb.extend(_make_online_data(50))
        batch = rb.sample()
        assert batch.batch_size == torch.Size([16])

    def test_sample_without_batch_size_raises(self):
        offline = _make_offline_buffer()
        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline,
            online_capacity=500,
        )
        rb.extend(_make_online_data(50))
        with pytest.raises(ValueError, match="batch_size must be provided"):
            rb.sample()

    @pytest.mark.parametrize("fraction", [0.25, 0.5, 0.75])
    def test_offline_fraction_respected_exactly(self, fraction):
        # Tag offline source=0, online source=1 so we can count exactly.
        offline = ReplayBuffer(storage=LazyTensorStorage(2000))
        offline.extend(
            TensorDict(
                {
                    "observation": torch.randn(2000, 4),
                    "source": torch.zeros(2000, dtype=torch.long),
                },
                [2000],
            )
        )
        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline,
            online_capacity=2000,
            offline_fraction=fraction,
            batch_size=32,
        )
        rb.extend(
            TensorDict(
                {
                    "observation": torch.randn(500, 4),
                    "source": torch.ones(500, dtype=torch.long),
                },
                [500],
            )
        )
        batch = rb.sample(100)
        offline_count = (batch["source"] == 0).sum().item()
        # Deterministic: exactly round(fraction * batch_size) offline samples.
        assert offline_count == round(fraction * 100)

    def test_anneal_reduces_offline_fraction(self):
        offline = _make_offline_buffer()
        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline,
            online_capacity=500,
            offline_fraction=0.8,
            batch_size=32,
        )
        # halfway: 0.8 * (1 - 0.5) = 0.4
        rb.anneal(step=50, total_steps=100)
        assert abs(rb.offline_fraction - 0.4) < 1e-6
        # fully annealed: offline fraction -> 0
        rb.anneal(step=100, total_steps=100)
        assert rb.offline_fraction == 0.0

    def test_anneal_clamps_past_total_steps(self):
        offline = _make_offline_buffer()
        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline,
            online_capacity=500,
            offline_fraction=0.5,
            batch_size=32,
        )
        rb.anneal(step=200, total_steps=100)
        assert rb.offline_fraction == 0.0  # does not go negative

    def test_fully_annealed_samples_online_only(self):
        offline = ReplayBuffer(storage=LazyTensorStorage(1000))
        offline.extend(
            TensorDict(
                {
                    "observation": torch.randn(1000, 4),
                    "source": torch.zeros(1000, dtype=torch.long),
                },
                [1000],
            )
        )
        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline,
            online_capacity=1000,
            offline_fraction=0.5,
            batch_size=32,
        )
        rb.extend(
            TensorDict(
                {
                    "observation": torch.randn(500, 4),
                    "source": torch.ones(500, dtype=torch.long),
                },
                [500],
            )
        )
        rb.anneal(step=100, total_steps=100)
        batch = rb.sample(64)
        assert (batch["source"] == 1).all()  # all online

    def test_len(self):
        offline = _make_offline_buffer(n=1000)
        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline,
            online_capacity=500,
            batch_size=32,
        )
        rb.extend(_make_online_data(50))
        assert len(rb) == 1050


class TestPrefillReplayBuffer:
    def test_prefill_exact_n_samples(self):
        offline = _make_offline_buffer(n=1000)
        target = ReplayBuffer(storage=LazyTensorStorage(10_000))
        prefill_replay_buffer(target, offline, n_samples=200)
        assert len(target) == 200

    def test_prefill_full_dataset(self):
        offline = _make_offline_buffer(n=300)
        target = ReplayBuffer(storage=LazyTensorStorage(10_000))
        prefill_replay_buffer(target, offline)
        assert len(target) == 300

    def test_prefill_caps_at_dataset_size(self):
        offline = _make_offline_buffer(n=100)
        target = ReplayBuffer(storage=LazyTensorStorage(10_000))
        prefill_replay_buffer(target, offline, n_samples=500)
        # cannot copy more than the dataset holds
        assert len(target) == 100

    def test_prefill_returns_buffer_for_chaining(self):
        offline = _make_offline_buffer(n=300)
        target = ReplayBuffer(storage=LazyTensorStorage(10_000))
        result = prefill_replay_buffer(target, offline, n_samples=50)
        assert result is target

    def test_prefill_respects_chunk_size(self):
        offline = _make_offline_buffer(n=1000)
        target = ReplayBuffer(storage=LazyTensorStorage(10_000))
        prefill_replay_buffer(target, offline, n_samples=250, chunk_size=37)
        assert len(target) == 250


class TestLoadDataset:
    def test_missing_prefix_raises(self):
        with pytest.raises(ValueError, match="must be prefixed"):
            load_dataset("halfcheetah-medium-v2")

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset source"):
            load_dataset("mujoco:hopper-v0")

    def test_minari_prefix_routes_to_minari(self, monkeypatch):
        captured = {}

        class FakeMinari:
            def __init__(self, dataset_id, **kwargs):
                captured["dataset_id"] = dataset_id
                captured["kwargs"] = kwargs

        import torchrl.data.datasets.minari_data as minari_mod

        monkeypatch.setattr(minari_mod, "MinariExperienceReplay", FakeMinari)
        load_dataset("minari:mujoco/hopper/expert-v0", batch_size=256)
        assert captured["dataset_id"] == "mujoco/hopper/expert-v0"
        assert captured["kwargs"] == {"batch_size": 256}

    def test_d4rl_prefix_routes_to_d4rl(self, monkeypatch):
        captured = {}

        class FakeD4RL:
            def __init__(self, dataset_id, **kwargs):
                captured["dataset_id"] = dataset_id
                captured["kwargs"] = kwargs

        import torchrl.data.datasets.d4rl as d4rl_mod

        monkeypatch.setattr(d4rl_mod, "D4RLExperienceReplay", FakeD4RL)
        load_dataset("d4rl:halfcheetah-medium-v2", split_trajs=True)
        assert captured["dataset_id"] == "halfcheetah-medium-v2"
        assert captured["kwargs"] == {"split_trajs": True}

    def test_string_construction_through_buffer(self, monkeypatch):
        """OfflineToOnlineReplayBuffer resolves string datasets via load_dataset."""
        offline = _make_offline_buffer()

        import torchrl.data.datasets.d4rl as d4rl_mod

        monkeypatch.setattr(
            d4rl_mod, "D4RLExperienceReplay", lambda dataset_id, **kw: offline
        )
        rb = OfflineToOnlineReplayBuffer(
            "d4rl:halfcheetah-medium-v2",
            online_capacity=500,
            batch_size=32,
        )
        assert rb.offline_buffer is offline


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
