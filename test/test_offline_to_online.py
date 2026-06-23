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
from torchrl.data.datasets import utils as dataset_utils
from torchrl.data.datasets.utils import load_dataset, register_dataset
from torchrl.data.replay_buffers.offline_to_online import prefill_replay_buffer
from torchrl.envs.libs.gym import _has_gym

# Running a SAC loss requires a tensordict new enough to support
# ``to_module(preserve_module_state=...)``; the offline-to-online wiring itself
# does not.
_LOSS_RUNNABLE = (
    "preserve_module_state"
    in __import__("inspect").signature(TensorDict.to_module).parameters
)


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

    def test_string_dataset_requires_batch_size(self):
        with pytest.raises(ValueError, match="batch_size must be provided"):
            OfflineToOnlineReplayBuffer(
                "d4rl:halfcheetah-medium-v2",
                online_capacity=500,
            )

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

    def test_prefill_string_dataset_uses_chunk_size_as_dataset_batch_size(
        self, monkeypatch
    ):
        offline = _make_offline_buffer(n=100)
        captured = {}

        def fake_d4rl(dataset_id, **kwargs):
            captured["dataset_id"] = dataset_id
            captured["kwargs"] = kwargs
            return offline

        import torchrl.data.datasets.d4rl as d4rl_mod

        monkeypatch.setattr(d4rl_mod, "D4RLExperienceReplay", fake_d4rl)
        target = ReplayBuffer(storage=LazyTensorStorage(1000))
        prefill_replay_buffer(
            target,
            "d4rl:halfcheetah-medium-v2",
            n_samples=20,
            chunk_size=7,
        )
        assert captured["dataset_id"] == "halfcheetah-medium-v2"
        assert captured["kwargs"] == {"batch_size": 7}
        assert len(target) == 20


class TestLoadDataset:
    def test_missing_prefix_raises(self):
        with pytest.raises(ValueError, match="must be prefixed"):
            load_dataset("halfcheetah-medium-v2")

    def test_unknown_prefix_raises(self):
        with pytest.raises(ValueError, match="Unknown dataset source"):
            load_dataset("mujoco:hopper-v0")

    def test_registry_includes_existing_dataset_backends(self):
        expected = {
            "atari",
            "atari_dqn",
            "d4rl",
            "gen_dgrl",
            "lerobot",
            "minari",
            "openml",
            "openx",
            "roboset",
            "vd4rl",
        }
        assert expected.issubset(dataset_utils._DATASET_REGISTRY)

    def test_register_dataset_routes_custom_factory(self):
        captured = {}

        class FakeDataset:
            def __init__(self, dataset_id, **kwargs):
                captured["dataset_id"] = dataset_id
                captured["kwargs"] = kwargs

        prefix = "test_backend"
        dataset_utils._DATASET_REGISTRY.pop(prefix, None)
        try:
            register_dataset(prefix, FakeDataset)
            dataset = load_dataset(f"{prefix}:demo-dataset", batch_size=11)
        finally:
            dataset_utils._DATASET_REGISTRY.pop(prefix, None)
        assert isinstance(dataset, FakeDataset)
        assert captured["dataset_id"] == "demo-dataset"
        assert captured["kwargs"] == {"batch_size": 11}

    def test_registry_routes_non_d4rl_builtin_prefix(self, monkeypatch):
        captured = {}

        class FakeOpenML:
            def __init__(self, dataset_id, **kwargs):
                captured["dataset_id"] = dataset_id
                captured["kwargs"] = kwargs

        class FakeModule:
            OpenMLExperienceReplay = FakeOpenML

        def import_module(name):
            assert name == "torchrl.data.datasets.openml"
            return FakeModule

        monkeypatch.setattr(dataset_utils.importlib, "import_module", import_module)
        load_dataset("openml:iris", batch_size=16)
        assert captured["dataset_id"] == "iris"
        assert captured["kwargs"] == {"batch_size": 16}

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

    def test_string_construction_forwards_batch_size_to_dataset(self, monkeypatch):
        captured = {}
        offline = _make_offline_buffer()

        def fake_d4rl(dataset_id, **kwargs):
            captured["dataset_id"] = dataset_id
            captured["kwargs"] = kwargs
            return offline

        import torchrl.data.datasets.d4rl as d4rl_mod

        monkeypatch.setattr(d4rl_mod, "D4RLExperienceReplay", fake_d4rl)
        rb = OfflineToOnlineReplayBuffer(
            "d4rl:halfcheetah-medium-v2",
            online_capacity=500,
            batch_size=32,
        )
        assert rb.offline_buffer is offline
        assert captured["dataset_id"] == "halfcheetah-medium-v2"
        assert captured["kwargs"] == {"batch_size": 32}


class _StubTrainer:
    """Minimal stand-in exposing the ``collected_frames`` the anneal hook reads."""

    def __init__(self, collected_frames: int = 0):
        self.collected_frames = collected_frames


class TestOfflineToOnlineReplayBufferHook:
    def test_extend_uses_collector_mask(self):
        from torchrl.trainers.algorithms.offline_to_online import (
            OfflineToOnlineReplayBufferHook,
        )

        offline = _make_offline_buffer()
        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline, online_capacity=500, batch_size=16
        )
        hook = OfflineToOnlineReplayBufferHook(rb)
        mask = torch.ones(2, 5, dtype=torch.bool)
        mask[0, 3:] = False  # 2 invalid rows -> 8 valid
        data = TensorDict(
            {
                "observation": torch.randn(2, 5, 4),
                ("collector", "mask"): mask,
            },
            batch_size=[2, 5],
        )
        hook.extend(data)
        assert len(rb.online_buffer) == 8
        # collector bookkeeping is not stored
        assert "collector" not in rb.online_buffer.sample(4).keys()

    def test_state_dict_roundtrip(self):
        from torchrl.trainers.algorithms.offline_to_online import (
            OfflineToOnlineReplayBufferHook,
        )

        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=_make_offline_buffer(), online_capacity=500, batch_size=16
        )
        hook = OfflineToOnlineReplayBufferHook(rb)
        hook.extend(_make_online_data(20))

        rb2 = OfflineToOnlineReplayBuffer(
            offline_dataset=_make_offline_buffer(), online_capacity=500, batch_size=16
        )
        hook2 = OfflineToOnlineReplayBufferHook(rb2)
        hook2.load_state_dict(hook.state_dict())
        assert len(rb2.online_buffer) == 20


class TestOfflineToOnlineAnnealHook:
    def test_anneal_decays_fraction(self):
        from torchrl.trainers.algorithms.offline_to_online import (
            OfflineToOnlineAnnealHook,
        )

        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=_make_offline_buffer(),
            online_capacity=500,
            offline_fraction=0.8,
            batch_size=16,
        )
        stub = _StubTrainer()
        hook = OfflineToOnlineAnnealHook(stub, rb, anneal_frames=100)

        stub.collected_frames = 0
        hook()
        assert rb.offline_fraction == pytest.approx(0.8)

        stub.collected_frames = 50
        hook()
        assert rb.offline_fraction == pytest.approx(0.4)

        stub.collected_frames = 100
        hook()
        assert rb.offline_fraction == pytest.approx(0.0)

        # clamps at 0 past anneal_frames
        stub.collected_frames = 200
        hook()
        assert rb.offline_fraction == 0.0


class TestOfflineToOnlineTrainer:
    def test_requires_offline_to_online_buffer(self):
        from torchrl.trainers.algorithms.offline_to_online import OfflineToOnlineTrainer

        plain = ReplayBuffer(storage=LazyTensorStorage(100))
        with pytest.raises(TypeError, match="OfflineToOnlineReplayBuffer"):
            OfflineToOnlineTrainer(
                collector=None,
                total_frames=1,
                frame_skip=1,
                optim_steps_per_batch=1,
                loss_module=None,
                replay_buffer=plain,
            )

    def test_hooks_drive_offline_online_flow(self):
        """The three hooks together grow the online buffer, keep the mixed batch
        flat, and anneal the offline fraction -- the data path the trainer runs,
        exercised without a loss so it is independent of the SAC/tensordict
        version."""
        from torchrl.trainers.algorithms.offline_to_online import (
            OfflineToOnlineAnnealHook,
            OfflineToOnlineReplayBufferHook,
        )

        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=_make_offline_buffer(),
            online_capacity=500,
            offline_fraction=0.5,
            batch_size=16,
        )
        rb_hook = OfflineToOnlineReplayBufferHook(rb, batch_size=16, device="cpu")
        stub = _StubTrainer()
        anneal = OfflineToOnlineAnnealHook(stub, rb, anneal_frames=100)

        for step in (20, 40, 60, 80, 100):
            rb_hook.extend(_make_online_data(20))  # pre_epoch
            sample = rb_hook.sample(None)  # process_optim_batch
            assert sample.batch_size == torch.Size([16])
            stub.collected_frames = step
            anneal()  # post_steps

        assert len(rb.online_buffer) == 100
        assert rb.offline_fraction == pytest.approx(0.0)

    @pytest.mark.skipif(
        not (_has_gym and _LOSS_RUNNABLE),
        reason="needs gym and a tensordict supporting to_module(preserve_module_state)",
    )
    def test_train_grows_online_and_anneals(self, tmp_path):
        import warnings

        from tensordict.nn import NormalParamExtractor, TensorDictModule
        from torch import nn

        from torchrl.collectors import Collector
        from torchrl.envs.libs.gym import GymEnv
        from torchrl.modules import MLP, ProbabilisticActor, TanhNormal, ValueOperator
        from torchrl.objectives import SACLoss, SoftUpdate
        from torchrl.trainers.algorithms.offline_to_online import OfflineToOnlineTrainer

        torch.manual_seed(0)
        env = GymEnv("Pendulum-v1")
        obs_dim = env.observation_spec["observation"].shape[-1]
        action_dim = env.action_spec.shape[-1]

        actor_net = nn.Sequential(
            MLP(in_features=obs_dim, out_features=2 * action_dim, num_cells=[32, 32]),
            NormalParamExtractor(),
        )
        actor_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=actor_module,
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
            MLP(in_features=obs_dim + action_dim, out_features=1, num_cells=[32, 32]),
            in_keys=["observation", "action"],
            out_keys=["state_action_value"],
        )

        loss = SACLoss(actor_network=actor, qvalue_network=qvalue)
        loss.make_value_estimator(gamma=0.99)
        target_updater = SoftUpdate(loss, eps=0.99)

        total_frames = 60
        frames_per_batch = 20
        collector = Collector(
            env,
            actor,
            frames_per_batch=frames_per_batch,
            total_frames=total_frames,
            init_random_frames=0,
        )

        # Seed an offline dataset from the same env so its keys match the online
        # transitions (no Minari/D4RL required).
        offline = ReplayBuffer(storage=LazyTensorStorage(200))
        offline.extend(env.rollout(50).reshape(-1).exclude("collector"))

        rb = OfflineToOnlineReplayBuffer(
            offline_dataset=offline,
            online_capacity=200,
            offline_fraction=0.5,
            batch_size=16,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            trainer = OfflineToOnlineTrainer(
                collector=collector,
                total_frames=total_frames,
                frame_skip=1,
                optim_steps_per_batch=1,
                loss_module=loss,
                replay_buffer=rb,
                anneal_frames=total_frames,
                optimizer=torch.optim.Adam(loss.parameters(), lr=1e-3),
                target_net_updater=target_updater,
                progress_bar=False,
                enable_logging=False,
            )
            # extend (pre_epoch), sample (process_optim_batch), anneal (post_steps)
            assert len(trainer._pre_epoch_ops) >= 1
            assert len(trainer._process_optim_batch_ops) >= 1
            assert len(trainer._post_steps_ops) >= 1

            trainer.train()

        # Online experience accumulated and the offline fraction annealed away.
        assert len(rb.online_buffer) > 0
        assert rb.offline_fraction < 0.5


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
