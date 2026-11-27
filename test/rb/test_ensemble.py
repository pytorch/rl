# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import functools
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
from unittest import mock

import numpy as np
import pytest
import torch
from _rb_common import _has_gym, _has_h5py, _has_tv
from tensordict import (
    assert_allclose_td,
    is_tensor_collection,
    tensorclass,
    TensorDict,
    TensorDictBase,
)
from torch.utils._pytree import tree_flatten

from torchrl.checkpoint import Checkpoint
from torchrl.collectors import Collector
from torchrl.collectors.utils import split_trajectories
from torchrl.data import (
    FlatStorageCheckpointer,
    MultiStep,
    NestedStorageCheckpointer,
    PrioritizedReplayBuffer,
    ReplayBuffer,
    ReplayBufferEnsemble,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.checkpointers import H5StorageCheckpointer
from torchrl.data.replay_buffers.samplers import (
    PrioritizedSampler,
    PrioritizedSliceSampler,
    RandomSampler,
    SamplerEnsemble,
    SamplerWithoutReplacement,
    SliceSampler,
    SliceSamplerWithoutReplacement,
    StalenessAwareSampler,
    StreamingSliceSampler,
)
from torchrl.data.replay_buffers.storages import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ListStorage,
    StorageEnsemble,
    TensorStorage,
)
from torchrl.data.replay_buffers.utils import tree_iter
from torchrl.data.replay_buffers.writers import (
    RoundRobinWriter,
    TensorDictMaxValueWriter,
    TensorDictRoundRobinWriter,
    WriterEnsemble,
)
from torchrl.envs import GymEnv, SerialEnv
from torchrl.envs.transforms.transforms import (
    Compose,
    RenameTransform,
    Resize,
    StepCounter,
    ToTensorImage,
)
from torchrl.modules import RandomPolicy
from torchrl.testing import CARTPOLE_VERSIONED, get_default_devices
from torchrl.testing.mocking_classes import CountingEnv


class TestEnsemble:
    def _make_data(self, data_type):
        if data_type is torch.Tensor:
            return torch.ones(90)
        if data_type is TensorDict:
            return TensorDict(
                {
                    "root": torch.arange(90),
                    "nested": TensorDict(
                        {"data": torch.arange(180).view(90, 2)}, batch_size=[90, 2]
                    ),
                },
                batch_size=[90],
            )
        raise NotImplementedError

    @staticmethod
    def _make_routed_member(max_size=8, *, sampler=None):
        if sampler is None:
            sampler = RandomSampler()
        return TensorDictReplayBuffer(
            storage=LazyTensorStorage(max_size),
            sampler=sampler,
            writer=TensorDictRoundRobinWriter(track_generations=True),
        )

    def test_routed_write_and_conditional_update(self):
        members = [self._make_routed_member(2) for _ in range(2)]
        rb = ReplayBufferEnsemble(*members, routing_key=("collector", "env_id"))
        data = TensorDict(
            {
                "value": torch.tensor([0, 10, 1, 11]),
                ("collector", "env_id"): torch.tensor([0, 1, 0, 1]),
            },
            [4],
        )

        metadata = rb.extend(data)

        assert metadata["buffer_ids"].tolist() == [0, 1, 0, 1]
        assert metadata["index"].tolist() == [0, 0, 1, 1]
        assert metadata["index_generation"].tolist() == [0, 0, 0, 0]
        assert members[0][:]["value"].tolist() == [0, 1]
        assert members[1][:]["value"].tolist() == [10, 11]

        rb.extend(
            TensorDict(
                {
                    "value": torch.tensor([2, 3]),
                    ("collector", "env_id"): torch.zeros(2, dtype=torch.long),
                },
                [2],
            )
        )
        result = rb.update_if_present(
            index=metadata.select("buffer_ids", "index"),
            generation=metadata["index_generation"],
            patch={"value": data["value"] + 100},
        )

        assert result.updated.tolist() == [False, True, False, True]
        assert members[0][:]["value"].tolist() == [2, 3]
        assert members[1][:]["value"].tolist() == [110, 111]
        assert rb.stats() == {
            "size": 4,
            "storage_size": 4,
            "sampleable_size": 4,
            "write_count": 6,
            "sample_calls": 0,
            "samples_returned": 0,
            "prefetch_queue_size": 0,
            "initialized": True,
            "capacity": 4,
            "utilization": 1.0,
            "num_buffers": 2,
        }
        restored_members = [self._make_routed_member(2) for _ in range(2)]
        restored = ReplayBufferEnsemble(
            *restored_members, routing_key=("collector", "env_id")
        )
        restored.load_state_dict(rb.state_dict())
        assert restored_members[0][:]["value"].tolist() == [2, 3]
        assert restored_members[1][:]["value"].tolist() == [110, 111]
        assert restored.stats()["write_count"] == 6

    def test_stats_size_tracks_sampleable_records(self):
        members = [
            ReplayBuffer(
                storage=LazyTensorStorage(4),
                batch_size=2,
                consume_after_n_samples=1,
            )
            for _ in range(2)
        ]
        for member in members:
            member.extend(torch.arange(4))
        members[0].sample()
        rb = ReplayBufferEnsemble(*members)

        stats = rb.stats()
        assert stats["size"] == stats["sampleable_size"] == 6
        assert stats["storage_size"] == 8

    @pytest.mark.parametrize(
        "device",
        [
            "cpu",
            pytest.param(
                "cuda",
                marks=[
                    pytest.mark.gpu,
                    pytest.mark.skipif(
                        not torch.cuda.is_available(), reason="requires CUDA"
                    ),
                ],
            ),
        ],
    )
    @pytest.mark.parametrize("submit", [False, True])
    @pytest.mark.parametrize("tensordict_patch", [False, True])
    def test_conditional_update_groups_records_by_member(
        self, device, submit, tensordict_patch
    ):
        members = [self._make_routed_member(3) for _ in range(4)]
        rb = ReplayBufferEnsemble(*members, routing_key="env_id")
        env_id = torch.tensor([2, 0, 3, 1, 1, 3, 0, 2, 0, 1, 2, 3])
        metadata = rb.extend(
            TensorDict({"value": torch.zeros(12), "env_id": env_id}, [12])
        )
        # Reusing one slot of member 3 makes a handle stale.
        rb.extend(
            TensorDict({"value": torch.full((1,), -1.0), "env_id": env_id[2:3]}, [1])
        )
        # Records arrive in a shuffled member order; member 0 gets none.
        rows = torch.tensor([5, 0, 3, 11, 7, 4, 9, 2])
        handles = metadata.select("buffer_ids", "index")[rows]
        generation = metadata["index_generation"][rows]
        patch = {"value": (rows + 100).float().to(device)}
        if tensordict_patch:
            handles = handles.to(device)
            generation = generation.to(device)
            patch = TensorDict(patch, batch_size=rows.shape)
        try:
            if submit:
                result = rb.submit_update_if_present(
                    index=handles, generation=generation, patch=patch
                ).result(timeout=5)
            else:
                result = rb.update_if_present(
                    index=handles, generation=generation, patch=patch
                )
        finally:
            rb.shutdown()

        handles = handles.cpu()
        stale = handles["buffer_ids"] == 3
        stale &= handles["index"] == metadata["index"][2]
        assert stale.sum() == 1
        assert result.updated.tolist() == (~stale).tolist()
        expected = torch.zeros(12)
        expected[rows] = (rows + 100).float()
        expected[2] = -1.0
        for member_id, member in enumerate(members):
            positions = (env_id == member_id).nonzero().reshape(-1)
            local = metadata["index"][positions]
            stored = member[:]["value"][local]
            assert stored.tolist() == expected[positions].tolist(), member_id

    @pytest.mark.parametrize("empty_member", [False, True])
    def test_checkpoint_drains_updates_and_restores_prefetch(
        self, monkeypatch, tmp_path, empty_member
    ):
        buffers = [
            ReplayBufferEnsemble(
                *[self._make_routed_member(8) for _ in range(2)],
                routing_key=("collector", "env_id"),
                p="sampleable",
                batch_size=4,
                prefetch=1,
                generator=torch.Generator().manual_seed(0),
            )
            for _ in range(2)
        ]
        replay, restored = buffers
        entered = threading.Event()
        release = threading.Event()
        original_update = replay.update_if_present

        def delayed_update(**kwargs):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Checkpoint test did not release the replay update.")
            return original_update(**kwargs)

        try:
            metadata = replay.extend(
                TensorDict(
                    {
                        "value": torch.arange(8),
                        ("collector", "env_id"): torch.zeros(8, dtype=torch.long)
                        if empty_member
                        else torch.arange(8) % 2,
                    },
                    [8],
                )
            )
            replay.sample()
            monkeypatch.setattr(replay, "update_if_present", delayed_update)
            update = replay.submit_update_if_present(
                index=metadata.select("buffer_ids", "index"),
                generation=metadata["index_generation"],
                patch={"value": torch.full((8,), 42)},
            )
            assert entered.wait(5)
            with ThreadPoolExecutor(max_workers=1) as executor:
                saved = executor.submit(
                    Checkpoint(replay=replay).save, tmp_path / "saved"
                )
                with pytest.raises(FuturesTimeoutError):
                    saved.result(timeout=0.05)
                release.set()
                saved.result(timeout=10)
            assert update.result().updated.all()
            Checkpoint(replay=restored).load(tmp_path / "saved")
            assert restored.stats()["write_count"] == 8
            assert len(restored[1]) == (0 if empty_member else 4)
            assert (restored[0][:]["value"] == 42).all()
            for _ in range(3):
                assert_allclose_td(replay.sample(), restored.sample())
            # Loading replay must not retain writable views into the checkpoint.
            restored[0].extend(
                TensorDict(
                    {
                        "value": torch.zeros(8, dtype=torch.long),
                        ("collector", "env_id"): torch.zeros(8, dtype=torch.long),
                    },
                    [8],
                )
            )
            Checkpoint(replay=restored).load(tmp_path / "saved")
            assert (restored[0][:]["value"] == 42).all()
        finally:
            release.set()
            for buffer in buffers:
                buffer.shutdown()

    @pytest.mark.parametrize("sampler_type", [SliceSampler, StreamingSliceSampler])
    def test_end_streams_orders_updates_and_resets_windows(
        self, sampler_type, monkeypatch
    ):
        done_key, terminal_key, cut_key = (
            ("flags", "end"),
            ("flags", "terminal"),
            ("flags", "cut"),
        )
        members = [
            TensorDictReplayBuffer(
                storage=LazyTensorStorage(9),
                writer=TensorDictRoundRobinWriter(track_generations=True),
                sampler=sampler_type(slice_len=3, end_key=done_key, cache_values=True),
                batch_size=3,
            )
            for _ in range(3)
        ]
        replay = ReplayBufferEnsemble(*members, routing_key="stream")
        data = TensorDict(
            {
                "stream": torch.zeros(11, dtype=torch.long),
                "value": torch.arange(11),
                done_key: torch.zeros(11, 1, dtype=torch.bool),
                terminal_key: torch.zeros(11, 1, dtype=torch.bool),
                cut_key: torch.zeros(11, 1, dtype=torch.bool),
            },
            [11],
        )
        replay.extend(data)
        terminal = data[:3].clone()
        terminal["stream"] = torch.ones(3, dtype=torch.long)
        terminal[done_key][-1] = True
        terminal[terminal_key][-1] = True
        replay.extend(terminal)
        # Populate the uniform boundary cache; keep an unfinished fresh window.
        members[0].sample()
        generation = members[0].writer.generations_of(torch.arange(9)).clone()
        writes = replay.stats()["write_count"]
        entered, release = threading.Event(), threading.Event()
        original_update = replay.update_if_present

        def delayed_update(**kwargs):
            entered.set()
            if not release.wait(10):
                raise TimeoutError("Stream-finalization update was not released.")
            return original_update(**kwargs)

        monkeypatch.setattr(replay, "update_if_present", delayed_update)
        update = replay.submit_update_if_present(
            index=TensorDict(
                {"buffer_ids": torch.tensor([0]), "index": torch.tensor([1])}, [1]
            ),
            generation=generation[1:2],
            patch={done_key: torch.zeros(1, 1, dtype=torch.bool)},
        )
        try:
            assert entered.wait(5)
            with ThreadPoolExecutor(max_workers=1) as executor:
                ended = executor.submit(
                    replay.end_streams,
                    end_key=done_key,
                    terminated_key=terminal_key,
                    truncated_key=cut_key,
                )
                with pytest.raises(FuturesTimeoutError):
                    ended.result(timeout=0.05)
                release.set()
                ended.result(timeout=10)
            assert update.result().updated.all()
            assert replay.stats()["write_count"] == writes
            torch.testing.assert_close(
                members[0].writer.generations_of(torch.arange(9)), generation
            )
            assert members[0][1][done_key].all() and members[0][1][cut_key].all()
            assert not members[0][1][terminal_key].any()
            assert (
                members[1][2][terminal_key].all() and not members[1][2][cut_key].any()
            )
            assert len(members[2]) == 0
            new = data[:3].clone()
            new["value"] = torch.arange(100, 103)
            replay.extend(new)
            for _ in range(12):
                values = members[0].sample()["value"]
                assert (values < 100).all() or (values >= 100).all()
                assert (values.diff() == 1).all()
        finally:
            release.set()
            replay.shutdown()

    @pytest.mark.parametrize("sampler_type", [SliceSampler, StreamingSliceSampler])
    def test_end_streams_with_per_stream_trajectory_key(self, sampler_type):
        done_key = ("next", "done")

        def make_member():
            return TensorDictReplayBuffer(
                storage=LazyTensorStorage(8),
                writer=TensorDictRoundRobinWriter(track_generations=True),
                sampler=sampler_type(
                    slice_len=2, end_key=done_key, traj_key="env_index"
                ),
                batch_size=2,
            )

        def records(env_index):
            return TensorDict(
                {
                    "env_index": torch.as_tensor(env_index),
                    "value": torch.arange(len(env_index)),
                    done_key: torch.zeros(len(env_index), 1, dtype=torch.bool),
                },
                [len(env_index)],
            )

        # One environment per member: the stream is the trajectory and its tail closes.
        members = [make_member() for _ in range(2)]
        replay = ReplayBufferEnsemble(*members, routing_key="env_index")
        try:
            replay.extend(records([0, 0, 0]))
            replay.extend(records([1, 1]))
            replay.end_streams()
            assert members[0][:][done_key].flatten().tolist() == [False, False, True]
            assert members[1][:][done_key].flatten().tolist() == [False, True]
        finally:
            replay.shutdown()

        # Distinct ids within a member may be reused by restarted producers.
        mixed = make_member()
        mixed.extend(records([0, 1]))
        replay = ReplayBufferEnsemble(mixed)
        try:
            with pytest.raises(RuntimeError, match="same trajectory id"):
                replay.end_streams()
            assert not mixed[:][done_key].any()
        finally:
            replay.shutdown()

    @pytest.mark.parametrize("default_device", ["cpu", "meta"])
    def test_routing_dim_preserves_member_order(self, default_device):
        members = [self._make_routed_member() for _ in range(2)]
        rb = ReplayBufferEnsemble(*members, routing_dim=1)
        data = TensorDict(
            {"value": torch.tensor([[0, 10], [1, 11]])}, batch_size=[2, 2]
        )

        with torch.device(default_device):
            metadata = rb.extend(data)

        assert metadata.batch_size == (4,)
        assert metadata["buffer_ids"].tolist() == [0, 1, 0, 1]
        assert metadata["index"].tolist() == [0, 0, 1, 1]
        assert metadata["index_generation"].tolist() == [0, 0, 0, 0]
        assert members[0][:]["value"].tolist() == [0, 1]
        assert members[1][:]["value"].tolist() == [10, 11]

    @pytest.mark.parametrize("default_device", ["cpu", "meta"])
    def test_sampleable_routing_groups_member_samples(self, default_device):
        lengths = (1, 4, 8)
        samplers = [SliceSampler(slice_len=2) for _ in lengths]
        members = [
            self._make_routed_member(16, sampler=sampler) for sampler in samplers
        ]
        for member_id, (member, length) in enumerate(zip(members, lengths)):
            member.extend(
                TensorDict(
                    {
                        "value": torch.arange(length),
                        "episode": torch.full((length,), member_id),
                    },
                    [length],
                )
            )
            sampler = samplers[member_id]
            sampler.sample = mock.MagicMock(wraps=sampler.sample)
        generator = torch.Generator().manual_seed(0)
        rb = ReplayBufferEnsemble(
            *members,
            p="sampleable",
            num_buffer_sampled=1_000,
            generator=generator,
        )

        with torch.device(default_device):
            index, _ = rb.sampler.sample(rb.storage, 2_000)
        counts = torch.bincount(index["buffer_ids"], minlength=3)

        assert counts[0] == 0
        assert 0.25 < counts[1] / (counts[1] + counts[2]) < 0.35
        assert samplers[0].sample.call_count == 0
        assert samplers[1].sample.call_count == 1
        assert samplers[2].sample.call_count == 1
        assert rb.can_sample(2_000)

    def test_routed_write_configuration_validation(self):
        members = [self._make_routed_member() for _ in range(2)]
        with pytest.raises(ValueError, match="mutually exclusive"):
            ReplayBufferEnsemble(*members, routing_key="member", routing_dim=0)
        rb = ReplayBufferEnsemble(*members)
        with pytest.raises(RuntimeError, match="writes are disabled"):
            rb.extend(TensorDict({"value": torch.arange(2)}, [2]))

    @staticmethod
    def _routed_values(values, route):
        return TensorDict(
            {"value": torch.as_tensor(values), "route": torch.as_tensor(route)},
            [len(values)],
        )

    def _make_demo_online(self, demo_size=4, online_size=4, **kwargs):
        demo = self._make_routed_member(
            8, sampler=PrioritizedSampler(8, alpha=1.0, beta=0.5, eps=1e-8)
        )
        online = self._make_routed_member(8)
        rb = ReplayBufferEnsemble(demo, online, routing_key="route", **kwargs)
        rb.extend(
            self._routed_values(
                torch.arange(demo_size + online_size),
                [0] * demo_size + [1] * online_size,
            )
        )
        return demo, online, rb

    def test_mixed_member_samplers_fill_neutral_priority_weight(self):
        demo, online, rb = self._make_demo_online(sample_from_all=True, batch_size=64)
        demo.update_priority(torch.arange(4), torch.tensor([1.0, 100.0, 100.0, 100.0]))

        sample = rb.sample()

        from_demo = sample["index", "buffer_ids"] == 0
        weight = sample["priority_weight"]
        local_index = sample["index", "index"]
        assert from_demo.any() and (~from_demo).any()
        assert (weight[~from_demo] == 1).all()
        expected = torch.where(local_index[from_demo] == 0, 1.0, 0.1)
        torch.testing.assert_close(weight[from_demo], expected, atol=1e-4, rtol=0)

    def test_mixed_member_samplers_reject_other_missing_info(self):
        stale = self._make_routed_member(8, sampler=StalenessAwareSampler())
        plain = self._make_routed_member(8)
        rb = ReplayBufferEnsemble(
            stale, plain, routing_key="route", sample_from_all=True, batch_size=4
        )
        data = self._routed_values(torch.arange(4), [0, 0, 1, 1])
        data.set("policy_version", torch.zeros(4, dtype=torch.long))
        rb.extend(data)

        with pytest.raises(KeyError, match=r"staleness.*member\(s\) \[1\]"):
            rb.sample()

    def test_mixed_member_index_layouts_raise(self):
        sliced = self._make_routed_member(8, sampler=SliceSampler(slice_len=2))
        plain = self._make_routed_member(8)
        rb = ReplayBufferEnsemble(
            sliced, plain, routing_key="route", sample_from_all=True, batch_size=4
        )
        data = self._routed_values(torch.arange(8), [0, 0, 0, 0, 1, 1, 1, 1])
        data.set("episode", torch.zeros(8, dtype=torch.long))
        rb.extend(data)

        with pytest.raises(ValueError, match="index layouts"):
            rb.sample()

    def test_update_tensordict_priority_routes_to_members(self):
        demo, online, rb = self._make_demo_online(sample_from_all=True, batch_size=16)
        sample = rb.sample()
        local_index = sample["index", "index"]
        from_demo = sample["index", "buffer_ids"] == 0
        target = local_index[from_demo][0]
        hit = from_demo & (local_index == target)
        sample.set("td_error", torch.where(hit, 1e4, 1e-4))

        rb.update_tensordict_priority(sample)

        demo_index = demo.sample(256, return_info=True)[1]["index"]
        assert (demo_index == target).float().mean() > 0.95

    def test_update_priority_takes_routed_index(self):
        demo, online, rb = self._make_demo_online()
        index = TensorDict(
            {"buffer_ids": torch.tensor([0, 1]), "index": torch.tensor([3, 0])}, [2]
        )

        rb.update_priority(index, torch.tensor([1e4, 1e4]))

        demo_index = demo.sample(256, return_info=True)[1]["index"]
        assert (demo_index == 3).float().mean() > 0.95

    def test_update_priority_plain_index_keeps_previous_behavior(self):
        demo, online, rb = self._make_demo_online()
        before = demo.sampler.state_dict()

        with pytest.warns(UserWarning, match="update_priority"):
            rb.update_priority(torch.tensor([0, 1]), torch.tensor([1e4, 1e4]))

        after = demo.sampler.state_dict()
        assert all(
            torch.equal(before[key], after[key])
            for key in before
            if isinstance(before[key], torch.Tensor)
        )

    def test_fractions_exact_quotas_flat_output(self):
        demo, online, rb = self._make_demo_online(fractions=[0.25, 0.75], batch_size=8)

        for _ in range(5):
            sample = rb.sample()
            assert sample.batch_size == (8,)
            assert sample["index", "buffer_ids"].tolist() == [0, 0, 1, 1, 1, 1, 1, 1]
            stored = torch.where(
                sample["index", "buffer_ids"] == 0,
                sample["index", "index"],
                sample["index", "index"] + 4,
            )
            assert sample["value"].tolist() == stored.tolist()
            assert (sample["priority_weight"][2:] == 1).all()

    @pytest.mark.parametrize(
        "fractions,batch_size,expected",
        [
            ([0.5, 0.5], 3, [2, 1]),
            ([0.3, 0.7], 256, [77, 179]),
            ([1.0, 0.0], 5, [5, 0]),
        ],
    )
    def test_fractions_largest_remainder_counts(self, fractions, batch_size, expected):
        demo, online, rb = self._make_demo_online(fractions=fractions)

        sample = rb.sample(batch_size)

        counts = torch.bincount(sample["index", "buffer_ids"], minlength=2)
        assert counts.tolist() == expected

    def test_fractions_empty_member_blocks_sampling(self):
        demo, online, rb = self._make_demo_online(
            online_size=0, fractions=[0.5, 0.5], batch_size=4
        )

        assert not rb.can_sample(4)
        with pytest.raises(RuntimeError, match=r"member\(s\) \[1\]"):
            rb.sample()
        rb.fractions = [1.0, 0.0]
        assert rb.can_sample(4)
        assert rb.sample()["index", "buffer_ids"].tolist() == [0, 0, 0, 0]

    def test_fractions_member_subset_renormalizes(self):
        demo, online, rb = self._make_demo_online(fractions=[0.25, 0.75], batch_size=4)

        subset = rb[1:]

        assert subset.fractions.tolist() == [1.0]
        assert (subset.sample(4)["value"] >= 4).all()

    def test_fractions_validation(self):
        members = [self._make_routed_member() for _ in range(2)]
        with pytest.raises(ValueError, match="fractions"):
            ReplayBufferEnsemble(*members, fractions=[0.5, 0.5], p=[0.5, 0.5])
        with pytest.raises(ValueError, match="fractions"):
            ReplayBufferEnsemble(*members, fractions=[0.5, 0.5], sample_from_all=True)
        with pytest.raises(ValueError, match="sum to 1"):
            ReplayBufferEnsemble(*members, fractions=[0.5, 0.6])
        with pytest.raises(ValueError, match="non-negative"):
            ReplayBufferEnsemble(*members, fractions=[1.5, -0.5])
        with pytest.raises(ValueError, match="one fraction per member"):
            ReplayBufferEnsemble(*members, fractions=[1.0])
        weighted = ReplayBufferEnsemble(*members, p=[0.9, 0.1])
        with pytest.raises(ValueError, match="exclusive"):
            weighted.fractions = [0.5, 0.5]
        assert weighted.fractions is None

    def test_fractions_slice_members_count_slices(self):
        members = [
            self._make_routed_member(
                16, sampler=SliceSampler(slice_len=2, traj_key="episode")
            )
            for _ in range(2)
        ]
        rb = ReplayBufferEnsemble(*members, routing_key="route", fractions=[0.5, 0.5])
        data = self._routed_values(torch.arange(16), [0] * 8 + [1] * 8)
        data.set("episode", torch.arange(16) // 4)
        rb.extend(data)

        sample = rb.sample(8)

        assert sample["index", "buffer_ids"].tolist() == [0, 0, 0, 0, 1, 1, 1, 1]
        pairs = sample["value"].reshape(4, 2)
        assert ((pairs[:, 1] - pairs[:, 0]) == 1).all()
        assert torch.bincount(rb.sample(6)["index", "buffer_ids"]).tolist() == [4, 2]
        with pytest.raises(ValueError, match="multiple of the slice length"):
            rb.sample(7)

    def test_fractions_flat_priority_round_trip(self):
        demo, online, rb = self._make_demo_online(fractions=[0.5, 0.5], batch_size=8)
        sample = rb.sample()
        from_demo = sample["index", "buffer_ids"] == 0
        target = sample["index", "index"][from_demo][0]
        hit = from_demo & (sample["index", "index"] == target)
        sample.set("td_error", torch.where(hit, 1e4, 1e-4))

        rb.update_tensordict_priority(sample)

        demo_index = demo.sample(256, return_info=True)[1]["index"]
        assert (demo_index == target).float().mean() > 0.95

    def _make_mask_ensemble(self, **kwargs):
        members = [self._make_routed_member(8) for _ in range(2)]
        rb = ReplayBufferEnsemble(
            *members, routing_key=("collector", "route"), routing_mode="mask", **kwargs
        )
        return members, rb

    def test_mask_routing_writes_every_copy(self):
        members, rb = self._make_mask_ensemble()
        data = TensorDict(
            {
                "value": torch.tensor([0, 1, 2]),
                ("collector", "route"): torch.tensor(
                    [[True, True], [False, True], [True, True]]
                ),
            },
            [3],
        )

        addresses = rb.extend(data)

        assert addresses.batch_size == (3, 2)
        assert addresses["present"].tolist() == [
            [True, True],
            [False, True],
            [True, True],
        ]
        assert addresses["buffer_ids"].tolist() == [[0, 1], [0, 1], [0, 1]]
        assert addresses["index"].tolist() == [[0, 0], [-1, 1], [1, 2]]
        assert addresses["index_generation"].tolist() == [[0, 0], [-1, 0], [0, 0]]
        assert members[0][:]["value"].tolist() == [0, 2]
        assert members[1][:]["value"].tolist() == [0, 1, 2]
        assert ("collector", "route") not in members[1][:].keys(True)

    def test_mask_routing_update_if_present_patches_every_copy(self):
        members, rb = self._make_mask_ensemble()
        data = TensorDict(
            {
                "value": torch.tensor([0, 1, 2]),
                ("collector", "route"): torch.tensor(
                    [[True, True], [False, True], [True, False]]
                ),
            },
            [3],
        )
        addresses = rb.extend(data)

        result = rb.update_if_present(
            index=addresses.select("buffer_ids", "index", "present"),
            generation=addresses["index_generation"],
            patch={"value": torch.tensor([100, 101, 102])},
        )

        assert result.updated.tolist() == [[True, True], [False, True], [True, False]]
        assert members[0][:]["value"].tolist() == [100, 102]
        assert members[1][:]["value"].tolist() == [100, 101]

    def test_mask_routing_validation(self):
        members = [self._make_routed_member() for _ in range(2)]
        with pytest.raises(ValueError, match="routing_mode"):
            ReplayBufferEnsemble(*members, routing_key="route", routing_mode="ids")
        with pytest.raises(ValueError, match="routing_key"):
            ReplayBufferEnsemble(*members, routing_dim=0, routing_mode="mask")
        rb = ReplayBufferEnsemble(*members, routing_key="route", routing_mode="mask")
        with pytest.raises(TypeError, match="boolean"):
            rb.extend(self._routed_values([0, 1], [[1, 1], [0, 1]]))
        with pytest.raises(ValueError, match=r"\[\*B, 2\]"):
            rb.extend(self._routed_values([0, 1], [[True], [False]]))

    def test_mask_routing_fragmented_slices_never_cross_gaps(self):
        def fragmented():
            return SliceSampler(
                slice_len=2,
                traj_key=("collector", "traj_ids"),
                step_key="step_count",
                fragmented=True,
            )

        members = [self._make_routed_member(64, sampler=fragmented()) for _ in range(2)]
        rb = ReplayBufferEnsemble(
            *members, routing_key="route", routing_mode="mask", fractions=[0.5, 0.5]
        )

        def steps(values, demo_steps):
            values = torch.as_tensor(values)
            demo = torch.tensor([int(value) in demo_steps for value in values])
            return TensorDict(
                {
                    "step_count": values.unsqueeze(-1),
                    ("collector", "traj_ids"): torch.full_like(values, 7),
                    "route": torch.stack([demo, torch.ones_like(demo)], -1),
                },
                [values.numel()],
            )

        rb.extend(steps(range(23), {0, 1, 2, 3, 10, 11, 12, 20, 21, 22}))
        rb.extend(steps(range(23, 26), {23, 24, 25}))

        demo_pairs = []
        for _ in range(50):
            sample = rb.sample(16)
            from_demo = sample["index", "buffer_ids"] == 0
            demo_pairs.append(sample["step_count"][from_demo].reshape(-1, 2))
        demo_pairs = torch.cat(demo_pairs)
        assert ((demo_pairs[:, 1] - demo_pairs[:, 0]) == 1).all()
        assert ((demo_pairs[:, 0] == 22) & (demo_pairs[:, 1] == 23)).any()

    def test_update_tensordict_priority_plain_members(self):
        storages = StorageEnsemble(LazyTensorStorage(8), LazyTensorStorage(8))
        for member, storage in enumerate(storages._storages):
            storage.set(
                torch.arange(4),
                TensorDict({"value": torch.arange(4) + 4 * member}, [4]),
            )
        samplers = SamplerEnsemble(
            PrioritizedSampler(8, alpha=1.0, beta=0.5),
            RandomSampler(),
            fractions=[0.5, 0.5],
        )
        rb = ReplayBufferEnsemble(
            storages=storages,
            samplers=samplers,
            writers=WriterEnsemble(RoundRobinWriter(), RoundRobinWriter()),
            batch_size=8,
        )
        sample = rb.sample()
        from_demo = sample["index", "buffer_ids"] == 0
        target = sample["index", "index"][from_demo][0]
        hit = from_demo & (sample["index", "index"] == target)
        sample.set("td_error", torch.where(hit, 1e4, 1e-4))

        rb.update_tensordict_priority(sample)

        member_index, _ = samplers[0].sample(storages._storages[0], 256)
        assert (member_index == target).float().mean() > 0.95

    def test_mask_patch_is_per_record_when_features_match_members(self):
        members, rb = self._make_mask_ensemble()
        data = TensorDict(
            {
                "action": torch.zeros(3, 2),
                ("collector", "route"): torch.tensor(
                    [[True, True], [False, True], [True, True]]
                ),
            },
            [3],
        )
        addresses = rb.extend(data)

        rb.update_if_present(
            index=addresses.select("buffer_ids", "index", "present"),
            generation=addresses["index_generation"],
            patch={"action": torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])},
        )

        assert members[0][:]["action"].tolist() == [[1.0, 2.0], [5.0, 6.0]]
        assert members[1][:]["action"].tolist() == [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]

    def test_mask_routing_without_copies_keeps_generation(self):
        members, rb = self._make_mask_ensemble()
        addresses = rb.extend(
            TensorDict(
                {
                    "value": torch.tensor([0, 1]),
                    ("collector", "route"): torch.zeros(2, 2, dtype=torch.bool),
                },
                [2],
            )
        )

        assert addresses["index_generation"].tolist() == [[-1, -1], [-1, -1]]
        assert not addresses["present"].any()

    def test_fractions_follow_slice_len_changes(self):
        samplers = [SliceSampler(slice_len=2, traj_key="episode") for _ in range(2)]
        members = [
            self._make_routed_member(32, sampler=sampler) for sampler in samplers
        ]
        rb = ReplayBufferEnsemble(*members, routing_key="route", fractions=[0.5, 0.5])
        data = self._routed_values(torch.arange(32), [0] * 16 + [1] * 16)
        data.set("episode", torch.arange(32) // 8)
        rb.extend(data)
        assert torch.bincount(rb.sample(12)["index", "buffer_ids"]).tolist() == [6, 6]

        for sampler in samplers:
            sampler.slice_len = 4

        assert torch.bincount(rb.sample(12)["index", "buffer_ids"]).tolist() == [8, 4]

    def test_fractions_getter_returns_a_copy(self):
        demo, online, rb = self._make_demo_online(fractions=[0.5, 0.5], batch_size=8)

        rb.fractions[:] = torch.tensor([0.25, 0.75])

        assert rb.sample()["index", "buffer_ids"].tolist() == [0] * 4 + [1] * 4

    def test_fractions_with_tensor_members(self):
        members = [ReplayBuffer(storage=ListStorage(8)) for _ in range(2)]
        members[0].extend(torch.arange(4))
        members[1].extend(torch.arange(4) + 10)
        rb = ReplayBufferEnsemble(*members, fractions=[0.25, 0.75])

        sample = rb.sample(4)

        assert sample.shape == (4,)
        assert (sample[:1] < 4).all() and (sample[1:] >= 10).all()

    def test_fractions_reject_batch_time_slices(self):
        members = [
            self._make_routed_member(
                16,
                sampler=SliceSampler(
                    slice_len=2, traj_key="episode", output_layout="batch_time"
                ),
            )
            for _ in range(2)
        ]
        with pytest.raises(ValueError, match="flat sampler output"):
            ReplayBufferEnsemble(*members, fractions=[0.5, 0.5]).sampler.quota_counts(8)

    def test_fractions_state_dict_round_trip(self):
        demo, online, rb = self._make_demo_online(fractions=[0.5, 0.5], batch_size=8)
        rb.fractions = [0.25, 0.75]
        state = rb.state_dict()

        restored_demo, restored_online, restored = self._make_demo_online(
            fractions=[0.5, 0.5], batch_size=8
        )
        restored.load_state_dict(state)

        assert restored.fractions.tolist() == [0.25, 0.75]

    @pytest.mark.parametrize("checkpoint", ["state_dict", "dumps"])
    def test_loading_a_checkpoint_without_fractions_clears_them(
        self, checkpoint, tmp_path
    ):
        demo, online, rb = self._make_demo_online(batch_size=8)
        restored_demo, restored_online, restored = self._make_demo_online(
            fractions=[0.5, 0.5], batch_size=8
        )

        if checkpoint == "state_dict":
            restored.load_state_dict(rb.state_dict())
        else:
            rb.dumps(tmp_path)
            restored.loads(tmp_path)

        assert restored.fractions is None

    def test_fractions_dumps_round_trip(self, tmp_path):
        demo, online, rb = self._make_demo_online(fractions=[0.5, 0.5], batch_size=8)
        rb.fractions = [0.25, 0.75]
        rb.dumps(tmp_path)

        restored_demo, restored_online, restored = self._make_demo_online(
            fractions=[0.5, 0.5], batch_size=8
        )
        restored.loads(tmp_path)

        assert restored.fractions.tolist() == [0.25, 0.75]

    def test_fractions_change_during_prefetch_keeps_labels(self):
        members = [
            TensorDictReplayBuffer(storage=LazyTensorStorage(500)) for _ in range(2)
        ]
        for member_id, member in enumerate(members):
            member.extend(TensorDict({"member": torch.full((500,), member_id)}, [500]))
        rb = ReplayBufferEnsemble(
            *members, fractions=[0.1, 0.9], batch_size=1024, prefetch=2
        )

        for step in range(100):
            rb.fractions = [0.9, 0.1] if step % 2 else [0.1, 0.9]
            sample = rb.sample()
            assert (sample["member"] == sample["index", "buffer_ids"]).all()

    def test_fractions_reject_non_strict_slices(self):
        members = [
            self._make_routed_member(
                16,
                sampler=SliceSampler(
                    slice_len=4, traj_key="episode", strict_length=False
                ),
            )
            for _ in range(2)
        ]
        with pytest.raises(ValueError, match="strict_length"):
            ReplayBufferEnsemble(*members, fractions=[0.5, 0.5]).sampler.quota_counts(8)

    def test_fractions_stats_count_multidim_members(self):
        members = [
            TensorDictReplayBuffer(storage=LazyTensorStorage(20, ndim=2))
            for _ in range(2)
        ]
        for member in members:
            member.extend(TensorDict({"obs": torch.arange(10).view(2, 5)}, [2, 5]))
        rb = ReplayBufferEnsemble(*members, fractions=[0.5, 0.5])

        rb.sample(6)

        assert rb.stats()["samples_returned"] == 6

    def test_update_priority_one_value_per_masked_record(self):
        demo, online = (
            TensorDictReplayBuffer(
                storage=LazyTensorStorage(5),
                sampler=PrioritizedSampler(5, alpha=1.0, beta=1.0),
            )
            for _ in range(2)
        )
        rb = ReplayBufferEnsemble(
            demo, online, routing_key="route", routing_mode="mask"
        )
        meta = rb.extend(TensorDict({"route": torch.ones(3, 2, dtype=torch.bool)}, [3]))

        rb.update_priority(meta, torch.tensor([1.0, 1.0, 1e4]))

        for member in (demo, online):
            index = member.sample(256, return_info=True)[1]["index"]
            assert (index == 2).float().mean() > 0.95

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
    def test_fractions_mixed_index_devices(self):
        demo = TensorDictReplayBuffer(
            storage=LazyTensorStorage(8, device="cuda"),
            sampler=PrioritizedSampler(8, alpha=1.0, beta=0.5),
        )
        online = TensorDictReplayBuffer(storage=LazyTensorStorage(8, device="cuda"))
        demo.extend(TensorDict({"value": torch.arange(4)}, [4], device="cuda"))
        online.extend(TensorDict({"value": torch.arange(4) + 4}, [4], device="cuda"))
        rb = ReplayBufferEnsemble(demo, online, fractions=[0.5, 0.5])

        sample = rb.sample(8)

        assert sample["value"].device.type == "cuda"
        assert (sample["value"][:4] < 4).all() and (sample["value"][4:] >= 4).all()

    def _make_sampler(self, sampler_type):
        if sampler_type is SamplerWithoutReplacement:
            return SamplerWithoutReplacement(drop_last=True)
        if sampler_type is RandomSampler:
            return RandomSampler()
        raise NotImplementedError

    def _make_storage(self, storage_type, data_type):
        if storage_type is LazyMemmapStorage:
            return LazyMemmapStorage(max_size=100)
        if storage_type is TensorStorage:
            if data_type is TensorDict:
                return TensorStorage(TensorDict(batch_size=[100]))
            elif data_type is torch.Tensor:
                return TensorStorage(torch.zeros(100))
            else:
                raise NotImplementedError
        if storage_type is ListStorage:
            return ListStorage(max_size=100)
        raise NotImplementedError

    def _make_collate(self, storage_type):
        if storage_type is ListStorage:
            return torch.stack
        else:
            return self._robust_stack

    @staticmethod
    def _robust_stack(tensor_list):
        if not isinstance(tensor_list, (tuple, list)):
            return tensor_list
        if all(tensor.shape == tensor_list[0].shape for tensor in tensor_list[1:]):
            return torch.stack(list(tensor_list))
        if is_tensor_collection(tensor_list[0]):
            return torch.cat(list(tensor_list))
        return torch.nested.nested_tensor(list(tensor_list))

    @pytest.mark.parametrize(
        "storage_type", [LazyMemmapStorage, TensorStorage, ListStorage]
    )
    @pytest.mark.parametrize("data_type", [torch.Tensor, TensorDict])
    @pytest.mark.parametrize("p", [[0.0, 0.9, 0.1], None])
    @pytest.mark.parametrize("num_buffer_sampled", [3, 16, None])
    @pytest.mark.parametrize("batch_size", [48, None])
    @pytest.mark.parametrize("sampler_type", [RandomSampler, SamplerWithoutReplacement])
    def test_rb(
        self, storage_type, sampler_type, data_type, p, num_buffer_sampled, batch_size
    ):
        storages = [self._make_storage(storage_type, data_type) for _ in range(3)]
        collate_fn = self._make_collate(storage_type)
        data = [self._make_data(data_type) for _ in range(3)]
        samplers = [self._make_sampler(sampler_type) for _ in range(3)]
        sub_batch_size = (
            batch_size // 3
            if issubclass(sampler_type, SamplerWithoutReplacement)
            and batch_size is not None
            else None
        )
        error_catcher = (
            pytest.raises(
                ValueError,
                match="Samplers with drop_last=True must work with a predictable batch-size",
            )
            if batch_size is None
            and issubclass(sampler_type, SamplerWithoutReplacement)
            else contextlib.nullcontext()
        )
        rbs = None
        with error_catcher:
            rbs = (rb0, rb1, rb2) = [
                ReplayBuffer(
                    storage=storage,
                    sampler=sampler,
                    collate_fn=collate_fn,
                    batch_size=sub_batch_size,
                )
                for (storage, sampler) in zip(storages, samplers)
            ]
        if rbs is None:
            return
        for datum, rb in zip(data, rbs):
            rb.extend(datum)
        rb = ReplayBufferEnsemble(
            *rbs, p=p, num_buffer_sampled=num_buffer_sampled, batch_size=batch_size
        )
        if batch_size is not None:
            for batch_iter in rb:
                assert isinstance(batch_iter, (torch.Tensor, TensorDictBase))
                break
            batch_sample, info = rb.sample(return_info=True)
        else:
            batch_iter = None
            batch_sample, info = rb.sample(48, return_info=True)
        assert isinstance(batch_sample, (torch.Tensor, TensorDictBase))
        if isinstance(batch_sample, TensorDictBase):
            assert "root" in batch_sample.keys()
            assert "nested" in batch_sample.keys()
            assert ("nested", "data") in batch_sample.keys(True)
            if p is not None:
                if batch_iter is not None:
                    buffer_ids = batch_iter.get(("index", "buffer_ids"))
                    assert isinstance(buffer_ids, torch.Tensor), batch_iter
                    assert 0 not in buffer_ids.unique().tolist()

                buffer_ids = batch_sample.get(("index", "buffer_ids"))
                assert isinstance(buffer_ids, torch.Tensor), buffer_ids
                assert 0 not in buffer_ids.unique().tolist()
            if num_buffer_sampled is not None:
                if batch_iter is not None:
                    assert batch_iter.shape == torch.Size(
                        [num_buffer_sampled, 48 // num_buffer_sampled]
                    )
                assert batch_sample.shape == torch.Size(
                    [num_buffer_sampled, 48 // num_buffer_sampled]
                )
            else:
                if batch_iter is not None:
                    assert batch_iter.shape == torch.Size([3, 16])
                assert batch_sample.shape == torch.Size([3, 16])

    def _prepare_dual_replay_buffer(self, explicit=False):
        torch.manual_seed(0)
        rb0 = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(10),
            transform=Compose(
                ToTensorImage(in_keys=["pixels", ("next", "pixels")]),
                Resize(32, in_keys=["pixels", ("next", "pixels")]),
                RenameTransform([("some", "key")], ["renamed"]),
            ),
        )
        rb1 = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(10),
            transform=Compose(
                ToTensorImage(in_keys=["pixels", ("next", "pixels")]),
                Resize(32, in_keys=["pixels", ("next", "pixels")]),
                RenameTransform(["another_key"], ["renamed"]),
            ),
        )
        if explicit:
            storages = StorageEnsemble(
                rb0._storage, rb1._storage, transforms=[rb0._transform, rb1._transform]
            )
            writers = WriterEnsemble(rb0._writer, rb1._writer)
            samplers = SamplerEnsemble(rb0._sampler, rb1._sampler, p=[0.5, 0.5])
            collate_fns = [rb0._collate_fn, rb1._collate_fn]
            rb = ReplayBufferEnsemble(
                storages=storages,
                samplers=samplers,
                writers=writers,
                collate_fns=collate_fns,
                transform=Resize(33, in_keys=["pixels"], out_keys=["pixels33"]),
            )
        else:
            rb = ReplayBufferEnsemble(
                rb0,
                rb1,
                p=[0.5, 0.5],
                transform=Resize(33, in_keys=["pixels"], out_keys=["pixels33"]),
            )
        data0 = TensorDict(
            {
                "pixels": torch.randint(255, (10, 244, 244, 3)),
                ("next", "pixels"): torch.randint(255, (10, 244, 244, 3)),
                ("some", "key"): torch.randn(10),
            },
            batch_size=[10],
        )
        data1 = TensorDict(
            {
                "pixels": torch.randint(255, (10, 64, 64, 3)),
                ("next", "pixels"): torch.randint(255, (10, 64, 64, 3)),
                "another_key": torch.randn(10),
            },
            batch_size=[10],
        )
        rb0.extend(data0)
        rb1.extend(data1)
        return rb, rb0, rb1

    @pytest.mark.skipif(not _has_tv, reason="torchvision not found")
    def test_rb_transform(self):
        rb, rb0, rb1 = self._prepare_dual_replay_buffer()
        for _ in range(2):
            sample = rb.sample(10)
            assert sample["next", "pixels"].shape == torch.Size([2, 5, 3, 32, 32])
            assert sample["pixels"].shape == torch.Size([2, 5, 3, 32, 32])
            assert sample["pixels33"].shape == torch.Size([2, 5, 3, 33, 33])
            assert sample["renamed"].shape == torch.Size([2, 5])

    @pytest.mark.skipif(not _has_tv, reason="torchvision not found")
    @pytest.mark.parametrize("explicit", [False, True])
    def test_rb_indexing(self, explicit):
        rb, rb0, rb1 = self._prepare_dual_replay_buffer(explicit=explicit)
        if explicit:
            # indirect checks
            assert rb[0]._storage is rb0._storage
            assert rb[1]._storage is rb1._storage
        else:
            assert rb[0] is rb0
            assert rb[1] is rb1
        assert rb[:] is rb

        torch.manual_seed(0)
        sample1 = rb.sample(6)
        # tensor
        torch.manual_seed(0)
        sample0 = rb[torch.tensor([0, 1])].sample(6)
        assert_allclose_td(sample0, sample1)
        # slice
        torch.manual_seed(0)
        sample0 = rb[:2].sample(6)
        assert_allclose_td(sample0, sample1)
        # np.ndarray
        torch.manual_seed(0)
        sample0 = rb[np.array([0, 1])].sample(6)
        assert_allclose_td(sample0, sample1)
        # list
        torch.manual_seed(0)
        sample0 = rb[[0, 1]].sample(6)
        assert_allclose_td(sample0, sample1)

        # direct indexing
        sample1 = rb[:, :3]
        # tensor
        sample0 = rb[torch.tensor([0, 1]), :3]
        assert_allclose_td(sample0, sample1)
        # slice
        torch.manual_seed(0)
        sample0 = rb[:2, :3]
        assert_allclose_td(sample0, sample1)
        # np.ndarray
        torch.manual_seed(0)
        sample0 = rb[np.array([0, 1]), :3]
        assert_allclose_td(sample0, sample1)
        # list
        torch.manual_seed(0)
        sample0 = rb[[0, 1], :3]
        assert_allclose_td(sample0, sample1)

        # check indexing of components
        assert isinstance(rb.storage[:], StorageEnsemble)
        assert isinstance(rb.storage[:2], StorageEnsemble)
        assert isinstance(rb.storage[torch.tensor([0, 1])], StorageEnsemble)
        assert isinstance(rb.storage[np.array([0, 1])], StorageEnsemble)
        assert isinstance(rb.storage[[0, 1]], StorageEnsemble)
        assert isinstance(rb.storage[1], LazyMemmapStorage)

        rb.storage[:, :3]
        rb.storage[:2, :3]
        rb.storage[torch.tensor([0, 1]), :3]
        rb.storage[np.array([0, 1]), :3]
        rb.storage[[0, 1], :3]

        assert isinstance(rb.sampler[:], SamplerEnsemble)
        assert isinstance(rb.sampler[:2], SamplerEnsemble)
        assert isinstance(rb.sampler[torch.tensor([0, 1])], SamplerEnsemble)
        assert isinstance(rb.sampler[np.array([0, 1])], SamplerEnsemble)
        assert isinstance(rb.sampler[[0, 1]], SamplerEnsemble)
        assert isinstance(rb.sampler[1], RandomSampler)

        assert isinstance(rb.writer[:], WriterEnsemble)
        assert isinstance(rb.writer[:2], WriterEnsemble)
        assert isinstance(rb.writer[torch.tensor([0, 1])], WriterEnsemble)
        assert isinstance(rb.writer[np.array([0, 1])], WriterEnsemble)
        assert isinstance(rb.writer[[0, 1]], WriterEnsemble)
        assert isinstance(rb.writer[0], RoundRobinWriter)


def _rbtype(datatype):
    if datatype in ("pytree", "tensorclass"):
        return [
            (ReplayBuffer, RandomSampler),
            (PrioritizedReplayBuffer, RandomSampler),
            (ReplayBuffer, SamplerWithoutReplacement),
            (PrioritizedReplayBuffer, SamplerWithoutReplacement),
        ]
    return [
        (ReplayBuffer, RandomSampler),
        (ReplayBuffer, SamplerWithoutReplacement),
        (PrioritizedReplayBuffer, None),
        (TensorDictReplayBuffer, RandomSampler),
        (TensorDictReplayBuffer, SamplerWithoutReplacement),
        (TensorDictPrioritizedReplayBuffer, None),
    ]


class TestRBMultidim:
    @tensorclass
    class MyData:
        x: torch.Tensor
        y: torch.Tensor
        z: torch.Tensor

    def _make_data(self, datatype, datadim):
        if datadim == 1:
            shape = [12]
        elif datadim == 2:
            shape = [4, 3]
        else:
            raise NotImplementedError
        if datatype == "pytree":
            return {
                "x": (torch.ones(*shape, 2), (torch.ones(*shape, 3))),
                "y": [
                    {"z": torch.ones(shape)},
                    torch.ones((*shape, 1), dtype=torch.bool),
                ],
            }
        elif datatype == "tensordict":
            return TensorDict(
                {"x": torch.ones(*shape, 2), "y": {"z": torch.ones(*shape, 3)}}, shape
            )
        elif datatype == "tensorclass":
            return self.MyData(
                x=torch.ones(*shape, 2),
                y=torch.ones(*shape, 3),
                z=torch.ones((*shape, 1), dtype=torch.bool),
                batch_size=shape,
            )

    datatype_rb_tuples = [
        [datatype, *rbtype]
        for datatype in ["pytree", "tensordict", "tensorclass"]
        for rbtype in _rbtype(datatype)
    ]

    @pytest.mark.parametrize("datatype,rbtype,sampler_cls", datatype_rb_tuples)
    @pytest.mark.parametrize("datadim", [1, 2])
    @pytest.mark.parametrize("storage_cls", [LazyMemmapStorage, LazyTensorStorage])
    def test_rb_multidim(self, datatype, datadim, rbtype, storage_cls, sampler_cls):
        data = self._make_data(datatype, datadim)
        if rbtype not in (PrioritizedReplayBuffer, TensorDictPrioritizedReplayBuffer):
            rbtype = functools.partial(rbtype, sampler=sampler_cls())
        else:
            rbtype = functools.partial(rbtype, alpha=0.9, beta=1.1)

        rb = rbtype(storage=storage_cls(100, ndim=datadim), batch_size=4)
        assert str(rb)  # check str works
        rb.extend(data)
        assert str(rb)
        assert len(rb) == 12
        data = rb[:]
        if datatype in ("tensordict", "tensorclass"):
            assert data.numel() == 12
        else:
            assert all(
                leaf.shape[:datadim].numel() == 12 for leaf in tree_flatten(data)[0]
            )
        s = rb.sample()
        assert str(rb)
        if datatype in ("tensordict", "tensorclass"):
            assert (s.exclude("index") == 1).all()
            assert s.numel() == 4
        else:
            for leaf in tree_iter(s):
                assert leaf.shape[0] == 4
                assert (leaf == 1).all()

    @pytest.mark.skipif(not _has_gym, reason="gym required for this test.")
    @pytest.mark.parametrize(
        "writer_cls",
        [TensorDictMaxValueWriter, RoundRobinWriter, TensorDictRoundRobinWriter],
    )
    @pytest.mark.parametrize("storage_cls", [LazyMemmapStorage, LazyTensorStorage])
    @pytest.mark.parametrize(
        "rbtype",
        [
            functools.partial(ReplayBuffer, batch_size=8),
            functools.partial(TensorDictReplayBuffer, batch_size=8),
        ],
    )
    @pytest.mark.parametrize(
        "sampler_cls",
        [
            functools.partial(SliceSampler, num_slices=2, strict_length=False),
            RandomSampler,
            functools.partial(
                SliceSamplerWithoutReplacement, num_slices=2, strict_length=False
            ),
            functools.partial(PrioritizedSampler, alpha=1.0, beta=1.0, max_capacity=10),
            functools.partial(
                PrioritizedSliceSampler,
                alpha=1.0,
                beta=1.0,
                max_capacity=10,
                num_slices=2,
                strict_length=False,
            ),
        ],
    )
    @pytest.mark.parametrize(
        "transform",
        [
            None,
            [
                lambda: split_trajectories,
                functools.partial(MultiStep, gamma=0.9, n_steps=3),
            ],
        ],
    )
    @pytest.mark.parametrize("env_device", get_default_devices())
    def test_rb_multidim_collector(
        self, rbtype, storage_cls, writer_cls, sampler_cls, transform, env_device
    ):
        torch.manual_seed(0)
        env = SerialEnv(2, lambda: GymEnv(CARTPOLE_VERSIONED()), device=env_device)
        env.set_seed(0)
        collector = Collector(
            env,
            RandomPolicy(env.action_spec),
            frames_per_batch=4,
            total_frames=16,
            device=env_device,
        )
        if writer_cls is TensorDictMaxValueWriter:
            with pytest.raises(
                ValueError,
                match="TensorDictMaxValueWriter is not compatible with storages with more than one dimension",
            ):
                rb = rbtype(
                    storage=storage_cls(max_size=10, ndim=2),
                    sampler=sampler_cls(),
                    writer=writer_cls(),
                    delayed_init=False,
                )
            return
        rb = rbtype(
            storage=storage_cls(max_size=10, ndim=2),
            sampler=sampler_cls(),
            writer=writer_cls(),
        )
        if not isinstance(rb.sampler, SliceSampler) and transform is not None:
            pytest.skip("no need to test this combination")
        if transform:
            for t in transform:
                rb.append_transform(t())
        try:
            for i, data in enumerate(collector):  # noqa: B007
                assert data.device == torch.device(env_device)
                rb.extend(data)
                if isinstance(rb, TensorDictReplayBuffer) and transform is not None:
                    # this should fail bc we can't set the indices after executing the transform.
                    with pytest.raises(
                        RuntimeError, match="Failed to set the metadata"
                    ):
                        rb.sample()
                    return
                s = rb.sample()
                assert s.device == torch.device("cpu")
                rbtot = rb[:]
                assert rbtot.shape[0] == 2
                assert len(rb) == rbtot.numel()
                if transform is not None:
                    assert s.ndim == 2
        except Exception:
            raise

    @pytest.mark.parametrize("strict_length", [True, False])
    def test_done_slicesampler(self, strict_length):
        env = SerialEnv(
            3,
            [
                lambda: CountingEnv(max_steps=31).add_truncated_keys(),
                lambda: CountingEnv(max_steps=32).add_truncated_keys(),
                lambda: CountingEnv(max_steps=33).add_truncated_keys(),
            ],
        )
        full_action_spec = CountingEnv(max_steps=32).full_action_spec
        policy = lambda td: td.update(
            full_action_spec.zero((3,)).apply_(lambda x: x + 1)
        )
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(200, ndim=2),
            sampler=SliceSampler(
                slice_len=32,
                strict_length=strict_length,
                truncated_key=("next", "truncated"),
            ),
            batch_size=128,
        )

        # env.add_truncated_keys()

        for i in range(50):
            r = env.rollout(
                50, policy=policy, break_when_any_done=False, set_truncated=True
            )
            rb.extend(r)

            sample = rb.sample()

            assert sample["next", "done"].sum() == 128 // 32, (
                i,
                sample["next", "done"].sum(),
            )
            assert (split_trajectories(sample)["next", "done"].sum(-2) == 1).all()


@pytest.mark.skipif(not _has_gym, reason="gym required")
class TestCheckpointers:
    @pytest.mark.parametrize("storage_type", [LazyMemmapStorage, LazyTensorStorage])
    @pytest.mark.parametrize(
        "checkpointer",
        [FlatStorageCheckpointer, H5StorageCheckpointer, NestedStorageCheckpointer],
    )
    @pytest.mark.parametrize("frames_per_batch", [22, 122])
    def test_simple_env(self, storage_type, checkpointer, tmpdir, frames_per_batch):
        env = GymEnv(CARTPOLE_VERSIONED(), device=None)
        env.set_seed(0)
        torch.manual_seed(0)
        collector = Collector(
            env,
            policy=env.rand_step,
            total_frames=200,
            frames_per_batch=frames_per_batch,
        )
        rb = ReplayBuffer(storage=storage_type(100))
        rb_test = ReplayBuffer(storage=storage_type(100))
        if torch.__version__ < "2.4.0.dev" and checkpointer in (
            H5StorageCheckpointer,
            NestedStorageCheckpointer,
        ):
            with pytest.raises(ValueError, match="Unsupported torch version"):
                checkpointer()
            return
        rb.storage.checkpointer = checkpointer()
        rb_test.storage.checkpointer = checkpointer()
        for data in collector:
            rb.extend(data)
            rb.dumps(tmpdir)
            rb_test.loads(tmpdir)
            assert_allclose_td(rb_test[:], rb[:])
            assert rb.writer._cursor == rb_test._writer._cursor

    @pytest.mark.parametrize("storage_type", [LazyMemmapStorage, LazyTensorStorage])
    @pytest.mark.parametrize("frames_per_batch", [22, 122])
    @pytest.mark.parametrize(
        "checkpointer",
        [FlatStorageCheckpointer, NestedStorageCheckpointer, H5StorageCheckpointer],
    )
    def test_multi_env(self, storage_type, checkpointer, tmpdir, frames_per_batch):
        env = SerialEnv(
            3,
            lambda: GymEnv(CARTPOLE_VERSIONED(), device=None).append_transform(
                StepCounter()
            ),
        )
        env.set_seed(0)
        torch.manual_seed(0)
        collector = Collector(
            env,
            policy=env.rand_step,
            total_frames=200,
            frames_per_batch=frames_per_batch,
        )
        rb = ReplayBuffer(storage=storage_type(100, ndim=2))
        rb_test = ReplayBuffer(storage=storage_type(100, ndim=2))
        if torch.__version__ < "2.4.0.dev" and checkpointer in (
            H5StorageCheckpointer,
            NestedStorageCheckpointer,
        ):
            with pytest.raises(ValueError, match="Unsupported torch version"):
                checkpointer()
            return
        rb.storage.checkpointer = checkpointer()
        rb_test.storage.checkpointer = checkpointer()
        for data in collector:
            rb.extend(data)
            assert rb.storage.max_size == 102
            if frames_per_batch > 100:
                assert rb.storage._is_full
                assert len(rb) == 102
                # Checks that when writing to the buffer with a batch greater than the total
                # size, we get the last step written properly.
                assert (rb[:]["next", "step_count"][:, -1] != 0).any()
            rb.dumps(tmpdir)
            rb.dumps(tmpdir)
            rb_test.loads(tmpdir)
            assert_allclose_td(rb_test[:], rb[:])
            assert rb.writer._cursor == rb_test._writer._cursor


@pytest.mark.parametrize("storage_type", [LazyMemmapStorage, LazyTensorStorage])
@pytest.mark.parametrize(
    "checkpointer",
    [
        FlatStorageCheckpointer,
        NestedStorageCheckpointer,
        pytest.param(
            H5StorageCheckpointer,
            marks=pytest.mark.skipif(not _has_h5py, reason="h5py required"),
        ),
    ],
)
class TestCheckpointRoundTrip:
    @staticmethod
    def _ted_batch(nsteps, done, offset=0):
        obs = torch.arange(3 * nsteps, dtype=torch.float32).view(3, nsteps) + offset
        done = torch.full((3, nsteps, 1), done)
        return TensorDict(
            {
                "observation": obs,
                "action": torch.zeros(3, nsteps),
                "done": torch.zeros_like(done),
                "terminated": torch.zeros_like(done),
                "next": {
                    "observation": obs + 1,
                    "done": done,
                    "terminated": done,
                    "reward": torch.ones(3, nsteps, 1),
                },
            },
            batch_size=[3, nsteps],
        )

    @pytest.mark.parametrize("refill_steps", [0, 7])
    def test_checkpoint_after_empty(
        self, storage_type, checkpointer, refill_steps, tmpdir
    ):
        """Rows emptied from the buffer do not reach a checkpoint, refilled or not."""
        rb = ReplayBuffer(storage=storage_type(102, ndim=2))
        rb_test = ReplayBuffer(storage=storage_type(102, ndim=2))
        rb.storage.checkpointer = checkpointer()
        rb_test.storage.checkpointer = checkpointer()
        rb.extend(self._ted_batch(34, done=True))
        rb.empty()
        if refill_steps:
            rb.extend(self._ted_batch(refill_steps, done=False))
        rb.dumps(tmpdir)
        rb_test.loads(tmpdir)
        assert len(rb_test) == len(rb) == 3 * refill_steps
        batch = self._ted_batch(5, done=False, offset=100)
        rb.extend(batch)
        rb_test.extend(batch)
        assert_allclose_td(rb_test[:], rb[:])

    def test_resave_loaded_empty_checkpoint(self, storage_type, checkpointer, tmpdir):
        """A buffer loaded from an empty checkpoint can be saved and reloaded."""
        rb = ReplayBuffer(storage=storage_type(102, ndim=2))
        rb.storage.checkpointer = checkpointer()
        rb.extend(self._ted_batch(34, done=True))
        rb.empty()
        rb.dumps(tmpdir / "first")
        rb_loaded = ReplayBuffer(storage=storage_type(102, ndim=2))
        rb_loaded.storage.checkpointer = checkpointer()
        rb_loaded.loads(tmpdir / "first")
        rb_loaded.dumps(tmpdir / "second")
        rb_test = ReplayBuffer(storage=storage_type(102, ndim=2))
        rb_test.storage.checkpointer = checkpointer()
        rb_test.loads(tmpdir / "second")
        assert len(rb_test) == 0
        batch = self._ted_batch(5, done=False, offset=100)
        rb_test.extend(batch)
        assert_allclose_td(rb_test[:], batch)

    def test_resave_loaded_partial_checkpoint(self, storage_type, checkpointer, tmpdir):
        """Saving a buffer loaded from a partial checkpoint keeps its done flags."""
        rb = ReplayBuffer(storage=storage_type(102, ndim=2))
        rb.storage.checkpointer = checkpointer()
        rb.extend(self._ted_batch(7, done=True))
        rb.dumps(tmpdir / "first")
        rb_loaded = ReplayBuffer(storage=storage_type(102, ndim=2))
        rb_loaded.storage.checkpointer = checkpointer()
        rb_loaded.loads(tmpdir / "first")
        rb_loaded.dumps(tmpdir / "second")
        rb_test = ReplayBuffer(storage=storage_type(102, ndim=2))
        rb_test.storage.checkpointer = checkpointer()
        rb_test.loads(tmpdir / "second")
        assert rb_test[:]["next", "done"].all()
        assert_allclose_td(rb_test[:], rb[:])

    def test_full_storage_slice_cursor(self, storage_type, checkpointer, tmpdir):
        """A full buffer last written through a slice round-trips its transitions."""
        rb = ReplayBuffer(storage=storage_type(10))
        rb.storage.checkpointer = checkpointer()
        rb.extend(self._ted_batch(10, done=False)[0])
        rb.storage.set(slice(0, 4), self._ted_batch(4, done=False, offset=10)[0])
        rb.dumps(tmpdir)
        rb_test = ReplayBuffer(storage=storage_type(10))
        rb_test.storage.checkpointer = checkpointer()
        rb_test.loads(tmpdir)
        assert_allclose_td(rb_test[:], rb[:])


@pytest.mark.parametrize("storage_type", [LazyMemmapStorage, LazyTensorStorage])
def test_checkpoint_empty_with_save_hook_only(storage_type, tmpdir):
    """An emptied buffer with a save hook and no load hook loads back empty."""
    rb = ReplayBuffer(storage=storage_type(102, ndim=2))
    rb.register_save_hook(lambda data, path=None: data)
    rb.extend(TestCheckpointRoundTrip._ted_batch(34, done=True))
    rb.empty()
    rb.dumps(tmpdir)
    rb_test = ReplayBuffer(storage=storage_type(102, ndim=2))
    rb_test.loads(tmpdir)
    assert len(rb_test) == 0
    batch = TestCheckpointRoundTrip._ted_batch(5, done=False, offset=100)
    rb_test.extend(batch)
    assert_allclose_td(rb_test[:], batch)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
