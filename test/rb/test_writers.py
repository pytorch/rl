# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse

import pytest
import torch
from tensordict import TensorDict
from torch import multiprocessing as mp

from torchrl.data import (
    PrioritizedReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import samplers
from torchrl.data.replay_buffers.samplers import (
    PrioritizedSampler,
    RandomSampler,
    SamplerWithoutReplacement,
)
from torchrl.data.replay_buffers.storages import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ListStorage,
)
from torchrl.data.replay_buffers.writers import (
    TensorDictMaxValueWriter,
    TensorDictRoundRobinWriter,
)
from torchrl.testing import get_default_devices


class TestMaxValueWriter:
    @pytest.mark.parametrize("size", [20, 25, 30])
    @pytest.mark.parametrize("batch_size", [1, 10, 15])
    @pytest.mark.parametrize("reward_ranges", [(0.25, 0.5, 1.0)])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_max_value_writer(self, size, batch_size, reward_ranges, device):
        torch.manual_seed(0)
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(size, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=batch_size,
            writer=TensorDictMaxValueWriter(rank_key="key"),
        )

        max_reward1, max_reward2, max_reward3 = reward_ranges

        td = TensorDict(
            {
                "key": torch.clamp_max(torch.rand(size), max=max_reward1),
                "obs": torch.rand(size),
            },
            batch_size=size,
            device=device,
        )
        rb.extend(td)
        sample = rb.sample()
        assert (sample.get("key") <= max_reward1).all()
        assert (0 <= sample.get("key")).all()
        assert len(sample.get("index").unique()) == len(sample.get("index"))

        td = TensorDict(
            {
                "key": torch.clamp(torch.rand(size), min=max_reward1, max=max_reward2),
                "obs": torch.rand(size),
            },
            batch_size=size,
            device=device,
        )
        rb.extend(td)
        sample = rb.sample()
        assert (sample.get("key") <= max_reward2).all()
        assert (max_reward1 <= sample.get("key")).all()
        assert len(sample.get("index").unique()) == len(sample.get("index"))

        td = TensorDict(
            {
                "key": torch.clamp(torch.rand(size), min=max_reward2, max=max_reward3),
                "obs": torch.rand(size),
            },
            batch_size=size,
            device=device,
        )

        for sample in td:
            rb.add(sample)

        sample = rb.sample()
        assert (sample.get("key") <= max_reward3).all()
        assert (max_reward2 <= sample.get("key")).all()
        assert len(sample.get("index").unique()) == len(sample.get("index"))

        # Finally, test the case when no obs should be added
        td = TensorDict(
            {
                "key": torch.zeros(size),
                "obs": torch.rand(size),
            },
            batch_size=size,
            device=device,
        )
        rb.extend(td)
        sample = rb.sample()
        assert (sample.get("key") != 0).all()

    @pytest.mark.parametrize("size", [20, 25, 30])
    @pytest.mark.parametrize("batch_size", [1, 10, 15])
    @pytest.mark.parametrize("reward_ranges", [(0.25, 0.5, 1.0)])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_max_value_writer_serialize(
        self, size, batch_size, reward_ranges, device, tmpdir
    ):
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(size, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=batch_size,
            writer=TensorDictMaxValueWriter(rank_key="key"),
        )

        max_reward1, max_reward2, max_reward3 = reward_ranges

        td = TensorDict(
            {
                "key": torch.clamp_max(torch.rand(size), max=max_reward1),
                "obs": torch.rand(size),
            },
            batch_size=size,
            device=device,
        )
        rb.extend(td)
        rb.writer.dumps(tmpdir)
        # check we can dump twice
        rb.writer.dumps(tmpdir)
        other = TensorDictMaxValueWriter(rank_key="key")
        other.loads(tmpdir)
        assert len(rb.writer._current_top_values) == len(other._current_top_values)
        torch.testing.assert_close(
            torch.tensor(rb.writer._current_top_values),
            torch.tensor(other._current_top_values),
        )

    @pytest.mark.parametrize("size", [[], [1], [2, 3]])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("reduction", ["max", "min", "mean", "median", "sum"])
    def test_max_value_writer_reduce(self, size, device, reduction):
        torch.manual_seed(0)
        batch_size = 4
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(1, device=device),
            sampler=SamplerWithoutReplacement(),
            batch_size=batch_size,
            writer=TensorDictMaxValueWriter(rank_key="key", reduction=reduction),
        )

        key = torch.rand(batch_size, *size, device=device)
        obs = torch.rand(batch_size, *size, device=device)
        td = TensorDict(
            {"key": key, "obs": obs},
            batch_size=batch_size,
            device=device,
        )
        rb.extend(td)
        sample = rb.sample()
        if reduction == "max":
            rank_key = torch.stack([k.max() for k in key.unbind(0)])
        elif reduction == "min":
            rank_key = torch.stack([k.min() for k in key.unbind(0)])
        elif reduction == "mean":
            rank_key = torch.stack([k.mean() for k in key.unbind(0)])
        elif reduction == "median":
            rank_key = torch.stack([k.median() for k in key.unbind(0)])
        elif reduction == "sum":
            rank_key = torch.stack([k.sum() for k in key.unbind(0)])

        top_rank = torch.argmax(rank_key)
        assert (sample.get("obs") == obs[top_rank]).all()


class TestMultiProc:
    @staticmethod
    def worker(rb, q0, q1):
        td = TensorDict({"a": torch.ones(10), "next": {"reward": torch.ones(10)}}, [10])
        rb.extend(td)
        q0.put("extended")
        extended = q1.get(timeout=5)
        assert extended == "extended"
        assert len(rb) == 21, len(rb)
        assert (rb["a"][:9] == 2).all()
        q0.put("finish")

    @staticmethod
    def async_prb_worker(rb, worker_id, q):
        td = TensorDict(
            {
                "obs": torch.full((4, 1), worker_id, dtype=torch.float32),
                "prio": {"td_error": torch.linspace(0.1, 1.0, 4) + worker_id},
            },
            [4],
        )
        rb.extend(td)
        q.put("finish")

    @staticmethod
    def async_generic_prb_worker(rb, worker_id, q):
        data = TensorDict(
            {"obs": torch.full((4, 1), worker_id, dtype=torch.float32)},
            [4],
        )
        rb.extend(data)
        q.put("finish")

    def exec_multiproc_rb(
        self,
        storage_type=LazyMemmapStorage,
        init=True,
        writer_type=TensorDictRoundRobinWriter,
        sampler_type=RandomSampler,
        device=None,
    ):
        rb = TensorDictReplayBuffer(
            storage=storage_type(21), writer=writer_type(), sampler=sampler_type()
        )
        if init:
            td = TensorDict(
                {"a": torch.zeros(10), "next": {"reward": torch.ones(10)}},
                [10],
                device=device,
            )
            rb.extend(td)
        q0 = mp.Queue(1)
        q1 = mp.Queue(1)
        proc = mp.Process(target=self.worker, args=(rb, q0, q1))
        proc.start()
        try:
            extended = q0.get(timeout=100)
            assert extended == "extended"
            assert len(rb) == 20
            assert (rb["a"][10:20] == 1).all()
            td = TensorDict({"a": torch.zeros(10) + 2}, [10])
            rb.extend(td)
            q1.put("extended")
            finish = q0.get(timeout=5)
            assert finish == "finish"
        finally:
            proc.join()

    def test_multiproc_rb(self):
        return self.exec_multiproc_rb()

    def test_error_list(self):
        # list storage cannot be shared
        with pytest.raises(RuntimeError, match="Cannot share a storage of type"):
            self.exec_multiproc_rb(storage_type=ListStorage)

    def test_error_maxwriter(self):
        # TensorDictMaxValueWriter cannot be shared
        with pytest.raises(RuntimeError, match="cannot be shared between processes"):
            self.exec_multiproc_rb(writer_type=TensorDictMaxValueWriter)

    def test_error_prb(self):
        # PrioritizedSampler cannot be shared
        if samplers.SumSegmentTreeFp32 is None:
            pytest.skip("PrioritizedSampler extension is unavailable.")
        with pytest.raises(
            RuntimeError,
            match="cannot be shared between processes.*sync=False",
        ):
            self.exec_multiproc_rb(
                sampler_type=lambda: PrioritizedSampler(21, alpha=1.1, beta=0.5)
            )

    def test_prioritized_sampler_shared_error_mentions_sync_false(self, monkeypatch):
        sampler = PrioritizedSampler.__new__(PrioritizedSampler)
        monkeypatch.setattr(samplers, "get_spawning_popen", lambda: object())
        with pytest.raises(RuntimeError, match="sync=False"):
            sampler.__getstate__()

    def test_shared_prefetch_error_mentions_fix(self):
        with pytest.raises(
            ValueError,
            match="Cannot share prefetched replay buffers.*prefetch=0.*shared=False",
        ):
            TensorDictReplayBuffer(
                storage=LazyTensorStorage(10),
                batch_size=2,
                prefetch=1,
                shared=True,
            )

    def test_async_prioritized_rb_multiproc_writes(self):
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            priority_key=("prio", "td_error"),
            storage=LazyMemmapStorage(32, shared_init=True),
            batch_size=4,
            shared=True,
            sync=False,
        )
        q = mp.Queue()
        processes = []
        for worker_id in range(2):
            proc = mp.Process(
                target=self.async_prb_worker,
                args=(rb, worker_id, q),
            )
            processes.append(proc)
            proc.start()

        for proc in processes:
            proc.join()
            assert proc.exitcode == 0
            assert q.get(timeout=5) == "finish"

        assert rb.write_count == 8
        sample = rb.sample()
        assert rb._prioritized_sampler_write_count == 8
        assert sample["obs"].shape == (4, 1)
        assert "priority_weight" in sample.keys()
        assert "index" in sample.keys()

        sample["prio", "td_error"] = torch.ones(sample.shape) * 10
        rb.update_tensordict_priority(sample)
        assert rb.prioritized_sampler._max_priority[0] is not None

    def test_async_generic_prioritized_rb_multiproc_writes(self):
        rb = PrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            storage=LazyMemmapStorage(32),
            batch_size=4,
            sync=False,
        )
        rb.extend(TensorDict({"obs": torch.zeros((1, 1))}, [1]))
        rb.empty()
        rb.share(True)
        q = mp.Queue()
        processes = []
        for worker_id in range(2):
            proc = mp.Process(
                target=self.async_generic_prb_worker,
                args=(rb, worker_id, q),
            )
            processes.append(proc)
            proc.start()

        for proc in processes:
            proc.join()
            assert proc.exitcode == 0
            assert q.get(timeout=5) == "finish"

        assert rb.write_count == 8
        sample, info = rb.sample(return_info=True)
        assert rb._prioritized_sampler_write_count == 8
        assert sample["obs"].shape == (4, 1)
        assert "priority_weight" in info
        assert "index" in info

        rb.update_priority(info["index"], torch.ones(4) * 10)
        assert rb.prioritized_sampler._max_priority[0] is not None

    def test_error_noninit(self):
        # list storage cannot be shared
        with pytest.raises(RuntimeError, match="it has not been initialized yet"):
            self.exec_multiproc_rb(init=False)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
