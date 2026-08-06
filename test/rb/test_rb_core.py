# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import functools
import json

import pytest
import torch
import torchrl
from _rb_common import OLD_TORCH, ReplayBufferRNG, TensorDictReplayBufferRNG
from tensordict import assert_allclose_td, TensorDict, TensorDictBase

from torchrl._utils import rl_warnings
from torchrl.data import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    Sequence,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.sample_units import SampleUnit, Transition
from torchrl.data.replay_buffers.samplers import (
    ConsumingSampler,
    PrioritizedSampler,
    RandomSampler,
    SamplerWithoutReplacement,
    SliceSampler,
)

from torchrl.data.replay_buffers.storages import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ListStorage,
    TensorStorage,
)
from torchrl.data.replay_buffers.writers import ImmutableDatasetWriter, RoundRobinWriter
from torchrl.envs.transforms.transforms import Transform
from torchrl.objectives.llm import MCAdvantage


class _UnshareableWriteStateTransform(Transform):
    requires_shared_write_state = True

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        return tensordict


def test_replay_buffer_direct_service_client_is_identity():
    replay_buffer = ReplayBuffer(storage=ListStorage(4), service_backend="direct")
    assert replay_buffer.client() is replay_buffer
    assert replay_buffer.start() is replay_buffer
    assert replay_buffer.service_backend == "direct"
    assert replay_buffer.is_alive
    replay_buffer.shutdown()
    assert not replay_buffer.is_alive
    replay_buffer.shutdown()


def test_replay_buffer_read_write_all_in_order():
    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(6))
    rb_slice = TensorDictReplayBuffer(storage=LazyTensorStorage(6))
    data = TensorDict({"obs": torch.arange(6), "reward": torch.zeros(6)}, [6])
    rb.extend(data)
    rb_slice.extend(data.clone())

    all_data = rb.read_all_in_order()
    assert_allclose_td(all_data, rb[:])
    assert all_data["obs"].tolist() == list(range(6))
    all_data["value_target"] = all_data["obs"] + 1
    rb.write_all(all_data)
    rb_slice[:] = all_data.clone()

    updated = rb.read_all_in_order()
    assert_allclose_td(updated, rb[:])
    assert_allclose_td(updated, rb_slice[:])
    assert updated["value_target"].tolist() == list(range(1, 7))


def test_replay_buffer_read_write_all_in_order_with_end():
    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10))
    rb_slice = TensorDictReplayBuffer(storage=LazyTensorStorage(10))
    rb.extend(TensorDict({"obs": torch.arange(6)}, [6]))
    rb_slice.extend(TensorDict({"obs": torch.arange(6)}, [6]))

    partial = rb.read_all_in_order(end=3)
    assert_allclose_td(partial, rb[:3])
    partial["obs"] = partial["obs"] + 10
    rb.write_all(partial, end=3)
    rb_slice[:3] = partial.clone()

    updated = rb.read_all_in_order()
    assert_allclose_td(updated, rb_slice[:])
    assert updated["obs"].tolist() == [10, 11, 12, 3, 4, 5]


def test_replay_buffer_read_write_all_in_order_matches_full_slice_ndim2():
    rb = TensorDictReplayBuffer(storage=LazyTensorStorage(6, ndim=2))
    rb_slice = TensorDictReplayBuffer(storage=LazyTensorStorage(6, ndim=2))
    data = TensorDict(
        {"obs": torch.arange(6).reshape(2, 3), "reward": torch.zeros(2, 3)},
        [2, 3],
    )
    rb.extend(data)
    rb_slice.extend(data.clone())

    all_data = rb.read_all_in_order()
    assert_allclose_td(all_data, rb[:])
    all_data["value_target"] = all_data["obs"] + 1
    rb.write_all(all_data)
    rb_slice[:] = all_data.clone()

    assert_allclose_td(rb.read_all_in_order(), rb[:])
    assert_allclose_td(rb.read_all_in_order(), rb_slice[:])


def test_replay_buffer_share_auto_shares_write_stateful_transform():
    transform = MCAdvantage(grpo_size=2, prompt_key="group_id", trajectory_return="sum")
    rb = ReplayBuffer(storage=ListStorage(10))
    rb.append_transform(transform)
    assert not transform.is_shared
    rb.share()
    assert transform.is_shared


def test_replay_buffer_append_transform_shares_when_buffer_is_shared():
    transform = MCAdvantage(grpo_size=2, prompt_key="group_id", trajectory_return="sum")
    rb = ReplayBuffer(storage=ListStorage(10), shared=True)
    rb.append_transform(transform)
    assert transform.is_shared


def test_replay_buffer_insert_transform_shares_when_buffer_is_shared():
    transform = MCAdvantage(grpo_size=2, prompt_key="group_id", trajectory_return="sum")
    rb = ReplayBuffer(storage=ListStorage(10), shared=True)
    rb.insert_transform(0, transform)
    assert transform.is_shared


def test_replay_buffer_share_rejects_unshareable_write_stateful_transform():
    transform = _UnshareableWriteStateTransform()
    rb = ReplayBuffer(storage=ListStorage(10))
    rb.append_transform(transform)
    with pytest.raises(RuntimeError, match="write state"):
        rb.share()


def test_replay_buffer_append_transform_rejects_unshareable_when_shared():
    transform = _UnshareableWriteStateTransform()
    rb = ReplayBuffer(storage=ListStorage(10), shared=True)
    with pytest.raises(RuntimeError, match="write state"):
        rb.append_transform(transform)


def test_replay_buffer_insert_transform_rejects_unshareable_when_shared():
    transform = _UnshareableWriteStateTransform()
    rb = ReplayBuffer(storage=ListStorage(10), shared=True)
    with pytest.raises(RuntimeError, match="write state"):
        rb.insert_transform(0, transform)


class TestRNG:
    def test_rb_rng(self):
        state = torch.random.get_rng_state()
        rb = ReplayBufferRNG(
            sampler=RandomSampler(), storage=LazyTensorStorage(100), delayed_init=False
        )
        assert rb.initialized
        rb.extend(torch.arange(100))
        rb._rng.set_state(state)
        a = rb.sample(32)
        rb._rng.set_state(state)
        b = rb.sample(32)
        assert (a == b).all()
        c = rb.sample(32)
        assert (a != c).any()

    def test_prb_rng(self):
        state = torch.random.get_rng_state()
        rb = ReplayBuffer(
            sampler=PrioritizedSampler(100, 1.0, 1.0),
            storage=LazyTensorStorage(100),
            generator=torch.Generator(),
        )
        rb.extend(torch.arange(100))
        rb.update_priority(index=torch.arange(100), priority=torch.arange(1, 101))

        rb._rng.set_state(state)
        a = rb.sample(32)

        rb._rng.set_state(state)
        b = rb.sample(32)
        assert (a == b).all()

        c = rb.sample(32)
        assert (a != c).any()

    def test_slice_rng(self):
        state = torch.random.get_rng_state()
        rb = ReplayBuffer(
            sampler=SliceSampler(num_slices=4),
            storage=LazyTensorStorage(100),
            generator=torch.Generator(),
        )
        done = torch.zeros(100, 1, dtype=torch.bool)
        done[49] = 1
        done[-1] = 1
        data = TensorDict(
            {
                "data": torch.arange(100),
                ("next", "done"): done,
            },
            batch_size=[100],
        )
        rb.extend(data)

        rb._rng.set_state(state)
        a = rb.sample(32)

        rb._rng.set_state(state)
        b = rb.sample(32)
        assert (a == b).all()

        c = rb.sample(32)
        assert (a != c).any()

    def test_rng_state_dict(self):
        state = torch.random.get_rng_state()
        rb = ReplayBufferRNG(sampler=RandomSampler(), storage=LazyTensorStorage(100))
        rb.extend(torch.arange(100))
        rb._rng.set_state(state)
        sd = rb.state_dict()
        assert sd.get("_rng") is not None
        a = rb.sample(32)

        rb.load_state_dict(sd)
        b = rb.sample(32)
        assert (a == b).all()
        c = rb.sample(32)
        assert (a != c).any()

    def test_rng_dumps(self, tmpdir):
        state = torch.random.get_rng_state()
        rb = ReplayBufferRNG(sampler=RandomSampler(), storage=LazyTensorStorage(100))
        rb.extend(torch.arange(100))
        rb._rng.set_state(state)
        rb.dumps(tmpdir)
        a = rb.sample(32)

        rb.loads(tmpdir)
        b = rb.sample(32)
        assert (a == b).all()
        c = rb.sample(32)
        assert (a != c).any()


@pytest.mark.parametrize(
    "rbtype,storage",
    [
        (ReplayBuffer, None),
        (ReplayBuffer, ListStorage),
        (ReplayBufferRNG, None),
        (ReplayBufferRNG, ListStorage),
        (PrioritizedReplayBuffer, None),
        (PrioritizedReplayBuffer, ListStorage),
        (TensorDictReplayBuffer, None),
        (TensorDictReplayBuffer, ListStorage),
        (TensorDictReplayBuffer, LazyTensorStorage),
        (TensorDictReplayBuffer, LazyMemmapStorage),
        (TensorDictReplayBufferRNG, None),
        (TensorDictReplayBufferRNG, ListStorage),
        (TensorDictReplayBufferRNG, LazyTensorStorage),
        (TensorDictReplayBufferRNG, LazyMemmapStorage),
        (TensorDictPrioritizedReplayBuffer, None),
        (TensorDictPrioritizedReplayBuffer, ListStorage),
        (TensorDictPrioritizedReplayBuffer, LazyTensorStorage),
        (TensorDictPrioritizedReplayBuffer, LazyMemmapStorage),
    ],
)
@pytest.mark.parametrize("size", [3, 5, 100])
@pytest.mark.parametrize("prefetch", [0])
class TestBuffers:
    default_constr = {
        ReplayBuffer: ReplayBuffer,
        PrioritizedReplayBuffer: functools.partial(
            PrioritizedReplayBuffer, alpha=0.8, beta=0.9
        ),
        TensorDictReplayBuffer: TensorDictReplayBuffer,
        TensorDictPrioritizedReplayBuffer: functools.partial(
            TensorDictPrioritizedReplayBuffer, alpha=0.8, beta=0.9
        ),
        TensorDictReplayBufferRNG: TensorDictReplayBufferRNG,
        ReplayBufferRNG: ReplayBufferRNG,
    }

    def _get_rb(self, rbtype, size, storage, prefetch):
        if storage is not None:
            storage = storage(size)
        rb = self.default_constr[rbtype](
            storage=storage, prefetch=prefetch, batch_size=3
        )
        return rb

    def _get_datum(self, rbtype):
        if rbtype in (ReplayBuffer, ReplayBufferRNG):
            data = torch.randint(100, (1,))
        elif rbtype is PrioritizedReplayBuffer:
            data = torch.randint(100, (1,))
        elif rbtype in (TensorDictReplayBuffer, TensorDictReplayBufferRNG):
            data = TensorDict({"a": torch.randint(100, (1,))}, [])
        elif rbtype is TensorDictPrioritizedReplayBuffer:
            data = TensorDict({"a": torch.randint(100, (1,))}, [])
        else:
            raise NotImplementedError(rbtype)
        return data

    def _get_data(self, rbtype, size):
        if rbtype in (ReplayBuffer, ReplayBufferRNG):
            data = [torch.randint(100, (1,)) for _ in range(size)]
        elif rbtype is PrioritizedReplayBuffer:
            data = [torch.randint(100, (1,)) for _ in range(size)]
        elif rbtype in (TensorDictReplayBuffer, TensorDictReplayBufferRNG):
            data = TensorDict(
                {
                    "a": torch.randint(100, (size,)),
                    "b": TensorDict({"c": torch.randint(100, (size,))}, [size]),
                },
                [size],
            )
        elif rbtype is TensorDictPrioritizedReplayBuffer:
            data = TensorDict(
                {
                    "a": torch.randint(100, (size,)),
                    "b": TensorDict({"c": torch.randint(100, (size,))}, [size]),
                },
                [size],
            )
        else:
            raise NotImplementedError(rbtype)
        return data

    def test_cursor_position2(self, rbtype, storage, size, prefetch):
        torch.manual_seed(0)
        rb = self._get_rb(rbtype, storage=storage, size=size, prefetch=prefetch)
        batch1 = self._get_data(rbtype, size=5)
        cond = (
            OLD_TORCH and size < len(batch1) and isinstance(rb.storage, TensorStorage)
        )
        with (
            pytest.warns(
                UserWarning,
                match="A cursor of length superior to the storage capacity was provided",
            )
            if cond
            else contextlib.nullcontext()
        ):
            rb.extend(batch1)

        # Added fewer data than storage max size
        if size > 5 or storage is None:
            assert rb.writer._cursor == 5
        # Added more data than storage max size
        elif size < 5:
            assert rb.writer._cursor == 5 - size
        # Added as data as storage max size
        else:
            assert rb.writer._cursor == 0
            batch2 = self._get_data(rbtype, size=size - 1)
            rb.extend(batch2)
            assert rb.writer._cursor == size - 1

    def test_add(self, rbtype, storage, size, prefetch):
        torch.manual_seed(0)
        rb = self._get_rb(rbtype, storage=storage, size=size, prefetch=prefetch)
        data = self._get_datum(rbtype)
        rb.add(data)
        s = rb.sample(1)[0]
        if isinstance(s, TensorDictBase):
            s = s.select(*data.keys(True), strict=False)
            data = data.select(*s.keys(True), strict=False)
            assert (s == data).all()
            assert list(s.keys(True, True))
        else:
            assert (s == data).all()

    def test_empty(self, rbtype, storage, size, prefetch):
        torch.manual_seed(0)
        rb = self._get_rb(rbtype, storage=storage, size=size, prefetch=prefetch)
        data = self._get_datum(rbtype)
        for _ in range(2):
            rb.add(data)
            s = rb.sample(1)[0]
            if isinstance(s, TensorDictBase):
                s = s.select(*data.keys(True), strict=False)
                data = data.select(*s.keys(True), strict=False)
                assert (s == data).all()
                assert list(s.keys(True, True))
            else:
                assert (s == data).all()
            rb.empty()
            with pytest.raises(
                RuntimeError, match="Cannot sample from an empty storage"
            ):
                rb.sample()

    def test_extend(self, rbtype, storage, size, prefetch):
        torch.manual_seed(0)
        rb = self._get_rb(rbtype, storage=storage, size=size, prefetch=prefetch)
        data = self._get_data(rbtype, size=5)
        cond = OLD_TORCH and size < len(data) and isinstance(rb.storage, TensorStorage)
        with (
            pytest.warns(
                UserWarning,
                match="A cursor of length superior to the storage capacity was provided",
            )
            if cond
            else contextlib.nullcontext()
        ):
            rb.extend(data)
        length = len(rb)
        for d in data[-length:]:
            for b in rb.storage:
                if isinstance(b, TensorDictBase):
                    keys = set(d.keys()).intersection(b.keys())
                    b = b.exclude("index").select(*keys, strict=False)
                    keys = set(d.keys()).intersection(b.keys())
                    d = d.select(*keys, strict=False)

                value = b == d
                if isinstance(value, (torch.Tensor, TensorDictBase)):
                    value = value.all()
                if value:
                    break
            else:
                raise RuntimeError("did not find match")

    def test_sample(self, rbtype, storage, size, prefetch):
        torch.manual_seed(0)
        rb = self._get_rb(rbtype, storage=storage, size=size, prefetch=prefetch)
        data = self._get_data(rbtype, size=5)
        cond = OLD_TORCH and size < len(data) and isinstance(rb.storage, TensorStorage)
        with (
            pytest.warns(
                UserWarning,
                match="A cursor of length superior to the storage capacity was provided",
            )
            if cond
            else contextlib.nullcontext()
        ):
            rb.extend(data)
        new_data = rb.sample()
        if not isinstance(new_data, (torch.Tensor, TensorDictBase)):
            new_data = new_data[0]

        for d in new_data:
            for b in data:
                if isinstance(b, TensorDictBase):
                    keys = set(d.keys()).intersection(b.keys())
                    b = b.exclude("index").select(*keys, strict=False)
                    keys = set(d.keys()).intersection(b.keys())
                    d = d.select(*keys, strict=False)

                value = b == d
                if isinstance(value, (torch.Tensor, TensorDictBase)):
                    value = value.all()
                if value:
                    break
            else:
                raise RuntimeError("did not find matching value")

    def test_index(self, rbtype, storage, size, prefetch):
        torch.manual_seed(0)
        rb = self._get_rb(rbtype, storage=storage, size=size, prefetch=prefetch)
        data = self._get_data(rbtype, size=5)
        cond = OLD_TORCH and size < len(data) and isinstance(rb.storage, TensorStorage)
        with (
            pytest.warns(
                UserWarning,
                match="A cursor of length superior to the storage capacity was provided",
            )
            if cond
            else contextlib.nullcontext()
        ):
            rb.extend(data)
        d1 = rb[2]
        d2 = rb.storage[2]
        if type(d1) is not type(d2):
            d1 = d1[0]
        b = d1 == d2
        if not isinstance(b, bool):
            b = b.all()
        assert b

    def test_index_nonfull(self, rbtype, storage, size, prefetch):
        # checks that indexing the buffer before it's full gives the accurate view of the data
        rb = self._get_rb(rbtype, storage=storage, size=size, prefetch=prefetch)
        data = self._get_data(rbtype, size=size - 1)
        rb.extend(data)
        assert len(rb[: size - 1]) == size - 1
        assert len(rb[size - 2 :]) == 1


def test_replay_buffer_set_at_():
    """Tests that set_at_ writes through to storage in-place."""
    rb = ReplayBuffer(
        storage=LazyTensorStorage(10),
        batch_size=5,
    )
    data = TensorDict({"a": torch.zeros(10), "b": torch.ones(10)}, batch_size=[10])
    rb.extend(data)
    # Modify key "a" at indices [2, 5]
    rb.set_at_("a", torch.tensor([99.0, 99.0]), torch.tensor([2, 5]))
    assert rb["a"][2] == 99.0
    assert rb["a"][5] == 99.0
    assert rb["a"][0] == 0.0  # unchanged
    assert rb["b"][2] == 1.0  # other key unchanged


def test_replay_buffer_set_():
    """Tests that set_ writes through to storage in-place."""
    rb = ReplayBuffer(
        storage=LazyTensorStorage(10),
        batch_size=5,
    )
    data = TensorDict({"a": torch.zeros(10), "b": torch.ones(10)}, batch_size=[10])
    rb.extend(data)
    rb.set_("a", torch.full((10,), 42.0))
    assert (rb["a"] == 42.0).all()
    assert (rb["b"] == 1.0).all()  # other key unchanged


def test_replay_buffer_update_():
    """Tests that update_ writes through to storage in-place."""
    rb = ReplayBuffer(
        storage=LazyTensorStorage(10),
        batch_size=5,
    )
    data = TensorDict({"a": torch.zeros(10), "b": torch.ones(10)}, batch_size=[10])
    rb.extend(data)
    update = TensorDict(
        {"a": torch.full((10,), 7.0), "b": torch.full((10,), 8.0)},
        batch_size=[10],
    )
    rb.update_(update)
    assert (rb["a"] == 7.0).all()
    assert (rb["b"] == 8.0).all()


def test_multi_loops():
    """Tests that one can iterate multiple times over a buffer without rep."""
    rb = ReplayBuffer(
        batch_size=5, storage=ListStorage(10), sampler=SamplerWithoutReplacement()
    )
    rb.extend(torch.zeros(10))
    for i, d in enumerate(rb):  # noqa: B007
        assert (d == 0).all()
    assert i == 1
    for i, d in enumerate(rb):  # noqa: B007
        assert (d == 0).all()
    assert i == 1


def test_batch_errors():
    """Tests error messages related to batch-size"""
    rb = ReplayBuffer(
        storage=ListStorage(10), sampler=SamplerWithoutReplacement(drop_last=False)
    )
    rb.extend(torch.zeros(10))
    rb.sample(3)  # that works
    with pytest.raises(
        RuntimeError,
        match="Cannot iterate over the replay buffer. Batch_size was not specified",
    ):
        for _ in rb:
            pass
    with pytest.raises(RuntimeError, match="batch_size not specified"):
        rb.sample()
    with pytest.raises(ValueError, match="Samplers with drop_last=True"):
        ReplayBuffer(
            storage=ListStorage(10), sampler=SamplerWithoutReplacement(drop_last=True)
        )
    # that works
    ReplayBuffer(
        storage=ListStorage(10),
    )
    rb = ReplayBuffer(
        storage=ListStorage(10),
        sampler=SamplerWithoutReplacement(drop_last=False),
        batch_size=3,
    )
    rb.extend(torch.zeros(10))
    for _ in rb:
        pass
    rb.sample()


@pytest.mark.skipif(not torchrl._utils.RL_WARNINGS, reason="RL_WARNINGS is not set")
def test_add_warning():
    if not rl_warnings():
        return
    rb = ReplayBuffer(storage=ListStorage(10), batch_size=3)
    with pytest.warns(
        UserWarning,
        match=r"Using `add\(\)` with a TensorDict that has batch_size",
    ):
        rb.add(TensorDict(batch_size=[1]))


@pytest.mark.parametrize("stack", [False, True])
@pytest.mark.parametrize("reduction", ["min", "max", "mean", "median"])
def test_rb_trajectories(stack, reduction):
    traj_td = TensorDict(
        {"obs": torch.randn(3, 4, 5), "actions": torch.randn(3, 4, 2)},
        batch_size=[3, 4],
    )
    if stack:
        traj_td = torch.stack([td.to_tensordict() for td in traj_td], 0)

    rb = TensorDictPrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.9,
        priority_key="td_error",
        storage=ListStorage(5),
        batch_size=3,
    )
    rb.extend(traj_td)
    sampled_td = rb.sample()
    sampled_td.set("td_error", torch.rand(3, 4))
    rb.update_tensordict_priority(sampled_td)
    sampled_td = rb.sample(include_info=True)
    assert (sampled_td.get("priority_weight") > 0).all()
    assert sampled_td.batch_size == torch.Size([3, 4])

    # set back the trajectory length
    sampled_td_filtered = sampled_td.to_tensordict().exclude(
        "priority_weight", "index", "td_error"
    )
    sampled_td_filtered.batch_size = [3, 4]


def test_shared_storage_prioritized_sampler():
    n = 100

    storage = LazyMemmapStorage(n)
    writer = RoundRobinWriter()
    sampler0 = RandomSampler()
    sampler1 = PrioritizedSampler(max_capacity=n, alpha=0.7, beta=1.1)

    rb0 = ReplayBuffer(storage=storage, writer=writer, sampler=sampler0, batch_size=10)
    rb1 = ReplayBuffer(storage=storage, writer=writer, sampler=sampler1, batch_size=10)

    data = TensorDict({"a": torch.arange(50)}, [50])

    # Extend rb0. rb1 should be aware of changes to storage.
    rb0.extend(data)

    assert len(rb0) == 50
    assert len(storage) == 50
    assert len(rb1) == 50

    rb0.sample()
    rb1.sample()

    assert rb1._sampler._sum_tree.query(0, 10) == 10
    assert rb1._sampler._sum_tree.query(0, 50) == 50
    assert rb1._sampler._sum_tree.query(0, 70) == 50


class TestReplayBufferConsumption:
    def test_replay_buffer_consume_after_one_list_storage(self):
        torch.manual_seed(0)
        data = [torch.tensor(i) for i in range(5)]
        rb = ReplayBuffer(
            storage=ListStorage(5),
            batch_size=3,
            consume_after_n_samples=1,
        )
        rb.extend(data)

        sample, info = rb.sample(return_info=True)

        assert len(sample) == 3
        assert len(rb) == 2
        assert set(sample.tolist()) == set(info["index"].tolist())

        sample = rb.sample()
        assert len(sample) == 2
        assert len(rb) == 0
        with pytest.raises(RuntimeError, match="Cannot sample from an empty storage"):
            rb.sample()

    def test_replay_buffer_consume_after_two_list_storage(self):
        torch.manual_seed(0)
        data = [torch.tensor(i) for i in range(4)]
        rb = ReplayBuffer(
            storage=ListStorage(4),
            batch_size=4,
            consume_after_n_samples=2,
        )
        rb.extend(data)

        first = rb.sample()
        assert set(first.tolist()) == set(range(4))
        assert len(rb) == 4

        second = rb.sample()
        assert set(second.tolist()) == set(range(4))
        assert len(rb) == 0

    def test_replay_buffer_consume_add_extend_after_exhaustion(self):
        torch.manual_seed(0)
        rb = ReplayBuffer(
            storage=ListStorage(10),
            consume_after_n_samples=1,
        )
        rb.extend([torch.tensor(0), torch.tensor(1)])
        assert set(rb.sample(2).tolist()) == {0, 1}
        assert len(rb) == 0

        rb.add(torch.tensor(2))
        assert len(rb) == 1
        assert rb.sample(1).item() == 2
        assert len(rb) == 0

        rb.extend([torch.tensor(3), torch.tensor(4), torch.tensor(5)])
        assert len(rb) == 3
        assert set(rb.sample(10).tolist()) == {3, 4, 5}
        assert len(rb) == 0

    def test_replay_buffer_consume_interleaved_writes(self):
        torch.manual_seed(0)
        rb = ReplayBuffer(
            storage=ListStorage(10),
            consume_after_n_samples=1,
        )
        rb.extend([torch.tensor(0), torch.tensor(1), torch.tensor(2)])
        first = set(rb.sample(2).tolist())
        assert len(first) == 2
        assert len(rb) == 1

        rb.extend([torch.tensor(3), torch.tensor(4)])
        second = set(rb.sample(10).tolist())
        assert second == set(range(5)) - first
        assert len(rb) == 0

    def test_replay_buffer_consume_reuses_consumed_slots_before_cursor(self):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(100),
            consume_after_n_samples=1,
        )
        rb.extend(torch.arange(100))
        rb._writer._cursor = 99

        consumed_index = torch.tensor([10, 20, 30, 40])
        rb._sampler._sample_count[consumed_index] = 1
        rb._sampler._live_mask[consumed_index] = False
        rb._sampler._append_free_indices(consumed_index)

        write_index = rb.extend(torch.arange(1000, 1004))

        assert write_index.tolist() == consumed_index.tolist()
        assert rb._writer._cursor == 99
        assert len(rb) == 100
        assert rb.storage.get(consumed_index).tolist() == [1000, 1001, 1002, 1003]

    def test_replay_buffer_consume_reuses_slots_then_cursor(self):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(100),
            consume_after_n_samples=1,
        )
        rb.extend(torch.arange(100))
        rb._writer._cursor = 99

        consumed_index = torch.tensor([10, 20])
        rb._sampler._sample_count[consumed_index] = 1
        rb._sampler._live_mask[consumed_index] = False
        rb._sampler._append_free_indices(consumed_index)

        write_index = rb.extend(torch.arange(1000, 1004))

        assert write_index.tolist() == [10, 20, 99, 0]
        assert rb._writer._cursor == 1
        assert len(rb) == 100
        assert rb.storage.get(torch.tensor([10, 20, 99, 0])).tolist() == [
            1000,
            1001,
            1002,
            1003,
        ]

    def test_replay_buffer_consume_reuses_slots_then_wraparound_cursor(self):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(100),
            consume_after_n_samples=1,
        )
        rb.extend(torch.arange(100))
        rb._writer._cursor = 99

        consumed_index = torch.tensor([10, 20, 30, 40])
        rb._sampler._sample_count[consumed_index] = 1
        rb._sampler._live_mask[consumed_index] = False
        rb._sampler._append_free_indices(consumed_index)

        write_index = rb.extend(torch.arange(1000, 1008))

        expected_index = torch.tensor([10, 20, 30, 40, 99, 0, 1, 2])
        assert write_index.tolist() == expected_index.tolist()
        assert rb._writer._cursor == 3
        assert len(rb) == 100
        assert rb.storage.get(expected_index).tolist() == list(range(1000, 1008))

    def test_replay_buffer_consume_skips_reused_slots_after_wraparound(self):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(100),
            consume_after_n_samples=1,
        )
        rb.extend(torch.arange(100))
        rb._writer._cursor = 0

        consumed_index = torch.tensor([10, 20, 30, 40])
        rb._sampler._sample_count[consumed_index] = 1
        rb._sampler._live_mask[consumed_index] = False
        rb._sampler._append_free_indices(consumed_index)

        write_index = rb.extend(torch.arange(1000, 1015))

        expected_index = torch.tensor(
            [10, 20, 30, 40, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11]
        )
        assert write_index.tolist() == expected_index.tolist()
        assert rb._writer._cursor == 12
        assert len(rb) == 100
        assert rb.storage.get(expected_index).tolist() == list(range(1000, 1015))

    def test_replay_buffer_consume_skips_reused_slots_when_advancing_cursor(self):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(8),
            consume_after_n_samples=1,
        )
        rb.extend(torch.arange(8))
        rb._writer._cursor = 0

        consumed_index = torch.tensor([1, 2])
        rb._sampler._sample_count[consumed_index] = 1
        rb._sampler._live_mask[consumed_index] = False
        rb._sampler._append_free_indices(consumed_index)

        write_index = rb.extend(torch.arange(100, 106))

        assert write_index.tolist() == [1, 2, 0, 3, 4, 5]
        assert rb._writer._cursor == 6
        assert rb.storage.get(write_index).tolist() == [100, 101, 102, 103, 104, 105]

    def test_replay_buffer_consume_reuse_and_append_keeps_tensor_storage_len(self):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(10),
            consume_after_n_samples=1,
        )
        rb.extend(torch.arange(5))

        consumed_index = torch.tensor([1, 2])
        rb._sampler._sample_count[consumed_index] = 1
        rb._sampler._live_mask[consumed_index] = False
        rb._sampler._append_free_indices(consumed_index)

        write_index = rb.extend(torch.arange(100, 104))

        assert write_index.tolist() == [1, 2, 5, 6]
        assert len(rb.storage) == 7
        assert len(rb) == 7
        assert rb.storage.get(write_index).tolist() == [100, 101, 102, 103]

    def test_replay_buffer_consume_iterates_until_exhausted(self):
        torch.manual_seed(0)
        rb = ReplayBuffer(
            storage=ListStorage(5),
            batch_size=2,
            sampler=ConsumingSampler(),
        )
        rb.extend([torch.tensor(i) for i in range(5)])

        samples = []
        for i, batch in enumerate(rb):
            if i == 5:
                raise RuntimeError("Iteration didn't terminate")
            samples.extend(batch.tolist())

        assert set(samples) == set(range(5))
        assert len(samples) == 5
        assert len(rb) == 0

    def test_replay_buffer_consume_state_dict_roundtrip(self):
        torch.manual_seed(0)
        data = [torch.tensor(i) for i in range(5)]
        rb = ReplayBuffer(
            storage=ListStorage(5),
            batch_size=5,
            consume_after_n_samples=2,
        )
        rb.extend(data)
        assert set(rb.sample().tolist()) == set(range(5))
        assert len(rb) == 5

        rb2 = ReplayBuffer(
            storage=ListStorage(5),
            batch_size=5,
            consume_after_n_samples=2,
        )
        rb2.load_state_dict(rb.state_dict())

        assert len(rb2) == 5
        assert set(rb2.sample().tolist()) == set(range(5))
        assert len(rb2) == 0

    def test_replay_buffer_consume_state_dict_preserves_free_slots(self):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(5),
            batch_size=2,
            consume_after_n_samples=1,
        )
        rb.extend(torch.arange(5))
        consumed = rb.sample(return_info=True)[1]["index"]

        rb2 = ReplayBuffer(
            storage=LazyTensorStorage(5),
            batch_size=2,
            consume_after_n_samples=1,
        )
        rb2.load_state_dict(rb.state_dict())

        write_index = rb2.extend(torch.arange(100, 102))

        assert write_index.tolist() == consumed.tolist()
        assert rb2.storage.get(write_index).tolist() == [100, 101]

    def test_replay_buffer_consume_load_state_dict_decouples_from_source(self):
        # loading a state dict must clone the incoming tensors so that
        # mutating the source afterwards does not corrupt the sampler state
        # (e.g. tensors mmap-backed by torch.load(mmap=True))
        rb = ReplayBuffer(
            storage=LazyTensorStorage(5),
            batch_size=2,
            consume_after_n_samples=1,
        )
        rb.extend(torch.arange(5))
        rb.sample()

        sampler_sd = rb.sampler.state_dict()
        sampler2 = ConsumingSampler()
        sampler2.load_state_dict(sampler_sd)

        before_count = sampler2._sample_count.clone()
        before_mask = sampler2._live_mask.clone()
        sampler_sd["_sample_count"].fill_(42)
        sampler_sd["_live_mask"].fill_(False)
        assert (sampler2._sample_count == before_count).all()
        assert (sampler2._live_mask == before_mask).all()
        if sampler_sd["_free_indices"] is not None:
            before_free = sampler2._free_indices.clone()
            sampler_sd["_free_indices"].fill_(-1)
            assert (sampler2._free_indices == before_free).all()

    def test_replay_buffer_consume_dumps_loads(self, tmpdir):
        torch.manual_seed(0)
        rb = ReplayBuffer(
            storage=LazyTensorStorage(5),
            batch_size=5,
            consume_after_n_samples=2,
        )
        rb.extend(torch.arange(5))
        assert set(rb.sample().tolist()) == set(range(5))

        rb2 = ReplayBuffer(
            storage=LazyTensorStorage(5),
            batch_size=5,
            consume_after_n_samples=2,
        )
        rb2.extend(torch.zeros(5, dtype=torch.long))
        rb.dumps(tmpdir)
        rb2.loads(tmpdir)

        assert len(rb2) == 5
        assert set(rb2.sample().tolist()) == set(range(5))
        assert len(rb2) == 0

    def test_replay_buffer_consume_tensor_storage_prepopulated(self):
        torch.manual_seed(0)
        rb = ReplayBuffer(
            storage=TensorStorage(torch.arange(5)),
            batch_size=3,
            consume_after_n_samples=1,
        )

        assert len(rb) == 5
        assert len(rb.sample()) == 3
        assert len(rb) == 2
        assert len(rb.sample()) == 2
        assert len(rb) == 0

    def test_tensordict_replay_buffer_consume_lazy_tensor_storage(self):
        torch.manual_seed(0)
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(5),
            consume_after_n_samples=1,
        )
        rb.extend(TensorDict({"a": torch.arange(4)}, [4]))

        sample = rb.sample(2)
        assert sample["index"].numel() == 2
        assert sample["index"].unique().numel() == 2
        assert len(rb) == 2

        sample = rb.sample(10)
        assert sample["index"].numel() == 2
        assert len(rb) == 0

    def test_tensordict_replay_buffer_consume_lazy_memmap_storage(self, tmpdir):
        torch.manual_seed(0)
        rb = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(5, scratch_dir=tmpdir),
            consume_after_n_samples=1,
        )
        rb.extend(TensorDict({"a": torch.arange(4)}, [4]))

        assert rb.sample(2)["index"].numel() == 2
        assert len(rb) == 2
        assert rb.sample(10)["index"].numel() == 2
        assert len(rb) == 0

    def test_replay_buffer_consume_unsupported_modes(self):
        with pytest.raises(ValueError, match="Prefetching is not supported"):
            ReplayBuffer(
                storage=ListStorage(10),
                batch_size=2,
                consume_after_n_samples=1,
                prefetch=1,
            )

        with pytest.raises(ValueError, match="Prefetching is not supported"):
            ReplayBuffer(
                storage=ListStorage(10),
                sampler=ConsumingSampler(),
                batch_size=2,
                prefetch=1,
            )

        with pytest.raises(ValueError, match="only supports the default RandomSampler"):
            ReplayBuffer(
                storage=ListStorage(10),
                sampler=SamplerWithoutReplacement(),
                consume_after_n_samples=1,
            )

        try:
            prioritized_sampler = PrioritizedSampler(10, 1.0, 1.0)
        except RuntimeError as err:
            if "SumSegmentTree" not in str(err):
                raise
        else:
            with pytest.raises(
                ValueError, match="only supports the default RandomSampler"
            ):
                ReplayBuffer(
                    storage=ListStorage(10),
                    sampler=prioritized_sampler,
                    consume_after_n_samples=1,
                )

        with pytest.raises(ValueError, match="1-dimensional storages"):
            ReplayBuffer(
                storage=LazyTensorStorage(10, ndim=2),
                sampler=ConsumingSampler(),
            )

    def test_replay_buffer_consume_requires_write_at_writer(self):
        class WriterWithoutWriteAt(RoundRobinWriter):
            write_at = None

        with pytest.raises(TypeError, match="requires a writer with a callable"):
            ReplayBuffer(
                storage=ListStorage(10),
                writer=WriterWithoutWriteAt(),
                consume_after_n_samples=1,
            )


@pytest.mark.parametrize("size", [10, 15, 20])
@pytest.mark.parametrize("drop_last", [True, False])
def test_replay_buffer_iter(size, drop_last):
    torch.manual_seed(0)
    storage = ListStorage(size)
    sampler = SamplerWithoutReplacement(drop_last=drop_last)
    writer = RoundRobinWriter()

    rb = ReplayBuffer(storage=storage, sampler=sampler, writer=writer, batch_size=3)
    rb.extend([torch.randint(100, (1,)) for _ in range(size)])

    for i, _ in enumerate(rb):
        if i == 20:
            # guard against infinite loop if error is introduced
            raise RuntimeError("Iteration didn't terminate")

    if drop_last:
        assert i == size // 3 - 1
    else:
        assert i == (size - 1) // 3


def test_replay_buffer_prefetch_queue_length():
    """Test that the prefetch queue maintains the correct length.

    This test verifies that after sampling from a replay buffer with prefetching
    enabled, the prefetch queue has exactly `prefetch` items computing in the
    background (no off-by-one error).
    """
    torch.manual_seed(0)

    rb = ReplayBuffer(storage=ListStorage(max_size=100), batch_size=2, prefetch=2)

    rb.extend(torch.arange(100))

    _ = rb.sample()

    assert (
        len(rb._prefetch_queue) == 2
    ), f"Expected prefetch queue to have 2 items, but got {len(rb._prefetch_queue)}."


class TestBufferStats:
    def test_stats_tracks_writes(self):
        rb = ReplayBuffer(storage=LazyTensorStorage(10), batch_size=2)
        stats = rb.stats()
        assert stats["size"] == 0
        assert stats["write_count"] == 0
        assert stats["capacity"] == 10
        assert stats["utilization"] == 0.0
        assert stats["prefetch_queue_size"] == 0
        assert stats["initialized"]
        rb.extend(torch.arange(15))
        stats = rb.stats()
        assert stats["size"] == 10
        assert stats["write_count"] == 15
        assert stats["utilization"] == 1.0

    def test_stats_is_side_effect_free(self):
        rb = ReplayBuffer(storage=LazyTensorStorage(10), batch_size=2)
        rb.extend(torch.arange(4))
        before = rb.stats()
        rb.stats()
        rb.sample()
        assert rb.stats() == before

    def test_stats_does_not_initialize_buffer(self):
        rb = ReplayBuffer(storage=LazyTensorStorage(10), delayed_init=True)
        stats = rb.stats()
        assert stats["initialized"] is False
        assert stats["size"] == 0
        assert stats["capacity"] == 10
        assert stats["utilization"] == 0.0
        assert not rb.initialized
        rb.extend(torch.arange(3))
        stats = rb.stats()
        assert stats["initialized"] is True
        assert stats["size"] == 3

    def test_stats_after_empty(self):
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
        rb.extend(torch.arange(6))
        rb.empty(empty_write_count=False)
        stats = rb.stats()
        assert stats["size"] == 0
        assert stats["write_count"] == 6
        rb.empty()
        stats = rb.stats()
        assert stats["write_count"] == 0

    def test_stats_is_serializable(self):
        rb = ReplayBuffer(storage=LazyTensorStorage(10))
        rb.extend(torch.arange(4))
        deserialized = json.loads(json.dumps(rb.stats()))
        assert deserialized["size"] == 4

    def test_stats_with_non_counting_writer(self):
        # dataset buffers use ImmutableDatasetWriter, which has no _write_count
        storage = LazyTensorStorage(10)
        rb = ReplayBuffer(storage=storage, writer=ImmutableDatasetWriter())
        storage.set(torch.arange(10), torch.zeros(10))
        stats = rb.stats()
        assert stats["size"] == 10
        assert stats["write_count"] == 0
        assert stats["capacity"] == 10


class _RepeatTwiceUnit(SampleUnit):
    """Toy unit doubling every anchor and recording per-record provenance."""

    def expand(self, index, info, storage):
        index = torch.as_tensor(index).repeat_interleave(2)
        info = dict(info)
        info["unit_repeat"] = torch.arange(index.numel()) % 2
        return index, info


class TestSampleUnit:
    """Executable spec for the SampleUnit composition point (#4039, PR 1).

    Contract pinned by this class:

    - ``sample_unit=None`` (default) and ``sample_unit=Transition()`` are
      behaviorally identical: same sampled data under the same generator
      state, same info entries.
    - A unit's ``expand`` runs after the anchor sampler and before storage
      read and index bookkeeping, so ``info["index"]`` reports the expanded
      indices and the returned batch is built from them.
    - Metadata a unit adds to ``info`` flows into ``sample(return_info=True)``
      and becomes keys of TensorDict samples.
    - ``sample_unit`` must be a ``SampleUnit`` instance; anything else raises
      ``TypeError`` at construction.
    """

    def test_default_and_transition_are_identical(self):
        data = torch.arange(20)
        samples = {}
        for name, unit in (("default", None), ("transition", Transition())):
            generator = torch.Generator()
            generator.manual_seed(0)
            rb = ReplayBuffer(
                storage=LazyTensorStorage(20),
                batch_size=4,
                generator=generator,
                sample_unit=unit,
            )
            rb.extend(data)
            samples[name] = rb.sample(return_info=True)
        default_sample, default_info = samples["default"]
        transition_sample, transition_info = samples["transition"]
        torch.testing.assert_close(default_sample, transition_sample)
        assert set(default_info) == set(transition_info)
        torch.testing.assert_close(
            torch.as_tensor(default_info["index"]),
            torch.as_tensor(transition_info["index"]),
        )

    def test_transition_adds_no_info_keys(self):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(10), batch_size=4, sample_unit=Transition()
        )
        rb.extend(torch.arange(10))
        _, info = rb.sample(return_info=True)
        assert "unit_repeat" not in info

    def test_custom_unit_expands_batch_and_metadata(self):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(10),
            batch_size=4,
            sample_unit=_RepeatTwiceUnit(),
        )
        rb.extend(torch.arange(10, dtype=torch.float32))
        sample, info = rb.sample(return_info=True)
        assert sample.shape[0] == 8
        torch.testing.assert_close(sample[0::2], sample[1::2])
        index = torch.as_tensor(info["index"])
        assert index.numel() == 8
        assert (index[0::2] == index[1::2]).all()
        assert info["unit_repeat"].numel() == 8

    def test_tensordict_buffer_carries_unit_metadata(self):
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(10),
            batch_size=4,
            sample_unit=_RepeatTwiceUnit(),
        )
        rb.extend(TensorDict({"obs": torch.randn(10, 3)}, batch_size=[10]))
        sample = rb.sample()
        assert sample.batch_size[0] == 8
        assert "unit_repeat" in sample.keys()
        torch.testing.assert_close(sample["obs"][0::2], sample["obs"][1::2])

    def test_invalid_sample_unit_raises(self):
        with pytest.raises(TypeError, match="sample_unit"):
            ReplayBuffer(storage=LazyTensorStorage(10), sample_unit=object())

    def test_prioritized_buffer_with_transition_unit(self):
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.9,
            storage=LazyTensorStorage(10),
            batch_size=4,
            sample_unit=Transition(),
        )
        rb.extend(TensorDict({"obs": torch.randn(10, 3)}, batch_size=[10]))
        sample = rb.sample()
        assert sample.batch_size[0] == 4
        rb.update_tensordict_priority(sample)


class TestSequenceUnit:
    """Executable spec for the Sequence sample unit (#4039, piece 2).

    Contract pinned by this class:

    - ``Sequence(length, episode_boundary="pad", done_key=("next","done"))``
      expands each anchor into the ``length`` records that follow it in
      stored-time order, wrapping physical ring indices when an episode spans
      the storage seam.
    - Boundary policies: ``"pad"`` keeps the anchor and marks entries past the
      episode end invalid, clamping their indices inside the episode;
      ``"stop"`` shifts the anchor backward so the sequence ends at the
      boundary (full-length, fully valid), falling back to pad behavior when
      the episode is shorter than ``length``; ``"include_reset"`` crosses the
      boundary with all entries valid.
    - ``expand`` adds per-record ``"sequence_id"``, ``"step_in_sequence"`` and
      ``"validity_mask"`` entries to ``info``, and expands per-anchor entries
      such as prioritized weights to the record count.
    - ``sample(batch_size=B)`` therefore returns ``B * length`` records.
    """

    def _sequence_cls(self):
        # Sequence is part of the public API: use the public import path.
        return Sequence

    def _make_storage(self, capacity=10, done_at=(5, 9)):
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(capacity), batch_size=4)
        size = capacity
        done = torch.zeros(size, 1, dtype=torch.bool)
        for idx in done_at:
            done[idx] = True
        rb.extend(
            TensorDict(
                {
                    "obs": torch.arange(size, dtype=torch.float32),
                    ("next", "done"): done,
                },
                batch_size=[size],
            )
        )
        return rb

    def _expand(self, rb, unit, anchors):
        index, info = unit.expand(
            torch.as_tensor(anchors, dtype=torch.long), {}, rb._storage
        )
        return torch.as_tensor(index), info

    def test_expansion_is_consecutive_within_episode(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage()
        index, info = self._expand(rb, Sequence(length=4), [1])
        assert index.tolist() == [1, 2, 3, 4]
        assert info["sequence_id"].tolist() == [0, 0, 0, 0]
        assert info["step_in_sequence"].tolist() == [0, 1, 2, 3]
        assert info["validity_mask"].all()

    def test_pad_masks_tail_and_clamps_inside_episode(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage()
        index, info = self._expand(rb, Sequence(length=4, episode_boundary="pad"), [4])
        assert index.tolist() == [4, 5, 5, 5]
        assert info["validity_mask"].tolist() == [True, True, False, False]
        obs = rb[:]["obs"][index]
        assert obs.tolist() == [4.0, 5.0, 5.0, 5.0]

    def test_stop_shifts_anchor_to_end_at_boundary(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage()
        index, info = self._expand(rb, Sequence(length=4, episode_boundary="stop"), [4])
        assert index.tolist() == [2, 3, 4, 5]
        assert info["validity_mask"].all()

    def test_stop_falls_back_to_pad_for_short_episode(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage(done_at=(1, 9))
        index, info = self._expand(rb, Sequence(length=4, episode_boundary="stop"), [0])
        assert index.tolist() == [0, 1, 1, 1]
        assert info["validity_mask"].tolist() == [True, True, False, False]

    def test_include_reset_crosses_boundary(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage()
        index, info = self._expand(
            rb, Sequence(length=4, episode_boundary="include_reset"), [4]
        )
        assert index.tolist() == [4, 5, 6, 7]
        assert info["validity_mask"].all()
        done = rb[:]["next", "done"].squeeze(-1)[index]
        assert done.tolist() == [False, True, False, False]

    def test_wraparound_seam(self):
        Sequence = self._sequence_cls()
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10), batch_size=4)
        size = 14
        rb.extend(
            TensorDict(
                {
                    "obs": torch.arange(size, dtype=torch.float32),
                    ("next", "done"): torch.zeros(size, 1, dtype=torch.bool),
                },
                batch_size=[size],
            )
        )
        index, info = self._expand(
            rb, Sequence(length=4, episode_boundary="include_reset"), [8]
        )
        assert index.tolist() == [8, 9, 0, 1]
        obs = rb[:]["obs"][index]
        assert obs.tolist() == [8.0, 9.0, 10.0, 11.0]
        assert info["validity_mask"].all()

    def test_multiple_anchors_sequence_ids(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage()
        index, info = self._expand(rb, Sequence(length=3), [0, 6])
        assert index.tolist() == [0, 1, 2, 6, 7, 8]
        assert info["sequence_id"].tolist() == [0, 0, 0, 1, 1, 1]
        assert info["step_in_sequence"].tolist() == [0, 1, 2, 0, 1, 2]

    def test_metadata_flows_into_tensordict_sample(self):
        Sequence = self._sequence_cls()
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(20),
            batch_size=2,
            sample_unit=Sequence(length=4),
        )
        rb.extend(
            TensorDict(
                {
                    "obs": torch.arange(20, dtype=torch.float32),
                    ("next", "done"): torch.zeros(20, 1, dtype=torch.bool),
                },
                batch_size=[20],
            )
        )
        sample = rb.sample()
        assert sample.batch_size[0] == 8
        for key in ("sequence_id", "step_in_sequence", "validity_mask"):
            assert key in sample.keys()
        valid = sample["validity_mask"].reshape(2, 4)
        obs = sample["obs"].reshape(2, 4)
        step = sample["step_in_sequence"].reshape(2, 4).float()
        starts = obs - step
        assert valid[:, 0].all()
        assert (valid.int().diff(dim=1) <= 0).all()
        for row in range(2):
            row_valid = valid[row]
            assert (starts[row][row_valid] == starts[row][0]).all()

    def test_prioritized_weights_expanded_to_records(self):
        Sequence = self._sequence_cls()
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(20),
            sampler=PrioritizedSampler(20, alpha=0.7, beta=0.9),
            batch_size=2,
            sample_unit=Sequence(length=4),
        )
        rb.extend(
            TensorDict(
                {
                    "obs": torch.arange(20, dtype=torch.float32),
                    ("next", "done"): torch.zeros(20, 1, dtype=torch.bool),
                },
                batch_size=[20],
            )
        )
        sample, info = rb.sample(return_info=True)
        assert sample.batch_size[0] == 8
        for value in info.values():
            assert torch.as_tensor(value).reshape(-1).shape[0] in (8,)

    def test_invalid_length_raises(self):
        Sequence = self._sequence_cls()
        with pytest.raises(ValueError):
            Sequence(length=0)
        with pytest.raises(ValueError):
            Sequence(length=4, episode_boundary="teleport")

    def test_include_reset_partially_filled_buffer(self):
        # Anchors close to the write head of a partially filled buffer must
        # not produce indices past the written region (repro: capacity 100,
        # 10 written, anchor 8, length 4 used to raise an IndexError).
        Sequence = self._sequence_cls()
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(100), batch_size=2)
        rb.extend(
            TensorDict(
                {
                    "obs": torch.arange(10, dtype=torch.float32),
                    ("next", "done"): torch.zeros(10, 1, dtype=torch.bool),
                },
                batch_size=[10],
            )
        )
        unit = Sequence(length=4, episode_boundary="include_reset")
        index, info = self._expand(rb, unit, [8])
        assert index.tolist() == [8, 9, 9, 9]
        assert info["validity_mask"].tolist() == [True, True, False, False]
        # reading the storage with the produced indices must not raise
        rb._storage.get(index)

    def test_include_reset_does_not_splice_across_write_cursor(self):
        # On a full ring buffer the record after the newest one is the oldest
        # record: include_reset must clamp at the write cursor instead of
        # splicing new and old data with an all-True validity mask.
        Sequence = self._sequence_cls()
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10), batch_size=2)
        size = 14
        rb.extend(
            TensorDict(
                {
                    "obs": torch.arange(size, dtype=torch.float32),
                    ("next", "done"): torch.zeros(size, 1, dtype=torch.bool),
                },
                batch_size=[size],
            )
        )
        # physical slots 0..3 hold obs 10..13 (newest at slot 3),
        # slots 4..9 hold obs 4..9 (oldest at slot 4)
        unit = Sequence(length=4, episode_boundary="include_reset")
        index, info = self._expand(rb, unit, [2])
        assert index.tolist() == [2, 3, 3, 3]
        assert info["validity_mask"].tolist() == [True, True, False, False]
        obs = rb[:]["obs"][index]
        assert obs.tolist() == [12.0, 13.0, 13.0, 13.0]

    def test_requires_tensordict_storage(self):
        Sequence = self._sequence_cls()
        unit = Sequence(length=3)
        rb = ReplayBuffer(storage=ListStorage(10), batch_size=2)
        with pytest.raises(TypeError, match="TensorStorage"):
            unit.expand(torch.tensor([0, 1]), {}, rb._storage)
        # plain-tensor TensorStorage is rejected as well
        rb = ReplayBuffer(storage=LazyTensorStorage(10), batch_size=2)
        rb.extend(torch.arange(5))
        with pytest.raises(TypeError, match="TensorDict"):
            unit.expand(torch.tensor([0, 1]), {}, rb._storage)

    def test_scalar_info_entries_pass_through(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage()
        unit = Sequence(length=3)
        _, info = unit.expand(
            torch.tensor([0, 6]),
            {"scalar_meta": 3.0, "per_anchor": torch.tensor([1.0, 2.0])},
            rb._storage,
        )
        assert info["scalar_meta"] == 3.0
        assert info["per_anchor"].tolist() == [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]

    def test_custom_nested_done_key(self):
        Sequence = self._sequence_cls()
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10), batch_size=2)
        done = torch.zeros(10, 1, dtype=torch.bool)
        done[5] = done[9] = True
        rb.extend(
            TensorDict(
                {
                    "obs": torch.arange(10, dtype=torch.float32),
                    ("stats", "episode_end"): done,
                },
                batch_size=[10],
            )
        )
        unit = Sequence(length=4, done_key=("stats", "episode_end"))
        index, info = self._expand(rb, unit, [4])
        assert index.tolist() == [4, 5, 5, 5]
        assert info["validity_mask"].tolist() == [True, True, False, False]

    @pytest.mark.parametrize(
        ("boundary", "expected_index", "expected_validity"),
        [
            ("pad", [3, 4, 4], [True, True, False]),
            ("stop", [2, 3, 4], [True, True, True]),
        ],
    )
    def test_none_done_key_uses_storage_boundary(
        self, boundary, expected_index, expected_validity
    ):
        Sequence = self._sequence_cls()
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10), batch_size=2)
        rb.extend(TensorDict({"obs": torch.arange(5)}, batch_size=[5]))
        index, info = self._expand(
            rb,
            Sequence(length=3, episode_boundary=boundary, done_key=None),
            [3],
        )
        assert index.tolist() == expected_index
        assert info["validity_mask"].tolist() == expected_validity

    @pytest.mark.gpu
    @pytest.mark.skipif(
        not torch.cuda.is_available() and not torch.backends.mps.is_available(),
        reason="needs a non-CPU device (CUDA or MPS)",
    )
    def test_non_cpu_storage(self):
        # Storage on an accelerator with anchors on CPU (as samplers produce
        # them): expansion must not mix devices, and the returned indices
        # live on the anchor device like Transition's.
        Sequence = self._sequence_cls()
        device = "cuda" if torch.cuda.is_available() else "mps"
        done = torch.zeros(10, 1, dtype=torch.bool)
        done[5] = done[9] = True
        data = TensorDict(
            {
                "obs": torch.arange(10, dtype=torch.float32),
                ("next", "done"): done,
            },
            batch_size=[10],
        ).to(device)
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(10, device=device),
            batch_size=2,
            sample_unit=Sequence(length=3),
        )
        rb.extend(data)
        anchors = torch.tensor([4], dtype=torch.long)
        for boundary in ("pad", "stop", "include_reset"):
            unit = Sequence(length=4, episode_boundary=boundary)
            index, info = unit.expand(anchors, {}, rb._storage)
            assert index.device == anchors.device
            assert info["validity_mask"].device == anchors.device
        sample = rb.sample()
        assert sample["obs"].shape[0] == 6


class TestSequenceBurnInBootstrap:
    """Executable spec for Sequence burn-in, bootstrap and stride (#4039, piece 3).

    Contract pinned by this class:

    - ``Sequence(..., burn_in=0, bootstrap=0, stride=1)`` extends the window
      around each anchor: ``burn_in`` records before the anchor, the learning
      region of ``length`` records starting at the anchor, then ``bootstrap``
      records after it. Total records per anchor:
      ``burn_in + length + bootstrap``.
    - A new per-record ``"learning_mask"`` info entry is True exactly on the
      learning region; ``"validity_mask"`` keeps its meaning (real, in-episode
      data). ``"step_in_sequence"`` runs 0..total-1 across the whole window.
    - Burn-in never shifts the anchor: entries before the anchor's episode
      start are invalid and clamped to the episode start, whatever the
      boundary policy. Bootstrap entries follow ``episode_boundary`` at the
      episode end like any tail entry.
    - ``stride`` spaces the records of the window uniformly.
    - Defaults reproduce the base Sequence behavior exactly.
    - ``burn_in < 0``, ``bootstrap < 0`` or ``stride < 1`` raise ``ValueError``.
    """

    def _sequence_cls(self):
        # Sequence is part of the public API: use the public import path.
        return Sequence

    def _make_storage(self, capacity=10, done_at=(5, 9)):
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(capacity), batch_size=4)
        done = torch.zeros(capacity, 1, dtype=torch.bool)
        for idx in done_at:
            done[idx] = True
        rb.extend(
            TensorDict(
                {
                    "obs": torch.arange(capacity, dtype=torch.float32),
                    ("next", "done"): done,
                },
                batch_size=[capacity],
            )
        )
        return rb

    def _expand(self, rb, unit, anchors):
        index, info = unit.expand(
            torch.as_tensor(anchors, dtype=torch.long), {}, rb._storage
        )
        return torch.as_tensor(index), info

    def test_burn_in_prepends_records(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage()
        index, info = self._expand(rb, Sequence(length=2, burn_in=2), [8])
        assert index.tolist() == [6, 7, 8, 9]
        assert info["learning_mask"].tolist() == [False, False, True, True]
        assert info["validity_mask"].all()
        assert info["step_in_sequence"].tolist() == [0, 1, 2, 3]

    def test_burn_in_clamped_at_episode_start(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage()
        index, info = self._expand(rb, Sequence(length=2, burn_in=2), [6])
        assert index.tolist() == [6, 6, 6, 7]
        assert info["validity_mask"].tolist() == [False, False, True, True]
        assert info["learning_mask"].tolist() == [False, False, True, True]

    def test_bootstrap_appends_records(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage()
        index, info = self._expand(rb, Sequence(length=2, bootstrap=1), [6])
        assert index.tolist() == [6, 7, 8]
        assert info["learning_mask"].tolist() == [True, True, False]
        assert info["validity_mask"].all()

    def test_bootstrap_masked_at_episode_end(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage()
        index, info = self._expand(
            rb, Sequence(length=2, bootstrap=1, episode_boundary="pad"), [8]
        )
        assert index.tolist() == [8, 9, 9]
        assert info["validity_mask"].tolist() == [True, True, False]
        assert info["learning_mask"].tolist() == [True, True, False]

    def test_stop_shifts_bootstrap_inside_episode(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage()
        index, info = self._expand(
            rb, Sequence(length=2, bootstrap=1, episode_boundary="stop"), [4]
        )
        assert index.tolist() == [3, 4, 5]
        assert info["validity_mask"].tolist() == [True, True, True]
        assert info["learning_mask"].tolist() == [True, True, False]
        assert info["anchor_index"].tolist() == [4, 4, 4]

    def test_stride_spaces_the_window(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage(done_at=(9,))
        index, info = self._expand(rb, Sequence(length=3, stride=2), [0])
        assert index.tolist() == [0, 2, 4]
        assert info["step_in_sequence"].tolist() == [0, 1, 2]
        assert info["validity_mask"].all()

    def test_defaults_match_base_sequence(self):
        Sequence = self._sequence_cls()
        rb = self._make_storage()
        base_index, base_info = self._expand(rb, Sequence(length=3), [1, 6])
        ext_index, ext_info = self._expand(
            rb, Sequence(length=3, burn_in=0, bootstrap=0, stride=1), [1, 6]
        )
        assert base_index.tolist() == ext_index.tolist()
        torch.testing.assert_close(
            base_info["validity_mask"], ext_info["validity_mask"]
        )
        assert ext_info["learning_mask"].all()

    def test_window_size_through_sampling(self):
        Sequence = self._sequence_cls()
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(30),
            batch_size=2,
            sample_unit=Sequence(length=3, burn_in=2, bootstrap=1),
        )
        rb.extend(
            TensorDict(
                {
                    "obs": torch.arange(30, dtype=torch.float32),
                    ("next", "done"): torch.zeros(30, 1, dtype=torch.bool),
                },
                batch_size=[30],
            )
        )
        sample = rb.sample()
        assert sample.batch_size[0] == 2 * (2 + 3 + 1)
        assert "learning_mask" in sample.keys()
        assert sample["learning_mask"].sum() == 2 * 3

    def test_include_reset_does_not_read_unwritten_slots(self):
        Sequence = self._sequence_cls()
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(20), batch_size=2)
        rb.extend(
            TensorDict(
                {
                    "obs": torch.arange(10, dtype=torch.float32),
                    ("next", "done"): torch.zeros(10, 1, dtype=torch.bool),
                },
                batch_size=[10],
            )
        )
        backward = Sequence(length=3, burn_in=2, episode_boundary="include_reset")
        index, info = backward.expand(torch.tensor([0]), {}, rb._storage)
        assert index.min() >= 0
        assert index.max() < 10
        assert info["validity_mask"].tolist() == [False, False, True, True, True]
        forward = Sequence(length=4, episode_boundary="include_reset")
        index, info = forward.expand(torch.tensor([8]), {}, rb._storage)
        assert index.max() < 10
        assert info["validity_mask"].tolist() == [True, True, False, False]

    def test_validation(self):
        Sequence = self._sequence_cls()
        with pytest.raises(ValueError):
            Sequence(length=3, burn_in=-1)
        with pytest.raises(ValueError):
            Sequence(length=3, bootstrap=-1)
        with pytest.raises(ValueError):
            Sequence(length=3, stride=0)

    def test_include_reset_burn_in_respects_write_cursor(self):
        # Burn-in walks backward from the anchor: on a full ring buffer it
        # must clamp at the oldest record instead of wrapping past the write
        # seam into the newest data.
        Sequence = self._sequence_cls()
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10), batch_size=2)
        size = 14
        rb.extend(
            TensorDict(
                {
                    "obs": torch.arange(size, dtype=torch.float32),
                    ("next", "done"): torch.zeros(size, 1, dtype=torch.bool),
                },
                batch_size=[size],
            )
        )
        # physical slots 0..3 hold obs 10..13 (newest at slot 3),
        # slots 4..9 hold obs 4..9 (oldest at slot 4)
        unit = Sequence(length=2, burn_in=2, episode_boundary="include_reset")
        index, info = unit.expand(torch.tensor([5]), {}, rb._storage)
        assert index.tolist() == [4, 4, 5, 6]
        assert info["validity_mask"].tolist() == [False, True, True, True]
        obs = rb[:]["obs"][index]
        assert obs.tolist() == [4.0, 4.0, 5.0, 6.0]


class TestSequencePrioritySemantics:
    """Executable spec for sequence priority semantics and distribution
    invariance (#4039, piece 4, first half).

    Contract pinned by this class:

    - Priorities live per anchor. The Sequence unit adds a per-record
      ``"anchor_index"`` info entry (the storage index of each record's
      anchor) so priorities can be updated for the anchors of sampled
      sequences through the ordinary ``update_priority`` path.
    - Per-anchor sampler entries such as importance weights are expanded
      block-constant: reshaped to ``[anchors, window]``, every row is
      constant.
    - Range expansion does not change the anchor selection distribution:
      anchors drawn through a Sequence unit follow the same distribution as
      anchors drawn through Transition, both for uniform and prioritized
      sampling.

    The distribution tests are statistical: seeded, with wide tolerance
    bands chosen to keep them deterministic in CI.
    """

    def _sequence_cls(self):
        # Sequence is part of the public API: use the public import path.
        return Sequence

    def _make_rb(self, sampler=None, unit=None, capacity=20, batch_size=4):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(capacity),
            sampler=sampler if sampler is not None else RandomSampler(),
            batch_size=batch_size,
            sample_unit=unit,
            generator=torch.Generator().manual_seed(0),
        )
        rb.extend(
            TensorDict(
                {
                    "obs": torch.arange(capacity, dtype=torch.float32),
                    ("next", "done"): torch.zeros(capacity, 1, dtype=torch.bool),
                },
                batch_size=[capacity],
            )
        )
        return rb

    def _anchor_counts(self, rb, unit_length, draws=400, capacity=20):
        counts = torch.zeros(capacity)
        for _ in range(draws):
            _, info = rb.sample(return_info=True)
            if unit_length == 1:
                anchors = torch.as_tensor(info["index"]).reshape(-1)
            else:
                anchors = torch.as_tensor(info["anchor_index"]).reshape(-1)[
                    ::unit_length
                ]
            counts += torch.bincount(anchors, minlength=capacity)
        return counts / counts.sum()

    def test_anchor_index_metadata(self):
        Sequence = self._sequence_cls()
        rb = self._make_rb()
        index, info = Sequence(length=3).expand(torch.tensor([1, 6]), {}, rb._storage)
        assert info["anchor_index"].tolist() == [1, 1, 1, 6, 6, 6]

    def test_user_anchor_index_ignored_without_sequence_unit(self):
        rb = TensorDictReplayBuffer(storage=LazyTensorStorage(4))
        data = TensorDict(
            {
                "index": torch.tensor([0, 1]),
                "anchor_index": torch.tensor([3, 3]),
                "td_error": torch.tensor([1.0, 2.0]),
            },
            batch_size=[2],
        )
        assert rb._anchor_reduced_priority(data, data["td_error"]) is None

    def test_weights_are_block_constant(self):
        Sequence = self._sequence_cls()
        length = 4
        rb = self._make_rb(
            sampler=PrioritizedSampler(20, alpha=0.7, beta=0.9),
            unit=Sequence(length=length),
        )
        rb.update_priority(index=torch.arange(20), priority=torch.rand(20) + 0.1)
        _, info = rb.sample(return_info=True)
        records = 4 * length
        structural = {
            "index",
            "index_generation",
            "sequence_id",
            "step_in_sequence",
            "anchor_index",
        }
        checked = 0
        for key, value in info.items():
            value = torch.as_tensor(value).reshape(-1)
            if (
                key in structural
                or value.numel() != records
                or value.dtype == torch.bool
            ):
                continue
            blocks = value.reshape(4, length).float()
            torch.testing.assert_close(blocks, blocks[:, :1].expand_as(blocks))
            checked += 1
        assert checked > 0

    def test_uniform_anchor_distribution_unchanged(self):
        Sequence = self._sequence_cls()
        expected = 1.0 / 20
        for unit, unit_length in ((None, 1), (Sequence(length=2), 2)):
            rb = self._make_rb(unit=unit)
            freqs = self._anchor_counts(rb, unit_length)
            assert (freqs > 0.4 * expected).all()
            assert (freqs < 1.9 * expected).all()

    def test_prioritized_anchor_distribution_unchanged(self):
        Sequence = self._sequence_cls()
        priorities = torch.ones(20)
        priorities[10:] = 9.0
        for unit, unit_length in ((None, 1), (Sequence(length=2), 2)):
            rb = self._make_rb(
                sampler=PrioritizedSampler(20, alpha=1.0, beta=1.0),
                unit=unit,
            )
            rb.update_priority(index=torch.arange(20), priority=priorities)
            freqs = self._anchor_counts(rb, unit_length)
            high_share = freqs[10:].sum().item()
            assert 0.8 < high_share < 0.98

    def test_update_tensordict_priority_routes_to_anchors(self):
        # update_tensordict_priority on an expanded sample must write to the
        # anchor slots only, reducing per-record priorities with a max over
        # each anchor's valid records: padded records never pollute the
        # priorities of unrelated storage slots, and duplicate anchors are
        # reduced before writing (well-defined, last-write-wins-free).
        Sequence = self._sequence_cls()
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=1.0,
            beta=1.0,
            storage=LazyTensorStorage(20),
            batch_size=2,
            sample_unit=Sequence(length=4),
        )
        done = torch.zeros(20, 1, dtype=torch.bool)
        done[9] = done[19] = True
        rb.extend(
            TensorDict(
                {
                    "obs": torch.arange(20, dtype=torch.float32),
                    ("next", "done"): done,
                },
                batch_size=[20],
            )
        )
        rb.update_priority(index=torch.arange(20), priority=torch.ones(20))
        # deterministic "sample": two windows sharing anchor 7, crossing the
        # episode end at 9 (last record padded and invalid)
        unit = Sequence(length=4, episode_boundary="pad")
        index, info = unit.expand(torch.tensor([7, 7]), {}, rb._storage)
        data = rb._storage.get(index)
        data.set("index", index)
        data.set("anchor_index", info["anchor_index"])
        data.set("validity_mask", info["validity_mask"])
        assert info["validity_mask"].tolist() == [True, True, True, False] * 2
        # huge errors on the padded (invalid) records must be ignored
        data.set("td_error", torch.tensor([1.0, 2.0, 3.0, 50.0, 4.0, 5.0, 6.0, 60.0]))
        before = torch.tensor([float(rb.sampler._sum_tree[i]) for i in range(20)])
        rb.update_tensordict_priority(data)
        after = torch.tensor([float(rb.sampler._sum_tree[i]) for i in range(20)])
        changed = (before != after).nonzero().squeeze(-1)
        assert changed.tolist() == [7]
        # max over the valid records of both duplicate windows
        assert after[7].item() == pytest.approx(6.0, rel=1e-4)
        # end-to-end: a sampled batch carries the metadata automatically and
        # only anchor slots are touched
        sample = rb.sample()
        sample.set("td_error", torch.rand(sample.shape[0]) + 0.5)
        before = torch.tensor([float(rb.sampler._sum_tree[i]) for i in range(20)])
        rb.update_tensordict_priority(sample)
        after = torch.tensor([float(rb.sampler._sum_tree[i]) for i in range(20)])
        changed = set((before != after).nonzero().squeeze(-1).tolist())
        anchors = set(sample.get("anchor_index").reshape(-1).tolist())
        assert changed.issubset(anchors)

    def test_update_priority_via_anchor_index(self):
        Sequence = self._sequence_cls()
        rb = self._make_rb(
            sampler=PrioritizedSampler(20, alpha=1.0, beta=1.0),
            unit=Sequence(length=2),
        )
        rb.update_priority(index=torch.arange(20), priority=torch.ones(20))
        _, info = rb.sample(return_info=True)
        boosted = int(torch.as_tensor(info["anchor_index"]).reshape(-1)[0])
        rb.update_priority(
            index=torch.tensor([boosted]), priority=torch.tensor([1000.0])
        )
        freqs = self._anchor_counts(rb, unit_length=2, draws=200)
        assert freqs[boosted] > 0.5


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
