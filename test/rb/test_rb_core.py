# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import functools

import pytest
import torch
import torchrl
from _rb_common import OLD_TORCH, ReplayBufferRNG, TensorDictReplayBufferRNG
from tensordict import assert_allclose_td, TensorDict, TensorDictBase

from torchrl._utils import rl_warnings
from torchrl.data import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
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
from torchrl.data.replay_buffers.writers import RoundRobinWriter
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
