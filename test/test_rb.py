# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import numpy as np
import pytest
import torch
from _utils_internal import get_available_devices
from torchrl.data import (
    PrioritizedReplayBuffer,
    ReplayBuffer,
    TensorDict,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import (
    rb_prototype,
    samplers,
    TensorDictPrioritizedReplayBuffer,
    writers,
)
from torchrl.data.replay_buffers.storages import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ListStorage,
)
from torchrl.data.tensordict.tensordict import assert_allclose_td, TensorDictBase


collate_fn_dict = {
    ListStorage: lambda x: torch.stack(x, 0),
    LazyTensorStorage: lambda x: x,
    LazyMemmapStorage: lambda x: x,
    None: lambda x: torch.stack(x, 0),
}


@pytest.mark.parametrize(
    "rb_type", [rb_prototype.ReplayBuffer, rb_prototype.TensorDictReplayBuffer]
)
@pytest.mark.parametrize(
    "sampler", [samplers.RandomSampler, samplers.PrioritizedSampler]
)
@pytest.mark.parametrize("writer", [writers.RoundRobinWriter])
@pytest.mark.parametrize("storage", [ListStorage, LazyTensorStorage, LazyMemmapStorage])
@pytest.mark.parametrize("size", [3, 100])
class TestPrototypeBuffers:
    def _get_rb(self, rb_type, size, sampler, writer, storage):
        collate_fn = collate_fn_dict[storage]

        if storage is not None:
            storage = storage(size)

        sampler_args = {}
        if sampler is samplers.PrioritizedSampler:
            sampler_args = {"max_capacity": size, "alpha": 0.8, "beta": 0.9}

        sampler = sampler(**sampler_args)
        writer = writer()
        rb = rb_type(
            collate_fn=collate_fn, storage=storage, sampler=sampler, writer=writer
        )
        return rb

    def _get_datum(self, rb_type):
        if rb_type is rb_prototype.ReplayBuffer:
            data = torch.randint(100, (1,))
        elif rb_type is rb_prototype.TensorDictReplayBuffer:
            data = TensorDict({"a": torch.randint(100, (1,))}, [])
        else:
            raise NotImplementedError(rb_type)
        return data

    def _get_data(self, rbtype, size):
        if rbtype is rb_prototype.ReplayBuffer:
            data = torch.randint(100, (size, 1))
        elif rbtype is rb_prototype.TensorDictReplayBuffer:
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

    def test_add(self, rb_type, sampler, writer, storage, size):
        torch.manual_seed(0)
        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        data = self._get_datum(rb_type)
        rb.add(data)
        s = rb._storage[0]
        if isinstance(s, TensorDictBase):
            assert (s == data.select(*s.keys())).all()
        else:
            assert (s == data).all()

    def test_extend(self, rb_type, sampler, writer, storage, size):
        torch.manual_seed(0)
        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        data = self._get_data(rb_type, size=5)
        rb.extend(data)
        length = len(rb)
        for d in data[-length:]:
            found_similar = False
            for b in rb._storage:
                if isinstance(b, TensorDictBase):
                    b = b.exclude("index").select(*set(d.keys()).intersection(b.keys()))
                    d = d.select(*set(d.keys()).intersection(b.keys()))

                value = b == d
                if isinstance(value, (torch.Tensor, TensorDictBase)):
                    value = value.all()
                if value:
                    found_similar = True
                    break
            assert found_similar

    def test_sample(self, rb_type, sampler, writer, storage, size):
        torch.manual_seed(0)
        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        data = self._get_data(rb_type, size=5)
        rb.extend(data)
        new_data = rb.sample(3)
        if not isinstance(new_data, (torch.Tensor, TensorDictBase)):
            new_data = new_data[0]

        for d in new_data:
            found_similar = False
            for b in data:
                if isinstance(b, TensorDictBase):
                    b = b.exclude("index").select(*set(d.keys()).intersection(b.keys()))
                    d = d.select(*set(d.keys()).intersection(b.keys()))

                value = b == d
                if isinstance(value, (torch.Tensor, TensorDictBase)):
                    value = value.all()
                if value:
                    found_similar = True
                    break
            if not found_similar:
                d
            assert found_similar, (d, data)

    def test_index(self, rb_type, sampler, writer, storage, size):
        torch.manual_seed(0)
        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        data = self._get_data(rb_type, size=5)
        rb.extend(data)
        d1 = rb[2]
        d2 = rb._storage[2]
        if type(d1) is not type(d2):
            d1 = d1[0]
        b = d1 == d2
        if not isinstance(b, bool):
            b = b.all()
        assert b


@pytest.mark.parametrize("priority_key", ["pk", "td_error"])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("device", get_available_devices())
def test_prototype_prb(priority_key, contiguous, device):
    torch.manual_seed(0)
    np.random.seed(0)
    rb = rb_prototype.TensorDictReplayBuffer(
        sampler=samplers.PrioritizedSampler(5, alpha=0.7, beta=0.9),
        collate_fn=None if contiguous else lambda x: torch.stack(x, 0),
        priority_key=priority_key,
    )
    td1 = TensorDict(
        source={
            "a": torch.randn(3, 1),
            priority_key: torch.rand(3, 1) / 10,
            "_idx": torch.arange(3).view(3, 1),
        },
        batch_size=[3],
    ).to(device)
    rb.extend(td1)
    s, _ = rb.sample(2)
    assert s.batch_size == torch.Size(
        [
            2,
        ]
    )
    assert (td1[s.get("_idx").squeeze()].get("a") == s.get("a")).all()
    assert_allclose_td(td1[s.get("_idx").squeeze()].select("a"), s.select("a"))

    # test replacement
    td2 = TensorDict(
        source={
            "a": torch.randn(5, 1),
            priority_key: torch.rand(5, 1) / 10,
            "_idx": torch.arange(5).view(5, 1),
        },
        batch_size=[5],
    ).to(device)
    rb.extend(td2)
    s, _ = rb.sample(5)
    assert s.batch_size == torch.Size([5])
    assert (td2[s.get("_idx").squeeze()].get("a") == s.get("a")).all()
    assert_allclose_td(td2[s.get("_idx").squeeze()].select("a"), s.select("a"))

    # test strong update
    # get all indices that match first item
    idx = s.get("_idx")
    idx_match = (idx == idx[0]).nonzero()[:, 0]
    s.set_at_(
        priority_key,
        torch.ones(
            idx_match.numel(),
            1,
            device=device,
        )
        * 100000000,
        idx_match,
    )
    val = s.get("a")[0]

    idx0 = s.get("_idx")[0]
    rb.update_tensordict_priority(s)
    s, _ = rb.sample(5)
    assert (val == s.get("a")).sum() >= 1
    torch.testing.assert_allclose(
        td2[idx0].get("a").view(1), s.get("a").unique().view(1)
    )

    # test updating values of original td
    td2.set_("a", torch.ones_like(td2.get("a")))
    s, _ = rb.sample(5)
    torch.testing.assert_allclose(
        td2[idx0].get("a").view(1), s.get("a").unique().view(1)
    )


@pytest.mark.parametrize("stack", [False, True])
def test_rb_prototype_trajectories(stack):
    traj_td = TensorDict(
        {"obs": torch.randn(3, 4, 5), "actions": torch.randn(3, 4, 2)},
        batch_size=[3, 4],
    )
    if stack:
        traj_td = torch.stack([td.to_tensordict() for td in traj_td], 0)

    rb = rb_prototype.TensorDictReplayBuffer(
        sampler=samplers.PrioritizedSampler(
            5,
            alpha=0.7,
            beta=0.9,
        ),
        collate_fn=lambda x: torch.stack(x, 0),
        priority_key="td_error",
    )
    rb.extend(traj_td)
    sampled_td, _ = rb.sample(3)
    sampled_td.set("td_error", torch.rand(3))
    rb.update_tensordict_priority(sampled_td)
    sampled_td, _ = rb.sample(3, include_info=True)
    assert (sampled_td.get("_weight") > 0).all()
    assert sampled_td.batch_size == torch.Size([3])

    # set back the trajectory length
    sampled_td_filtered = sampled_td.to_tensordict().exclude(
        "_weight", "index", "td_error"
    )
    sampled_td_filtered.batch_size = [3, 4]


@pytest.mark.parametrize(
    "rbtype,storage",
    [
        (ReplayBuffer, None),
        (ReplayBuffer, ListStorage),
        (PrioritizedReplayBuffer, None),
        (PrioritizedReplayBuffer, ListStorage),
        (TensorDictReplayBuffer, None),
        (TensorDictReplayBuffer, ListStorage),
        (TensorDictReplayBuffer, LazyTensorStorage),
        (TensorDictReplayBuffer, LazyMemmapStorage),
        (TensorDictPrioritizedReplayBuffer, None),
        (TensorDictPrioritizedReplayBuffer, ListStorage),
        (TensorDictPrioritizedReplayBuffer, LazyTensorStorage),
        (TensorDictPrioritizedReplayBuffer, LazyMemmapStorage),
    ],
)
@pytest.mark.parametrize("size", [3, 100])
@pytest.mark.parametrize("prefetch", [0])
class TestBuffers:
    _default_params_rb = {}
    _default_params_td_rb = {}
    _default_params_prb = {"alpha": 0.8, "beta": 0.9}
    _default_params_td_prb = {"alpha": 0.8, "beta": 0.9}

    def _get_rb(self, rbtype, size, storage, prefetch):
        collate_fn = collate_fn_dict[storage]
        if storage is not None:
            storage = storage(size)
        if rbtype is ReplayBuffer:
            params = self._default_params_rb
        elif rbtype is PrioritizedReplayBuffer:
            params = self._default_params_prb
        elif rbtype is TensorDictReplayBuffer:
            params = self._default_params_td_rb
        elif rbtype is TensorDictPrioritizedReplayBuffer:
            params = self._default_params_td_prb
        else:
            raise NotImplementedError(rbtype)
        rb = rbtype(
            size=size,
            storage=storage,
            prefetch=prefetch,
            collate_fn=collate_fn,
            **params
        )
        return rb

    def _get_datum(self, rbtype):
        if rbtype is ReplayBuffer:
            data = torch.randint(100, (1,))
        elif rbtype is PrioritizedReplayBuffer:
            data = torch.randint(100, (1,))
        elif rbtype is TensorDictReplayBuffer:
            data = TensorDict({"a": torch.randint(100, (1,))}, [])
        elif rbtype is TensorDictPrioritizedReplayBuffer:
            data = TensorDict({"a": torch.randint(100, (1,))}, [])
        else:
            raise NotImplementedError(rbtype)
        return data

    def _get_data(self, rbtype, size):
        if rbtype is ReplayBuffer:
            data = [torch.randint(100, (1,)) for _ in range(size)]
        elif rbtype is PrioritizedReplayBuffer:
            data = [torch.randint(100, (1,)) for _ in range(size)]
        elif rbtype is TensorDictReplayBuffer:
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

    def test_add(self, rbtype, storage, size, prefetch):
        torch.manual_seed(0)
        rb = self._get_rb(rbtype, storage=storage, size=size, prefetch=prefetch)
        data = self._get_datum(rbtype)
        rb.add(data)
        s = rb._storage[0]
        if isinstance(s, TensorDictBase):
            assert (s == data.select(*s.keys())).all()
        else:
            assert (s == data).all()

    def test_extend(self, rbtype, storage, size, prefetch):
        torch.manual_seed(0)
        rb = self._get_rb(rbtype, storage=storage, size=size, prefetch=prefetch)
        data = self._get_data(rbtype, size=5)
        rb.extend(data)
        length = len(rb)
        for d in data[-length:]:
            found_similar = False
            for b in rb._storage:
                if isinstance(b, TensorDictBase):
                    b = b.exclude("index").select(*set(d.keys()).intersection(b.keys()))
                    d = d.select(*set(d.keys()).intersection(b.keys()))

                value = b == d
                if isinstance(value, (torch.Tensor, TensorDictBase)):
                    value = value.all()
                if value:
                    found_similar = True
                    break
            assert found_similar

    def test_sample(self, rbtype, storage, size, prefetch):
        torch.manual_seed(0)
        rb = self._get_rb(rbtype, storage=storage, size=size, prefetch=prefetch)
        data = self._get_data(rbtype, size=5)
        rb.extend(data)
        new_data = rb.sample(3)
        if not isinstance(new_data, (torch.Tensor, TensorDictBase)):
            new_data = new_data[0]

        for d in new_data:
            found_similar = False
            for b in data:
                if isinstance(b, TensorDictBase):
                    b = b.exclude("index").select(*set(d.keys()).intersection(b.keys()))
                    d = d.select(*set(d.keys()).intersection(b.keys()))

                value = b == d
                if isinstance(value, (torch.Tensor, TensorDictBase)):
                    value = value.all()
                if value:
                    found_similar = True
                    break
            if not found_similar:
                d
            assert found_similar, (d, data)

    def test_index(self, rbtype, storage, size, prefetch):
        torch.manual_seed(0)
        rb = self._get_rb(rbtype, storage=storage, size=size, prefetch=prefetch)
        data = self._get_data(rbtype, size=5)
        rb.extend(data)
        d1 = rb[2]
        d2 = rb._storage[2]
        if type(d1) is not type(d2):
            d1 = d1[0]
        b = d1 == d2
        if not isinstance(b, bool):
            b = b.all()
        assert b


@pytest.mark.parametrize("priority_key", ["pk", "td_error"])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("device", get_available_devices())
def test_prb(priority_key, contiguous, device):
    torch.manual_seed(0)
    np.random.seed(0)
    rb = TensorDictPrioritizedReplayBuffer(
        5,
        alpha=0.7,
        beta=0.9,
        collate_fn=None if contiguous else lambda x: torch.stack(x, 0),
        priority_key=priority_key,
    )
    td1 = TensorDict(
        source={
            "a": torch.randn(3, 1),
            priority_key: torch.rand(3, 1) / 10,
            "_idx": torch.arange(3).view(3, 1),
        },
        batch_size=[3],
    ).to(device)
    rb.extend(td1)
    s = rb.sample(2)
    assert s.batch_size == torch.Size(
        [
            2,
        ]
    )
    assert (td1[s.get("_idx").squeeze()].get("a") == s.get("a")).all()
    assert_allclose_td(td1[s.get("_idx").squeeze()].select("a"), s.select("a"))

    # test replacement
    td2 = TensorDict(
        source={
            "a": torch.randn(5, 1),
            priority_key: torch.rand(5, 1) / 10,
            "_idx": torch.arange(5).view(5, 1),
        },
        batch_size=[5],
    ).to(device)
    rb.extend(td2)
    s = rb.sample(5)
    assert s.batch_size == torch.Size([5])
    assert (td2[s.get("_idx").squeeze()].get("a") == s.get("a")).all()
    assert_allclose_td(td2[s.get("_idx").squeeze()].select("a"), s.select("a"))

    # test strong update
    # get all indices that match first item
    idx = s.get("_idx")
    idx_match = (idx == idx[0]).nonzero()[:, 0]
    s.set_at_(
        priority_key,
        torch.ones(
            idx_match.numel(),
            1,
            device=device,
        )
        * 100000000,
        idx_match,
    )
    val = s.get("a")[0]

    idx0 = s.get("_idx")[0]
    rb.update_priority(s)
    s = rb.sample(5)
    assert (val == s.get("a")).sum() >= 1
    torch.testing.assert_close(td2[idx0].get("a").view(1), s.get("a").unique().view(1))

    # test updating values of original td
    td2.set_("a", torch.ones_like(td2.get("a")))
    s = rb.sample(5)
    torch.testing.assert_close(td2[idx0].get("a").view(1), s.get("a").unique().view(1))


@pytest.mark.parametrize("stack", [False, True])
def test_rb_trajectories(stack):
    traj_td = TensorDict(
        {"obs": torch.randn(3, 4, 5), "actions": torch.randn(3, 4, 2)},
        batch_size=[3, 4],
    )
    if stack:
        traj_td = torch.stack([td.to_tensordict() for td in traj_td], 0)

    rb = TensorDictPrioritizedReplayBuffer(
        5,
        alpha=0.7,
        beta=0.9,
        collate_fn=lambda x: torch.stack(x, 0),
        priority_key="td_error",
    )
    rb.extend(traj_td)
    sampled_td = rb.sample(3)
    sampled_td.set("td_error", torch.rand(3))
    rb.update_priority(sampled_td)
    sampled_td = rb.sample(3, return_weight=True)
    assert (sampled_td.get("_weight") > 0).all()
    assert sampled_td.batch_size == torch.Size([3])

    # set back the trajectory length
    sampled_td_filtered = sampled_td.to_tensordict().exclude(
        "_weight", "index", "td_error"
    )
    sampled_td_filtered.batch_size = [3, 4]


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
