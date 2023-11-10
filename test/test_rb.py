# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import contextlib
import importlib
import pickle
import sys
from functools import partial
from unittest import mock

import numpy as np
import pytest
import torch
from _utils_internal import get_default_devices, make_tc
from packaging.version import parse
from tensordict import is_tensorclass, tensorclass
from tensordict.tensordict import assert_allclose_td, TensorDict, TensorDictBase
from torchrl.data import (
    PrioritizedReplayBuffer,
    RemoteTensorDictReplayBuffer,
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import samplers, writers
from torchrl.data.replay_buffers.samplers import (
    PrioritizedSampler,
    RandomSampler,
    SamplerWithoutReplacement,
)

from torchrl.data.replay_buffers.storages import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ListStorage,
    TensorStorage,
)
from torchrl.data.replay_buffers.writers import (
    RoundRobinWriter,
    TensorDictMaxValueWriter,
)
from torchrl.envs.transforms.transforms import (
    BinarizeReward,
    CatFrames,
    CatTensors,
    CenterCrop,
    DiscreteActionProjection,
    DoubleToFloat,
    FiniteTensorDictCheck,
    FlattenObservation,
    GrayScale,
    gSDENoise,
    ObservationNorm,
    PinMemoryTransform,
    Resize,
    RewardClipping,
    RewardScaling,
    SqueezeTransform,
    ToTensorImage,
    UnsqueezeTransform,
    VecNorm,
)

OLD_TORCH = parse(torch.__version__) < parse("2.0.0")
_has_tv = importlib.util.find_spec("torchvision") is not None
_os_is_windows = sys.platform == "win32"


@pytest.mark.parametrize(
    "rb_type",
    [
        ReplayBuffer,
        TensorDictReplayBuffer,
        RemoteTensorDictReplayBuffer,
    ],
)
@pytest.mark.parametrize(
    "sampler", [samplers.RandomSampler, samplers.PrioritizedSampler]
)
@pytest.mark.parametrize("writer", [writers.RoundRobinWriter])
@pytest.mark.parametrize("storage", [ListStorage, LazyTensorStorage, LazyMemmapStorage])
@pytest.mark.parametrize("size", [3, 5, 100])
class TestComposableBuffers:
    def _get_rb(self, rb_type, size, sampler, writer, storage):

        if storage is not None:
            storage = storage(size)

        sampler_args = {}
        if sampler is samplers.PrioritizedSampler:
            sampler_args = {"max_capacity": size, "alpha": 0.8, "beta": 0.9}

        sampler = sampler(**sampler_args)
        writer = writer()
        rb = rb_type(storage=storage, sampler=sampler, writer=writer, batch_size=3)
        return rb

    def _get_datum(self, rb_type):
        if rb_type is ReplayBuffer:
            data = torch.randint(100, (1,))
        elif (
            rb_type is TensorDictReplayBuffer or rb_type is RemoteTensorDictReplayBuffer
        ):
            data = TensorDict({"a": torch.randint(100, (1,))}, [])
        else:
            raise NotImplementedError(rb_type)
        return data

    def _get_data(self, rb_type, size):
        if rb_type is ReplayBuffer:
            data = torch.randint(100, (size, 1))
        elif (
            rb_type is TensorDictReplayBuffer or rb_type is RemoteTensorDictReplayBuffer
        ):
            data = TensorDict(
                {
                    "a": torch.randint(100, (size,)),
                    "b": TensorDict({"c": torch.randint(100, (size,))}, [size]),
                },
                [size],
            )
        else:
            raise NotImplementedError(rb_type)
        return data

    def test_add(self, rb_type, sampler, writer, storage, size):
        if rb_type is RemoteTensorDictReplayBuffer and _os_is_windows:
            pytest.skip(
                "Distributed package support on Windows is a prototype feature and is subject to changes."
            )
        torch.manual_seed(0)
        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        data = self._get_datum(rb_type)
        rb.add(data)
        s = rb.sample(1)
        assert s.ndim, s
        s = s[0]
        if isinstance(s, TensorDictBase):
            s = s.select(*data.keys(True), strict=False)
            data = data.select(*s.keys(True), strict=False)
            assert (s == data).all()
            assert list(s.keys(True, True))
        else:
            assert (s == data).all()

    def test_cursor_position(self, rb_type, sampler, writer, storage, size):
        storage = storage(size)
        writer = writer()
        writer.register_storage(storage)
        batch1 = self._get_data(rb_type, size=5)
        cond = OLD_TORCH and size < len(batch1) and isinstance(storage, TensorStorage)
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
            writer.extend(batch1)

        # Added less data than storage max size
        if size > 5:
            assert writer._cursor == 5
        # Added more data than storage max size
        elif size < 5:
            assert writer._cursor == 5 - size
        # Added as data as storage max size
        else:
            assert writer._cursor == 0
            batch2 = self._get_data(rb_type, size=size - 1)
            writer.extend(batch2)
            assert writer._cursor == size - 1

    def test_extend(self, rb_type, sampler, writer, storage, size):
        if rb_type is RemoteTensorDictReplayBuffer and _os_is_windows:
            pytest.skip(
                "Distributed package support on Windows is a prototype feature and is subject to changes."
            )
        torch.manual_seed(0)
        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        data = self._get_data(rb_type, size=5)
        cond = OLD_TORCH and size < len(data) and isinstance(rb._storage, TensorStorage)
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
            rb.extend(data)
        length = len(rb)
        for d in data[-length:]:
            for b in rb._storage:
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
        data2 = self._get_data(rb_type, size=2 * size + 2)
        cond = (
            OLD_TORCH and size < len(data2) and isinstance(rb._storage, TensorStorage)
        )
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
            rb.extend(data2)

    def test_sample(self, rb_type, sampler, writer, storage, size):
        if rb_type is RemoteTensorDictReplayBuffer and _os_is_windows:
            pytest.skip(
                "Distributed package support on Windows is a prototype feature and is subject to changes."
            )
        torch.manual_seed(0)
        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        data = self._get_data(rb_type, size=5)
        cond = OLD_TORCH and size < len(data) and isinstance(rb._storage, TensorStorage)
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
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
                raise RuntimeError("did not find match")

    def test_index(self, rb_type, sampler, writer, storage, size):
        if rb_type is RemoteTensorDictReplayBuffer and _os_is_windows:
            pytest.skip(
                "Distributed package support on Windows is a prototype feature and is subject to changes."
            )
        torch.manual_seed(0)
        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        data = self._get_data(rb_type, size=5)
        cond = OLD_TORCH and size < len(data) and isinstance(rb._storage, TensorStorage)
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
            rb.extend(data)
        d1 = rb[2]
        d2 = rb._storage[2]
        if type(d1) is not type(d2):
            d1 = d1[0]
        b = d1 == d2
        if not isinstance(b, bool):
            b = b.all()
        assert b

    def test_pickable(self, rb_type, sampler, writer, storage, size):

        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        serialized = pickle.dumps(rb)
        rb2 = pickle.loads(serialized)
        assert rb.__dict__.keys() == rb2.__dict__.keys()
        for key in sorted(rb.__dict__.keys()):
            assert isinstance(rb.__dict__[key], type(rb2.__dict__[key]))


class TestStorages:
    def _get_tensor(self):
        return torch.randn(10, 11)

    def _get_tensordict(self):
        return TensorDict(
            {"data": torch.randn(10, 11), ("nested", "data"): torch.randn(10, 11, 3)},
            [10, 11],
        )

    def _get_tensorclass(self):
        data = self._get_tensordict()
        return make_tc(data)(**data, batch_size=data.shape)

    @pytest.mark.parametrize("storage_type", [TensorStorage])
    def test_errors(self, storage_type):
        with pytest.raises(ValueError, match="Expected storage to be non-null"):
            storage_type(None)
        data = torch.randn(3)
        with pytest.raises(
            ValueError, match="The max-size and the storage shape mismatch"
        ):
            storage_type(data, max_size=4)

    @pytest.mark.parametrize("data_type", ["tensor", "tensordict", "tensorclass"])
    @pytest.mark.parametrize("storage_type", [TensorStorage])
    def test_get_set(self, storage_type, data_type):
        if data_type == "tensor":
            data = self._get_tensor()
        elif data_type == "tensorclass":
            data = self._get_tensorclass()
        elif data_type == "tensordict":
            data = self._get_tensordict()
        else:
            raise NotImplementedError
        storage = storage_type(data)
        storage.set(range(10), torch.zeros_like(data))
        assert (storage.get(range(10)) == 0).all()

    @pytest.mark.parametrize("data_type", ["tensor", "tensordict", "tensorclass"])
    @pytest.mark.parametrize("storage_type", [TensorStorage])
    def test_state_dict(self, storage_type, data_type):
        if data_type == "tensor":
            data = self._get_tensor()
        elif data_type == "tensorclass":
            data = self._get_tensorclass()
        elif data_type == "tensordict":
            data = self._get_tensordict()
        else:
            raise NotImplementedError
        storage = storage_type(data)
        sd = storage.state_dict()
        storage2 = storage_type(torch.zeros_like(data))
        storage2.load_state_dict(sd)
        assert (storage.get(range(10)) == storage2.get(range(10))).all()
        assert type(storage.get(range(10))) is type(  # noqa: E721
            storage2.get(range(10))
        )

    @pytest.mark.skipif(
        not torch.cuda.device_count(),
        reason="not cuda device found to test rb storage.",
    )
    @pytest.mark.parametrize(
        "device_data,device_storage",
        [
            [torch.device("cuda"), torch.device("cpu")],
            [torch.device("cpu"), torch.device("cuda")],
            [torch.device("cpu"), "auto"],
            [torch.device("cuda"), "auto"],
        ],
    )
    @pytest.mark.parametrize("storage_type", [LazyMemmapStorage, LazyTensorStorage])
    @pytest.mark.parametrize("data_type", ["tensor", "tc", "td"])
    def test_storage_device(self, device_data, device_storage, storage_type, data_type):
        @tensorclass
        class TC:
            a: torch.Tensor

        if data_type == "tensor":
            data = torch.randn(3, device=device_data)
        elif data_type == "td":
            data = TensorDict(
                {"a": torch.randn(3, device=device_data)}, [], device=device_data
            )
        elif data_type == "tc":
            data = TC(
                a=torch.randn(3, device=device_data),
                batch_size=[],
                device=device_data,
            )
        else:
            raise NotImplementedError
        storage = storage_type(max_size=10, device=device_storage)
        if device_storage == "auto":
            device_storage = device_data
        if storage_type is LazyMemmapStorage and device_storage.type == "cuda":
            with pytest.warns(
                DeprecationWarning, match="Support for Memmap device other than CPU"
            ):
                storage.set(0, data)
        else:
            storage.set(0, data)
        assert storage.get(0).device.type == device_storage.type


@pytest.mark.parametrize("max_size", [1000])
@pytest.mark.parametrize("shape", [[3, 4]])
@pytest.mark.parametrize("storage", [LazyTensorStorage, LazyMemmapStorage])
class TestLazyStorages:
    def _get_nested_tensorclass(self, shape):
        @tensorclass
        class NestedTensorClass:
            key1: torch.Tensor
            key2: torch.Tensor

        @tensorclass
        class TensorClass:
            key1: torch.Tensor
            key2: torch.Tensor
            next: NestedTensorClass

        return TensorClass(
            key1=torch.ones(*shape),
            key2=torch.ones(*shape),
            next=NestedTensorClass(
                key1=torch.ones(*shape), key2=torch.ones(*shape), batch_size=shape
            ),
            batch_size=shape,
        )

    def _get_nested_td(self, shape):
        nested_td = TensorDict(
            {
                "key1": torch.ones(*shape),
                "key2": torch.ones(*shape),
                "next": TensorDict(
                    {
                        "key1": torch.ones(*shape),
                        "key2": torch.ones(*shape),
                    },
                    shape,
                ),
            },
            shape,
        )
        return nested_td

    def test_init(self, max_size, shape, storage):
        td = self._get_nested_td(shape)
        mystorage = storage(max_size=max_size)
        mystorage._init(td)
        assert mystorage._storage.shape == (max_size, *shape)

    def test_set(self, max_size, shape, storage):
        td = self._get_nested_td(shape)
        mystorage = storage(max_size=max_size)
        mystorage.set(list(range(td.shape[0])), td)
        assert mystorage._storage.shape == (max_size, *shape[1:])
        idx = list(range(1, td.shape[0] - 1))
        tc_sample = mystorage.get(idx)
        assert tc_sample.shape == torch.Size([td.shape[0] - 2, *td.shape[1:]])

    def test_init_tensorclass(self, max_size, shape, storage):
        tc = self._get_nested_tensorclass(shape)
        mystorage = storage(max_size=max_size)
        mystorage._init(tc)
        assert is_tensorclass(mystorage._storage)
        assert mystorage._storage.shape == (max_size, *shape)

    def test_set_tensorclass(self, max_size, shape, storage):
        tc = self._get_nested_tensorclass(shape)
        mystorage = storage(max_size=max_size)
        mystorage.set(list(range(tc.shape[0])), tc)
        assert mystorage._storage.shape == (max_size, *shape[1:])
        idx = list(range(1, tc.shape[0] - 1))
        tc_sample = mystorage.get(idx)
        assert tc_sample.shape == torch.Size([tc.shape[0] - 2, *tc.shape[1:]])


@pytest.mark.parametrize("priority_key", ["pk", "td_error"])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("device", get_default_devices())
def test_prototype_prb(priority_key, contiguous, device):
    torch.manual_seed(0)
    np.random.seed(0)
    rb = TensorDictReplayBuffer(
        sampler=samplers.PrioritizedSampler(5, alpha=0.7, beta=0.9),
        priority_key=priority_key,
        batch_size=5,
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
    s = rb.sample()
    assert s.batch_size == torch.Size([5])
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
    s = rb.sample()
    assert s.batch_size == torch.Size([5])
    assert (td2[s.get("_idx").squeeze()].get("a") == s.get("a")).all()
    assert_allclose_td(td2[s.get("_idx").squeeze()].select("a"), s.select("a"))

    # test strong update
    # get all indices that match first item
    idx = s.get("_idx")
    idx_match = (idx == idx[0]).nonzero()[:, 0]
    s.set_at_(
        priority_key,
        torch.ones(idx_match.numel(), 1, device=device) * 100000000,
        idx_match,
    )
    val = s.get("a")[0]

    idx0 = s.get("_idx")[0]
    rb.update_tensordict_priority(s)
    s = rb.sample()
    assert (val == s.get("a")).sum() >= 1
    torch.testing.assert_close(td2[idx0].get("a").view(1), s.get("a").unique().view(1))

    # test updating values of original td
    td2.set_("a", torch.ones_like(td2.get("a")))
    s = rb.sample()
    torch.testing.assert_close(td2[idx0].get("a").view(1), s.get("a").unique().view(1))


@pytest.mark.parametrize("stack", [False, True])
@pytest.mark.parametrize("datatype", ["tc", "tb"])
@pytest.mark.parametrize("reduction", ["min", "max", "median", "mean"])
def test_replay_buffer_trajectories(stack, reduction, datatype):
    traj_td = TensorDict(
        {"obs": torch.randn(3, 4, 5), "actions": torch.randn(3, 4, 2)},
        batch_size=[3, 4],
    )
    if datatype == "tc":
        c = make_tc(traj_td)
        traj_td = c(**traj_td, batch_size=traj_td.batch_size)
        assert is_tensorclass(traj_td)
    elif datatype != "tb":
        raise NotImplementedError

    if stack:
        traj_td = torch.stack(list(traj_td), 0)

    rb = TensorDictReplayBuffer(
        sampler=samplers.PrioritizedSampler(
            5,
            alpha=0.7,
            beta=0.9,
            reduction=reduction,
        ),
        priority_key="td_error",
        batch_size=3,
    )
    rb.extend(traj_td)
    sampled_td = rb.sample()
    if datatype == "tc":
        assert is_tensorclass(traj_td)
        return

    sampled_td.set("td_error", torch.rand(sampled_td.shape))
    rb.update_tensordict_priority(sampled_td)
    sampled_td = rb.sample(include_info=True)
    assert (sampled_td.get("_weight") > 0).all()
    assert sampled_td.batch_size == torch.Size([3, 4])

    # # set back the trajectory length
    # sampled_td_filtered = sampled_td.to_tensordict().exclude(
    #     "_weight", "index", "td_error"
    # )
    # sampled_td_filtered.batch_size = [3, 4]


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
@pytest.mark.parametrize("size", [3, 5, 100])
@pytest.mark.parametrize("prefetch", [0])
class TestBuffers:
    _default_params_rb = {}
    _default_params_td_rb = {}
    _default_params_prb = {"alpha": 0.8, "beta": 0.9}
    _default_params_td_prb = {"alpha": 0.8, "beta": 0.9}

    def _get_rb(self, rbtype, size, storage, prefetch):
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
        rb = rbtype(storage=storage, prefetch=prefetch, batch_size=3, **params)
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

    def test_cursor_position2(self, rbtype, storage, size, prefetch):
        torch.manual_seed(0)
        rb = self._get_rb(rbtype, storage=storage, size=size, prefetch=prefetch)
        batch1 = self._get_data(rbtype, size=5)
        cond = (
            OLD_TORCH and size < len(batch1) and isinstance(rb._storage, TensorStorage)
        )
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
            rb.extend(batch1)

        # Added less data than storage max size
        if size > 5 or storage is None:
            assert rb._writer._cursor == 5
        # Added more data than storage max size
        elif size < 5:
            assert rb._writer._cursor == 5 - size
        # Added as data as storage max size
        else:
            assert rb._writer._cursor == 0
            batch2 = self._get_data(rbtype, size=size - 1)
            rb.extend(batch2)
            assert rb._writer._cursor == size - 1

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
        cond = OLD_TORCH and size < len(data) and isinstance(rb._storage, TensorStorage)
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
            rb.extend(data)
        length = len(rb)
        for d in data[-length:]:
            found_similar = False
            for b in rb._storage:
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
        cond = OLD_TORCH and size < len(data) and isinstance(rb._storage, TensorStorage)
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
            rb.extend(data)
        new_data = rb.sample()
        if not isinstance(new_data, (torch.Tensor, TensorDictBase)):
            new_data = new_data[0]

        for d in new_data:
            found_similar = False
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
        cond = OLD_TORCH and size < len(data) and isinstance(rb._storage, TensorStorage)
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
            rb.extend(data)
        d1 = rb[2]
        d2 = rb._storage[2]
        if type(d1) is not type(d2):
            d1 = d1[0]
        b = d1 == d2
        if not isinstance(b, bool):
            b = b.all()
        assert b


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


@pytest.mark.parametrize("priority_key", ["pk", "td_error"])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("device", get_default_devices())
def test_prb(priority_key, contiguous, device):
    torch.manual_seed(0)
    np.random.seed(0)
    rb = TensorDictPrioritizedReplayBuffer(
        alpha=0.7,
        beta=0.9,
        priority_key=priority_key,
        storage=ListStorage(5),
        batch_size=5,
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
    s = rb.sample()
    assert s.batch_size == torch.Size([5])
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
    s = rb.sample()
    assert s.batch_size == torch.Size([5])
    assert (td2[s.get("_idx").squeeze()].get("a") == s.get("a")).all()
    assert_allclose_td(td2[s.get("_idx").squeeze()].select("a"), s.select("a"))

    # test strong update
    # get all indices that match first item
    idx = s.get("_idx")
    idx_match = (idx == idx[0]).nonzero()[:, 0]
    s.set_at_(
        priority_key,
        torch.ones(idx_match.numel(), 1, device=device) * 100000000,
        idx_match,
    )
    val = s.get("a")[0]

    idx0 = s.get("_idx")[0]
    rb.update_tensordict_priority(s)
    s = rb.sample()
    assert (val == s.get("a")).sum() >= 1
    torch.testing.assert_close(td2[idx0].get("a").view(1), s.get("a").unique().view(1))

    # test updating values of original td
    td2.set_("a", torch.ones_like(td2.get("a")))
    s = rb.sample()
    torch.testing.assert_close(td2[idx0].get("a").view(1), s.get("a").unique().view(1))


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
    assert (sampled_td.get("_weight") > 0).all()
    assert sampled_td.batch_size == torch.Size([3, 4])

    # set back the trajectory length
    sampled_td_filtered = sampled_td.to_tensordict().exclude(
        "_weight", "index", "td_error"
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


def test_append_transform():
    rb = ReplayBuffer(collate_fn=lambda x: torch.stack(x, 0), batch_size=1)
    td = TensorDict(
        {
            "observation": torch.randn(2, 4, 3, 16),
            "observation2": torch.randn(2, 4, 3, 16),
        },
        [],
    )
    rb.add(td)
    flatten = CatTensors(
        in_keys=["observation", "observation2"], out_key="observation_cat"
    )

    rb.append_transform(flatten)

    sampled = rb.sample()
    assert sampled.get("observation_cat").shape[-1] == 32


def test_init_transform():
    flatten = FlattenObservation(
        -2, -1, in_keys=["observation"], out_keys=["flattened"]
    )

    rb = ReplayBuffer(
        collate_fn=lambda x: torch.stack(x, 0), transform=flatten, batch_size=1
    )

    td = TensorDict({"observation": torch.randn(2, 4, 3, 16)}, [])
    rb.add(td)
    sampled = rb.sample()
    assert sampled.get("flattened").shape[-1] == 48


def test_insert_transform():
    flatten = FlattenObservation(
        -2, -1, in_keys=["observation"], out_keys=["flattened"]
    )
    rb = ReplayBuffer(
        collate_fn=lambda x: torch.stack(x, 0), transform=flatten, batch_size=1
    )
    td = TensorDict({"observation": torch.randn(2, 4, 3, 16, 1)}, [])
    rb.add(td)

    rb.insert_transform(0, SqueezeTransform(-1, in_keys=["observation"]))

    sampled = rb.sample()
    assert sampled.get("flattened").shape[-1] == 48

    with pytest.raises(ValueError):
        rb.insert_transform(10, SqueezeTransform(-1, in_keys=["observation"]))


transforms = [
    ToTensorImage,
    pytest.param(
        partial(RewardClipping, clamp_min=0.1, clamp_max=0.9), id="RewardClipping"
    ),
    BinarizeReward,
    pytest.param(
        partial(Resize, w=2, h=2),
        id="Resize",
        marks=pytest.mark.skipif(not _has_tv, reason="needs torchvision dependency"),
    ),
    pytest.param(
        partial(CenterCrop, w=1),
        id="CenterCrop",
        marks=pytest.mark.skipif(not _has_tv, reason="needs torchvision dependency"),
    ),
    pytest.param(
        partial(UnsqueezeTransform, unsqueeze_dim=-1), id="UnsqueezeTransform"
    ),
    pytest.param(partial(SqueezeTransform, squeeze_dim=-1), id="SqueezeTransform"),
    GrayScale,
    pytest.param(partial(ObservationNorm, loc=1, scale=2), id="ObservationNorm"),
    pytest.param(partial(CatFrames, dim=-3, N=4), id="CatFrames"),
    pytest.param(partial(RewardScaling, loc=1, scale=2), id="RewardScaling"),
    DoubleToFloat,
    VecNorm,
]


@pytest.mark.parametrize("transform", transforms)
def test_smoke_replay_buffer_transform(transform):
    rb = TensorDictReplayBuffer(
        transform=transform(in_keys=["observation"]), batch_size=1
    )

    # td = TensorDict({"observation": torch.randn(3, 3, 3, 16, 1), "action": torch.randn(3)}, [])
    td = TensorDict({"observation": torch.randn(3, 3, 3, 16, 3)}, [])
    rb.add(td)

    m = mock.Mock()
    m.side_effect = [td.unsqueeze(0)]
    rb._transform.forward = m
    # rb._transform.__len__ = lambda *args: 3
    rb.sample()
    assert rb._transform.forward.called

    # was_called = [False]
    # forward = rb._transform.forward
    # def new_forward(*args, **kwargs):
    #     was_called[0] = True
    #     return forward(*args, **kwargs)
    # rb._transform.forward = new_forward
    # rb.sample()
    # assert was_called[0]


transforms = [
    partial(DiscreteActionProjection, num_actions_effective=1, max_actions=3),
    FiniteTensorDictCheck,
    gSDENoise,
    PinMemoryTransform,
]


@pytest.mark.parametrize("transform", transforms)
def test_smoke_replay_buffer_transform_no_inkeys(transform):
    if transform == PinMemoryTransform and not torch.cuda.is_available():
        raise pytest.skip("No CUDA device detected, skipping PinMemory")
    rb = ReplayBuffer(
        collate_fn=lambda x: torch.stack(x, 0), transform=transform(), batch_size=1
    )

    action = torch.zeros(3)
    action[..., 0] = 1
    td = TensorDict({"observation": torch.randn(3, 3, 3, 16, 1), "action": action}, [])
    rb.add(td)
    rb.sample()

    rb._transform = mock.MagicMock()
    rb._transform.__len__ = lambda *args: 3
    rb.sample()
    assert rb._transform.called


@pytest.mark.parametrize("size", [10, 15, 20])
@pytest.mark.parametrize("samples", [5, 9, 11, 14, 16])
@pytest.mark.parametrize("drop_last", [True, False])
def test_samplerwithoutrep(size, samples, drop_last):
    torch.manual_seed(0)
    storage = ListStorage(size)
    storage.set(range(size), range(size))
    assert len(storage) == size
    sampler = SamplerWithoutReplacement(drop_last=drop_last)
    visited = False
    for _ in range(10):
        _n_left = (
            sampler._sample_list.numel() if sampler._sample_list is not None else size
        )
        if samples > size and drop_last:
            with pytest.raises(
                ValueError,
                match=r"The batch size .* is greater than the storage capacity",
            ):
                idx, _ = sampler.sample(storage, samples)
            break
        idx, _ = sampler.sample(storage, samples)
        if drop_last or _n_left >= samples:
            assert idx.numel() == samples
            assert idx.unique().numel() == idx.numel()
        else:
            assert idx.numel() == _n_left
            visited = True
    if not drop_last and (size % samples > 0):
        assert visited
    else:
        assert not visited


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


class TestStateDict:
    @pytest.mark.parametrize("storage_in", ["tensor", "memmap"])
    @pytest.mark.parametrize("storage_out", ["tensor", "memmap"])
    @pytest.mark.parametrize("init_out", [True, False])
    def test_load_state_dict(self, storage_in, storage_out, init_out):
        buffer_size = 100
        if storage_in == "memmap":
            storage_in = LazyMemmapStorage(buffer_size, device="cpu")
        elif storage_in == "tensor":
            storage_in = LazyTensorStorage(buffer_size, device="cpu")
        if storage_out == "memmap":
            storage_out = LazyMemmapStorage(buffer_size, device="cpu")
        elif storage_out == "tensor":
            storage_out = LazyTensorStorage(buffer_size, device="cpu")

        replay_buffer = TensorDictReplayBuffer(
            pin_memory=False, prefetch=3, storage=storage_in, batch_size=3
        )
        # fill replay buffer with random data
        transition = TensorDict(
            {
                "observation": torch.ones(1, 4),
                "action": torch.ones(1, 2),
                "reward": torch.ones(1, 1),
                "dones": torch.ones(1, 1),
                "next": {"observation": torch.ones(1, 4)},
            },
            batch_size=1,
        )
        for _ in range(3):
            replay_buffer.extend(transition)

        state_dict = replay_buffer.state_dict()

        new_replay_buffer = TensorDictReplayBuffer(
            pin_memory=False,
            prefetch=3,
            storage=storage_out,
            batch_size=state_dict["_batch_size"],
        )
        if init_out:
            new_replay_buffer.extend(transition)

        new_replay_buffer.load_state_dict(state_dict)
        s = new_replay_buffer.sample()
        assert (s.exclude("index") == 1).all()


@pytest.mark.parametrize("size", [20, 25, 30])
@pytest.mark.parametrize("batch_size", [1, 10, 15])
@pytest.mark.parametrize("reward_ranges", [(0.25, 0.5, 1.0)])
@pytest.mark.parametrize("device", get_default_devices())
def test_max_value_writer(size, batch_size, reward_ranges, device):
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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
