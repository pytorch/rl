# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import functools
import importlib
import os
import pickle
import sys
from functools import partial
from unittest import mock

import numpy as np
import pytest
import torch

from packaging import version
from packaging.version import parse
from tensordict import (
    assert_allclose_td,
    is_tensor_collection,
    is_tensorclass,
    LazyStackedTensorDict,
    set_list_to_stack,
    tensorclass,
    TensorDict,
    TensorDictBase,
)
from torch import multiprocessing as mp
from torch.utils._pytree import tree_flatten, tree_map

from torchrl._utils import _replace_last
from torchrl.collectors import RandomPolicy, SyncDataCollector
from torchrl.collectors.utils import split_trajectories
from torchrl.data import (
    FlatStorageCheckpointer,
    History,
    MultiStep,
    NestedStorageCheckpointer,
    PrioritizedReplayBuffer,
    RemoteTensorDictReplayBuffer,
    ReplayBuffer,
    ReplayBufferEnsemble,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import samplers, writers
from torchrl.data.replay_buffers.checkpointers import H5StorageCheckpointer
from torchrl.data.replay_buffers.samplers import (
    PrioritizedSampler,
    PrioritizedSliceSampler,
    RandomSampler,
    SamplerEnsemble,
    SamplerWithoutReplacement,
    SliceSampler,
    SliceSamplerWithoutReplacement,
)
from torchrl.data.replay_buffers.scheduler import (
    LinearScheduler,
    SchedulerList,
    StepScheduler,
)

from torchrl.data.replay_buffers.storages import (
    LazyMemmapStorage,
    LazyStackStorage,
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
    BinarizeReward,
    CatFrames,
    CatTensors,
    CenterCrop,
    Compose,
    DiscreteActionProjection,
    DoubleToFloat,
    FiniteTensorDictCheck,
    FlattenObservation,
    GrayScale,
    gSDENoise,
    ObservationNorm,
    PinMemoryTransform,
    RenameTransform,
    Resize,
    RewardClipping,
    RewardScaling,
    SqueezeTransform,
    StepCounter,
    ToTensorImage,
    UnsqueezeTransform,
    VecNorm,
)


if os.getenv("PYTORCH_TEST_FBCODE"):
    from pytorch.rl.test._utils_internal import (
        capture_log_records,
        CARTPOLE_VERSIONED,
        get_default_devices,
        make_tc,
    )
    from pytorch.rl.test.mocking_classes import CountingEnv
else:
    from _utils_internal import (
        capture_log_records,
        CARTPOLE_VERSIONED,
        get_default_devices,
        make_tc,
    )
    from mocking_classes import CountingEnv

OLD_TORCH = parse(torch.__version__) < parse("2.0.0")
_has_tv = importlib.util.find_spec("torchvision") is not None
_has_gym = importlib.util.find_spec("gym") is not None
_has_snapshot = importlib.util.find_spec("torchsnapshot") is not None
_os_is_windows = sys.platform == "win32"
_has_transformers = importlib.util.find_spec("transformers") is not None
TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)

torch_2_3 = version.parse(
    ".".join([str(s) for s in version.parse(str(torch.__version__)).release])
) >= version.parse("2.3.0")

ReplayBufferRNG = functools.partial(ReplayBuffer, generator=torch.Generator())
TensorDictReplayBufferRNG = functools.partial(
    TensorDictReplayBuffer, generator=torch.Generator()
)


@pytest.mark.parametrize(
    "sampler",
    [
        samplers.RandomSampler,
        samplers.SamplerWithoutReplacement,
        samplers.PrioritizedSampler,
    ],
)
@pytest.mark.parametrize(
    "writer", [writers.RoundRobinWriter, writers.TensorDictMaxValueWriter]
)
@pytest.mark.parametrize(
    "rb_type,storage,datatype",
    [
        [ReplayBuffer, ListStorage, None],
        [ReplayBufferRNG, ListStorage, None],
        [TensorDictReplayBuffer, ListStorage, "tensordict"],
        [TensorDictReplayBufferRNG, ListStorage, "tensordict"],
        [RemoteTensorDictReplayBuffer, ListStorage, "tensordict"],
        [ReplayBuffer, LazyTensorStorage, "tensor"],
        [ReplayBuffer, LazyTensorStorage, "tensordict"],
        [ReplayBuffer, LazyTensorStorage, "pytree"],
        [ReplayBufferRNG, LazyTensorStorage, "tensor"],
        [ReplayBufferRNG, LazyTensorStorage, "tensordict"],
        [ReplayBufferRNG, LazyTensorStorage, "pytree"],
        [TensorDictReplayBuffer, LazyTensorStorage, "tensordict"],
        [TensorDictReplayBufferRNG, LazyTensorStorage, "tensordict"],
        [RemoteTensorDictReplayBuffer, LazyTensorStorage, "tensordict"],
        [ReplayBuffer, LazyMemmapStorage, "tensor"],
        [ReplayBuffer, LazyMemmapStorage, "tensordict"],
        [ReplayBuffer, LazyMemmapStorage, "pytree"],
        [ReplayBufferRNG, LazyMemmapStorage, "tensor"],
        [ReplayBufferRNG, LazyMemmapStorage, "tensordict"],
        [ReplayBufferRNG, LazyMemmapStorage, "pytree"],
        [TensorDictReplayBuffer, LazyMemmapStorage, "tensordict"],
        [TensorDictReplayBufferRNG, LazyMemmapStorage, "tensordict"],
        [RemoteTensorDictReplayBuffer, LazyMemmapStorage, "tensordict"],
    ],
)
@pytest.mark.parametrize("size", [3, 5, 100])
class TestComposableBuffers:
    def _get_rb(self, rb_type, size, sampler, writer, storage, compilable=False):

        if storage is not None:
            storage = storage(size, compilable=compilable)

        sampler_args = {}
        if sampler is samplers.PrioritizedSampler:
            sampler_args = {"max_capacity": size, "alpha": 0.8, "beta": 0.9}

        sampler = sampler(**sampler_args)
        writer = writer(compilable=compilable)
        rb = rb_type(
            storage=storage,
            sampler=sampler,
            writer=writer,
            batch_size=3,
            compilable=compilable,
        )
        return rb

    def _get_datum(self, datatype):
        if datatype is None:
            data = torch.randint(100, (1,))
        elif datatype == "tensor":
            data = torch.randint(100, (1,))
        elif datatype == "tensordict":
            data = TensorDict(
                {"a": torch.randint(100, (1,)), "next": {"reward": torch.randn(1)}}, []
            )
        elif datatype == "pytree":
            data = {
                "a": torch.randint(100, (1,)),
                "b": {"c": [torch.zeros(3), (torch.ones(2),)]},
                30: torch.zeros(2),
            }
        else:
            raise NotImplementedError(datatype)
        return data

    def _get_data(self, datatype, size):
        if datatype is None:
            data = torch.randint(100, (size, 1))
        elif datatype == "tensor":
            data = torch.randint(100, (size, 1))
        elif datatype == "tensordict":
            data = TensorDict(
                {
                    "a": torch.randint(100, (size, 1)),
                    "next": {"reward": torch.randn(size, 1)},
                },
                [size],
            )
        elif datatype == "pytree":
            data = {
                "a": torch.randint(100, (size, 1)),
                "b": {"c": [torch.zeros(size, 3), (torch.ones(size, 2),)]},
                30: torch.zeros(size, 2),
            }
        else:
            raise NotImplementedError(datatype)
        return data

    def test_rb_repr(self, rb_type, sampler, writer, storage, size, datatype):
        if rb_type is RemoteTensorDictReplayBuffer and _os_is_windows:
            pytest.skip(
                "Distributed package support on Windows is a prototype feature and is subject to changes."
            )
        torch.manual_seed(0)
        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        data = self._get_datum(datatype)
        if not is_tensor_collection(data) and writer is TensorDictMaxValueWriter:
            with pytest.raises(
                RuntimeError, match="expects data to be a tensor collection"
            ):
                rb.add(data)
            return
        rb.add(data)
        # we just check that str runs, not its value
        assert str(rb)
        rb.sample()
        assert str(rb)

    def test_add(self, rb_type, sampler, writer, storage, size, datatype):
        if rb_type is RemoteTensorDictReplayBuffer and _os_is_windows:
            pytest.skip(
                "Distributed package support on Windows is a prototype feature and is subject to changes."
            )
        torch.manual_seed(0)
        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        data = self._get_datum(datatype)
        if not is_tensor_collection(data) and writer is TensorDictMaxValueWriter:
            with pytest.raises(
                RuntimeError, match="expects data to be a tensor collection"
            ):
                rb.add(data)
            return
        rb.add(data)
        s, info = rb.sample(1, return_info=True)
        assert len(rb) == 1
        if isinstance(s, (torch.Tensor, TensorDictBase)):
            assert s.ndim, s
            s = s[0]
        else:

            def assert_ndim(tensor):
                assert tensor.shape[0] == 1

            tree_map(assert_ndim, s)
            s = tree_map(lambda s: s[0], s)
        if isinstance(s, TensorDictBase):
            s = s.select(*data.keys(True), strict=False)
            data = data.select(*s.keys(True), strict=False)
            assert (s == data).all()
            assert list(s.keys(True, True))
        else:
            flat_s = tree_flatten(s)[0]
            flat_data = tree_flatten(data)[0]
            assert all((_s == _data).all() for (_s, _data) in zip(flat_s, flat_data))

    def test_cursor_position(self, rb_type, sampler, writer, storage, size, datatype):
        storage = storage(size)
        writer = writer()
        writer.register_storage(storage)
        batch1 = self._get_data(datatype, size=5)
        cond = (
            OLD_TORCH
            and not isinstance(writer, TensorDictMaxValueWriter)
            and size < len(batch1)
            and isinstance(storage, TensorStorage)
        )

        if not is_tensor_collection(batch1) and isinstance(
            writer, TensorDictMaxValueWriter
        ):
            with pytest.raises(
                RuntimeError, match="expects data to be a tensor collection"
            ):
                writer.extend(batch1)
            return

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
            # if Max writer, we don't necessarily overwrite existing values so
            # we just check that the cursor is before the threshold
            if isinstance(writer, TensorDictMaxValueWriter):
                assert writer._cursor <= 5 - size
            else:
                assert writer._cursor == 5 - size
        # Added as data as storage max size
        else:
            assert writer._cursor == 0
            if not isinstance(writer, TensorDictMaxValueWriter):
                batch2 = self._get_data(datatype, size=size - 1)
                writer.extend(batch2)
                assert writer._cursor == size - 1

    def test_extend(self, rb_type, sampler, writer, storage, size, datatype):
        if rb_type is RemoteTensorDictReplayBuffer and _os_is_windows:
            pytest.skip(
                "Distributed package support on Windows is a prototype feature and is subject to changes."
            )
        torch.manual_seed(0)
        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        data_shape = 5
        data = self._get_data(datatype, size=data_shape)
        cond = (
            OLD_TORCH
            and writer is not TensorDictMaxValueWriter
            and size < len(data)
            and isinstance(rb._storage, TensorStorage)
        )
        if not is_tensor_collection(data) and writer is TensorDictMaxValueWriter:
            with pytest.raises(
                RuntimeError, match="expects data to be a tensor collection"
            ):
                rb.extend(data)
            return
        length = min(rb._storage.max_size, len(rb) + data_shape)
        if writer is TensorDictMaxValueWriter:
            data["next", "reward"][-length:] = 1_000_000
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
            rb.extend(data)
        length = len(rb)
        if is_tensor_collection(data):
            data_iter = data[-length:]
        else:

            def data_iter():
                for t in range(-length, -1):
                    yield tree_map(lambda x, t=t: x[t], data)

            data_iter = data_iter()
        for d in data_iter:
            for b in rb._storage:
                if isinstance(b, TensorDictBase):
                    keys = set(d.keys()).intersection(b.keys())
                    b = b.exclude("index").select(*keys, strict=False)
                    keys = set(d.keys()).intersection(b.keys())
                    d = d.select(*keys, strict=False)
                if isinstance(b, (torch.Tensor, TensorDictBase)):
                    value = b == d
                    value = value.all()
                else:
                    d_flat = tree_flatten(d)[0]
                    b_flat = tree_flatten(b)[0]
                    value = all((_b == _d).all() for (_b, _d) in zip(b_flat, d_flat))
                if value:
                    break
            else:
                raise RuntimeError("did not find match")

        data2 = self._get_data(datatype, size=2 * size + 2)
        cond = (
            OLD_TORCH
            and writer is not TensorDictMaxValueWriter
            and size < len(data2)
            and isinstance(rb._storage, TensorStorage)
        )
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
            rb.extend(data2)

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
    )
    # Compiling on Windows requires "cl" compiler to be installed.
    # <https://github.com/pytorch/pytorch/blob/8231180147a096a703d8891756068c89365292e0/torch/_inductor/cpp_builder.py#L143>
    # Our Windows CI jobs do not have "cl", so skip this test.
    @pytest.mark.skipif(_os_is_windows, reason="windows tests do not support compile")
    @pytest.mark.parametrize("avoid_max_size", [False, True])
    def test_extend_sample_recompile(
        self, rb_type, sampler, writer, storage, size, datatype, avoid_max_size
    ):
        if rb_type is not ReplayBuffer:
            pytest.skip(
                "Only replay buffer of type 'ReplayBuffer' is currently supported."
            )
        if sampler is not RandomSampler:
            pytest.skip("Only sampler of type 'RandomSampler' is currently supported.")
        if storage is not LazyTensorStorage:
            pytest.skip(
                "Only storage of type 'LazyTensorStorage' is currently supported."
            )
        if writer is not RoundRobinWriter:
            pytest.skip(
                "Only writer of type 'RoundRobinWriter' is currently supported."
            )
        if datatype == "tensordict":
            pytest.skip("'tensordict' datatype is not currently supported.")

        torch._dynamo.reset_code_caches()

        # Number of times to extend the replay buffer
        num_extend = 10
        data_size = size

        # These two cases are separated because when the max storage size is
        # reached, the code execution path changes, causing necessary
        # recompiles.
        if avoid_max_size:
            storage_size = (num_extend + 1) * data_size
        else:
            storage_size = 2 * data_size

        rb = self._get_rb(
            rb_type=rb_type,
            sampler=sampler,
            writer=writer,
            storage=storage,
            size=storage_size,
            compilable=True,
        )
        data = self._get_data(datatype, size=data_size)

        @torch.compile
        def extend_and_sample(data):
            rb.extend(data)
            return rb.sample()

        # NOTE: The first three calls to 'extend' and 'sample' can currently
        # cause recompilations, so avoid capturing those.
        num_extend_before_capture = 3

        for _ in range(num_extend_before_capture):
            extend_and_sample(data)

        try:
            torch._logging.set_logs(recompiles=True)
            records = []
            capture_log_records(records, "torch._dynamo", "recompiles")

            for _ in range(num_extend - num_extend_before_capture):
                extend_and_sample(data)

        finally:
            torch._logging.set_logs()

        assert len(rb) == min((num_extend * data_size), storage_size)
        assert len(records) == 0

    def test_sample(self, rb_type, sampler, writer, storage, size, datatype):
        if rb_type is RemoteTensorDictReplayBuffer and _os_is_windows:
            pytest.skip(
                "Distributed package support on Windows is a prototype feature and is subject to changes."
            )
        torch.manual_seed(0)
        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        data = self._get_data(datatype, size=5)
        cond = (
            OLD_TORCH
            and writer is not TensorDictMaxValueWriter
            and size < len(data)
            and isinstance(rb._storage, TensorStorage)
        )
        if not is_tensor_collection(data) and writer is TensorDictMaxValueWriter:
            with pytest.raises(
                RuntimeError, match="expects data to be a tensor collection"
            ):
                rb.extend(data)
            return
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
            rb.extend(data)
        rb_sample = rb.sample()
        # if not isinstance(new_data, (torch.Tensor, TensorDictBase)):
        #     new_data = new_data[0]

        if is_tensor_collection(data) or isinstance(data, torch.Tensor):
            rb_sample_iter = rb_sample
        else:

            def data_iter_func(maxval, data=data):
                for t in range(maxval):
                    yield tree_map(lambda x, t=t: x[t], data)

            rb_sample_iter = data_iter_func(rb._batch_size, rb_sample)

        for single_sample in rb_sample_iter:

            if is_tensor_collection(data) or isinstance(data, torch.Tensor):
                data_iter = data
            else:
                data_iter = data_iter_func(5, data)

            for data_sample in data_iter:
                if isinstance(data_sample, TensorDictBase):
                    keys = set(single_sample.keys()).intersection(data_sample.keys())
                    data_sample = data_sample.exclude("index").select(
                        *keys, strict=False
                    )
                    keys = set(single_sample.keys()).intersection(data_sample.keys())
                    single_sample = single_sample.select(*keys, strict=False)

                if isinstance(data_sample, (torch.Tensor, TensorDictBase)):
                    value = data_sample == single_sample
                    value = value.all()
                else:
                    d_flat = tree_flatten(single_sample)[0]
                    b_flat = tree_flatten(data_sample)[0]
                    value = all((_b == _d).all() for (_b, _d) in zip(b_flat, d_flat))

                if value:
                    break
            else:
                raise RuntimeError("did not find match")

    def test_index(self, rb_type, sampler, writer, storage, size, datatype):
        if rb_type is RemoteTensorDictReplayBuffer and _os_is_windows:
            pytest.skip(
                "Distributed package support on Windows is a prototype feature and is subject to changes."
            )
        torch.manual_seed(0)
        rb = self._get_rb(
            rb_type=rb_type, sampler=sampler, writer=writer, storage=storage, size=size
        )
        data = self._get_data(datatype, size=5)
        cond = (
            OLD_TORCH
            and writer is not TensorDictMaxValueWriter
            and size < len(data)
            and isinstance(rb._storage, TensorStorage)
        )
        if not is_tensor_collection(data) and writer is TensorDictMaxValueWriter:
            with pytest.raises(
                RuntimeError, match="expects data to be a tensor collection"
            ):
                rb.extend(data)
            return
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
            rb.extend(data)
        d1 = rb[2]
        d2 = rb._storage[2]
        if type(d1) is not type(d2):
            d1 = d1[0]
        if is_tensor_collection(data) or isinstance(data, torch.Tensor):
            b = d1 == d2
            if not isinstance(b, bool):
                b = b.all()
        else:
            d1_flat = tree_flatten(d1)[0]
            d2_flat = tree_flatten(d2)[0]
            b = all((_d1 == _d2).all() for (_d1, _d2) in zip(d1_flat, d2_flat))
        assert b

    def test_pickable(self, rb_type, sampler, writer, storage, size, datatype):

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

    def _get_pytree(self):
        return {
            "a": torch.randint(100, (10, 11, 1)),
            "b": {"c": [torch.zeros(10, 11), (torch.ones(10, 11),)]},
            30: torch.zeros(10, 11),
        }

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

    def test_existsok_lazymemmap(self, tmpdir):
        storage0 = LazyMemmapStorage(10, scratch_dir=tmpdir)
        rb = ReplayBuffer(storage=storage0)
        rb.extend(TensorDict(a=torch.randn(3), batch_size=[3]))

        storage1 = LazyMemmapStorage(10, scratch_dir=tmpdir)
        rb = ReplayBuffer(storage=storage1)
        with pytest.raises(RuntimeError, match="existsok"):
            rb.extend(TensorDict(a=torch.randn(3), batch_size=[3]))

        storage2 = LazyMemmapStorage(10, scratch_dir=tmpdir, existsok=True)
        rb = ReplayBuffer(storage=storage2)
        rb.extend(TensorDict(a=torch.randn(3), batch_size=[3]))

    @pytest.mark.parametrize(
        "data_type", ["tensor", "tensordict", "tensorclass", "pytree"]
    )
    @pytest.mark.parametrize("storage_type", [TensorStorage])
    def test_get_set(self, storage_type, data_type):
        if data_type == "tensor":
            data = self._get_tensor()
        elif data_type == "tensorclass":
            data = self._get_tensorclass()
        elif data_type == "tensordict":
            data = self._get_tensordict()
        elif data_type == "pytree":
            data = self._get_pytree()
        else:
            raise NotImplementedError
        storage = storage_type(data)
        if data_type == "pytree":
            storage.set(range(10), tree_map(torch.zeros_like, data))

            def check(x):
                assert (x == 0).all()

            tree_map(check, storage.get(range(10)))
        else:
            storage.set(range(10), torch.zeros_like(data))
            assert (storage.get(range(10)) == 0).all()

    @pytest.mark.parametrize(
        "data_type", ["tensor", "tensordict", "tensorclass", "pytree"]
    )
    @pytest.mark.parametrize("storage_type", [TensorStorage])
    def test_state_dict(self, storage_type, data_type):
        if data_type == "tensor":
            data = self._get_tensor()
        elif data_type == "tensorclass":
            data = self._get_tensorclass()
        elif data_type == "tensordict":
            data = self._get_tensordict()
        elif data_type == "pytree":
            data = self._get_pytree()
        else:
            raise NotImplementedError
        storage = storage_type(data)
        if data_type == "pytree":
            with pytest.raises(TypeError, match="are not supported by"):
                storage.state_dict()
            return
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

        if (
            storage_type is LazyMemmapStorage
            and device_storage != "auto"
            and device_storage.type != "cpu"
        ):
            with pytest.raises(ValueError, match="Memory map device other than CPU"):
                storage_type(max_size=10, device=device_storage)
            return
        storage = storage_type(max_size=10, device=device_storage)
        storage.set(0, data)
        if device_storage != "auto":
            assert storage.get(0).device.type == device_storage.type
        else:
            assert storage.get(0).device.type == storage.device.type

    @pytest.mark.parametrize("storage_in", ["tensor", "memmap"])
    @pytest.mark.parametrize("storage_out", ["tensor", "memmap"])
    @pytest.mark.parametrize("init_out", [True, False])
    @pytest.mark.parametrize(
        "backend", ["torch"] + (["torchsnapshot"] if _has_snapshot else [])
    )
    def test_storage_state_dict(self, storage_in, storage_out, init_out, backend):
        os.environ["CKPT_BACKEND"] = backend
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

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
    )
    @pytest.mark.skipif(_os_is_windows, reason="windows tests do not support compile")
    # This test checks if the `torch._dynamo.disable` wrapper around
    # `TensorStorage._rand_given_ndim` is still necessary.
    def test__rand_given_ndim_recompile(self):
        torch._dynamo.reset_code_caches()

        # Number of times to extend the replay buffer
        num_extend = 10
        data_size = 100
        storage_size = (num_extend + 1) * data_size
        sample_size = 3

        storage = LazyTensorStorage(storage_size, compilable=True)
        sampler = RandomSampler()

        # Override to avoid the `torch._dynamo.disable` wrapper
        storage._rand_given_ndim = storage._rand_given_ndim_impl

        @torch.compile
        def extend_and_sample(data):
            storage.set(torch.arange(data_size) + len(storage), data)
            return sampler.sample(storage, sample_size)

        data = torch.randint(100, (data_size, 1))

        try:
            torch._logging.set_logs(recompiles=True)
            records = []
            capture_log_records(records, "torch._dynamo", "recompiles")

            for _ in range(num_extend):
                extend_and_sample(data)

        finally:
            torch._logging.set_logs()

        assert len(storage) == num_extend * data_size
        assert len(records) == 8, (
            "If this ever decreases, that's probably good news and the "
            "`torch._dynamo.disable` wrapper around "
            "`TensorStorage._rand_given_ndim` can be removed."
        )

    @pytest.mark.parametrize("storage_type", [LazyMemmapStorage, LazyTensorStorage])
    def test_extend_lazystack(self, storage_type):

        rb = ReplayBuffer(
            storage=storage_type(6),
            batch_size=2,
        )
        td1 = TensorDict(a=torch.rand(5, 4, 8), batch_size=5)
        td2 = TensorDict(a=torch.rand(5, 3, 8), batch_size=5)
        ltd = LazyStackedTensorDict(td1, td2, stack_dim=1)
        rb.extend(ltd)
        rb.sample(3)
        assert len(rb) == 5

    @pytest.mark.parametrize("device_data", get_default_devices())
    @pytest.mark.parametrize("storage_type", [LazyMemmapStorage, LazyTensorStorage])
    @pytest.mark.parametrize("data_type", ["tensor", "tc", "td", "pytree"])
    @pytest.mark.parametrize("isinit", [True, False])
    def test_storage_dumps_loads(
        self, device_data, storage_type, data_type, isinit, tmpdir
    ):
        torch.manual_seed(0)

        dir_rb = tmpdir / "rb"
        dir_save = tmpdir / "save"
        dir_rb.mkdir()
        dir_save.mkdir()
        torch.manual_seed(0)

        @tensorclass
        class TC:
            tensor: torch.Tensor
            td: TensorDict
            text: str

        if data_type == "tensor":
            data = torch.randint(10, (3,), device=device_data)
        elif data_type == "pytree":
            data = {
                "a": torch.randint(10, (3,), device=device_data),
                "b": {"c": [torch.ones(3), (-torch.ones(3, 2),)]},
                30: -torch.ones(3, 1),
            }
        elif data_type == "td":
            data = TensorDict(
                {
                    "a": torch.randint(10, (3,), device=device_data),
                    "b": TensorDict(
                        {"c": torch.randint(10, (3,), device=device_data)},
                        batch_size=[3],
                    ),
                },
                batch_size=[3],
                device=device_data,
            )
        elif data_type == "tc":
            data = TC(
                tensor=torch.randint(10, (3,), device=device_data),
                td=TensorDict(
                    {"c": torch.randint(10, (3,), device=device_data)}, batch_size=[3]
                ),
                text="some text",
                batch_size=[3],
                device=device_data,
            )
        else:
            raise NotImplementedError

        if storage_type in (LazyMemmapStorage,):
            storage = storage_type(max_size=10, scratch_dir=dir_rb)
        else:
            storage = storage_type(max_size=10)

        # We cast the device to CPU as CUDA isn't automatically cast to CPU when using range() index
        if data_type == "pytree":
            storage.set(range(3), tree_map(lambda x: x.cpu(), data))
        else:
            storage.set(range(3), data.cpu())

        storage.dumps(dir_save)
        # check we can dump twice
        storage.dumps(dir_save)

        storage_recover = storage_type(max_size=10)
        if isinit:
            if data_type == "pytree":
                storage_recover.set(
                    range(3), tree_map(lambda x: x.cpu().clone().zero_(), data)
                )
            else:
                storage_recover.set(range(3), data.cpu().clone().zero_())

        if data_type in ("tensor", "pytree") and not isinit:
            with pytest.raises(
                RuntimeError,
                match="Cannot fill a non-initialized pytree-based TensorStorage",
            ):
                storage_recover.loads(dir_save)
            return
        storage_recover.loads(dir_save)
        # tree_map with more than one pytree is only available in torch >= 2.3
        if torch_2_3:
            if data_type in ("tensor", "pytree"):
                tree_map(
                    torch.testing.assert_close,
                    tree_flatten(storage[:])[0],
                    tree_flatten(storage_recover[:])[0],
                )
            else:
                assert_allclose_td(storage[:], storage_recover[:])
        if data == "tc":
            assert storage._storage.text == storage_recover._storage.text

    def test_add_list_of_tds(self):
        rb = ReplayBuffer(storage=LazyTensorStorage(100))
        rb.extend([TensorDict({"a": torch.randn(2, 3)}, [2])])
        assert len(rb) == 1
        assert rb[:].shape == torch.Size([1, 2])

    @pytest.mark.parametrize(
        "storage_type,collate_fn",
        [
            (LazyTensorStorage, None),
            (LazyMemmapStorage, None),
            (ListStorage, torch.stack),
        ],
    )
    def test_storage_inplace_writing(self, storage_type, collate_fn):
        rb = ReplayBuffer(storage=storage_type(102), collate_fn=collate_fn)
        data = TensorDict(
            {"a": torch.arange(100), ("b", "c"): torch.arange(100)}, [100]
        )
        rb.extend(data)
        assert len(rb) == 100
        rb[3:4] = TensorDict(
            {"a": torch.tensor([0]), ("b", "c"): torch.tensor([0])}, [1]
        )
        assert (rb[3:4] == 0).all()
        assert len(rb) == 100
        assert rb._writer._cursor == 100
        rb[10:20] = TensorDict(
            {"a": torch.tensor([0] * 10), ("b", "c"): torch.tensor([0] * 10)}, [10]
        )
        assert (rb[10:20] == 0).all()
        assert len(rb) == 100
        assert rb._writer._cursor == 100
        rb[torch.arange(30, 40)] = TensorDict(
            {"a": torch.tensor([0] * 10), ("b", "c"): torch.tensor([0] * 10)}, [10]
        )
        assert (rb[30:40] == 0).all()
        assert len(rb) == 100

    @pytest.mark.parametrize(
        "storage_type,collate_fn",
        [
            (LazyTensorStorage, None),
            (LazyMemmapStorage, None),
            (ListStorage, torch.stack),
        ],
    )
    def test_storage_inplace_writing_transform(self, storage_type, collate_fn):
        rb = ReplayBuffer(storage=storage_type(102), collate_fn=collate_fn)
        rb.append_transform(lambda x: x + 1, invert=True)
        rb.append_transform(lambda x: x + 1)
        data = TensorDict(
            {"a": torch.arange(100), ("b", "c"): torch.arange(100)}, [100]
        )
        rb.extend(data)
        assert len(rb) == 100
        rb[3:4] = TensorDict(
            {"a": torch.tensor([0]), ("b", "c"): torch.tensor([0])}, [1]
        )
        assert (rb[3:4] == 2).all(), rb[3:4]["a"]
        assert len(rb) == 100
        assert rb._writer._cursor == 100
        rb[10:20] = TensorDict(
            {"a": torch.tensor([0] * 10), ("b", "c"): torch.tensor([0] * 10)}, [10]
        )
        assert (rb[10:20] == 2).all()
        assert len(rb) == 100
        assert rb._writer._cursor == 100
        rb[torch.arange(30, 40)] = TensorDict(
            {"a": torch.tensor([0] * 10), ("b", "c"): torch.tensor([0] * 10)}, [10]
        )
        assert (rb[30:40] == 2).all()
        assert len(rb) == 100

    @pytest.mark.parametrize(
        "storage_type,collate_fn",
        [
            (LazyTensorStorage, None),
            # (LazyMemmapStorage, None),
            (ListStorage, TensorDict.maybe_dense_stack),
        ],
    )
    def test_storage_inplace_writing_newkey(self, storage_type, collate_fn):
        rb = ReplayBuffer(storage=storage_type(102), collate_fn=collate_fn)
        data = TensorDict(
            {"a": torch.arange(100), ("b", "c"): torch.arange(100)}, [100]
        )
        rb.extend(data)
        assert len(rb) == 100
        rb[3:4] = TensorDict(
            {"a": torch.tensor([0]), ("b", "c"): torch.tensor([0]), "d": torch.ones(1)},
            [1],
        )
        assert "d" in rb[3]
        assert "d" in rb[3:4]
        if storage_type is not ListStorage:
            assert "d" in rb[3:5]
        else:
            # a lazy stack doesn't show exclusive fields
            assert "d" not in rb[3:5]

    @pytest.mark.parametrize("storage_type", [LazyTensorStorage, LazyMemmapStorage])
    def test_storage_inplace_writing_ndim(self, storage_type):
        rb = ReplayBuffer(storage=storage_type(102, ndim=2))
        data = TensorDict(
            {
                "a": torch.arange(50).expand(2, 50),
                ("b", "c"): torch.arange(50).expand(2, 50),
            },
            [2, 50],
        )
        rb.extend(data)
        assert len(rb) == 100
        rb[0, 3:4] = TensorDict(
            {"a": torch.tensor([0]), ("b", "c"): torch.tensor([0])}, [1]
        )
        assert (rb[0, 3:4] == 0).all()
        assert (rb[1, 3:4] != 0).all()
        assert rb._writer._cursor == 50
        rb[1, 5:6] = TensorDict(
            {"a": torch.tensor([0]), ("b", "c"): torch.tensor([0])}, [1]
        )
        assert (rb[1, 5:6] == 0).all()
        assert rb._writer._cursor == 50
        rb[:, 7:8] = TensorDict(
            {"a": torch.tensor([0]), ("b", "c"): torch.tensor([0])}, [1]
        ).expand(2, 1)
        assert (rb[:, 7:8] == 0).all()
        assert rb._writer._cursor == 50
        # test broadcasting
        rb[:, 10:20] = TensorDict(
            {"a": torch.tensor([0] * 10), ("b", "c"): torch.tensor([0] * 10)}, [10]
        )
        assert (rb[:, 10:20] == 0).all()
        assert len(rb) == 100

    @pytest.mark.parametrize("max_size", [1000, None])
    @pytest.mark.parametrize("stack_dim", [-1, 0])
    def test_lazy_stack_storage(self, max_size, stack_dim):
        # Create an instance of LazyStackStorage with given parameters
        storage = LazyStackStorage(max_size=max_size, stack_dim=stack_dim)
        # Create a ReplayBuffer using the created storage
        rb = ReplayBuffer(storage=storage)
        # Generate some random data to add to the buffer
        torch.manual_seed(0)
        data0 = TensorDict(a=torch.randn((10,)), b=torch.rand(4), c="a string!")
        data1 = TensorDict(a=torch.randn((11,)), b=torch.rand(4), c="another string!")
        # Add the data to the buffer
        rb.add(data0)
        rb.add(data1)
        # Sample from the buffer
        sample = rb.sample(10)
        # Check that the sampled data has the correct shape and type
        assert isinstance(sample, LazyStackedTensorDict)
        assert sample["b"].shape[0] == 10
        assert all(isinstance(item, str) for item in sample["c"])
        # If densify is True, check that the sampled data is dense
        sample = sample.densify(layout=torch.jagged)
        assert isinstance(sample["a"], torch.Tensor)
        assert sample["a"].shape[0] == 10


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

    def test_extend_list_pytree(self, max_size, shape, storage):
        memory = ReplayBuffer(
            storage=storage(max_size=max_size),
            sampler=SamplerWithoutReplacement(),
        )
        data = [
            (
                torch.full(shape, i),
                {"a": torch.full(shape, i), "b": (torch.full(shape, i))},
                [torch.full(shape, i)],
            )
            for i in range(10)
        ]
        memory.extend(data)
        assert len(memory) == 10
        assert len(memory._storage) == 10
        sample = memory.sample(10)
        for leaf in tree_iter(sample):
            assert (leaf.unique(sorted=True) == torch.arange(10)).all()
        memory = ReplayBuffer(
            storage=storage(max_size=max_size),
            sampler=SamplerWithoutReplacement(),
        )
        t1x4 = torch.Tensor([0.1, 0.2, 0.3, 0.4])
        t1x1 = torch.Tensor([0.01])
        with pytest.raises(
            RuntimeError, match="Stacking the elements of the list resulted in an error"
        ):
            memory.extend([t1x4, t1x1, t1x4 + 0.4, t1x1 + 0.01])


@pytest.mark.parametrize("priority_key", ["pk", "td_error"])
@pytest.mark.parametrize("contiguous", [True, False])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("alpha", [0.0, 0.7])
def test_ptdrb(priority_key, contiguous, alpha, device):
    torch.manual_seed(0)
    np.random.seed(0)
    rb = TensorDictReplayBuffer(
        sampler=samplers.PrioritizedSampler(5, alpha=alpha, beta=0.9),
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
        device=device,
    )
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
        device=device,
    )
    rb.extend(td2)
    s = rb.sample()
    assert s.batch_size == torch.Size([5])
    assert (td2[s.get("_idx").squeeze()].get("a") == s.get("a")).all()
    assert_allclose_td(td2[s.get("_idx").squeeze()].select("a"), s.select("a"))

    if (
        alpha == 0.0
    ):  # when alpha is 0.0, sampling is uniform, so no need to check priority sampling
        return

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
    rbcls = functools.partial(TensorDictReplayBuffer, priority_key="td_error")
    if datatype == "tc":
        c = make_tc(traj_td)
        rbcls = functools.partial(ReplayBuffer, storage=LazyTensorStorage(100))
        traj_td = c(**traj_td, batch_size=traj_td.batch_size)
        assert is_tensorclass(traj_td)
    elif datatype != "tb":
        raise NotImplementedError

    if stack:
        traj_td = torch.stack(list(traj_td), 0)

    rb = rbcls(
        sampler=samplers.PrioritizedSampler(
            5,
            alpha=0.7,
            beta=0.9,
            reduction=reduction,
        ),
        batch_size=3,
    )
    rb.extend(traj_td)
    if datatype == "tc":
        sampled_td, info = rb.sample(return_info=True)
        index = info["index"]
    else:
        sampled_td = rb.sample()
    if datatype == "tc":
        assert is_tensorclass(traj_td)
        return

    sampled_td.set("td_error", torch.rand(sampled_td.shape))
    if datatype == "tc":
        rb.update_priority(index, sampled_td)
        sampled_td, info = rb.sample(return_info=True)
        assert (info["_weight"] > 0).all()
        assert sampled_td.batch_size == torch.Size([3, 4])
    else:
        rb.update_tensordict_priority(sampled_td)
        sampled_td = rb.sample(include_info=True)
        assert (sampled_td.get("_weight") > 0).all()
        assert sampled_td.batch_size == torch.Size([3, 4])

    # # set back the trajectory length
    # sampled_td_filtered = sampled_td.to_tensordict().exclude(
    #     "_weight", "index", "td_error"
    # )
    # sampled_td_filtered.batch_size = [3, 4]


class TestRNG:
    def test_rb_rng(self):
        state = torch.random.get_rng_state()
        rb = ReplayBufferRNG(sampler=RandomSampler(), storage=LazyTensorStorage(100))
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
            OLD_TORCH and size < len(batch1) and isinstance(rb._storage, TensorStorage)
        )
        with pytest.warns(
            UserWarning,
            match="A cursor of length superior to the storage capacity was provided",
        ) if cond else contextlib.nullcontext():
            rb.extend(batch1)

        # Added fewer data than storage max size
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

    def test_index_nonfull(self, rbtype, storage, size, prefetch):
        # checks that indexing the buffer before it's full gives the accurate view of the data
        rb = self._get_rb(rbtype, storage=storage, size=size, prefetch=prefetch)
        data = self._get_data(rbtype, size=size - 1)
        rb.extend(data)
        assert len(rb[: size - 1]) == size - 1
        assert len(rb[size - 2 :]) == 1


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


class TestTransforms:
    def test_append_transform(self):
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

    def test_init_transform(self):
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

    def test_insert_transform(self):
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
            marks=pytest.mark.skipif(
                not _has_tv, reason="needs torchvision dependency"
            ),
        ),
        pytest.param(
            partial(CenterCrop, w=1),
            id="CenterCrop",
            marks=pytest.mark.skipif(
                not _has_tv, reason="needs torchvision dependency"
            ),
        ),
        pytest.param(partial(UnsqueezeTransform, dim=-1), id="UnsqueezeTransform"),
        pytest.param(partial(SqueezeTransform, dim=-1), id="SqueezeTransform"),
        GrayScale,
        pytest.param(partial(ObservationNorm, loc=1, scale=2), id="ObservationNorm"),
        pytest.param(partial(CatFrames, dim=-3, N=4), id="CatFrames"),
        pytest.param(partial(RewardScaling, loc=1, scale=2), id="RewardScaling"),
        DoubleToFloat,
        VecNorm,
    ]

    @pytest.mark.parametrize("transform", transforms)
    def test_smoke_replay_buffer_transform(self, transform):
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

    transforms2 = [
        partial(DiscreteActionProjection, num_actions_effective=1, max_actions=3),
        FiniteTensorDictCheck,
        gSDENoise,
        PinMemoryTransform,
    ]

    @pytest.mark.parametrize("transform", transforms2)
    def test_smoke_replay_buffer_transform_no_inkeys(self, transform):
        if transform == PinMemoryTransform and not torch.cuda.is_available():
            raise pytest.skip("No CUDA device detected, skipping PinMemory")
        rb = ReplayBuffer(
            collate_fn=lambda x: torch.stack(x, 0), transform=transform(), batch_size=1
        )

        action = torch.zeros(3)
        action[..., 0] = 1
        td = TensorDict(
            {"observation": torch.randn(3, 3, 3, 16, 1), "action": action}, []
        )
        rb.add(td)
        rb.sample()

        rb._transform = mock.MagicMock()
        rb._transform.__len__ = lambda *args: 3
        rb.sample()
        assert rb._transform.called

    @pytest.mark.parametrize("at_init", [True, False])
    def test_transform_nontensor(self, at_init):
        def t(x):
            return tree_map(lambda y: y * 0, x)

        if at_init:
            rb = ReplayBuffer(storage=LazyMemmapStorage(100), transform=t)
        else:
            rb = ReplayBuffer(storage=LazyMemmapStorage(100))
            rb.append_transform(t)
        data = {
            "a": torch.randn(3),
            "b": {"c": (torch.zeros(2), [torch.ones(1)])},
            30: -torch.ones(()),
        }
        rb.add(data)

        def assert0(x):
            assert (x == 0).all()

        s = rb.sample(10)
        tree_map(assert0, s)

    def test_transform_inv(self):
        rb = ReplayBuffer(storage=LazyMemmapStorage(10), batch_size=4)
        data = TensorDict({"a": torch.zeros(10)}, [10])

        def t(data):
            data += 1
            return data

        rb.append_transform(t, invert=True)
        rb.extend(data)
        assert (data == 1).all()


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
        rb._writer.dumps(tmpdir)
        # check we can dump twice
        rb._writer.dumps(tmpdir)
        other = TensorDictMaxValueWriter(rank_key="key")
        other.loads(tmpdir)
        assert len(rb._writer._current_top_values) == len(other._current_top_values)
        torch.testing.assert_close(
            torch.tensor(rb._writer._current_top_values),
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
        with pytest.raises(RuntimeError, match="cannot be shared between processes"):
            self.exec_multiproc_rb(
                sampler_type=lambda: PrioritizedSampler(21, alpha=1.1, beta=0.5)
            )

    def test_error_noninit(self):
        # list storage cannot be shared
        with pytest.raises(RuntimeError, match="it has not been initialized yet"):
            self.exec_multiproc_rb(init=False)


class TestSamplers:
    @pytest.mark.parametrize(
        "backend", ["torch"] + (["torchsnapshot"] if _has_snapshot else [])
    )
    def test_sampler_without_rep_state_dict(self, backend):
        os.environ["CKPT_BACKEND"] = backend
        torch.manual_seed(0)

        n_samples = 3
        buffer_size = 100
        storage_in = LazyTensorStorage(buffer_size, device="cpu")
        storage_out = LazyTensorStorage(buffer_size, device="cpu")

        replay_buffer = TensorDictReplayBuffer(
            storage=storage_in,
            sampler=SamplerWithoutReplacement(),
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
        for _ in range(n_samples):
            replay_buffer.extend(transition.clone())
        for _ in range(n_samples):
            s = replay_buffer.sample(batch_size=1)
            assert (s.exclude("index") == 1).all()

        replay_buffer.extend(torch.zeros_like(transition))

        state_dict = replay_buffer.state_dict()

        new_replay_buffer = TensorDictReplayBuffer(
            storage=storage_out,
            batch_size=state_dict["_batch_size"],
            sampler=SamplerWithoutReplacement(),
        )

        new_replay_buffer.load_state_dict(state_dict)
        s = new_replay_buffer.sample(batch_size=1)
        assert (s.exclude("index") == 0).all()

    def test_sampler_without_rep_dumps_loads(self, tmpdir):
        d0 = tmpdir + "/save0"
        d1 = tmpdir + "/save1"
        d2 = tmpdir + "/dump"
        replay_buffer = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(max_size=100, scratch_dir=d0, device="cpu"),
            sampler=SamplerWithoutReplacement(drop_last=True),
            batch_size=8,
        )
        replay_buffer2 = TensorDictReplayBuffer(
            storage=LazyMemmapStorage(max_size=100, scratch_dir=d1, device="cpu"),
            sampler=SamplerWithoutReplacement(drop_last=True),
            batch_size=8,
        )
        td = TensorDict(
            {"a": torch.arange(0, 27), ("b", "c"): torch.arange(1, 28)}, batch_size=[27]
        )
        replay_buffer.extend(td)
        for _ in replay_buffer:
            break
        replay_buffer.dumps(d2)
        replay_buffer2.loads(d2)
        assert (
            replay_buffer.sampler._sample_list == replay_buffer2.sampler._sample_list
        ).all()
        s = replay_buffer2.sample(3)
        assert (s["a"] == s["b", "c"] - 1).all()

    @pytest.mark.parametrize("drop_last", [False, True])
    def test_sampler_without_replacement_cap_prefetch(self, drop_last):
        torch.manual_seed(0)
        data = TensorDict({"a": torch.arange(11)}, batch_size=[11])
        rb = ReplayBuffer(
            storage=LazyTensorStorage(11),
            sampler=SamplerWithoutReplacement(drop_last=drop_last),
            batch_size=2,
            prefetch=3,
        )
        rb.extend(data)

        for _ in range(100):
            s = set()
            for i, d in enumerate(rb):
                assert i <= (4 + int(not drop_last)), i
                s = s.union(set(d["a"].tolist()))
            assert i == (4 + int(not drop_last)), i
            if drop_last:
                assert s != set(range(11))
            else:
                assert s == set(range(11))

    @pytest.mark.parametrize(
        "batch_size,num_slices,slice_len,prioritized",
        [
            [100, 20, None, True],
            [100, 20, None, False],
            [120, 30, None, False],
            [100, None, 5, False],
            [120, None, 4, False],
            [101, None, 101, False],
        ],
    )
    @pytest.mark.parametrize("episode_key", ["episode", ("some", "episode")])
    @pytest.mark.parametrize("done_key", ["done", ("some", "done")])
    @pytest.mark.parametrize("match_episode", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_slice_sampler(
        self,
        batch_size,
        num_slices,
        slice_len,
        prioritized,
        episode_key,
        done_key,
        match_episode,
        device,
    ):
        torch.manual_seed(0)
        storage = LazyMemmapStorage(100)
        episode = torch.zeros(100, dtype=torch.int, device=device)
        episode[:30] = 1
        episode[30:55] = 2
        episode[55:70] = 3
        episode[70:] = 4
        steps = torch.cat(
            [torch.arange(30), torch.arange(25), torch.arange(15), torch.arange(30)], 0
        )

        done = torch.zeros(100, 1, dtype=torch.bool)
        done[torch.tensor([29, 54, 69, 99])] = 1

        data = TensorDict(
            {
                # we only use episode_key if we want the sampler to access it
                episode_key if match_episode else "whatever_episode": episode,
                "another_episode": episode,
                "obs": torch.randn((3, 4, 5)).expand(100, 3, 4, 5),
                "act": torch.randn((20,)).expand(100, 20),
                "steps": steps,
                "count": torch.arange(100),
                "other": torch.randn((20, 50)).expand(100, 20, 50),
                done_key: done,
                _replace_last(done_key, "terminated"): done,
            },
            [100],
            device=device,
        )
        storage.set(range(100), data)
        if slice_len is not None and slice_len > 15:
            # we may have to sample trajs shorter than slice_len
            strict_length = False
        else:
            strict_length = True

        if prioritized:
            num_steps = data.shape[0]
            sampler = PrioritizedSliceSampler(
                max_capacity=num_steps,
                alpha=0.7,
                beta=0.9,
                num_slices=num_slices,
                traj_key=episode_key,
                end_key=done_key,
                slice_len=slice_len,
                strict_length=strict_length,
                truncated_key=_replace_last(done_key, "truncated"),
            )
            index = torch.arange(0, num_steps, 1)
            sampler.extend(index)
            sampler.update_priority(index, 1)
        else:
            sampler = SliceSampler(
                num_slices=num_slices,
                traj_key=episode_key,
                end_key=done_key,
                slice_len=slice_len,
                strict_length=strict_length,
                truncated_key=_replace_last(done_key, "truncated"),
            )
        if slice_len is not None:
            num_slices = batch_size // slice_len
        trajs_unique_id = set()
        too_short = False
        count_unique = set()
        for _ in range(50):
            index, info = sampler.sample(storage, batch_size=batch_size)
            samples = storage._storage[index]
            if strict_length:
                # check that trajs are ok
                samples = samples.view(num_slices, -1)

                unique_another_episode = (
                    samples["another_episode"].unique(dim=1).squeeze()
                )
                assert unique_another_episode.shape == torch.Size([num_slices]), (
                    num_slices,
                    samples,
                )
                assert (
                    samples["steps"][..., 1:] - 1 == samples["steps"][..., :-1]
                ).all()
            if isinstance(index, tuple):
                index_numel = index[0].numel()
            else:
                index_numel = index.numel()

            too_short = too_short or index_numel < batch_size
            trajs_unique_id = trajs_unique_id.union(
                samples["another_episode"].view(-1).tolist()
            )
            count_unique = count_unique.union(samples.get("count").view(-1).tolist())

            truncated = info[_replace_last(done_key, "truncated")]
            terminated = info[_replace_last(done_key, "terminated")]
            assert (truncated | terminated).view(num_slices, -1)[:, -1].all()
            assert (
                terminated
                == samples[_replace_last(done_key, "terminated")].view_as(terminated)
            ).all()
            done = info[done_key]
            assert done.view(num_slices, -1)[:, -1].all()

            if len(count_unique) == 100:
                # all items have been sampled
                break
        else:
            raise AssertionError(
                f"Not all items can be sampled: {set(range(100))-count_unique} are missing"
            )

        if strict_length:
            assert not too_short
        else:
            assert too_short

        assert len(trajs_unique_id) == 4

    @pytest.mark.parametrize("sampler", [SliceSampler, SliceSamplerWithoutReplacement])
    def test_slice_sampler_at_capacity(self, sampler):
        torch.manual_seed(0)

        trajectory0 = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3])
        trajectory1 = torch.arange(2).repeat_interleave(6)
        trajectory = torch.stack([trajectory0, trajectory1], 0)

        td = TensorDict(
            {"trajectory": trajectory, "steps": torch.arange(12).expand(2, 12)}, [2, 12]
        )

        rb = ReplayBuffer(
            sampler=sampler(traj_key="trajectory", num_slices=2),
            storage=LazyTensorStorage(20, ndim=2),
            batch_size=6,
        )

        rb.extend(td)

        for s in rb:
            if (s["steps"] == 9).any():
                break
        else:
            raise AssertionError

    def test_slice_sampler_errors(self):
        device = "cpu"
        batch_size, num_slices = 100, 20

        episode = torch.zeros(100, dtype=torch.int, device=device)
        episode[:30] = 1
        episode[30:55] = 2
        episode[55:70] = 3
        episode[70:] = 4
        steps = torch.cat(
            [torch.arange(30), torch.arange(25), torch.arange(15), torch.arange(30)], 0
        )

        done = torch.zeros(100, 1, dtype=torch.bool)
        done[torch.tensor([29, 54, 69])] = 1

        data = TensorDict(
            {
                # we only use episode_key if we want the sampler to access it
                "episode": episode,
                "another_episode": episode,
                "obs": torch.randn((3, 4, 5)).expand(100, 3, 4, 5),
                "act": torch.randn((20,)).expand(100, 20),
                "steps": steps,
                "other": torch.randn((20, 50)).expand(100, 20, 50),
                ("next", "done"): done,
            },
            [100],
            device=device,
        )

        data_wrong_done = data.clone(False)
        data_wrong_done.rename_key_("episode", "_")
        data_wrong_done["next", "done"] = done.unsqueeze(1).expand(100, 5, 1)
        storage = LazyMemmapStorage(100)
        storage.set(range(100), data_wrong_done)
        sampler = SliceSampler(num_slices=num_slices)
        with pytest.raises(
            RuntimeError,
            match="Expected the end-of-trajectory signal to be 1-dimensional",
        ):
            index, _ = sampler.sample(storage, batch_size=batch_size)

        storage = ListStorage(100)
        storage.set(range(100), data)
        sampler = SliceSampler(num_slices=num_slices)
        with pytest.raises(
            RuntimeError,
            match="Could not get a tensordict out of the storage, which is required for SliceSampler to compute the trajectories.",
        ):
            index, _ = sampler.sample(storage, batch_size=batch_size)

    @pytest.mark.parametrize("batch_size,num_slices", [[20, 4], [4, 2]])
    @pytest.mark.parametrize("episode_key", ["episode", ("some", "episode")])
    @pytest.mark.parametrize("done_key", ["done", ("some", "done")])
    @pytest.mark.parametrize("match_episode", [True, False])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_slice_sampler_without_replacement(
        self,
        batch_size,
        num_slices,
        episode_key,
        done_key,
        match_episode,
        device,
    ):
        torch.manual_seed(0)
        storage = LazyMemmapStorage(100)
        episode = torch.zeros(100, dtype=torch.int, device=device)
        steps = []
        done = torch.zeros(100, 1, dtype=torch.bool)
        for i in range(0, 100, 5):
            episode[i : i + 5] = i // 5
            steps.append(torch.arange(5))
            done[i + 4] = 1
        steps = torch.cat(steps)

        data = TensorDict(
            {
                # we only use episode_key if we want the sampler to access it
                episode_key if match_episode else "whatever_episode": episode,
                "another_episode": episode,
                "obs": torch.randn((3, 4, 5)).expand(100, 3, 4, 5),
                "act": torch.randn((20,)).expand(100, 20),
                "steps": steps,
                "other": torch.randn((20, 50)).expand(100, 20, 50),
                done_key: done,
            },
            [100],
            device=device,
        )
        storage.set(range(100), data)
        sampler = SliceSamplerWithoutReplacement(
            num_slices=num_slices, traj_key=episode_key, end_key=done_key
        )
        trajs_unique_id = set()
        for i in range(5):
            index, info = sampler.sample(storage, batch_size=batch_size)
            samples = storage._storage[index]

            # check that trajs are ok
            samples = samples.view(num_slices, -1)
            assert samples["another_episode"].unique(
                dim=1
            ).squeeze().shape == torch.Size([num_slices])
            assert (samples["steps"][..., 1:] - 1 == samples["steps"][..., :-1]).all()
            cur_episodes = samples["another_episode"].view(-1).tolist()
            for ep in cur_episodes:
                assert ep not in trajs_unique_id, i
            trajs_unique_id = trajs_unique_id.union(
                cur_episodes,
            )
        done_recon = info[("next", "truncated")] | info[("next", "terminated")]
        assert done_recon.view(num_slices, -1)[:, -1].all()
        done = info[("next", "done")]
        assert done.view(num_slices, -1)[:, -1].all()

    def test_slice_sampler_left_right(self):
        torch.manual_seed(0)
        data = TensorDict(
            {"obs": torch.arange(1, 11).repeat(10), "eps": torch.arange(100) // 10 + 1},
            [100],
        )

        for N in (2, 4):
            rb = TensorDictReplayBuffer(
                sampler=SliceSampler(num_slices=10, traj_key="eps", span=(N, N)),
                batch_size=50,
                storage=LazyMemmapStorage(100),
            )
            rb.extend(data)

            for _ in range(10):
                sample = rb.sample()
                sample = split_trajectories(sample)
                assert (sample["next", "truncated"].squeeze(-1).sum(-1) == 1).all()
                assert ((sample["obs"] == 0).sum(-1) <= N).all(), sample["obs"]
                assert ((sample["eps"] == 0).sum(-1) <= N).all()
                for i in range(sample.shape[0]):
                    curr_eps = sample[i]["eps"]
                    curr_eps = curr_eps[curr_eps != 0]
                    assert curr_eps.unique().numel() == 1

    def test_slice_sampler_left_right_ndim(self):
        torch.manual_seed(0)
        data = TensorDict(
            {"obs": torch.arange(1, 11).repeat(12), "eps": torch.arange(120) // 10 + 1},
            [120],
        )
        data = data.reshape(4, 30)

        for N in (2, 4):
            rb = TensorDictReplayBuffer(
                sampler=SliceSampler(num_slices=10, traj_key="eps", span=(N, N)),
                batch_size=50,
                storage=LazyMemmapStorage(100, ndim=2),
            )
            rb.extend(data)

            for _ in range(10):
                sample = rb.sample()
                sample = split_trajectories(sample)
                assert (sample["next", "truncated"].squeeze(-1).sum(-1) <= 1).all()
                assert ((sample["obs"] == 0).sum(-1) <= N).all(), sample["obs"]
                assert ((sample["eps"] == 0).sum(-1) <= N).all()
                for i in range(sample.shape[0]):
                    curr_eps = sample[i]["eps"]
                    curr_eps = curr_eps[curr_eps != 0]
                    assert curr_eps.unique().numel() == 1

    def test_slice_sampler_strictlength(self):

        torch.manual_seed(0)

        data = TensorDict(
            {
                "traj": torch.cat(
                    [
                        torch.ones(2, dtype=torch.int),
                        torch.zeros(10, dtype=torch.int),
                    ],
                    dim=0,
                ),
                "x": torch.arange(12),
            },
            [12],
        )

        buffer = ReplayBuffer(
            storage=LazyTensorStorage(12),
            sampler=SliceSampler(num_slices=2, strict_length=True, traj_key="traj"),
            batch_size=8,
        )
        buffer.extend(data)

        for _ in range(50):
            sample = buffer.sample()
            assert sample.shape == torch.Size([8])
            assert (sample["traj"] == 0).all()

        buffer = ReplayBuffer(
            storage=LazyTensorStorage(12),
            sampler=SliceSampler(num_slices=2, strict_length=False, traj_key="traj"),
            batch_size=8,
        )
        buffer.extend(data)

        for _ in range(50):
            sample = buffer.sample()
            if sample.shape == torch.Size([6]):
                assert (sample["traj"] != 0).any()
            else:
                assert len(sample["traj"].unique()) == 1

    @pytest.mark.parametrize("ndim", [1, 2])
    @pytest.mark.parametrize("strict_length", [True, False])
    @pytest.mark.parametrize("circ", [False, True])
    @pytest.mark.parametrize("at_capacity", [False, True])
    def test_slice_sampler_prioritized(self, ndim, strict_length, circ, at_capacity):
        torch.manual_seed(0)
        out = []
        for t in range(5):
            length = (t + 1) * 5
            done = torch.zeros(length, 1, dtype=torch.bool)
            done[-1] = 1
            priority = 10 if t == 0 else 1
            traj = TensorDict(
                {
                    "traj": torch.full((length,), t),
                    "step_count": torch.arange(length),
                    "done": done,
                    "priority": torch.full((length,), priority),
                },
                batch_size=length,
            )
            out.append(traj)
        data = torch.cat(out)
        if ndim == 2:
            data = torch.stack([data, data])
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(data.numel() - at_capacity, ndim=ndim),
            sampler=PrioritizedSliceSampler(
                max_capacity=data.numel() - at_capacity,
                alpha=1.0,
                beta=1.0,
                end_key="done",
                slice_len=10,
                strict_length=strict_length,
                cache_values=True,
            ),
            batch_size=50,
        )
        if not circ:
            # Simplest case: the buffer is full but no overlap
            index = rb.extend(data)
        else:
            # The buffer is 2/3 -> 1/3 overlapping
            rb.extend(data[..., : data.shape[-1] // 3])
            index = rb.extend(data)
        rb.update_priority(index, data["priority"])
        samples = []
        found_shorter_batch = False
        for _ in range(100):
            samples.append(rb.sample())
            if samples[-1].numel() < 50:
                found_shorter_batch = True
        samples = torch.cat(samples)
        if strict_length:
            assert not found_shorter_batch
        else:
            assert found_shorter_batch
        # the first trajectory has a very high priority, but should only appear
        # if strict_length=False.
        if strict_length:
            assert (samples["traj"] != 0).all(), samples["traj"].unique()
        else:
            assert (samples["traj"] == 0).any()
            # Check that all samples of the first traj contain all elements (since it's too short to fulfill 10 elts)
            sc = samples[samples["traj"] == 0]["step_count"]
            assert (sc == 1).sum() == (sc == 2).sum()
            assert (sc == 1).sum() == (sc == 4).sum()
        assert rb._sampler._cache
        rb.extend(data)
        assert not rb._sampler._cache

    @pytest.mark.parametrize("ndim", [1, 2])
    @pytest.mark.parametrize("strict_length", [True, False])
    @pytest.mark.parametrize("circ", [False, True])
    @pytest.mark.parametrize(
        "span", [False, [False, False], [False, True], 3, [False, 3]]
    )
    def test_slice_sampler_prioritized_span(self, ndim, strict_length, circ, span):
        torch.manual_seed(0)
        out = []
        # 5 trajs of length 3, 6, 9, 12 and 15
        for t in range(5):
            length = (t + 1) * 3
            done = torch.zeros(length, 1, dtype=torch.bool)
            done[-1] = 1
            priority = 1
            traj = TensorDict(
                {
                    "traj": torch.full((length,), t),
                    "step_count": torch.arange(length),
                    "done": done,
                    "priority": torch.full((length,), priority),
                },
                batch_size=length,
            )
            out.append(traj)
        data = torch.cat(out)
        if ndim == 2:
            data = torch.stack([data, data])
        rb = TensorDictReplayBuffer(
            storage=LazyTensorStorage(data.numel(), ndim=ndim),
            sampler=PrioritizedSliceSampler(
                max_capacity=data.numel(),
                alpha=1.0,
                beta=1.0,
                end_key="done",
                slice_len=5,
                strict_length=strict_length,
                cache_values=True,
                span=span,
            ),
            batch_size=5,
        )
        if not circ:
            # Simplest case: the buffer is full but no overlap
            index = rb.extend(data)
        else:
            # The buffer is 2/3 -> 1/3 overlapping
            rb.extend(data[..., : data.shape[-1] // 3])
            index = rb.extend(data)
        rb.update_priority(index, data["priority"])
        found_traj_0 = False
        found_traj_4_truncated_right = False
        for i, s in enumerate(rb):
            t = s["traj"].unique().tolist()
            assert len(t) == 1
            t = t[0]
            if t == 0:
                found_traj_0 = True
            if t == 4 and s.numel() < 5:
                if s["step_count"][0] > 10:
                    found_traj_4_truncated_right = True
                if s["step_count"][0] == 0:
                    pass
            if i == 1000:
                break
        assert not rb._sampler.span[0]
        # if rb._sampler.span[0]:
        #     assert found_traj_4_truncated_left
        if rb._sampler.span[1]:
            assert found_traj_4_truncated_right
        else:
            assert not found_traj_4_truncated_right
        if strict_length and not rb._sampler.span[1]:
            assert not found_traj_0
        else:
            assert found_traj_0

    @pytest.mark.parametrize("max_priority_within_buffer", [True, False])
    def test_prb_update_max_priority(self, max_priority_within_buffer):
        rb = ReplayBuffer(
            storage=LazyTensorStorage(11),
            sampler=PrioritizedSampler(
                max_capacity=11,
                alpha=1.0,
                beta=1.0,
                max_priority_within_buffer=max_priority_within_buffer,
            ),
        )
        for data in torch.arange(20):
            idx = rb.add(data)
            rb.update_priority(idx, 21 - data)
            if data <= 10:
                # The max is always going to be the first value
                assert rb._sampler._max_priority[0] == 21
                assert rb._sampler._max_priority[1] == 0
            elif not max_priority_within_buffer:
                # The max is the historical max, which was at idx 0
                assert rb._sampler._max_priority[0] == 21
                assert rb._sampler._max_priority[1] == 0
            else:
                # the max is the current max. Find it and compare
                sumtree = torch.as_tensor(
                    [rb._sampler._sum_tree[i] for i in range(rb._sampler._max_capacity)]
                )
                assert rb._sampler._max_priority[0] == sumtree.max()
                assert rb._sampler._max_priority[1] == sumtree.argmax()
        idx = rb.extend(torch.arange(10))
        rb.update_priority(idx, 12)
        if max_priority_within_buffer:
            assert rb._sampler._max_priority[0] == 12
            assert rb._sampler._max_priority[1] == 0
        else:
            assert rb._sampler._max_priority[0] == 21
            assert rb._sampler._max_priority[1] == 0

    def test_prb_ndim(self):
        """This test lists all the possible ways of updating the priority of a PRB with RB, TRB and TPRB.

        All tests are done for 1d and 2d TDs.

        """
        torch.manual_seed(0)
        np.random.seed(0)

        # first case: 1d, RB
        rb = ReplayBuffer(
            sampler=PrioritizedSampler(max_capacity=100, alpha=1.0, beta=1.0),
            storage=LazyTensorStorage(100),
            batch_size=4,
        )
        data = TensorDict({"a": torch.arange(10), "p": torch.ones(10) / 2}, [10])
        idx = rb.extend(data)
        assert (torch.tensor([rb._sampler._sum_tree[i] for i in range(10)]) == 1).all()
        rb.update_priority(idx, 2)
        assert (torch.tensor([rb._sampler._sum_tree[i] for i in range(10)]) == 2).all()
        s, info = rb.sample(return_info=True)
        rb.update_priority(info["index"], 3)
        assert (
            torch.tensor([rb._sampler._sum_tree[i] for i in range(10)])[info["index"]]
            == 3
        ).all()

        # second case: 1d, TRB
        rb = TensorDictReplayBuffer(
            sampler=PrioritizedSampler(max_capacity=100, alpha=1.0, beta=1.0),
            storage=LazyTensorStorage(100),
            batch_size=4,
        )
        data = TensorDict({"a": torch.arange(10), "p": torch.ones(10) / 2}, [10])
        idx = rb.extend(data)
        assert (torch.tensor([rb._sampler._sum_tree[i] for i in range(10)]) == 1).all()
        rb.update_priority(idx, 2)
        assert (torch.tensor([rb._sampler._sum_tree[i] for i in range(10)]) == 2).all()
        s = rb.sample()
        rb.update_priority(s["index"], 3)
        assert (
            torch.tensor([rb._sampler._sum_tree[i] for i in range(10)])[s["index"]] == 3
        ).all()

        # third case: 1d TPRB
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=1.0,
            beta=1.0,
            storage=LazyTensorStorage(100),
            batch_size=4,
            priority_key="p",
        )
        data = TensorDict({"a": torch.arange(10), "p": torch.ones(10) / 2}, [10])
        idx = rb.extend(data)
        assert (torch.tensor([rb._sampler._sum_tree[i] for i in range(10)]) == 1).all()
        rb.update_priority(idx, 2)
        assert (torch.tensor([rb._sampler._sum_tree[i] for i in range(10)]) == 2).all()
        s = rb.sample()

        s["p"] = torch.ones(4) * 10_000
        rb.update_tensordict_priority(s)
        assert (
            torch.tensor([rb._sampler._sum_tree[i] for i in range(10)])[s["index"]]
            == 10_000
        ).all()

        s2 = rb.sample()
        # All indices in s2 must be from s since we set a very high priority to these items
        assert (s2["index"].unsqueeze(0) == s["index"].unsqueeze(1)).any(0).all()

        # fourth case: 2d RB
        rb = ReplayBuffer(
            sampler=PrioritizedSampler(max_capacity=100, alpha=1.0, beta=1.0),
            storage=LazyTensorStorage(100, ndim=2),
            batch_size=4,
        )
        data = TensorDict(
            {"a": torch.arange(5).expand(2, 5), "p": torch.ones(2, 5) / 2}, [2, 5]
        )
        idx = rb.extend(data)
        assert (torch.tensor([rb._sampler._sum_tree[i] for i in range(10)]) == 1).all()
        rb.update_priority(idx, 2)
        assert (torch.tensor([rb._sampler._sum_tree[i] for i in range(10)]) == 2).all()

        s, info = rb.sample(return_info=True)
        rb.update_priority(info["index"], 3)
        priorities = torch.tensor(
            [rb._sampler._sum_tree[i] for i in range(10)]
        ).reshape((5, 2))
        assert (priorities[info["index"]] == 3).all()

        # fifth case: 2d TRB
        # 2d
        rb = TensorDictReplayBuffer(
            sampler=PrioritizedSampler(max_capacity=100, alpha=1.0, beta=1.0),
            storage=LazyTensorStorage(100, ndim=2),
            batch_size=4,
        )
        data = TensorDict(
            {"a": torch.arange(5).expand(2, 5), "p": torch.ones(2, 5) / 2}, [2, 5]
        )
        idx = rb.extend(data)
        assert (torch.tensor([rb._sampler._sum_tree[i] for i in range(10)]) == 1).all()
        rb.update_priority(idx, 2)
        assert (torch.tensor([rb._sampler._sum_tree[i] for i in range(10)]) == 2).all()

        s = rb.sample()
        rb.update_priority(s["index"], 10_000)
        priorities = torch.tensor(
            [rb._sampler._sum_tree[i] for i in range(10)]
        ).reshape((5, 2))
        assert (priorities[s["index"].unbind(-1)] == 10_000).all()

        s2 = rb.sample()
        assert (
            (s2["index"].unsqueeze(0) == s["index"].unsqueeze(1)).all(-1).any(0).all()
        )

        # Sixth case: 2d TDPRB
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=1.0,
            beta=1.0,
            storage=LazyTensorStorage(100, ndim=2),
            batch_size=4,
            priority_key="p",
        )
        data = TensorDict(
            {"a": torch.arange(5).expand(2, 5), "p": torch.ones(2, 5) / 2}, [2, 5]
        )
        idx = rb.extend(data)
        assert (torch.tensor([rb._sampler._sum_tree[i] for i in range(10)]) == 1).all()
        rb.update_priority(idx, torch.ones(()) * 2)
        assert (torch.tensor([rb._sampler._sum_tree[i] for i in range(10)]) == 2).all()
        s = rb.sample()
        # setting the priorities to a value that is so big that the buffer will resample them
        s["p"] = torch.ones(4) * 10_000
        rb.update_tensordict_priority(s)
        priorities = torch.tensor(
            [rb._sampler._sum_tree[i] for i in range(10)]
        ).reshape((5, 2))
        assert (priorities[s["index"].unbind(-1)] == 10_000).all()

        s2 = rb.sample()
        assert (
            (s2["index"].unsqueeze(0) == s["index"].unsqueeze(1)).all(-1).any(0).all()
        )


def test_prioritized_slice_sampler_doc_example():
    sampler = PrioritizedSliceSampler(max_capacity=9, num_slices=3, alpha=0.7, beta=0.9)
    rb = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(9), sampler=sampler, batch_size=6
    )
    data = TensorDict(
        {
            "observation": torch.randn(9, 16),
            "action": torch.randn(9, 1),
            "episode": torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2], dtype=torch.long),
            "steps": torch.tensor([0, 1, 2, 0, 1, 2, 0, 1, 2], dtype=torch.long),
            ("next", "observation"): torch.randn(9, 16),
            ("next", "reward"): torch.randn(9, 1),
            ("next", "done"): torch.tensor(
                [0, 0, 1, 0, 0, 1, 0, 0, 1], dtype=torch.bool
            ).unsqueeze(1),
        },
        batch_size=[9],
    )
    rb.extend(data)
    sample, info = rb.sample(return_info=True)
    # print("episode", sample["episode"].tolist())
    # print("steps", sample["steps"].tolist())
    # print("weight", info["_weight"].tolist())

    priority = torch.tensor([0, 3, 3, 0, 0, 0, 1, 1, 1])
    rb.update_priority(torch.arange(0, 9, 1), priority=priority)
    sample, info = rb.sample(return_info=True)
    # print("episode", sample["episode"].tolist())
    # print("steps", sample["steps"].tolist())
    # print("weight", info["_weight"].tolist())


@pytest.mark.parametrize("device", get_default_devices())
def test_prioritized_slice_sampler_episodes(device):
    num_slices = 10
    batch_size = 20

    episode = torch.zeros(100, dtype=torch.int, device=device)
    episode[:30] = 1
    episode[30:55] = 2
    episode[55:70] = 3
    episode[70:] = 4
    steps = torch.cat(
        [torch.arange(30), torch.arange(25), torch.arange(15), torch.arange(30)], 0
    )
    done = torch.zeros(100, 1, dtype=torch.bool)
    done[torch.tensor([29, 54, 69])] = 1

    data = TensorDict(
        {
            "observation": torch.randn(100, 16),
            "action": torch.randn(100, 4),
            "episode": episode,
            "steps": steps,
            ("next", "observation"): torch.randn(100, 16),
            ("next", "reward"): torch.randn(100, 1),
            ("next", "done"): done,
        },
        batch_size=[100],
        device=device,
    )

    num_steps = data.shape[0]
    sampler = PrioritizedSliceSampler(
        max_capacity=num_steps,
        alpha=0.7,
        beta=0.9,
        num_slices=num_slices,
    )

    rb = TensorDictReplayBuffer(
        storage=LazyMemmapStorage(100),
        sampler=sampler,
        batch_size=batch_size,
    )
    rb.extend(data)

    episodes = []
    for _ in range(10):
        sample = rb.sample()
        episodes.append(sample["episode"])
    assert {1, 2, 3, 4} == set(
        torch.cat(episodes).cpu().tolist()
    ), "all episodes are expected to be sampled at least once"

    index = torch.arange(0, num_steps, 1)
    new_priorities = torch.cat(
        [torch.ones(30), torch.zeros(25), torch.ones(15), torch.zeros(30)], 0
    )
    sampler.update_priority(index, new_priorities)

    episodes = []
    for _ in range(10):
        sample = rb.sample()
        episodes.append(sample["episode"])
    assert {1, 3} == set(
        torch.cat(episodes).cpu().tolist()
    ), "after priority update, only episode 1 and 3 are expected to be sampled"


@pytest.mark.parametrize("alpha", [0.6, torch.tensor(1.0)])
@pytest.mark.parametrize("beta", [0.7, torch.tensor(0.1)])
@pytest.mark.parametrize("gamma", [0.1])
@pytest.mark.parametrize("total_steps", [200])
@pytest.mark.parametrize("n_annealing_steps", [100])
@pytest.mark.parametrize("anneal_every_n", [10, 159])
@pytest.mark.parametrize("alpha_min", [0, 0.2])
@pytest.mark.parametrize("beta_max", [1, 1.4])
def test_prioritized_parameter_scheduler(
    alpha,
    beta,
    gamma,
    total_steps,
    n_annealing_steps,
    anneal_every_n,
    alpha_min,
    beta_max,
):
    rb = TensorDictPrioritizedReplayBuffer(
        alpha=alpha, beta=beta, storage=ListStorage(max_size=1000)
    )
    data = TensorDict({"data": torch.randn(1000, 5)}, batch_size=1000)
    rb.extend(data)
    alpha_scheduler = LinearScheduler(
        rb, param_name="alpha", final_value=alpha_min, num_steps=n_annealing_steps
    )
    beta_scheduler = StepScheduler(
        rb,
        param_name="beta",
        gamma=gamma,
        n_steps=anneal_every_n,
        max_value=beta_max,
        mode="additive",
    )

    scheduler = SchedulerList(schedulers=(alpha_scheduler, beta_scheduler))

    alpha = alpha if torch.is_tensor(alpha) else torch.tensor(alpha)
    alpha_min = torch.tensor(alpha_min)
    expected_alpha_vals = torch.linspace(alpha, alpha_min, n_annealing_steps + 1)
    expected_alpha_vals = torch.nn.functional.pad(
        expected_alpha_vals, (0, total_steps - n_annealing_steps), value=alpha_min
    )

    expected_beta_vals = [beta]
    annealing_steps = total_steps // anneal_every_n
    gammas = torch.arange(0, annealing_steps + 1, dtype=torch.float32) * gamma
    expected_beta_vals = (
        (beta + gammas).repeat_interleave(anneal_every_n).clip(None, beta_max)
    )
    for i in range(total_steps):
        curr_alpha = rb.sampler.alpha
        torch.testing.assert_close(
            curr_alpha
            if torch.is_tensor(curr_alpha)
            else torch.tensor(curr_alpha).float(),
            expected_alpha_vals[i],
            msg=f"expected {expected_alpha_vals[i]}, got {curr_alpha}",
        )
        curr_beta = rb.sampler.beta
        torch.testing.assert_close(
            curr_beta
            if torch.is_tensor(curr_beta)
            else torch.tensor(curr_beta).float(),
            expected_beta_vals[i],
            msg=f"expected {expected_beta_vals[i]}, got {curr_beta}",
        )
        rb.sample(20)
        scheduler.step()


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
        assert isinstance(rb._storage[:], StorageEnsemble)
        assert isinstance(rb._storage[:2], StorageEnsemble)
        assert isinstance(rb._storage[torch.tensor([0, 1])], StorageEnsemble)
        assert isinstance(rb._storage[np.array([0, 1])], StorageEnsemble)
        assert isinstance(rb._storage[[0, 1]], StorageEnsemble)
        assert isinstance(rb._storage[1], LazyMemmapStorage)

        rb._storage[:, :3]
        rb._storage[:2, :3]
        rb._storage[torch.tensor([0, 1]), :3]
        rb._storage[np.array([0, 1]), :3]
        rb._storage[[0, 1], :3]

        assert isinstance(rb._sampler[:], SamplerEnsemble)
        assert isinstance(rb._sampler[:2], SamplerEnsemble)
        assert isinstance(rb._sampler[torch.tensor([0, 1])], SamplerEnsemble)
        assert isinstance(rb._sampler[np.array([0, 1])], SamplerEnsemble)
        assert isinstance(rb._sampler[[0, 1]], SamplerEnsemble)
        assert isinstance(rb._sampler[1], RandomSampler)

        assert isinstance(rb._writer[:], WriterEnsemble)
        assert isinstance(rb._writer[:2], WriterEnsemble)
        assert isinstance(rb._writer[torch.tensor([0, 1])], WriterEnsemble)
        assert isinstance(rb._writer[np.array([0, 1])], WriterEnsemble)
        assert isinstance(rb._writer[[0, 1]], WriterEnsemble)
        assert isinstance(rb._writer[0], RoundRobinWriter)


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
        if os.getenv("PYTORCH_TEST_FBCODE"):
            from pytorch.rl.test._utils_internal import CARTPOLE_VERSIONED
        else:
            from _utils_internal import CARTPOLE_VERSIONED

        torch.manual_seed(0)
        env = SerialEnv(2, lambda: GymEnv(CARTPOLE_VERSIONED()), device=env_device)
        env.set_seed(0)
        collector = SyncDataCollector(
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
                )
            return
        rb = rbtype(
            storage=storage_cls(max_size=10, ndim=2),
            sampler=sampler_cls(),
            writer=writer_cls(),
        )
        if not isinstance(rb._sampler, SliceSampler) and transform is not None:
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
        collector = SyncDataCollector(
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
            assert rb._writer._cursor == rb_test._writer._cursor

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
        collector = SyncDataCollector(
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
            assert rb._storage.max_size == 102
            if frames_per_batch > 100:
                assert rb._storage._is_full
                assert len(rb) == 102
                # Checks that when writing to the buffer with a batch greater than the total
                # size, we get the last step written properly.
                assert (rb[:]["next", "step_count"][:, -1] != 0).any()
            rb.dumps(tmpdir)
            rb.dumps(tmpdir)
            rb_test.loads(tmpdir)
            assert_allclose_td(rb_test[:], rb[:])
            assert rb._writer._cursor == rb_test._writer._cursor


class TestHistory:
    @pytest.fixture(scope="class", autouse=True)
    def set_context(self):
        with set_list_to_stack(True):
            yield

    def test_history_construct(self):
        hst0 = History(role="user", content="a message")
        assert not hst0.shape
        hst1 = History(role="user", content="another message")
        with pytest.raises(RuntimeError, match="unsqueeze"):
            hst0.append(hst1)
        hst0 = hst0.unsqueeze(0)

        # In an env.step, we typically have one more piece of history to add to the stack
        assert not hst1.shape
        assert not hst1.batch_size
        assert not hst1.batch_dims
        # test out-place
        hst0_copy = hst0.copy()
        hst0b = hst0.append(hst1, inplace=False)
        assert hst0b is not hst0
        assert (hst0 == hst0_copy).all()
        assert (hst0b[:-1] == hst0).all()

        # test in-place
        hst0b = hst0.append(hst1)
        assert hst0b is hst0
        assert hst0b.shape == (2,)

        assert hst0b.content == ["a message", "another message"]
        hst2 = History(
            role=["assistant", "user"],
            content=["i'm the assistant", "i'm the user"],
            batch_size=2,
        )
        assert hst2[0].role == "assistant"
        assert hst2[0].content == "i'm the assistant"
        assert hst2[1].role == "user"
        assert hst2[1].content == "i'm the user"
        with pytest.raises(RuntimeError, match="The new history to extend"):
            hst0.extend(hst1)

        # test out-place
        hst0_copy = hst0.copy()
        hst0b = hst0.extend(hst2, inplace=False)
        assert hst0b is not hst0
        assert (hst0 == hst0_copy).all()
        assert (hst0b[:-2] == hst0).all()

        # test in-place
        hst0b = hst0.extend(hst2)

        assert hst0b is hst0
        assert hst0.__dict__["_tensordict"].shape == (4,)
        assert hst0.shape == (4,)
        assert hst0.role == ["user", "user", "assistant", "user"]
        assert hst0.content == [
            "a message",
            "another message",
            "i'm the assistant",
            "i'm the user",
        ]

    def test_history_construct_ndim(self):
        hst0 = History(role="user", content="a message").unsqueeze(0).unsqueeze(0)
        hst1 = History(role="user", content="another message").unsqueeze(0)

        # test out-place
        hst0_copy = hst0.copy()
        hst0b = hst0.append(hst1, inplace=False, dim=1)
        assert hst0b is not hst0
        assert (hst0 == hst0_copy).all()
        assert (hst0b[:, :-1] == hst0).all()

        # test in-place
        hst0b = hst0.append(hst1, dim=1)
        assert hst0b is hst0
        assert hst0b.shape == (
            1,
            2,
        )

        assert hst0b.content == [["a message", "another message"]]
        hst2 = History(
            role=["assistant", "user"],
            content=["i'm the assistant", "i'm the user"],
            batch_size=2,
        ).unsqueeze(0)

        # test out-place
        hst0_copy = hst0.copy()
        hst0b = hst0.extend(hst2, inplace=False, dim=1)
        assert hst0b is not hst0
        assert (hst0 == hst0_copy).all()
        assert (hst0b[:, :-2] == hst0).all()

        # test in-place
        hst0b = hst0.extend(hst2, dim=1)

        assert hst0b is hst0
        assert hst0.__dict__["_tensordict"].shape == (
            1,
            4,
        )
        assert hst0.shape == (
            1,
            4,
        )
        assert hst0.role == [["user", "user", "assistant", "user"]]
        assert hst0.content == [
            [
                "a message",
                "another message",
                "i'm the assistant",
                "i'm the user",
            ]
        ]

    @pytest.fixture(scope="class")
    def mock_history(self):
        history0 = History(
            role="system",
            content="""CONTENT
        This is the setup""",
        )
        history1 = History(
            role="user",
            content="""CONTENT
        This is the first user prompt""",
        )
        history2 = History(
            role="assistant",
            content="""CONTENT
        This is the second prompt, the first for the assistant.""",
        )
        history = torch.stack([history0, history1, history2])
        return history

    @pytest.fixture(scope="class")
    def tokenizer(self):
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained("GPT2")
        yield tokenizer

    @pytest.mark.skipif(not _has_transformers, reason="requires transformers library")
    def test_history_template(self, mock_history, tokenizer):
        history = mock_history
        data_str = history.apply_chat_template(
            tokenizer=tokenizer, add_generation_prompt=False
        )
        assert isinstance(data_str, str)
        data_token = history.apply_chat_template(
            tokenizer=tokenizer, tokenize=True, add_generation_prompt=False
        )
        assert isinstance(data_token, torch.Tensor)

        # test add_generation_prompt
        data_str = history.apply_chat_template(
            tokenizer=tokenizer, add_generation_prompt=True
        )
        assert isinstance(data_str, str)
        assert data_str.endswith("<|im_start|>assistant\n"), data_str

    @pytest.mark.skipif(not _has_transformers, reason="requires transformers library")
    def test_history_template_recover(self, mock_history, tokenizer):
        history = mock_history
        data_str = history.apply_chat_template(tokenizer=tokenizer)
        # Test inverse
        recovered = history._inv_chatml(data_str)
        assert recovered.role == history.role
        assert recovered.content == history.content
        data_token = history.apply_chat_template(
            tokenizer=tokenizer, tokenize=True, add_generation_prompt=False
        )
        recovered = history._inv_chatml(tokenizer.batch_decode(data_token)[0])

    def test_history_spec(self):
        history = History(
            role=["system", "user", "assistant", "user"],
            content=[
                "i'm the system",
                "i'm the user",
                "I'm the assistant",
                "I'm the user again",
            ],
        )
        spec = history.default_spec()
        r = spec.zero()
        assert isinstance(r, History)
        assert spec.is_in(r)
        assert spec.is_in(history)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
