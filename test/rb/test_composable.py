# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import pickle
import sys

import pytest
import torch
from _rb_common import (
    _os_is_windows,
    OLD_TORCH,
    ReplayBufferRNG,
    TensorDictReplayBufferRNG,
    TORCH_VERSION,
)
from packaging import version
from tensordict import is_tensor_collection, TensorDict, TensorDictBase
from torch.utils._pytree import tree_flatten, tree_map

from torchrl.data import (
    RemoteTensorDictReplayBuffer,
    ReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers import samplers, writers
from torchrl.data.replay_buffers.samplers import RandomSampler
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
from torchrl.testing import capture_log_records


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
    def _get_rb(
        self, rb_type, size, sampler, writer, storage, compilable=False, **kwargs
    ):
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
            **kwargs,
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

        with (
            pytest.warns(
                UserWarning,
                match="A cursor of length superior to the storage capacity was provided",
            )
            if cond
            else contextlib.nullcontext()
        ):
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
            and isinstance(rb.storage, TensorStorage)
        )
        if not is_tensor_collection(data) and writer is TensorDictMaxValueWriter:
            with pytest.raises(
                RuntimeError, match="expects data to be a tensor collection"
            ):
                rb.extend(data)
            return
        length = min(rb.storage.max_size, len(rb) + data_shape)
        if writer is TensorDictMaxValueWriter:
            data["next", "reward"][-length:] = 1_000_000
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
        if is_tensor_collection(data):
            data_iter = data[-length:]
        else:

            def data_iter():
                for t in range(-length, -1):
                    yield tree_map(lambda x, t=t: x[t], data)

            data_iter = data_iter()
        for d in data_iter:
            for b in rb.storage:
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
            and isinstance(rb.storage, TensorStorage)
        )
        with (
            pytest.warns(
                UserWarning,
                match="A cursor of length superior to the storage capacity was provided",
            )
            if cond
            else contextlib.nullcontext()
        ):
            rb.extend(data2)

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
    )
    # Compiling on Windows requires "cl" compiler to be installed.
    # <https://github.com/pytorch/pytorch/blob/8231180147a096a703d8891756068c89365292e0/torch/_inductor/cpp_builder.py#L143>
    # Our Windows CI jobs do not have "cl", so skip this test.
    @pytest.mark.skipif(_os_is_windows, reason="windows tests do not support compile")
    @pytest.mark.skipif(
        sys.version_info >= (3, 14),
        reason="torch.compile is not supported on Python 3.14+",
    )
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
            and isinstance(rb.storage, TensorStorage)
        )
        if not is_tensor_collection(data) and writer is TensorDictMaxValueWriter:
            with pytest.raises(
                RuntimeError, match="expects data to be a tensor collection"
            ):
                rb.extend(data)
            return
        with (
            pytest.warns(
                UserWarning,
                match="A cursor of length superior to the storage capacity was provided",
            )
            if cond
            else contextlib.nullcontext()
        ):
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
            and isinstance(rb.storage, TensorStorage)
        )
        if not is_tensor_collection(data) and writer is TensorDictMaxValueWriter:
            with pytest.raises(
                RuntimeError, match="expects data to be a tensor collection"
            ):
                rb.extend(data)
            return
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
            rb_type=rb_type,
            sampler=sampler,
            writer=writer,
            storage=storage,
            size=size,
            delayed_init=False,
        )
        serialized = pickle.dumps(rb)
        rb2 = pickle.loads(serialized)
        assert rb.__dict__.keys() == rb2.__dict__.keys()
        for key in sorted(rb.__dict__.keys()):
            assert isinstance(rb.__dict__[key], type(rb2.__dict__[key]))


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
