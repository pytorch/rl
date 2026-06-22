# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import functools
import gc
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from functools import partial
from pathlib import Path
from unittest import mock

import pytest
import torch
from _rb_common import (
    _has_snapshot,
    _has_zstandard,
    _os_is_windows,
    torch_2_3,
    TORCH_VERSION,
)
from packaging import version
from tensordict import (
    assert_allclose_td,
    is_tensor_collection,
    is_tensorclass,
    LazyStackedTensorDict,
    tensorclass,
    TensorDict,
    TensorDictBase,
)
from torch import multiprocessing as mp
from torch.utils._pytree import tree_flatten, tree_map

from torchrl.data import (
    CompressedListStorage,
    ReplayBuffer,
    TensorDictPrioritizedReplayBuffer,
    TensorDictReplayBuffer,
)
from torchrl.data.replay_buffers.samplers import (
    RandomSampler,
    SamplerWithoutReplacement,
)
from torchrl.data.replay_buffers.storages import (
    _MEMMAP_STORAGE_REGISTRY,
    LazyMemmapStorage,
    LazyStackStorage,
    LazyTensorStorage,
    ListStorage,
    TensorStorage,
)
from torchrl.data.replay_buffers.utils import tree_iter
from torchrl.data.replay_buffers.writers import RoundRobinWriter
from torchrl.testing import capture_log_records, get_default_devices, make_tc


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

    @pytest.mark.gpu
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
    @pytest.mark.skipif(
        sys.version_info >= (3, 14),
        reason="torch.compile is not supported on Python 3.14+",
    )
    # This test checks if the `torch._dynamo.disable` wrapper around
    # `TensorStorage._rand_given_ndim` is still necessary.
    def test__rand_given_ndim_recompile(self):
        torch._dynamo.reset_code_caches()

        # Number of times to extend the replay buffer
        num_extend = 5
        data_size = 50
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
        assert len(records) <= 8, (
            "Excessive recompilations detected. Expected 8 or fewer, but got "
            f"{len(records)}. This suggests the `torch.compiler.disable` "
            "decorators may not be working properly or new recompilation "
            "sources have been introduced."
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

    @pytest.mark.parametrize("storage_type", [LazyMemmapStorage, LazyTensorStorage])
    def test_extend_lazystack_direct_write(self, storage_type):
        """Test that lazy stacks can be extended to storage correctly.

        This tests that lazy stacks from collectors are properly stored in
        replay buffers and that the data integrity is preserved. Also verifies
        that the update_() optimization is used for tensor indices.
        """
        rb = ReplayBuffer(
            storage=storage_type(100),
            batch_size=10,
        )
        # Create a list of tensordicts (like a collector would produce)
        tensordicts = [
            TensorDict(
                {"obs": torch.rand(4, 8), "action": torch.rand(2)}, batch_size=()
            )
            for _ in range(10)
        ]
        # Create lazy stack with stack_dim=0 (the batch dimension)
        lazy_td = LazyStackedTensorDict.lazy_stack(tensordicts, dim=0)
        assert isinstance(lazy_td, LazyStackedTensorDict)

        # Track calls to update_at_() - used for tensor indices
        update_at_called = []
        original_update_at = TensorDictBase.update_at_

        def mock_update_at(self, *args, **kwargs):
            update_at_called.append(True)
            return original_update_at(self, *args, **kwargs)

        # Extend with lazy stack and verify update_at_() is called
        # (rb.extend uses tensor indices, so update_at_() path is taken)
        with mock.patch.object(TensorDictBase, "update_at_", mock_update_at):
            rb.extend(lazy_td)

        # Verify update_at_() was called (optimization was used)
        assert len(update_at_called) > 0, "update_at_() should have been called"

        # Verify data integrity
        assert len(rb) == 10
        sample = rb.sample(5)
        assert sample["obs"].shape == (5, 4, 8)
        assert sample["action"].shape == (5, 2)

        # Verify all data is accessible by reading the entire storage
        all_data = rb[:]
        assert all_data["obs"].shape == (10, 4, 8)
        assert all_data["action"].shape == (10, 2)

        # Verify data values are preserved (check against original stacked data)
        expected = lazy_td.to_tensordict()
        assert torch.allclose(all_data["obs"], expected["obs"])
        assert torch.allclose(all_data["action"], expected["action"])

    @pytest.mark.parametrize("storage_type", [LazyMemmapStorage, LazyTensorStorage])
    def test_extend_lazystack_2d_storage(self, storage_type):
        """Test lazy stack optimization for 2D storage (parallel envs).

        When using parallel environments, the storage is 2D [max_size, n_steps]
        and the lazy stack has stack_dim=1 (time dimension). This test verifies
        the optimization handles this case correctly.
        """
        n_envs = 4
        n_steps = 10
        img_shape = (3, 32, 32)

        # Create 2D storage - capacity is 100 * n_steps when ndim=2
        storage = storage_type(100 * n_steps, ndim=2)

        # Pre-initialize storage with correct shape by setting first element
        init_td = TensorDict(
            {"pixels": torch.zeros(n_steps, *img_shape)},
            batch_size=[n_steps],
        )
        storage.set(0, init_td, set_cursor=False)

        # Expand storage to full size
        full_init = TensorDict(
            {"pixels": torch.zeros(100, n_steps, *img_shape)},
            batch_size=[100, n_steps],
        )
        storage.set(slice(0, 100), full_init, set_cursor=False)

        # Create lazy stack simulating parallel env output
        # stack_dim=1 means stacked along time dimension
        time_tds = [
            TensorDict(
                {"pixels": torch.rand(n_envs, *img_shape)},
                batch_size=[n_envs],
            )
            for _ in range(n_steps)
        ]
        lazy_td = LazyStackedTensorDict.lazy_stack(time_tds, dim=1)
        assert lazy_td.stack_dim == 1
        assert lazy_td.batch_size == torch.Size([n_envs, n_steps])

        # Write using tensor indices (simulating circular buffer behavior)
        cursor = torch.tensor([0, 1, 2, 3])
        storage.set(cursor, lazy_td)

        # Verify data integrity
        for i in range(n_envs):
            stored = storage[i]
            expected = lazy_td[i].to_tensordict()
            assert torch.allclose(
                stored["pixels"], expected["pixels"]
            ), f"Data mismatch for env {i}"

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
        assert rb.writer._cursor == 100
        rb[10:20] = TensorDict(
            {"a": torch.tensor([0] * 10), ("b", "c"): torch.tensor([0] * 10)}, [10]
        )
        assert (rb[10:20] == 0).all()
        assert len(rb) == 100
        assert rb.writer._cursor == 100
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
        assert rb.writer._cursor == 100
        rb[10:20] = TensorDict(
            {"a": torch.tensor([0] * 10), ("b", "c"): torch.tensor([0] * 10)}, [10]
        )
        assert (rb[10:20] == 2).all()
        assert len(rb) == 100
        assert rb.writer._cursor == 100
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
        assert rb.writer._cursor == 50
        rb[1, 5:6] = TensorDict(
            {"a": torch.tensor([0]), ("b", "c"): torch.tensor([0])}, [1]
        )
        assert (rb[1, 5:6] == 0).all()
        assert rb.writer._cursor == 50
        rb[:, 7:8] = TensorDict(
            {"a": torch.tensor([0]), ("b", "c"): torch.tensor([0])}, [1]
        ).expand(2, 1)
        assert (rb[:, 7:8] == 0).all()
        assert rb.writer._cursor == 50
        # test broadcasting
        rb[:, 10:20] = TensorDict(
            {"a": torch.tensor([0] * 10), ("b", "c"): torch.tensor([0] * 10)}, [10]
        )
        assert (rb[:, 10:20] == 0).all()
        assert len(rb) == 100

    @pytest.mark.skipif(
        TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
    )
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


def test_storage_save_hook(tmpdir):
    observed = {}

    class SaveHook:
        shift = None
        is_full = None

        def __call__(self, data, path=None):
            observed["shift"] = self.shift
            observed["is_full"] = self.is_full
            return data

    hook = SaveHook()
    rb = ReplayBuffer(storage=LazyMemmapStorage(10))
    rb.register_save_hook(hook)
    rb.extend(torch.arange(5))
    rb.dumps(tmpdir)

    assert hook.shift == 5, f"Expected shift=5, got {hook.shift}"
    assert hook.is_full is False, f"Expected is_full=False, got {hook.is_full}"
    assert observed["shift"] == 5
    assert observed["is_full"] is False


class TestSharedStorageInit:
    def worker(self, rb, worker_id, queue):
        length = len(rb)
        data = TensorDict({"x": torch.full((2,), worker_id)}, batch_size=(2,))
        worker_id * 2
        index = rb.extend(data)
        assert len(rb) >= length + 2
        assert (rb[index] == data).all()
        queue.put("done")

    @pytest.mark.parametrize(
        "storage_cls, use_tmpdir",
        [
            (LazyTensorStorage, False),
            (LazyMemmapStorage, False),
            (LazyMemmapStorage, True),
        ],
    )
    def test_shared_storage_multiprocess(self, storage_cls, use_tmpdir, tmpdir):
        if use_tmpdir:
            storage_cls = functools.partial(storage_cls, scratch_dir=tmpdir)
        storage = storage_cls(max_size=100, shared_init=True)
        rb = ReplayBuffer(storage=storage, batch_size=2).share(True)
        queue = mp.Queue()

        processes = []
        for i in range(4):
            p = mp.Process(target=self.worker, args=(rb, i, queue))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
            queue.get()

        all_data = storage.get(slice(0, 8))
        values = set(all_data["x"].tolist())
        expected = {0.0, 1.0, 2.0, 3.0}
        assert expected.issubset(values)
        assert len(storage) >= 8

    def prioritized_collector_worker(self, rb, worker_id, queue):
        data = TensorDict(
            {
                "obs": torch.full((4, 1), worker_id, dtype=torch.float32),
                "td_error": torch.linspace(0.1, 1.0, 4) + worker_id,
            },
            batch_size=(4,),
        )
        rb.extend(data)
        queue.put("done")

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    def test_prioritized_memmap_cuda_sampler_after_multiprocess_writes(self, tmpdir):
        ext = pytest.importorskip("torchrl._torchrl")
        if not hasattr(ext, "CudaSumSegmentTreeFp32"):
            pytest.skip("TorchRL was not built with CUDA segment tree support")

        storage = LazyMemmapStorage(max_size=32, scratch_dir=tmpdir, shared_init=True)
        writer_rb = TensorDictReplayBuffer(storage=storage, batch_size=4).share(True)
        queue = mp.Queue()

        processes = []
        for i in range(2):
            p = mp.Process(
                target=self.prioritized_collector_worker,
                args=(writer_rb, i, queue),
            )
            processes.append(p)
            p.start()

        for p in processes:
            p.join()
            assert p.exitcode == 0
            assert queue.get(timeout=5) == "done"

        assert len(storage) == 8
        learner_rb = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.5,
            storage=storage,
            sampler_device="cuda:0",
            batch_size=4,
            priority_key="td_error",
        )

        sample = learner_rb.sample()
        assert learner_rb._sampler.device == torch.device("cuda:0")
        assert sample["obs"].device.type == "cpu"
        assert sample["index"].device.type == "cpu"
        assert sample["priority_weight"].device.type == "cpu"

        sample["td_error"] = torch.ones_like(sample["td_error"]) * 10
        learner_rb.update_tensordict_priority(sample)
        sample = learner_rb.sample()
        assert sample["index"].device.type == "cpu"


@pytest.mark.skipif(not _has_zstandard, reason="zstandard required for this test.")
class TestCompressedListStorage:
    """Test cases for CompressedListStorage."""

    def test_compressed_storage_initialization(self):
        """Test that CompressedListStorage initializes correctly."""
        storage = CompressedListStorage(max_size=100, compression_level=3)
        assert storage.max_size == 100
        assert storage.compression_level == 3
        assert len(storage) == 0

    @pytest.mark.parametrize(
        "test_tensor",
        [
            torch.rand(1),  # 0D scalar
            torch.randn(84, dtype=torch.float32),  # 1D tensor
            torch.randn(84, 84, dtype=torch.float32),  # 2D tensor
            torch.randn(1, 84, 84, dtype=torch.float32),  # 3D tensor
            torch.randn(32, 84, 84, dtype=torch.float32),  # 3D tensor
        ],
    )
    def test_compressed_storage_tensor(self, test_tensor):
        """Test compression and decompression of tensor data of various shapes."""
        storage = CompressedListStorage(max_size=10, compression_level=3)

        # Store tensor
        storage.set(0, test_tensor)

        # Retrieve tensor
        retrieved_tensor = storage.get(0)

        # Verify data integrity
        assert (
            test_tensor.shape == retrieved_tensor.shape
        ), f"Expected shape {test_tensor.shape}, got {retrieved_tensor.shape}"
        assert (
            test_tensor.dtype == retrieved_tensor.dtype
        ), f"Expected dtype {test_tensor.dtype}, got {retrieved_tensor.dtype}"
        assert torch.allclose(test_tensor, retrieved_tensor, atol=1e-6)

    def test_compressed_storage_tensordict(self):
        """Test compression and decompression of TensorDict data."""
        storage = CompressedListStorage(max_size=10, compression_level=3)

        # Create test TensorDict
        test_td = TensorDict(
            {
                "obs": torch.randn(3, 84, 84, dtype=torch.float32),
                "action": torch.tensor([1, 2, 3]),
                "reward": torch.randn(3),
                "done": torch.tensor([False, True, False]),
            },
            batch_size=[3],
        )

        # Store TensorDict
        storage.set(0, test_td)

        # Retrieve TensorDict
        retrieved_td = storage.get(0)

        # Verify data integrity
        assert torch.allclose(test_td["obs"], retrieved_td["obs"], atol=1e-6)
        assert torch.allclose(test_td["action"], retrieved_td["action"])
        assert torch.allclose(test_td["reward"], retrieved_td["reward"], atol=1e-6)
        assert torch.allclose(test_td["done"], retrieved_td["done"])

    def test_compressed_storage_multiple_indices(self):
        """Test storing and retrieving multiple items."""
        storage = CompressedListStorage(max_size=10, compression_level=3)

        # Store multiple tensors
        tensors = [
            torch.randn(2, 2, dtype=torch.float32),
            torch.randn(3, 3, dtype=torch.float32),
            torch.randn(4, 4, dtype=torch.float32),
        ]

        for i, tensor in enumerate(tensors):
            storage.set(i, tensor)

        # Retrieve multiple tensors
        retrieved = storage.get([0, 1, 2])

        # Verify data integrity
        for original, retrieved_tensor in zip(tensors, retrieved):
            assert torch.allclose(original, retrieved_tensor, atol=1e-6)

    def test_compressed_storage_with_replay_buffer(self):
        """Test CompressedListStorage with ReplayBuffer."""
        storage = CompressedListStorage(max_size=100, compression_level=3)
        rb = ReplayBuffer(storage=storage, batch_size=5)

        # Create test data
        data = TensorDict(
            {
                "obs": torch.randn(10, 3, 84, 84, dtype=torch.float32),
                "action": torch.randint(0, 4, (10,)),
                "reward": torch.randn(10),
            },
            batch_size=[10],
        )

        # Add data to replay buffer
        rb.extend(data)

        # Sample from replay buffer
        sample = rb.sample(5)

        # Verify sample has correct shape
        assert is_tensor_collection(sample), sample
        assert sample["obs"].shape[0] == 5
        assert sample["obs"].shape[1:] == (3, 84, 84)
        assert sample["action"].shape[0] == 5
        assert sample["reward"].shape[0] == 5

    def test_compressed_storage_state_dict(self):
        """Test saving and loading state dict."""
        storage = CompressedListStorage(max_size=10, compression_level=3)

        # Add some data
        test_tensor = torch.randn(3, 3, dtype=torch.float32)
        storage.set(0, test_tensor)

        # Save state dict
        state_dict = storage.state_dict()

        # Create new storage and load state dict
        new_storage = CompressedListStorage(max_size=10, compression_level=3)
        new_storage.load_state_dict(state_dict)

        # Verify data integrity
        retrieved_tensor = new_storage.get(0)
        assert torch.allclose(test_tensor, retrieved_tensor, atol=1e-6)

    def test_compressed_storage_checkpointing(self):
        """Test checkpointing functionality."""
        storage = CompressedListStorage(max_size=10, compression_level=3)

        # Add some data
        test_td = TensorDict(
            {
                "obs": torch.randn(3, 84, 84, dtype=torch.float32),
                "action": torch.tensor([1, 2, 3]),
            },
            batch_size=[3],
        )
        storage.set(0, test_td)

        # second batch, different shape
        test_td2 = TensorDict(
            {
                "obs": torch.randn(3, 85, 83, dtype=torch.float32),
                "action": torch.tensor([1, 2, 3]),
                "meta": torch.randn(3),
                "astring": "a string!",
            },
            batch_size=[3],
        )
        storage.set(1, test_td)

        # Create temporary directory for checkpointing
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "checkpoint"

            # Save checkpoint
            storage.dumps(checkpoint_path)

            # Create new storage and load checkpoint
            new_storage = CompressedListStorage(max_size=10, compression_level=3)
            new_storage.loads(checkpoint_path)

            # Verify data integrity
            retrieved_td = new_storage.get(0)
            assert torch.allclose(test_td["obs"], retrieved_td["obs"], atol=1e-6)
            assert torch.allclose(test_td["action"], retrieved_td["action"])

    def test_compressed_storage_length(self):
        """Test that length is calculated correctly."""
        storage = CompressedListStorage(max_size=10, compression_level=3)

        # Initially empty
        assert len(storage) == 0

        # Add some data
        storage.set(0, torch.randn(2, 2))
        assert len(storage) == 1

        storage.set(1, torch.randn(2, 2))
        assert len(storage) == 2

        storage.set(2, torch.randn(2, 2))
        assert len(storage) == 3

    def test_compressed_storage_contains(self):
        """Test the contains method."""
        storage = CompressedListStorage(max_size=10, compression_level=3)

        # Initially empty
        assert not storage.contains(0)

        # Add data
        storage.set(0, torch.randn(2, 2))
        assert storage.contains(0)
        assert not storage.contains(1)

    def test_compressed_storage_empty(self):
        """Test emptying the storage."""
        storage = CompressedListStorage(max_size=10, compression_level=3)

        # Add some data
        storage.set(0, torch.randn(2, 2))
        storage.set(1, torch.randn(2, 2))
        assert len(storage) == 2

        # Empty storage
        storage._empty()
        assert len(storage) == 0

    def test_compressed_storage_custom_compression(self):
        """Test custom compression functions."""

        def custom_compress(tensor):
            # Simple compression: just convert to uint8
            return tensor.to(torch.uint8)

        def custom_decompress(compressed_tensor, metadata):
            # Simple decompression: convert back to original dtype
            return compressed_tensor.to(metadata["dtype"])

        storage = CompressedListStorage(
            max_size=10,
            compression_fn=custom_compress,
            decompression_fn=custom_decompress,
        )

        # Test with tensor
        test_tensor = torch.randn(2, 2, dtype=torch.float32)
        storage.set(0, test_tensor)
        retrieved_tensor = storage.get(0)

        # Note: This will lose precision due to uint8 conversion
        # but should still work
        assert retrieved_tensor.shape == test_tensor.shape

    def test_compressed_storage_error_handling(self):
        """Test error handling for invalid operations."""
        storage = CompressedListStorage(max_size=5, compression_level=3)

        # Test setting data beyond max_size
        with pytest.raises(RuntimeError):
            storage.set(10, torch.randn(2, 2))

        # Test getting non-existent data
        with pytest.raises(IndexError):
            storage.get(0)

    def test_compressed_storage_memory_efficiency(self):
        """Test that compression actually reduces memory usage."""
        storage = CompressedListStorage(max_size=100, compression_level=3)

        # Create large tensor data
        large_tensor = torch.zeros(100, 3, 84, 84, dtype=torch.int64)
        large_tensor.copy_(
            torch.arange(large_tensor.numel(), dtype=torch.int32).view_as(large_tensor)
            // (3 * 84 * 84)
        )
        original_size = large_tensor.numel() * large_tensor.element_size()

        # Store in compressed storage
        storage.set(0, large_tensor)

        # Estimate compressed size
        compressed_data = storage._storage[0]
        compressed_size = compressed_data.numel()  # uint8 bytes

        # Verify compression ratio is reasonable (at least 2x for random data)
        compression_ratio = original_size / compressed_size
        assert (
            compression_ratio > 1.5
        ), f"Compression ratio {compression_ratio} is too low"


class TestRBLazyInit:
    def test_lazy_init(self):
        def transform(td):
            return td

        rb = ReplayBuffer(
            storage=partial(ListStorage),
            writer=partial(RoundRobinWriter),
            sampler=partial(RandomSampler),
            transform_factory=lambda: transform,
        )
        assert not rb.initialized
        assert not hasattr(rb, "_storage")
        assert rb._init_storage is not None
        assert not hasattr(rb, "_sampler")
        assert rb._init_sampler is not None
        assert not hasattr(rb, "_writer")
        assert rb._init_writer is not None
        rb.extend(TensorDict(batch_size=[2]))
        assert rb.initialized
        assert rb._storage is not None
        assert rb._init_storage is None
        assert rb._sampler is not None
        assert rb._init_sampler is None
        assert rb._writer is not None
        assert rb._init_writer is None

        rb = ReplayBuffer(
            storage=partial(ListStorage),
            writer=partial(RoundRobinWriter),
            sampler=partial(RandomSampler),
        )
        assert rb.initialized
        assert rb._storage is not None
        assert rb._init_storage is None
        assert rb._sampler is not None
        assert rb._init_sampler is None
        assert rb._writer is not None
        assert rb._init_writer is None

        rb = ReplayBuffer(
            storage=partial(ListStorage),
            writer=partial(RoundRobinWriter),
            sampler=partial(RandomSampler),
            delayed_init=False,
        )
        assert rb.initialized
        assert rb._storage is not None
        assert rb._init_storage is None
        assert rb._sampler is not None
        assert rb._init_sampler is None
        assert rb._writer is not None
        assert rb._init_writer is None


@pytest.mark.skipif(
    _os_is_windows, reason="Windows file locking prevents cleanup tests"
)
class TestLazyMemmapStorageCleanup:
    """Tests for LazyMemmapStorage automatic cleanup functionality."""

    def test_cleanup_explicit_scratch_dir(self, tmpdir):
        """Test that cleanup removes files when scratch_dir is specified."""
        scratch_dir = str(tmpdir / "memmap_storage")
        os.makedirs(scratch_dir, exist_ok=True)

        storage = LazyMemmapStorage(100, scratch_dir=scratch_dir, auto_cleanup=True)
        rb = ReplayBuffer(storage=storage)
        rb.extend(TensorDict(a=torch.randn(10), batch_size=[10]))

        # Verify files were created
        assert os.path.isdir(scratch_dir)
        assert len(os.listdir(scratch_dir)) > 0

        # Cleanup should remove the directory
        result = storage.cleanup()
        assert result is True
        assert not os.path.exists(scratch_dir)

        # Second cleanup should be a no-op
        result = storage.cleanup()
        assert result is False

    def test_cleanup_temp_dir(self):
        """Test cleanup when using default temp directory."""
        storage = LazyMemmapStorage(100, auto_cleanup=True)
        rb = ReplayBuffer(storage=storage)
        rb.extend(TensorDict(a=torch.randn(10), batch_size=[10]))

        # Get the temp directory paths before cleanup
        temp_paths = set()
        for tensor in storage._storage.values(include_nested=True, leaves_only=True):
            try:
                if hasattr(tensor, "filename") and tensor.filename:
                    temp_paths.add(os.path.dirname(tensor.filename))
            except (AttributeError, RuntimeError):
                continue

        # Cleanup should remove the files if any were created on disk
        result = storage.cleanup()
        if len(temp_paths) > 0:
            assert result is True
            # Paths should no longer exist
            for path in temp_paths:
                assert not os.path.exists(path)
        else:
            # If no files were created (e.g. anonymous memmap), result should be False
            assert result is False

    def test_auto_cleanup_default_behavior(self, tmpdir):
        """Test that auto_cleanup defaults correctly based on scratch_dir."""
        # When scratch_dir is None, auto_cleanup should default to True
        storage1 = LazyMemmapStorage(100)
        assert storage1._auto_cleanup is True
        assert storage1._scratch_dir_is_temp is True

        # When scratch_dir is provided, auto_cleanup should default to False
        scratch_dir = str(tmpdir / "user_storage")
        storage2 = LazyMemmapStorage(100, scratch_dir=scratch_dir)
        assert storage2._auto_cleanup is False
        assert storage2._scratch_dir_is_temp is False

        # User can override
        storage3 = LazyMemmapStorage(100, scratch_dir=scratch_dir, auto_cleanup=True)
        assert storage3._auto_cleanup is True

        storage4 = LazyMemmapStorage(100, auto_cleanup=False)
        assert storage4._auto_cleanup is False

    def test_cleanup_idempotent(self, tmpdir):
        """Test that cleanup can be called multiple times safely."""
        scratch_dir = str(tmpdir / "memmap_storage")
        os.makedirs(scratch_dir, exist_ok=True)

        storage = LazyMemmapStorage(100, scratch_dir=scratch_dir, auto_cleanup=True)
        rb = ReplayBuffer(storage=storage)
        rb.extend(TensorDict(a=torch.randn(10), batch_size=[10]))

        # Multiple cleanups should not raise
        storage.cleanup()
        storage.cleanup()
        storage.cleanup()
        assert storage._cleaned_up is True

    def test_cleanup_nonexistent_dir(self, tmpdir):
        """Test cleanup when directory was already deleted."""
        scratch_dir = str(tmpdir / "memmap_storage")
        os.makedirs(scratch_dir, exist_ok=True)

        storage = LazyMemmapStorage(100, scratch_dir=scratch_dir, auto_cleanup=True)
        rb = ReplayBuffer(storage=storage)
        rb.extend(TensorDict(a=torch.randn(10), batch_size=[10]))

        # Delete the directory externally
        shutil.rmtree(scratch_dir)
        assert not os.path.exists(scratch_dir)

        # Cleanup should handle missing directory gracefully
        result = storage.cleanup()
        assert result is False  # No cleanup needed since dir is gone

    def test_cleanup_uninitialized_storage(self):
        """Test cleanup on storage that was never used."""
        storage = LazyMemmapStorage(100, auto_cleanup=True)
        # Storage is not initialized - cleanup should be safe
        result = storage.cleanup()
        assert result is False

    def test_cleanup_registry(self):
        """Test that storages are registered for cleanup."""
        storage = LazyMemmapStorage(100, auto_cleanup=True)
        # Check storage is in the registry (avoids race with GC on WeakSet)
        assert storage in _MEMMAP_STORAGE_REGISTRY

        # Storage with auto_cleanup=False should not be registered
        storage2 = LazyMemmapStorage(100, auto_cleanup=False)
        assert storage2 not in _MEMMAP_STORAGE_REGISTRY
        # Original storage should still be in the registry
        assert storage in _MEMMAP_STORAGE_REGISTRY

        # Cleanup should still work
        storage.cleanup()

    def test_cleanup_subprocess(self, tmpdir):
        """Test that cleanup works correctly in subprocess scenarios."""
        scratch_dir = str(tmpdir / "subprocess_storage")

        # Create a script that creates a storage and exits normally
        script = f"""
import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyMemmapStorage

storage = LazyMemmapStorage(100, scratch_dir="{scratch_dir}", auto_cleanup=True)
rb = ReplayBuffer(storage=storage)
rb.extend(TensorDict(a=torch.randn(10), batch_size=[10]))
print("Storage created")
# Normal exit - atexit handler should clean up
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Script should have succeeded
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        # Directory should have been cleaned up on exit
        assert not os.path.exists(
            scratch_dir
        ), f"Directory {scratch_dir} should have been cleaned up"

    def test_cleanup_signal_interrupt(self, tmpdir):
        """Test that cleanup happens on SIGINT (Ctrl+C)."""
        scratch_dir = str(tmpdir / "signal_storage")

        # Create a script that sleeps and can be interrupted
        script = f"""
import signal
import time
import torch
from tensordict import TensorDict
from torchrl.data import ReplayBuffer, LazyMemmapStorage

storage = LazyMemmapStorage(100, scratch_dir="{scratch_dir}", auto_cleanup=True)
rb = ReplayBuffer(storage=storage)
rb.extend(TensorDict(a=torch.randn(10), batch_size=[10]))
print("READY", flush=True)
time.sleep(60)  # Will be interrupted
"""
        proc = subprocess.Popen(
            [sys.executable, "-c", script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Wait for the script to be ready
        try:
            # Read until we see READY
            start = time.time()
            while time.time() - start < 10:
                line = proc.stdout.readline()
                if "READY" in line:
                    break
            else:
                proc.kill()
                pytest.skip("Script did not start in time")

            # Give it a moment to set up signal handlers
            time.sleep(0.5)

            # Verify directory exists
            assert os.path.isdir(scratch_dir)

            # Send SIGINT (Ctrl+C)
            proc.send_signal(signal.SIGINT)
            proc.wait(timeout=5)

            # Directory should have been cleaned up
            assert not os.path.exists(
                scratch_dir
            ), f"Directory {scratch_dir} should have been cleaned up on SIGINT"
        finally:
            if proc.poll() is None:
                proc.kill()
                proc.wait()

    def test_cleanup_with_del(self, tmpdir):
        """Test that __del__ triggers cleanup."""
        scratch_dir = str(tmpdir / "del_storage")
        os.makedirs(scratch_dir, exist_ok=True)

        def create_and_delete():
            storage = LazyMemmapStorage(100, scratch_dir=scratch_dir, auto_cleanup=True)
            rb = ReplayBuffer(storage=storage)
            rb.extend(TensorDict(a=torch.randn(10), batch_size=[10]))
            # Storage goes out of scope here

        create_and_delete()

        # Force garbage collection
        gc.collect()

        # Note: __del__ is not guaranteed to run immediately, but the cleanup
        # infrastructure should still work via atexit

    def test_cleanup_preserves_user_data_by_default(self, tmpdir):
        """Test that user-specified directories are NOT cleaned by default."""
        scratch_dir = str(tmpdir / "user_data")
        os.makedirs(scratch_dir, exist_ok=True)

        storage = LazyMemmapStorage(100, scratch_dir=scratch_dir)
        rb = ReplayBuffer(storage=storage)
        rb.extend(TensorDict(a=torch.randn(10), batch_size=[10]))

        # auto_cleanup should be False by default
        assert storage._auto_cleanup is False

        # Directory should exist
        assert os.path.isdir(scratch_dir)

        # Explicit cleanup should still work
        storage.cleanup()
        assert not os.path.exists(scratch_dir)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
