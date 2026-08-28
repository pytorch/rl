# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import contextlib
import functools

import numpy as np
import pytest
import torch
from _rb_common import _has_gym, _has_tv
from tensordict import (
    assert_allclose_td,
    is_tensor_collection,
    tensorclass,
    TensorDict,
    TensorDictBase,
)
from torch.utils._pytree import tree_flatten

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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
