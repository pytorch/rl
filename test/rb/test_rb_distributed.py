# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import os

import sys
import time
from functools import partial

import pytest
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from _rb_common import _has_ray
from tensordict import TensorDict
from torchrl._utils import logger as torchrl_logger
from torchrl.data import RayReplayBuffer, ReplayBuffer
from torchrl.data.replay_buffers import RemoteTensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import (
    RandomSampler,
    SamplerWithoutReplacement,
)
from torchrl.data.replay_buffers.storages import LazyMemmapStorage, LazyTensorStorage
from torchrl.data.replay_buffers.writers import RoundRobinWriter
from torchrl.objectives.llm import MCAdvantage

RETRY_COUNT = 3
RETRY_BACKOFF = 3


class ReplayBufferNode(RemoteTensorDictReplayBuffer):
    def __init__(self, capacity: int, scratch_dir=None):
        super().__init__(
            storage=LazyMemmapStorage(
                max_size=capacity, scratch_dir=scratch_dir, device=torch.device("cpu")
            ),
            sampler=RandomSampler(),
            writer=RoundRobinWriter(),
            collate_fn=lambda x: x,
        )


def construct_buffer_test(rank, name, world_size):
    if name == "TRAINER":
        buffer = _construct_buffer("BUFFER")
        assert type(buffer) is torch._C._distributed_rpc.PyRRef


def add_to_buffer_remotely_test(rank, name, world_size):
    if name == "TRAINER":
        buffer = _construct_buffer("BUFFER")
        res, _ = _add_random_tensor_dict_to_buffer(buffer)
        assert type(res) is int
        assert res == 0


def sample_from_buffer_remotely_returns_correct_tensordict_test(rank, name, world_size):
    if name == "TRAINER":
        buffer = _construct_buffer("BUFFER")
        _, inserted = _add_random_tensor_dict_to_buffer(buffer)
        sampled = _sample_from_buffer(buffer, 1)
        assert type(sampled) is type(inserted) is TensorDict
        a_sample = sampled["a"]
        a_insert = inserted["a"]
        assert (a_sample == a_insert).all()


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="Distributed package support on Windows is a prototype feature and is subject to changes.",
)
@pytest.mark.parametrize("names", [["BUFFER", "TRAINER"]])
@pytest.mark.parametrize(
    "func",
    [
        construct_buffer_test,
        add_to_buffer_remotely_test,
        sample_from_buffer_remotely_returns_correct_tensordict_test,
    ],
)
def test_funcs(names, func):
    world_size = len(names)
    with mp.Pool(world_size) as pool:
        pool.starmap(
            init_rpc, ((rank, name, world_size) for rank, name in enumerate(names))
        )
        pool.starmap(
            func, ((rank, name, world_size) for rank, name in enumerate(names))
        )
        pool.apply_async(shutdown)


def init_rpc(rank, name, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    str_init_method = "tcp://localhost:10030"
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16, init_method=str_init_method
    )
    rpc.init_rpc(
        name,
        rank=rank,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=options,
        world_size=world_size,
    )


def shutdown():
    rpc.shutdown()


def _construct_buffer(target):
    for _ in range(RETRY_COUNT):
        try:
            buffer_rref = rpc.remote(target, ReplayBufferNode, args=(1000,))
            return buffer_rref
        except Exception as e:
            torchrl_logger.info(f"Failed to connect: {e}")
            time.sleep(RETRY_BACKOFF)
    raise RuntimeError("Unable to connect to replay buffer")


def _add_random_tensor_dict_to_buffer(buffer):
    rand_td = TensorDict({"a": torch.randint(100, (1,))}, [])
    return (
        rpc.rpc_sync(
            buffer.owner(),
            ReplayBufferNode.add,
            args=(
                buffer,
                rand_td,
            ),
        ),
        rand_td,
    )


def _sample_from_buffer(buffer, batch_size):
    return rpc.rpc_sync(
        buffer.owner(), ReplayBufferNode.sample, args=(buffer, batch_size)
    )


def _make_mcadvantage_traj(group_id, rewards):
    n_steps = len(rewards)
    return TensorDict(
        {
            "group_id": torch.full((n_steps,), group_id),
            ("next", "reward"): torch.tensor(rewards).reshape(n_steps, 1),
            ("next", "done"): torch.tensor([False] * (n_steps - 1) + [True]).reshape(
                n_steps, 1
            ),
        },
        batch_size=[n_steps],
    )


@pytest.mark.skipif(not _has_ray, reason="ray required for this test.")
class TestRayRB:
    @pytest.fixture(autouse=True, scope="module")
    def cleanup(self):
        import ray

        ray.shutdown()
        torchrl_logger.info("Initializing Ray.")
        ray.init(num_cpus=1)
        yield
        torchrl_logger.info("Shutting down Ray.")
        ray.shutdown()

    def test_ray_rb(self):
        rb = RayReplayBuffer(
            storage=partial(LazyTensorStorage, 100), ray_init_config={"num_cpus": 1}
        )
        try:
            rb.extend(
                TensorDict(
                    {"x": torch.ones(100, 2), "y": torch.ones(100, 2)}, batch_size=100
                )
            )
            assert rb.write_count == 100
            assert len(rb) == 100
            assert rb.sample(2).shape == (2,)
        finally:
            rb.close()

    def test_ray_rb_iter(self):
        rb = RayReplayBuffer(
            storage=partial(LazyTensorStorage, 100),
            ray_init_config={"num_cpus": 1},
            sampler=SamplerWithoutReplacement,
            batch_size=25,
        )
        try:
            rb.extend(
                TensorDict(
                    {
                        "x": torch.ones(
                            100,
                        ),
                        "y": torch.ones(
                            100,
                        ),
                    },
                    batch_size=100,
                )
            )
            for _ in range(2):
                for d in rb:
                    torchrl_logger.info(f"d: {d}")
                    assert d is not None
                    assert d.shape == (25,)
        finally:
            rb.close()

    def test_ray_rb_serialization(self):
        import ray

        class Worker:
            def __init__(self, rb):
                self.rb = rb

            def run(self):
                self.rb.extend(TensorDict({"x": torch.ones(100)}, batch_size=100))

        rb = RayReplayBuffer(
            storage=partial(LazyTensorStorage, 100), ray_init_config={"num_cpus": 1}
        )
        try:
            client = rb.client()
            assert not hasattr(client, "shutdown")
            assert not hasattr(client, "close")
            remote_worker = ray.remote(Worker).remote(client)
            ray.get(remote_worker.run.remote())
            assert len(rb) == 100
        finally:
            rb.close()

    def test_construct_from_replay_buffer_service_backend(self):
        import ray

        rb = ReplayBuffer(
            storage=partial(LazyTensorStorage, 100),
            service_backend="ray",
            service_backend_options={
                "ray_init_config": {"num_cpus": 1},
                "remote_config": {"num_cpus": 0},
            },
        )
        try:
            assert isinstance(rb, RayReplayBuffer)
            assert isinstance(rb, ReplayBuffer)
            assert rb.service_backend == "ray"
            assert rb.transport_kind == "ray"
            clients = rb.clients(2)
            assert clients[0] is not clients[1]
            rb.extend(TensorDict({"x": torch.ones(4)}, batch_size=4))
            assert len(rb.client()) == 4
        finally:
            rb.shutdown()
        assert not rb.is_alive
        assert ray.is_initialized()
        rb.shutdown()

    def test_ray_replay_with_gloo_transport(self):
        rb = ReplayBuffer(
            storage=partial(LazyTensorStorage, 100),
            batch_size=4,
            service_backend="ray",
            service_backend_options={"remote_config": {"num_cpus": 0}},
            transport="distributed",
            transport_options={"backend": "gloo", "timeout": 30.0},
        )
        try:
            client = rb.client()
            indices = client.extend(
                TensorDict({"x": torch.arange(8)}, batch_size=[8]), timeout=30.0
            )
            assert indices.tolist() == list(range(8))
            assert len(client) == 8
            assert client.write_count == 8
            assert client.sample(timeout=30.0).shape == (4,)
            with pytest.raises((RuntimeError, ValueError)):
                client.extend(
                    TensorDict({"x": torch.arange(4)}, batch_size=[4]),
                    timeout=30.0,
                )
            assert not hasattr(client, "shutdown")
        finally:
            rb.shutdown()

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
    def test_ray_replay_with_nccl_transport(self):
        rb = ReplayBuffer(
            storage=partial(LazyTensorStorage, 100, device="cuda"),
            batch_size=4,
            service_backend="ray",
            service_backend_options={"remote_config": {"num_gpus": 1}},
            transport="distributed",
            transport_options={"backend": "nccl", "timeout": 30.0},
        )
        try:
            client = rb.client()
            client.extend(
                TensorDict(
                    {"x": torch.arange(8, device="cuda")},
                    batch_size=[8],
                    device="cuda",
                ),
                timeout=30.0,
            )
            sample = client.sample(timeout=30.0)
            assert sample.shape == (4,)
            assert sample.device.type == "cuda"
        finally:
            rb.shutdown()

    def test_ray_rb_mcadvantage_transform_factory(self):
        rb = RayReplayBuffer(
            storage=partial(LazyTensorStorage, 10),
            transform_factory=partial(
                MCAdvantage,
                grpo_size=2,
                prompt_key="group_id",
                trajectory_return="sum",
            ),
            ray_init_config={"num_cpus": 1},
            remote_config={"num_cpus": 0},
            batch_size=2,
        )
        try:
            rb.extend(_make_mcadvantage_traj(0, [0.0]))
            assert len(rb) == 0
            rb.extend(_make_mcadvantage_traj(0, [1.0]))
            assert len(rb) == 2
            sample = rb.sample()
            assert sample.shape == (2,)
            assert "advantage" in sample.keys()
        finally:
            rb.close()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
