# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import os

import sys
import time

import pytest
import torch
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
from tensordict import TensorDict
from torchrl._utils import logger as torchrl_logger
from torchrl.data.replay_buffers import RemoteTensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.data.replay_buffers.writers import RoundRobinWriter

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


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
