# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Sample latency benchmarking (using RPC)
======================================
A rough benchmark of sample latency using different storage types over the network using `torch.rpc`.
Run this script with --rank=0 and --rank=1 flags set in separate processes - these ranks correspond to the trainer worker and buffer worker respectively, and both need to be initialised.
e.g. to benchmark LazyMemmapStorage, run the following commands using either two separate shells or multiprocessing.
    - python3 benchmark_sample_latency_over_rpc.py --rank=0 --storage=LazyMemmapStorage
    - python3 benchmark_sample_latency_over_rpc.py --rank=1 --storage=LazyMemmapStorage
This code is based on examples/distributed/distributed_replay_buffer.py.
"""
import argparse
import os
import pickle
import sys
import time
import timeit
from datetime import datetime

import torch
import torch.distributed.rpc as rpc
from tensordict import TensorDict
from torchrl._utils import logger as torchrl_logger
from torchrl.data.replay_buffers import RemoteTensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ListStorage,
)
from torchrl.data.replay_buffers.writers import RoundRobinWriter

RETRY_LIMIT = 2
RETRY_DELAY_SECS = 3
REPLAY_BUFFER_NODE = "ReplayBuffer"
TRAINER_NODE = "Trainer"
TENSOR_SIZE = 3 * 86 * 86
BUFFER_SIZE = 1001
BATCH_SIZE = 256
REPEATS = 1000

storage_options = {
    "LazyMemmapStorage": LazyMemmapStorage,
    "LazyTensorStorage": LazyTensorStorage,
    "ListStorage": ListStorage,
}

storage_arg_options = {
    "LazyMemmapStorage": {"scratch_dir": "/tmp/", "device": torch.device("cpu")},
    "LazyTensorStorage": {},
    "ListStorage": {},
}
parser = argparse.ArgumentParser(
    description="RPC Replay Buffer Example",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter,
)

parser.add_argument(
    "--rank",
    type=int,
    default=-1,
    help="Node Rank [0 = Replay Buffer, 1 = Dummy Trainer, 2+ = Dummy Data Collector]",
)

parser.add_argument(
    "--storage",
    type=str,
    default="LazyMemmapStorage",
    help="Storage type [LazyMemmapStorage, LazyTensorStorage, ListStorage]",
)


class DummyTrainerNode:
    def __init__(self) -> None:
        self.id = rpc.get_worker_info().id
        self.replay_buffer = self._create_replay_buffer()
        self._ret = None

    def train(self, batch_size: int) -> None:
        start_time = timeit.default_timer()
        ret = rpc.rpc_sync(
            self.replay_buffer.owner(),
            ReplayBufferNode.sample,
            args=(self.replay_buffer, batch_size),
        )
        if storage_type == "ListStorage":
            self._ret = ret[0]
        else:
            if self._ret is None:
                self._ret = ret
            else:
                self._ret.update_(ret)
        # make sure the content is read
        self._ret["observation"] + 1
        self._ret["next_observation"] + 1
        return timeit.default_timer() - start_time

    def _create_replay_buffer(self) -> rpc.RRef:
        while True:
            try:
                replay_buffer_info = rpc.get_worker_info(REPLAY_BUFFER_NODE)
                buffer_rref = rpc.remote(
                    replay_buffer_info, ReplayBufferNode, args=(1000000,)
                )
                torchrl_logger.info(f"Connected to replay buffer {replay_buffer_info}")
                return buffer_rref
            except Exception:
                torchrl_logger.info("Failed to connect to replay buffer")
                time.sleep(RETRY_DELAY_SECS)


class ReplayBufferNode(RemoteTensorDictReplayBuffer):
    def __init__(self, capacity: int):
        super().__init__(
            storage=storage_options[storage_type](
                max_size=capacity, **storage_arg_options[storage_type]
            ),
            sampler=RandomSampler(),
            writer=RoundRobinWriter(),
            collate_fn=lambda x: x,
        )
        tds = TensorDict(
            {
                "observation": torch.randn(
                    BUFFER_SIZE,
                    TENSOR_SIZE,
                ),
                "next_observation": torch.randn(
                    BUFFER_SIZE,
                    TENSOR_SIZE,
                ),
            },
            batch_size=[BUFFER_SIZE],
        )
        self.extend(tds)


if __name__ == "__main__":
    args = parser.parse_args()
    rank = args.rank
    storage_type = args.storage

    torchrl_logger.info(f"Rank: {rank}; Storage: {storage_type}")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16, init_method="tcp://localhost:10002", rpc_timeout=120
    )
    if rank == 0:
        # rank 0 is the trainer
        rpc.init_rpc(
            TRAINER_NODE,
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )
        trainer = DummyTrainerNode()
        results = []
        for i in range(REPEATS):
            result = trainer.train(batch_size=BATCH_SIZE)
            if i == 0:
                continue
            results.append(result)
            torchrl_logger.info(f"{i}, {results[-1]}")

        with open(
            f'./benchmark_{datetime.now().strftime("%d-%m-%Y%H:%M:%S")};batch_size={BATCH_SIZE};tensor_size={TENSOR_SIZE};repeat={REPEATS};storage={storage_type}.pkl',
            "wb+",
        ) as f:
            pickle.dump(results, f)

        tensor_results = torch.tensor(results)
        torchrl_logger.info(f"Mean: {torch.mean(tensor_results)}")
        breakpoint()
    elif rank == 1:
        # rank 1 is the replay buffer
        # replay buffer waits passively for construction instructions from trainer node
        rpc.init_rpc(
            REPLAY_BUFFER_NODE,
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )
        breakpoint()
    else:
        sys.exit(1)
    rpc.shutdown()
