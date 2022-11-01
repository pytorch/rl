import argparse
import os
import pickle
import sys
import time
import timeit
from datetime import datetime
from functools import wraps
from typing import Union, List

import torch
import torch.distributed.rpc as rpc
from torchrl.data.replay_buffers.rb_prototype import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import (
    LazyMemmapStorage,
    LazyTensorStorage,
    ListStorage,
)
from torchrl.data.replay_buffers.writers import RoundRobinWriter
from torchrl.data.tensordict import TensorDict

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
    "LazyMemmapStorage": dict(scratch_dir="/tmp/", device=torch.device("cpu")),
    "LazyTensorStorage": dict(),
    "ListStorage": dict(),
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
    help="Storage type [LazyMemmapStorage, LazyTensorStorage]",
)


def accept_remote_rref_invocation(func):
    @wraps(func)
    def unpack_rref_and_invoke_function(self, *args, **kwargs):
        if isinstance(self, torch._C._distributed_rpc.PyRRef):
            self = self.local_value()
        return func(self, *args, **kwargs)

    return unpack_rref_and_invoke_function


class DummyTrainerNode:
    def __init__(self) -> None:
        print("DummyTrainerNode")
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
        if self._ret is None:
            self._ret = ret
        else:
            self._ret[0].update_(ret[0])
        # make sure the content is read
        self._ret[0]["observation"] + 1
        self._ret[0]["next_observation"] + 1
        dt = timeit.default_timer() - start_time
        return dt

    def _create_replay_buffer(self) -> rpc.RRef:
        while True:
            try:
                replay_buffer_info = rpc.get_worker_info(REPLAY_BUFFER_NODE)
                buffer_rref = rpc.remote(
                    replay_buffer_info, ReplayBufferNode, args=(1000000,)
                )
                print(f"Connected to replay buffer {replay_buffer_info}")
                return buffer_rref
            except Exception:
                print("Failed to connect to replay buffer")
                time.sleep(RETRY_DELAY_SECS)


class ReplayBufferNode(TensorDictReplayBuffer):
    def __init__(self, capacity: int) -> None:
        super().__init__(
            storage=storage_options[storage_type](
                max_size=capacity, **storage_arg_options[storage_type]
            ),
            sampler=RandomSampler(),
            writer=RoundRobinWriter(),
            collate_fn=lambda x: x,
        )

        self.id = rpc.get_worker_info().id
        print("ReplayBufferNode constructed")
        tds = TensorDict(
            {
                "observation": torch.randn(BUFFER_SIZE, TENSOR_SIZE, ),
                "next_observation": torch.randn(BUFFER_SIZE, TENSOR_SIZE, ),
            },
            batch_size=[BUFFER_SIZE],
        )
        print("Built random contents")
        self.extend(tds)
        print("Extended tensor dict")

    @accept_remote_rref_invocation
    def sample(self, batch_size: int) -> TensorDict:
        return super().sample(batch_size)

    @accept_remote_rref_invocation
    def extend(self, tensordicts: Union[List, TensorDict]) -> torch.Tensor:
        return super().extend(tensordicts)


if __name__ == "__main__":
    args = parser.parse_args()
    rank = args.rank
    storage_type = args.storage

    print(f"Rank: {rank}; Storage: {storage_type}")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16, init_method="tcp://localhost:10000", rpc_timeout=120
    )
    if rank == 0:
        # rank 0 is the trainer
        rpc.init_rpc(
            TRAINER_NODE,
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )
        print(f"Initialised Trainer Node {rank}")
        trainer = DummyTrainerNode()
        results = []
        for i in range(REPEATS):
            result = trainer.train(batch_size=BATCH_SIZE)
            if i == 0:
                continue
            results.append(result)
            print(i, results[-1])

        print(f"Results: {results}")
        with open(
            f'./benchmark_{datetime.now().strftime("%d-%m-%Y%H:%M:%S")};batch_size={BATCH_SIZE};tensor_size={TENSOR_SIZE};repeat={REPEATS};storage={storage_type}.pkl',
            "wb+",
        ) as f:
            pickle.dump(results, f)

        tensor_results = torch.tensor(results)
        print(f"Mean: {torch.mean(tensor_results)}")
        breakpoint()
    elif rank == 1:
        # rank 1 is the replay buffer
        # replay buffer waits passively for construction instructions from trainer node
        print(REPLAY_BUFFER_NODE)
        rpc.init_rpc(
            REPLAY_BUFFER_NODE,
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )
        print(f"Initialised RB Node {rank}")
        breakpoint()
    else:
        sys.exit(1)
    rpc.shutdown()
