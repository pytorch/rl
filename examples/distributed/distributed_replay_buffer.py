import argparse
import os
import random
import sys
import time
from datetime import datetime
from functools import wraps

import torch
import torch.distributed.rpc as rpc
from torchrl.data.replay_buffers.rb_prototype import TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import RandomSampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.data.replay_buffers.writers import RoundRobinWriter
from torchrl.data.tensordict import TensorDict

RETRY_LIMIT = 2
RETRY_DELAY_SECS = 3
REPLAY_BUFFER_NODE = "ReplayBuffer"
TRAINER_NODE = "Trainer"

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


def accept_remote_rref_invocation(func):
    @wraps(func)
    def unpack_rref_and_invoke_function(self, *args, **kwargs):
        if isinstance(self, torch._C._distributed_rpc.PyRRef):
            self = self.local_value()
        return func(self, *args, **kwargs)

    return unpack_rref_and_invoke_function


class DummyDataCollectorNode:
    def __init__(self, replay_buffer) -> None:
        self.id = rpc.get_worker_info().id
        self.replay_buffer = replay_buffer
        print("Data Collector Node constructed")

    def __submit_random_item_async(self) -> rpc.RRef:
        td = TensorDict({"a": torch.randint(100, (1,))}, [])
        return rpc.remote(
            self.replay_buffer.owner(),
            ReplayBufferNode.add,
            args=(
                self.replay_buffer,
                td,
            ),
        )

    @accept_remote_rref_invocation
    def collect(self):
        for elem in range(50):
            time.sleep(random.randint(1, 4))
            print(
                f"[{self.id}] Collector submission {elem}: {self.__submit_random_item_async().to_here()}"
            )


class DummyTrainerNode:
    def __init__(self) -> None:
        print("DummyTrainerNode")
        self.id = rpc.get_worker_info().id
        self.replay_buffer = self._create_replay_buffer()
        self._create_and_launch_data_collectors()

    def train(self, iterations: int) -> None:
        for iteration in range(iterations):
            print(f"[{self.id}] Training Iteration: {iteration}")
            time.sleep(3)
            batch = rpc.rpc_sync(
                self.replay_buffer.owner(),
                ReplayBufferNode.sample,
                args=(self.replay_buffer, 16),
            )
            print(f"[{self.id}] Sample Obtained Iteration: {iteration}")
            print(f"{batch}")

    def _create_replay_buffer(self) -> rpc.RRef:
        while True:
            try:
                replay_buffer_info = rpc.get_worker_info(REPLAY_BUFFER_NODE)
                buffer_rref = rpc.remote(
                    replay_buffer_info, ReplayBufferNode, args=(10000,)
                )
                print(f"Connected to replay buffer {replay_buffer_info}")
                return buffer_rref
            except Exception:
                print("Failed to connect to replay buffer")
                time.sleep(RETRY_DELAY_SECS)

    def _create_and_launch_data_collectors(self) -> None:
        data_collector_number = 2
        retries = 0
        data_collectors = []
        data_collector_infos = []
        # discover launched data collector nodes (with retry to allow collectors to dynamically join)
        while True:
            try:
                data_collector_info = rpc.get_worker_info(
                    f"DataCollector{data_collector_number}"
                )
                print(f"Data collector info: {data_collector_info}")
                dc_ref = rpc.remote(
                    data_collector_info,
                    DummyDataCollectorNode,
                    args=(self.replay_buffer,),
                )
                data_collectors.append(dc_ref)
                data_collector_infos.append(data_collector_info)
                data_collector_number += 1
                retries = 0
            except Exception:
                retries += 1
                print(
                    f"Failed to connect to DataCollector{data_collector_number} with {retries} retries"
                )
                if retries >= RETRY_LIMIT:
                    print(f"{len(data_collectors)} data collectors")
                    for data_collector_info, data_collector in zip(
                        data_collector_infos, data_collectors
                    ):
                        rpc.remote(
                            data_collector_info,
                            DummyDataCollectorNode.collect,
                            args=(data_collector,),
                        )
                    break
                else:
                    time.sleep(RETRY_DELAY_SECS)


class ReplayBufferNode(TensorDictReplayBuffer):
    def __init__(self, capacity: int) -> None:
        super().__init__(
            storage=LazyMemmapStorage(
                max_size=capacity, scratch_dir="/tmp/", device=torch.device("cpu")
            ),
            sampler=RandomSampler(),
            writer=RoundRobinWriter(),
            collate_fn=lambda x: x,
        )
        self.id = rpc.get_worker_info().id
        print("ReplayBufferNode constructed")

    @accept_remote_rref_invocation
    def sample(self, batch_size: int) -> TensorDict:
        if len(self) <= batch_size:
            print(
                f'[{self.id}] Empty Buffer Sampling at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
            )
            return None
        else:
            print(
                f'[{self.id}] Replay Buffer Sampling at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")}'
            )
            batch = super().sample(batch_size)
            return batch

    @accept_remote_rref_invocation
    def add(self, data: TensorDict) -> None:
        res = super().add(data)
        print(
            f'[{self.id}] Replay Buffer Insertion at {datetime.now().strftime("%d/%m/%Y %H:%M:%S")} with {len(self)} elements'
        )
        return res


if __name__ == "__main__":
    args = parser.parse_args()
    rank = args.rank
    print(f"Rank: {rank}")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    str_init_method = "tcp://localhost:10000"
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16, init_method=str_init_method
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
        trainer.train(100)
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
    elif rank >= 2:
        # rank 2+ is a new data collector node
        # data collectors also wait passively for construction instructions from trainer node
        rpc.init_rpc(
            f"DataCollector{rank}",
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )
        print(f"Initialised DC Node {rank}")
        breakpoint()
    else:
        sys.exit(1)
    rpc.shutdown()
