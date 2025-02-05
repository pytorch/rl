# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import os
import random
import time

import torch
import torch.distributed.rpc as rpc
import tqdm
from tensordict import TensorDict

from torchrl._utils import accept_remote_rref_invocation, logger as torchrl_logger
from torchrl.data.replay_buffers import RemoteReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.data.replay_buffers.writers import RoundRobinWriter

RETRY_LIMIT = 2
RETRY_DELAY_SECS = 3

REPLAY_BUFFER_NODE = "ReplayBuffer"
TRAINER_NODE = "Trainer"


class CollectorNode:
    """Data collector node responsible for collecting experiences used for learning.

    Args:
        replay_buffer (rpc.RRef): the RRef associated with the construction of the replay buffer
        frames_per_batch (int): the ``frames_per_batch`` of the collector. This serves as an example of hyperparameters
            to be passed to the collector.

    """

    def __init__(self, replay_buffer: rpc.RRef, frames_per_batch: int = 128) -> None:
        self.id = rpc.get_worker_info().id
        self.replay_buffer = replay_buffer
        # Write your collector here
        #  self.collector = SyncDataCollector(...)
        assert frames_per_batch > 0
        self.frames_per_batch = frames_per_batch
        torchrl_logger.info("Data Collector Node constructed")

    def _submit_item_async(self) -> rpc.RRef:
        """Function that collects data and populates the replay buffer."""
        # Replace this by a call to next() over the data collector
        done = torch.zeros(self.frames_per_batch, 1, dtype=torch.bool)
        done[..., -1, 0] = True
        td = TensorDict(
            {
                "action": torch.randint(
                    100,
                    (
                        self.frames_per_batch,
                        1,
                    ),
                ),
                "done": torch.zeros(self.frames_per_batch, dtype=torch.bool),
                "observation": torch.randn(self.frames_per_batch, 4),
                "step_count": torch.arange(self.frames_per_batch),
                "terminated": torch.zeros(self.frames_per_batch, dtype=torch.bool),
                "truncated": torch.zeros(self.frames_per_batch, dtype=torch.bool),
                "next": {
                    "done": done,
                    "observation": torch.randn(self.frames_per_batch, 4),
                    "reward": torch.randn(self.frames_per_batch, 1),
                    "step_count": torch.arange(1, self.frames_per_batch + 1),
                    "terminated": torch.zeros_like(done),
                    "truncated": done,
                },
            },
            [self.frames_per_batch],
        )
        return rpc.remote(
            self.replay_buffer.owner(),
            ReplayBufferNode.extend,
            args=(
                self.replay_buffer,
                td,
            ),
        )

    @accept_remote_rref_invocation
    def collect(self):
        """Method that begins experience collection (we just generate random TensorDicts in this example).

        `accept_remote_rref_invocation` enables this method to be invoked remotely provided the class instantiation
        `rpc.RRef` is provided in place of the object reference.
        """
        for elem in range(50):
            time.sleep(random.randint(1, 4))
            item = self._submit_item_async()
            torchrl_logger.info(
                f"Collector [{self.id}] submission {elem}: {item.to_here()}"
            )


class TrainerNode:
    """Trainer node responsible for learning from experiences sampled from an experience replay buffer."""

    def __init__(self, replay_buffer_node="ReplayBuffer", world_size=3) -> None:
        self.replay_buffer_node = replay_buffer_node
        self.world_size = world_size
        torchrl_logger.info("TrainerNode")
        self.id = rpc.get_worker_info().id
        self.replay_buffer = self._create_replay_buffer()
        self._create_and_launch_data_collectors()

    def train(self, iterations: int) -> None:
        """Write your training loop here."""
        for iteration in tqdm.tqdm(range(iterations)):
            torchrl_logger.info(f"[{self.id}] Training Iteration: {iteration}")
            # # Wait until the buffer has elements
            while not rpc.rpc_sync(
                self.replay_buffer.owner(),
                ReplayBufferNode.__len__,
                args=(self.replay_buffer,),
            ):
                continue

            batch = rpc.rpc_sync(
                self.replay_buffer.owner(),
                ReplayBufferNode.sample,
                args=(self.replay_buffer, 16),
            )

            torchrl_logger.info(f"[{self.id}] Sample Obtained Iteration: {iteration}")
            torchrl_logger.info(f"{batch}")
            # Process the sample here: forward, backward, ...

    def _create_replay_buffer(self) -> rpc.RRef:
        def connect():
            replay_buffer_info = rpc.get_worker_info(self.replay_buffer_node)
            buffer_rref = rpc.remote(
                replay_buffer_info, ReplayBufferNode, args=(10000,)
            )
            torchrl_logger.info(f"Connected to replay buffer {replay_buffer_info}")
            return buffer_rref

        while True:
            try:
                return connect()
            except Exception as e:
                torchrl_logger.info(f"Failed to connect to replay buffer: {e}")
                time.sleep(RETRY_DELAY_SECS)

    def _create_and_launch_data_collectors(self) -> None:
        data_collector_number = self.world_size - 2
        self.data_collectors = []
        self.data_collector_infos = []
        # discover launched data collector nodes (with retry to allow collectors to dynamically join)
        def connect(n, retry):
            data_collector_info = rpc.get_worker_info(
                f"DataCollector{n + 2}"  # 2, 3, 4, ...
            )
            torchrl_logger.info(
                f"Data collector info: {data_collector_info}-retry={retry}"
            )
            dc_ref = rpc.remote(
                data_collector_info,
                CollectorNode,
                args=(self.replay_buffer,),
            )
            self.data_collectors.append(dc_ref)
            self.data_collector_infos.append(data_collector_info)

        for n in range(data_collector_number):
            for retry in range(RETRY_LIMIT):
                try:
                    connect(n, retry)
                    break
                except Exception as e:
                    torchrl_logger.info(
                        f"Failed to connect to DataCollector{n} with {retry} retries (err={e})"
                    )
                    time.sleep(RETRY_DELAY_SECS)
            else:
                raise Exception
        for collector, data_collector_info in zip(
            self.data_collectors, self.data_collector_infos
        ):
            rpc.remote(
                data_collector_info,
                CollectorNode.collect,
                args=(collector,),
            )


class ReplayBufferNode(RemoteReplayBuffer):
    """Experience replay buffer node that is capable of accepting remote connections. Being a `RemoteReplayBuffer`
    means all of its public methods are remotely invokable using `torch.rpc`.
    Using a LazyMemmapStorage is highly advised in distributed settings with shared storage due to the lower serialisation
    cost of MemoryMappedTensors as well as the ability to specify file storage locations which can improve ability to recover from node failures.

    Args:
        capacity (int): the maximum number of elements that can be stored in the replay buffer.
    """

    def __init__(self, capacity: int):
        super().__init__(
            storage=LazyMemmapStorage(
                max_size=capacity, scratch_dir="/tmp/", device=torch.device("cpu")
            ),
            sampler=SliceSampler(num_slices=4),
            writer=RoundRobinWriter(),
            batch_size=32,
        )


def main(rank, world_size, **tensorpipe_kwargs):
    """Dispatcher for the distributed workflow.

    rank 0 will be assigned the TRAINER job,
    rank 1 will be assigned the REPLAY BUFFER job,
    rank 2 to world_size-1 will be assigned the COLLECTOR jobs.

    """
    torchrl_logger.info(f"Rank: {rank}")

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    #

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16, **tensorpipe_kwargs
    )

    if rank == 0:
        # rank 0 is the trainer
        torchrl_logger.info(f"Init RPC on {TRAINER_NODE}...")
        rpc.init_rpc(
            TRAINER_NODE,
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
            world_size=world_size,
        )
        torchrl_logger.info(f"Initialised {TRAINER_NODE}")
        trainer = TrainerNode(replay_buffer_node=REPLAY_BUFFER_NODE)
        trainer.train(100)
        rpc.shutdown()
    elif rank == 1:
        # rank 1 is the replay buffer
        #  replay buffer waits passively for construction instructions from trainer node
        torchrl_logger.info(f"Init RPC on {REPLAY_BUFFER_NODE}...")
        rpc.init_rpc(
            REPLAY_BUFFER_NODE,
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
            world_size=world_size,
        )
        torchrl_logger.info(f"Initialised {REPLAY_BUFFER_NODE}")
        rpc.shutdown()
    else:
        # rank 2+ is a new data collector node
        # data collectors also wait passively for construction instructions from trainer node
        torchrl_logger.info(f"Init RPC on DataCollector{rank}")
        rpc.init_rpc(
            f"DataCollector{rank}",
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
            world_size=world_size,
        )
        torchrl_logger.info(f"Initialised DataCollector{rank}")
        rpc.shutdown()
    print("exiting", rank)
