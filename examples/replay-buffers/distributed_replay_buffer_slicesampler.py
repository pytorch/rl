"""
Example use of a distributed replay buffer
==========================================

To launch this script, run

```bash
$ # In terminal0: Trainer node
$ python examples/replay-buffers/distributed_replay_buffer_slicesampler.py --rank=0
$ # In terminal1: Replay buffer node
$ python examples/replay-buffers/distributed_replay_buffer_slicesampler.py --rank=1
$ # In terminal2 to N: Collector nodes
$ python examples/replay-buffers/distributed_replay_buffer_slicesampler.py --rank=2

```
"""

import argparse
import os
import random
import sys
import time

import torch
import torch.distributed.rpc as rpc
from tensordict import TensorDict
from torchrl._utils import accept_remote_rref_invocation, logger as torchrl_logger, accept_remote_rref_udf_invocation
from torchrl.data.replay_buffers import RemoteReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.data.replay_buffers.writers import RoundRobinWriter

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

    def __init__(self) -> None:
        torchrl_logger.info("TrainerNode")
        self.id = rpc.get_worker_info().id
        self.replay_buffer = self._create_replay_buffer()
        self._create_and_launch_data_collectors()

    def train(self, iterations: int) -> None:
        """Write your training loop here."""
        for iteration in range(iterations):
            torchrl_logger.info(f"[{self.id}] Training Iteration: {iteration}")
            # # Wait until the buffer has elements
            while not rpc.rpc_sync(
                self.replay_buffer.owner(),
                ReplayBufferNode.__len__,
                args=(self.replay_buffer,)
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
        while True:
            try:
                replay_buffer_info = rpc.get_worker_info(REPLAY_BUFFER_NODE)
                buffer_rref = rpc.remote(
                    replay_buffer_info, ReplayBufferNode, args=(10000,)
                )
                torchrl_logger.info(f"Connected to replay buffer {replay_buffer_info}")
                return buffer_rref
            except Exception as e:
                torchrl_logger.info(f"Failed to connect to replay buffer: {e}")
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
                torchrl_logger.info(f"Data collector info: {data_collector_info}")
                dc_ref = rpc.remote(
                    data_collector_info,
                    CollectorNode,
                    args=(self.replay_buffer,),
                )
                data_collectors.append(dc_ref)
                data_collector_infos.append(data_collector_info)
                data_collector_number += 1
                retries = 0
            except Exception:
                retries += 1
                torchrl_logger.info(
                    f"Failed to connect to DataCollector{data_collector_number} with {retries} retries"
                )
                if retries >= RETRY_LIMIT:
                    torchrl_logger.info(f"{len(data_collectors)} data collectors")
                    for data_collector_info, data_collector in zip(
                        data_collector_infos, data_collectors
                    ):
                        rpc.remote(
                            data_collector_info,
                            CollectorNode.collect,
                            args=(data_collector,),
                        )
                    break
                else:
                    time.sleep(RETRY_DELAY_SECS)


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


if __name__ == "__main__":

    args = parser.parse_args()
    rank = args.rank
    torchrl_logger.info(f"Rank: {rank}")

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
        torchrl_logger.info(f"Initialised Trainer Node {rank}")
        trainer = TrainerNode()
        trainer.train(100)
        breakpoint()
    elif rank == 1:
        # rank 1 is the replay buffer
        # replay buffer waits passively for construction instructions from trainer node
        torchrl_logger.info(REPLAY_BUFFER_NODE)
        rpc.init_rpc(
            REPLAY_BUFFER_NODE,
            rank=rank,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )
        torchrl_logger.info(f"Initialised RB Node {rank}")
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
        torchrl_logger.info(f"Initialised DC Node {rank}")
        breakpoint()
    else:
        sys.exit(1)
    rpc.shutdown()
