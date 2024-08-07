"""
Example use of a distributed replay buffer
==========================================

This example illustrates how a skeleton reinforcement learning algorithm can be implemented in a distributed fashion
with communication between nodes/workers handled using `torch.rpc`.
It focusses on how to set up a replay buffer worker that accepts remote operation requests efficiently, and so omits
any learning component such as parameter updates that may be required for a complete distributed reinforcement learning
algorithm implementation.

In this model, >= 1 data collectors workers are responsible for collecting experiences in an environment, the replay
buffer worker receives all of these experiences and exposes them to a trainer that is responsible for making parameter
updates to any required models.

To launch this script, run

```bash
$ torchrun examples/replay-buffers/distributed_replay_buffer_torchrun.py --nnodes=1
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
from torchrl._utils import (
    accept_remote_rref_invocation,
    logger as torchrl_logger,
)
from torchrl.data.replay_buffers import RemoteReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.data.replay_buffers.storages import LazyMemmapStorage
from torchrl.data.replay_buffers.writers import RoundRobinWriter
from distributed_rb_utils import TrainerNode, CollectorNode, ReplayBufferNode

REPLAY_BUFFER_NODE = "ReplayBuffer"
TRAINER_NODE = "Trainer"

if __name__ == "__main__":

    rank = int(os.environ["LOCAL_RANK"])
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
