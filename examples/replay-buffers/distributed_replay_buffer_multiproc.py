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
python examples/replay-buffers/distributed_replay_buffer_multiproc.py
```

"""

import os
import sys
import time

import torch.distributed.rpc as rpc

from distributed_rb_utils import main
from torch import multiprocessing as mp


REPLAY_BUFFER_NODE = "ReplayBuffer"
TRAINER_NODE = "Trainer"

if __name__ == "__main__":
    ctx = mp.get_context("spawn")
    procs = []
    world_size = 3
    for i in range(world_size):
        procs.append(ctx.Process(target=main, args=(i, world_size)))
        procs[-1].start()

    for p in reversed(procs):
        p.join()
