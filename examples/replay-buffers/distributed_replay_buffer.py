"""
Example use of a distributed replay buffer (custom)
===================================================

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
$ # In terminal0: Trainer node
$ python examples/replay-buffers/distributed_replay_buffer.py --rank=0
$ # In terminal1: Replay buffer node
$ python examples/replay-buffers/distributed_replay_buffer.py --rank=1
$ # In terminal2 to N: Collector nodes
$ python examples/replay-buffers/distributed_replay_buffer.py --rank=2

```
"""

import argparse

from distributed_rb_utils import main

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
parser.add_argument("--world_size", type=int, default=3, help="Number of nodes/workers")


if __name__ == "__main__":

    args = parser.parse_args()
    rank = args.rank
    world_size = args.world_size

    main(rank, world_size, str_init_method="tcp://localhost:10000")
