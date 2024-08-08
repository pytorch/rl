"""
Example use of a distributed replay buffer (submitit)
=====================================================

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
python examples/replay-buffers/distributed_replay_buffer_submitit.py
```

"""

import submitit

from distributed_rb_utils import main
from torch import multiprocessing as mp

DEFAULT_SLURM_CONF = {
    "timeout_min": 10,
    "slurm_partition": "train",
    "slurm_cpus_per_task": 32,
    "slurm_gpus_per_node": 0,
}  #: Default value of the SLURM jobs

if __name__ == "__main__":

    executor = submitit.AutoExecutor(folder="log_test")
    executor.update_parameters(**DEFAULT_SLURM_CONF)

    ctx = mp.get_context("spawn")
    jobs = []
    world_size = 3
    for i in range(world_size):
        job = executor.submit(main, i, world_size)
        jobs.append(job)

    for i in range(world_size):
        jobs[i].result()
