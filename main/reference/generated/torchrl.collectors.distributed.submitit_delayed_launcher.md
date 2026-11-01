# submitit_delayed_launcher

*class*torchrl.collectors.distributed.submitit_delayed_launcher(*num_jobs*, *framework='distributed'*, *backend='gloo'*, *tcpport='10003'*, *submitit_main_conf: dict = {'slurm_cpus_per_task': 32, 'slurm_gpus_per_node': 1, 'slurm_partition': 'train', 'timeout_min': 10}*, *submitit_collection_conf: dict = {'slurm_cpus_per_task': 32, 'slurm_gpus_per_node': 0, 'slurm_partition': 'train', 'timeout_min': 10}*)[[source]](../../_modules/torchrl/collectors/distributed/utils.html#submitit_delayed_launcher)

Delayed launcher for submitit.

In some cases, launched jobs cannot spawn other jobs on their own and this
can only be done at the jump-host level.

In these cases, the `submitit_delayed_launcher()` can be used to
pre-launch collector nodes that will wait for the main worker to provide
the launching instruction.

Parameters:

- **num_jobs** (*int*) - the number of collection jobs to be launched.
- **framework** (*str**,**optional*) - the framework to use. Can be either `"distributed"`
or `"rpc"`. `"distributed"` requires a [`DistributedCollector`](torchrl.collectors.distributed.DistributedCollector.html#torchrl.collectors.distributed.DistributedCollector)
collector whereas `"rpc"` requires a [`RPCCollector`](torchrl.collectors.distributed.RPCCollector.html#torchrl.collectors.distributed.RPCCollector).
Defaults to `"distributed"`.
- **backend** (*str**,**optional*) - torch.distributed backend in case `framework`
points to `"distributed"`. This value must match the one passed to
the collector, otherwise main and satellite nodes will fail to
reach the rendezvous and hang forever (ie no exception will be raised!)
Defaults to `'gloo'`.
- **tcpport** (*int**or**str**,**optional*) - the TCP port to use.
Defaults to `torchrl.collectors.distributed.default_configs.TCP_PORT`
- **submitit_main_conf** (*dict**,**optional*) - the main node configuration to be passed to submitit.
Defaults to `torchrl.collectors.distributed.default_configs.DEFAULT_SLURM_CONF_MAIN`
- **submitit_collection_conf** (*dict**,**optional*) - the configuration to be passed to submitit.
Defaults to `torchrl.collectors.distributed.default_configs.DEFAULT_SLURM_CONF`

Examples

```
>>> num_jobs=2
>>> @submitit_delayed_launcher(num_jobs=num_jobs)
... def main():
... from torchrl.modules.utils.utils import RandomPolicyfrom torchrl.envs.libs.gym import GymEnv
... from torchrl.data import BoundedContinuous
... collector = DistributedCollector(
... [EnvCreator(lambda: GymEnv("Pendulum-v1"))] * num_jobs,
... policy=RandomPolicy(BoundedContinuous(-1, 1, shape=(1,))),
... launcher="submitit_delayed",
... )
... for data in collector:
... print(data)
...
>>> if __name__ == "__main__":
... main()
...
```