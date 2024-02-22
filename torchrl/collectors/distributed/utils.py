import subprocess
import time

from torchrl._utils import logger as torchrl_logger, VERBOSE
from torchrl.collectors.distributed.default_configs import (
    DEFAULT_SLURM_CONF,
    DEFAULT_SLURM_CONF_MAIN,
    TCP_PORT,
)
from torchrl.collectors.distributed.generic import _distributed_init_delayed
from torchrl.collectors.distributed.rpc import _rpc_init_collection_node

try:
    import submitit

    _has_submitit = True
except ModuleNotFoundError as err:
    _has_submitit = False
    SUBMITIT_ERR = err


class submitit_delayed_launcher:
    """Delayed launcher for submitit.

    In some cases, launched jobs cannot spawn other jobs on their own and this
    can only be done at the jump-host level.

    In these cases, the :func:`submitit_delayed_launcher` can be used to
    pre-launch collector nodes that will wait for the main worker to provide
    the launching instruction.

    Args:
        num_jobs (int): the number of collection jobs to be launched.
        framework (str, optional): the framework to use. Can be either ``"distributed"``
            or ``"rpc"``. ``"distributed"`` requires a :class:`~.DistributedDataCollector`
            collector whereas ``"rpc"`` requires a :class:`RPCDataCollector`.
            Defaults to ``"distributed"``.
        backend (str, optional): torch.distributed backend in case ``framework``
            points to ``"distributed"``. This value must match the one passed to
            the collector, otherwise main and satellite nodes will fail to
            reach the rendezvous and hang forever (ie no exception will be raised!)
            Defaults to ``'gloo'``.
        tcpport (int or str, optional): the TCP port to use.
            Defaults to :obj:`torchrl.collectors.distributed.default_configs.TCP_PORT`
        submitit_main_conf (dict, optional): the main node configuration to be passed to submitit.
            Defaults to :obj:`torchrl.collectors.distributed.default_configs.DEFAULT_SLURM_CONF_MAIN`
        submitit_collection_conf (dict, optional): the configuration to be passed to submitit.
            Defaults to :obj:`torchrl.collectors.distributed.default_configs.DEFAULT_SLURM_CONF`

    Examples:
        >>> num_jobs=2
        >>> @submitit_delayed_launcher(num_jobs=num_jobs)
        ... def main():
        ...     from torchrl.envs.utils import RandomPolicy
                from torchrl.envs.libs.gym import GymEnv
        ...     from torchrl.data import BoundedTensorSpec
        ...     collector = DistributedDataCollector(
        ...         [EnvCreator(lambda: GymEnv("Pendulum-v1"))] * num_jobs,
        ...         policy=RandomPolicy(BoundedTensorSpec(-1, 1, shape=(1,))),
        ...         launcher="submitit_delayed",
        ...     )
        ...     for data in collector:
        ...         print(data)
        ...
        >>> if __name__ == "__main__":
        ...     main()
        ...
    """

    _VERBOSE = VERBOSE  # for debugging

    def __init__(
        self,
        num_jobs,
        framework="distributed",
        backend="gloo",
        tcpport=TCP_PORT,
        submitit_main_conf: dict = DEFAULT_SLURM_CONF_MAIN,
        submitit_collection_conf: dict = DEFAULT_SLURM_CONF,
    ):
        self.num_jobs = num_jobs
        self.backend = backend
        self.framework = framework
        self.submitit_collection_conf = submitit_collection_conf
        self.submitit_main_conf = submitit_main_conf
        self.tcpport = tcpport

    def __call__(self, main_func):
        def exec_fun():
            if not _has_submitit:
                raise ModuleNotFoundError(
                    "Failed to import submitit. Check installation of the library."
                ) from SUBMITIT_ERR
            # submit main
            executor = submitit.AutoExecutor(folder="log_test")
            executor.update_parameters(**self.submitit_main_conf)
            main_job = executor.submit(main_func)
            # listen to output file looking for IP address
            torchrl_logger.info(f"job id: {main_job.job_id}")
            time.sleep(2.0)
            node = None
            while not node:
                cmd = f"squeue -j {main_job.job_id} -o %N | tail -1"
                node = subprocess.check_output(cmd, shell=True, text=True).strip()
                try:
                    node = int(node)
                except ValueError:
                    time.sleep(0.5)
                    continue
            torchrl_logger.info(f"node: {node}")
            # by default, sinfo will truncate the node name at char 20, we increase this to 200
            cmd = f"sinfo -n {node} -O nodeaddr:200 | tail -1"
            rank0_ip = subprocess.check_output(cmd, shell=True, text=True).strip()
            torchrl_logger.info(f"IP: {rank0_ip}")
            world_size = self.num_jobs + 1

            # submit jobs
            executor = submitit.AutoExecutor(folder="log_test")
            executor.update_parameters(**self.submitit_collection_conf)
            jobs = []
            if self.framework == "rpc":
                from .rpc import DEFAULT_TENSORPIPE_OPTIONS

                tensorpipe_options = DEFAULT_TENSORPIPE_OPTIONS
            for i in range(self.num_jobs):
                rank = i + 1
                if self.framework == "distributed":
                    job = executor.submit(
                        _distributed_init_delayed,
                        rank,
                        self.backend,
                        rank0_ip,
                        self.tcpport,
                        world_size,
                        self._VERBOSE,
                    )
                elif self.framework == "rpc":
                    job = executor.submit(
                        _rpc_init_collection_node,
                        rank,
                        rank0_ip,
                        self.tcpport,
                        world_size,
                        None,
                        tensorpipe_options,
                    )
                else:
                    raise NotImplementedError(f"Unknown framework {self.framework}.")
                jobs.append(job)
            for job in jobs:
                job.result()
            main_job.result()

        return exec_fun
