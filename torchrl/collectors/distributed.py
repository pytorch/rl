import os
import socket
import time
from typing import OrderedDict

import submitit

from torch.distributed import rpc

from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import (
    _DataCollector,
    MultiSyncDataCollector,
    RandomPolicy,
    SyncDataCollector,
)
from torchrl.envs import EnvBase, EnvCreator
from torchrl.envs.vec_env import _BatchedEnv

MAX_TIME_TO_CONNECT = 1000
DEFAULT_SLURM_CONF = {
    "timeout_min": 10,
    "slurm_partition": "train",
    "slurm_cpus_per_task": 32,
    "slurm_gpus_per_node": 0,
}


def collect(rank, rank0_ip):
    os.environ["MASTER_ADDR"] = str(rank0_ip)
    os.environ["MASTER_PORT"] = "29500"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        init_method=f"tcp://{rank0_ip}:10002",
        rpc_timeout=MAX_TIME_TO_CONNECT,
        _transports=["uv"],
    )
    print("init rpc")
    rpc.init_rpc(
        f"COLLECTOR_NODE_{rank}",
        rank=rank,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=options,
    )
    print("waiting...")
    time.sleep(10_000)
    rpc.shutdown()


class DistributedDataCollector(_DataCollector):
    def __init__(
        self,
        env_makers,
        policy,
        collector_class,
        num_workers_per_collector,
        frames_per_batch,
        total_frames,
        slurm_kwargs=None,
        collector_kwargs=None,
        device_maps=None,
    ):
        if collector_class == "async":
            collector_class = MultiaSyncDataCollector
        elif collector_class == "sync":
            collector_class = MultiSyncDataCollector
        elif collector_class == "single":
            collector_class = SyncDataCollector
        self.collector_class = collector_class
        self.env_constructors = env_makers
        self.policy = policy
        self.num_workers = len(env_makers)
        self.frames_per_batch = frames_per_batch
        self.num_workers_per_collector = num_workers_per_collector
        self.total_frames = total_frames
        self.slurm_kwargs = (
            slurm_kwargs if slurm_kwargs is not None else DEFAULT_SLURM_CONF
        )
        self.device_maps = device_maps
        self.collector_kwargs = collector_kwargs if collector_kwargs is not None else {}

        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        print("IP address", IPAddr)
        self.IPAddr = IPAddr
        os.environ["MASTER_ADDR"] = str(IPAddr)
        os.environ["MASTER_PORT"] = "29500"
        os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16,
            init_method="tcp://localhost:10002",
            rpc_timeout=10_000,
            _transports=["uv"],
        )
        self.options = options
        print("init rpc")
        rpc.init_rpc(
            "TRAINER_NODE",
            rank=0,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
        )
        self._init_workers()

    def _init_workers(self):
        self.collector_infos = []
        self.collector_rrefs = []
        self.futures = []
        self.executors = []
        for i in range(self.num_workers):
            print("Submitting job")
            executor = submitit.AutoExecutor(folder="log_test")
            executor.update_parameters(**self.slurm_kwargs)
            job = executor.submit(collect, i + 1, self.IPAddr)  # will compute add(5, 7)
            print("job id", job.job_id)  # ID of your job
            self.executors.append(executor)
            if self.device_maps is not None:
                self.options.set_device_map(f"COLLECTOR_NODE_{i}", self.device_maps)

        for i in range(self.num_workers):
            counter = 0
            time_interval = 1.0
            while True:
                counter += 1
                time.sleep(time_interval)
                try:
                    print("trying to connect to collector node")
                    collector_info = rpc.get_worker_info(f"COLLECTOR_NODE_{i+1}")
                    break
                except RuntimeError as err:
                    if counter * time_interval > MAX_TIME_TO_CONNECT:
                        raise RuntimeError("Could not connect to remote node") from err
                    continue
            env_make = self.env_constructors[i]
            if not isinstance(env_make, (EnvBase, EnvCreator)):
                env_make = EnvCreator(env_make)
            print("Making collector in remote node")
            collector_rref = rpc.remote(
                collector_info,
                self.collector_class,
                args=(
                    [env_make] * self.num_workers_per_collector
                    if self.collector_class is not SyncDataCollector
                    else env_make,
                    self.policy,
                ),
                kwargs={
                    "frames_per_batch": self.frames_per_batch,
                    "total_frames": self.total_frames,
                    "split_trajs": False,
                    **self.collector_kwargs,
                },
            )
            print("Asking for the first batch")
            future = rpc.rpc_async(
                collector_info, self.collector_class.next, args=(collector_rref,)
            )

            self.collector_infos.append(collector_info)
            self.collector_rrefs.append(collector_rref)
            self.futures.append(future)

    def iterator(self):
        total_frames = 0
        while total_frames < self.total_frames:
            for i, future in enumerate(self.futures):
                if future.done():
                    data = future.value()
                    total_frames += data.numel()
                    self.futures[i] = rpc.rpc_async(
                        self.collector_infos[i],
                        self.collector_class.next,
                        args=(self.collector_rrefs[i],),
                    )
                    yield data

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        for worker in self.collector_infos:
            seed = rpc.rpc_sync(worker, self.collector_class.set_seed, args=(seed,))

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    def shutdown(self):
        rpc.shutdown()
