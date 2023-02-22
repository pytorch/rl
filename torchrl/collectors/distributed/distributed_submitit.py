import os
import socket
import time
from typing import OrderedDict

SUBMITIT_ERR = None
try:
    import submitit
    _has_submitit = False
except ModuleNotFoundError as err:
    _has_submitit = False
    SUBMITIT_ERR = err
import torch.cuda

from torch.distributed import rpc

from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import (
    _DataCollector,
    MultiSyncDataCollector,
    SyncDataCollector,
)
from torchrl.envs import EnvBase, EnvCreator

SLEEP_INTERVAL = 1e-6
MAX_TIME_TO_CONNECT = 1000
DEFAULT_SLURM_CONF = {
    "timeout_min": 10,
    "slurm_partition": "train",
    "slurm_cpus_per_task": 32,
    "slurm_gpus_per_node": 0,
}

def init_master(world_size, env_constructors, collector_class, num_workers_per_collector, policy, frames_per_batch, total_frames, collector_kwargs):
    hostname = socket.gethostname()
    IPAddr = socket.gethostbyname(hostname)

    print("IP address", IPAddr)

    os.environ["MASTER_ADDR"] = str(IPAddr)
    os.environ["MASTER_PORT"] = "29500"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        init_method=f"tcp://{IPAddr}:10003",
        rpc_timeout=10_000,
        _transports=["uv"],
        # Currently fails when nodes have more than 0 gpus avail,
        # even when no device is made visible
        devices=[],
    )
    print("init rpc")
    rpc.init_rpc(
        "TRAINER_NODE",
        rank=0,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=options,
        world_size=world_size,
    )

    num_workers = world_size-1
    time_interval = 1.0
    collector_infos = []
    for i in range(num_workers):
        counter = 0
        while True:
            counter += 1
            time.sleep(time_interval)
            try:
                print(f"trying to connect to collector node {i + 1}")
                collector_info = rpc.get_worker_info(f"COLLECTOR_NODE_{i + 1}")
                break
            except RuntimeError as err:
                if counter * time_interval > MAX_TIME_TO_CONNECT:
                    raise RuntimeError(
                        "Could not connect to remote node"
                        ) from err
                continue
        collector_infos.append(collector_info)

    collector_rrefs = []
    for i in range(num_workers):
        env_make = env_constructors[i]
        if not isinstance(env_make, (EnvBase, EnvCreator)):
            env_make = EnvCreator(env_make)
        print("Making collector in remote node")
        collector_rref = rpc.remote(
            collector_infos[i],
            collector_class,
            args=(
                [env_make] * num_workers_per_collector
                if collector_class is not SyncDataCollector
                else env_make,
                policy,
            ),
            kwargs={
                "frames_per_batch": frames_per_batch,
                "total_frames": total_frames,
                "split_trajs": False,
                **collector_kwargs,
            },
        )
        collector_rrefs.append(collector_rref)

    futures = []
    for i in range(num_workers):
        print("Asking for the first batch")
        future = rpc.rpc_async(
            collector_infos[i],
            collector_class.next,
            args=(collector_rrefs[i],),
        )

        futures.append(future)

    for f in futures:
        print(f.wait())

def init_collection_node(rank, rank0_ip, world_size):
    """Sets up RPC on the distant node.

    Args:
        rank (int): the rank of the process;
        rank0_ip (str): the IP address of the master process (rank 0)
        world_size (int): the total number of nodes, including master.

    """
    os.environ["MASTER_ADDR"] = str(rank0_ip)
    os.environ["MASTER_PORT"] = "29500"
    os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # os.environ['TP_SOCKET_IFNAME']='lo'
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        init_method=f"tcp://{rank0_ip}:10003",
        rpc_timeout=MAX_TIME_TO_CONNECT,
        _transports=["uv"],
        # Currently fails when nodes have more than 0 gpus avail,
        # even when no device is made visible
        devices=[],
    )
    print("init rpc")
    rpc.init_rpc(
        f"COLLECTOR_NODE_{rank}",
        rank=rank,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=options,
        world_size=world_size,
    )
    print("waiting...")
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
        sync=False,
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
        # make private to avoid changes from users during collection
        self._sync = sync
        if self._sync:
            if self.frames_per_batch % self.num_workers != 0:
                raise RuntimeError(
                    f"Cannot dispatch {self.frames_per_batch} frames across {self.num_workers}. "
                    f"Consider using a number of frames per batch that is divisible by the number of workers."
                )
            self.frames_per_batch = self.frames_per_batch // self.num_workers

        self.num_workers_per_collector = num_workers_per_collector
        self.total_frames = total_frames
        self.slurm_kwargs = (
            slurm_kwargs if slurm_kwargs is not None else DEFAULT_SLURM_CONF
        )
        self.device_maps = device_maps
        self.collector_kwargs = collector_kwargs if collector_kwargs is not None else {}

        # os.environ['TP_SOCKET_IFNAME'] = 'lo'

        self._init_workers()

    def _init_workers(self):
        self.collector_infos = []
        self.collector_rrefs = []
        self.futures = []
        self.executors = []
        executor = submitit.AutoExecutor(folder="log_test")
        executor.update_parameters(**self.slurm_kwargs)
        job_master = executor.submit(
            init_master, self.num_workers + 1,
            self.env_constructors,
            self.collector_class,
            self.num_workers_per_collector,
            self.policy,
            self.frames_per_batch,
            self.total_frames,
            self.collector_kwargs
        )
        # let's wait until we figure out what the IP address is
        IPAddr = None
        count = 0
        while IPAddr is None and count < 100:
            count += 1
            stdout_master = job_master.stdout()
            for line in stdout_master.split("\n"):
                if "IP address" in line:
                    IPAddr = line.split("IP address: ")
                    break
            else:
                time.sleep(1.0)
        if count >= 100:
            raise RuntimeError("Failed to get the IP address of master node.")

        for i in range(self.num_workers):
            print("Submitting job")
            job = executor.submit(
                init_collection_node, i + 1, self.IPAddr, self.num_workers + 1
            )
            print("job id", job.job_id)  # ID of your job
            self.executors.append(executor)


    def iterator(self):
        # placeholder
        for i in range(100000):
            time.sleep(1.0)
            yield i

    def _next_async(self):
        for i, future in enumerate(self.futures):
            if future.done():
                data = future.value()
                self.futures[i] = rpc.rpc_async(
                    self.collector_infos[i],
                    self.collector_class.next,
                    args=(self.collector_rrefs[i],),
                )
                return data
            else:
                time.sleep(SLEEP_INTERVAL)

    def _next_sync(self):
        data = []
        for i, future in enumerate(self.futures):
            if future.done():
                data += [future.value()]
                self.futures[i] = rpc.rpc_async(
                    self.collector_infos[i],
                    self.collector_class.next,
                    args=(self.collector_rrefs[i],),
                )
            else:
                time.sleep(SLEEP_INTERVAL)
        data = torch.cat(data)
        return data

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        for worker in self.collector_infos:
            seed = rpc.rpc_sync(worker, self.collector_class.set_seed, args=(seed,))

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    def shutdown(self):
        rpc.shutdown()
