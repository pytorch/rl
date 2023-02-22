import os
import socket
import time
from datetime import timedelta
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
TCP_PORT=os.environ.get("TCP_PORT", "10003")


def rpc_init_collection_node(rank, rank0_ip, world_size, ):
    """Sets up RPC on the distant node.

    Args:
        rank (int): the rank of the process;
        rank0_ip (str): the IP address of the master process (rank 0)
        world_size (int): the total number of nodes, including master.

    """
    os.environ["MASTER_ADDR"] = str(rank0_ip)
    os.environ["MASTER_PORT"] = "29500"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # os.environ['TP_SOCKET_IFNAME']='lo'
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        init_method=f"tcp://{rank0_ip}:{TCP_PORT}",
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


def distributed_init_collection_node(
    rank,
    rank0_ip,
    sync,
    world_size,
    backend,
    collector_class,
    env_make,
    policy,
    frames_per_batch,
    total_frames,
    collector_kwargs,
):
    if not backend.startswith("distributed"):
        raise RuntimeError(f"Backend {backend} is incompatible with the collector.")
    _, backend = backend.split(":")
    torch.distributed.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(MAX_TIME_TO_CONNECT),
        init_method=f"tcp://{rank0_ip}:{TCP_PORT}",
    )
    collector = collector_class(
        env_make,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        split_trajs=False,
        **collector_kwargs,
    )
    if not sync:
        _store = torch.distributed.TCPStore(
            host_name=rank0_ip,
            port=int(TCP_PORT)+1,
            world_size=world_size,
            is_master=False,
        )
    for data in collector:
        if sync:
            data.gather_and_stack(dest=0)
        else:
            _store.set(f"NODE_{rank}_status", "busy")
            data.isend(dst=0)
            while _store.get(f"NODE_{rank}_status") != "continue":
                time.sleep(1e-4)

    collector.shutdown()


class DistributedDataCollector(_DataCollector):
    """A distributed data collector with submitit backend.

    To find more about submitit, visit https://github.com/facebookincubator/submitit

    Supports sync and async data collection.

    Args:
        env_makers (list of callables or EnvBase instances): a list of the
            same length as the number of nodes to be launched.
        policy (Callable[[TensorDict], TensorDict]): a callable that populates
            the tensordict with an `"action"` field.
        collector_class (type): a collector class for the remote node. Can be
            :class:`torchrl.collectors.SyncDataCollector`, :class:`torchrl.collectors.MultiSyncDataCollector`, :class:`torchrl.collectors.MultiaSyncDataCollector`
            or a derived class of these.
        frames_per_batch (int): the number of frames to be gathered in each
            batch.
        total_frames (int): the total number of frames to be collected from the
            distributed collector.
        num_workers_per_collector (int): the number of copies of the
            env constructor that is to be used on the remote nodes.
            Defaults to 1 (a single env per collector).
        sync (bool): if ``True``, the resulting tensordict is a stack of all the
            tensordicts collected on each node. If ``False`` (default), each
            tensordict results from a separate node in a "first-ready,
            first-served" fashion.
        slurm_kwargs (dict): a dictionary of parameters to be passed to the
            submitit executor.
        collector_kwargs (dict): a dictionary of parameters to be passed to the
            remote data-collector.
        backend (str): must be one of "rpc" or "distributed:<distributed_backed>".
            <distributed_backed> is one of "gloo", "mpi", "nccl" or "ucc". See
            the torch.distributed documentation for more information.
            Defaults to "distributed:gloo".
    """

    def __init__(
        self,
        env_makers,
        policy,
        collector_class,
        frames_per_batch,
        total_frames,
        num_workers_per_collector=1,
        sync=False,
        slurm_kwargs=None,
        collector_kwargs=None,
        backend="distributed:gloo",
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
            self._frames_per_batch_corrected = self.frames_per_batch // self.num_workers
        else:
            self._frames_per_batch_corrected = self.frames_per_batch

        self.num_workers_per_collector = num_workers_per_collector
        self.total_frames = total_frames
        self.slurm_kwargs = (
            slurm_kwargs if slurm_kwargs is not None else DEFAULT_SLURM_CONF
        )
        self.collector_kwargs = collector_kwargs if collector_kwargs is not None else {}
        self.backend = backend

        # os.environ['TP_SOCKET_IFNAME'] = 'lo'

        self._init_workers()

    def _init_master_dist(
        self,
        world_size,
        backend,
    ):
        _, backend = backend.split(":")
        torch.distributed.init_process_group(
            backend,
            rank=0,
            world_size=world_size,
            timeout=timedelta(MAX_TIME_TO_CONNECT),
            init_method=f"tcp://{self.IPAddr}:{TCP_PORT}",
        )
        env_constructor = self.env_constructors[0]
        pseudo_collector = SyncDataCollector(
            env_constructor,
            self.policy,
            frames_per_batch=self._frames_per_batch_corrected,
            total_frames=self.total_frames,
            split_trajs=False,
        )
        if not self._sync:
            self._store = torch.distributed.TCPStore(
                host_name=self.IPAddr,
                port=int(TCP_PORT)+1,
                world_size=self.num_workers+1,
                is_master=True,
            )
        for data in pseudo_collector:
            break
        if not issubclass(self.collector_class, SyncDataCollector):
            # Multi-data collectors
            self._out_tensordict = data.expand(
                (self.num_workers, self.num_workers_per_collector, *data.shape)
            ).to_tensordict()
        else:
            # Multi-data collectors
            self._out_tensordict = data.expand(
                (self.num_workers, *data.shape)
            ).to_tensordict()

    def _init_master_rpc(
        self,
        world_size,
        env_constructors,
        collector_class,
        num_workers_per_collector,
        policy,
        frames_per_batch,
        total_frames,
        collector_kwargs,
    ):
        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16,
            init_method=f"tcp://{self.IPAddr}:{TCP_PORT}",
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

        num_workers = world_size - 1
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
                        raise RuntimeError("Could not connect to remote node") from err
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
        self.futures = futures
        self.collector_rrefs = collector_rrefs
        self.collector_infos = collector_infos

    def _init_worker_rpc(self, executor, i):
        job = executor.submit(
            rpc_init_collection_node, i + 1, self.IPAddr, self.num_workers + 1
        )
        return job

    def _init_worker_dist(self, executor, i):
        job = executor.submit(
            distributed_init_collection_node,
            i + 1,
            self.IPAddr,
            self._sync,
            self.num_workers + 1,
            self.backend,
            self.collector_class,
            self.env_constructors[i],
            self.policy,
            self._frames_per_batch_corrected,
            self.total_frames,
            self.collector_kwargs,
        )
        return job

    def _init_workers(self):
        executor = submitit.AutoExecutor(folder="log_test")
        executor.update_parameters(**self.slurm_kwargs)

        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        self.IPAddr = IPAddr
        os.environ["MASTER_ADDR"] = str(self.IPAddr)
        os.environ["MASTER_PORT"] = "29500"
        # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

        for i in range(self.num_workers):
            print("Submitting job")
            if self.backend.startswith("rpc"):
                job = self._init_worker_rpc(
                    executor,
                    i,
                )
            else:
                job = self._init_worker_dist(
                    executor,
                    i,
                )
            print("job id", job.job_id)  # ID of your job

        if self.backend.startswith("rpc"):
            self._init_master_rpc(
                self.num_workers + 1,
                self.env_constructors,
                self.collector_class,
                self.num_workers_per_collector,
                self.policy,
                self._frames_per_batch_corrected,
                self.total_frames,
                self.collector_kwargs,
            )
        else:
            self._init_master_dist(self.num_workers + 1, self.backend)

    def iterator(self):
        if self.backend.startswith("rcp"):
            yield from self._iterator_rpc()
        else:
            yield from self._iterator_dist()

    def _iterator_dist(self):
        total_frames = 0
        if not self._sync:
            trackers = []
            for i in range(self.num_workers):
                trackers.append(
                    self._out_tensordict[i].irecv(
                        src=i + 1,
                        return_premature=True
                        )
                )

        while total_frames < self.total_frames:
            if self._sync:
                self._out_tensordict.gather_and_stack(dest=0)
                data = self._out_tensordict.to_tensordict()
            else:
                data = None
                while data is None:
                    for i in range(self.num_workers):
                        if all(_data.wait() for _data in trackers[i]):
                            data = self._out_tensordict[i].to_tensordict()
                            self._store.set(f"NODE_{i+1}_status", "continue")
                            trackers[i] = self._out_tensordict[i].irecv(
                                    src=i + 1,
                                    return_premature=True
                                )
                            break
            total_frames += data.numel()
            yield data

    def _iterator_rpc(self):
        total_frames = 0
        while total_frames < self.total_frames:
            if self._sync:
                data = self._next_sync_rpc()
            else:
                data = self._next_async_rpc()
            total_frames += data.numel()
            yield data

    def _next_async_rpc(self):
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

    def _next_sync_rpc(self):
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
