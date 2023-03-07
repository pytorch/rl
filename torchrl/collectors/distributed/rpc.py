# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Generic distributed data-collector using torch.distributed.rpc backend."""

import os
import socket
import time
from typing import OrderedDict

SUBMITIT_ERR = None
try:
    import submitit

    _has_submitit = True
except ModuleNotFoundError as err:
    _has_submitit = False
    SUBMITIT_ERR = err
import torch.cuda
from tensordict import TensorDict
from torch import multiprocessing as mp, nn

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
TCP_PORT = os.environ.get("TCP_PORT", "10003")
IDLE_TIMEOUT = os.environ.get("RCP_IDLE_TIMEOUT", 10)


def _rpc_init_collection_node(
    rank,
    rank0_ip,
    tcp_port,
    world_size,
):
    """Sets up RPC on the distant node.

    Args:
        rank (int): the rank of the process;
        rank0_ip (str): the IP address of the master process (rank 0)
        tcp_port (str or int): the TCP port of the master process
        world_size (int): the total number of nodes, including master.

    """
    proc = mp.Process(
        target=_rpc_init_collection_node_proc,
        args=(rank, rank0_ip, tcp_port, world_size),
    )
    proc.start()
    proc.join()


def _rpc_init_collection_node_proc(
    rank,
    rank0_ip,
    tcp_port,
    world_size,
):
    os.environ["MASTER_ADDR"] = str(rank0_ip)
    os.environ["MASTER_PORT"] = "29500"
    # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"
    # os.environ['TP_SOCKET_IFNAME']='lo'
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        init_method=f"tcp://{rank0_ip}:{tcp_port}",
        rpc_timeout=MAX_TIME_TO_CONNECT,
        _transports=["uv"],
        # Currently fails when nodes have more than 0 gpus avail,
        # even when no device is made visible
        devices=list(range(torch.cuda.device_count())),
        device_maps={f"COLLECTOR_NODE_{rank}": {0: 0} for rank in range(1, world_size)},
    )
    print("init rpc")
    rpc.init_rpc(
        f"COLLECTOR_NODE_{rank}",
        rank=rank,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=options,
        world_size=world_size,
    )
    rpc.shutdown()


class RPCDataCollector(_DataCollector):
    """An RPC-based distributed data collector.

    Supports sync and async data collection.

    Args:
        create_env_fn (list of callables or EnvBase instances): a list of the
            same length as the number of nodes to be launched.
        policy (Callable[[TensorDict], TensorDict]): a callable that populates
            the tensordict with an `"action"` field.
        frames_per_batch (int): the number of frames to be gathered in each
            batch.
        total_frames (int): the total number of frames to be collected from the
            distributed collector.
        collector_class (type or str, optional): a collector class for the remote node. Can be
            :class:`torchrl.collectors.SyncDataCollector`,
            :class:`torchrl.collectors.MultiSyncDataCollector`,
            :class:`torchrl.collectors.MultiaSyncDataCollector`
            or a derived class of these. The strings "single", "sync" and
            "async" correspond to respective class.
            Defaults to :class:`torchrl.collectors.SyncDataCollector`.
        collector_kwargs (dict, optional): a dictionary of parameters to be passed to the
            remote data-collector.
        num_workers_per_collector (int, optional): the number of copies of the
            env constructor that is to be used on the remote nodes.
            Defaults to 1 (a single env per collector).
            On a single worker node all the sub-workers will be
            executing the same environment. If different environments need to
            be executed, they should be dispatched across worker nodes, not
            subnodes.
        sync (bool, optional): if ``True``, the resulting tensordict is a stack of all the
            tensordicts collected on each node. If ``False`` (default), each
            tensordict results from a separate node in a "first-ready,
            first-served" fashion.
        slurm_kwargs (dict): a dictionary of parameters to be passed to the
            submitit executor.
        update_after_each_batch (bool, optional): if ``True``, the weights will
            be updated after each collection. For ``sync=True``, this means that
            all workers will see their weights updated. For ``sync=False``,
            only the worker from which the data has been gathered will be
            updated.
            Defaults to ``False``, ie. updates have to be executed manually
            through
            ``torchrl.collectors.distributed.DistributedDataCollector.update_policy_weights_()``
        max_weight_update_interval (int, optional): the maximum number of
            batches that can be collected before the policy weights of a worker
            is updated.
            For sync collections, this parameter is overwritten by ``update_after_each_batch``.
            For async collections, it may be that one worker has not seen its
            parameters being updated for a certain time even if ``update_after_each_batch``
            is turned on.
            Defaults to -1 (no forced update).
        launcher (str, optional): how jobs should be launched.
            Can be one of "submitit" or "mp" for multiprocessing. The former
            can launch jobs across multiple nodes, whilst the latter will only
            launch jobs on a single machine. "submitit" requires the homonymous
            library to be installed.
            To find more about submitit, visit
            https://github.com/facebookincubator/submitit
            Defaults to "submitit".
        tcp_port (int, optional): the TCP port to be used. Defaults to 10003.

    """

    def __init__(
        self,
        create_env_fn,
        policy,
        frames_per_batch,
        total_frames,
        collector_class=SyncDataCollector,
        collector_kwargs=None,
        num_workers_per_collector=1,
        sync=False,
        slurm_kwargs=None,
        storing_device="cpu",
        update_after_each_batch=False,
        max_weight_update_interval=-1,
        launcher="submitit",
        tcp_port=None,
    ):
        if collector_class == "async":
            collector_class = MultiaSyncDataCollector
        elif collector_class == "sync":
            collector_class = MultiSyncDataCollector
        elif collector_class == "single":
            collector_class = SyncDataCollector
        self.collector_class = collector_class
        self.env_constructors = create_env_fn
        self.policy = policy
        if isinstance(policy, nn.Module):
            policy_weights = TensorDict(dict(policy.named_parameters()), [])
        else:
            policy_weights = TensorDict({}, [])
        self.policy_weights = policy_weights
        self.num_workers = len(create_env_fn)
        self.frames_per_batch = frames_per_batch
        self.storing_device = storing_device
        # make private to avoid changes from users during collection
        self._sync = sync
        self.update_after_each_batch = update_after_each_batch
        self.max_weight_update_interval = max_weight_update_interval
        self.launcher = launcher
        self._batches_since_weight_update = [0 for _ in range(self.num_workers)]
        if tcp_port is None:
            self.tcp_port = os.environ.get("TCP_PORT", "10003")
        else:
            self.tcp_port = str(tcp_port)

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
        collector_kwargs = collector_kwargs if collector_kwargs is not None else {}
        self.collector_kwargs = (
            collector_kwargs
            if isinstance(collector_kwargs, (list, tuple))
            else [collector_kwargs] * self.num_workers
        )

        self._init_workers()

    def _init_master_rpc(
        self,
        world_size,
    ):
        options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16,
            init_method=f"tcp://{self.IPAddr}:{self.tcp_port}",
            rpc_timeout=10_000,
            _transports=["uv"],
            # Currently fails when nodes have more than 0 gpus avail,
            # even when no device is made visible
            devices=list(range(torch.cuda.device_count())),
            device_maps={
                f"COLLECTOR_NODE_{rank}": {0: 0} for rank in range(1, world_size)
            },
        )
        print("init rpc")
        rpc.init_rpc(
            "TRAINER_NODE",
            rank=0,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
            world_size=world_size,
        )

    def _launch_workers(
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
                    "total_frames": -1,
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
        if self.launcher == "submitit":
            if not _has_submitit:
                raise ImportError("submitit not found.") from SUBMITIT_ERR
            job = executor.submit(
                _rpc_init_collection_node,
                i + 1,
                self.IPAddr,
                self.tcp_port,
                self.num_workers + 1,
            )
            print("job id", job.job_id)  # ID of your job
            return job
        elif self.launcher == "mp":
            job = mp.Process(
                target=_rpc_init_collection_node,
                args=(i + 1, self.IPAddr, self.tcp_port, self.num_workers + 1),
            )
            job.start()
            return job
        else:
            raise NotImplementedError(f"Unknown launcher {self.launcher}")

    def _init_workers(self):
        if self.launcher == "submitit":
            executor = submitit.AutoExecutor(folder="log_test")
            executor.update_parameters(**self.slurm_kwargs)
        else:
            executor = None

        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        self.IPAddr = IPAddr
        os.environ["MASTER_ADDR"] = str(self.IPAddr)
        os.environ["MASTER_PORT"] = "29500"
        # os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

        self.jobs = []
        for i in range(self.num_workers):
            print("Submitting job")
            job = self._init_worker_rpc(
                executor,
                i,
            )
            self.jobs.append(job)

        self._init_master_rpc(
            self.num_workers + 1,
        )
        self._launch_workers(
            self.num_workers + 1,
            self.env_constructors,
            self.collector_class,
            self.num_workers_per_collector,
            self.policy,
            self._frames_per_batch_corrected,
            self.total_frames,
            self.collector_kwargs,
        )

    def iterator(self):
        print("Iterating")
        self._collected_frames = 0
        while self._collected_frames < self.total_frames:
            if self._sync:
                data = self._next_sync_rpc()
            else:
                data = self._next_async_rpc()
            yield data

    def update_policy_weights_(self) -> None:
        futures = []
        for i in range(self.num_workers):
            print(f"calling update on worker {i}")
            futures.append(
                rpc.rpc_async(
                    self.collector_infos[i],
                    self.collector_class.update_policy_weights_,
                    args=(self.collector_rrefs[i], self.policy_weights.detach()),
                )
            )
        for i in range(self.num_workers):
            print(f"waiting for worker {i}")
            self.futures[i].wait()

    def _next_async_rpc(self):
        while True:
            for i, future in enumerate(self.futures):
                if future.done():
                    data = future.value()
                    self._collected_frames += data.numel()
                    if self._collected_frames < self.total_frames:
                        self.futures[i] = rpc.rpc_async(
                            self.collector_infos[i],
                            self.collector_class.next,
                            args=(self.collector_rrefs[i],),
                        )
                    else:
                        self.futures[i] = None
                    return data.to(self.storing_device)

    def _next_sync_rpc(self):
        data = []
        while len(data) < self.num_workers:
            for i, future in enumerate(self.futures):
                # the order is NOT guaranteed: should we change that?
                if future.done():
                    data += [future.value()]
                    self.futures[i] = rpc.rpc_async(
                        self.collector_infos[i],
                        self.collector_class.next,
                        args=(self.collector_rrefs[i],),
                    )
        data = torch.cat(data).to(self.storing_device)
        self._collected_frames += data.numel()
        return data

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        for worker in self.collector_infos:
            seed = rpc.rpc_sync(worker, self.collector_class.set_seed, args=(seed,))

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    def shutdown(self):
        for future in self.futures:
            # clear the futures
            while future is not None and not future.done():
                continue
        for i in range(self.num_workers):
            rpc.rpc_sync(
                self.collector_infos[i],
                self.collector_class.shutdown,
                args=(self.collector_rrefs[i],),
                timeout=int(IDLE_TIMEOUT),
            )
        rpc.shutdown(timeout=int(IDLE_TIMEOUT))
        if self.launcher == "mp":
            for job in self.jobs:
                job.join(int(IDLE_TIMEOUT))
        elif self.launcher == "submitit":
            for job in self.jobs:
                _ = job.result()
        else:
            raise NotImplementedError(f"Unknown launcher {self.launcher}")
