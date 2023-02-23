"""
Generic distributed data-collector using torch.distributed backend
==================================================================

"""

import os
import socket
import time
import warnings
from datetime import timedelta
from typing import OrderedDict

import torch.cuda
from tensordict import TensorDict
from torch import multiprocessing as mp, nn

from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import (
    _DataCollector,
    MultiSyncDataCollector,
    SyncDataCollector,
)

SUBMITIT_ERR = None
try:
    import submitit

    _has_submitit = False
except ModuleNotFoundError as err:
    _has_submitit = False
    SUBMITIT_ERR = err

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


def distributed_init_collection_node(
    rank,
    rank0_ip,
    sync,
    world_size,
    backend,
    collector_class,
    num_workers,
    env_make,
    policy,
    frames_per_batch,
    total_frames,
    collector_kwargs,
    verbose=True,
):
    if verbose:
        print(f"node with rank {rank} -- launching distributed")
    torch.distributed.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(MAX_TIME_TO_CONNECT),
        init_method=f"tcp://{rank0_ip}:{TCP_PORT}",
    )
    if verbose:
        print(f"node with rank {rank} -- creating collector")
    if not issubclass(collector_class, SyncDataCollector):
        env_make = [env_make] * num_workers
    elif num_workers != 1:
        raise RuntimeError(
            "SyncDataCollector and subclasses can only support a single environment."
        )
    collector = collector_class(
        env_make,
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        split_trajs=False,
        **collector_kwargs,
    )
    if verbose:
        print(f"node with rank {rank} -- creating store")
    # The store carries instructions for the node
    _store = torch.distributed.TCPStore(
        host_name=rank0_ip,
        port=int(TCP_PORT) + 1,
        world_size=world_size,
        is_master=False,
    )
    if isinstance(policy, nn.Module):
        policy_weights = TensorDict(dict(policy.named_parameters()), [])
    else:
        policy_weights = TensorDict({}, [])
    collector_iter = iter(collector)
    _store.set(f"NODE_{rank}_status", "busy")
    if verbose:
        print(f"node with rank {rank} -- first data")
    data = next(collector_iter)
    if verbose:
        print(f"node with rank {rank} -- loop")
    while True:
        instruction = _store.get(f"NODE_{rank}_status")
        if instruction == b"busy":
            _store.set(f"NODE_{rank}_status", "sending")
            if verbose:
                print(f"node with rank {rank} -- sending {data}")
            # sync and async only differ by how they send data
            if sync:
                data.gather_and_stack(dest=0)
            else:
                data.isend(dst=0)
            _store.set(f"NODE_{rank}_status", "done")
            # wait for instruction
            while _store.get(f"NODE_{rank}_status") == b"done":
                continue
            instruction = _store.get(f"NODE_{rank}_status")
        if instruction == b"continue":
            if verbose:
                print(f"node with rank {rank} -- new data")
            _store.set(f"NODE_{rank}_status", "busy")
            data = next(collector_iter)
            continue
        elif instruction == b"shutdown":
            if verbose:
                print(f"node with rank {rank} -- shutting down")
            try:
                collector.shutdown()
            except Exception:
                pass
            _store.set(f"NODE_{rank}_status", "down")
            break
        elif instruction == b"update_weights":
            if sync:
                policy_weights.recv(0)
            else:
                # wthout further arguments, irecv blocks until weights have
                # been updated
                policy_weights.irecv(0)
            # the policy has been updated: we can simply update the weights
            collector.update_policy_weights_()
        elif instruction.startswith(b"seeding"):
            seed = int(instruction.split(b"seeding_"))
            new_seed = collector.set_seed(seed)
            _store.set(f"NODE_{rank}_status", "seeded")
            _store.set(f"NODE_{rank}_seed", str(new_seed))
        else:
            raise RuntimeError(f"Instruction {instruction} is not recognised")
    if not collector.closed:
        collector.shutdown()
    return


class DistributedDataCollector(_DataCollector):
    """A distributed data collector with torch.distributed backend.

    Supports sync and async data collection.

    Args:
        env_makers (list of callables or EnvBase instances): a list of the
            same length as the number of nodes to be launched.
        policy (Callable[[TensorDict], TensorDict]): a callable that populates
            the tensordict with an `"action"` field.
        frames_per_batch (int): the number of frames to be gathered in each
            batch.
        total_frames (int): the total number of frames to be collected from the
            distributed collector.
        collector_class (type, optional): a collector class for the remote node. Can be
            :class:`torchrl.collectors.SyncDataCollector`,
            :class:`torchrl.collectors.MultiSyncDataCollector`,
            :class:`torchrl.collectors.MultiaSyncDataCollector`
            or a derived class of these.
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
        backend (str, optional): must a string "<distributed_backed>" where
            <distributed_backed> is one of "gloo", "mpi", "nccl" or "ucc". See
            the torch.distributed documentation for more information.
            Defaults to "gloo".
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
    """

    def __init__(
        self,
        env_makers,
        policy,
        frames_per_batch,
        total_frames,
        collector_class=SyncDataCollector,
        collector_kwargs=None,
        num_workers_per_collector=1,
        sync=False,
        slurm_kwargs=None,
        backend="gloo",
        storing_device="cpu",
        update_after_each_batch=False,
        max_weight_update_interval=-1,
        launcher="submitit",
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
        if isinstance(policy, nn.Module):
            policy_weights = TensorDict(dict(policy.named_parameters()), [])
        else:
            policy_weights = TensorDict({}, [])
        self.policy_weights = policy_weights
        self.num_workers = len(env_makers)
        self.frames_per_batch = frames_per_batch
        self.storing_device = storing_device
        self.update_after_each_batch = update_after_each_batch
        self.max_weight_update_interval = max_weight_update_interval
        self.launcher = launcher
        self._batches_since_weight_update = [0 for _ in range(self.num_workers)]

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
                port=int(TCP_PORT) + 1,
                world_size=self.num_workers + 1,
                is_master=True,
            )
        for data in pseudo_collector:
            break
        if not issubclass(self.collector_class, SyncDataCollector):
            # Multi-data collectors
            self._out_tensordict = (
                data.expand((self.num_workers, *data.shape))
                .to_tensordict()
                .to(self.storing_device)
            )
        else:
            # Multi-data collectors
            self._out_tensordict = (
                data.expand((self.num_workers, *data.shape))
                .to_tensordict()
                .to(self.storing_device)
            )

    def _init_worker_dist_submitit(self, executor, i):
        job = executor.submit(
            distributed_init_collection_node,
            i + 1,
            self.IPAddr,
            self._sync,
            self.num_workers + 1,
            self.backend,
            self.collector_class,
            self.num_workers_per_collector,
            self.env_constructors[i],
            self.policy,
            self._frames_per_batch_corrected,
            self.total_frames,
            self.collector_kwargs,
        )
        return job

    def _init_worker_dist_mp(self, i):
        job = mp.Process(
            target=distributed_init_collection_node,
            args=(
                i + 1,
                self.IPAddr,
                self._sync,
                self.num_workers + 1,
                self.backend,
                self.collector_class,
                self.num_workers_per_collector,
                self.env_constructors[i],
                self.policy,
                self._frames_per_batch_corrected,
                self.total_frames,
                self.collector_kwargs,
            ),
        )
        job.start()
        return job

    def _init_workers(self):

        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        self.IPAddr = IPAddr
        os.environ["MASTER_ADDR"] = str(self.IPAddr)
        os.environ["MASTER_PORT"] = "29500"
        if self.launcher == "mp":
            self.jobs = []
        elif self.launcher == "submitit":
            executor = submitit.AutoExecutor(folder="log_test")
            executor.update_parameters(**self.slurm_kwargs)
        for i in range(self.num_workers):
            print("Submitting job")
            if self.launcher == "submitit":
                job = self._init_worker_dist_submitit(
                    executor,
                    i,
                )
                print("job id", job.job_id)  # ID of your job
            elif self.launcher == "mp":
                job = self._init_worker_dist_mp(
                    i,
                )
                self.jobs.append(job)
                print("job launched")

        self._init_master_dist(self.num_workers + 1, self.backend)

    def iterator(self):
        yield from self._iterator_dist()

    def _iterator_dist(self):
        total_frames = 0
        if not self._sync:
            trackers = []
            for i in range(self.num_workers):
                trackers.append(
                    self._out_tensordict[i].irecv(src=i + 1, return_premature=True)
                )

        while total_frames < self.total_frames:
            if self._sync:
                self._out_tensordict.gather_and_stack(dest=0)
                data = self._out_tensordict.to_tensordict()
                if self.update_after_each_batch:
                    self.update_policy_weights_()
                else:
                    for j in range(self.num_workers):
                        self._batches_since_weight_update[j] += 1
            else:
                data = None
                while data is None:
                    for i in range(self.num_workers):
                        rank = i + 1
                        # if all(_tracker.is_completed() for _tracker in trackers[i]):
                        if self._store.get(f"NODE_{rank}_status") == b"done":
                            for _tracker in trackers[i]:
                                _tracker.wait()
                            data = self._out_tensordict[i].to_tensordict()
                            data._origin = rank
                            if self.update_after_each_batch:
                                self.update_policy_weights_(rank)
                            self._store.set(f"NODE_{rank}_status", "continue")
                            trackers[i] = self._out_tensordict[i].irecv(
                                src=i + 1, return_premature=True
                            )
                            for j in range(self.num_workers):
                                self._batches_since_weight_update[j] += j != i
                            break
            total_frames += data.numel()
            yield data

            if self.max_weight_update_interval > -1:
                for j in range(self.num_workers):
                    rank = j + 1
                    if (
                        self._batches_since_weight_update[j]
                        > self.max_weight_update_interval
                    ):
                        self.update_policy_weights_(rank)

        for i in range(self.num_workers):
            rank = i + 1
            self._store.set(f"NODE_{rank}_status", "shutdown")

    def update_policy_weights_(self, worker_rank=None) -> None:
        """Updates the weights of the worker nodes.

        Args:
            worker_rank (int, optional): if provided, only this worker weights
                will be updated.
        """
        if worker_rank is not None and worker_rank < 1:
            raise RuntimeError("worker_rank must be greater than 1")
        workers = range(self.num_workers) if worker_rank is None else [worker_rank - 1]
        for i in workers:
            rank = i + 1
            self._store.set(f"NODE_{rank}_status", "update_weights")
            if self._sync:
                self.policy_weights.send(rank)
            else:
                self.policy_weights.isend(rank)
            self._batches_since_weight_update[rank - 1] = 0

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        for i in range(self.num_workers):
            rank = i + 1
            self._store.set(f"NODE_{rank}_status", f"seeding_{seed}")
            count = 0
            interval = 1e-4
            while self._store.get(f"NODE_{rank}_status") != b"seeded":
                count += 1
                if count > 1e6:
                    raise RuntimeError(
                        f"Failed to seed during the allowed interval of {count*interval}s."
                    )
                time.sleep(interval)
            new_seed = self._store.get(f"NODE_{rank}_seed")
            seed = int(new_seed)
        return seed

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    def shutdown(self):
        for i in range(self.num_workers):
            rank = i + 1
            self._store.set(f"NODE_{rank}_status", "shutdown")
            count = 0
            interval = 1e-4
            while self._store.get(f"NODE_{rank}_status") != b"down":
                count += 1
                if count > 1e6:
                    warnings.warn(
                        f"Failed to shutdown during the allowed interval of {count*interval}s. "
                        f"The remote process may of may not be inactive."
                    )
                time.sleep(interval)
            if self.launcher == "mp":
                if not self.jobs[i].is_alive():
                    continue
                self.jobs[i].join(timeout=10)
        print("collector shut down")
