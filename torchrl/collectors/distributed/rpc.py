# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Generic distributed data-collector using torch.distributed.rpc backend."""
import collections
import os
import socket
import time
import warnings
from copy import copy, deepcopy
from typing import OrderedDict

from torchrl.collectors.distributed import DEFAULT_SLURM_CONF
from torchrl.collectors.distributed.default_configs import (
    DEFAULT_TENSORPIPE_OPTIONS,
    IDLE_TIMEOUT,
    TCP_PORT,
)
from torchrl.collectors.utils import split_trajectories
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs.utils import _convert_exploration_type

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
from torchrl._utils import VERBOSE

from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import (
    DataCollectorBase,
    DEFAULT_EXPLORATION_TYPE,
    MultiSyncDataCollector,
    SyncDataCollector,
)
from torchrl.envs import EnvBase, EnvCreator


def _rpc_init_collection_node(
    rank,
    rank0_ip,
    tcp_port,
    world_size,
    visible_device,
    tensorpipe_options,
    verbose=VERBOSE,
):
    os.environ["MASTER_ADDR"] = str(rank0_ip)
    os.environ["MASTER_PORT"] = str(tcp_port)

    if isinstance(visible_device, list):
        pass
    elif isinstance(visible_device, (str, int, torch.device)):
        visible_device = [visible_device]
    elif visible_device is None:
        pass
    else:
        raise RuntimeError(f"unrecognised dtype {type(visible_device)}")

    options = rpc.TensorPipeRpcBackendOptions(
        devices=visible_device,
        **tensorpipe_options,
    )
    if verbose:
        print(
            f"init rpc with master addr: {os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}"
        )
    rpc.init_rpc(
        f"COLLECTOR_NODE_{rank}",
        rank=rank,
        backend=rpc.BackendType.TENSORPIPE,
        rpc_backend_options=options,
        world_size=world_size,
    )
    rpc.shutdown()


class RPCDataCollector(DataCollectorBase):
    """An RPC-based distributed data collector.

    Supports sync and async data collection.

    Args:
        create_env_fn (Callable or List[Callabled]): list of Callables, each returning an
            instance of :class:`~torchrl.envs.EnvBase`.
        policy (Callable, optional): Instance of TensorDictModule class.
            Must accept TensorDictBase object as input.
            If ``None`` is provided, the policy used will be a
            :class:`RandomPolicy` instance with the environment
            ``action_spec``.
        frames_per_batch (int): A keyword-only argument representing the
            total number of elements in a batch.
        total_frames (int): A keyword-only argument representing the
            total number of frames returned by the collector
            during its lifespan. If the ``total_frames`` is not divisible by
            ``frames_per_batch``, an exception is raised.
             Endless collectors can be created by passing ``total_frames=-1``.
        max_frames_per_traj (int, optional): Maximum steps per trajectory.
            Note that a trajectory can span over multiple batches (unless
            ``reset_at_each_iter`` is set to ``True``, see below).
            Once a trajectory reaches ``n_steps``, the environment is reset.
            If the environment wraps multiple environments together, the number
            of steps is tracked for each environment independently. Negative
            values are allowed, in which case this argument is ignored.
            Defaults to ``-1`` (i.e. no maximum number of steps).
        init_random_frames (int, optional): Number of frames for which the
            policy is ignored before it is called. This feature is mainly
            intended to be used in offline/model-based settings, where a
            batch of random trajectories can be used to initialize training.
            Defaults to ``-1`` (i.e. no random frames).
        reset_at_each_iter (bool, optional): Whether environments should be reset
            at the beginning of a batch collection.
            Defaults to ``False``.
        postproc (Callable, optional): A post-processing transform, such as
            a :class:`~torchrl.envs.Transform` or a :class:`~torchrl.data.postprocs.MultiStep`
            instance.
            Defaults to ``None``.
        split_trajs (bool, optional): Boolean indicating whether the resulting
            TensorDict should be split according to the trajectories.
            See :func:`~torchrl.collectors.utils.split_trajectories` for more
            information.
            Defaults to ``False``.
        exploration_type (str, optional): interaction mode to be used when
            collecting data. Must be one of ``ExplorationType.RANDOM``,
            ``ExplorationType.MODE`` or
            ``ExplorationType.MEAN``.
            Defaults to ``ExplorationType.RANDOM``
        reset_when_done (bool, optional): if ``True`` (default), an environment
            that return a ``True`` value in its ``"done"`` or ``"truncated"``
            entry will be reset at the corresponding indices.
        collector_class (type or str, optional): a collector class for the remote node. Can be
            :class:`~torchrl.collectors.SyncDataCollector`,
            :class:`~torchrl.collectors.MultiSyncDataCollector`,
            :class:`~torchrl.collectors.MultiaSyncDataCollector`
            or a derived class of these. The strings "single", "sync" and
            "async" correspond to respective class.
            Defaults to :class:`~torchrl.collectors.SyncDataCollector`.

            .. note::

              Support for :class:`MultiSyncDataCollector` and :class:`MultiaSyncDataCollector`
              is experimental, and :class:`~torchrl.collectors.SyncDataCollector`
              should always be preferred. If multiple simultaneous environment
              need to be executed on a single node, consider using a
              :class:`~torchrl.envs.ParallelEnv` instance.

        collector_kwargs (dict or list, optional): a dictionary of parameters to be passed to the
            remote data-collector. If a list is provided, each element will
            correspond to an individual set of keyword arguments for the
            dedicated collector.
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
        storing_device (int, str or torch.device, optional): the device where
            data will be stored and delivered by the iterator. Defaults to
            ``"cpu"``.
        update_after_each_batch (bool, optional): if ``True``, the weights will
            be updated after each collection. For ``sync=True``, this means that
            all workers will see their weights updated. For ``sync=False``,
            only the worker from which the data has been gathered will be
            updated.
            Defaults to ``False``, ie. updates have to be executed manually
            through
            :meth:`~torchrl.collectors.distributed.DistributedDataCollector.update_policy_weights_`.
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
        visible_devices (list of Union[int, torch.device, str], optional): a
            list of the same length as the number of nodes containing the
            device used to pass data to main.
        tensorpipe_options (dict, optional): a dictionary of keyword argument
            to pass to :class:`torch.distributed.rpc.TensorPipeRpcBackendOption`.

    """

    _VERBOSE = VERBOSE  # for debugging

    def __init__(
        self,
        create_env_fn,
        policy,
        *,
        frames_per_batch,
        total_frames,
        max_frames_per_traj=-1,
        init_random_frames=-1,
        reset_at_each_iter=False,
        postproc=None,
        split_trajs=False,
        exploration_type=DEFAULT_EXPLORATION_TYPE,
        exploration_mode=None,
        reset_when_done=True,
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
        visible_devices=None,
        tensorpipe_options=None,
    ):
        exploration_type = _convert_exploration_type(
            exploration_mode=exploration_mode, exploration_type=exploration_type
        )
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
        if self.update_after_each_batch and self.max_weight_update_interval > -1:
            raise RuntimeError(
                "Got conflicting udpate instructions: `update_after_each_batch` "
                "`max_weight_update_interval` are incompatible."
            )
        self.launcher = launcher
        self._batches_since_weight_update = [0 for _ in range(self.num_workers)]
        if tcp_port is None:
            self.tcp_port = os.environ.get("TCP_PORT", TCP_PORT)
        else:
            self.tcp_port = str(tcp_port)
        self.visible_devices = visible_devices
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
        self.slurm_kwargs = copy(DEFAULT_SLURM_CONF)
        if slurm_kwargs is not None:
            self.slurm_kwargs.update(slurm_kwargs)

        collector_kwargs = collector_kwargs if collector_kwargs is not None else {}
        self.collector_kwargs = (
            deepcopy(collector_kwargs)
            if isinstance(collector_kwargs, (list, tuple))
            else [copy(collector_kwargs) for _ in range(self.num_workers)]
        )

        # update collector kwargs
        for collector_kwarg in self.collector_kwargs:
            collector_kwarg["max_frames_per_traj"] = max_frames_per_traj
            collector_kwarg["init_random_frames"] = (
                init_random_frames // self.num_workers
            )
            if not self._sync and init_random_frames > 0:
                warnings.warn(
                    "async distributed data collection with init_random_frames > 0 "
                    "may have unforeseen consequences as we do not control that once "
                    "non-random data is being collected all nodes are returning non-random data. "
                    "If this is a feature that you feel should be fixed, please raise an issue on "
                    "torchrl's repo."
                )
            collector_kwarg["reset_at_each_iter"] = reset_at_each_iter
            collector_kwarg["exploration_type"] = exploration_type
            collector_kwarg["reset_when_done"] = reset_when_done

        if postproc is not None and hasattr(postproc, "to"):
            self.postproc = postproc.to(self.storing_device)
        else:
            self.postproc = postproc
        self.split_trajs = split_trajs

        if tensorpipe_options is None:
            self.tensorpipe_options = copy(DEFAULT_TENSORPIPE_OPTIONS)
        else:
            self.tensorpipe_options = copy(DEFAULT_TENSORPIPE_OPTIONS).update(
                tensorpipe_options
            )
        self._init()

    def _init_master_rpc(
        self,
        world_size,
    ):
        """Init RPC on main node."""
        options = rpc.TensorPipeRpcBackendOptions(**self.tensorpipe_options)
        if torch.cuda.device_count():
            if self.visible_devices:
                for i in range(self.num_workers):
                    rank = i + 1
                    options.set_device_map(
                        f"COLLECTOR_NODE_{rank}", {0: self.visible_devices[i]}
                    )
        if self._VERBOSE:
            print("init rpc")
        rpc.init_rpc(
            "TRAINER_NODE",
            rank=0,
            backend=rpc.BackendType.TENSORPIPE,
            rpc_backend_options=options,
            world_size=world_size,
        )

    def _start_workers(
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
        """Instantiate remote collectors."""
        num_workers = world_size - 1
        time_interval = 1.0
        collector_infos = []
        for i in range(num_workers):
            counter = 0
            while True:
                counter += 1
                time.sleep(time_interval)
                try:
                    if self._VERBOSE:
                        print(f"trying to connect to collector node {i + 1}")
                    collector_info = rpc.get_worker_info(f"COLLECTOR_NODE_{i + 1}")
                    break
                except RuntimeError as err:
                    if counter * time_interval > self.tensorpipe_options["rpc_timeout"]:
                        raise RuntimeError("Could not connect to remote node") from err
                    continue
            collector_infos.append(collector_info)

        collector_rrefs = []
        for i in range(num_workers):
            env_make = env_constructors[i]
            if not isinstance(env_make, (EnvBase, EnvCreator)):
                env_make = CloudpickleWrapper(env_make)
            if self._VERBOSE:
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
                    **collector_kwargs[i],
                },
            )
            collector_rrefs.append(collector_rref)

        futures = collections.deque(maxlen=self.num_workers)

        if not self._sync:
            for i in range(num_workers):
                if self._VERBOSE:
                    print("Asking for the first batch")
                future = rpc.rpc_async(
                    collector_infos[i],
                    collector_class.next,
                    args=(collector_rrefs[i],),
                )
                futures.append((future, i))
        self.futures = futures
        self.collector_rrefs = collector_rrefs
        self.collector_infos = collector_infos

    def _init_worker_rpc(self, executor, i):
        """Init RPC node if necessary."""
        visible_device = (
            self.visible_devices[i] if self.visible_devices is not None else None
        )
        if self.launcher == "submitit":
            if not _has_submitit:
                raise ImportError("submitit not found.") from SUBMITIT_ERR
            job = executor.submit(
                _rpc_init_collection_node,
                i + 1,
                self.IPAddr,
                self.tcp_port,
                self.num_workers + 1,
                visible_device,
                self.tensorpipe_options,
                self._VERBOSE,
            )
            if self._VERBOSE:
                print("job id", job.job_id)  # ID of your job
            return job
        elif self.launcher == "mp":
            job = mp.Process(
                target=_rpc_init_collection_node,
                args=(
                    i + 1,
                    self.IPAddr,
                    self.tcp_port,
                    self.num_workers + 1,
                    visible_device,
                    self.tensorpipe_options,
                    self._VERBOSE,
                ),
            )
            job.start()
            return job
        elif self.launcher == "submitit_delayed":
            # job is already launched
            return None
        else:
            raise NotImplementedError(f"Unknown launcher {self.launcher}")

    def _init(self):
        self._shutdown = False
        if self.launcher == "submitit":
            executor = submitit.AutoExecutor(folder="log_test")
            executor.update_parameters(**self.slurm_kwargs)
        else:
            executor = None

        hostname = socket.gethostname()
        if self.launcher != "mp":
            IPAddr = socket.gethostbyname(hostname)
        else:
            IPAddr = "localhost"
        self.IPAddr = IPAddr

        os.environ["MASTER_ADDR"] = str(self.IPAddr)
        os.environ["MASTER_PORT"] = str(self.tcp_port)

        self.jobs = []
        for i in range(self.num_workers):
            if self._VERBOSE:
                print(f"Submitting job {i}")
            job = self._init_worker_rpc(
                executor,
                i,
            )
            self.jobs.append(job)

        self._init_master_rpc(
            self.num_workers + 1,
        )
        self._start_workers(
            world_size=self.num_workers + 1,
            env_constructors=self.env_constructors,
            collector_class=self.collector_class,
            num_workers_per_collector=self.num_workers_per_collector,
            policy=self.policy,
            frames_per_batch=self._frames_per_batch_corrected,
            total_frames=self.total_frames,
            collector_kwargs=self.collector_kwargs,
        )

    def iterator(self):
        self._collected_frames = 0
        while self._collected_frames < self.total_frames:
            if self._sync:
                data = self._next_sync_rpc()
            else:
                data = self._next_async_rpc()

            if self.split_trajs:
                data = split_trajectories(data)
            if self.postproc is not None:
                data = self.postproc(data)
            yield data

            if self.max_weight_update_interval > -1 and not self._sync:
                for j in range(self.num_workers):
                    if (
                        self._batches_since_weight_update[j]
                        > self.max_weight_update_interval
                    ):
                        self.update_policy_weights_([j], wait=False)
            elif self.max_weight_update_interval > -1:
                ranks = [
                    1
                    for j in range(self.num_workers)
                    if self._batches_since_weight_update[j]
                    > self.max_weight_update_interval
                ]
                self.update_policy_weights_(ranks, wait=True)

    def update_policy_weights_(self, workers=None, wait=True) -> None:
        if workers is None:
            workers = list(range(self.num_workers))
        futures = []
        for i in workers:
            if self._VERBOSE:
                print(f"calling update on worker {i}")
            futures.append(
                rpc.rpc_async(
                    self.collector_infos[i],
                    self.collector_class.update_policy_weights_,
                    args=(self.collector_rrefs[i], self.policy_weights.detach()),
                )
            )
        if wait:
            for i in workers:
                if self._VERBOSE:
                    print(f"waiting for worker {i}")
                futures[i].wait()
                if self._VERBOSE:
                    print("got it!")

    def _next_async_rpc(self):
        if self._VERBOSE:
            print("next async")
        if not len(self.futures):
            raise StopIteration(
                f"The queue is empty, the collector has ran out of data after {self._collected_frames} collected frames."
            )
        while True:
            future, i = self.futures.popleft()
            if future.done():
                if self.update_after_each_batch:
                    self.update_policy_weights_(workers=(i,), wait=False)
                if self._VERBOSE:
                    print(f"future {i} is done")
                data = future.value()
                self._collected_frames += data.numel()
                if self._collected_frames < self.total_frames:
                    future = rpc.rpc_async(
                        self.collector_infos[i],
                        self.collector_class.next,
                        args=(self.collector_rrefs[i],),
                    )
                    self.futures.append((future, i))
                return data.to(self.storing_device)
            self.futures.append((future, i))

    def _next_sync_rpc(self):
        if self._VERBOSE:
            print("next sync: futures")
        if self.update_after_each_batch:
            self.update_policy_weights_()
        for i in range(self.num_workers):
            future = rpc.rpc_async(
                self.collector_infos[i],
                self.collector_class.next,
                args=(self.collector_rrefs[i],),
            )
            self.futures.append((future, i))
        data = []
        while len(self.futures):
            future, i = self.futures.popleft()
            # the order is NOT guaranteed: should we change that?
            if future.done():
                data += [future.value()]
                if self._VERBOSE:
                    print(
                        f"got data from {i} // data has len {len(data)} / {self.num_workers}"
                    )
            else:
                self.futures.append((future, i))
        data = torch.cat(data).to(self.storing_device)
        traj_ids = data.get(("collector", "traj_ids"), None)
        if traj_ids is not None:
            for i in range(1, self.num_workers):
                traj_ids[i] += traj_ids[i - 1].max()
            data.set_(("collector", "traj_ids"), traj_ids)
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
        if not hasattr(self, "_shutdown"):
            warnings.warn("shutdown has no effect has `_init` has not been called yet.")
            return
        if self._shutdown:
            return
        if self._VERBOSE:
            print("shutting down")
        for future, i in self.futures:
            # clear the futures
            while future is not None and not future.done():
                print(f"waiting for proc {i} to clear")
                future.wait()
        for i in range(self.num_workers):
            if self._VERBOSE:
                print(f"shutting down {i}")
            rpc.rpc_sync(
                self.collector_infos[i],
                self.collector_class.shutdown,
                args=(self.collector_rrefs[i],),
                timeout=int(IDLE_TIMEOUT),
            )
        if self._VERBOSE:
            print("rpc shutdown")
        rpc.shutdown(timeout=int(IDLE_TIMEOUT))
        if self.launcher == "mp":
            for job in self.jobs:
                job.join(int(IDLE_TIMEOUT))
        elif self.launcher == "submitit":
            for job in self.jobs:
                _ = job.result()
        elif self.launcher == "submitit_delayed":
            pass
        else:
            raise NotImplementedError(f"Unknown launcher {self.launcher}")
        self._shutdown = True
