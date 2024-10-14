# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

r"""Generic distributed data-collector using torch.distributed backend."""
from __future__ import annotations

import os
import socket
import warnings
from copy import copy, deepcopy
from datetime import timedelta
from typing import Callable, List, OrderedDict

import torch.cuda
from tensordict import TensorDict
from torch import nn

from torchrl._utils import _ProcessNoWarn, logger as torchrl_logger, VERBOSE

from torchrl.collectors import MultiaSyncDataCollector
from torchrl.collectors.collectors import (
    DataCollectorBase,
    DEFAULT_EXPLORATION_TYPE,
    MultiSyncDataCollector,
    SyncDataCollector,
)
from torchrl.collectors.distributed.default_configs import (
    DEFAULT_SLURM_CONF,
    MAX_TIME_TO_CONNECT,
)
from torchrl.collectors.utils import _NON_NN_POLICY_WEIGHTS, split_trajectories
from torchrl.data.utils import CloudpickleWrapper
from torchrl.envs.common import EnvBase
from torchrl.envs.env_creator import EnvCreator

SUBMITIT_ERR = None
try:
    import submitit

    _has_submitit = True
except ModuleNotFoundError as err:
    _has_submitit = False
    SUBMITIT_ERR = err


def _distributed_init_collection_node(
    rank,
    rank0_ip,
    tcpport,
    world_size,
    backend,
    collector_class,
    num_workers,
    env_make,
    policy,
    frames_per_batch,
    collector_kwargs,
    update_interval,
    total_frames,
    verbose=VERBOSE,
):
    os.environ["MASTER_ADDR"] = str(rank0_ip)
    os.environ["MASTER_PORT"] = str(tcpport)

    if verbose:
        torchrl_logger.info(
            f"node with rank {rank} -- creating collector of type {collector_class}"
        )
    if not issubclass(collector_class, SyncDataCollector):
        env_make = [env_make] * num_workers
    else:
        collector_kwargs["return_same_td"] = True
        if num_workers != 1:
            raise RuntimeError(
                "SyncDataCollector and subclasses can only support a single environment."
            )

    if isinstance(policy, nn.Module):
        policy_weights = TensorDict.from_module(policy)
        policy_weights = policy_weights.data.lock_()
    else:
        warnings.warn(_NON_NN_POLICY_WEIGHTS)
        policy_weights = TensorDict(lock=True)

    collector = collector_class(
        env_make,
        policy,
        frames_per_batch=frames_per_batch,
        split_trajs=False,
        total_frames=total_frames,
        **collector_kwargs,
    )

    torchrl_logger.info(f"IP address: {rank0_ip} \ttcp port: {tcpport}")
    if verbose:
        torchrl_logger.info(f"node with rank {rank} -- launching distributed")
    torch.distributed.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(MAX_TIME_TO_CONNECT),
        # init_method=f"tcp://{rank0_ip}:{tcpport}",
    )
    if verbose:
        torchrl_logger.info(f"node with rank {rank} -- creating store")
    if verbose:
        torchrl_logger.info(f"node with rank {rank} -- loop")
    policy_weights.irecv(0)
    frames = 0
    for i, data in enumerate(collector):
        data.isend(dst=0)
        frames += data.numel()
        if (
            frames < total_frames
            and (i + 1) % update_interval == 0
            and not policy_weights.is_empty()
        ):
            policy_weights.irecv(0)

    if not collector.closed:
        collector.shutdown()
    del collector
    return


class DistributedSyncDataCollector(DataCollectorBase):
    """A distributed synchronous data collector with torch.distributed backend.

    Args:
        create_env_fn (Callable or List[Callabled]): list of Callables, each returning an
            instance of :class:`~torchrl.envs.EnvBase`.
        policy (Callable): Policy to be executed in the environment.
            Must accept :class:`tensordict.tensordict.TensorDictBase` object as input.
            If ``None`` is provided, the policy used will be a
            :class:`~torchrl.collectors.RandomPolicy` instance with the environment
            ``action_spec``.
            Accepted policies are usually subclasses of :class:`~tensordict.nn.TensorDictModuleBase`.
            This is the recommended usage of the collector.
            Other callables are accepted too:
            If the policy is not a ``TensorDictModuleBase`` (e.g., a regular :class:`~torch.nn.Module`
            instances) it will be wrapped in a `nn.Module` first.
            Then, the collector will try to assess if these
            modules require wrapping in a :class:`~tensordict.nn.TensorDictModule` or not.
            - If the policy forward signature matches any of ``forward(self, tensordict)``,
              ``forward(self, td)`` or ``forward(self, <anything>: TensorDictBase)`` (or
              any typing with a single argument typed as a subclass of ``TensorDictBase``)
              then the policy won't be wrapped in a :class:`~tensordict.nn.TensorDictModule`.
            - In all other cases an attempt to wrap it will be undergone as such: ``TensorDictModule(policy, in_keys=env_obs_key, out_keys=env.action_keys)``.

    Keyword Args:
        frames_per_batch (int): A keyword-only argument representing the total
            number of elements in a batch.
        total_frames (int): A keyword-only argument representing the total
            number of frames returned by the collector
            during its lifespan. If the ``total_frames`` is not divisible by
            ``frames_per_batch``, an exception is raised.
             Endless collectors can be created by passing ``total_frames=-1``.
             Defaults to ``-1`` (endless collector).
        device (int, str or torch.device, optional): The generic device of the
            collector. The ``device`` args fills any non-specified device: if
            ``device`` is not ``None`` and any of ``storing_device``, ``policy_device`` or
            ``env_device`` is not specified, its value will be set to ``device``.
            Defaults to ``None`` (No default device).
            Lists of devices are supported.
        storing_device (int, str or torch.device, optional): The *remote* device on which
            the output :class:`~tensordict.TensorDict` will be stored.
            If ``device`` is passed and ``storing_device`` is ``None``, it will
            default to the value indicated by ``device``.
            For long trajectories, it may be necessary to store the data on a different
            device than the one where the policy and env are executed.
            Defaults to ``None`` (the output tensordict isn't on a specific device,
            leaf tensors sit on the device where they were created).
            Lists of devices are supported.
        env_device (int, str or torch.device, optional): The *remote* device on which
            the environment should be cast (or executed if that functionality is
            supported). If not specified and the env has a non-``None`` device,
            ``env_device`` will default to that value. If ``device`` is passed
            and ``env_device=None``, it will default to ``device``. If the value
            as such specified of ``env_device`` differs from ``policy_device``
            and one of them is not ``None``, the data will be cast to ``env_device``
            before being passed to the env (i.e., passing different devices to
            policy and env is supported). Defaults to ``None``.
            Lists of devices are supported.
        policy_device (int, str or torch.device, optional): The *remote* device on which
            the policy should be cast.
            If ``device`` is passed and ``policy_device=None``, it will default
            to ``device``. If the value as such specified of ``policy_device``
            differs from ``env_device`` and one of them is not ``None``,
            the data will be cast to ``policy_device`` before being passed to
            the policy (i.e., passing different devices to policy and env is
            supported). Defaults to ``None``.
            Lists of devices are supported.
        max_frames_per_traj (int, optional): Maximum steps per trajectory.
            Note that a trajectory can span across multiple batches (unless
            ``reset_at_each_iter`` is set to ``True``, see below).
            Once a trajectory reaches ``n_steps``, the environment is reset.
            If the environment wraps multiple environments together, the number
            of steps is tracked for each environment independently. Negative
            values are allowed, in which case this argument is ignored.
            Defaults to ``None`` (i.e., no maximum number of steps).
        init_random_frames (int, optional): Number of frames for which the
            policy is ignored before it is called. This feature is mainly
            intended to be used in offline/model-based settings, where a
            batch of random trajectories can be used to initialize training.
            If provided, it will be rounded up to the closest multiple of frames_per_batch.
            Defaults to ``None`` (i.e. no random frames).
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
        exploration_type (ExplorationType, optional): interaction mode to be used when
            collecting data. Must be one of ``torchrl.envs.utils.ExplorationType.DETERMINISTIC``,
            ``torchrl.envs.utils.ExplorationType.RANDOM``, ``torchrl.envs.utils.ExplorationType.MODE``
            or ``torchrl.envs.utils.ExplorationType.MEAN``.
        collector_class (type or str, optional): a collector class for the remote node. Can be
            :class:`~torchrl.collectors.SyncDataCollector`,
            :class:`~torchrl.collectors.MultiSyncDataCollector`,
            :class:`~torchrl.collectors.MultiaSyncDataCollector`
            or a derived class of these. The strings "single", "sync" and
            "async" correspond to respective class.
            Defaults to :class:`~torchrl.collectors.SyncDataCollector`.
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
        slurm_kwargs (dict): a dictionary of parameters to be passed to the
            submitit executor.
        backend (str, optional): must a string "<distributed_backed>" where
            <distributed_backed> is one of ``"gloo"``, ``"mpi"``, ``"nccl"`` or ``"ucc"``. See
            the torch.distributed documentation for more information.
            Defaults to ``"gloo"``.
        max_weight_update_interval (int, optional): the maximum number of
            batches that can be collected before the policy weights of a worker
            is updated.
            For sync collections, this parameter is overwritten by ``update_after_each_batch``.
            For async collections, it may be that one worker has not seen its
            parameters being updated for a certain time even if ``update_after_each_batch``
            is turned on.
            Defaults to -1 (no forced update).
        update_interval (int, optional): the frequency at which the policy is
            updated. Defaults to 1.
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
        *,
        frames_per_batch: int,
        total_frames: int = -1,
        device: torch.device | List[torch.device] = None,
        storing_device: torch.device | List[torch.device] = None,
        env_device: torch.device | List[torch.device] = None,
        policy_device: torch.device | List[torch.device] = None,
        max_frames_per_traj: int = -1,
        init_random_frames: int = -1,
        reset_at_each_iter: bool = False,
        postproc: Callable | None = None,
        split_trajs: bool = False,
        exploration_type: "ExporationType" = DEFAULT_EXPLORATION_TYPE,  # noqa
        collector_class=SyncDataCollector,
        collector_kwargs=None,
        num_workers_per_collector=1,
        slurm_kwargs=None,
        backend="gloo",
        max_weight_update_interval=-1,
        update_interval=1,
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
            policy_weights = TensorDict.from_module(policy)
            policy_weights = policy_weights.data.lock_()
        else:
            warnings.warn(_NON_NN_POLICY_WEIGHTS)
            policy_weights = TensorDict(lock=True)

        self.policy_weights = policy_weights
        self.num_workers = len(create_env_fn)
        self.frames_per_batch = frames_per_batch

        self.device = device
        self.storing_device = storing_device
        self.env_device = env_device
        self.policy_device = policy_device

        self.storing_device = storing_device
        # make private to avoid changes from users during collection
        self.update_interval = update_interval
        self.total_frames_per_collector = total_frames // self.num_workers
        if self.total_frames_per_collector * self.num_workers != total_frames:
            raise RuntimeError(
                f"Cannot dispatch {total_frames} frames across {self.num_workers}. "
                f"Consider using a number of frames that is divisible by the number of workers."
            )
        self.max_weight_update_interval = max_weight_update_interval
        self.launcher = launcher
        self._batches_since_weight_update = [0 for _ in range(self.num_workers)]
        if tcp_port is None:
            self.tcp_port = os.environ.get("TCP_PORT", "10003")
        else:
            self.tcp_port = str(tcp_port)

        if self.frames_per_batch % self.num_workers != 0:
            raise RuntimeError(
                f"Cannot dispatch {self.frames_per_batch} frames across {self.num_workers}. "
                f"Consider using a number of frames per batch that is divisible by the number of workers."
            )
        self._frames_per_batch_corrected = self.frames_per_batch // self.num_workers

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
        for i, collector_kwarg in enumerate(self.collector_kwargs):
            collector_kwarg["max_frames_per_traj"] = max_frames_per_traj
            collector_kwarg["init_random_frames"] = (
                init_random_frames // self.num_workers
            )
            collector_kwarg["reset_at_each_iter"] = reset_at_each_iter
            collector_kwarg["exploration_type"] = exploration_type
            collector_kwarg["device"] = self.device[i]
            collector_kwarg["storing_device"] = self.storing_device[i]
            collector_kwarg["env_device"] = self.env_device[i]
            collector_kwarg["policy_device"] = self.policy_device[i]

        self.postproc = postproc
        self.split_trajs = split_trajs

        self.backend = backend

        # os.environ['TP_SOCKET_IFNAME'] = 'lo'

        self._init_workers()
        self._make_container()

    @property
    def device(self) -> List[torch.device]:
        return self._device

    @property
    def storing_device(self) -> List[torch.device]:
        return self._storing_device

    @property
    def env_device(self) -> List[torch.device]:
        return self._env_device

    @property
    def policy_device(self) -> List[torch.device]:
        return self._policy_device

    @device.setter
    def device(self, value):
        if isinstance(value, (tuple, list)):
            if len(value) != self.num_workers:
                raise RuntimeError(
                    "The number of devices passed to the collector must match the number of workers."
                )
            self._device = value
        else:
            self._device = [value] * self.num_workers

    @storing_device.setter
    def storing_device(self, value):
        if isinstance(value, (tuple, list)):
            if len(value) != self.num_workers:
                raise RuntimeError(
                    "The number of devices passed to the collector must match the number of workers."
                )
            self._storing_device = value
        else:
            self._storing_device = [value] * self.num_workers

    @env_device.setter
    def env_device(self, value):
        if isinstance(value, (tuple, list)):
            if len(value) != self.num_workers:
                raise RuntimeError(
                    "The number of devices passed to the collector must match the number of workers."
                )
            self._env_device = value
        else:
            self._env_device = [value] * self.num_workers

    @policy_device.setter
    def policy_device(self, value):
        if isinstance(value, (tuple, list)):
            if len(value) != self.num_workers:
                raise RuntimeError(
                    "The number of devices passed to the collector must match the number of workers."
                )
            self._policy_device = value
        else:
            self._policy_device = [value] * self.num_workers

    def _init_master_dist(
        self,
        world_size,
        backend,
    ):
        TCP_PORT = self.tcp_port
        torchrl_logger.info("init master...")
        torch.distributed.init_process_group(
            backend,
            rank=0,
            world_size=world_size,
            timeout=timedelta(MAX_TIME_TO_CONNECT),
            init_method=f"tcp://{self.IPAddr}:{TCP_PORT}",
        )
        torchrl_logger.info("done")

    def _make_container(self):
        env_constructor = self.env_constructors[0]
        pseudo_collector = SyncDataCollector(
            env_constructor,
            self.policy,
            frames_per_batch=self._frames_per_batch_corrected,
            total_frames=self.total_frames,
            split_trajs=False,
        )
        for _data in pseudo_collector:
            break
        self._tensordict_out = _data.expand((self.num_workers, *_data.shape))
        self._single_tds = self._tensordict_out.unbind(0)
        self._tensordict_out.lock_()
        pseudo_collector.shutdown()
        del pseudo_collector

    def _init_worker_dist_submitit(self, executor, i):
        TCP_PORT = self.tcp_port
        env_make = self.env_constructors[i]
        if not isinstance(env_make, (EnvBase, EnvCreator)):
            env_make = CloudpickleWrapper(env_make)
        job = executor.submit(
            _distributed_init_collection_node,
            i + 1,
            self.IPAddr,
            int(TCP_PORT),
            self.num_workers + 1,
            self.backend,
            self.collector_class,
            self.num_workers_per_collector,
            env_make,
            self.policy,
            self._frames_per_batch_corrected,
            self.collector_kwargs[i],
            self.update_interval,
            self.total_frames_per_collector,
        )
        return job

    def _init_worker_dist_mp(self, i):
        TCP_PORT = self.tcp_port
        env_make = self.env_constructors[i]
        if not isinstance(env_make, (EnvBase, EnvCreator)):
            env_make = CloudpickleWrapper(env_make)
        job = _ProcessNoWarn(
            target=_distributed_init_collection_node,
            args=(
                i + 1,
                self.IPAddr,
                int(TCP_PORT),
                self.num_workers + 1,
                self.backend,
                self.collector_class,
                self.num_workers_per_collector,
                env_make,
                self.policy,
                self._frames_per_batch_corrected,
                self.collector_kwargs[i],
                self.update_interval,
                self.total_frames_per_collector,
            ),
        )
        job.start()
        return job

    def _init_workers(self):

        hostname = socket.gethostname()
        IPAddr = socket.gethostbyname(hostname)
        torchrl_logger.info(f"Server IP address: {IPAddr}")
        self.IPAddr = IPAddr
        os.environ["MASTER_ADDR"] = str(self.IPAddr)
        os.environ["MASTER_PORT"] = str(self.tcp_port)

        self.jobs = []
        if self.launcher == "submitit":
            if not _has_submitit:
                raise ImportError("submitit not found.") from SUBMITIT_ERR
            executor = submitit.AutoExecutor(folder="log_test")
            executor.update_parameters(**self.slurm_kwargs)
        for i in range(self.num_workers):
            torchrl_logger.info("Submitting job")
            if self.launcher == "submitit":
                job = self._init_worker_dist_submitit(
                    executor,
                    i,
                )
                torchrl_logger.info(f"job id {job.job_id}")  # ID of your job
            elif self.launcher == "mp":
                job = self._init_worker_dist_mp(
                    i,
                )
                torchrl_logger.info("job launched")
            self.jobs.append(job)
        self._init_master_dist(self.num_workers + 1, self.backend)

    def iterator(self):
        yield from self._iterator_dist()

    def _iterator_dist(self):

        total_frames = 0
        j = -1
        while total_frames < self.total_frames:
            j += 1
            if j % self.update_interval == 0 and not self.policy_weights.is_empty():
                for i in range(self.num_workers):
                    rank = i + 1
                    self.policy_weights.isend(rank)

            trackers = []
            for i in range(self.num_workers):
                rank = i + 1
                trackers.append(
                    self._single_tds[i].irecv(src=rank, return_premature=True)
                )
            for tracker in trackers:
                for _tracker in tracker:
                    _tracker.wait()

            data = self._tensordict_out.clone()
            traj_ids = data.get(("collector", "traj_ids"), None)
            if traj_ids is not None:
                for i in range(1, self.num_workers):
                    traj_ids[i] += traj_ids[i - 1].max()
                data.set_(("collector", "traj_ids"), traj_ids)
            total_frames += data.numel()
            if self.split_trajs:
                data = split_trajectories(data)
            if self.postproc is not None:
                data = self.postproc(data)
            yield data

    def update_policy_weights_(self, worker_rank=None) -> None:
        raise NotImplementedError

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        raise NotImplementedError

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    def shutdown(self):
        pass
