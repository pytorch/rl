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
from typing import Any, Callable, OrderedDict, Sequence

import torch.cuda
from tensordict import TensorDict, TensorDictBase
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
    TCP_PORT,
)
from torchrl.collectors.utils import _NON_NN_POLICY_WEIGHTS, split_trajectories
from torchrl.collectors.weight_update import WeightUpdaterBase
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


def _node_init_dist(rank, world_size, backend, rank0_ip, tcpport, verbose):
    os.environ["MASTER_ADDR"] = str(rank0_ip)
    os.environ["MASTER_PORT"] = str(tcpport)

    if verbose:
        torchrl_logger.info(
            f"Rank0 IP address: '{rank0_ip}' \ttcp port: '{tcpport}', backend={backend}."
        )
        torchrl_logger.info(
            f"node with rank {rank} with world_size {world_size} -- launching distributed"
        )
    torch.distributed.init_process_group(
        backend,
        rank=rank,
        world_size=world_size,
        timeout=timedelta(MAX_TIME_TO_CONNECT),
        init_method=f"tcp://{rank0_ip}:{tcpport}",
    )
    if verbose:
        torchrl_logger.info(f"Connected!\nNode with rank {rank} -- creating store")
    # The store carries instructions for the node
    _store = torch.distributed.TCPStore(
        host_name=rank0_ip,
        port=tcpport + 1,
        world_size=world_size,
        is_master=False,
        timeout=timedelta(10),
    )
    return _store


def _distributed_init_delayed(
    rank,
    backend,
    rank0_ip,
    tcpport,
    world_size,
    verbose=False,
):
    """Initializer for contexts where jobs cannot be launched from main node.

    This function will wait for the main worker to send the launch command.
    """
    _store = _node_init_dist(rank, world_size, backend, rank0_ip, tcpport, verbose)
    # wait...
    objects = [
        None,
    ] * world_size
    output_list = [None]
    torch.distributed.scatter_object_list(output_list, objects, src=0)
    output = output_list[0]
    sync = output["sync"]
    collector_class = output["collector_class"]
    num_workers = output["num_workers"]
    env_make = output["env_make"]
    policy = output["policy"]
    frames_per_batch = output["frames_per_batch"]
    collector_kwargs = output["collector_kwargs"]
    _run_collector(
        _store,
        sync,
        collector_class,
        num_workers,
        env_make,
        policy,
        frames_per_batch,
        collector_kwargs,
        verbose=verbose,
    )


def _distributed_init_collection_node(
    rank,
    rank0_ip,
    tcpport,
    sync,
    world_size,
    backend,
    collector_class,
    num_workers,
    env_make,
    policy,
    policy_factory,
    frames_per_batch,
    collector_kwargs,
    verbose=True,
):
    _store = _node_init_dist(rank, world_size, backend, rank0_ip, tcpport, verbose)
    _run_collector(
        _store,
        sync,
        collector_class,
        num_workers,
        env_make,
        policy,
        policy_factory,
        frames_per_batch,
        collector_kwargs,
        verbose=verbose,
    )


def _run_collector(
    _store,
    sync,
    collector_class,
    num_workers,
    env_make,
    policy,
    policy_factory,
    frames_per_batch,
    collector_kwargs,
    verbose=True,
):
    rank = torch.distributed.get_rank()
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
        if collector_kwargs.get("weight_updater") is None and (
            policy_factory is None
            or (isinstance(policy_factory, Sequence) and not any(policy_factory))
        ):
            warnings.warn(_NON_NN_POLICY_WEIGHTS)
        policy_weights = TensorDict(lock=True)

    collector = collector_class(
        env_make,
        policy,
        policy_factory=policy_factory,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        split_trajs=False,
        **collector_kwargs,
    )
    total_frames = 0
    if verbose:
        torchrl_logger.info(f"node with rank {rank} -- loop")
    while True:
        instruction = _store.get(f"NODE_{rank}_in")
        if verbose:
            torchrl_logger.info(
                f"node with rank {rank} -- new instruction: {instruction}"
            )
        _store.delete_key(f"NODE_{rank}_in")
        if instruction == b"continue":
            _store.set(f"NODE_{rank}_status", b"busy")
            if verbose:
                torchrl_logger.info(f"node with rank {rank} -- new data")
            data = collector.next()
            total_frames += data.numel()
            if verbose:
                torchrl_logger.info(f"got data, total frames = {total_frames}")
                torchrl_logger.info(f"node with rank {rank} -- sending {data}")
            if _store.get("TRAINER_status") == b"alive":
                data.isend(dst=0)
                if verbose:
                    torchrl_logger.info(f"node with rank {rank} -- setting to 'done'")
                if not sync:
                    _store.set(f"NODE_{rank}_status", b"done")
        elif instruction == b"shutdown":
            if verbose:
                torchrl_logger.info(f"node with rank {rank} -- shutting down")
            try:
                collector.shutdown()
            except Exception:
                pass
            _store.set(f"NODE_{rank}_out", b"down")
            break
        elif instruction == b"update_weights":
            if sync:
                policy_weights.recv(0)
            else:
                # without further arguments, irecv blocks until weights have
                # been updated
                policy_weights.irecv(0)
            # the policy has been updated: we can simply update the weights
            collector.update_policy_weights_(policy_weights=policy_weights)
            _store.set(f"NODE_{rank}_out", b"updated")
        elif instruction.startswith(b"seeding"):
            seed = int(instruction.split(b"seeding_"))
            new_seed = collector.set_seed(seed)
            _store.set(f"NODE_{rank}_out", b"seeded")
            _store.set(f"NODE_{rank}_seed", str(new_seed).encode("utf-8"))
        else:
            raise RuntimeError(f"Instruction {instruction} is not recognised")
    if not collector.closed:
        collector.shutdown()
    del collector
    return


class DistributedDataCollector(DataCollectorBase):
    """A distributed data collector with torch.distributed backend.

    Supports sync and async data collection.

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

            .. note:: If the policy needs to be passed as a policy factory (e.g., in case it mustn't be serialized /
                pickled directly), the :arg:`policy_factory` should be used instead.

    Keyword Args:
        policy_factory (Callable[[], Callable], list of Callable[[], Callable], optional): a callable
            (or list of callables) that returns a policy instance. This is exclusive with the `policy` argument.

            .. note:: `policy_factory` comes in handy whenever the policy cannot be serialized.

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
        collector_class (Type or str, optional): a collector class for the remote node. Can be
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
        sync (bool, optional): if ``True``, the resulting tensordict is a stack of all the
            tensordicts collected on each node. If ``False`` (default), each
            tensordict results from a separate node in a "first-ready,
            first-served" fashion.
        slurm_kwargs (dict): a dictionary of parameters to be passed to the
            submitit executor.
        backend (str, optional): must a string "<distributed_backed>" where
            <distributed_backed> is one of ``"gloo"``, ``"mpi"``, ``"nccl"`` or ``"ucc"``. See
            the torch.distributed documentation for more information.
            Defaults to ``"gloo"``.
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
            Can be one of "submitit" or "mp" for multiprocessing.
            Use "submitit_delayed" if your cluster does not support spawning
            jobs from existing jobs.
            The former can launch jobs across multiple nodes, whilst the latter will only
            launch jobs on a single machine. "submitit" requires the homonymous
            library to be installed.
            To find more about submitit, visit
            https://github.com/facebookincubator/submitit and check our examples
            to learn more.
            Defaults to ``"submitit"``.
        tcp_port (int, optional): the TCP port to be used. Defaults to 10003.
        weight_updater (WeightUpdaterBase or constructor, optional): An instance of :class:`~torchrl.collectors.WeightUpdaterBase`
            or its subclass, responsible for updating the policy weights on distributed inference workers.
            If not provided, a :class:`~torchrl.collectors.distributed.DistributedWeightUpdater` will be used by
            default, which handles weight synchronization across distributed workers.
            Consider using a constructor if the updater needs to be serialized.

    """

    _VERBOSE = VERBOSE  # for debugging

    def __init__(
        self,
        create_env_fn,
        policy: Callable[[TensorDictBase], TensorDictBase] | None = None,
        *,
        policy_factory: Callable[[], Callable]
        | list[Callable[[] | Callable]]
        | None = None,
        frames_per_batch: int,
        total_frames: int = -1,
        device: torch.device | list[torch.device] | None = None,
        storing_device: torch.device | list[torch.device] | None = None,
        env_device: torch.device | list[torch.device] | None = None,
        policy_device: torch.device | list[torch.device] | None = None,
        max_frames_per_traj: int = -1,
        init_random_frames: int = -1,
        reset_at_each_iter: bool = False,
        postproc: Callable | None = None,
        split_trajs: bool = False,
        exploration_type: ExporationType = DEFAULT_EXPLORATION_TYPE,  # noqa
        collector_class: type = SyncDataCollector,
        collector_kwargs: dict[str, Any] | None = None,
        num_workers_per_collector: int = 1,
        sync: bool = False,
        slurm_kwargs: dict[str, Any] | None = None,
        backend: str = "gloo",
        update_after_each_batch: bool = False,
        max_weight_update_interval: int = -1,
        launcher: str = "submitit",
        tcp_port: int | None = None,
        weight_updater: WeightUpdaterBase
        | Callable[[], WeightUpdaterBase]
        | None = None,
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
        if not isinstance(policy_factory, Sequence):
            policy_factory = [policy_factory for _ in range(len(self.env_constructors))]
        self.policy_factory = policy_factory
        if isinstance(policy, nn.Module):
            policy_weights = TensorDict.from_module(policy)
            policy_weights = policy_weights.data.lock_()
        elif any(policy_factory):
            policy_weights = None
            if weight_updater is None:
                raise RuntimeError(
                    "weight_updater must be passed along with " "a policy_factory."
                )
        else:
            if not any(policy_factory):
                warnings.warn(_NON_NN_POLICY_WEIGHTS)
            policy_weights = TensorDict(lock=True)
        self.policy_weights = policy_weights
        self.num_workers = len(create_env_fn)
        self.frames_per_batch = frames_per_batch
        self.requested_frames_per_batch = frames_per_batch

        self.device = device
        self.storing_device = storing_device
        self.env_device = env_device
        self.policy_device = policy_device

        # make private to avoid changes from users during collection
        self._sync = sync
        self.update_after_each_batch = update_after_each_batch
        self.max_weight_update_interval = max_weight_update_interval
        if self.update_after_each_batch and self.max_weight_update_interval > -1:
            raise RuntimeError(
                "Got conflicting update instructions: `update_after_each_batch` "
                "`max_weight_update_interval` are incompatible."
            )
        self.launcher = launcher
        self._batches_since_weight_update = [0 for _ in range(self.num_workers)]
        if tcp_port is None:
            self.tcp_port = os.environ.get("TCP_PORT", TCP_PORT)
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
        if weight_updater is None:
            weight_updater = DistributedWeightUpdater(
                store=self._store,
                policy_weights=self.policy_weights,
                num_workers=self.num_workers,
                sync=self._sync,
            )
        self.weight_updater = weight_updater

    @property
    def device(self) -> list[torch.device]:
        return self._device

    @property
    def storing_device(self) -> list[torch.device]:
        return self._storing_device

    @property
    def env_device(self) -> list[torch.device]:
        return self._env_device

    @property
    def policy_device(self) -> list[torch.device]:
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
        if self._VERBOSE:
            torchrl_logger.info(
                f"launching main node with tcp port '{self.tcp_port}' and "
                f"IP '{self.IPAddr}'. rank: 0, world_size: {world_size}, backend={backend}."
            )
        os.environ["MASTER_ADDR"] = str(self.IPAddr)
        os.environ["MASTER_PORT"] = str(self.tcp_port)

        TCP_PORT = self.tcp_port
        torch.distributed.init_process_group(
            backend,
            rank=0,
            world_size=world_size,
            timeout=timedelta(MAX_TIME_TO_CONNECT),
            init_method=f"tcp://{self.IPAddr}:{TCP_PORT}",
        )
        if self._VERBOSE:
            torchrl_logger.info("main initiated! Launching store...")
        self._store = torch.distributed.TCPStore(
            host_name=self.IPAddr,
            port=int(TCP_PORT) + 1,
            world_size=self.num_workers + 1,
            is_master=True,
            timeout=timedelta(10),
        )
        if self._VERBOSE:
            torchrl_logger.info("done. Setting status to 'alive'")
        self._store.set("TRAINER_status", b"alive")

    def _make_container(self):
        if self._VERBOSE:
            torchrl_logger.info("making container")
        env_constructor = self.env_constructors[0]
        kwargs = self.collector_kwargs[0]
        pseudo_collector = SyncDataCollector(
            env_constructor,
            policy=self.policy,
            policy_factory=self.policy_factory[0],
            frames_per_batch=self._frames_per_batch_corrected,
            total_frames=-1,
            split_trajs=False,
            **kwargs,
        )
        for _data in pseudo_collector:
            break
        if self._VERBOSE:
            torchrl_logger.info(f"got data {_data}")
            torchrl_logger.info("expanding...")
        self._tensordict_out = _data.expand((self.num_workers, *_data.shape))
        if self._VERBOSE:
            torchrl_logger.info("locking")
        if self._sync:
            self._tensordict_out.lock_()
            self._tensordict_out_unbind = self._tensordict_out.unbind(0)
            for td in self._tensordict_out_unbind:
                td.lock_()
        else:
            self._tensordict_out = self._tensordict_out.unbind(0)
            for td in self._tensordict_out:
                td.lock_()
        if self._VERBOSE:
            torchrl_logger.info("storage created:")
            torchrl_logger.info("shutting down...")
        pseudo_collector.shutdown()
        if self._VERBOSE:
            torchrl_logger.info("dummy collector shut down!")
        del pseudo_collector

    def _init_worker_dist_submitit(self, executor, i):
        env_make = self.env_constructors[i]
        if not isinstance(env_make, (EnvBase, EnvCreator)):
            env_make = CloudpickleWrapper(env_make)
        TCP_PORT = self.tcp_port
        job = executor.submit(
            _distributed_init_collection_node,
            i + 1,
            self.IPAddr,
            int(TCP_PORT),
            self._sync,
            self.num_workers + 1,
            self.backend,
            self.collector_class,
            self.num_workers_per_collector,
            env_make,
            self.policy,
            self.policy_factory[i],
            self._frames_per_batch_corrected,
            self.collector_kwargs[i],
            self._VERBOSE,
        )
        return job

    def _init_worker_dist_submitit_delayed(self):
        def get_env_make(i):
            env_make = self.env_constructors[i]
            if not isinstance(env_make, (EnvBase, EnvCreator)):
                env_make = CloudpickleWrapper(env_make)
            return env_make

        self._init_master_dist(self.num_workers + 1, self.backend)
        objects = [
            {
                "sync": self._sync,
                "collector_class": self.collector_class,
                "num_workers": self.num_workers_per_collector,
                "env_make": get_env_make(i),
                "policy": self.policy,
                "policy_factory": self.policy_factory[i],
                "frames_per_batch": self._frames_per_batch_corrected,
                "collector_kwargs": self.collector_kwargs[i],
            }
            for i in range(self.num_workers)
        ]
        objects = [None] + objects
        torch.distributed.scatter_object_list([None], objects, src=0)

    def _init_worker_dist_mp(self, i):
        env_make = self.env_constructors[i]
        if not isinstance(env_make, (EnvBase, EnvCreator)):
            env_make = CloudpickleWrapper(env_make)
        TCP_PORT = self.tcp_port
        job = _ProcessNoWarn(
            target=_distributed_init_collection_node,
            args=(
                i + 1,
                self.IPAddr,
                int(TCP_PORT),
                self._sync,
                self.num_workers + 1,
                self.backend,
                self.collector_class,
                self.num_workers_per_collector,
                env_make,
                self.policy,
                self.policy_factory[i],
                self._frames_per_batch_corrected,
                self.collector_kwargs[i],
                self._VERBOSE,
            ),
        )
        job.start()
        return job

    def _init_workers(self):

        if self.launcher != "mp":
            hostname = socket.gethostname()
            IPAddr = socket.gethostbyname(hostname)
        else:
            IPAddr = "localhost"
        if self._VERBOSE:
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
        if self.launcher == "submitit_delayed":
            self._init_worker_dist_submitit_delayed()
        else:
            for i in range(self.num_workers):
                if self._VERBOSE:
                    torchrl_logger.info("Submitting job")
                if self.launcher == "submitit":
                    job = self._init_worker_dist_submitit(
                        executor,
                        i,
                    )
                    if self._VERBOSE:
                        torchrl_logger.info(f"job id {job.job_id}")  # ID of your job
                elif self.launcher == "mp":
                    job = self._init_worker_dist_mp(
                        i,
                    )
                    if self._VERBOSE:
                        torchrl_logger.info("job launched")
                self.jobs.append(job)
            self._init_master_dist(self.num_workers + 1, self.backend)

    def iterator(self):
        yield from self._iterator_dist()

    def _iterator_dist(self):
        if self._VERBOSE:
            torchrl_logger.info("iterating...")

        total_frames = 0
        if not self._sync:
            for rank in range(1, self.num_workers + 1):
                if self._VERBOSE:
                    torchrl_logger.info(f"sending 'continue' to {rank}")
                self._store.set(f"NODE_{rank}_in", b"continue")
            trackers = []
            for i in range(self.num_workers):
                rank = i + 1
                trackers.append(
                    self._tensordict_out[i].irecv(src=rank, return_premature=True)
                )

        while total_frames < self.total_frames:
            if self._sync:
                data, total_frames = self._next_sync(total_frames)
            else:
                data, total_frames = self._next_async(total_frames, trackers)

            if self.split_trajs:
                data = split_trajectories(data)
            if self.postproc is not None:
                data = self.postproc(data)
            yield data

            if self.max_weight_update_interval > -1:
                for j in range(self.num_workers):
                    rank = j + 1
                    if (
                        self._batches_since_weight_update[j]
                        > self.max_weight_update_interval
                    ):
                        self.update_policy_weights_(
                            policy_weights=None, worker_ids=rank
                        )

        for i in range(self.num_workers):
            rank = i + 1
            if self._VERBOSE:
                torchrl_logger.info(f"shutting down rank {rank}.")
            self._store.set(f"NODE_{rank}_in", b"shutdown")

    def _next_sync(self, total_frames):
        # in the 'sync' case we should update before collecting the data
        if self.update_after_each_batch:
            self.update_policy_weights_()
        else:
            for j in range(self.num_workers):
                self._batches_since_weight_update[j] += 1

        if total_frames < self.total_frames:
            for rank in range(1, self.num_workers + 1):
                if self._VERBOSE:
                    torchrl_logger.info(f"sending 'continue' to {rank}")
                self._store.set(f"NODE_{rank}_in", b"continue")
        trackers = []
        for i in range(self.num_workers):
            rank = i + 1
            trackers.append(
                self._tensordict_out_unbind[i].irecv(src=rank, return_premature=True)
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
        return data, total_frames

    def _next_async(self, total_frames, trackers):
        data = None
        while data is None:
            for i in range(self.num_workers):
                rank = i + 1
                if self._store.get(f"NODE_{rank}_status") == b"done":
                    for _tracker in trackers[i]:
                        _tracker.wait()
                    data = self._tensordict_out[i].clone()
                    if self.update_after_each_batch:
                        self.update_policy_weights_(worker_ids=rank)
                    total_frames += data.numel()
                    if total_frames < self.total_frames:
                        if self._VERBOSE:
                            torchrl_logger.info(f"sending 'continue' to {rank}")
                        self._store.set(f"NODE_{rank}_in", b"continue")
                    trackers[i] = self._tensordict_out[i].irecv(
                        src=i + 1, return_premature=True
                    )
                    for j in range(self.num_workers):
                        self._batches_since_weight_update[j] += j != i
                    break
        return data, total_frames

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        for i in range(self.num_workers):
            rank = i + 1
            self._store.set(f"NODE_{rank}_in", f"seeding_{seed}".encode())
            status = self._store.get(f"NODE_{rank}_out")
            if status != b"updated":
                raise RuntimeError(f"Expected 'seeded' but got status {status}.")
            self._store.delete_key(f"NODE_{rank}_out")
            new_seed = self._store.get(f"NODE_{rank}_seed")
            self._store.delete_key(f"NODE_{rank}_seed")
            seed = int(new_seed)
        return seed

    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    def shutdown(self, timeout: float | None = None) -> None:
        self._store.set("TRAINER_status", b"shutdown")
        for i in range(self.num_workers):
            rank = i + 1
            if self._VERBOSE:
                torchrl_logger.info(f"shutting down node with rank={rank}")
            self._store.set(f"NODE_{rank}_in", b"shutdown")
        for i in range(self.num_workers):
            rank = i + 1
            if self._VERBOSE:
                torchrl_logger.info(f"getting status of node {rank}")
            status = self._store.get(f"NODE_{rank}_out")
            if status != b"down":
                raise RuntimeError(f"Expected 'down' but got status {status}.")
            self._store.delete_key(f"NODE_{rank}_out")
        for i in range(self.num_workers):
            if self.launcher == "mp":
                if not self.jobs[i].is_alive():
                    continue
                self.jobs[i].join(timeout=10)
            elif self.launcher == "submitit":
                self.jobs[i].result()
            elif self.launcher == "submitit_delayed":
                pass
        if self._VERBOSE:
            torchrl_logger.info("collector shut down")


class DistributedWeightUpdater(WeightUpdaterBase):
    """A remote weight updater for synchronizing policy weights across distributed workers.

    The `DistributedWeightUpdater` class provides a mechanism for updating the weights
    of a policy across distributed inference workers. It is designed to work with the
    :class:`~torchrl.collectors.distributed.DistributedDataCollector` to ensure that each worker receives the latest policy weights.
    This class is typically used in distributed data collection scenarios where multiple workers
    need to be kept in sync with the central policy weights.

    Args:
        store (dict[str, str]): A dictionary-like store used for communication between the server
            and the distributed workers.
        policy_weights (TensorDictBase): The current weights of the policy that need to be distributed
            to the workers.
        num_workers (int): The number of distributed workers that will receive the updated policy weights.
        sync (bool): if ``True``, the sync happens synchronously (the server waits for the worker to have completed
            the update to restart the run).

    Methods:
        update_weights: Updates the weights on specified or all distributed workers.
        all_worker_ids: Returns a list of all worker identifiers (not implemented in this class).
        _sync_weights_with_worker: Synchronizes the server weights with a specific worker (not implemented).
        _get_server_weights: Retrieves the latest weights from the server (not implemented).
        _maybe_map_weights: Optionally maps server weights before distribution (not implemented).

    .. note::
        This class assumes that the server weights can be directly applied to the distributed workers
        without any additional processing. If your use case requires more complex weight mapping or
        synchronization logic, consider extending `WeightUpdaterBase` with a custom implementation.

    Raises:
        RuntimeError: If the worker rank is less than 1 or if the status returned by the store is not "updated".

    .. seealso:: :class:`~torchrl.collectors.WeightUpdaterBase` and
        :class:`~torchrl.collectors.distributed.DistributedDataCollector`.

    """

    _VERBOSE = True

    def __init__(
        self,
        store: dict[str, str],
        policy_weights: TensorDictBase,
        num_workers: int,
        sync: bool,
    ):
        self._store = store
        self.policy_weights = policy_weights
        self.num_workers = num_workers
        self._sync = sync
        self._batches_since_weight_update = [0 for _ in range(self.num_workers)]

    def _sync_weights_with_worker(
        self, worker_id: int | torch.device, server_weights: TensorDictBase
    ) -> TensorDictBase:
        raise NotImplementedError

    def _get_server_weights(self) -> TensorDictBase:
        raise NotImplementedError

    def _maybe_map_weights(self, server_weights: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError

    def all_worker_ids(self) -> list[int] | list[torch.device]:
        raise NotImplementedError

    def push_weights(
        self,
        weights: TensorDictBase | None = None,
        worker_ids: torch.device | int | list[int] | list[torch.device] | None = None,
    ):
        worker_rank = worker_ids
        if isinstance(worker_ids, int):
            if worker_rank is not None and worker_rank < 1:
                raise RuntimeError("worker_rank must be greater than 1")
            worker_rank = [worker_rank - 1]
        workers = range(self.num_workers) if worker_rank is None else worker_rank
        weights = self.policy_weights if weights is None else weights
        for i in workers:
            rank = i + 1
            if self._VERBOSE:
                torchrl_logger.info(f"updating weights of {rank}")
            self._store.set(f"NODE_{rank}_in", b"update_weights")
            if self._sync:
                weights.send(rank)
            else:
                weights.isend(rank)
            self._batches_since_weight_update[i] = 0
            status = self._store.get(f"NODE_{rank}_out")
            if status != b"updated":
                raise RuntimeError(f"Expected 'updated' but got status {status}.")
            self._store.delete_key(f"NODE_{rank}_out")
