# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict, TensorDictBase

from torchrl._utils import as_remote, logger as torchrl_logger
from torchrl.collectors._base import DataCollectorBase
from torchrl.collectors._constants import DEFAULT_EXPLORATION_TYPE
from torchrl.collectors._multi_async import MultiaSyncDataCollector
from torchrl.collectors._multi_sync import MultiSyncDataCollector
from torchrl.collectors._single import SyncDataCollector
from torchrl.collectors.utils import _NON_NN_POLICY_WEIGHTS, split_trajectories
from torchrl.collectors.weight_update import RayWeightUpdater, WeightUpdaterBase
from torchrl.data import ReplayBuffer
from torchrl.envs.common import EnvBase
from torchrl.envs.env_creator import EnvCreator
from torchrl.weight_update.weight_sync_schemes import WeightSyncScheme

RAY_ERR = None
try:
    import ray
    from ray._private.services import get_node_ip_address

    _has_ray = True
except ImportError as err:
    _has_ray = False
    RAY_ERR = err

DEFAULT_RAY_INIT_CONFIG = {
    "address": None,
    "num_cpus": None,
    "num_gpus": None,
    "resources": None,
    "object_store_memory": None,
    "local_mode": False,
    "ignore_reinit_error": False,
    "include_dashboard": None,
    "dashboard_host": "127.0.0.1",
    "dashboard_port": None,
    "job_config": None,
    "configure_logging": True,
    "logging_level": "info",
    "logging_format": None,
    "log_to_driver": True,
    "namespace": None,
    "runtime_env": None,
}

DEFAULT_REMOTE_CLASS_CONFIG = {
    "num_cpus": 1,
    "num_gpus": 0.2 if torch.cuda.is_available() else None,
    "memory": 2 * 1024**3,
}


def print_remote_collector_info(self):
    """Prints some information about the remote collector."""
    s = (
        f"Created remote collector with in machine "
        f"{get_node_ip_address()} using gpus {ray.get_gpu_ids()}"
    )
    # torchrl_logger.warning(s)
    torchrl_logger.debug(s)


class RayCollector(DataCollectorBase):
    """Distributed data collector with `Ray <https://docs.ray.io/>`_ backend.

    This Python class serves as a ray-based solution to instantiate and coordinate multiple
    data collectors in a distributed cluster. Like TorchRL non-distributed collectors, this
    collector is an iterable that yields TensorDicts until a target number of collected
    frames is reached, but handles distributed data collection under the hood.

    The class dictionary input parameter "ray_init_config" can be used to provide the kwargs to
    call Ray initialization method ray.init(). If "ray_init_config" is not provided, the default
    behavior is to autodetect an existing Ray cluster or start a new Ray instance locally if no
    existing cluster is found. Refer to Ray documentation for advanced initialization kwargs.

    Similarly, dictionary input parameter "remote_configs" can be used to specify the kwargs for
    ray.remote() when called to create each remote collector actor, including collector compute
    resources.The sum of all collector resources should be available in the cluster. Refer to Ray
    documentation for advanced configuration of the ray.remote() method. Default kwargs are:

    >>> kwargs = {
    ...     "num_cpus": 1,
    ...     "num_gpus": 0.2,
    ...     "memory": 2 * 1024 ** 3,
    ... }


    The coordination between collector instances can be specified as "synchronous" or "asynchronous".
    In synchronous coordination, this class waits for all remote collectors to collect a rollout,
    concatenates all rollouts into a single TensorDict instance and finally yields the concatenated
    data. On the other hand, if the coordination is to be carried out asynchronously, this class
    provides the rollouts as they become available from individual remote collectors.

    Args:
        create_env_fn (Callable or List[Callabled]): list of Callables, each returning an
            instance of :class:`~torchrl.envs.EnvBase`.
        policy (Callable, optional): Policy to be executed in the environment.
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
                pickled directly), the ``policy_factory`` should be used instead.

    Keyword Args:
        policy_factory (Callable[[], Callable], list of Callable[[], Callable], optional): a callable
            (or list of callables) that returns a policy instance. This is exclusive with the `policy` argument.

            .. note:: `policy_factory` comes in handy whenever the policy cannot be serialized.

        trust_policy (bool, optional): if ``True``, a non-TensorDictModule policy will be trusted to be
            assumed to be compatible with the collector. This defaults to ``True`` for CudaGraphModules
            and ``False`` otherwise.
        frames_per_batch (int): A keyword-only argument representing the
            total number of elements in a batch.
        total_frames (int, Optional): lower bound of the total number of frames returned by the collector.
            The iterator will stop once the total number of frames equates or exceeds the total number of
            frames passed to the collector. Default value is -1, which mean no target total number of frames
            (i.e. the collector will run indefinitely).
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
        create_env_kwargs (dict, optional): Dictionary of kwargs for
            ``create_env_fn``.
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
        collector_class (Python class or constructor): a collector class to be remotely instantiated. Can be
            :class:`~torchrl.collectors.SyncDataCollector`,
            :class:`~torchrl.collectors.MultiSyncDataCollector`,
            :class:`~torchrl.collectors.MultiaSyncDataCollector`
            or a derived class of these.
            Defaults to :class:`~torchrl.collectors.SyncDataCollector`.
        collector_kwargs (dict or list, optional): a dictionary of parameters to be passed to the
            remote data-collector. If a list is provided, each element will
            correspond to an individual set of keyword arguments for the
            dedicated collector.
        num_workers_per_collector (int): the number of copies of the
            env constructor that is to be used on the remote nodes.
            Defaults to 1 (a single env per collector).
            On a single worker node all the sub-workers will be
            executing the same environment. If different environments need to
            be executed, they should be dispatched across worker nodes, not
            subnodes.
        ray_init_config (dict, Optional): kwargs used to call ray.init().
        remote_configs (list of dicts, Optional): ray resource specs for each remote collector.
            A single dict can be provided as well, and will be used in all collectors.
        num_collectors (int, Optional): total number of collectors to be instantiated.
        sync (bool): if ``True``, the resulting tensordict is a stack of all the
            tensordicts collected on each node. If ``False`` (default), each
            tensordict results from a separate node in a "first-ready,
            first-served" fashion.
        update_after_each_batch (bool, optional): if ``True``, the weights will
            be updated after each collection. For ``sync=True``, this means that
            all workers will see their weights updated. For ``sync=False``,
            only the worker from which the data has been gathered will be
            updated.
            This is equivalent to `max_weight_update_interval=0`.
            Defaults to ``False``, i.e. updates have to be executed manually
            through
            :meth:`torchrl.collectors.DataCollector.update_policy_weights_`
        max_weight_update_interval (int, optional): the maximum number of
            batches that can be collected before the policy weights of a worker
            is updated.
            For sync collections, this parameter is overwritten by ``update_after_each_batch``.
            For async collections, it may be that one worker has not seen its
            parameters being updated for a certain time even if ``update_after_each_batch``
            is turned on.
            Defaults to -1 (no forced update).
        replay_buffer (RayReplayBuffer, optional): if provided, the collector will not yield tensordicts
            but populate the buffer instead. Defaults to ``None``.

            .. note:: although it is not enfoced (to allow users to implement their own replay buffer class), a
                :class:`~torchrl.data.RayReplayBuffer` instance should be used here.
        weight_updater (WeightUpdaterBase or constructor, optional): (Deprecated) An instance of :class:`~torchrl.collectors.WeightUpdaterBase`
            or its subclass, responsible for updating the policy weights on remote inference workers managed by Ray.
            If not provided, a :class:`~torchrl.collectors.RayWeightUpdater` will be used by default, leveraging
            Ray's distributed capabilities.
            Consider using a constructor if the updater needs to be serialized.
        weight_sync_schemes (dict[str, WeightSyncScheme], optional): Dictionary of weight sync schemes for
            SENDING weights to remote collector workers. Keys are model identifiers (e.g., "policy")
            and values are WeightSyncScheme instances configured to send weights via Ray.
            This is the recommended way to configure weight synchronization for propagating weights
            from the main process to remote collectors. If not provided,
            defaults to ``{"policy": RayWeightSyncScheme()}``.

            .. note:: Weight synchronization is lazily initialized. When using ``policy_factory``
                without a central ``policy``, weight sync is deferred until the first call to
                :meth:`~torchrl.collectors.DataCollector.update_policy_weights_` with actual weights.
                This allows sub-collectors to each have their own independent policies created via
                the factory. If you have a central policy and want to sync its weights to remote
                collectors, call ``update_policy_weights_(policy)`` before starting iteration.

        weight_recv_schemes (dict[str, WeightSyncScheme], optional): Dictionary of weight sync schemes for
            RECEIVING weights from a parent process or training loop. Keys are model identifiers (e.g., "policy")
            and values are WeightSyncScheme instances configured to receive weights.
            This is typically used when RayCollector is itself a worker in a larger distributed setup.
            Defaults to ``None``.
        use_env_creator (bool, optional): if ``True``, the environment constructor functions will be wrapped
            in :class:`~torchrl.envs.EnvCreator`. This is useful for multiprocessed settings where shared memory
            needs to be managed, but Ray has its own object storage mechanism, so this is typically not needed.
            Defaults to ``False``.

    Examples:
        >>> from torch import nn
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.collectors import SyncDataCollector
        >>> from torchrl.collectors.distributed import RayCollector
        >>> env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
        >>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
        >>> distributed_collector = RayCollector(
        ...     create_env_fn=[env_maker],
        ...     policy=policy,
        ...     collector_class=SyncDataCollector,
        ...     max_frames_per_traj=50,
        ...     init_random_frames=-1,
        ...     reset_at_each_iter=-False,
        ...     collector_kwargs={
        ...         "device": "cpu",
        ...         "storing_device": "cpu",
        ...     },
        ...     num_collectors=1,
        ...     total_frames=10000,
        ...     frames_per_batch=200,
        ... )
        >>> for i, data in enumerate(collector):
        ...     if i == 2:
        ...         print(data)
        ...         break
    """

    def __init__(
        self,
        create_env_fn: Callable | EnvBase | list[Callable] | list[EnvBase],
        policy: Callable[[TensorDictBase], TensorDictBase] | None = None,
        *,
        policy_factory: Callable[[], Callable]
        | list[Callable[[], Callable]]
        | None = None,
        trust_policy: bool | None = None,
        frames_per_batch: int,
        total_frames: int = -1,
        device: torch.device | list[torch.device] | None = None,
        storing_device: torch.device | list[torch.device] | None = None,
        env_device: torch.device | list[torch.device] | None = None,
        policy_device: torch.device | list[torch.device] | None = None,
        max_frames_per_traj=-1,
        init_random_frames=-1,
        reset_at_each_iter=False,
        postproc=None,
        split_trajs=False,
        exploration_type=DEFAULT_EXPLORATION_TYPE,
        collector_class: Callable[[TensorDict], TensorDict] = SyncDataCollector,
        collector_kwargs: dict[str, Any] | list[dict] | None = None,
        num_workers_per_collector: int = 1,
        sync: bool = False,
        ray_init_config: dict[str, Any] | None = None,
        remote_configs: dict[str, Any] | list[dict[str, Any]] | None = None,
        num_collectors: int | None = None,
        update_after_each_batch: bool = False,
        max_weight_update_interval: int = -1,
        replay_buffer: ReplayBuffer | None = None,
        weight_updater: WeightUpdaterBase
        | Callable[[], WeightUpdaterBase]
        | None = None,
        weight_sync_schemes: dict[str, WeightSyncScheme] | None = None,
        weight_recv_schemes: dict[str, WeightSyncScheme] | None = None,
        use_env_creator: bool = False,
        no_cuda_sync: bool | None = None,
    ):
        self.frames_per_batch = frames_per_batch
        if remote_configs is None:
            remote_configs = DEFAULT_REMOTE_CLASS_CONFIG

        if ray_init_config is None:
            ray_init_config = DEFAULT_RAY_INIT_CONFIG

        if collector_kwargs is None:
            collector_kwargs = {}
        if replay_buffer is not None:
            if isinstance(collector_kwargs, dict):
                collector_kwargs.setdefault("replay_buffer", replay_buffer)
            else:
                collector_kwargs = [
                    ck.setdefault("replay_buffer", replay_buffer)
                    for ck in collector_kwargs
                ]

        # Make sure input parameters are consistent
        def check_consistency_with_num_collectors(param, param_name, num_collectors):
            """Checks that if param is a list, it has length num_collectors."""
            if isinstance(param, list):
                if len(param) != num_collectors:
                    raise ValueError(
                        f"Inconsistent RayDistributedCollector parameters, {param_name} is a list of length "
                        f"{len(param)} but the specified number of collectors is {num_collectors}."
                    )
            else:
                param = [param] * num_collectors
            return param

        if num_collectors:
            create_env_fn = check_consistency_with_num_collectors(
                create_env_fn, "create_env_fn", num_collectors
            )
            collector_kwargs = check_consistency_with_num_collectors(
                collector_kwargs, "collector_kwargs", num_collectors
            )
            remote_configs = check_consistency_with_num_collectors(
                remote_configs, "remote_config", num_collectors
            )

        def check_list_length_consistency(*lists):
            """Checks that all input lists have the same length.

            If any non-list input is given, it is converted to a list
            of the same length as the others by repeating the same
            element multiple times.
            """
            lengths = set()
            new_lists = []
            for lst in lists:
                if isinstance(lst, list):
                    lengths.add(len(lst))
                    new_lists.append(lst)
                else:
                    new_lst = [lst] * max(lengths)
                    new_lists.append(new_lst)
                    lengths.add(len(new_lst))
            if len(lengths) > 1:
                raise ValueError(
                    "Inconsistent RayDistributedCollector parameters. create_env_fn, "
                    "collector_kwargs and remote_configs are lists of different length."
                )
            else:
                return new_lists

        out_lists = check_list_length_consistency(
            create_env_fn, collector_kwargs, remote_configs
        )
        create_env_fn, collector_kwargs, remote_configs = out_lists
        num_collectors = len(create_env_fn)

        if use_env_creator:
            for i in range(len(create_env_fn)):
                if not isinstance(create_env_fn[i], (EnvBase, EnvCreator)):
                    create_env_fn[i] = EnvCreator(create_env_fn[i])

        # If ray available, try to connect to an existing Ray cluster or start one and connect to it.
        if not _has_ray:
            raise RuntimeError(
                "ray library not found, unable to create a DistributedCollector. "
            ) from RAY_ERR
        if not ray.is_initialized():
            ray.init(**ray_init_config)
        if not ray.is_initialized():
            raise RuntimeError("Ray could not be initialized.")

        # Define collector_class, monkey patch it with as_remote and print_remote_collector_info methods
        if collector_class == "async":
            collector_class = MultiaSyncDataCollector
        elif collector_class == "sync":
            collector_class = MultiSyncDataCollector
        elif collector_class == "single":
            collector_class = SyncDataCollector
        elif not isinstance(collector_class, type) or not issubclass(
            collector_class, DataCollectorBase
        ):
            raise TypeError(
                "The collector_class must be an instance of DataCollectorBase."
            )
        if not hasattr(collector_class, "as_remote"):
            collector_class.as_remote = as_remote
        if not hasattr(collector_class, "print_remote_collector_info"):
            collector_class.print_remote_collector_info = print_remote_collector_info

        self.no_cuda_sync = no_cuda_sync
        self.replay_buffer = replay_buffer
        if not isinstance(policy_factory, Sequence):
            policy_factory = [policy_factory] * len(create_env_fn)
        self.policy_factory = policy_factory
        self.policy = policy  # Store policy for weight extraction
        self.trust_policy = trust_policy
        if isinstance(policy, nn.Module):
            policy_weights = TensorDict.from_module(policy)
            policy_weights = policy_weights.data.lock_()
        else:
            policy_weights = TensorDict(lock=True)
            if weight_updater is None:
                warnings.warn(_NON_NN_POLICY_WEIGHTS)
        self.policy_weights = policy_weights
        self.collector_class = collector_class
        self.collected_frames = 0
        self.split_trajs = split_trajs
        self.total_frames = total_frames
        self.num_collectors = num_collectors

        self.update_after_each_batch = update_after_each_batch
        self.max_weight_update_interval = max_weight_update_interval

        self.collector_kwargs = (
            collector_kwargs if collector_kwargs is not None else [{}]
        )
        self.device = device
        self.storing_device = storing_device
        self.env_device = env_device
        self.policy_device = policy_device
        self._batches_since_weight_update = [0 for _ in range(self.num_collectors)]
        self._sync = sync
        self._collection_thread = None
        self._stop_event = threading.Event()

        if self._sync:
            if frames_per_batch % self.num_collectors != 0:
                raise RuntimeError(
                    f"Cannot dispatch {frames_per_batch} frames across {self.num_collectors}. "
                    f"Consider using a number of frames per batch that is divisible by the number of workers."
                )
            self._frames_per_batch_corrected = frames_per_batch // self.num_collectors
        else:
            self._frames_per_batch_corrected = frames_per_batch

        # update collector kwargs
        for i, collector_kwarg in enumerate(self.collector_kwargs):
            # Don't pass policy_factory if we have a policy - remote collectors need the policy object
            # to be able to apply weight updates
            if policy is None:
                collector_kwarg["policy_factory"] = policy_factory[i]
            collector_kwarg["max_frames_per_traj"] = max_frames_per_traj
            collector_kwarg["init_random_frames"] = (
                init_random_frames // self.num_collectors
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
            collector_kwarg["split_trajs"] = False
            collector_kwarg["frames_per_batch"] = self._frames_per_batch_corrected
            collector_kwarg["device"] = self.device[i]
            collector_kwarg["storing_device"] = self.storing_device[i]
            collector_kwarg["env_device"] = self.env_device[i]
            collector_kwarg["policy_device"] = self.policy_device[i]
            if "trust_policy" not in collector_kwarg:
                collector_kwarg["trust_policy"] = self.trust_policy
            if "no_cuda_sync" not in collector_kwarg and self.no_cuda_sync is not None:
                collector_kwarg["no_cuda_sync"] = no_cuda_sync

        self.postproc = postproc

        # Create remote instances of the collector class
        self._remote_collectors = []
        if self.num_collectors > 0:
            self.add_collectors(
                create_env_fn,
                num_workers_per_collector,
                policy,
                collector_kwargs,
                remote_configs,
            )
        # Set up weight synchronization - prefer new schemes over legacy updater
        if weight_updater is None and weight_sync_schemes is None:
            # Default to Ray weight sync scheme for Ray collectors
            from torchrl.weight_update import RayWeightSyncScheme

            weight_sync_schemes = {"policy": RayWeightSyncScheme()}

        if weight_sync_schemes is not None:
            torchrl_logger.debug("RayCollector: Using weight sync schemes")
            # Use new weight synchronization system
            self._weight_sync_schemes = weight_sync_schemes

            # Initialize schemes on the sender (main process) side
            # Pass remote collectors as the "workers" for Ray schemes
            for model_id, scheme in self._weight_sync_schemes.items():
                torchrl_logger.debug(
                    f"RayCollector: Initializing sender for model '{model_id}'"
                )
                scheme.init_on_sender(
                    model_id=model_id,
                    remote_collectors=self.remote_collectors,
                    model=self.policy if model_id == "policy" else None,
                    context=self,
                )

            # Set up receiver schemes on remote collectors
            # This enables the remote collectors to receive weight updates
            for remote_collector in self.remote_collectors:
                torchrl_logger.debug(
                    f"RayCollector: Registering scheme receiver for remote collector {remote_collector}"
                )
                fut = remote_collector.register_scheme_receiver.remote(
                    self._weight_sync_schemes, synchronize_weights=False
                )
                ray.get(fut)

            self.weight_updater = None  # Don't use legacy system
        else:
            torchrl_logger.debug("RayCollector: Using legacy weight updater system")
            # Fall back to legacy weight updater system
            if weight_updater is None:
                weight_updater = RayWeightUpdater(
                    policy_weights=policy_weights,
                    remote_collectors=self.remote_collectors,
                    max_interval=self.max_weight_update_interval,
                )
            self.weight_updater = weight_updater
            self._weight_sync_schemes = None

        # Always initialize this flag - legacy system doesn't need lazy init
        # but we set it for consistency
        self._weight_sync_initialized = False

        # Set up weight receivers if provided
        if weight_recv_schemes is not None:
            torchrl_logger.debug("RayCollector: Setting up weight receivers...")
            self.register_scheme_receiver(weight_recv_schemes)

        if not self._weight_sync_initialized:
            self._lazy_initialize_weight_sync()

        # Print info of all remote workers (fire and forget - no need to wait)
        for e in self.remote_collectors:
            e.print_remote_collector_info.remote()

    def _lazy_initialize_weight_sync(self) -> None:
        """Initialize weight synchronization lazily on first update_policy_weights_() call.

        This method performs the initial weight synchronization that was deferred from __init__.
        It must be called before collection begins if weights need to be synced from a central policy.

        The synchronization is done here (not in __init__) because:
        1. When using policy_factory, there may be no central policy to sync from
        2. Users may want to train the policy first before syncing weights
        3. Different sub-collectors may have different policies via policy_factory
        """
        if self._weight_sync_initialized:
            return

        if self._weight_sync_schemes is None:
            # Legacy weight updater system doesn't use lazy init
            self._weight_sync_initialized = True
            return

        torchrl_logger.debug("RayCollector: Performing lazy weight synchronization")

        # Cascade synchronize_weights to remote collectors
        torchrl_logger.debug(
            "RayCollector: Cascading synchronize_weights to remote collectors"
        )
        self._sync_futures = []
        for remote_collector in self.remote_collectors:
            for model_id in self._weight_sync_schemes:
                self._sync_futures.append(
                    remote_collector.cascade_execute.remote(
                        f"_receiver_schemes['{model_id}'].connect"
                    )
                )

        # Synchronize weights for each scheme
        for model_id, scheme in self._weight_sync_schemes.items():
            torchrl_logger.debug(
                f"RayCollector: Synchronizing weights for model '{model_id}'"
            )
            scheme.connect()

        # Block sync
        torchrl_logger.debug(
            "RayCollector: Waiting for weight synchronization to finish"
        )
        ray.get(self._sync_futures)
        self._weight_sync_initialized = True
        torchrl_logger.debug("RayCollector: Weight synchronization complete")

    def _weight_update_impl(
        self,
        policy_or_weights: TensorDictBase | nn.Module | dict | None = None,
        *,
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
        model_id: str | None = None,
        weights_dict: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Override to trigger lazy weight sync initialization on first call.

        When using policy_factory without a central policy, weight synchronization
        is deferred until this method is called with actual weights.
        """
        # Trigger lazy initialization if not already done
        if not self._weight_sync_initialized:
            self._lazy_initialize_weight_sync()

        # Call parent implementation
        return super()._weight_update_impl(
            policy_or_weights=policy_or_weights,
            worker_ids=worker_ids,
            model_id=model_id,
            weights_dict=weights_dict,
            **kwargs,
        )

    # def _send_weights_scheme(self, *, scheme, processed_weights, worker_ids, model_id):
    #     if not worker_ids:
    #         worker_ids = list(range(self.num_collectors))
    #     futures = []
    #     for worker_id in worker_ids:
    #         torchrl_logger.debug(f"RayCollector: Sending weights to remote worker {worker_id}")
    #         # Call irecv
    #         fut = self.remote_collectors[worker_id].cascade_execute.remote(f"_receiver_schemes['{model_id}'].receive")
    #         futures.append(fut)
    #     torchrl_logger.debug(f"RayCollector: calling isend")
    #     scheme.send(weights=processed_weights, worker_ids=worker_ids)
    #     torchrl_logger.debug(f"RayCollector: Waiting for {len(futures)} irecv calls to finish")
    #     ray.get(futures)

    def _extract_weights_if_needed(self, weights: Any, model_id: str) -> Any:
        """Extract weights from a model if needed.

        For Ray collectors, when weights is None and we have a weight sync scheme,
        extract fresh weights from the tracked policy model.
        """
        scheme = (
            self._weight_sync_schemes.get(model_id)
            if self._weight_sync_schemes
            else None
        )

        if weights is None and scheme is not None:
            # Extract fresh weights from the scheme's model
            model = scheme.model
            if model is not None:
                from torchrl.weight_update.weight_sync_schemes import WeightStrategy

                strategy = WeightStrategy(extract_as=scheme.strategy_str)
                return strategy.extract_weights(model)

        # Fall back to base class behavior
        return super()._extract_weights_if_needed(weights, model_id)

    @property
    def num_workers(self):
        return self.num_collectors

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
            self._device = value
        else:
            self._device = [value] * self.num_collectors

    @storing_device.setter
    def storing_device(self, value):
        if isinstance(value, (tuple, list)):
            self._storing_device = value
        else:
            self._storing_device = [value] * self.num_collectors

    @env_device.setter
    def env_device(self, value):
        if isinstance(value, (tuple, list)):
            self._env_device = value
        else:
            self._env_device = [value] * self.num_collectors

    @policy_device.setter
    def policy_device(self, value):
        if isinstance(value, (tuple, list)):
            self._policy_device = value
        else:
            self._policy_device = [value] * self.num_collectors

    @staticmethod
    def _make_collector(cls, *, env_maker, policy, other_params):
        """Create a single collector instance."""
        if policy is not None:
            other_params["policy"] = policy
        collector = cls(
            env_maker,
            total_frames=-1,
            **other_params,
        )
        return collector

    def add_collectors(
        self,
        create_env_fn,
        num_envs,
        policy,
        collector_kwargs,
        remote_configs,
    ):
        """Creates and adds a number of remote collectors to the set."""
        for i, (env_maker, other_params, remote_config) in enumerate(
            zip(create_env_fn, collector_kwargs, remote_configs)
        ):
            # Add worker_idx to params so remote collectors know their index
            other_params = dict(other_params)  # Make a copy to avoid mutating original
            other_params["worker_idx"] = i

            cls = self.collector_class.as_remote(remote_config).remote
            collector = self._make_collector(
                cls,
                env_maker=[env_maker] * num_envs
                if num_envs > 1
                or (
                    isinstance(self.collector_class, type)
                    and not issubclass(self.collector_class, SyncDataCollector)
                )
                else env_maker,
                policy=policy,
                other_params=other_params,
            )
            self._remote_collectors.append(collector)

    def local_policy(self):
        """Returns local collector."""
        return self._local_policy

    @property
    def remote_collectors(self):
        """Returns list of remote collectors."""
        return self._remote_collectors

    def stop_remote_collectors(self):
        """Stops all remote collectors."""
        for _ in range(len(self._remote_collectors)):
            collector = self.remote_collectors.pop()
            # collector.__ray_terminate__.remote()  # This will kill the actor but let pending tasks finish
            ray.kill(
                collector
            )  # This will interrupt any running tasks on the actor, causing them to fail immediately

    def iterator(self):
        # Warn if weight sync wasn't initialized before collection starts
        if not self._weight_sync_initialized and self._weight_sync_schemes is not None:
            warnings.warn(
                "RayCollector iteration started before weight synchronization was initialized. "
                "Call update_policy_weights_(policy_or_weights) before iterating to sync weights "
                "from a central policy to remote collectors. If using policy_factory with "
                "independent policies on each collector, you can ignore this warning.",
                UserWarning,
                stacklevel=2,
            )

        def proc(data):
            # When using RayReplayBuffer, sub-collectors write directly to buffer
            # and return None, so skip processing
            if data is None:
                return None
            if self.split_trajs:
                data = split_trajectories(data)
            if self.postproc is not None:
                data = self.postproc(data)
            return data

        if self._sync:
            meth = self._sync_iterator
        else:
            meth = self._async_iterator
        yield from (proc(data) for data in meth())

    async def _asyncio_iterator(self):
        def proc(data):
            # When using RayReplayBuffer, sub-collectors write directly to buffer
            # and return None, so skip processing
            if data is None:
                return None
            if self.split_trajs:
                data = split_trajectories(data)
            if self.postproc is not None:
                data = self.postproc(data)
            return data

        if self._sync:
            for d in self._sync_iterator():
                yield proc(d)
        else:
            for d in self._async_iterator():
                yield proc(d)

    def _sync_iterator(self) -> Iterator[TensorDictBase]:
        """Collects one data batch per remote collector in each iteration."""
        while (
            self.collected_frames < self.total_frames and not self._stop_event.is_set()
        ):
            if self.update_after_each_batch or self.max_weight_update_interval > -1:
                torchrl_logger.debug("Updating weights on all workers")
                self.update_policy_weights_()

            # Ask for batches to all remote workers.
            pending_tasks = [e.next.remote() for e in self.remote_collectors]

            # Wait for all rollouts
            samples_ready = []
            while len(samples_ready) < self.num_collectors:
                samples_ready, samples_not_ready = ray.wait(
                    pending_tasks, num_returns=len(pending_tasks)
                )

            # Retrieve and concatenate Tensordicts
            out_td = []
            for r in pending_tasks:
                rollouts = ray.get(r)
                ray.internal.free(
                    r
                )  # should not be necessary, deleted automatically when ref count is down to 0
                out_td.append(rollouts)

            # Handle case where replay_buffer is used and rollouts are None
            if out_td[0] is None:
                # Sub-collectors are writing directly to RayReplayBuffer
                # Track frames and yield None to signal completion
                self.collected_frames += self.frames_per_batch
                yield None
            else:
                # Normal case: concatenate and yield rollouts
                if len(rollouts.batch_size):
                    out_td = torch.stack(out_td)
                else:
                    out_td = torch.cat(out_td)

                self.collected_frames += out_td.numel()
                yield out_td

        # Only auto-shutdown if not running in a background thread.
        # When using replay buffer, users should explicitly manage shutdown order.
        if self._collection_thread is None:
            self.shutdown(shutdown_ray=False)

    def _run_collection_loop(self):
        """Runs the collection loop in a background thread."""
        try:
            for _ in self.iterator():
                if self._stop_event.is_set():
                    break
                # When RayReplayBuffer is configured, sub-collectors write directly
                # to the buffer and data will be None. Otherwise, data contains rollouts.
        except Exception as e:
            torchrl_logger.error(f"Error in collection thread: {e}")
            raise

    def start(self):
        """Starts the RayCollector in a background thread."""
        if self.replay_buffer is None:
            raise RuntimeError(
                "Replay buffer must be defined for background execution."
            )
        if self._collection_thread is None or not self._collection_thread.is_alive():
            self._stop_event.clear()
            self._collection_thread = threading.Thread(
                target=self._run_collection_loop, daemon=True
            )
            self._collection_thread.start()

    async def async_shutdown(self, shutdown_ray: bool = False):
        """Finishes processes started by the collector during async execution.

        Args:
            shutdown_ray (bool): If True, also shutdown the Ray cluster. Defaults to False.
                Note: Setting this to True will kill all Ray actors in the cluster, including
                any replay buffers or other services. Only set to True if you're sure you want
                to shut down the entire Ray cluster.

        """
        self._stop_event.set()
        if self._collection_thread is not None and self._collection_thread.is_alive():
            self._collection_thread.join(timeout=5.0)
        self.stop_remote_collectors()
        if shutdown_ray:
            ray.shutdown()

    def _async_iterator(self) -> Iterator[TensorDictBase]:
        """Collects a data batch from a single remote collector in each iteration."""
        pending_tasks = {}
        for index, collector in enumerate(self.remote_collectors):
            future = collector.next.remote()
            pending_tasks[future] = index

        while (
            self.collected_frames < self.total_frames and not self._stop_event.is_set()
        ):
            if not len(list(pending_tasks.keys())) == len(self.remote_collectors):
                raise RuntimeError("Missing pending tasks, something went wrong")

            # Wait for first worker to finish
            wait_results = ray.wait(list(pending_tasks.keys()))
            future = wait_results[0][0]
            collector_index = pending_tasks.pop(future)
            collector = self.remote_collectors[collector_index]

            # Retrieve single rollouts
            out_td = ray.get(future)
            ray.internal.free(
                [future]
            )  # should not be necessary, deleted automatically when ref count is down to 0

            # Track collected frames - use frames_per_batch since out_td might be None
            # when using RayReplayBuffer (sub-collectors write directly to buffer)
            self.collected_frames += self.frames_per_batch

            yield out_td

            if self.update_after_each_batch or self.max_weight_update_interval > -1:
                torchrl_logger.debug(f"Updating weights on worker {collector_index}")
                self.update_policy_weights_(worker_ids=collector_index + 1)

            # Schedule a new collection task
            future = collector.next.remote()
            pending_tasks[future] = collector_index

        # Wait for the in-process collections tasks to finish.
        refs = list(pending_tasks.keys())
        ray.wait(refs, num_returns=len(refs))

        # Cancel the in-process collections tasks
        # for ref in refs:
        #     ray.cancel(
        #         object_ref=ref,
        #         force=False,
        #     )
        if self._collection_thread is None:
            self.shutdown()

    def set_seed(self, seed: int, static_seed: bool = False) -> list[int]:
        """Calls parent method for each remote collector iteratively and returns final seed."""
        for collector in self.remote_collectors:
            seed = ray.get(object_refs=collector.set_seed.remote(seed, static_seed))
        return seed

    def state_dict(self) -> list[OrderedDict]:
        """Calls parent method for each remote collector and returns a list of results."""
        futures = [
            collector.state_dict.remote() for collector in self.remote_collectors
        ]
        results = ray.get(object_refs=futures)
        return results

    def load_state_dict(self, state_dict: OrderedDict | list[OrderedDict]) -> None:
        """Calls parent method for each remote collector."""
        if isinstance(state_dict, OrderedDict):
            state_dicts = [state_dict]
        if len(state_dict) == 1:
            state_dicts = state_dict * len(self.remote_collectors)
        for collector, state_dict in zip(self.remote_collectors, state_dicts):
            collector.load_state_dict.remote(state_dict)

    def shutdown(
        self, timeout: float | None = None, shutdown_ray: bool = False
    ) -> None:
        """Finishes processes started by the collector.

        Args:
            timeout (float, optional): Timeout for stopping the collection thread.
            shutdown_ray (bool): If True, also shutdown the Ray cluster. Defaults to False.
                Note: Setting this to True will kill all Ray actors in the cluster, including
                any replay buffers or other services. Only set to True if you're sure you want
                to shut down the entire Ray cluster.

        """
        self._stop_event.set()
        if self._collection_thread is not None and self._collection_thread.is_alive():
            self._collection_thread.join(
                timeout=timeout if timeout is not None else 5.0
            )
        self.stop_remote_collectors()
        if shutdown_ray:
            ray.shutdown()

    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}()"
        return string
