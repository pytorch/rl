from __future__ import annotations

import _pickle
import abc

import contextlib
import sys
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import CudaGraphModule, TensorDictModule
from tensordict.utils import _zip_strict
from torch import multiprocessing as mp, nn
from torchrl import logger as torchrl_logger
from torchrl._utils import (
    _check_for_faulty_process,
    _get_mp_ctx,
    _make_process_no_warn_cls,
    _mp_sharing_strategy_for_spawn,
    _set_mp_start_method_if_unset,
    RL_WARNINGS,
)
from torchrl.collectors._base import BaseCollector
from torchrl.collectors._constants import (
    _InterruptorManager,
    _is_osx,
    DEFAULT_EXPLORATION_TYPE,
    ExplorationType,
    INSTANTIATE_TIMEOUT,
)
from torchrl.collectors._runner import _main_async_collector
from torchrl.collectors._single import Collector
from torchrl.collectors.utils import _make_meta_policy_cm, _TrajectoryPool
from torchrl.collectors.weight_update import WeightUpdaterBase
from torchrl.data import ReplayBuffer
from torchrl.data.utils import CloudpickleWrapper, DEVICE_TYPING
from torchrl.envs import EnvBase, EnvCreator
from torchrl.envs.llm.transforms import PolicyVersion
from torchrl.weight_update import (
    MultiProcessWeightSyncScheme,
    SharedMemWeightSyncScheme,
    WeightSyncScheme,
)
from torchrl.weight_update.utils import _resolve_model


class _MultiCollectorMeta(abc.ABCMeta):
    """Metaclass for MultiCollector that dispatches based on sync parameter.

    When MultiCollector is instantiated with sync=True or sync=False, the metaclass
    intercepts the call and returns the appropriate subclass instance:
    - sync=True: returns MultiSyncCollector (alias: MultiSyncCollector)
    - sync=False: returns MultiAsyncCollector (alias: MultiAsyncCollector)
    """

    def __call__(cls, *args, sync: bool | None = None, **kwargs):
        # Only dispatch if we're instantiating MultiCollector directly (not a subclass)
        # and sync is explicitly provided
        if cls.__name__ == "MultiCollector" and sync is not None:
            if sync:
                from torchrl.collectors._multi_sync import MultiSyncCollector

                return MultiSyncCollector(*args, **kwargs)
            else:
                from torchrl.collectors._multi_async import MultiAsyncCollector

                return MultiAsyncCollector(*args, **kwargs)
        return super().__call__(*args, **kwargs)


class MultiCollector(BaseCollector, metaclass=_MultiCollectorMeta):
    """Runs a given number of DataCollectors on separate processes.

    Args:
        create_env_fn (List[Callabled]): list of Callables, each returning an
            instance of :class:`~torchrl.envs.EnvBase`.
        policy (Callable): Policy to be executed in the environment.
            Must accept :class:`tensordict.tensordict.TensorDictBase` object as input.
            If ``None`` is provided (default), the policy used will be a
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

            - In all other cases an attempt to wrap it will be undergone as such:
              ``TensorDictModule(policy, in_keys=env_obs_key, out_keys=env.action_keys)``.

            .. note:: If the policy needs to be passed as a policy factory (e.g., in case it mustn't be serialized /
                pickled directly), the ``policy_factory`` should be used instead.

            .. note:: When using ``weight_sync_schemes``, both ``policy`` and ``policy_factory`` can be provided together.
                In this case, the ``policy`` is used ONLY for weight extraction (via ``TensorDict.from_module()``) to
                set up weight synchronization, but it is NOT sent to workers and its weights are NOT depopulated.
                The ``policy_factory`` is what actually gets passed to workers to create their local policy instances.
                This is useful when the policy is hard to serialize but you have a copy on the main node for
                weight synchronization purposes.

    Keyword Args:
        sync (bool, optional): if ``True``, the collector will run in sync mode (:class:`~torchrl.collectors.MultiSyncCollector`). If
            `False`, the collector will run in async mode (:class:`~torchrl.collectors.MultiAsyncCollector`).
        policy_factory (Callable[[], Callable], list of Callable[[], Callable], optional): a callable
            (or list of callables) that returns a policy instance.

            When not using ``weight_sync_schemes``, this is mutually exclusive with the ``policy`` argument.

            When using ``weight_sync_schemes``, both ``policy`` and ``policy_factory`` can be provided:
            the ``policy`` is used for weight extraction only, while ``policy_factory`` creates policies on workers.

            .. note:: `policy_factory` comes in handy whenever the policy cannot be serialized.

            .. warning:: `policy_factory` is currently not compatible with multiprocessed data
                collectors.

        num_workers (int, optional): number of workers to use. If `create_env_fn` is a list, this will be ignored.
            Defaults to `None` (workers determined by the `create_env_fn` length).
        frames_per_batch (int, Sequence[int]): A keyword-only argument representing the
            total number of elements in a batch. If a sequence is provided, represents the number of elements in a
            batch per worker. Total number of elements in a batch is then the sum over the sequence.
        total_frames (int, optional): A keyword-only argument representing the
            total number of frames returned by the collector
            during its lifespan. If the ``total_frames`` is not divisible by
            ``frames_per_batch``, an exception is raised.
            Endless collectors can be created by passing ``total_frames=-1``.
            Defaults to ``-1`` (never ending collector).
        device (int, str or torch.device, optional): The generic device of the
            collector. The ``device`` args fills any non-specified device: if
            ``device`` is not ``None`` and any of ``storing_device``, ``policy_device`` or
            ``env_device`` is not specified, its value will be set to ``device``.
            Defaults to ``None`` (No default device).
            Supports a list of devices if one wishes to indicate a different device
            for each worker. The list must be as long as the number of workers.
        storing_device (int, str or torch.device, optional): The device on which
            the output :class:`~tensordict.TensorDict` will be stored.
            If ``device`` is passed and ``storing_device`` is ``None``, it will
            default to the value indicated by ``device``.
            For long trajectories, it may be necessary to store the data on a different
            device than the one where the policy and env are executed.
            Defaults to ``None`` (the output tensordict isn't on a specific device,
            leaf tensors sit on the device where they were created).
            Supports a list of devices if one wishes to indicate a different device
            for each worker. The list must be as long as the number of workers.
        env_device (int, str or torch.device, optional): The device on which
            the environment should be cast (or executed if that functionality is
            supported). If not specified and the env has a non-``None`` device,
            ``env_device`` will default to that value. If ``device`` is passed
            and ``env_device=None``, it will default to ``device``. If the value
            as such specified of ``env_device`` differs from ``policy_device``
            and one of them is not ``None``, the data will be cast to ``env_device``
            before being passed to the env (i.e., passing different devices to
            policy and env is supported). Defaults to ``None``.
            Supports a list of devices if one wishes to indicate a different device
            for each worker. The list must be as long as the number of workers.
        policy_device (int, str or torch.device, optional): The device on which
            the policy should be cast.
            If ``device`` is passed and ``policy_device=None``, it will default
            to ``device``. If the value as such specified of ``policy_device``
            differs from ``env_device`` and one of them is not ``None``,
            the data will be cast to ``policy_device`` before being passed to
            the policy (i.e., passing different devices to policy and env is
            supported). Defaults to ``None``.
            Supports a list of devices if one wishes to indicate a different device
            for each worker. The list must be as long as the number of workers.
        create_env_kwargs (dict, optional): A dictionary with the
            keyword arguments used to create an environment. If a list is
            provided, each of its elements will be assigned to a sub-collector.
        collector_class (Python class or constructor): a collector class to be remotely instantiated. Can be
            :class:`~torchrl.collectors.Collector`,
            :class:`~torchrl.collectors.MultiSyncCollector`,
            :class:`~torchrl.collectors.MultiAsyncCollector`
            or a derived class of these.
            Defaults to :class:`~torchrl.collectors.Collector`.
        max_frames_per_traj (int, optional): Maximum steps per trajectory.
            Note that a trajectory can span across multiple batches (unless
            ``reset_at_each_iter`` is set to ``True``, see below).
            Once a trajectory reaches ``n_steps``, the environment is reset.
            If the environment wraps multiple environments together, the number
            of steps is tracked for each environment independently. Negative
            values are allowed, in which case this argument is ignored.
            Defaults to ``None`` (i.e. no maximum number of steps).
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
        reset_when_done (bool, optional): if ``True`` (default), an environment
            that return a ``True`` value in its ``"done"`` or ``"truncated"``
            entry will be reset at the corresponding indices.
        update_at_each_batch (boolm optional): if ``True``, :meth:`update_policy_weights_()`
            will be called before (sync) or after (async) each data collection.
            Defaults to ``False``.
        preemptive_threshold (:obj:`float`, optional): a value between 0.0 and 1.0 that specifies the ratio of workers
            that will be allowed to finished collecting their rollout before the rest are forced to end early.
        num_threads (int, optional): number of threads for this process.
            Defaults to the number of workers.
        num_sub_threads (int, optional): number of threads of the subprocesses.
            Should be equal to one plus the number of processes launched within
            each subprocess (or one if a single process is launched).
            Defaults to 1 for safety: if none is indicated, launching multiple
            workers may charge the cpu load too much and harm performance.
        cat_results (str, int or None): (:class:`~torchrl.collectors.MultiSyncCollector` exclusively).
            If ``"stack"``, the data collected from the workers will be stacked along the
            first dimension. This is the preferred behavior as it is the most compatible
            with the rest of the library.
            If ``0``, results will be concatenated along the first dimension
            of the outputs, which can be the batched dimension if the environments are
            batched or the time dimension if not.
            A ``cat_results`` value of ``-1`` will always concatenate results along the
            time dimension. This should be preferred over the default. Intermediate values
            are also accepted.
            Defaults to ``"stack"``.

            .. note:: From v0.5, this argument will default to ``"stack"`` for a better
                interoperability with the rest of the library.

        set_truncated (bool, optional): if ``True``, the truncated signals (and corresponding
            ``"done"`` but not ``"terminated"``) will be set to ``True`` when the last frame of
            a rollout is reached. If no ``"truncated"`` key is found, an exception is raised.
            Truncated keys can be set through ``env.add_truncated_keys``.
            Defaults to ``False``.
        use_buffers (bool, optional): if ``True``, a buffer will be used to stack the data.
            This isn't compatible with environments with dynamic specs. Defaults to ``True``
            for envs without dynamic specs, ``False`` for others.
        replay_buffer (ReplayBuffer, optional): if provided, the collector will not yield tensordicts
            but populate the buffer instead. Defaults to ``None``.
        extend_buffer (bool, optional): if `True`, the replay buffer is extended with entire rollouts and not
            with single steps. Defaults to `True` for multiprocessed data collectors.
        local_init_rb (bool, optional): if ``False``, the collector will use fake data to initialize
            the replay buffer in the main process (legacy behavior). If ``True``, the storage-level
            coordination will handle initialization with real data from worker processes.
            Defaults to ``None``, which maintains backward compatibility but shows a deprecation warning.
            This parameter is deprecated and will be removed in v0.12.
        trust_policy (bool, optional): if ``True``, a non-TensorDictModule policy will be trusted to be
            assumed to be compatible with the collector. This defaults to ``True`` for CudaGraphModules
            and ``False`` otherwise.
        compile_policy (bool or Dict[str, Any], optional): if ``True``, the policy will be compiled
            using :func:`~torch.compile` default behaviour. If a dictionary of kwargs is passed, it
            will be used to compile the policy.
        cudagraph_policy (bool or Dict[str, Any], optional): if ``True``, the policy will be wrapped
            in :class:`~tensordict.nn.CudaGraphModule` with default kwargs.
            If a dictionary of kwargs is passed, it will be used to wrap the policy.
        no_cuda_sync (bool): if ``True``, explicit CUDA synchronizations calls will be bypassed.
            For environments running directly on CUDA (`IsaacLab <https://github.com/isaac-sim/IsaacLab/>`_
            or `ManiSkills <https://github.com/haosulab/ManiSkill/>`_) cuda synchronization may cause unexpected
            crashes.
            Defaults to ``False``.
        weight_updater (WeightUpdaterBase or constructor, optional): An instance of :class:`~torchrl.collectors.WeightUpdaterBase`
            or its subclass, responsible for updating the policy weights on remote inference workers.
            If not provided, a :class:`~torchrl.collectors.MultiProcessedWeightUpdater` will be used by default,
            which handles weight synchronization across multiple processes.
            Consider using a constructor if the updater needs to be serialized.
        weight_sync_schemes (dict[str, WeightSyncScheme], optional): Dictionary of weight sync schemes for
            SENDING weights to worker sub-collectors. Keys are model identifiers (e.g., "policy")
            and values are WeightSyncScheme instances configured to send weights to child processes.
            If not provided, a :class:`~torchrl.collectors.MultiProcessWeightSyncScheme` will be used by default.
            This is for propagating weights DOWN the hierarchy (parent -> children).
        weight_recv_schemes (dict[str, WeightSyncScheme], optional): Dictionary of weight sync schemes for
            RECEIVING weights from parent collectors. Keys are model identifiers (e.g., "policy")
            and values are WeightSyncScheme instances configured to receive weights.
            This enables cascading in hierarchies like: RPCDataCollector -> MultiSyncCollector -> Collector.
            Received weights are automatically propagated to sub-collectors if matching model_ids exist.
            Defaults to ``None``.
        track_policy_version (bool or PolicyVersion, optional): if ``True``, the collector will track the version of the policy.
            This will be mediated by the :class:`~torchrl.envs.llm.transforms.policy_version.PolicyVersion` transform, which will be added to the environment.
            Alternatively, a :class:`~torchrl.envs.llm.transforms.policy_version.PolicyVersion` instance can be passed, which will be used to track
            the policy version.
            Defaults to `False`.
        worker_idx (int, optional): the index of the worker.

    Examples:
        >>> from torchrl.collectors import MultiCollector
        >>> from torchrl.envs import GymEnv
        >>>
        >>> def make_env():
        ...     return GymEnv("CartPole-v1")
        >>>
        >>> # Synchronous collection (for on-policy algorithms like PPO)
        >>> sync_collector = MultiCollector(
        ...     create_env_fn=[make_env] * 4,  # 4 parallel workers
        ...     policy=my_policy,
        ...     frames_per_batch=1000,
        ...     total_frames=100000,
        ...     sync=True,  # All workers complete before batch is delivered
        ... )
        >>>
        >>> # Asynchronous collection (for off-policy algorithms like SAC)
        >>> async_collector = MultiCollector(
        ...     create_env_fn=[make_env] * 4,
        ...     policy=my_policy,
        ...     frames_per_batch=1000,
        ...     total_frames=100000,
        ...     sync=False,  # First-come-first-serve delivery
        ... )
        >>>
        >>> # Iterate over collected data
        >>> for data in sync_collector:
        ...     # data is a TensorDict with collected transitions
        ...     pass
        >>> sync_collector.shutdown()

    """

    def __init__(
        self,
        create_env_fn: Sequence[Callable[[], EnvBase]],
        policy: None
        | (TensorDictModule | Callable[[TensorDictBase], TensorDictBase]) = None,
        *,
        num_workers: int | None = None,
        policy_factory: Callable[[], Callable]
        | list[Callable[[], Callable]]
        | None = None,
        frames_per_batch: int | Sequence[int],
        total_frames: int | None = -1,
        device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        storing_device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        env_device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        policy_device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        create_env_kwargs: Sequence[dict] | None = None,
        collector_class: type | Callable[[], BaseCollector] | None = None,
        max_frames_per_traj: int | None = None,
        init_random_frames: int | None = None,
        reset_at_each_iter: bool = False,
        postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
        split_trajs: bool | None = None,
        exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
        reset_when_done: bool = True,
        update_at_each_batch: bool = False,
        preemptive_threshold: float | None = None,
        num_threads: int | None = None,
        num_sub_threads: int = 1,
        cat_results: str | int | None = None,
        set_truncated: bool = False,
        use_buffers: bool | None = None,
        replay_buffer: ReplayBuffer | None = None,
        extend_buffer: bool = True,
        local_init_rb: bool | None = None,
        trust_policy: bool | None = None,
        compile_policy: bool | dict[str, Any] | None = None,
        cudagraph_policy: bool | dict[str, Any] | None = None,
        no_cuda_sync: bool = False,
        weight_updater: WeightUpdaterBase
        | Callable[[], WeightUpdaterBase]
        | None = None,
        weight_sync_schemes: dict[str, WeightSyncScheme] | None = None,
        weight_recv_schemes: dict[str, WeightSyncScheme] | None = None,
        track_policy_version: bool = False,
        worker_idx: int | None = None,
    ):
        self.closed = True
        self.worker_idx = worker_idx

        # Set up workers and environment functions
        create_env_fn, total_frames_per_batch = self._setup_workers_and_env_fns(
            create_env_fn, num_workers, frames_per_batch
        )

        # Set up basic configuration
        self.set_truncated = set_truncated
        self.num_sub_threads = num_sub_threads
        self.num_threads = num_threads
        self.create_env_fn = create_env_fn
        self._read_compile_kwargs(compile_policy, cudagraph_policy)

        # Set up environment kwargs
        self.create_env_kwargs = self._setup_env_kwargs(create_env_kwargs)

        # Set up devices
        storing_devices, policy_devices, env_devices = self._get_devices(
            storing_device=storing_device,
            env_device=env_device,
            policy_device=policy_device,
            device=device,
        )
        self.storing_device = storing_devices
        self.policy_device = policy_devices
        self.env_device = env_devices
        self.collector_class = collector_class
        del storing_device, env_device, policy_device, device
        self.no_cuda_sync = no_cuda_sync

        # Set up replay buffer
        self._use_buffers = use_buffers
        self.replay_buffer = replay_buffer
        self._setup_multi_replay_buffer(local_init_rb, replay_buffer, extend_buffer)

        # Set up policy and weights
        if trust_policy is None:
            trust_policy = policy is not None and isinstance(policy, CudaGraphModule)
        self.trust_policy = trust_policy

        policy_factory = self._setup_policy_factory(policy_factory)

        # Set up weight synchronization
        if weight_sync_schemes is None and weight_updater is None:
            weight_sync_schemes = {}
        elif weight_sync_schemes is not None and weight_updater is not None:
            raise TypeError(
                "Cannot specify both weight_sync_schemes and weight_updater."
            )
        if (
            weight_sync_schemes is not None
            and not weight_sync_schemes
            and weight_updater is None
            and (isinstance(policy, nn.Module) or any(policy_factory))
        ):
            # Set up a default local shared-memory sync scheme for the policy.
            # This is used to propagate weights from the orchestrator policy
            # (possibly combined with a policy_factory) down to worker policies.
            weight_sync_schemes["policy"] = SharedMemWeightSyncScheme()

        self._setup_multi_weight_sync(weight_updater, weight_sync_schemes)

        # Store policy and policy_factory - temporary set to make them visible to the receiver
        self.policy = policy
        self.policy_factory = policy_factory

        # Set up weight receivers if provided
        if weight_recv_schemes is not None:
            self.register_scheme_receiver(weight_recv_schemes)

        self._setup_multi_policy_and_weights(
            self.policy, self.policy_factory, weight_updater, weight_sync_schemes
        )

        # Set up policy version tracking
        self._setup_multi_policy_version_tracking(track_policy_version)

        # # Set up fallback policy for weight extraction
        # self._setup_fallback_policy(policy, policy_factory, weight_sync_schemes)

        # Set up total frames and other parameters
        self._setup_multi_total_frames(
            total_frames, total_frames_per_batch, frames_per_batch
        )
        self.reset_at_each_iter = reset_at_each_iter
        self.postprocs = postproc
        self.max_frames_per_traj = (
            int(max_frames_per_traj) if max_frames_per_traj is not None else 0
        )

        # Set up split trajectories
        self.requested_frames_per_batch = total_frames_per_batch
        self.reset_when_done = reset_when_done
        self._setup_split_trajs(split_trajs, reset_when_done)

        # Set up other parameters
        self.init_random_frames = (
            int(init_random_frames) if init_random_frames is not None else 0
        )
        self.update_at_each_batch = update_at_each_batch
        self.exploration_type = exploration_type
        self.frames_per_worker = np.inf

        # Set up preemptive threshold
        self._setup_preemptive_threshold(preemptive_threshold)

        # Run worker processes
        try:
            self._run_processes()
        except Exception as e:
            self.shutdown(raise_on_error=False)
            raise e

        # Set up frame tracking and other options
        self._exclude_private_keys = True
        self._frames = 0
        self._iter = -1

        # Validate cat_results
        self._validate_cat_results(cat_results)

    def _setup_workers_and_env_fns(
        self,
        create_env_fn: Sequence[Callable] | Callable,
        num_workers: int | None,
        frames_per_batch: int | Sequence[int],
    ) -> tuple[list[Callable], int]:
        """Set up workers and environment functions."""
        if isinstance(create_env_fn, Sequence):
            self.num_workers = len(create_env_fn)
        else:
            self.num_workers = num_workers
            create_env_fn = [create_env_fn] * self.num_workers

        if (
            isinstance(frames_per_batch, Sequence)
            and len(frames_per_batch) != self.num_workers
        ):
            raise ValueError(
                "If `frames_per_batch` is provided as a sequence, it should contain exactly one value per worker."
                f"Got {len(frames_per_batch)} values for {self.num_workers} workers."
            )

        self._frames_per_batch = frames_per_batch
        total_frames_per_batch = (
            sum(frames_per_batch)
            if isinstance(frames_per_batch, Sequence)
            else frames_per_batch
        )

        return create_env_fn, total_frames_per_batch

    def _setup_env_kwargs(
        self, create_env_kwargs: Sequence[dict] | dict | None
    ) -> list[dict]:
        """Set up environment kwargs for each worker."""
        if isinstance(create_env_kwargs, Mapping):
            create_env_kwargs = [create_env_kwargs] * self.num_workers
        elif create_env_kwargs is None:
            create_env_kwargs = [{}] * self.num_workers
        elif isinstance(create_env_kwargs, (tuple, list)):
            create_env_kwargs = list(create_env_kwargs)
            if len(create_env_kwargs) != self.num_workers:
                raise ValueError(
                    f"len(create_env_kwargs) must be equal to num_workers, got {len(create_env_kwargs)=} and {self.num_workers=}"
                )
        return create_env_kwargs

    def _setup_multi_replay_buffer(
        self,
        local_init_rb: bool | None,
        replay_buffer: ReplayBuffer | None,
        extend_buffer: bool,
    ) -> None:
        """Set up replay buffer for multi-process collector."""
        # Handle local_init_rb deprecation
        if local_init_rb is None:
            local_init_rb = False
            if replay_buffer is not None and not local_init_rb:
                warnings.warn(
                    "local_init_rb=False is deprecated and will be removed in v0.12. "
                    "The new storage-level initialization provides better performance.",
                    FutureWarning,
                )
        self.local_init_rb = local_init_rb

        self._check_replay_buffer_init()

        self.extend_buffer = extend_buffer

        if (
            replay_buffer is not None
            and hasattr(replay_buffer, "shared")
            and not replay_buffer.shared
        ):
            torchrl_logger.warning("Replay buffer is not shared. Sharing it.")
            replay_buffer.share()

    def _setup_policy_factory(
        self, policy_factory: Callable | list[Callable] | None
    ) -> list[Callable | None]:
        """Set up policy factory for each worker."""
        if not isinstance(policy_factory, Sequence):
            policy_factory = [policy_factory] * self.num_workers
        return policy_factory

    def _setup_multi_policy_and_weights(
        self,
        policy: TensorDictModule | Callable | None,
        policy_factory: list[Callable | None],
        weight_updater: WeightUpdaterBase | Callable | None,
        weight_sync_schemes: dict[str, WeightSyncScheme] | None,
    ) -> None:
        """Set up policy for multi-process collector.

        With weight sync schemes: validates and stores policy without weight extraction.
        With weight updater: extracts weights and creates stateful policies.

        When both policy and policy_factory are provided (with weight_sync_schemes):
        - The policy is used ONLY for weight extraction via get_model()
        - The policy is NOT depopulated of its weights
        - The policy is NOT sent to workers
        - The policy_factory is used to create policies on workers
        """
        if any(policy_factory) and policy is not None:
            if weight_sync_schemes is None:
                raise TypeError(
                    "policy_factory and policy are mutually exclusive when not using weight_sync_schemes. "
                    "When using weight_sync_schemes, policy can be provided alongside policy_factory "
                    "for weight extraction purposes only (the policy will not be sent to workers)."
                )
            # Store policy as fallback for weight extraction only
            # The policy keeps its weights and is NOT sent to workers
            self._fallback_policy = policy

        if weight_sync_schemes is not None:
            weight_sync_policy = weight_sync_schemes.get("policy")
            if weight_sync_policy is None:
                return
            # # If we only have a policy_factory (no policy instance), the scheme must
            # # be pre-initialized on the sender, since there is no policy on the
            # # collector to extract weights from.
            # if any(p is not None for p in policy_factory) and policy is None:
            #     if not weight_sync_policy.initialized_on_sender:
            #         raise RuntimeError(
            #             "the weight sync scheme must be initialized on sender ahead of time "
            #             "when passing a policy_factory without a policy instance on the collector. "
            #             f"Got {policy_factory=}"
            #         )
            # # When a policy instance is provided alongside a policy_factory, the scheme
            # # can rely on the collector context (and its policy) to extract weights.
            # # Weight sync scheme initialization then happens in _run_processes where
            # # pipes and workers are available.
        else:
            # Using legacy weight updater - extract weights and create stateful policies
            self._setup_multi_policy_and_weights_legacy(
                policy, policy_factory, weight_updater, weight_sync_schemes
            )

    def _setup_multi_policy_and_weights_legacy(
        self,
        policy: TensorDictModule | Callable | None,
        policy_factory: list[Callable | None],
        weight_updater: WeightUpdaterBase | Callable | None,
        weight_sync_schemes: dict[str, WeightSyncScheme] | None,
    ) -> None:
        """Set up policy and extract weights for each device.

        Creates stateful policies with weights extracted and placed in shared memory.
        Used with weight updater for in-place weight replacement.
        """
        self._policy_weights_dict = {}
        self._fallback_policy = None  # Policy to use for weight extraction fallback

        if not any(policy_factory):
            for policy_device, env_maker, env_maker_kwargs in _zip_strict(
                self.policy_device, self.create_env_fn, self.create_env_kwargs
            ):
                policy_new_device, get_weights_fn = self._get_policy_and_device(
                    policy=policy,
                    policy_device=policy_device,
                    env_maker=env_maker,
                    env_maker_kwargs=env_maker_kwargs,
                )
                if type(policy_new_device) is not type(policy):
                    policy = policy_new_device
                weights = (
                    TensorDict.from_module(policy_new_device)
                    if isinstance(policy_new_device, nn.Module)
                    else TensorDict()
                )
                # For multi-process collectors, ensure weights are in shared memory
                if policy_device and policy_device.type == "cpu":
                    weights = weights.share_memory_()
                self._policy_weights_dict[policy_device] = weights
                # Store the first policy instance for fallback weight extraction
                if self._fallback_policy is None:
                    self._fallback_policy = policy_new_device
            self._get_weights_fn = get_weights_fn
            if weight_updater is None:
                # For multiprocessed collectors, use MultiProcessWeightSyncScheme by default
                if weight_sync_schemes is None:
                    weight_sync_schemes = {"policy": MultiProcessWeightSyncScheme()}
                    self._weight_sync_schemes = weight_sync_schemes
        elif weight_updater is None:
            warnings.warn(
                "weight_updater is None, but policy_factory is provided. This means that the server will "
                "not know how to send the weights to the workers. If the workers can handle their weight synchronization "
                "on their own (via some specialized worker type / constructor) this may well work, but make sure "
                "your weight synchronization strategy is properly set. To suppress this warning, you can use "
                "RemoteModuleWeightUpdater() which enforces explicit weight passing when calling update_policy_weights_(weights). "
                "This will work whenever your inference and training policies are nn.Module instances with similar structures."
            )

    def _setup_multi_weight_sync(
        self,
        weight_updater: WeightUpdaterBase | Callable | None,
        weight_sync_schemes: dict[str, WeightSyncScheme] | None,
    ) -> None:
        """Set up weight synchronization for multi-process collector."""
        if weight_sync_schemes is not None:
            # Use weight sync schemes for weight distribution
            self._weight_sync_schemes = weight_sync_schemes
            # Senders will be created in _run_processes
            self.weight_updater = None
        else:
            # Use weight updater for weight distribution
            self.weight_updater = weight_updater
            self._weight_sync_schemes = None

    def _setup_multi_policy_version_tracking(
        self, track_policy_version: bool | PolicyVersion
    ) -> None:
        """Set up policy version tracking for multi-process collector."""
        self.policy_version_tracker = track_policy_version
        if PolicyVersion is not None:
            if isinstance(track_policy_version, bool) and track_policy_version:
                self.policy_version_tracker = PolicyVersion()
            elif hasattr(track_policy_version, "increment_version"):
                self.policy_version_tracker = track_policy_version
            else:
                self.policy_version_tracker = None
        else:
            if track_policy_version:
                raise ImportError(
                    "PolicyVersion is not available. Please install the LLM dependencies or set track_policy_version=False."
                )
            self.policy_version_tracker = None

    # TODO: Remove this
    def _setup_fallback_policy(
        self,
        policy: TensorDictModule | Callable | None,
        policy_factory: list[Callable | None],
        weight_sync_schemes: dict[str, WeightSyncScheme] | None,
    ) -> None:
        """Set up fallback policy for weight extraction when using policy_factory."""
        # _fallback_policy is already set in _setup_multi_policy_and_weights if a policy was provided
        # If policy_factory was used, create a policy instance to use as fallback
        if policy is None and any(policy_factory) and weight_sync_schemes is not None:
            if not hasattr(self, "_fallback_policy") or self._fallback_policy is None:
                first_factory = (
                    policy_factory[0]
                    if isinstance(policy_factory, list)
                    else policy_factory
                )
                if first_factory is not None:
                    # Create a policy instance for weight extraction
                    # This will be a reference to a policy with the same structure
                    # For shared memory, modifications to any policy will be visible here
                    self._fallback_policy = first_factory()

    def _setup_multi_total_frames(
        self,
        total_frames: int,
        total_frames_per_batch: int,
        frames_per_batch: int | Sequence[int],
    ) -> None:
        """Validate and set total frames for multi-process collector."""
        if total_frames is None or total_frames < 0:
            total_frames = float("inf")
        else:
            remainder = total_frames % total_frames_per_batch
            if remainder != 0 and RL_WARNINGS:
                warnings.warn(
                    f"total_frames ({total_frames}) is not exactly divisible by frames_per_batch ({total_frames_per_batch}). "
                    f"This means {total_frames_per_batch - remainder} additional frames will be collected. "
                    "To silence this message, set the environment variable RL_WARNINGS to False."
                )
        self.total_frames = (
            int(total_frames) if total_frames != float("inf") else total_frames
        )

    def _setup_split_trajs(
        self, split_trajs: bool | None, reset_when_done: bool
    ) -> None:
        """Set up split trajectories option."""
        if split_trajs is None:
            split_trajs = False
        elif not reset_when_done and split_trajs:
            raise RuntimeError(
                "Cannot split trajectories when reset_when_done is False."
            )
        self.split_trajs = split_trajs

    def _setup_preemptive_threshold(self, preemptive_threshold: float | None) -> None:
        """Set up preemptive threshold for early stopping."""
        if preemptive_threshold is not None:
            if _is_osx:
                raise NotImplementedError(
                    "Cannot use preemption on OSX due to Queue.qsize() not being implemented on this platform."
                )
            self.preemptive_threshold = np.clip(preemptive_threshold, 0.0, 1.0)
            manager = _InterruptorManager()
            manager.start()
            self.interruptor = manager._Interruptor()
        else:
            self.preemptive_threshold = 1.0
            self.interruptor = None

    def _should_use_random_frames(self) -> bool:
        """Determine if random frames should be used instead of the policy.

        When a replay buffer is provided, uses `replay_buffer.write_count` as the
        global step counter to support `.start()` mode where `_frames` isn't updated
        until after collection. Otherwise, uses the internal `_frames` counter.

        Returns:
            bool: True if random frames should be used, False otherwise.
        """
        if self.init_random_frames is None or self.init_random_frames <= 0:
            return False
        # Use replay_buffer.write_count when available for accurate counting in .start() mode
        if self.replay_buffer is not None:
            return self.replay_buffer.write_count < self.init_random_frames
        return self._frames < self.init_random_frames

    def _validate_cat_results(self, cat_results: str | int | None) -> None:
        """Validate cat_results parameter."""
        if cat_results is not None and (
            not isinstance(cat_results, (int, str))
            or (isinstance(cat_results, str) and cat_results != "stack")
        ):
            raise ValueError(
                "cat_results must be a string ('stack') "
                f"or an integer representing the cat dimension. Got {cat_results}."
            )
        # Lazy import to avoid circular dependency
        from torchrl.collectors._multi_sync import MultiSyncCollector

        if not isinstance(self, MultiSyncCollector) and cat_results not in (
            "stack",
            None,
        ):
            raise ValueError(
                "cat_results can only be used with ``MultiSyncCollector``."
            )
        self.cat_results = cat_results

    def _check_replay_buffer_init(self):
        if self.replay_buffer is None:
            return
        is_init = hasattr(self.replay_buffer, "_storage") and getattr(
            self.replay_buffer._storage, "initialized", True
        )
        if not is_init:
            if self.local_init_rb:
                # New behavior: storage handles all coordination itself
                # Nothing to do here - the storage will coordinate during first write
                self.replay_buffer.share()
                return

            # Legacy behavior: fake tensordict initialization
            if isinstance(self.create_env_fn[0], EnvCreator):
                fake_td = self.create_env_fn[0].meta_data.tensordict
            elif isinstance(self.create_env_fn[0], EnvBase):
                fake_td = self.create_env_fn[0].fake_tensordict()
            else:
                fake_td = self.create_env_fn[0](
                    **self.create_env_kwargs[0]
                ).fake_tensordict()
            fake_td["collector", "traj_ids"] = torch.zeros(
                fake_td.shape, dtype=torch.long
            )
            # Use extend to avoid time-related transforms to fail
            self.replay_buffer.extend(fake_td.unsqueeze(-1))
            self.replay_buffer.empty()

    @classmethod
    def _total_workers_from_env(cls, env_creators):
        if isinstance(env_creators, (tuple, list)):
            return sum(
                cls._total_workers_from_env(env_creator) for env_creator in env_creators
            )
        from torchrl.envs import ParallelEnv

        if isinstance(env_creators, ParallelEnv):
            return env_creators.num_workers
        return 1

    def _get_devices(
        self,
        *,
        storing_device: torch.device,
        policy_device: torch.device,
        env_device: torch.device,
        device: torch.device,
    ):
        # convert all devices to lists
        if not isinstance(storing_device, (list, tuple)):
            storing_device = [
                storing_device,
            ] * self.num_workers
        if not isinstance(policy_device, (list, tuple)):
            policy_device = [
                policy_device,
            ] * self.num_workers
        if not isinstance(env_device, (list, tuple)):
            env_device = [
                env_device,
            ] * self.num_workers
        if not isinstance(device, (list, tuple)):
            device = [
                device,
            ] * self.num_workers
        if not (
            len(device)
            == len(storing_device)
            == len(policy_device)
            == len(env_device)
            == self.num_workers
        ):
            raise RuntimeError(
                f"THe length of the devices does not match the number of workers: {self.num_workers}."
            )
        storing_device, policy_device, env_device = zip(
            *[
                Collector._get_devices(
                    storing_device=storing_device,
                    policy_device=policy_device,
                    env_device=env_device,
                    device=device,
                )
                for (storing_device, policy_device, env_device, device) in zip(
                    storing_device, policy_device, env_device, device
                )
            ]
        )
        return storing_device, policy_device, env_device

    def frames_per_batch_worker(self, *, worker_idx: int | None = None) -> int:
        raise NotImplementedError

    @property
    def _queue_len(self) -> int:
        raise NotImplementedError

    def _recv_and_check(
        self,
        pipe,
        *,
        timeout: float | None = None,
        check_interval: float = 1.0,
        worker_idx: int | None = None,
    ):
        """Receive from a pipe while periodically checking worker health.

        This method prevents the main process from hanging indefinitely if a worker
        dies while we're waiting for a response. It polls the pipe with a timeout
        and checks if all worker processes are still alive between polls.

        The overhead is minimal: if data is already available, `poll()` returns
        immediately and no health check is performed. Health checks only run
        when actually waiting for a slow response.

        Args:
            pipe: The pipe to receive from.
            timeout: Maximum total time to wait for a message (seconds).
                If None (default), wait indefinitely but still check worker health
                periodically.
            check_interval: How often to check worker health (seconds). Default 1.0.
            worker_idx: Optional worker index for error messages.

        Returns:
            The received message.

        Raises:
            RuntimeError: If a worker process dies while waiting.
            TimeoutError: If no message is received within the timeout (only if
                timeout is not None).
        """
        # Fast path: check if data is already available (no overhead)
        if pipe.poll(0):
            return pipe.recv()

        # Slow path: wait with periodic health checks
        elapsed = 0.0
        while timeout is None or elapsed < timeout:
            if pipe.poll(check_interval):
                return pipe.recv()
            elapsed += check_interval
            # Check if any worker has died
            _check_for_faulty_process(self.procs)
            torchrl_logger.debug(
                f"MultiCollector._recv_and_check: Still waiting after {elapsed:.1f}s"
                + (f" for worker {worker_idx}" if worker_idx is not None else "")
            )

        # Final check before timeout
        _check_for_faulty_process(self.procs)
        worker_info = f" from worker {worker_idx}" if worker_idx is not None else ""
        raise TimeoutError(
            f"Timed out after {timeout}s waiting for message{worker_info}. "
            f"All workers are still alive - this may indicate a deadlock or very slow operation."
        )

    def _run_processes(self) -> None:
        if self.num_threads is None:
            total_workers = self._total_workers_from_env(self.create_env_fn)
            self.num_threads = max(
                1, torch.get_num_threads() - total_workers
            )  # 1 more thread for this proc

        # Set up for worker processes
        torch.set_num_threads(self.num_threads)
        ctx = _get_mp_ctx()
        # Best-effort global init (only if unset) to keep other mp users consistent.
        _set_mp_start_method_if_unset(ctx.get_start_method())
        if sys.platform == "linux" and ctx.get_start_method() == "spawn":
            # On older PyTorch versions (< 2.8), pickling Process objects for "spawn"
            # can pass file descriptors for shared storages, causing spawn-time failures.
            # The strategy function returns "file_system" for old PyTorch, None otherwise.
            strategy = _mp_sharing_strategy_for_spawn()
            if strategy is not None:
                mp.set_sharing_strategy(strategy)
        queue_out = ctx.Queue(self._queue_len)  # sends data from proc to main
        self.procs = []
        self._traj_pool = _TrajectoryPool(ctx=ctx, lock=True)

        # Create all pipes upfront (needed for weight sync scheme initialization)
        # Store as list of (parent, child) tuples for use in worker creation
        pipe_pairs = [ctx.Pipe() for _ in range(self.num_workers)]
        # Extract parent pipes for external use (e.g., polling, receiving messages)
        self.pipes = [pipe_parent for pipe_parent, _ in pipe_pairs]

        _ProcessNoWarnCtx = _make_process_no_warn_cls(ctx)
        # Initialize all weight sync schemes now that pipes are available
        # Both SharedMemWeightSyncScheme (uses queues) and MultiProcessWeightSyncScheme (uses pipes)
        # can be initialized here since all required resources exist
        if self._weight_sync_schemes:
            for model_id, scheme in self._weight_sync_schemes.items():
                if not scheme.initialized_on_sender:
                    torchrl_logger.debug(
                        f"Init weight sync scheme {type(scheme).__name__} for {model_id=}."
                    )
                    scheme.init_on_sender(model_id=model_id, context=self, ctx=ctx)

        # Create a policy on the right device
        policy_factory = self.policy_factory
        has_policy_factory = any(policy_factory)
        if has_policy_factory:
            policy_factory = [
                CloudpickleWrapper(_policy_factory)
                for _policy_factory in policy_factory
            ]

        for i, (env_fun, env_fun_kwargs) in enumerate(
            zip(self.create_env_fn, self.create_env_kwargs)
        ):
            pipe_parent, pipe_child = pipe_pairs[i]  # use pre-created pipes
            if env_fun.__class__.__name__ != "EnvCreator" and not isinstance(
                env_fun, EnvBase
            ):  # to avoid circular imports
                env_fun = CloudpickleWrapper(env_fun)

            policy_device = self.policy_device[i]
            storing_device = self.storing_device[i]
            env_device = self.env_device[i]

            # Prepare policy for worker based on weight synchronization method.
            # IMPORTANT: when a policy_factory is provided, the policy instance
            # is used ONLY on the main process (for weight extraction etc.) and
            # is NOT sent to workers.
            policy = self.policy

            if self._weight_sync_schemes:
                # With weight sync schemes, send stateless policies.
                # Schemes handle weight distribution on worker side.
                if has_policy_factory:
                    # Factory will create policy in worker; don't send policy.
                    policy_to_send = None
                    cm = contextlib.nullcontext()
                elif policy is not None:
                    # Send a stateless policy down to workers: schemes apply weights.
                    policy_to_send = policy
                    cm = _make_meta_policy_cm(
                        policy, mp_start_method=ctx.get_start_method()
                    )
                else:
                    policy_to_send = None
                    cm = contextlib.nullcontext()
            elif hasattr(self, "_policy_weights_dict"):
                # LEGACY:
                # With weight updater, use in-place weight replacement.
                # Take the weights and locally dispatch them to the policy before sending.
                # This ensures a given set of shared weights for a device are shared
                # for all policies that rely on that device.
                policy_weights = self._policy_weights_dict.get(policy_device)
                if has_policy_factory:
                    # Even in legacy mode, when a policy_factory is present, do not
                    # send the stateful policy down to workers.
                    policy_to_send = None
                    cm = contextlib.nullcontext()
                else:
                    policy_to_send = policy
                    if policy is not None and policy_weights is not None:
                        cm = policy_weights.to_module(policy)
                    else:
                        cm = contextlib.nullcontext()
            else:
                # Parameter-less policy.
                cm = contextlib.nullcontext()
                # When a policy_factory exists, never send the policy instance.
                policy_to_send = None if has_policy_factory else policy

            with cm:
                kwargs = {
                    "policy_factory": policy_factory[i],
                    "pipe_child": pipe_child,
                    "queue_out": queue_out,
                    "create_env_fn": env_fun,
                    "create_env_kwargs": env_fun_kwargs,
                    "policy": policy_to_send,
                    "max_frames_per_traj": self.max_frames_per_traj,
                    "frames_per_batch": self.frames_per_batch_worker(worker_idx=i),
                    "reset_at_each_iter": self.reset_at_each_iter,
                    "policy_device": policy_device,
                    "storing_device": storing_device,
                    "env_device": env_device,
                    "exploration_type": self.exploration_type,
                    "reset_when_done": self.reset_when_done,
                    "idx": i,
                    "interruptor": self.interruptor,
                    "set_truncated": self.set_truncated,
                    "use_buffers": self._use_buffers,
                    "replay_buffer": self.replay_buffer,
                    "extend_buffer": self.extend_buffer,
                    "traj_pool": self._traj_pool,
                    "trust_policy": self.trust_policy,
                    "compile_policy": self.compiled_policy_kwargs
                    if self.compiled_policy
                    else False,
                    "cudagraph_policy": self.cudagraphed_policy_kwargs
                    if self.cudagraphed_policy
                    else False,
                    "no_cuda_sync": self.no_cuda_sync,
                    "collector_class": self.collector_class,
                    "postproc": self.postprocs
                    if self.replay_buffer is not None
                    else None,
                    "weight_sync_schemes": self._weight_sync_schemes,
                    "worker_idx": i,  # Worker index for queue-based weight distribution
                    "init_random_frames": self.init_random_frames,
                    "profile_config": self._profile_config,
                }
                proc = _ProcessNoWarnCtx(
                    target=_main_async_collector,
                    num_threads=self.num_sub_threads,
                    _start_method=ctx.get_start_method(),
                    kwargs=kwargs,
                )
                # proc.daemon can't be set as daemonic processes may be launched by the process itself
                try:
                    proc.start()
                except TypeError as err:
                    if "cannot pickle" in str(err):
                        raise RuntimeError(
                            "A non-serializable object was passed to the collector workers."
                        ) from err
                except RuntimeError as err:
                    if "Cowardly refusing to serialize non-leaf tensor" in str(err):
                        raise RuntimeError(
                            "At least one of the tensors in the policy, replay buffer, environment constructor or postprocessor requires gradients. "
                            "This is not supported in multiprocessed data collectors.\n- For ReplayBuffer transforms, use a `transform_factory` instead with `delayed_init=True`.\n"
                            "- Make sure your environment constructor does not reference tensors already instantiated on the main process.\n"
                            "- Since no gradient can be propagated through the Collector pipes, the backward graph is never needed. Consider using detached tensors instead."
                        ) from err
                    elif "_share_fd_: only available on CPU" in str(
                        err
                    ) or "_share_filename_: only available on CPU" in str(err):
                        # This is a common failure mode on older PyTorch versions when using the
                        # "spawn" multiprocessing start method: the process object contains a
                        # CUDA/MPS tensor (or a module/buffer on a non-CPU device), which must be
                        # pickled when spawning workers.
                        #
                        # See: https://github.com/pytorch/pytorch/issues/87688#issuecomment-1968901877
                        start_method = None
                        try:
                            start_method = mp.get_start_method(allow_none=True)
                        except Exception:
                            # Best effort: some environments may disallow querying here.
                            start_method = None
                        raise RuntimeError(
                            "Failed to start a collector worker process because a non-CPU tensor "
                            "was captured in the worker process arguments and had to be serialized "
                            "(pickled) at process start.\n\n"
                            f"Detected multiprocessing start method: {start_method!r}.\n\n"
                            "Workarounds:\n"
                            "- Keep any tensors/modules referenced by your collector constructor "
                            "(policy, replay buffer, postprocs, env factory captures, etc.) on CPU "
                            "when using a spawning start method (common on macOS/Windows).\n"
                            "- Or set the multiprocessing start method to 'fork' *before* creating "
                            "the collector (Unix only). Example:\n\n"
                            "    import torch.multiprocessing as mp\n"
                            "    if __name__ == '__main__':\n"
                            "        mp.set_start_method('fork', force=True)\n\n"
                            "Upstream context: https://github.com/pytorch/pytorch/issues/87688#issuecomment-1968901877"
                        ) from err
                    else:
                        raise err
                except ValueError as err:
                    if "bad value(s) in fds_to_keep" in str(err):
                        # This error occurs on old Python versions (e.g., 3.9) with old PyTorch (e.g., 2.3)
                        # when using the spawn multiprocessing start method. The spawn implementation tries to
                        # preserve file descriptors across exec, but some descriptors may be invalid/closed.
                        # This is a compatibility issue with old Python multiprocessing implementations.
                        python_version = (
                            f"{sys.version_info.major}.{sys.version_info.minor}"
                        )
                        raise RuntimeError(
                            f"Failed to start collector worker process due to file descriptor issues "
                            f"with spawn multiprocessing on Python {python_version}.\n\n"
                            f"This is a known compatibility issue with old Python/PyTorch stacks. "
                            f"Consider upgrading to Python >= 3.10 and PyTorch >= 2.5, or use the 'fork' "
                            f"multiprocessing start method on Unix systems.\n\n"
                            f"Workarounds:\n"
                            f"- Upgrade Python to >= 3.10 and PyTorch to >= 2.5\n"
                            f"- On Unix systems, force fork start method:\n"
                            f"  import torch.multiprocessing as mp\n"
                            f"  if __name__ == '__main__':\n"
                            f"      mp.set_start_method('fork', force=True)\n\n"
                            f"Upstream Python issue: https://github.com/python/cpython/issues/87706"
                        ) from err
                except _pickle.PicklingError as err:
                    if "<lambda>" in str(err):
                        raise RuntimeError(
                            """Can't open a process with doubly cloud-pickled lambda function.
This error is likely due to an attempt to use a ParallelEnv in a
multiprocessed data collector. To do this, consider wrapping your
lambda function in an `torchrl.envs.EnvCreator` wrapper as follows:
`env = ParallelEnv(N, EnvCreator(my_lambda_function))`.
This will not only ensure that your lambda function is cloud-pickled once, but
also that the state dict is synchronised across processes if needed."""
                        ) from err
                pipe_child.close()
                self.procs.append(proc)

        # Synchronize initial weights with workers AFTER starting processes but BEFORE waiting for "instantiated"
        # This must happen after proc.start() but before workers send "instantiated" to avoid deadlock:
        # Workers will call receiver.collect() during init and may block waiting for data
        if self._weight_sync_schemes:
            # start with policy
            policy_scheme = self._weight_sync_schemes.get("policy")
            if policy_scheme is not None:
                policy_scheme.connect()
            for key, scheme in self._weight_sync_schemes.items():
                if key == "policy":
                    continue
                scheme.connect()

        # Wait for workers to be ready
        for i, pipe_parent in enumerate(self.pipes):
            pipe_parent.poll(timeout=INSTANTIATE_TIMEOUT)
            try:
                msg = pipe_parent.recv()
            except EOFError as e:
                raise RuntimeError(
                    f"Worker {i} failed to initialize and closed the connection before sending status. "
                    f"This typically indicates that the worker process crashed during initialization. "
                    f"Check the worker process logs for the actual error."
                ) from e
            if msg != "instantiated":
                # Check if it's an error dict from worker
                if isinstance(msg, dict) and msg.get("error"):
                    # Reconstruct the exception from the worker
                    exc_type_name = msg["exception_type"]
                    exc_msg = msg["exception_msg"]
                    traceback_str = msg["traceback"]

                    # Try to get the actual exception class
                    exc_class = None
                    exc_module = msg["exception_module"]

                    if exc_module == "builtins":
                        # Get from builtins
                        import builtins

                        exc_class = getattr(builtins, exc_type_name, None)
                    else:
                        # Try to import from the module
                        try:
                            import importlib

                            mod = importlib.import_module(exc_module)
                            exc_class = getattr(mod, exc_type_name, None)
                        except Exception:
                            pass

                    # Re-raise with original exception type if possible
                    if exc_class is not None:
                        raise exc_class(
                            f"{exc_msg}\n\nWorker traceback:\n{traceback_str}"
                        )
                    else:
                        # Fall back to RuntimeError if we can't get the original type
                        raise RuntimeError(
                            f"Worker {i} raised {exc_type_name}: {exc_msg}\n\nWorker traceback:\n{traceback_str}"
                        )
                else:
                    # Legacy string error message
                    raise RuntimeError(msg)

        self.queue_out = queue_out
        self.closed = False

    _running_free = False

    def start(self):
        """Starts the collector(s) for asynchronous data collection.

        The collected data is stored in the provided replay buffer. This method initiates the background collection of
        data across multiple processes, allowing for decoupling of data collection and training.

        Raises:
            RuntimeError: If no replay buffer is defined during the collector's initialization.

        Example:
            >>> from torchrl.modules import RandomPolicy            >>>             >>> import time
            >>> from functools import partial
            >>>
            >>> import tqdm
            >>>
            >>> from torchrl.collectors import MultiAsyncCollector
            >>> from torchrl.data import LazyTensorStorage, ReplayBuffer
            >>> from torchrl.envs import GymEnv, set_gym_backend
            >>> import ale_py
            >>>
            >>> # Set the gym backend to gymnasium
            >>> set_gym_backend("gymnasium").set()
            >>>
            >>> if __name__ == "__main__":
            ...     # Create a random policy for the Pong environment
            ...     env_fn = partial(GymEnv, "ALE/Pong-v5")
            ...     policy = RandomPolicy(env_fn().action_spec)
            ...
            ...     # Initialize a shared replay buffer
            ...     rb = ReplayBuffer(storage=LazyTensorStorage(10000), shared=True)
            ...
            ...     # Create a multi-async data collector with 16 environments
            ...     num_envs = 16
            ...     collector = MultiAsyncCollector(
            ...         [env_fn] * num_envs,
            ...         policy=policy,
            ...         replay_buffer=rb,
            ...         frames_per_batch=num_envs * 16,
            ...         total_frames=-1,
            ...     )
            ...
            ...     # Progress bar to track the number of collected frames
            ...     pbar = tqdm.tqdm(total=100_000)
            ...
            ...     # Start the collector asynchronously
            ...     collector.start()
            ...
            ...     # Track the write count of the replay buffer
            ...     prec_wc = 0
            ...     while True:
            ...         wc = rb.write_count
            ...         c = wc - prec_wc
            ...         prec_wc = wc
            ...
            ...         # Update the progress bar
            ...         pbar.update(c)
            ...         pbar.set_description(f"Write Count: {rb.write_count}")
            ...
            ...         # Check the write count every 0.5 seconds
            ...         time.sleep(0.5)
            ...
            ...         # Stop when the desired number of frames is reached
            ...         if rb.write_count . 100_000:
            ...             break
            ...
            ...     # Shut down the collector
            ...     collector.async_shutdown()
        """
        if self.replay_buffer is None:
            raise RuntimeError("Replay buffer must be defined for execution.")
        self._running_free = True
        torchrl_logger.debug(
            f"MultiCollector.start(): Sending run_free to {len(self.pipes)} workers..."
        )
        for i, pipe in enumerate(self.pipes):
            pipe.send((None, "run_free"))
            torchrl_logger.debug(f"MultiCollector.start(): Sent run_free to worker {i}")

    @contextlib.contextmanager
    def pause(self):
        """Context manager that pauses the collector if it is running free."""
        if self._running_free:
            for pipe in self.pipes:
                pipe.send((None, "pause"))
            # Make sure all workers are paused
            for i in range(len(self.pipes)):
                # Use timeout with health check to avoid hanging if a worker dies
                timeout = 30.0
                check_interval = 1.0
                elapsed = 0.0
                while elapsed < timeout:
                    try:
                        idx, msg = self.queue_out.get(timeout=check_interval)
                        break
                    except Exception:
                        elapsed += check_interval
                        _check_for_faulty_process(self.procs)
                else:
                    _check_for_faulty_process(self.procs)
                    raise TimeoutError(
                        f"Timed out waiting for worker {i} to pause after {timeout}s"
                    )
                if msg != "paused":
                    raise ValueError(f"Expected paused, but got {msg=}.")
                torchrl_logger.debug(f"Worker {idx} is paused.")
            self._running_free = False
            yield None
            for pipe in self.pipes:
                pipe.send((None, "restart"))
            self._running_free = True
        else:
            raise RuntimeError("Collector cannot be paused.")

    def enable_profile(self, **kwargs) -> None:
        """Enable profiling for collector worker rollouts.

        For multi-process collectors, this sends the profile configuration
        to the specified workers. Must be called before iteration starts.

        See :meth:`BaseCollector.enable_profile` for full documentation.
        """
        # First, call parent to validate and set _profile_config
        super().enable_profile(**kwargs)

        # Send profile config to workers that should be profiled
        if self._profile_config is not None:
            for idx in self._profile_config.workers:
                if idx < self.num_workers:
                    self.pipes[idx].send((self._profile_config, "enable_profile"))

            # Wait for confirmation from workers
            for idx in self._profile_config.workers:
                if idx < self.num_workers:
                    if self.pipes[idx].poll(INSTANTIATE_TIMEOUT):
                        _, msg = self.pipes[idx].recv()
                        if msg != "profile_enabled":
                            raise RuntimeError(
                                f"Worker {idx}: Expected 'profile_enabled' message, got {msg}"
                            )
                    else:
                        raise TimeoutError(
                            f"Worker {idx}: Timed out waiting for profile confirmation."
                        )

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            # an AttributeError will typically be raised if the collector is deleted when the program ends.
            # In the future, insignificant changes to the close method may change the error type.
            # We excplicitely assume that any error raised during closure in
            # __del__ will not affect the program.
            pass

    def shutdown(
        self,
        timeout: float | None = None,
        close_env: bool = True,
        raise_on_error: bool = True,
    ) -> None:
        """Shuts down all processes. This operation is irreversible.

        Args:
            timeout (float, optional): The timeout for closing pipes between workers.
            close_env (bool, optional): Whether to close the environment. Defaults to `True`.
            raise_on_error (bool, optional): Whether to raise an error if the shutdown fails. Defaults to `True`.
        """
        if not close_env:
            raise RuntimeError(
                f"Cannot shutdown {type(self).__name__} collector without environment being closed."
            )
        try:
            self._shutdown_main(timeout)
        except Exception as e:
            if raise_on_error:
                raise e
            else:
                pass

    def _shutdown_main(self, timeout: float | None = None) -> None:
        if timeout is None:
            timeout = 10
        try:
            if self.closed:
                return
            _check_for_faulty_process(self.procs)
            all_closed = [False] * self.num_workers
            rep = 0
            for idx in range(self.num_workers):
                if all_closed[idx]:
                    continue
                if not self.procs[idx].is_alive():
                    continue
                self.pipes[idx].send((None, "close"))

            while not all(all_closed) and rep < 1000:
                rep += 1
                for idx in range(self.num_workers):
                    if all_closed[idx]:
                        continue
                    if not self.procs[idx].is_alive():
                        all_closed[idx] = True
                        continue
                    try:
                        if self.pipes[idx].poll(timeout / 1000 / self.num_workers):
                            msg = self.pipes[idx].recv()
                            if msg != "closed":
                                raise RuntimeError(f"got {msg} but expected 'close'")
                            all_closed[idx] = True
                        else:
                            continue
                    except BrokenPipeError:
                        all_closed[idx] = True
                        continue
            self.closed = True

            self.queue_out.close()
            for pipe in self.pipes:
                pipe.close()
            for proc in self.procs:
                proc.join(1.0)
        finally:
            import torchrl

            num_threads = min(
                torchrl._THREAD_POOL_INIT,
                torch.get_num_threads()
                + self._total_workers_from_env(self.create_env_fn),
            )
            torch.set_num_threads(num_threads)

            for proc in self.procs:
                if proc.is_alive():
                    proc.terminate()

    def async_shutdown(self, timeout: float | None = None):
        return self.shutdown(timeout=timeout)

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        """Sets the seeds of the environments stored in the DataCollector.

        Args:
            seed: integer representing the seed to be used for the environment.
            static_seed (bool, optional): if ``True``, the seed is not incremented.
                Defaults to False

        Returns:
            Output seed. This is useful when more than one environment is
            contained in the DataCollector, as the seed will be incremented for
            each of these. The resulting seed is the seed of the last
            environment.

        Examples:
            >>> from torchrl.envs import ParallelEnv
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> from tensordict.nn import TensorDictModule
            >>> from torch import nn
            >>> env_fn = lambda: GymEnv("Pendulum-v1")
            >>> env_fn_parallel = lambda: ParallelEnv(6, env_fn)
            >>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
            >>> collector = Collector(env_fn_parallel, policy, frames_per_batch=100, total_frames=300)
            >>> out_seed = collector.set_seed(1)  # out_seed = 6

        """
        _check_for_faulty_process(self.procs)
        for idx in range(self.num_workers):
            self.pipes[idx].send(((seed, static_seed), "seed"))
            new_seed, msg = self._recv_and_check(self.pipes[idx], worker_idx=idx)
            if msg != "seeded":
                raise RuntimeError(f"Expected msg='seeded', got {msg}")
            seed = new_seed
        self.reset()
        return seed

    def reset(self, reset_idx: Sequence[bool] | None = None) -> None:
        """Resets the environments to a new initial state.

        Args:
            reset_idx: Optional. Sequence indicating which environments have
                to be reset. If None, all environments are reset.

        """
        _check_for_faulty_process(self.procs)

        if reset_idx is None:
            reset_idx = [True for _ in range(self.num_workers)]
        for idx in range(self.num_workers):
            if reset_idx[idx]:
                self.pipes[idx].send((None, "reset"))
        for idx in range(self.num_workers):
            if reset_idx[idx]:
                j, msg = self._recv_and_check(self.pipes[idx], worker_idx=idx)
                if msg != "reset":
                    raise RuntimeError(f"Expected msg='reset', got {msg}")

    def state_dict(self) -> OrderedDict:
        """Returns the state_dict of the data collector.

        Each field represents a worker containing its own state_dict.

        """
        for idx in range(self.num_workers):
            self.pipes[idx].send((None, "state_dict"))
        state_dict = OrderedDict()
        for idx in range(self.num_workers):
            _state_dict, msg = self._recv_and_check(self.pipes[idx], worker_idx=idx)
            if msg != "state_dict":
                raise RuntimeError(f"Expected msg='state_dict', got {msg}")
            state_dict[f"worker{idx}"] = _state_dict
        state_dict.update({"frames": self._frames, "iter": self._iter})

        return state_dict

    def load_state_dict(self, state_dict: OrderedDict) -> None:
        """Loads the state_dict on the workers.

        Args:
            state_dict (OrderedDict): state_dict of the form
                ``{"worker0": state_dict0, "worker1": state_dict1}``.

        """
        for idx in range(self.num_workers):
            self.pipes[idx].send((state_dict[f"worker{idx}"], "load_state_dict"))
        for idx in range(self.num_workers):
            _, msg = self._recv_and_check(self.pipes[idx], worker_idx=idx)
            if msg != "loaded":
                raise RuntimeError(f"Expected msg='loaded', got {msg}")
        self._frames = state_dict["frames"]
        self._iter = state_dict["iter"]

    def increment_version(self):
        """Increment the policy version."""
        if self.policy_version_tracker is not None:
            if not hasattr(self.policy_version_tracker, "increment_version"):
                raise RuntimeError(
                    "Policy version tracker is not a PolicyVersion instance. Please pass a PolicyVersion instance to the collector."
                )
            self.policy_version_tracker.increment_version()

    @property
    def policy_version(self) -> str | int | None:
        """The current policy version."""
        if not hasattr(self.policy_version_tracker, "version"):
            return None
        return self.policy_version_tracker.version

    def get_policy_version(self) -> str | int | None:
        """Get the current policy version.

        This method exists to support remote calls in Ray actors, since properties
        cannot be accessed directly through Ray's RPC mechanism.

        Returns:
            The current version number (int) or UUID (str), or None if version tracking is disabled.
        """
        return self.policy_version

    def getattr_policy(self, attr):
        """Get an attribute from the policy of the first worker.

        Args:
            attr (str): The attribute name to retrieve from the policy.

        Returns:
            The attribute value from the policy of the first worker.

        Raises:
            AttributeError: If the attribute doesn't exist on the policy.
        """
        _check_for_faulty_process(self.procs)

        # Send command to first worker (index 0)
        self.pipes[0].send((attr, "getattr_policy"))
        result, msg = self._recv_and_check(self.pipes[0], worker_idx=0)
        if msg != "getattr_policy":
            raise RuntimeError(f"Expected msg='getattr_policy', got {msg}")

        # If the worker returned an AttributeError, re-raise it
        if isinstance(result, AttributeError):
            raise result

        return result

    def getattr_env(self, attr):
        """Get an attribute from the environment of the first worker.

        Args:
            attr (str): The attribute name to retrieve from the environment.

        Returns:
            The attribute value from the environment of the first worker.

        Raises:
            AttributeError: If the attribute doesn't exist on the environment.
        """
        _check_for_faulty_process(self.procs)

        # Send command to first worker (index 0)
        self.pipes[0].send((attr, "getattr_env"))
        result, msg = self._recv_and_check(self.pipes[0], worker_idx=0)
        if msg != "getattr_env":
            raise RuntimeError(f"Expected msg='getattr_env', got {msg}")

        # If the worker returned an AttributeError, re-raise it
        if isinstance(result, AttributeError):
            raise result

        return result

    def getattr_rb(self, attr):
        """Get an attribute from the replay buffer."""
        return getattr(self.replay_buffer, attr)

    def get_model(self, model_id: str):
        """Get model instance by ID (for weight sync schemes).

        Args:
            model_id: Model identifier (e.g., "policy", "value_net")

        Returns:
            The model instance

        Raises:
            ValueError: If model_id is not recognized
        """
        if model_id == "policy":
            # Return the fallback policy instance
            if (fallback_policy := getattr(self, "_fallback_policy", None)) is not None:
                return fallback_policy
            elif hasattr(self, "policy") and self.policy is not None:
                return self.policy
            else:
                raise ValueError(f"No policy found for model_id '{model_id}'")
        else:
            # Try to resolve via attribute access
            return _resolve_model(self, model_id)

    def get_cached_weights(self, model_id: str):
        """Get cached shared memory weights if available (for weight sync schemes).

        Args:
            model_id: Model identifier

        Returns:
            Cached TensorDict weights or None if not available
        """
        if model_id == "policy" and hasattr(self, "_policy_weights_dict"):
            # Get the policy device (first device if list)
            policy_device = self.policy_device
            if isinstance(policy_device, (list, tuple)):
                policy_device = policy_device[0] if len(policy_device) > 0 else None

            # Return cached weights for this device
            return self._policy_weights_dict.get(policy_device)
        return None

    def _weight_update_impl(
        self,
        policy_or_weights: TensorDictBase | nn.Module | dict | None = None,
        *,
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
        model_id: str | None = None,
        weights_dict: dict[str, Any] | None = None,
        **kwargs,
    ) -> None:
        """Update weights on workers.

        Weight sync schemes now use background threads on the receiver side.
        The scheme's send() method:
        1. Puts weights in the queue (or updates shared memory)
        2. Sends a "receive" instruction to the worker's background thread
        3. Waits for acknowledgment (if sync=True)

        No pipe signaling is needed - the scheme handles everything internally.
        """
        # Call parent implementation which calls scheme.send()
        # The scheme handles instruction delivery and acknowledgments
        super()._weight_update_impl(
            policy_or_weights=policy_or_weights,
            worker_ids=worker_ids,
            model_id=model_id,
            weights_dict=weights_dict,
            **kwargs,
        )

    # for RPC
    def receive_weights(self, policy_or_weights: TensorDictBase | None = None):
        return super().receive_weights(policy_or_weights)

    # for RPC
    def _receive_weights_scheme(self):
        return super()._receive_weights_scheme()


# Backward-compatible alias (deprecated, use MultiCollector instead)
MultiCollector = MultiCollector
