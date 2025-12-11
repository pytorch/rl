from __future__ import annotations

import contextlib
import threading
import warnings
import weakref
from collections import OrderedDict
from collections.abc import Callable, Iterator, Sequence
from textwrap import indent
from typing import Any

import torch

from tensordict import LazyStackedTensorDict, TensorDict, TensorDictBase
from tensordict.nn import CudaGraphModule, TensorDictModule, TensorDictModuleBase
from torch import nn
from torchrl import compile_with_warmup, logger as torchrl_logger
from torchrl._utils import (
    _ends_with,
    _make_ordinal_device,
    _replace_last,
    accept_remote_rref_udf_invocation,
    prod,
    RL_WARNINGS,
)
from torchrl.collectors._base import DataCollectorBase
from torchrl.collectors._constants import (
    cudagraph_mark_step_begin,
    DEFAULT_EXPLORATION_TYPE,
    ExplorationType,
)
from torchrl.collectors.utils import _TrajectoryPool, split_trajectories
from torchrl.collectors.weight_update import WeightUpdaterBase
from torchrl.data import ReplayBuffer
from torchrl.data.utils import DEVICE_TYPING
from torchrl.envs import EnvBase, EnvCreator, StepCounter, TransformedEnv
from torchrl.envs.common import _do_nothing
from torchrl.envs.llm.transforms import PolicyVersion
from torchrl.envs.utils import (
    _aggregate_end_of_traj,
    _make_compatible_policy,
    set_exploration_type,
)
from torchrl.modules import RandomPolicy
from torchrl.weight_update import WeightSyncScheme
from torchrl.weight_update.utils import _resolve_model


@accept_remote_rref_udf_invocation
class SyncDataCollector(DataCollectorBase):
    """Generic data collector for RL problems. Requires an environment constructor and a policy.

    Args:
        create_env_fn (Callable or EnvBase): a callable that returns an instance of
            :class:`~torchrl.envs.EnvBase` class, or the env itself.
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
                pickled directly), the ``policy_factory`` should be used instead.

    Keyword Args:
        policy_factory (Callable[[], Callable], optional): a callable that returns
            a policy instance. This is exclusive with the `policy` argument.

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
        storing_device (int, str or torch.device, optional): The device on which
            the output :class:`~tensordict.TensorDict` will be stored.
            If ``device`` is passed and ``storing_device`` is ``None``, it will
            default to the value indicated by ``device``.
            For long trajectories, it may be necessary to store the data on a different
            device than the one where the policy and env are executed.
            Defaults to ``None`` (the output tensordict isn't on a specific device,
            leaf tensors sit on the device where they were created).
        env_device (int, str or torch.device, optional): The device on which
            the environment should be cast (or executed if that functionality is
            supported). If not specified and the env has a non-``None`` device,
            ``env_device`` will default to that value. If ``device`` is passed
            and ``env_device=None``, it will default to ``device``. If the value
            as such specified of ``env_device`` differs from ``policy_device``
            and one of them is not ``None``, the data will be cast to ``env_device``
            before being passed to the env (i.e., passing different devices to
            policy and env is supported). Defaults to ``None``.
        policy_device (int, str or torch.device, optional): The device on which
            the policy should be cast.
            If ``device`` is passed and ``policy_device=None``, it will default
            to ``device``. If the value as such specified of ``policy_device``
            differs from ``env_device`` and one of them is not ``None``,
            the data will be cast to ``policy_device`` before being passed to
            the policy (i.e., passing different devices to policy and env is
            supported). Defaults to ``None``.
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

            .. warning:: Postproc is not applied when a replay buffer is used and items are added to the buffer
                as they are produced (`extend_buffer=False`). The recommended usage is to use `extend_buffer=True`.

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
        return_same_td (bool, optional): if ``True``, the same TensorDict
            will be returned at each iteration, with its values
            updated. This feature should be used cautiously: if the same
            tensordict is added to a replay buffer for instance,
            the whole content of the buffer will be identical.
            Default is ``False``.
        interruptor (_Interruptor, optional):
            An _Interruptor object that can be used from outside the class to control rollout collection.
            The _Interruptor class has methods ´start_collection´ and ´stop_collection´, which allow to implement
            strategies such as preeptively stopping rollout collection.
            Default is ``False``.
        set_truncated (bool, optional): if ``True``, the truncated signals (and corresponding
            ``"done"`` but not ``"terminated"``) will be set to ``True`` when the last frame of
            a rollout is reached. If no ``"truncated"`` key is found, an exception is raised.
            Truncated keys can be set through ``env.add_truncated_keys``.
            Defaults to ``False``.
        use_buffers (bool, optional): if ``True``, a buffer will be used to stack the data.
            This isn't compatible with environments with dynamic specs. Defaults to ``True``
            for envs without dynamic specs, ``False`` for others.
        replay_buffer (ReplayBuffer, optional): if provided, the collector will not yield tensordicts
            but populate the buffer instead.
            Defaults to ``None``.

            .. seealso:: By default (``extend_buffer=True``), the buffer is extended with entire rollouts.
                If the buffer needs to be populated with individual frames as they are collected,
                set ``extend_buffer=False`` (deprecated).

            .. warning:: Using a replay buffer with a `postproc` or `split_trajs=True` requires
                `extend_buffer=True`, as the whole batch needs to be observed to apply these transforms.

        extend_buffer (bool, optional): if `True`, the replay buffer is extended with entire rollouts and not
            with single steps. Defaults to `True`.

            .. note:: Setting this to `False` is deprecated and will be removed in a future version.
                Extending the buffer with entire rollouts is the recommended approach for better
                compatibility with postprocessing and trajectory splitting.
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
            This is typically not used in :class:`~torchrl.collectors.SyncDataCollector` as it operates in a single-process environment.
            Consider using a constructor if the updater needs to be serialized.
        weight_sync_schemes (dict[str, WeightSyncScheme], optional): **Not supported for SyncDataCollector**.
            SyncDataCollector is a leaf collector and cannot send weights to sub-collectors.
            Providing this parameter will raise a ValueError.
            Use ``weight_recv_schemes`` if you need to receive weights from a parent collector.
        weight_recv_schemes (dict[str, WeightSyncScheme], optional): Dictionary of weight sync schemes for
            RECEIVING weights from parent collectors. Keys are model identifiers (e.g., "policy")
            and values are WeightSyncScheme instances configured to receive weights.
            This enables cascading weight updates in hierarchies like:
            RPCDataCollector -> MultiSyncDataCollector -> SyncDataCollector.
            Defaults to ``None``.
        track_policy_version (bool or PolicyVersion, optional): if ``True``, the collector will track the version of the policy.
            This will be mediated by the :class:`~torchrl.envs.llm.transforms.policy_version.PolicyVersion` transform, which will be added to the environment.
            Alternatively, a :class:`~torchrl.envs.llm.transforms.policy_version.PolicyVersion` instance can be passed, which will be used to track
            the policy version.
            Defaults to `False`.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from tensordict.nn import TensorDictModule
        >>> from torch import nn
        >>> env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
        >>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
        >>> collector = SyncDataCollector(
        ...     create_env_fn=env_maker,
        ...     policy=policy,
        ...     total_frames=2000,
        ...     max_frames_per_traj=50,
        ...     frames_per_batch=200,
        ...     init_random_frames=-1,
        ...     reset_at_each_iter=False,
        ...     device="cpu",
        ...     storing_device="cpu",
        ... )
        >>> for i, data in enumerate(collector):
        ...     if i == 2:
        ...         print(data)
        ...         break
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                collector: TensorDict(
                    fields={
                        traj_ids: Tensor(shape=torch.Size([200]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        step_count: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                        truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([200]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([200, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                step_count: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                truncated: Tensor(shape=torch.Size([200, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([200]),
            device=cpu,
            is_shared=False)
        >>> del collector

    The collector delivers batches of data that are marked with a ``"time"``
    dimension.

    Examples:
        >>> assert data.names[-1] == "time"

    """

    _ignore_rb: bool = False

    def __init__(
        self,
        create_env_fn: (
            EnvBase | EnvCreator | Sequence[Callable[[], EnvBase]]  # noqa: F821
        ),  # noqa: F821
        policy: None
        | (TensorDictModule | Callable[[TensorDictBase], TensorDictBase]) = None,
        *,
        policy_factory: Callable[[], Callable] | None = None,
        frames_per_batch: int,
        total_frames: int = -1,
        device: DEVICE_TYPING | None = None,
        storing_device: DEVICE_TYPING | None = None,
        policy_device: DEVICE_TYPING | None = None,
        env_device: DEVICE_TYPING | None = None,
        create_env_kwargs: dict[str, Any] | None = None,
        max_frames_per_traj: int | None = None,
        init_random_frames: int | None = None,
        reset_at_each_iter: bool = False,
        postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
        split_trajs: bool | None = None,
        exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
        return_same_td: bool = False,
        reset_when_done: bool = True,
        interruptor=None,
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
        **kwargs,
    ):
        self.closed = True
        self.worker_idx = worker_idx

        # Note: weight_sync_schemes can be used to send weights to components
        # within the environment (e.g., RayModuleTransform), not just sub-collectors

        # Initialize environment
        env = self._init_env(create_env_fn, create_env_kwargs)

        # Initialize policy
        policy = self._init_policy(policy, policy_factory, env, trust_policy)
        self._read_compile_kwargs(compile_policy, cudagraph_policy)

        # Handle trajectory pool and validate kwargs
        self._traj_pool_val = kwargs.pop("traj_pool", None)
        if kwargs:
            raise TypeError(
                f"Keys {list(kwargs.keys())} are unknown to {type(self).__name__}."
            )

        # Set up devices and synchronization
        self._setup_devices(
            device=device,
            storing_device=storing_device,
            policy_device=policy_device,
            env_device=env_device,
            no_cuda_sync=no_cuda_sync,
        )

        self.env: EnvBase = env
        del env

        # Set up policy version tracking
        self._setup_policy_version_tracking(track_policy_version)

        # Set up replay buffer
        self._setup_replay_buffer(
            replay_buffer=replay_buffer,
            extend_buffer=extend_buffer,
            local_init_rb=local_init_rb,
            postproc=postproc,
            split_trajs=split_trajs,
            return_same_td=return_same_td,
            use_buffers=use_buffers,
        )

        self.closed = False

        # Validate reset_when_done
        if not reset_when_done:
            raise ValueError("reset_when_done is deprecated.")
        self.reset_when_done = reset_when_done
        self.n_env = self.env.batch_size.numel()

        # Register collector with policy and env
        if hasattr(policy, "register_collector"):
            policy.register_collector(self)
        if hasattr(self.env, "register_collector"):
            self.env.register_collector(self)

        # Set up policy and weights
        self._setup_policy_and_weights(policy)

        # Apply environment device
        self._apply_env_device()

        # Set up max frames per trajectory
        self._setup_max_frames_per_traj(max_frames_per_traj)

        # Validate and set total frames
        self.reset_at_each_iter = reset_at_each_iter
        self._setup_total_frames(total_frames, frames_per_batch)

        # Set up init random frames
        self._setup_init_random_frames(init_random_frames, frames_per_batch)

        # Set up postproc
        self._setup_postproc(postproc)

        # Calculate frames per batch
        self._setup_frames_per_batch(frames_per_batch)

        # Set exploration and other options
        self.exploration_type = (
            exploration_type if exploration_type else DEFAULT_EXPLORATION_TYPE
        )
        self.return_same_td = return_same_td
        self.set_truncated = set_truncated

        # Create shuttle and rollout buffers
        self._make_shuttle()
        self._maybe_make_final_rollout(make_rollout=self._use_buffers)
        self._set_truncated_keys()

        # Set split trajectories option
        if split_trajs is None:
            split_trajs = False
        self.split_trajs = split_trajs
        self._exclude_private_keys = True

        # Set up interruptor and frame tracking
        self.interruptor = interruptor
        self._frames = 0
        self._iter = -1

        # Set up weight synchronization
        self._setup_weight_sync(weight_updater, weight_sync_schemes)

        # Set up weight receivers if provided
        if weight_recv_schemes is not None:
            self.register_scheme_receiver(weight_recv_schemes)

    def _init_env(
        self,
        create_env_fn: EnvBase | EnvCreator | Callable[[], EnvBase],
        create_env_kwargs: dict[str, Any] | None,
    ) -> EnvBase:
        """Initialize and configure the environment."""
        from torchrl.envs.batched_envs import BatchedEnvBase

        if create_env_kwargs is None:
            create_env_kwargs = {}

        if not isinstance(create_env_fn, EnvBase):
            env = create_env_fn(**create_env_kwargs)
        else:
            env = create_env_fn
            if create_env_kwargs:
                if not isinstance(env, BatchedEnvBase):
                    raise RuntimeError(
                        "kwargs were passed to SyncDataCollector but they can't be set "
                        f"on environment of type {type(create_env_fn)}."
                    )
                env.update_kwargs(create_env_kwargs)
        return env

    def _init_policy(
        self,
        policy: TensorDictModule | Callable | None,
        policy_factory: Callable[[], Callable] | None,
        env: EnvBase,
        trust_policy: bool | None,
    ) -> TensorDictModule | Callable:
        """Initialize and configure the policy before device placement / wrapping."""
        if policy is None:
            if policy_factory is not None:
                policy = policy_factory()
            else:
                policy = RandomPolicy(env.full_action_spec)
        elif policy_factory is not None:
            raise TypeError("policy_factory cannot be used with policy argument.")

        if trust_policy is None:
            trust_policy = isinstance(policy, (RandomPolicy, CudaGraphModule))
        self.trust_policy = trust_policy

        return policy

    def _setup_devices(
        self,
        device: DEVICE_TYPING | None,
        storing_device: DEVICE_TYPING | None,
        policy_device: DEVICE_TYPING | None,
        env_device: DEVICE_TYPING | None,
        no_cuda_sync: bool,
    ) -> None:
        """Set up devices and synchronization functions."""
        storing_device, policy_device, env_device = self._get_devices(
            storing_device=storing_device,
            policy_device=policy_device,
            env_device=env_device,
            device=device,
        )

        self.storing_device = storing_device
        self._sync_storage = self._get_sync_fn(storing_device)

        self.env_device = env_device
        self._sync_env = self._get_sync_fn(env_device)

        self.policy_device = policy_device
        self._sync_policy = self._get_sync_fn(policy_device)

        self.device = device
        self.no_cuda_sync = no_cuda_sync
        self._cast_to_policy_device = self.policy_device != self.env_device

    def _get_sync_fn(self, device: torch.device | None) -> Callable:
        """Get the appropriate synchronization function for a device."""
        if device is not None and device.type != "cuda":
            # Cuda handles sync
            if torch.cuda.is_available():
                return torch.cuda.synchronize
            elif torch.backends.mps.is_available() and hasattr(torch, "mps"):
                return torch.mps.synchronize
            elif hasattr(torch, "npu") and torch.npu.is_available():
                return torch.npu.synchronize
            elif device.type == "cpu":
                return _do_nothing
            else:
                raise RuntimeError("Non supported device")
        else:
            return _do_nothing

    def _setup_policy_version_tracking(
        self, track_policy_version: bool | PolicyVersion
    ) -> None:
        """Set up policy version tracking if requested."""
        self.policy_version_tracker = track_policy_version
        if isinstance(track_policy_version, bool) and track_policy_version:
            from torchrl.envs.batched_envs import BatchedEnvBase

            if isinstance(self.env, BatchedEnvBase):
                raise RuntimeError(
                    "BatchedEnvBase is not supported for policy version tracking. Please add the PolicyVersion transform to the environment manually, "
                    "and pass that transform to the collector."
                )
            self.policy_version_tracker = PolicyVersion()
            self.env = self.env.append_transform(self.policy_version_tracker)  # type: ignore
        elif hasattr(track_policy_version, "increment_version"):
            self.policy_version_tracker = track_policy_version
            self.env = self.env.append_transform(self.policy_version_tracker)  # type: ignore
        else:
            self.policy_version_tracker = None

    def _setup_replay_buffer(
        self,
        replay_buffer: ReplayBuffer | None,
        extend_buffer: bool,
        local_init_rb: bool | None,
        postproc: Callable | None,
        split_trajs: bool | None,
        return_same_td: bool,
        use_buffers: bool | None,
    ) -> None:
        """Set up replay buffer configuration and validate compatibility."""
        self.replay_buffer = replay_buffer
        self.extend_buffer = extend_buffer

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

        # Validate replay buffer compatibility
        if self.replay_buffer is not None and not self._ignore_rb:
            if postproc is not None and not self.extend_buffer:
                raise TypeError(
                    "postproc must be None when a replay buffer is passed, or extend_buffer must be set to True."
                )
            if split_trajs not in (None, False) and not self.extend_buffer:
                raise TypeError(
                    "split_trajs must be None/False when a replay buffer is passed, or extend_buffer must be set to True."
                )
            if return_same_td:
                raise TypeError(
                    "return_same_td must be False when a replay buffer is passed, or extend_buffer must be set to True."
                )
            if use_buffers:
                raise TypeError("replay_buffer is exclusive with use_buffers.")

        if use_buffers is None:
            use_buffers = not self.env._has_dynamic_specs and self.replay_buffer is None
        self._use_buffers = use_buffers

    def _setup_policy_and_weights(self, policy: TensorDictModule | Callable) -> None:
        """Set up policy, wrapped policy, and extract weights."""
        # Store weak reference to original policy before any transformations
        # This allows update_policy_weights_ to sync from the original when no scheme is configured
        if isinstance(policy, nn.Module):
            self._orig_policy_ref = weakref.ref(policy)
        else:
            self._orig_policy_ref = None

        # Check if policy has meta-device parameters (sent from weight sync schemes)
        # In that case, skip device placement - weights will come from the receiver
        has_meta_params = False
        if isinstance(policy, nn.Module):
            for p in policy.parameters():
                if p.device.type == "meta":
                    has_meta_params = True
                    break

        if has_meta_params:
            # Policy has meta params - sent from weight sync schemes
            # Skip device placement, weights will come from receiver
            # Keep policy on meta device until weights are loaded
            if not self.trust_policy:
                self.policy = policy
                env = getattr(self, "env", None)
                try:
                    wrapped_policy = _make_compatible_policy(
                        policy=policy,
                        observation_spec=getattr(env, "observation_spec", None),
                        env=self.env,
                    )
                except (TypeError, AttributeError, ValueError) as err:
                    raise TypeError(
                        "Failed to wrap the policy. If the policy needs to be trusted, set trust_policy=True. Scroll up for more details."
                    ) from err
                self._wrapped_policy = wrapped_policy
            else:
                self.policy = self._wrapped_policy = policy

            # For meta-parameter policies, keep the internal (worker-side) policy
            # as the reference for collector state_dict / load_state_dict.
            if isinstance(self.policy, nn.Module):
                self._policy_w_state_dict = self.policy

            # Don't extract weights yet - they're on meta device (empty)
            self.policy_weights = TensorDict()
            self.get_weights_fn = None
        else:
            # Normal path: move policy to correct device
            policy, self.get_weights_fn = self._get_policy_and_device(policy=policy)

            if not self.trust_policy:
                self.policy = policy
                env = getattr(self, "env", None)
                try:
                    wrapped_policy = _make_compatible_policy(
                        policy=policy,
                        observation_spec=getattr(env, "observation_spec", None),
                        env=self.env,
                    )
                except (TypeError, AttributeError, ValueError) as err:
                    raise TypeError(
                        "Failed to wrap the policy. If the policy needs to be trusted, set trust_policy=True. Scroll up for more details."
                    ) from err
                self._wrapped_policy = wrapped_policy
            else:
                self.policy = self._wrapped_policy = policy

            # Use the internal, unwrapped policy (cast to the correct device) as the
            # reference for state_dict / load_state_dict and legacy weight extractors.
            if isinstance(self.policy, nn.Module):
                self._policy_w_state_dict = self.policy

            # Extract policy weights from the uncompiled wrapped policy
            # Access _wrapped_policy_uncompiled directly to avoid triggering compilation.
            if isinstance(self._wrapped_policy_uncompiled, nn.Module):
                self.policy_weights = TensorDict.from_module(
                    self._wrapped_policy_uncompiled, as_module=True
                ).data
            else:
                self.policy_weights = TensorDict()

        # If policy doesn't have meta params, compile immediately
        # Otherwise, defer until first use (after weights are loaded)
        if not has_meta_params and (self.compiled_policy or self.cudagraphed_policy):
            self._wrapped_policy_maybe_compiled = self._compile_wrapped_policy(
                self._wrapped_policy_uncompiled
            )

    def _compile_wrapped_policy(self, policy):
        """Apply compilation and/or cudagraph to a policy."""
        if self.compiled_policy:
            policy = compile_with_warmup(policy, **self.compiled_policy_kwargs)
        if self.cudagraphed_policy:
            policy = CudaGraphModule(
                policy,
                in_keys=[],
                out_keys=[],
                device=self.policy_device,
                **self.cudagraphed_policy_kwargs,
            )
        return policy

    @property
    def _wrapped_policy(self):
        """Returns the compiled policy, compiling it lazily if needed."""
        if (policy := self._wrapped_policy_maybe_compiled) is None:
            if self.compiled_policy or self.cudagraphed_policy:
                policy = (
                    self._wrapped_policy_maybe_compiled
                ) = self._compile_wrapped_policy(self._wrapped_policy_uncompiled)
            else:
                policy = (
                    self._wrapped_policy_maybe_compiled
                ) = self._wrapped_policy_uncompiled
        return policy

    @property
    def _orig_policy(self):
        """Returns the original policy passed to the collector, if still alive."""
        if self._orig_policy_ref is not None:
            return self._orig_policy_ref()
        return None

    @_wrapped_policy.setter
    def _wrapped_policy(self, value):
        """Allow setting the wrapped policy during initialization."""
        self._wrapped_policy_uncompiled = value
        self._wrapped_policy_maybe_compiled = None

    def _apply_env_device(self) -> None:
        """Apply device to environment if specified."""
        if self.env_device:
            self.env: EnvBase = self.env.to(self.env_device)
        elif self.env.device is not None:
            # Use the device of the env if none was provided
            self.env_device = self.env.device

        # Check if we need to cast to env device
        self._cast_to_env_device = self._cast_to_policy_device or (
            self.env.device != self.storing_device
        )

    def _setup_max_frames_per_traj(self, max_frames_per_traj: int | None) -> None:
        """Set up maximum frames per trajectory and add StepCounter if needed."""
        self.max_frames_per_traj = (
            int(max_frames_per_traj) if max_frames_per_traj is not None else 0
        )
        if self.max_frames_per_traj is not None and self.max_frames_per_traj > 0:
            # Check that there is no StepCounter yet
            for key in self.env.output_spec.keys(True, True):
                if isinstance(key, str):
                    key = (key,)
                if "step_count" in key:
                    raise ValueError(
                        "A 'step_count' key is already present in the environment "
                        "and the 'max_frames_per_traj' argument may conflict with "
                        "a 'StepCounter' that has already been set. "
                        "Possible solutions: Set max_frames_per_traj to 0 or "
                        "remove the StepCounter limit from the environment transforms."
                    )
            self.env = TransformedEnv(
                self.env, StepCounter(max_steps=self.max_frames_per_traj)
            )

    def _setup_total_frames(self, total_frames: int, frames_per_batch: int) -> None:
        """Validate and set total frames."""
        if total_frames is None or total_frames < 0:
            total_frames = float("inf")
        else:
            remainder = total_frames % frames_per_batch
            if remainder != 0 and RL_WARNINGS:
                warnings.warn(
                    f"total_frames ({total_frames}) is not exactly divisible by frames_per_batch ({frames_per_batch}). "
                    f"This means {frames_per_batch - remainder} additional frames will be collected."
                    "To silence this message, set the environment variable RL_WARNINGS to False."
                )
        self.total_frames = (
            int(total_frames) if total_frames != float("inf") else total_frames
        )

    def _setup_init_random_frames(
        self, init_random_frames: int | None, frames_per_batch: int
    ) -> None:
        """Set up initial random frames."""
        self.init_random_frames = (
            int(init_random_frames) if init_random_frames not in (None, -1) else 0
        )
        if (
            init_random_frames not in (-1, None, 0)
            and init_random_frames % frames_per_batch != 0
            and RL_WARNINGS
        ):
            warnings.warn(
                f"init_random_frames ({init_random_frames}) is not exactly a multiple of frames_per_batch ({frames_per_batch}), "
                f" this results in more init_random_frames than requested"
                f" ({-(-init_random_frames // frames_per_batch) * frames_per_batch})."
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )

    def _setup_postproc(self, postproc: Callable | None) -> None:
        """Set up post-processing transform."""
        self.postproc = postproc
        if (
            self.postproc is not None
            and hasattr(self.postproc, "to")
            and self.storing_device
        ):
            postproc = self.postproc.to(self.storing_device)
            if postproc is not self.postproc and postproc is not None:
                self.postproc = postproc

    def _setup_frames_per_batch(self, frames_per_batch: int) -> None:
        """Calculate and validate frames per batch."""
        if frames_per_batch % self.n_env != 0 and RL_WARNINGS:
            warnings.warn(
                f"frames_per_batch ({frames_per_batch}) is not exactly divisible by the number of batched environments ({self.n_env}), "
                f" this results in more frames_per_batch per iteration that requested"
                f" ({-(-frames_per_batch // self.n_env) * self.n_env}). "
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )
        self.frames_per_batch = -(-frames_per_batch // self.n_env)
        self.requested_frames_per_batch = self.frames_per_batch * self.n_env

    def _setup_weight_sync(
        self,
        weight_updater: WeightUpdaterBase | Callable | None,
        weight_sync_schemes: dict[str, WeightSyncScheme] | None,
    ) -> None:
        """Set up weight synchronization system."""
        if weight_sync_schemes is not None:
            # Use new simplified weight synchronization system
            self._weight_sync_schemes = weight_sync_schemes
            # Initialize and synchronize schemes that need sender-side setup
            # (e.g., RayModuleTransformScheme for updating transforms in the env)
            for model_id, scheme in weight_sync_schemes.items():
                if not scheme.initialized_on_sender:
                    scheme.init_on_sender(model_id=model_id, context=self)
                if not scheme.synchronized_on_sender:
                    scheme.connect()
            self.weight_updater = None  # Don't use legacy system
        elif weight_updater is not None:
            # Use legacy weight updater system if explicitly provided
            if not isinstance(weight_updater, WeightUpdaterBase):
                if callable(weight_updater):
                    weight_updater = weight_updater()
                else:
                    raise TypeError(
                        f"weight_updater must be a subclass of WeightUpdaterBase. Got {type(weight_updater)} instead."
                    )
            warnings.warn(
                "Using WeightUpdaterBase is deprecated. Please use weight_sync_schemes instead. "
                "This will be removed in a future version.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.weight_updater = weight_updater
            self._weight_sync_schemes = None
        else:
            # No weight sync needed for single-process collectors
            self.weight_updater = None
            self._weight_sync_schemes = None

    @property
    def _traj_pool(self):
        pool = getattr(self, "_traj_pool_val", None)
        if pool is None:
            pool = self._traj_pool_val = _TrajectoryPool()
        return pool

    def _make_shuttle(self):
        # Shuttle is a deviceless tensordict that just carried data from env to policy and policy to env
        with torch.no_grad():
            self._carrier = self.env.reset()
        if self.policy_device != self.env_device or self.env_device is None:
            self._shuttle_has_no_device = True
            self._carrier.clear_device_()
        else:
            self._shuttle_has_no_device = False

        traj_ids = self._traj_pool.get_traj_and_increment(
            self.n_env, device=self.storing_device
        ).view(self.env.batch_size)
        self._carrier.set(
            ("collector", "traj_ids"),
            traj_ids,
        )

    def _maybe_make_final_rollout(self, make_rollout: bool):
        if make_rollout:
            with torch.no_grad():
                self._final_rollout = self.env.fake_tensordict()

            # If storing device is not None, we use this to cast the storage.
            # If it is None and the env and policy are on the same device,
            # the storing device is already the same as those, so we don't need
            # to consider this use case.
            # In all other cases, we can't really put a device on the storage,
            # since at least one data source has a device that is not clear.
            if self.storing_device:
                self._final_rollout = self._final_rollout.to(
                    self.storing_device, non_blocking=True
                )
            else:
                # erase all devices
                self._final_rollout.clear_device_()

        # Check if policy has meta-device parameters (not yet initialized)
        has_meta_params = False
        if hasattr(self, "_wrapped_policy_uncompiled") and isinstance(
            self._wrapped_policy_uncompiled, nn.Module
        ):
            for p in self._wrapped_policy_uncompiled.parameters():
                if p.device.type == "meta":
                    has_meta_params = True
                    break

        # If the policy has a valid spec, we use it
        self._policy_output_keys = set()
        if (
            make_rollout
            and hasattr(
                self._wrapped_policy_uncompiled
                if has_meta_params
                else self._wrapped_policy,
                "spec",
            )
            and (
                self._wrapped_policy_uncompiled
                if has_meta_params
                else self._wrapped_policy
            ).spec
            is not None
            and all(
                v is not None
                for v in (
                    self._wrapped_policy_uncompiled
                    if has_meta_params
                    else self._wrapped_policy
                ).spec.values(True, True)
            )
        ):
            if any(
                key not in self._final_rollout.keys(isinstance(key, tuple))
                for key in (
                    self._wrapped_policy_uncompiled
                    if has_meta_params
                    else self._wrapped_policy
                ).spec.keys(True, True)
            ):
                # if policy spec is non-empty, all the values are not None and the keys
                # match the out_keys we assume the user has given all relevant information
                # the policy could have more keys than the env:
                policy_spec = (
                    self._wrapped_policy_uncompiled
                    if has_meta_params
                    else self._wrapped_policy
                ).spec
                if policy_spec.ndim < self._final_rollout.ndim:
                    policy_spec = policy_spec.expand(self._final_rollout.shape)
                for key, spec in policy_spec.items(True, True):
                    self._policy_output_keys.add(key)
                    if key in self._final_rollout.keys(True):
                        continue
                    self._final_rollout.set(key, spec.zero())
        elif (
            not make_rollout
            and hasattr(
                self._wrapped_policy_uncompiled
                if has_meta_params
                else self._wrapped_policy,
                "out_keys",
            )
            and (
                self._wrapped_policy_uncompiled
                if has_meta_params
                else self._wrapped_policy
            ).out_keys
        ):
            self._policy_output_keys = list(
                (
                    self._wrapped_policy_uncompiled
                    if has_meta_params
                    else self._wrapped_policy
                ).out_keys
            )
        elif has_meta_params:
            # Policy has meta params and no spec/out_keys - defer initialization
            # Mark that we need to initialize later when weights are loaded
            self._policy_output_keys = set()
            if make_rollout:
                # We'll populate keys on first actual rollout after weights are loaded
                self._final_rollout_needs_init = True
        else:
            if make_rollout:
                # otherwise, we perform a small number of steps with the policy to
                # determine the relevant keys with which to pre-populate _final_rollout.
                # This is the safest thing to do if the spec has None fields or if there is
                # no spec at all.
                # See #505 for additional context.
                self._final_rollout.update(self._carrier.copy())
            with torch.no_grad():
                policy_input = self._carrier.copy()
                if self.policy_device:
                    policy_input = policy_input.to(self.policy_device)
                # we cast to policy device, we'll deal with the device later
                policy_input_copy = policy_input.copy()
                policy_input_clone = (
                    policy_input.clone()
                )  # to test if values have changed in-place
                if self.compiled_policy:
                    cudagraph_mark_step_begin()
                policy_output = self._wrapped_policy(policy_input)

                # check that we don't have exclusive keys, because they don't appear in keys
                def check_exclusive(val):
                    if (
                        isinstance(val, LazyStackedTensorDict)
                        and val._has_exclusive_keys
                    ):
                        raise RuntimeError(
                            "LazyStackedTensorDict with exclusive keys are not permitted in collectors. "
                            "Consider using a placeholder for missing keys."
                        )

                policy_output._fast_apply(
                    check_exclusive, call_on_nested=True, filter_empty=True
                )

                # Use apply, because it works well with lazy stacks
                # Edge-case of this approach: the policy may change the values in-place and only by a tiny bit
                # or occasionally. In these cases, the keys will be missed (we can't detect if the policy has
                # changed them here).
                # This will cause a failure to update entries when policy and env device mismatch and
                # casting is necessary.
                def filter_policy(name, value_output, value_input, value_input_clone):
                    if (value_input is None) or (
                        (value_output is not value_input)
                        and (
                            value_output.device != value_input_clone.device
                            or ~torch.isclose(value_output, value_input_clone).any()
                        )
                    ):
                        return value_output

                filtered_policy_output = policy_output.apply(
                    filter_policy,
                    policy_input_copy,
                    policy_input_clone,
                    default=None,
                    filter_empty=True,
                    named=True,
                )
                self._policy_output_keys = list(
                    self._policy_output_keys.union(
                        set(filtered_policy_output.keys(True, True))
                    )
                )
                if make_rollout:
                    self._final_rollout.update(
                        policy_output.select(*self._policy_output_keys)
                    )
                del filtered_policy_output, policy_output, policy_input

        _env_output_keys = []
        for spec in ["full_observation_spec", "full_done_spec", "full_reward_spec"]:
            _env_output_keys += list(self.env.output_spec[spec].keys(True, True))
        self._env_output_keys = _env_output_keys
        if make_rollout:
            self._final_rollout = (
                self._final_rollout.unsqueeze(-1)
                .expand(*self.env.batch_size, self.frames_per_batch)
                .clone()
                .zero_()
            )

            # in addition to outputs of the policy, we add traj_ids to
            # _final_rollout which will be collected during rollout
            self._final_rollout.set(
                ("collector", "traj_ids"),
                torch.zeros(
                    *self._final_rollout.batch_size,
                    dtype=torch.int64,
                    device=self.storing_device,
                ),
            )
            self._final_rollout.refine_names(..., "time")

    def _set_truncated_keys(self):
        self._truncated_keys = []
        if self.set_truncated:
            if not any(_ends_with(key, "truncated") for key in self.env.done_keys):
                raise RuntimeError(
                    "set_truncated was set to True but no truncated key could be found "
                    "in the environment. Make sure the truncated keys are properly set using "
                    "`env.add_truncated_keys()` before passing the env to the collector."
                )
            self._truncated_keys = [
                key for key in self.env.done_keys if _ends_with(key, "truncated")
            ]

    @classmethod
    def _get_devices(
        cls,
        *,
        storing_device: torch.device,
        policy_device: torch.device,
        env_device: torch.device,
        device: torch.device,
    ):
        device = _make_ordinal_device(torch.device(device) if device else device)
        storing_device = _make_ordinal_device(
            torch.device(storing_device) if storing_device else device
        )
        policy_device = _make_ordinal_device(
            torch.device(policy_device) if policy_device else device
        )
        env_device = _make_ordinal_device(
            torch.device(env_device) if env_device else device
        )
        if storing_device is None and (env_device == policy_device):
            storing_device = env_device
        return storing_device, policy_device, env_device

    # for RPC
    def next(self):
        return super().next()

    # for RPC
    def update_policy_weights_(
        self,
        policy_or_weights: TensorDictBase | TensorDictModuleBase | dict | None = None,
        *,
        worker_ids: int | list[int] | torch.device | list[torch.device] | None = None,
        **kwargs,
    ) -> None:
        if "policy_weights" in kwargs:
            warnings.warn(
                "`policy_weights` is deprecated. Use `policy_or_weights` instead.",
                DeprecationWarning,
            )
            policy_or_weights = kwargs.pop("policy_weights")

        super().update_policy_weights_(
            policy_or_weights=policy_or_weights, worker_ids=worker_ids, **kwargs
        )

    def _maybe_fallback_update(
        self,
        policy_or_weights: TensorDictBase | TensorDictModuleBase | dict | None = None,
        *,
        model_id: str | None = None,
    ) -> None:
        """Copy weights from original policy to internal policy when no scheme configured."""
        if model_id is not None and model_id != "policy":
            return

        # Get source weights - either from argument or from original policy
        if policy_or_weights is not None:
            weights = self._extract_weights_if_needed(policy_or_weights, "policy")
        elif self._orig_policy is not None:
            weights = TensorDict.from_module(self._orig_policy)
        else:
            return

        # Apply to internal policy
        if (
            hasattr(self, "_policy_w_state_dict")
            and self._policy_w_state_dict is not None
        ):
            TensorDict.from_module(self._policy_w_state_dict).data.update_(weights.data)

    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        """Sets the seeds of the environments stored in the DataCollector.

        Args:
            seed (int): integer representing the seed to be used for the environment.
            static_seed(bool, optional): if ``True``, the seed is not incremented.
                Defaults to False

        Returns:
            Output seed. This is useful when more than one environment is contained in the DataCollector, as the
            seed will be incremented for each of these. The resulting seed is the seed of the last environment.

        Examples:
            >>> from torchrl.envs import ParallelEnv
            >>> from torchrl.envs.libs.gym import GymEnv
            >>> from tensordict.nn import TensorDictModule
            >>> from torch import nn
            >>> env_fn = lambda: GymEnv("Pendulum-v1")
            >>> env_fn_parallel = ParallelEnv(6, env_fn)
            >>> policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
            >>> collector = SyncDataCollector(env_fn_parallel, policy, total_frames=300, frames_per_batch=100)
            >>> out_seed = collector.set_seed(1)  # out_seed = 6

        """
        out = self.env.set_seed(seed, static_seed=static_seed)
        return out

    def _increment_frames(self, numel):
        self._frames += numel
        completed = self._frames >= self.total_frames
        if completed:
            self.env.close()
        return completed

    def iterator(self) -> Iterator[TensorDictBase]:
        """Iterates through the DataCollector.

        Yields: TensorDictBase objects containing (chunks of) trajectories

        """
        if (
            not self.no_cuda_sync
            and self.storing_device
            and self.storing_device.type == "cuda"
        ):
            stream = torch.cuda.Stream(self.storing_device, priority=-1)
            event = stream.record_event()
            streams = [stream]
            events = [event]
        elif not self.no_cuda_sync and self.storing_device is None:
            streams = []
            events = []
            # this way of checking cuda is robust to lazy stacks with mismatching shapes
            cuda_devices = set()

            def cuda_check(tensor: torch.Tensor):
                if tensor.is_cuda:
                    cuda_devices.add(tensor.device)

            if not self._use_buffers:
                # This may be a bit dangerous as `torch.device("cuda")` may not have a precise
                # device associated, whereas `tensor.device` always has
                for spec in self.env.specs.values(True, True):
                    if spec.device is not None and spec.device.type == "cuda":
                        if ":" not in str(spec.device):
                            raise RuntimeError(
                                "A cuda spec did not have a device associated. Make sure to "
                                "pass `'cuda:device_num'` to each spec device."
                            )
                        cuda_devices.add(spec.device)
            else:
                self._final_rollout.apply(cuda_check, filter_empty=True)
            for device in cuda_devices:
                streams.append(torch.cuda.Stream(device, priority=-1))
                events.append(streams[-1].record_event())
        else:
            streams = []
            events = []
        with contextlib.ExitStack() as stack:
            for stream in streams:
                stack.enter_context(torch.cuda.stream(stream))

            while self._frames < self.total_frames:
                self._iter += 1
                torchrl_logger.debug("Collector: rollout.")
                tensordict_out = self.rollout()
                if tensordict_out is None:
                    # if a replay buffer is passed and self.extend_buffer=False, there is no tensordict_out
                    #  frames are updated within the rollout function
                    torchrl_logger.debug("Collector: No tensordict_out. Yielding.")
                    yield
                    continue
                self._increment_frames(tensordict_out.numel())
                tensordict_out = self._postproc(tensordict_out)
                torchrl_logger.debug("Collector: postproc done.")
                if self.return_same_td:
                    # This is used with multiprocessed collectors to use the buffers
                    # stored in the tensordict.
                    if events:
                        for event in events:
                            event.record()
                            event.synchronize()
                    yield tensordict_out
                elif self.replay_buffer is not None and not self._ignore_rb:
                    self.replay_buffer.extend(tensordict_out)
                    torchrl_logger.debug(
                        f"Collector: Added {tensordict_out.numel()} frames to replay buffer. "
                        "Buffer write count: {self.replay_buffer.write_count}. Yielding."
                    )
                    yield
                else:
                    # we must clone the values, as the tensordict is updated in-place.
                    # otherwise the following code may break:
                    # >>> for i, data in enumerate(collector):
                    # >>>      if i == 0:
                    # >>>          data0 = data
                    # >>>      elif i == 1:
                    # >>>          data1 = data
                    # >>>      else:
                    # >>>          break
                    # >>> assert data0["done"] is not data1["done"]
                    yield tensordict_out.clone()

    def start(self):
        """Starts the collector in a separate thread for asynchronous data collection.

        The collected data is stored in the provided replay buffer. This method is useful when you want to decouple data
        collection from training, allowing your training loop to run independently of the data collection process.

        Raises:
            RuntimeError: If no replay buffer is defined during the collector's initialization.

        Example:
            >>> from torchrl.modules import RandomPolicy            >>>             >>> import time
            >>> from functools import partial
            >>>
            >>> import tqdm
            >>>
            >>> from torchrl.collectors import SyncDataCollector
            >>> from torchrl.data import LazyTensorStorage, ReplayBuffer
            >>> from torchrl.envs import GymEnv, set_gym_backend
            >>> import ale_py
            >>>
            >>> # Set the gym backend to gymnasium
            >>> set_gym_backend("gymnasium").set()
            >>>
            >>> if __name__ == "__main__":
            ...     # Create a random policy for the Pong environment
            ...     env = GymEnv("ALE/Pong-v5")
            ...     policy = RandomPolicy(env.action_spec)
            ...
            ...     # Initialize a shared replay buffer
            ...     rb = ReplayBuffer(storage=LazyTensorStorage(1000), shared=True)
            ...
            ...     # Create a synchronous data collector
            ...     collector = SyncDataCollector(
            ...         env,
            ...         policy=policy,
            ...         replay_buffer=rb,
            ...         frames_per_batch=256,
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
        if not self.is_running():
            self._stop = False
            self._thread = threading.Thread(target=self._run_iterator)
            self._thread.daemon = (
                True  # So that the thread dies when the main program exits
            )
            self._thread.start()

    def _run_iterator(self):
        for _ in self:
            if self._stop:
                return

    def is_running(self):
        return hasattr(self, "_thread") and self._thread.is_alive()

    def async_shutdown(
        self, timeout: float | None = None, close_env: bool = True
    ) -> None:
        """Finishes processes started by ray.init() during async execution."""
        self._stop = True
        if hasattr(self, "_thread") and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        self.shutdown(close_env=close_env)

    def _postproc(self, tensordict_out):
        if self.split_trajs:
            tensordict_out = split_trajectories(tensordict_out, prefix="collector")
        if self.postproc is not None:
            tensordict_out = self.postproc(tensordict_out)
        if self._exclude_private_keys:

            def is_private(key):
                if isinstance(key, str) and key.startswith("_"):
                    return True
                if isinstance(key, tuple) and any(_key.startswith("_") for _key in key):
                    return True
                return False

            excluded_keys = [
                key for key in tensordict_out.keys(True) if is_private(key)
            ]
            tensordict_out = tensordict_out.exclude(*excluded_keys, inplace=True)
        return tensordict_out

    def _update_traj_ids(self, env_output) -> None:
        # we can't use the reset keys because they're gone
        traj_sop = _aggregate_end_of_traj(
            env_output.get("next"), done_keys=self.env.done_keys
        )
        if traj_sop.any():
            device = self.storing_device

            traj_ids = self._carrier.get(("collector", "traj_ids"))
            if device is not None:
                traj_ids = traj_ids.to(device)
                traj_sop = traj_sop.to(device)
            elif traj_sop.device != traj_ids.device:
                traj_sop = traj_sop.to(traj_ids.device)

            pool = self._traj_pool
            new_traj = pool.get_traj_and_increment(
                traj_sop.sum(), device=traj_sop.device
            )
            traj_ids = traj_ids.masked_scatter(traj_sop, new_traj)
            self._carrier.set(("collector", "traj_ids"), traj_ids)

    @torch.no_grad()
    def rollout(self) -> TensorDictBase:
        """Computes a rollout in the environment using the provided policy.

        Returns:
            TensorDictBase containing the computed rollout.

        """
        if self.reset_at_each_iter:
            self._carrier.update(self.env.reset())

        # self._shuttle.fill_(("collector", "step_count"), 0)
        if self._use_buffers:
            self._final_rollout.fill_(("collector", "traj_ids"), -1)
        else:
            pass
        tensordicts = []
        with set_exploration_type(self.exploration_type):
            for t in range(self.frames_per_batch):
                if (
                    self.init_random_frames is not None
                    and self._frames < self.init_random_frames
                ):
                    self.env.rand_action(self._carrier)
                    if (
                        self.policy_device is not None
                        and self.policy_device != self.env_device
                    ):
                        # TODO: This may break with exclusive / ragged lazy stacks
                        self._carrier.apply(
                            lambda name, val: val.to(
                                device=self.policy_device, non_blocking=True
                            )
                            if name in self._policy_output_keys
                            else val,
                            out=self._carrier,
                            named=True,
                            nested_keys=True,
                        )
                else:
                    if self._cast_to_policy_device:
                        if self.policy_device is not None:
                            # This is unsafe if the shuttle is in pin_memory -- otherwise cuda will be happy with non_blocking
                            non_blocking = (
                                not self.no_cuda_sync
                                or self.policy_device.type == "cuda"
                            )
                            policy_input = self._carrier.to(
                                self.policy_device,
                                non_blocking=non_blocking,
                            )
                            if not self.no_cuda_sync:
                                self._sync_policy()
                        elif self.policy_device is None:
                            # we know the tensordict has a device otherwise we would not be here
                            # we can pass this, clear_device_ must have been called earlier
                            # policy_input = self._shuttle.clear_device_()
                            policy_input = self._carrier
                    else:
                        policy_input = self._carrier
                    # we still do the assignment for security
                    if self.compiled_policy:
                        cudagraph_mark_step_begin()
                    policy_output = self._wrapped_policy(policy_input)
                    if self.compiled_policy:
                        policy_output = policy_output.clone()
                    if self._carrier is not policy_output:
                        # ad-hoc update shuttle
                        self._carrier.update(
                            policy_output, keys_to_update=self._policy_output_keys
                        )

                if self._cast_to_env_device:
                    if self.env_device is not None:
                        non_blocking = (
                            not self.no_cuda_sync or self.env_device.type == "cuda"
                        )
                        env_input = self._carrier.to(
                            self.env_device, non_blocking=non_blocking
                        )
                        if not self.no_cuda_sync:
                            self._sync_env()
                    elif self.env_device is None:
                        # we know the tensordict has a device otherwise we would not be here
                        # we can pass this, clear_device_ must have been called earlier
                        # env_input = self._shuttle.clear_device_()
                        env_input = self._carrier
                else:
                    env_input = self._carrier
                env_output, env_next_output = self.env.step_and_maybe_reset(env_input)

                if self._carrier is not env_output:
                    # ad-hoc update shuttle
                    next_data = env_output.get("next")
                    if self._shuttle_has_no_device:
                        # Make sure
                        next_data.clear_device_()
                    self._carrier.set("next", next_data)

                if (
                    self.replay_buffer is not None
                    and not self._ignore_rb
                    and not self.extend_buffer
                ):
                    torchrl_logger.debug(
                        f"Collector: Adding {env_output.numel()} frames to replay buffer using add()."
                    )
                    self.replay_buffer.add(self._carrier)
                    if self._increment_frames(self._carrier.numel()):
                        return
                else:
                    if self.storing_device is not None:
                        torchrl_logger.debug(
                            f"Collector: Moving to {self.storing_device} and adding to queue."
                        )
                        non_blocking = (
                            not self.no_cuda_sync or self.storing_device.type == "cuda"
                        )
                        tensordicts.append(
                            self._carrier.to(
                                self.storing_device, non_blocking=non_blocking
                            )
                        )
                        if not self.no_cuda_sync:
                            self._sync_storage()
                    else:
                        tensordicts.append(self._carrier)

                # carry over collector data without messing up devices
                collector_data = self._carrier.get("collector").copy()
                self._carrier = env_next_output
                if self._shuttle_has_no_device:
                    self._carrier.clear_device_()
                self._carrier.set("collector", collector_data)
                self._update_traj_ids(env_output)

                if (
                    self.interruptor is not None
                    and self.interruptor.collection_stopped()
                ):
                    torchrl_logger.debug("Collector: Interruptor stopped.")
                    if (
                        self.replay_buffer is not None
                        and not self._ignore_rb
                        and not self.extend_buffer
                    ):
                        return
                    result = self._final_rollout
                    if self._use_buffers:
                        try:
                            torch.stack(
                                tensordicts,
                                self._final_rollout.ndim - 1,
                                out=self._final_rollout[..., : t + 1],
                            )
                        except RuntimeError:
                            with self._final_rollout.unlock_():
                                torch.stack(
                                    tensordicts,
                                    self._final_rollout.ndim - 1,
                                    out=self._final_rollout[..., : t + 1],
                                )
                    else:
                        result = TensorDict.maybe_dense_stack(tensordicts, dim=-1)
                    break
            else:
                if self._use_buffers:
                    torchrl_logger.debug("Returning final rollout within buffer.")
                    result = self._final_rollout
                    try:
                        result = torch.stack(
                            tensordicts,
                            self._final_rollout.ndim - 1,
                            out=self._final_rollout,
                        )

                    except RuntimeError:
                        with self._final_rollout.unlock_():
                            result = torch.stack(
                                tensordicts,
                                self._final_rollout.ndim - 1,
                                out=self._final_rollout,
                            )
                elif (
                    self.replay_buffer is not None
                    and not self._ignore_rb
                    and not self.extend_buffer
                ):
                    return
                else:
                    torchrl_logger.debug(
                        "Returning final rollout with NO buffer (maybe_dense_stack)."
                    )
                    result = TensorDict.maybe_dense_stack(tensordicts, dim=-1)
                    result.refine_names(..., "time")

        return self._maybe_set_truncated(result)

    def _maybe_set_truncated(self, final_rollout):
        last_step = (slice(None),) * (final_rollout.ndim - 1) + (-1,)
        for truncated_key in self._truncated_keys:
            truncated = final_rollout["next", truncated_key]
            truncated[last_step] = True
            final_rollout["next", truncated_key] = truncated
            done = final_rollout["next", _replace_last(truncated_key, "done")]
            final_rollout["next", _replace_last(truncated_key, "done")] = (
                done | truncated
            )
        return final_rollout

    @torch.no_grad()
    def reset(self, index=None, **kwargs) -> None:
        """Resets the environments to a new initial state."""
        # metadata
        collector_metadata = self._carrier.get("collector").clone()
        if index is not None:
            # check that the env supports partial reset
            if prod(self.env.batch_size) == 0:
                raise RuntimeError("resetting unique env with index is not permitted.")
            for reset_key, done_keys in zip(
                self.env.reset_keys, self.env.done_keys_groups
            ):
                _reset = torch.zeros(
                    self.env.full_done_spec[done_keys[0]].shape,
                    dtype=torch.bool,
                    device=self.env.device,
                )
                _reset[index] = 1
                self._carrier.set(reset_key, _reset)
        else:
            _reset = None
            self._carrier.zero_()

        self._carrier.update(self.env.reset(**kwargs), inplace=True)
        collector_metadata["traj_ids"] = (
            collector_metadata["traj_ids"] - collector_metadata["traj_ids"].min()
        )
        self._carrier["collector"] = collector_metadata

    def shutdown(
        self,
        timeout: float | None = None,
        close_env: bool = True,
        raise_on_error: bool = True,
    ) -> None:
        """Shuts down all workers and/or closes the local environment.

        Args:
            timeout (float, optional): The timeout for closing pipes between workers.
                No effect for this class.
            close_env (bool, optional): Whether to close the environment. Defaults to `True`.
            raise_on_error (bool, optional): Whether to raise an error if the shutdown fails. Defaults to `True`.
        """
        try:
            if not self.closed:
                self.closed = True
                del self._carrier
                if self._use_buffers:
                    del self._final_rollout
                if close_env and not self.env.is_closed:
                    self.env.close(raise_if_closed=raise_on_error)
                del self.env
            return
        except Exception as e:
            if raise_on_error:
                raise e
            else:
                pass

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            # an AttributeError will typically be raised if the collector is deleted when the program ends.
            # In the future, insignificant changes to the close method may change the error type.
            # We excplicitely assume that any error raised during closure in
            # __del__ will not affect the program.
            pass

    def state_dict(self) -> OrderedDict:
        """Returns the local state_dict of the data collector (environment and policy).

        Returns:
            an ordered dictionary with fields :obj:`"policy_state_dict"` and
            `"env_state_dict"`.

        """
        from torchrl.envs.batched_envs import BatchedEnvBase

        if isinstance(self.env, TransformedEnv):
            env_state_dict = self.env.transform.state_dict()
        elif isinstance(self.env, BatchedEnvBase):
            env_state_dict = self.env.state_dict()
        else:
            env_state_dict = OrderedDict()

        if hasattr(self, "_policy_w_state_dict"):
            policy_state_dict = self._policy_w_state_dict.state_dict()
            state_dict = OrderedDict(
                policy_state_dict=policy_state_dict,
                env_state_dict=env_state_dict,
            )
        else:
            state_dict = OrderedDict(env_state_dict=env_state_dict)

        state_dict.update({"frames": self._frames, "iter": self._iter})

        return state_dict

    def load_state_dict(self, state_dict: OrderedDict, **kwargs) -> None:
        """Loads a state_dict on the environment and policy.

        Args:
            state_dict (OrderedDict): ordered dictionary containing the fields
                `"policy_state_dict"` and :obj:`"env_state_dict"`.

        """
        strict = kwargs.get("strict", True)
        if strict or "env_state_dict" in state_dict:
            self.env.load_state_dict(state_dict["env_state_dict"], **kwargs)
        if strict or "policy_state_dict" in state_dict:
            if not hasattr(self, "_policy_w_state_dict"):
                raise ValueError(
                    "Underlying policy does not have state_dict to load policy_state_dict into."
                )
            self._policy_w_state_dict.load_state_dict(
                state_dict["policy_state_dict"], **kwargs
            )
        self._frames = state_dict["frames"]
        self._iter = state_dict["iter"]

    def __repr__(self) -> str:
        try:
            env_str = indent(f"env={self.env}", 4 * " ")
            policy_str = indent(f"policy={self._wrapped_policy}", 4 * " ")
            td_out_str = repr(getattr(self, "_final_rollout", None))
            if len(td_out_str) > 50:
                td_out_str = td_out_str[:50] + "..."
            td_out_str = indent(f"td_out={td_out_str}", 4 * " ")
            string = (
                f"{self.__class__.__name__}("
                f"\n{env_str},"
                f"\n{policy_str},"
                f"\n{td_out_str},"
                f"\nexploration={self.exploration_type})"
            )
            return string
        except Exception:
            return f"{type(self).__name__}(not_init)"

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
        """Get an attribute from the policy."""
        # send command to policy to return the attr
        return getattr(self._wrapped_policy, attr)

    def getattr_env(self, attr):
        """Get an attribute from the environment."""
        # send command to env to return the attr
        return getattr(self.env, attr)

    def getattr_rb(self, attr):
        """Get an attribute from the replay buffer."""
        # send command to rb to return the attr
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
            # Return the unwrapped policy instance for weight synchronization
            # The unwrapped policy has the same parameter structure as what's
            # extracted in the main process, avoiding key mismatches when
            # the policy is auto-wrapped (e.g., WrappablePolicy -> TensorDictModule)
            if hasattr(self, "policy") and self.policy is not None:
                return self.policy
            else:
                raise ValueError(f"No policy found for model_id '{model_id}'")
        else:
            return _resolve_model(self, model_id)

    def _receive_weights_scheme(self):
        return super()._receive_weights_scheme()
