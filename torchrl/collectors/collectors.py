# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import _pickle
import abc

import contextlib

import functools
import os
import queue
import sys
import time
import typing
import warnings
from collections import defaultdict, OrderedDict
from copy import deepcopy
from multiprocessing import connection, queues
from multiprocessing.managers import SyncManager
from textwrap import indent
from typing import Any, Callable, Dict, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

import torch
import torch.nn as nn
from tensordict import (
    LazyStackedTensorDict,
    TensorDict,
    TensorDictBase,
    TensorDictParams,
)
from tensordict.base import NO_DEFAULT
from tensordict.nn import CudaGraphModule, TensorDictModule
from tensordict.utils import Buffer
from torch import multiprocessing as mp
from torch.nn import Parameter
from torch.utils.data import IterableDataset

from torchrl._utils import (
    _check_for_faulty_process,
    _ends_with,
    _make_ordinal_device,
    _ProcessNoWarn,
    _replace_last,
    accept_remote_rref_udf_invocation,
    logger as torchrl_logger,
    prod,
    RL_WARNINGS,
    VERBOSE,
)
from torchrl.collectors.utils import split_trajectories
from torchrl.data import ReplayBuffer
from torchrl.data.tensor_specs import TensorSpec
from torchrl.data.utils import CloudpickleWrapper, DEVICE_TYPING
from torchrl.envs.common import _do_nothing, EnvBase
from torchrl.envs.env_creator import EnvCreator
from torchrl.envs.transforms import StepCounter, TransformedEnv
from torchrl.envs.utils import (
    _aggregate_end_of_traj,
    _make_compatible_policy,
    ExplorationType,
    RandomPolicy,
    set_exploration_type,
)

_TIMEOUT = 1.0
INSTANTIATE_TIMEOUT = 20
_MIN_TIMEOUT = 1e-3  # should be several orders of magnitude inferior wrt time spent collecting a trajectory
# MAX_IDLE_COUNT is the maximum number of times a Dataloader worker can timeout with his queue.
_MAX_IDLE_COUNT = int(os.environ.get("MAX_IDLE_COUNT", 1000))

DEFAULT_EXPLORATION_TYPE: ExplorationType = ExplorationType.RANDOM

_is_osx = sys.platform.startswith("darwin")


class _Interruptor:
    """A class for managing the collection state of a process.

    This class provides methods to start and stop collection, and to check
    whether collection has been stopped. The collection state is protected
    by a lock to ensure thread-safety.
    """

    # interrupter vs interruptor: google trends seems to indicate that "or" is more
    # widely used than "er" even if my IDE complains about that...
    def __init__(self):
        self._collect = True
        self._lock = mp.Lock()

    def start_collection(self):
        with self._lock:
            self._collect = True

    def stop_collection(self):
        with self._lock:
            self._collect = False

    def collection_stopped(self):
        with self._lock:
            return self._collect is False


class _InterruptorManager(SyncManager):
    """A custom SyncManager for managing the collection state of a process.

    This class extends the SyncManager class and allows to share an Interruptor object
    between processes.
    """

    pass


_InterruptorManager.register("_Interruptor", _Interruptor)


def recursive_map_to_cpu(dictionary: OrderedDict) -> OrderedDict:
    """Maps the tensors to CPU through a nested dictionary."""
    return OrderedDict(
        **{
            k: recursive_map_to_cpu(item)
            if isinstance(item, OrderedDict)
            else item.cpu()
            if isinstance(item, torch.Tensor)
            else item
            for k, item in dictionary.items()
        }
    )


class DataCollectorBase(IterableDataset, metaclass=abc.ABCMeta):
    """Base class for data collectors."""

    _iterator = None
    total_frames: int
    frames_per_batch: int
    trust_policy: bool
    compiled_policy: bool
    cudagraphed_policy: bool

    def _get_policy_and_device(
        self,
        policy: Callable[[Any], Any] | None = None,
        observation_spec: TensorSpec = None,
        policy_device: Any = NO_DEFAULT,
        env_maker: Any | None = None,
        env_maker_kwargs: dict | None = None,
    ) -> Tuple[TensorDictModule, Union[None, Callable[[], dict]]]:
        """Util method to get a policy and its device given the collector __init__ inputs.

        Args:
            policy (TensorDictModule, optional): a policy to be used
            observation_spec (TensorSpec, optional): spec of the observations
            policy_device (torch.device, optional): the device where the policy should be placed.
                Defaults to self.policy_device
            env_maker (a callable or a batched env, optional): the env_maker function for this device/policy pair.
            env_maker_kwargs (a dict, optional): the env_maker function kwargs.

        """
        if policy_device is NO_DEFAULT:
            policy_device = self.policy_device

        if not self.trust_policy:
            env = getattr(self, "env", None)
            policy = _make_compatible_policy(
                policy,
                observation_spec,
                env=env,
                env_maker=env_maker,
                env_maker_kwargs=env_maker_kwargs,
            )
        if not policy_device:
            return policy, None

        if isinstance(policy, nn.Module):
            param_and_buf = TensorDict.from_module(policy, as_module=True)
        else:
            # Because we want to reach the warning
            param_and_buf = TensorDict()

        i = -1
        for p in param_and_buf.values(True, True):
            i += 1
            if p.device != policy_device:
                # Then we need casting
                break
        else:
            if i == -1 and not self.trust_policy:
                # We trust that the policy policy device is adequate
                warnings.warn(
                    "A policy device was provided but no parameter/buffer could be found in "
                    "the policy. Casting to policy_device is therefore impossible. "
                    "The collector will trust that the devices match. To suppress this "
                    "warning, set `trust_policy=True` when building the collector."
                )
            return policy, None

        def map_weight(
            weight,
            policy_device=policy_device,
        ):

            is_param = isinstance(weight, Parameter)
            is_buffer = isinstance(weight, Buffer)
            weight = weight.data
            if weight.device != policy_device:
                weight = weight.to(policy_device)
            elif weight.device.type in ("cpu", "mps"):
                weight = weight.share_memory_()
            if is_param:
                weight = Parameter(weight, requires_grad=False)
            elif is_buffer:
                weight = Buffer(weight)
            return weight

        # Create a stateless policy, then populate this copy with params on device
        get_original_weights = functools.partial(TensorDict.from_module, policy)
        with param_and_buf.to("meta").to_module(policy):
            policy = deepcopy(policy)

        param_and_buf.apply(
            functools.partial(map_weight),
            filter_empty=False,
        ).to_module(policy)
        return policy, get_original_weights

    def update_policy_weights_(
        self, policy_weights: Optional[TensorDictBase] = None
    ) -> None:
        """Updates the policy weights if the policy of the data collector and the trained policy live on different devices.

        Args:
            policy_weights (TensorDictBase, optional): if provided, a TensorDict containing
                the weights of the policy to be used for the udpdate.

        """
        if policy_weights is not None:
            self.policy_weights.data.update_(policy_weights)
        elif self.get_weights_fn is not None:
            self.policy_weights.data.update_(self.get_weights_fn())

    def __iter__(self) -> Iterator[TensorDictBase]:
        yield from self.iterator()

    def next(self):
        try:
            if self._iterator is None:
                self._iterator = iter(self)
            out = next(self._iterator)
            # if any, we don't want the device ref to be passed in distributed settings
            out.clear_device_()
            return out
        except StopIteration:
            return None

    @abc.abstractmethod
    def shutdown(self):
        raise NotImplementedError

    @abc.abstractmethod
    def iterator(self) -> Iterator[TensorDictBase]:
        raise NotImplementedError

    @abc.abstractmethod
    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def state_dict(self) -> OrderedDict:
        raise NotImplementedError

    @abc.abstractmethod
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        raise NotImplementedError

    def _read_compile_kwargs(self, compile_policy, cudagraph_policy):
        self.compiled_policy = compile_policy not in (False, None)
        self.cudagraphed_policy = cudagraph_policy not in (False, None)
        self.compiled_policy_kwargs = (
            {} if not isinstance(compile_policy, typing.Mapping) else compile_policy
        )
        self.cudagraphed_policy_kwargs = (
            {} if not isinstance(cudagraph_policy, typing.Mapping) else cudagraph_policy
        )

    def __repr__(self) -> str:
        string = f"{self.__class__.__name__}()"
        return string

    def __class_getitem__(self, index):
        raise NotImplementedError

    def __len__(self) -> int:
        if self.total_frames > 0:
            return -(self.total_frames // -self.frames_per_batch)
        raise RuntimeError("Non-terminating collectors do not have a length")


@accept_remote_rref_udf_invocation
class SyncDataCollector(DataCollectorBase):
    """Generic data collector for RL problems. Requires an environment constructor and a policy.

    Args:
        create_env_fn (Callable): a callable that returns an instance of
            :class:`~torchrl.envs.EnvBase` class.
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
        replay_buffer (ReplayBuffer, optional): if provided, the collector will not yield tensordict
            but populate the buffer instead. Defaults to ``None``.
        trust_policy (bool, optional): if ``True``, a non-TensorDictModule policy will be trusted to be
            assumed to be compatible with the collector. This defaults to ``True`` for CudaGraphModules
            and ``False`` otherwise.
        compile_policy (bool or Dict[str, Any], optional): if ``True``, the policy will be compiled
            using :func:`~torch.compile` default behaviour. If a dictionary of kwargs is passed, it
            will be used to compile the policy.
        cudagraph_policy (bool or Dict[str, Any], optional): if ``True``, the policy will be wrapped
            in :class:`~tensordict.nn.CudaGraphModule` with default kwargs.
            If a dictionary of kwargs is passed, it will be used to wrap the policy.

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

    def __init__(
        self,
        create_env_fn: Union[
            EnvBase, "EnvCreator", Sequence[Callable[[], EnvBase]]  # noqa: F821
        ],  # noqa: F821
        policy: Optional[
            Union[
                TensorDictModule,
                Callable[[TensorDictBase], TensorDictBase],
            ]
        ] = None,
        *,
        frames_per_batch: int,
        total_frames: int = -1,
        device: DEVICE_TYPING = None,
        storing_device: DEVICE_TYPING = None,
        policy_device: DEVICE_TYPING = None,
        env_device: DEVICE_TYPING = None,
        create_env_kwargs: dict | None = None,
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
        trust_policy: bool = None,
        compile_policy: bool | Dict[str, Any] | None = None,
        cudagraph_policy: bool | Dict[str, Any] | None = None,
        **kwargs,
    ):
        from torchrl.envs.batched_envs import BatchedEnvBase

        self.closed = True
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

        if policy is None:

            policy = RandomPolicy(env.full_action_spec)
        if trust_policy is None:
            trust_policy = isinstance(policy, (RandomPolicy, CudaGraphModule))
        self.trust_policy = trust_policy
        self._read_compile_kwargs(compile_policy, cudagraph_policy)

        ##########################
        # Trajectory pool
        self._traj_pool_val = kwargs.pop("traj_pool", None)
        if kwargs:
            raise TypeError(
                f"Keys {list(kwargs.keys())} are unknown to {type(self).__name__}."
            )

        ##########################
        # Setting devices:
        # The rule is the following:
        # - If no device is passed, all devices are assumed to work OOB.
        #   The tensordict used for output is not on any device (ie, actions and observations
        #   can be on a different device).
        # - If the ``device`` is passed, it is used for all devices (storing, env and policy)
        #   unless overridden by another kwarg.
        # - The rest of the kwargs control the respective device.
        storing_device, policy_device, env_device = self._get_devices(
            storing_device=storing_device,
            policy_device=policy_device,
            env_device=env_device,
            device=device,
        )

        self.storing_device = storing_device
        if self.storing_device is not None and self.storing_device.type != "cuda":
            # Cuda handles sync
            if torch.cuda.is_available():
                self._sync_storage = torch.cuda.synchronize
            elif torch.backends.mps.is_available() and hasattr(torch, "mps"):
                # Will break for older PT versions which don't have torch.mps
                self._sync_storage = torch.mps.synchronize
            elif self.storing_device.type == "cpu":
                self._sync_storage = _do_nothing
            else:
                raise RuntimeError("Non supported device")
        else:
            self._sync_storage = _do_nothing

        self.env_device = env_device
        if self.env_device is not None and self.env_device.type != "cuda":
            # Cuda handles sync
            if torch.cuda.is_available():
                self._sync_env = torch.cuda.synchronize
            elif torch.backends.mps.is_available() and hasattr(torch, "mps"):
                self._sync_env = torch.mps.synchronize
            elif self.env_device.type == "cpu":
                self._sync_env = _do_nothing
            else:
                raise RuntimeError("Non supported device")
        else:
            self._sync_env = _do_nothing
        self.policy_device = policy_device
        if self.policy_device is not None and self.policy_device.type != "cuda":
            # Cuda handles sync
            if torch.cuda.is_available():
                self._sync_policy = torch.cuda.synchronize
            elif torch.backends.mps.is_available() and hasattr(torch, "mps"):
                self._sync_policy = torch.mps.synchronize
            elif self.policy_device.type == "cpu":
                self._sync_policy = _do_nothing
            else:
                raise RuntimeError("Non supported device")
        else:
            self._sync_policy = _do_nothing
        self.device = device
        # Check if we need to cast things from device to device
        # If the policy has a None device and the env too, no need to cast (we don't know
        # and assume the user knows what she's doing).
        # If the devices match we're happy too.
        # Only if the values differ we need to cast
        self._cast_to_policy_device = self.policy_device != self.env_device

        self.env: EnvBase = env
        del env
        self.replay_buffer = replay_buffer
        if self.replay_buffer is not None:
            if postproc is not None:
                raise TypeError("postproc must be None when a replay buffer is passed.")
            if use_buffers:
                raise TypeError("replay_buffer is exclusive with use_buffers.")
        if use_buffers is None:
            use_buffers = not self.env._has_dynamic_specs and self.replay_buffer is None
        self._use_buffers = use_buffers
        self.replay_buffer = replay_buffer

        self.closed = False

        if not reset_when_done:
            raise ValueError("reset_when_done is deprectated.")
        self.reset_when_done = reset_when_done
        self.n_env = self.env.batch_size.numel()

        (self.policy, self.get_weights_fn,) = self._get_policy_and_device(
            policy=policy,
            observation_spec=self.env.observation_spec,
        )
        if isinstance(self.policy, nn.Module):
            self.policy_weights = TensorDict.from_module(self.policy, as_module=True)
        else:
            self.policy_weights = TensorDict()

        if self.compiled_policy:
            self.policy = torch.compile(self.policy, **self.compiled_policy_kwargs)
        if self.cudagraphed_policy:
            self.policy = CudaGraphModule(self.policy, **self.cudagraphed_policy_kwargs)

        if self.env_device:
            self.env: EnvBase = self.env.to(self.env_device)
        elif self.env.device is not None:
            # we did not receive an env device, we use the device of the env
            self.env_device = self.env.device

        # If the storing device is not the same as the policy device, we have
        # no guarantee that the "next" entry from the policy will be on the
        # same device as the collector metadata.
        self._cast_to_env_device = self._cast_to_policy_device or (
            self.env.device != self.storing_device
        )

        self.max_frames_per_traj = (
            int(max_frames_per_traj) if max_frames_per_traj is not None else 0
        )
        if self.max_frames_per_traj is not None and self.max_frames_per_traj > 0:
            # let's check that there is no StepCounter yet
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

        if total_frames is None or total_frames < 0:
            total_frames = float("inf")
        else:
            remainder = total_frames % frames_per_batch
            if remainder != 0 and RL_WARNINGS:
                warnings.warn(
                    f"total_frames ({total_frames}) is not exactly divisible by frames_per_batch ({frames_per_batch})."
                    f"This means {frames_per_batch - remainder} additional frames will be collected."
                    "To silence this message, set the environment variable RL_WARNINGS to False."
                )
        self.total_frames = (
            int(total_frames) if total_frames != float("inf") else total_frames
        )
        self.reset_at_each_iter = reset_at_each_iter
        self.init_random_frames = (
            int(init_random_frames) if init_random_frames is not None else 0
        )
        if (
            init_random_frames is not None
            and init_random_frames % frames_per_batch != 0
            and RL_WARNINGS
        ):
            warnings.warn(
                f"init_random_frames ({init_random_frames}) is not exactly a multiple of frames_per_batch ({frames_per_batch}), "
                f" this results in more init_random_frames than requested"
                f" ({-(-init_random_frames // frames_per_batch) * frames_per_batch})."
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )

        self.postproc = postproc
        if (
            self.postproc is not None
            and hasattr(self.postproc, "to")
            and self.storing_device
        ):
            self.postproc.to(self.storing_device)
        if frames_per_batch % self.n_env != 0 and RL_WARNINGS:
            warnings.warn(
                f"frames_per_batch ({frames_per_batch}) is not exactly divisible by the number of batched environments ({self.n_env}), "
                f" this results in more frames_per_batch per iteration that requested"
                f" ({-(-frames_per_batch // self.n_env) * self.n_env})."
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )
        self.requested_frames_per_batch = int(frames_per_batch)
        self.frames_per_batch = -(-frames_per_batch // self.n_env)
        self.exploration_type = (
            exploration_type if exploration_type else DEFAULT_EXPLORATION_TYPE
        )
        self.return_same_td = return_same_td
        self.set_truncated = set_truncated

        self._make_shuttle()
        if self._use_buffers:
            self._make_final_rollout()
        self._set_truncated_keys()

        if split_trajs is None:
            split_trajs = False
        self.split_trajs = split_trajs
        self._exclude_private_keys = True

        self.interruptor = interruptor
        self._frames = 0
        self._iter = -1

    @property
    def _traj_pool(self):
        pool = getattr(self, "_traj_pool_val", None)
        if pool is None:
            pool = self._traj_pool_val = _TrajectoryPool()
        return pool

    def _make_shuttle(self):
        # Shuttle is a deviceless tensordict that just carried data from env to policy and policy to env
        with torch.no_grad():
            self._shuttle = self.env.reset()
        if self.policy_device != self.env_device or self.env_device is None:
            self._shuttle_has_no_device = True
            self._shuttle.clear_device_()
        else:
            self._shuttle_has_no_device = False

        traj_ids = self._traj_pool.get_traj_and_increment(
            self.n_env, device=self.storing_device
        ).view(self.env.batch_size)
        self._shuttle.set(
            ("collector", "traj_ids"),
            traj_ids,
        )

    def _make_final_rollout(self):
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

        # If the policy has a valid spec, we use it
        self._policy_output_keys = set()
        if (
            hasattr(self.policy, "spec")
            and self.policy.spec is not None
            and all(v is not None for v in self.policy.spec.values(True, True))
        ):
            if any(
                key not in self._final_rollout.keys(isinstance(key, tuple))
                for key in self.policy.spec.keys(True, True)
            ):
                # if policy spec is non-empty, all the values are not None and the keys
                # match the out_keys we assume the user has given all relevant information
                # the policy could have more keys than the env:
                policy_spec = self.policy.spec
                if policy_spec.ndim < self._final_rollout.ndim:
                    policy_spec = policy_spec.expand(self._final_rollout.shape)
                for key, spec in policy_spec.items(True, True):
                    self._policy_output_keys.add(key)
                    if key in self._final_rollout.keys(True):
                        continue
                    self._final_rollout.set(key, spec.zero())

        else:
            # otherwise, we perform a small number of steps with the policy to
            # determine the relevant keys with which to pre-populate _final_rollout.
            # This is the safest thing to do if the spec has None fields or if there is
            # no spec at all.
            # See #505 for additional context.
            self._final_rollout.update(self._shuttle.copy())
            with torch.no_grad():
                policy_input = self._shuttle.copy()
                if self.policy_device:
                    policy_input = policy_input.to(self.policy_device)
                # we cast to policy device, we'll deal with the device later
                policy_input_copy = policy_input.copy()
                policy_input_clone = (
                    policy_input.clone()
                )  # to test if values have changed in-place
                policy_output = self.policy(policy_input)

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
                self._final_rollout.update(
                    policy_output.select(*self._policy_output_keys)
                )
                del filtered_policy_output, policy_output, policy_input

        _env_output_keys = []
        for spec in ["full_observation_spec", "full_done_spec", "full_reward_spec"]:
            _env_output_keys += list(self.env.output_spec[spec].keys(True, True))
        self._env_output_keys = _env_output_keys
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
        self, policy_weights: Optional[TensorDictBase] = None
    ) -> None:
        super().update_policy_weights_(policy_weights)

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
        if self.storing_device and self.storing_device.type == "cuda":
            stream = torch.cuda.Stream(self.storing_device, priority=-1)
            event = stream.record_event()
            streams = [stream]
            events = [event]
        elif self.storing_device is None:
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
                    if spec.device.type == "cuda":
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
                tensordict_out = self.rollout()
                if tensordict_out is None:
                    # if a replay buffer is passed, there is no tensordict_out
                    #  frames are updated within the rollout function
                    yield
                    continue
                self._increment_frames(tensordict_out.numel())

                if self.split_trajs:
                    tensordict_out = split_trajectories(
                        tensordict_out, prefix="collector"
                    )
                if self.postproc is not None:
                    tensordict_out = self.postproc(tensordict_out)
                if self._exclude_private_keys:

                    def is_private(key):
                        if isinstance(key, str) and key.startswith("_"):
                            return True
                        if isinstance(key, tuple) and any(
                            _key.startswith("_") for _key in key
                        ):
                            return True
                        return False

                    excluded_keys = [
                        key for key in tensordict_out.keys(True) if is_private(key)
                    ]
                    tensordict_out = tensordict_out.exclude(
                        *excluded_keys, inplace=True
                    )
                if self.return_same_td:
                    # This is used with multiprocessed collectors to use the buffers
                    # stored in the tensordict.
                    if events:
                        for event in events:
                            event.record()
                            event.synchronize()
                    yield tensordict_out
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

    def _update_traj_ids(self, env_output) -> None:
        # we can't use the reset keys because they're gone
        traj_sop = _aggregate_end_of_traj(
            env_output.get("next"), done_keys=self.env.done_keys
        )
        if traj_sop.any():
            device = self.storing_device

            traj_ids = self._shuttle.get(("collector", "traj_ids"))
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
            self._shuttle.set(("collector", "traj_ids"), traj_ids)

    @torch.no_grad()
    def rollout(self) -> TensorDictBase:
        """Computes a rollout in the environment using the provided policy.

        Returns:
            TensorDictBase containing the computed rollout.

        """
        if self.reset_at_each_iter:
            self._shuttle.update(self.env.reset())

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
                    self.env.rand_action(self._shuttle)
                else:
                    if self._cast_to_policy_device:
                        if self.policy_device is not None:
                            policy_input = self._shuttle.to(
                                self.policy_device, non_blocking=True
                            )
                            self._sync_policy()
                        elif self.policy_device is None:
                            # we know the tensordict has a device otherwise we would not be here
                            # we can pass this, clear_device_ must have been called earlier
                            # policy_input = self._shuttle.clear_device_()
                            policy_input = self._shuttle
                    else:
                        policy_input = self._shuttle
                    # we still do the assignment for security
                    policy_output = self.policy(policy_input)
                    if self._shuttle is not policy_output:
                        # ad-hoc update shuttle
                        self._shuttle.update(
                            policy_output, keys_to_update=self._policy_output_keys
                        )

                if self._cast_to_env_device:
                    if self.env_device is not None:
                        env_input = self._shuttle.to(self.env_device, non_blocking=True)
                        self._sync_env()
                    elif self.env_device is None:
                        # we know the tensordict has a device otherwise we would not be here
                        # we can pass this, clear_device_ must have been called earlier
                        # env_input = self._shuttle.clear_device_()
                        env_input = self._shuttle
                else:
                    env_input = self._shuttle
                env_output, env_next_output = self.env.step_and_maybe_reset(env_input)

                if self._shuttle is not env_output:
                    # ad-hoc update shuttle
                    next_data = env_output.get("next")
                    if self._shuttle_has_no_device:
                        # Make sure
                        next_data.clear_device_()
                    self._shuttle.set("next", next_data)

                if self.replay_buffer is not None:
                    self.replay_buffer.add(self._shuttle)
                    if self._increment_frames(self._shuttle.numel()):
                        return
                else:
                    if self.storing_device is not None:
                        tensordicts.append(
                            self._shuttle.to(self.storing_device, non_blocking=True)
                        )
                        self._sync_storage()
                    else:
                        tensordicts.append(self._shuttle)

                # carry over collector data without messing up devices
                collector_data = self._shuttle.get("collector").copy()
                self._shuttle = env_next_output
                if self._shuttle_has_no_device:
                    self._shuttle.clear_device_()
                self._shuttle.set("collector", collector_data)
                self._update_traj_ids(env_output)

                if (
                    self.interruptor is not None
                    and self.interruptor.collection_stopped()
                ):
                    if self.replay_buffer is not None:
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
                elif self.replay_buffer is not None:
                    return
                else:
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
        collector_metadata = self._shuttle.get("collector").clone()
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
                self._shuttle.set(reset_key, _reset)
        else:
            _reset = None
            self._shuttle.zero_()

        self._shuttle.update(self.env.reset(**kwargs), inplace=True)
        collector_metadata["traj_ids"] = (
            collector_metadata["traj_ids"] - collector_metadata["traj_ids"].min()
        )
        self._shuttle["collector"] = collector_metadata

    def shutdown(self) -> None:
        """Shuts down all workers and/or closes the local environment."""
        if not self.closed:
            self.closed = True
            del self._shuttle
            if self._use_buffers:
                del self._final_rollout
            if not self.env.is_closed:
                self.env.close()
            del self.env
        return

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

        if hasattr(self.policy, "state_dict"):
            policy_state_dict = self.policy.state_dict()
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
            self.policy.load_state_dict(state_dict["policy_state_dict"], **kwargs)
        self._frames = state_dict["frames"]
        self._iter = state_dict["iter"]

    def __repr__(self) -> str:
        env_str = indent(f"env={self.env}", 4 * " ")
        policy_str = indent(f"policy={self.policy}", 4 * " ")
        td_out_str = indent(f"td_out={self._final_rollout}", 4 * " ")
        string = (
            f"{self.__class__.__name__}("
            f"\n{env_str},"
            f"\n{policy_str},"
            f"\n{td_out_str},"
            f"\nexploration={self.exploration_type})"
        )
        return string


class _MultiDataCollector(DataCollectorBase):
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
            - In all other cases an attempt to wrap it will be undergone as such: ``TensorDictModule(policy, in_keys=env_obs_key, out_keys=env.action_keys)``.

    Keyword Args:
        frames_per_batch (int): A keyword-only argument representing the
            total number of elements in a batch.
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
        update_at_each_batch (boolm optional): if ``True``, :meth:`~.update_policy_weight_()`
            will be called before (sync) or after (async) each data collection.
            Defaults to ``False``.
        preemptive_threshold (float, optional): a value between 0.0 and 1.0 that specifies the ratio of workers
            that will be allowed to finished collecting their rollout before the rest are forced to end early.
        num_threads (int, optional): number of threads for this process.
            Defaults to the number of workers.
        num_sub_threads (int, optional): number of threads of the subprocesses.
            Should be equal to one plus the number of processes launched within
            each subprocess (or one if a single process is launched).
            Defaults to 1 for safety: if none is indicated, launching multiple
            workers may charge the cpu load too much and harm performance.
        cat_results (str, int or None): (:class:`~torchrl.collectors.MultiSyncDataCollector` exclusively).
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
        replay_buffer (ReplayBuffer, optional): if provided, the collector will not yield tensordict
            but populate the buffer instead. Defaults to ``None``.
        trust_policy (bool, optional): if ``True``, a non-TensorDictModule policy will be trusted to be
            assumed to be compatible with the collector. This defaults to ``True`` for CudaGraphModules
            and ``False`` otherwise.
        compile_policy (bool or Dict[str, Any], optional): if ``True``, the policy will be compiled
            using :func:`~torch.compile` default behaviour. If a dictionary of kwargs is passed, it
            will be used to compile the policy.
        cudagraph_policy (bool or Dict[str, Any], optional): if ``True``, the policy will be wrapped
            in :class:`~tensordict.nn.CudaGraphModule` with default kwargs.
            If a dictionary of kwargs is passed, it will be used to wrap the policy.

    """

    def __init__(
        self,
        create_env_fn: Sequence[Callable[[], EnvBase]],
        policy: Optional[
            Union[
                TensorDictModule,
                Callable[[TensorDictBase], TensorDictBase],
            ]
        ] = None,
        *,
        frames_per_batch: int,
        total_frames: Optional[int] = -1,
        device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        storing_device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        env_device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        policy_device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        create_env_kwargs: Optional[Sequence[dict]] = None,
        max_frames_per_traj: int | None = None,
        init_random_frames: int | None = None,
        reset_at_each_iter: bool = False,
        postproc: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        split_trajs: Optional[bool] = None,
        exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
        reset_when_done: bool = True,
        update_at_each_batch: bool = False,
        preemptive_threshold: float = None,
        num_threads: int = None,
        num_sub_threads: int = 1,
        cat_results: str | int | None = None,
        set_truncated: bool = False,
        use_buffers: bool | None = None,
        replay_buffer: ReplayBuffer | None = None,
        replay_buffer_chunk: bool = True,
        trust_policy: bool = None,
        compile_policy: bool | Dict[str, Any] | None = None,
        cudagraph_policy: bool | Dict[str, Any] | None = None,
    ):
        self.closed = True
        self.num_workers = len(create_env_fn)

        self.set_truncated = set_truncated
        self.num_sub_threads = num_sub_threads
        self.num_threads = num_threads
        self.create_env_fn = create_env_fn
        self._read_compile_kwargs(compile_policy, cudagraph_policy)
        self.create_env_kwargs = (
            create_env_kwargs
            if create_env_kwargs is not None
            else [{} for _ in range(self.num_workers)]
        )
        # Preparing devices:
        # We want the user to be able to choose, for each worker, on which
        # device will the policy live and which device will be used to store
        # data. Those devices may or may not match.
        # One caveat is that, if there is only one device for the policy, and
        # if there are multiple workers, sending the same device and policy
        # to be copied to each worker will result in multiple copies of the
        # same policy on the same device.
        # To go around this, we do the copies of the policy in the server
        # (this object) to each possible device, and send to all the
        # processes their copy of the policy.

        storing_devices, policy_devices, env_devices = self._get_devices(
            storing_device=storing_device,
            env_device=env_device,
            policy_device=policy_device,
            device=device,
        )

        # to avoid confusion
        self.storing_device = storing_devices
        self.policy_device = policy_devices
        self.env_device = env_devices

        del storing_device, env_device, policy_device, device

        self._use_buffers = use_buffers
        self.replay_buffer = replay_buffer
        self._check_replay_buffer_init()
        self.replay_buffer_chunk = replay_buffer_chunk
        if (
            replay_buffer is not None
            and hasattr(replay_buffer, "shared")
            and not replay_buffer.shared
        ):
            replay_buffer.share()

        self._policy_weights_dict = {}
        self._get_weights_fn_dict = {}

        if trust_policy is None:
            trust_policy = isinstance(policy, CudaGraphModule)
        self.trust_policy = trust_policy

        for policy_device, env_maker, env_maker_kwargs in zip(
            self.policy_device, self.create_env_fn, self.create_env_kwargs
        ):
            (policy_copy, get_weights_fn,) = self._get_policy_and_device(
                policy=policy,
                policy_device=policy_device,
                env_maker=env_maker,
                env_maker_kwargs=env_maker_kwargs,
            )
            if type(policy_copy) is not type(policy):
                policy = policy_copy
            weights = (
                TensorDict.from_module(policy_copy)
                if isinstance(policy_copy, nn.Module)
                else TensorDict()
            )
            self._policy_weights_dict[policy_device] = weights
            self._get_weights_fn_dict[policy_device] = get_weights_fn
        self.policy = policy

        if total_frames is None or total_frames < 0:
            total_frames = float("inf")
        else:
            remainder = total_frames % frames_per_batch
            if remainder != 0 and RL_WARNINGS:
                warnings.warn(
                    f"total_frames ({total_frames}) is not exactly divisible by frames_per_batch ({frames_per_batch})."
                    f"This means {frames_per_batch - remainder} additional frames will be collected."
                    "To silence this message, set the environment variable RL_WARNINGS to False."
                )
        self.total_frames = (
            int(total_frames) if total_frames != float("inf") else total_frames
        )
        self.reset_at_each_iter = reset_at_each_iter
        self.postprocs = postproc
        self.max_frames_per_traj = (
            int(max_frames_per_traj) if max_frames_per_traj is not None else 0
        )
        self.requested_frames_per_batch = int(frames_per_batch)
        self.reset_when_done = reset_when_done
        if split_trajs is None:
            split_trajs = False
        elif not self.reset_when_done and split_trajs:
            raise RuntimeError(
                "Cannot split trajectories when reset_when_done is False."
            )
        self.split_trajs = split_trajs
        self.init_random_frames = (
            int(init_random_frames) if init_random_frames is not None else 0
        )
        self.update_at_each_batch = update_at_each_batch
        self.exploration_type = exploration_type
        self.frames_per_worker = np.inf
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
        self._run_processes()
        self._exclude_private_keys = True
        self._frames = 0
        self._iter = -1
        if cat_results is not None and (
            not isinstance(cat_results, (int, str))
            or (isinstance(cat_results, str) and cat_results != "stack")
        ):
            raise ValueError(
                "cat_results must be a string ('stack') "
                f"or an integer representing the cat dimension. Got {cat_results}."
            )
        if not isinstance(self, MultiSyncDataCollector) and cat_results not in (
            "stack",
            None,
        ):
            raise ValueError(
                "cat_results can only be used with ``MultiSyncDataCollector``."
            )
        self.cat_results = cat_results

    def _check_replay_buffer_init(self):
        if self.replay_buffer is None:
            return
        is_init = getattr(self.replay_buffer._storage, "initialized", True)
        if not is_init:
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

            self.replay_buffer.add(fake_td)
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
                SyncDataCollector._get_devices(
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

    @property
    def frames_per_batch_worker(self):
        raise NotImplementedError

    def update_policy_weights_(self, policy_weights=None) -> None:
        if isinstance(policy_weights, TensorDictParams):
            policy_weights = policy_weights.data
        for _device in self._policy_weights_dict:
            if policy_weights is not None:
                self._policy_weights_dict[_device].data.update_(policy_weights)
            elif self._get_weights_fn_dict[_device] is not None:
                original_weights = self._get_weights_fn_dict[_device]()
                if original_weights is None:
                    # if the weights match in identity, we can spare a call to update_
                    continue
                if isinstance(original_weights, TensorDictParams):
                    original_weights = original_weights.data
                self._policy_weights_dict[_device].data.update_(original_weights)

    @property
    def _queue_len(self) -> int:
        raise NotImplementedError

    def _run_processes(self) -> None:
        if self.num_threads is None:
            total_workers = self._total_workers_from_env(self.create_env_fn)
            self.num_threads = max(
                1, torch.get_num_threads() - total_workers
            )  # 1 more thread for this proc

        torch.set_num_threads(self.num_threads)
        queue_out = mp.Queue(self._queue_len)  # sends data from proc to main
        self.procs = []
        self.pipes = []
        self._traj_pool = _TrajectoryPool(lock=True)

        for i, (env_fun, env_fun_kwargs) in enumerate(
            zip(self.create_env_fn, self.create_env_kwargs)
        ):
            pipe_parent, pipe_child = mp.Pipe()  # send messages to procs
            if env_fun.__class__.__name__ != "EnvCreator" and not isinstance(
                env_fun, EnvBase
            ):  # to avoid circular imports
                env_fun = CloudpickleWrapper(env_fun)

            # Create a policy on the right device
            policy_device = self.policy_device[i]
            storing_device = self.storing_device[i]
            env_device = self.env_device[i]
            policy = self.policy
            policy_weights = self._policy_weights_dict[policy_device]
            if policy is not None and policy_weights is not None:
                cm = policy_weights.to_module(policy)
            else:
                cm = contextlib.nullcontext()
            with cm:
                kwargs = {
                    "pipe_parent": pipe_parent,
                    "pipe_child": pipe_child,
                    "queue_out": queue_out,
                    "create_env_fn": env_fun,
                    "create_env_kwargs": env_fun_kwargs,
                    "policy": policy,
                    "max_frames_per_traj": self.max_frames_per_traj,
                    "frames_per_batch": self.frames_per_batch_worker,
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
                    "replay_buffer_chunk": self.replay_buffer_chunk,
                    "traj_pool": self._traj_pool,
                    "trust_policy": self.trust_policy,
                    "compile_policy": self.compiled_policy_kwargs
                    if self.compiled_policy
                    else False,
                    "cudagraph_policy": self.cudagraphed_policy_kwargs
                    if self.cudagraphed_policy
                    else False,
                }
                proc = _ProcessNoWarn(
                    target=_main_async_collector,
                    num_threads=self.num_sub_threads,
                    kwargs=kwargs,
                )
                # proc.daemon can't be set as daemonic processes may be launched by the process itself
                try:
                    proc.start()
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
                self.pipes.append(pipe_parent)
        for pipe_parent in self.pipes:
            pipe_parent.poll(timeout=INSTANTIATE_TIMEOUT)
            msg = pipe_parent.recv()
            if msg != "instantiated":
                raise RuntimeError(msg)
        self.queue_out = queue_out
        self.closed = False

    def __del__(self):
        try:
            self.shutdown()
        except Exception:
            # an AttributeError will typically be raised if the collector is deleted when the program ends.
            # In the future, insignificant changes to the close method may change the error type.
            # We excplicitely assume that any error raised during closure in
            # __del__ will not affect the program.
            pass

    def shutdown(self) -> None:
        """Shuts down all processes. This operation is irreversible."""
        self._shutdown_main()

    def _shutdown_main(self) -> None:
        try:
            if self.closed:
                return
            _check_for_faulty_process(self.procs)
            self.closed = True
            for idx in range(self.num_workers):
                if not self.procs[idx].is_alive():
                    continue
                try:
                    self.pipes[idx].send((None, "close"))

                    if self.pipes[idx].poll(10.0):
                        msg = self.pipes[idx].recv()
                        if msg != "closed":
                            raise RuntimeError(f"got {msg} but expected 'close'")
                    else:
                        continue
                except BrokenPipeError:
                    continue

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
            >>> collector = SyncDataCollector(env_fn_parallel, policy, frames_per_batch=100, total_frames=300)
            >>> out_seed = collector.set_seed(1)  # out_seed = 6

        """
        _check_for_faulty_process(self.procs)
        for idx in range(self.num_workers):
            self.pipes[idx].send(((seed, static_seed), "seed"))
            new_seed, msg = self.pipes[idx].recv()
            if msg != "seeded":
                raise RuntimeError(f"Expected msg='seeded', got {msg}")
            seed = new_seed
        self.reset()
        return seed

    def reset(self, reset_idx: Optional[Sequence[bool]] = None) -> None:
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
                j, msg = self.pipes[idx].recv()
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
            _state_dict, msg = self.pipes[idx].recv()
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
            _, msg = self.pipes[idx].recv()
            if msg != "loaded":
                raise RuntimeError(f"Expected msg='loaded', got {msg}")
        self._frames = state_dict["frames"]
        self._iter = state_dict["iter"]


@accept_remote_rref_udf_invocation
class MultiSyncDataCollector(_MultiDataCollector):
    """Runs a given number of DataCollectors on separate processes synchronously.

    .. aafig::

            +----------------------------------------------------------------------+
            |            "MultiSyncDataCollector"                 |                |
            |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|                |
            |   "Collector 1" |  "Collector 2"  |  "Collector 3"  |     Main       |
            |~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~|
            | "env1" | "env2" | "env3" | "env4" | "env5" | "env6" |                |
            |~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~~~~~~~~~|
            |"reset" |"reset" |"reset" |"reset" |"reset" |"reset" |                |
            |        |        |        |        |        |        |                |
            |       "actor"   |        |        |       "actor"   |                |
            |                 |        |        |                 |                |
            | "step" | "step" |       "actor"   |                 |                |
            |        |        |                 |                 |                |
            |        |        |                 | "step" | "step" |                |
            |        |        |                 |        |        |                |
            |       "actor"   | "step" | "step" |       "actor"   |                |
            |                 |        |        |                 |                |
            |                 |       "actor"   |                 |                |
            |                 |                 |                 |                |
            |                       "yield batch of traj 1"------->"collect, train"|
            |                                                     |                |
            | "step" | "step" | "step" | "step" | "step" | "step" |                |
            |        |        |        |        |        |        |                |
            |       "actor"   |       "actor"   |        |        |                |
            |                 | "step" | "step" |       "actor"   |                |
            |                 |        |        |                 |                |
            | "step" | "step" |       "actor"   | "step" | "step" |                |
            |        |        |                 |        |        |                |
            |       "actor"   |                 |       "actor"   |                |
            |                       "yield batch of traj 2"------->"collect, train"|
            |                                                     |                |
            +----------------------------------------------------------------------+

    Envs can be identical or different.

    The collection starts when the next item of the collector is queried,
    and no environment step is computed in between the reception of a batch of
    trajectory and the start of the next collection.
    This class can be safely used with online RL sota-implementations.

    .. note:: Python requires multiprocessed code to be instantiated within a main guard:

            >>> from torchrl.collectors import MultiSyncDataCollector
            >>> if __name__ == "__main__":
            ...     # Create your collector here

        See https://docs.python.org/3/library/multiprocessing.html for more info.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from tensordict.nn import TensorDictModule
        >>> from torch import nn
        >>> from torchrl.collectors import MultiSyncDataCollector
        >>> if __name__ == "__main__":
        ...     env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
        ...     policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
        ...     collector = MultiSyncDataCollector(
        ...         create_env_fn=[env_maker, env_maker],
        ...         policy=policy,
        ...         total_frames=2000,
        ...         max_frames_per_traj=50,
        ...         frames_per_batch=200,
        ...         init_random_frames=-1,
        ...         reset_at_each_iter=False,
        ...         device="cpu",
        ...         storing_device="cpu",
        ...         cat_results="stack",
        ...     )
        ...     for i, data in enumerate(collector):
        ...         if i == 2:
        ...             print(data)
        ...             break
        ... collector.shutdown()
        ... del collector
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

    """

    __doc__ += _MultiDataCollector.__doc__

    # for RPC
    def next(self):
        return super().next()

    # for RPC
    def shutdown(self):
        if hasattr(self, "out_buffer"):
            del self.out_buffer
        if hasattr(self, "buffers"):
            del self.buffers
        return super().shutdown()

    # for RPC
    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        return super().set_seed(seed, static_seed)

    # for RPC
    def state_dict(self) -> OrderedDict:
        return super().state_dict()

    # for RPC
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        return super().load_state_dict(state_dict)

    # for RPC
    def update_policy_weights_(
        self, policy_weights: Optional[TensorDictBase] = None
    ) -> None:
        super().update_policy_weights_(policy_weights)

    @property
    def frames_per_batch_worker(self):
        if self.requested_frames_per_batch % self.num_workers != 0 and RL_WARNINGS:
            warnings.warn(
                f"frames_per_batch {self.requested_frames_per_batch} is not exactly divisible by the number of collector workers {self.num_workers},"
                f" this results in more frames_per_batch per iteration that requested."
                "To silence this message, set the environment variable RL_WARNINGS to False."
            )
        frames_per_batch_worker = -(
            -self.requested_frames_per_batch // self.num_workers
        )
        return frames_per_batch_worker

    @property
    def _queue_len(self) -> int:
        return self.num_workers

    def iterator(self) -> Iterator[TensorDictBase]:
        cat_results = self.cat_results
        if cat_results is None:
            cat_results = "stack"

        self.buffers = {}
        dones = [False for _ in range(self.num_workers)]
        workers_frames = [0 for _ in range(self.num_workers)]
        same_device = None
        self.out_buffer = None
        preempt = self.interruptor is not None and self.preemptive_threshold < 1.0

        while not all(dones) and self._frames < self.total_frames:
            _check_for_faulty_process(self.procs)
            if self.update_at_each_batch:
                self.update_policy_weights_()

            for idx in range(self.num_workers):
                if (
                    self.init_random_frames is not None
                    and self._frames < self.init_random_frames
                ):
                    msg = "continue_random"
                else:
                    msg = "continue"
                self.pipes[idx].send((None, msg))

            self._iter += 1

            if preempt:
                self.interruptor.start_collection()
                while self.queue_out.qsize() < int(
                    self.num_workers * self.preemptive_threshold
                ):
                    continue
                self.interruptor.stop_collection()
                # Now wait for stragglers to return
                while self.queue_out.qsize() < int(self.num_workers):
                    continue

            for _ in range(self.num_workers):
                new_data, j = self.queue_out.get()
                use_buffers = self._use_buffers
                if self.replay_buffer is not None:
                    idx = new_data
                    workers_frames[idx] = (
                        workers_frames[idx] + self.frames_per_batch_worker
                    )
                    continue
                elif j == 0 or not use_buffers:
                    try:
                        data, idx = new_data
                        self.buffers[idx] = data
                        if use_buffers is None and j > 0:
                            self._use_buffers = False
                    except TypeError:
                        if use_buffers is None:
                            self._use_buffers = True
                            idx = new_data
                        else:
                            raise
                else:
                    idx = new_data

                if preempt:
                    # mask buffers if cat, and create a mask if stack
                    if cat_results != "stack":
                        buffers = {}
                        for idx, buffer in self.buffers.items():
                            valid = buffer.get(("collector", "traj_ids")) != -1
                            if valid.ndim > 2:
                                valid = valid.flatten(0, -2)
                            if valid.ndim == 2:
                                valid = valid.any(0)
                            buffers[idx] = buffer[..., valid]
                    else:
                        for buffer in self.buffers.values():
                            with buffer.unlock_():
                                buffer.set(
                                    ("collector", "mask"),
                                    buffer.get(("collector", "traj_ids")) != -1,
                                )
                        buffers = self.buffers
                else:
                    buffers = self.buffers

                workers_frames[idx] = workers_frames[idx] + buffers[idx].numel()

                if workers_frames[idx] >= self.total_frames:
                    dones[idx] = True

            if self.replay_buffer is not None:
                yield
                self._frames += self.frames_per_batch_worker * self.num_workers
                continue

            # we have to correct the traj_ids to make sure that they don't overlap
            # We can count the number of frames collected for free in this loop
            n_collected = 0
            for idx in range(self.num_workers):
                buffer = buffers[idx]
                traj_ids = buffer.get(("collector", "traj_ids"))
                if preempt:
                    if cat_results == "stack":
                        mask_frames = buffer.get(("collector", "traj_ids")) != -1
                        n_collected += mask_frames.sum().cpu()
                    else:
                        n_collected += traj_ids.numel()
                else:
                    n_collected += traj_ids.numel()

            if same_device is None:
                prev_device = None
                same_device = True
                for item in self.buffers.values():
                    if prev_device is None:
                        prev_device = item.device
                    else:
                        same_device = same_device and (item.device == prev_device)

            if cat_results == "stack":
                stack = (
                    torch.stack if self._use_buffers else TensorDict.maybe_dense_stack
                )
                if same_device:
                    self.out_buffer = stack(list(buffers.values()), 0)
                else:
                    self.out_buffer = stack(
                        [item.cpu() for item in buffers.values()], 0
                    )
            else:
                if self._use_buffers is None:
                    torchrl_logger.warning(
                        "use_buffer not specified and not yet inferred from data, assuming `True`."
                    )
                elif not self._use_buffers:
                    raise RuntimeError(
                        "Cannot concatenate results with use_buffers=False"
                    )
                try:
                    if same_device:
                        self.out_buffer = torch.cat(list(buffers.values()), cat_results)
                    else:
                        self.out_buffer = torch.cat(
                            [item.cpu() for item in buffers.values()], cat_results
                        )
                except RuntimeError as err:
                    if (
                        preempt
                        and cat_results != -1
                        and "Sizes of tensors must match" in str(err)
                    ):
                        raise RuntimeError(
                            "The value provided to cat_results isn't compatible with the collectors outputs. "
                            "Consider using `cat_results=-1`."
                        )
                    raise

            # TODO: why do we need to do cat inplace and clone?
            if self.split_trajs:
                out = split_trajectories(self.out_buffer, prefix="collector")
            else:
                out = self.out_buffer
            if cat_results in (-1, "stack"):
                out.refine_names(*[None] * (out.ndim - 1) + ["time"])

            self._frames += n_collected

            if self.postprocs:
                self.postprocs = self.postprocs.to(out.device)
                out = self.postprocs(out)
            if self._exclude_private_keys:
                excluded_keys = [key for key in out.keys() if key.startswith("_")]
                if excluded_keys:
                    out = out.exclude(*excluded_keys)
            yield out
            del out

        del self.buffers
        self.out_buffer = None
        # We shall not call shutdown just yet as user may want to retrieve state_dict
        # self._shutdown_main()


@accept_remote_rref_udf_invocation
class MultiaSyncDataCollector(_MultiDataCollector):
    """Runs a given number of DataCollectors on separate processes asynchronously.

    .. aafig::


            +----------------------------------------------------------------------+
            |           "MultiConcurrentCollector"                |                |
            |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|                |
            |  "Collector 1"  |  "Collector 2"  |  "Collector 3"  |     "Main"     |
            |~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~~|~~~~~~~~~~~~~~~~|
            | "env1" | "env2" | "env3" | "env4" | "env5" | "env6" |                |
            |~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~|~~~~~~~~~~~~~~~~|
            |"reset" |"reset" |"reset" |"reset" |"reset" |"reset" |                |
            |        |        |        |        |        |        |                |
            |       "actor"   |        |        |       "actor"   |                |
            |                 |        |        |                 |                |
            | "step" | "step" |       "actor"   |                 |                |
            |        |        |                 |                 |                |
            |        |        |                 | "step" | "step" |                |
            |        |        |                 |        |        |                |
            |       "actor    | "step" | "step" |       "actor"   |                |
            |                 |        |        |                 |                |
            | "yield batch 1" |       "actor"   |                 |"collect, train"|
            |                 |                 |                 |                |
            | "step" | "step" |                 | "yield batch 2" |"collect, train"|
            |        |        |                 |                 |                |
            |        |        | "yield batch 3" |                 |"collect, train"|
            |        |        |                 |                 |                |
            +----------------------------------------------------------------------+

    Environment types can be identical or different.

    The collection keeps on occuring on all processes even between the time
    the batch of rollouts is collected and the next call to the iterator.
    This class can be safely used with offline RL sota-implementations.

    .. note:: Python requires multiprocessed code to be instantiated within a main guard:

            >>> from torchrl.collectors import MultiaSyncDataCollector
            >>> if __name__ == "__main__":
            ...     # Create your collector here

        See https://docs.python.org/3/library/multiprocessing.html for more info.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from tensordict.nn import TensorDictModule
        >>> from torch import nn
        >>> from torchrl.collectors import MultiaSyncDataCollector
        >>> if __name__ == "__main__":
        ...     env_maker = lambda: GymEnv("Pendulum-v1", device="cpu")
        ...     policy = TensorDictModule(nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"])
        ...     collector = MultiaSyncDataCollector(
        ...         create_env_fn=[env_maker, env_maker],
        ...         policy=policy,
        ...         total_frames=2000,
        ...         max_frames_per_traj=50,
        ...         frames_per_batch=200,
        ...         init_random_frames=-1,
        ...         reset_at_each_iter=False,
        ...         device="cpu",
        ...         storing_device="cpu",
        ...         cat_results="stack",
        ...     )
        ...     for i, data in enumerate(collector):
        ...         if i == 2:
        ...             print(data)
        ...             break
        ... collector.shutdown()
        ... del collector
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

    """

    __doc__ += _MultiDataCollector.__doc__

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.out_tensordicts = defaultdict(lambda: None)
        self.running = False

        if self.postprocs is not None:
            postproc = self.postprocs
            self.postprocs = {}
            for _device in self.storing_device:
                if _device not in self.postprocs:
                    self.postprocs[_device] = deepcopy(postproc).to(_device)

    # for RPC
    def next(self):
        return super().next()

    # for RPC
    def shutdown(self):
        if hasattr(self, "out_tensordicts"):
            del self.out_tensordicts
        return super().shutdown()

    # for RPC
    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        return super().set_seed(seed, static_seed)

    # for RPC
    def state_dict(self) -> OrderedDict:
        return super().state_dict()

    # for RPC
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        return super().load_state_dict(state_dict)

    # for RPC
    def update_policy_weights_(
        self, policy_weights: Optional[TensorDictBase] = None
    ) -> None:
        super().update_policy_weights_(policy_weights)

    @property
    def frames_per_batch_worker(self):
        return self.requested_frames_per_batch

    def _get_from_queue(self, timeout=None) -> Tuple[int, int, TensorDictBase]:
        new_data, j = self.queue_out.get(timeout=timeout)
        use_buffers = self._use_buffers
        if self.replay_buffer is not None:
            idx = new_data
        elif j == 0 or not use_buffers:
            try:
                data, idx = new_data
                self.out_tensordicts[idx] = data
                if use_buffers is None and j > 0:
                    use_buffers = self._use_buffers = False
            except TypeError:
                if use_buffers is None:
                    use_buffers = self._use_buffers = True
                    idx = new_data
                else:
                    raise
        else:
            idx = new_data
        out = self.out_tensordicts[idx]
        if not self.replay_buffer and (j == 0 or use_buffers):
            # we clone the data to make sure that we'll be working with a fixed copy
            out = out.clone()
        return idx, j, out

    @property
    def _queue_len(self) -> int:
        return 1

    def iterator(self) -> Iterator[TensorDictBase]:
        if self.update_at_each_batch:
            self.update_policy_weights_()

        for i in range(self.num_workers):
            if self.init_random_frames is not None and self.init_random_frames > 0:
                self.pipes[i].send((None, "continue_random"))
            else:
                self.pipes[i].send((None, "continue"))
        self.running = True

        workers_frames = [0 for _ in range(self.num_workers)]
        while self._frames < self.total_frames:
            self._iter += 1
            while True:
                try:
                    idx, j, out = self._get_from_queue(timeout=10.0)
                    break
                except TimeoutError:
                    _check_for_faulty_process(self.procs)
            if self.replay_buffer is None:
                worker_frames = out.numel()
                if self.split_trajs:
                    out = split_trajectories(out, prefix="collector")
            else:
                worker_frames = self.frames_per_batch_worker
            self._frames += worker_frames
            workers_frames[idx] = workers_frames[idx] + worker_frames
            if self.postprocs:
                out = self.postprocs[out.device](out)

            # the function blocks here until the next item is asked, hence we send the message to the
            # worker to keep on working in the meantime before the yield statement
            if (
                self.init_random_frames is not None
                and self._frames < self.init_random_frames
            ):
                msg = "continue_random"
            else:
                msg = "continue"
            self.pipes[idx].send((idx, msg))
            if out is not None and self._exclude_private_keys:
                excluded_keys = [key for key in out.keys() if key.startswith("_")]
                out = out.exclude(*excluded_keys)
            yield out

        # We don't want to shutdown yet, the user may want to call state_dict before
        # self._shutdown_main()
        self.running = False

    def _shutdown_main(self) -> None:
        if hasattr(self, "out_tensordicts"):
            del self.out_tensordicts
        return super()._shutdown_main()

    def reset(self, reset_idx: Optional[Sequence[bool]] = None) -> None:
        super().reset(reset_idx)
        if self.queue_out.full():
            time.sleep(_TIMEOUT)  # wait until queue is empty
        if self.queue_out.full():
            raise Exception("self.queue_out is full")
        if self.running:
            for idx in range(self.num_workers):
                if (
                    self.init_random_frames is not None
                    and self._frames < self.init_random_frames
                ):
                    self.pipes[idx].send((idx, "continue_random"))
                else:
                    self.pipes[idx].send((idx, "continue"))


@accept_remote_rref_udf_invocation
class aSyncDataCollector(MultiaSyncDataCollector):
    """Runs a single DataCollector on a separate process.

    This is mostly useful for offline RL paradigms where the policy being
    trained can differ from the policy used to collect data. In online
    settings, a regular DataCollector should be preferred. This class is
    merely a wrapper around a MultiaSyncDataCollector where a single process
    is being created.

    Args:
        create_env_fn (Callabled): Callable returning an instance of EnvBase
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
        frames_per_batch (int): A keyword-only argument representing the
            total number of elements in a batch.
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
        update_at_each_batch (boolm optional): if ``True``, :meth:`~.update_policy_weight_()`
            will be called before (sync) or after (async) each data collection.
            Defaults to ``False``.
        preemptive_threshold (float, optional): a value between 0.0 and 1.0 that specifies the ratio of workers
            that will be allowed to finished collecting their rollout before the rest are forced to end early.
        num_threads (int, optional): number of threads for this process.
            Defaults to the number of workers.
        num_sub_threads (int, optional): number of threads of the subprocesses.
            Should be equal to one plus the number of processes launched within
            each subprocess (or one if a single process is launched).
            Defaults to 1 for safety: if none is indicated, launching multiple
            workers may charge the cpu load too much and harm performance.
        set_truncated (bool, optional): if ``True``, the truncated signals (and corresponding
            ``"done"`` but not ``"terminated"``) will be set to ``True`` when the last frame of
            a rollout is reached. If no ``"truncated"`` key is found, an exception is raised.
            Truncated keys can be set through ``env.add_truncated_keys``.
            Defaults to ``False``.

    """

    def __init__(
        self,
        create_env_fn: Callable[[], EnvBase],
        policy: Optional[
            Union[
                TensorDictModule,
                Callable[[TensorDictBase], TensorDictBase],
            ]
        ],
        *,
        frames_per_batch: int,
        total_frames: Optional[int] = -1,
        device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        storing_device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        env_device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        policy_device: DEVICE_TYPING | Sequence[DEVICE_TYPING] | None = None,
        create_env_kwargs: Optional[Sequence[dict]] = None,
        max_frames_per_traj: int | None = None,
        init_random_frames: int | None = None,
        reset_at_each_iter: bool = False,
        postproc: Optional[Callable[[TensorDictBase], TensorDictBase]] = None,
        split_trajs: Optional[bool] = None,
        exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
        reset_when_done: bool = True,
        update_at_each_batch: bool = False,
        preemptive_threshold: float = None,
        num_threads: int = None,
        num_sub_threads: int = 1,
        set_truncated: bool = False,
        **kwargs,
    ):
        super().__init__(
            create_env_fn=[create_env_fn],
            policy=policy,
            total_frames=total_frames,
            create_env_kwargs=[create_env_kwargs],
            max_frames_per_traj=max_frames_per_traj,
            frames_per_batch=frames_per_batch,
            reset_at_each_iter=reset_at_each_iter,
            init_random_frames=init_random_frames,
            postproc=postproc,
            split_trajs=split_trajs,
            device=device,
            policy_device=policy_device,
            env_device=env_device,
            storing_device=storing_device,
            exploration_type=exploration_type,
            reset_when_done=reset_when_done,
            update_at_each_batch=update_at_each_batch,
            preemptive_threshold=preemptive_threshold,
            num_threads=num_threads,
            num_sub_threads=num_sub_threads,
            set_truncated=set_truncated,
        )

    # for RPC
    def next(self):
        return super().next()

    # for RPC
    def shutdown(self):
        return super().shutdown()

    # for RPC
    def set_seed(self, seed: int, static_seed: bool = False) -> int:
        return super().set_seed(seed, static_seed)

    # for RPC
    def state_dict(self) -> OrderedDict:
        return super().state_dict()

    # for RPC
    def load_state_dict(self, state_dict: OrderedDict) -> None:
        return super().load_state_dict(state_dict)


def _main_async_collector(
    pipe_parent: connection.Connection,
    pipe_child: connection.Connection,
    queue_out: queues.Queue,
    create_env_fn: Union[EnvBase, "EnvCreator", Callable[[], EnvBase]],  # noqa: F821
    create_env_kwargs: Dict[str, Any],
    policy: Callable[[TensorDictBase], TensorDictBase],
    max_frames_per_traj: int,
    frames_per_batch: int,
    reset_at_each_iter: bool,
    storing_device: Optional[Union[torch.device, str, int]],
    env_device: Optional[Union[torch.device, str, int]],
    policy_device: Optional[Union[torch.device, str, int]],
    idx: int = 0,
    exploration_type: ExplorationType = DEFAULT_EXPLORATION_TYPE,
    reset_when_done: bool = True,
    verbose: bool = VERBOSE,
    interruptor=None,
    set_truncated: bool = False,
    use_buffers: bool | None = None,
    replay_buffer: ReplayBuffer | None = None,
    replay_buffer_chunk: bool = True,
    traj_pool: _TrajectoryPool = None,
    trust_policy: bool = False,
    compile_policy: bool = False,
    cudagraph_policy: bool = False,
) -> None:
    pipe_parent.close()
    # init variables that will be cleared when closing
    collected_tensordict = data = next_data = data_in = inner_collector = dc_iter = None

    inner_collector = SyncDataCollector(
        create_env_fn,
        create_env_kwargs=create_env_kwargs,
        policy=policy,
        total_frames=-1,
        max_frames_per_traj=max_frames_per_traj,
        frames_per_batch=frames_per_batch,
        reset_at_each_iter=reset_at_each_iter,
        postproc=None,
        split_trajs=False,
        storing_device=storing_device,
        policy_device=policy_device,
        env_device=env_device,
        exploration_type=exploration_type,
        reset_when_done=reset_when_done,
        return_same_td=replay_buffer is None,
        interruptor=interruptor,
        set_truncated=set_truncated,
        use_buffers=use_buffers,
        replay_buffer=replay_buffer if replay_buffer_chunk else None,
        traj_pool=traj_pool,
        trust_policy=trust_policy,
        compile_policy=compile_policy,
        cudagraph_policy=cudagraph_policy,
    )
    use_buffers = inner_collector._use_buffers
    if verbose:
        torchrl_logger.info("Sync data collector created")
    dc_iter = iter(inner_collector)
    j = 0
    pipe_child.send("instantiated")

    has_timed_out = False
    counter = 0
    while True:
        _timeout = _TIMEOUT if not has_timed_out else 1e-3
        if pipe_child.poll(_timeout):
            counter = 0
            data_in, msg = pipe_child.recv()
            if verbose:
                torchrl_logger.info(f"worker {idx} received {msg}")
        else:
            if verbose:
                torchrl_logger.info(f"poll failed, j={j}, worker={idx}")
            # default is "continue" (after first iteration)
            # this is expected to happen if queue_out reached the timeout, but no new msg was waiting in the pipe
            # in that case, the main process probably expects the worker to continue collect data
            if has_timed_out:
                counter = 0
                # has_timed_out is True if the process failed to send data, which will
                # typically occur if main has taken another batch (i.e. the queue is Full).
                # In this case, msg is the previous msg sent by main, which will typically be "continue"
                # If it's not the case, it is not expected that has_timed_out is True.
                if msg not in ("continue", "continue_random"):
                    raise RuntimeError(f"Unexpected message after time out: msg={msg}")
            else:
                # if has_timed_out is False, then the time out does not come from the fact that the queue is Full.
                # this means that our process has been waiting for a command from main in vain, while main was not
                # receiving data.
                # This will occur if main is busy doing something else (e.g. computing loss etc).

                counter += _timeout
                if verbose:
                    torchrl_logger.info(f"worker {idx} has counter {counter}")
                if counter >= (_MAX_IDLE_COUNT * _TIMEOUT):
                    raise RuntimeError(
                        f"This process waited for {counter} seconds "
                        f"without receiving a command from main. Consider increasing the maximum idle count "
                        f"if this is expected via the environment variable MAX_IDLE_COUNT "
                        f"(current value is {_MAX_IDLE_COUNT})."
                        f"\nIf this occurs at the end of a function or program, it means that your collector has not been "
                        f"collected, consider calling `collector.shutdown()` before ending the program."
                    )
                continue
        if msg in ("continue", "continue_random"):
            if msg == "continue_random":
                inner_collector.init_random_frames = float("inf")
            else:
                inner_collector.init_random_frames = -1

            next_data = next(dc_iter)
            if pipe_child.poll(_MIN_TIMEOUT):
                # in this case, main send a message to the worker while it was busy collecting trajectories.
                # In that case, we skip the collected trajectory and get the message from main. This is faster than
                # sending the trajectory in the queue until timeout when it's never going to be received.
                continue

            if replay_buffer is not None:
                if not replay_buffer_chunk:
                    next_data.names = None
                    replay_buffer.extend(next_data)

                try:
                    queue_out.put((idx, j), timeout=_TIMEOUT)
                    if verbose:
                        torchrl_logger.info(f"worker {idx} successfully sent data")
                    j += 1
                    has_timed_out = False
                    continue
                except queue.Full:
                    if verbose:
                        torchrl_logger.info(f"worker {idx} has timed out")
                    has_timed_out = True
                    continue

            if j == 0 or not use_buffers:
                collected_tensordict = next_data
                if (
                    storing_device is not None
                    and collected_tensordict.device != storing_device
                ):
                    raise RuntimeError(
                        f"expected device to be {storing_device} but got {collected_tensordict.device}"
                    )
                if use_buffers:
                    # If policy and env are on cpu, we put in shared mem,
                    # if policy is on cuda and env on cuda, we are fine with this
                    # If policy is on cuda and env on cpu (or opposite) we put tensors that
                    # are on cpu in shared mem.
                    MPS_ERROR = (
                        "tensors on mps device cannot be put in shared memory. Make sure "
                        "the shared device (aka storing_device) is set to CPU."
                    )
                    if collected_tensordict.device is not None:
                        # placehoder in case we need different behaviors
                        if collected_tensordict.device.type in ("cpu",):
                            collected_tensordict.share_memory_()
                        elif collected_tensordict.device.type in ("mps",):
                            raise RuntimeError(MPS_ERROR)
                        elif collected_tensordict.device.type == "cuda":
                            collected_tensordict.share_memory_()
                        else:
                            raise NotImplementedError(
                                f"Device {collected_tensordict.device} is not supported in multi-collectors yet."
                            )
                    else:
                        # make sure each cpu tensor is shared - assuming non-cpu devices are shared
                        def cast_tensor(x, MPS_ERROR=MPS_ERROR):
                            if x.device.type in ("cpu",):
                                x.share_memory_()
                            if x.device.type in ("mps",):
                                RuntimeError(MPS_ERROR)

                        collected_tensordict.apply(cast_tensor, filter_empty=True)
                data = (collected_tensordict, idx)
            else:
                if next_data is not collected_tensordict:
                    raise RuntimeError(
                        "SyncDataCollector should return the same tensordict modified in-place."
                    )
                data = idx  # flag the worker that has sent its data
            try:
                queue_out.put((data, j), timeout=_TIMEOUT)
                if verbose:
                    torchrl_logger.info(f"worker {idx} successfully sent data")
                j += 1
                has_timed_out = False
                continue
            except queue.Full:
                if verbose:
                    torchrl_logger.info(f"worker {idx} has timed out")
                has_timed_out = True
                continue

        elif msg == "update":
            inner_collector.update_policy_weights_()
            pipe_child.send((j, "updated"))
            has_timed_out = False
            continue

        elif msg == "seed":
            data_in, static_seed = data_in
            new_seed = inner_collector.set_seed(data_in, static_seed=static_seed)
            torch.manual_seed(data_in)
            np.random.seed(data_in)
            pipe_child.send((new_seed, "seeded"))
            has_timed_out = False
            continue

        elif msg == "reset":
            inner_collector.reset()
            pipe_child.send((j, "reset"))
            continue

        elif msg == "state_dict":
            state_dict = inner_collector.state_dict()
            # send state_dict to cpu first
            state_dict = recursive_map_to_cpu(state_dict)
            pipe_child.send((state_dict, "state_dict"))
            has_timed_out = False
            continue

        elif msg == "load_state_dict":
            state_dict = data_in
            inner_collector.load_state_dict(state_dict)
            del state_dict
            pipe_child.send((j, "loaded"))
            has_timed_out = False
            continue

        elif msg == "close":
            del collected_tensordict, data, next_data, data_in
            inner_collector.shutdown()
            del inner_collector, dc_iter
            pipe_child.send("closed")
            if verbose:
                torchrl_logger.info(f"collector {idx} closed")
            break

        else:
            raise Exception(f"Unrecognized message {msg}")


def _make_meta_params(param):
    is_param = isinstance(param, Parameter)

    pd = param.detach().to("meta")

    if is_param:
        pd = Parameter(pd, requires_grad=False)
    return pd


class _TrajectoryPool:
    def __init__(self, ctx=None, lock: bool = False):
        self.ctx = ctx
        self._traj_id = torch.zeros((), device="cpu", dtype=torch.int).share_memory_()
        if ctx is None:
            self.lock = contextlib.nullcontext() if not lock else mp.RLock()
        else:
            self.lock = contextlib.nullcontext() if not lock else ctx.RLock()

    def get_traj_and_increment(self, n=1, device=None):
        with self.lock:
            v = self._traj_id.item()
            out = torch.arange(v, v + n).to(device)
            self._traj_id.copy_(1 + out[-1].item())
        return out
