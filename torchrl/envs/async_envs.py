# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
import multiprocessing
import os
import queue
import threading
import traceback
import warnings
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import as_completed, FIRST_COMPLETED, ThreadPoolExecutor, wait
from multiprocessing import Queue
from numbers import Integral
from typing import Literal

import torch
from tensordict import (
    lazy_stack,
    LazyStackedTensorDict,
    maybe_dense_stack,
    TensorDict,
    TensorDictBase,
)
from tensordict.tensorclass import NonTensorData, NonTensorStack
from tensordict.utils import _zip_strict, expand_as_right, NestedKey

from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.data.tensor_specs import NonTensor
from torchrl.envs._async_exchange import _receive_batch, _SharedSlotExchange
from torchrl.envs.common import _EnvPostInit, EnvBase


class _ValidatedCPUAffinity(tuple):
    pass


def _validate_cpu_affinity(
    cpu_affinity: Sequence[int], *, option_name: str
) -> tuple[int, ...]:
    if isinstance(cpu_affinity, _ValidatedCPUAffinity):
        return cpu_affinity
    if not hasattr(os, "sched_setaffinity") or not hasattr(os, "sched_getaffinity"):
        raise NotImplementedError(
            f"{option_name} requires os.sched_getaffinity and "
            "os.sched_setaffinity, which are only available on Linux."
        )
    try:
        cpu_affinity = tuple(cpu_affinity)
    except TypeError as err:
        raise TypeError(f"{option_name} must be a sequence of CPU indices.") from err
    if not cpu_affinity:
        raise ValueError(f"{option_name} must contain at least one CPU index.")
    if any(
        isinstance(cpu, bool) or not isinstance(cpu, Integral) for cpu in cpu_affinity
    ):
        raise TypeError(f"{option_name} must contain only integer CPU indices.")
    if any(cpu < 0 for cpu in cpu_affinity):
        raise ValueError(f"{option_name} CPU indices must be non-negative.")
    cpu_affinity = _ValidatedCPUAffinity(int(cpu) for cpu in cpu_affinity)
    available_cpus = os.sched_getaffinity(0)
    unavailable_cpus = sorted(set(cpu_affinity).difference(available_cpus))
    if unavailable_cpus:
        raise ValueError(
            f"{option_name} contains CPUs unavailable to this process: "
            f"{unavailable_cpus}."
        )
    return cpu_affinity


class _AsyncEnvMeta(_EnvPostInit):
    """A metaclass for asynchronous environment pools that determines the backend implementation to use based on the provided arguments.

    This class is responsible for instantiating the appropriate subclass of `AsyncEnvPool` based on the specified
    backend, such as threading or multiprocessing.
    """

    def __call__(cls, *args, **kwargs):
        backend = kwargs.get("backend", "threading")
        if cls is AsyncEnvPool:
            if backend == "threading":
                instance: ThreadingAsyncEnvPool = ThreadingAsyncEnvPool(*args, **kwargs)
            elif backend == "multiprocessing":
                instance: ProcessorAsyncEnvPool = ProcessorAsyncEnvPool(*args, **kwargs)
            elif backend == "asyncio":
                raise NotImplementedError
                # instance: AsyncioAsyncEnvPool = AsyncioAsyncEnvPool(*args, **kwargs)
            else:
                raise NotImplementedError
            return instance
        else:
            return super().__call__(*args, **kwargs)


class AsyncEnvPool(EnvBase, metaclass=_AsyncEnvMeta):
    """A base class for asynchronous environment pools, providing a common interface for managing multiple environments concurrently.

    This class supports different backends for parallel execution, such as threading
    and multiprocessing, and provides methods for asynchronous stepping and resetting
    of environments.

    .. note:: This class and its subclasses should work when nested in with :class:`~torchrl.envs.TransformedEnv` and
        batched environments, but users won't currently be able to use the async features of the base environment when
        it's nested in these classes. One should prefer nested transformed envs within an `AsyncEnvPool` instead.
        If this is not possible, please raise an issue.

    Args:
        env_makers (Callable[[], EnvBase] | EnvBase | list[EnvBase] | list[Callable[[], EnvBase]]):
            A callable or list of callables that create environment instances, or
            environment instances themselves.
        backend (Literal["threading", "multiprocessing", "asyncio"], optional):
            The backend to use for parallel execution. Defaults to `"threading"`.
        stack (Literal["dense", "maybe_dense", "lazy"], optional):
            The method to use for stacking environment outputs. Defaults to `"dense"`.
        exchange (Literal["queue", "shm", "auto"], optional): Data exchange
            used by the multiprocessing backend. ``"queue"`` supports dynamic
            data and copies received tensors out of multiprocessing shared
            memory so retaining results does not retain one mapping per tensor.
            ``"shm"`` stores fixed-shape tensor data in shared slots and sends
            only readiness descriptors through queues; it requires identical,
            fixed-shape, CPU, tensor-only schemas across workers. ``"auto"``
            selects ``"shm"`` when the env schema supports it and falls back to
            ``"queue"`` otherwise (the resolution is reported by
            :attr:`resolved_exchange` and logged on fallback). Defaults to
            ``"queue"``.

            .. warning::
                The default will change from ``"queue"`` to ``"auto"`` in
                v0.15 for the multiprocessing backend. A ``FutureWarning`` is
                emitted when the default is relied upon.
        worker_affinity (Sequence[Sequence[int]] or Callable[[int], Sequence[int]], optional):
            Optional Linux CPU placement for multiprocessing workers. This is
            useful when environment workers share a constrained CPU set with
            CPU-heavy simulators or other services: without affinity, the
            scheduler may place them on the same CPUs and increase environment
            step-time jitter. Most users should leave this as ``None``.

            TorchRL can discover which CPUs the current process may use, but
            it cannot infer which CPUs the application has reserved for other
            work or how many threads an environment and its subprocesses need.
            It therefore does not choose a partition automatically. Provide
            one mask per worker process, or a callable mapping a worker
            index to its mask. The mask is applied before environment threads
            and factories start and is inherited by subprocesses they start.
            Defaults to ``None``. See :ref:`async_env_pool_cpu_affinity` for an
            example and deployment guidance.
        envs_per_worker (int, optional): Number of environments hosted by each
            multiprocessing worker process. Environments within a worker are
            executed concurrently in threads. This option is only supported by
            the multiprocessing backend. Defaults to ``1``.
        create_env_kwargs (dict, optional):
            Keyword arguments to pass to the environment maker. Defaults to `{}`.

    Attributes:
        min_get (int): Minimum number of environments to process in a batch.
        env_makers (list): List of environment makers or environments.
        num_envs (int): Number of environments in the pool.
        envs_per_worker (int): Number of environments hosted by each
            multiprocessing worker.
        backend (str): Backend used for parallel execution.
        stack (str): Method used for stacking environment outputs.

    Examples:
        >>> from functools import partial
        >>> from torchrl.envs import AsyncEnvPool, GymEnv
        >>> import torch
        >>> # Choose backend
        >>> backend = "threading"
        >>> env = AsyncEnvPool([partial(GymEnv, "Pendulum-v1"), partial(GymEnv, "CartPole-v1")], stack="lazy", backend=backend)
        >>> assert env.batch_size == (2,)
        >>> # Execute a sync reset
        >>> reset = env.reset()
        >>> print(reset)
        LazyStackedTensorDict(
            fields={
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                env_index: NonTensorStack(
                    [0, 1],
                    batch_size=torch.Size([2]),
                    device=None),
                observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            exclusive_fields={
            },
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False,
            stack_dim=0)
        >>> # Execute a sync step
        >>> s = env.rand_step(reset)
        >>> print(s)
        LazyStackedTensorDict(
            fields={
                action: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                env_index: NonTensorStack(
                    [0, 1],
                    batch_size=torch.Size([2]),
                    device=None),
                next: LazyStackedTensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    exclusive_fields={
                    },
                    batch_size=torch.Size([2]),
                    device=None,
                    is_shared=False,
                    stack_dim=0),
                observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            exclusive_fields={
            },
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False,
            stack_dim=0)
        >>> s = env.step_mdp(s)
        >>> # Execute an asynchronous step in env 0
        >>> s0 = s[0]
        >>> s0["action"] = torch.randn(1).clamp(-1, 1)
        >>> # We must tell the env which data this is from
        >>> s0["env_index"] = 0
        >>> env.async_step_send(s0)
        >>> # Receive data
        >>> s0_result = env.async_step_recv()
        >>> print('result', s0_result)
        result LazyStackedTensorDict(
            fields={
                action: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                env_index: NonTensorStack(
                    [0],
                    batch_size=torch.Size([1]),
                    device=None),
                next: LazyStackedTensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([1, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    exclusive_fields={
                    },
                    batch_size=torch.Size([1]),
                    device=None,
                    is_shared=False,
                    stack_dim=0),
                observation: Tensor(shape=torch.Size([1, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            exclusive_fields={
            },
            batch_size=torch.Size([1]),
            device=None,
            is_shared=False,
            stack_dim=0)
        >>> # Close env
        >>> env.close()


    """

    _env_idx_key = "env_index"

    def __init__(
        self,
        env_makers: Callable[[], EnvBase]
        | EnvBase
        | list[EnvBase]
        | list[Callable[[], EnvBase]],
        *,
        backend: Literal["threading", "multiprocessing", "asyncio"] = "threading",
        stack: Literal["dense", "maybe_dense", "lazy"] = "dense",
        exchange: Literal["queue", "shm", "auto"] | None = None,
        worker_affinity: Sequence[Sequence[int]]
        | Callable[[int], Sequence[int]]
        | None = None,
        envs_per_worker: int = 1,
        create_env_kwargs: dict | list[dict] | None = None,
    ) -> None:
        if not isinstance(env_makers, Sequence):
            env_makers = [env_makers]
        self.min_get = 1
        self.env_makers = env_makers
        self.num_envs = len(env_makers)
        self.backend = backend
        if (
            isinstance(envs_per_worker, bool)
            or not isinstance(envs_per_worker, int)
            or envs_per_worker < 1
        ):
            raise ValueError(
                "envs_per_worker must be a positive integer, got "
                f"{envs_per_worker!r}."
            )
        if backend != "multiprocessing" and envs_per_worker != 1:
            raise ValueError(
                "envs_per_worker is only supported with backend='multiprocessing'."
            )
        self.envs_per_worker = envs_per_worker
        self.worker_affinity = worker_affinity
        self._worker_affinity = None
        if worker_affinity is not None:
            if backend != "multiprocessing":
                raise ValueError(
                    "worker_affinity is only supported with "
                    "backend='multiprocessing'."
                )
            num_workers = (self.num_envs + envs_per_worker - 1) // envs_per_worker
            if callable(worker_affinity):
                affinity_masks = [
                    worker_affinity(worker_index) for worker_index in range(num_workers)
                ]
            else:
                affinity_masks = list(worker_affinity)
                if len(affinity_masks) != num_workers:
                    raise ValueError(
                        "worker_affinity must provide one CPU mask per "
                        f"worker process, got {len(affinity_masks)} masks for "
                        f"{num_workers} worker processes."
                    )
            self._worker_affinity = [
                _validate_cpu_affinity(
                    affinity,
                    option_name=f"worker_affinity[{worker_index}]",
                )
                for worker_index, affinity in enumerate(affinity_masks)
            ]
        if exchange is None:
            if backend == "multiprocessing":
                warnings.warn(
                    "The default exchange of AsyncEnvPool with the "
                    "multiprocessing backend will change from 'queue' to "
                    "'auto' in v0.15. Pass exchange='queue' to keep the "
                    "current behavior, or exchange='auto' to adopt the "
                    "future default now.",
                    FutureWarning,
                )
            exchange = "queue"
        if exchange not in ("queue", "shm", "auto"):
            raise ValueError(
                f"exchange must be 'queue', 'shm' or 'auto', got {exchange!r}."
            )
        if backend != "multiprocessing" and exchange == "shm":
            raise ValueError(
                "exchange='shm' is only supported with backend='multiprocessing'."
            )
        self.exchange = exchange
        if create_env_kwargs is None:
            create_env_kwargs = {}
        if isinstance(create_env_kwargs, Mapping):
            create_env_kwargs = [create_env_kwargs] * self.num_envs
        if len(create_env_kwargs) != self.num_envs:
            raise ValueError(
                f"create_env_kwargs must be a dict or a list of dicts with length {self.num_envs}"
            )
        self.create_env_kwargs = create_env_kwargs

        self.stack = stack
        if stack == "dense":
            self._stack_func = torch.stack
        elif stack == "maybe_dense":
            self._stack_func = maybe_dense_stack
        elif stack == "lazy":
            self._stack_func = lazy_stack
        else:
            raise NotImplementedError

        output_spec, input_spec = self._setup()
        input_spec["full_state_spec"].set(
            self._env_idx_key, NonTensor(example_data=0, shape=input_spec.shape)
        )
        self.__dict__["_output_spec"] = output_spec
        self.__dict__["_input_spec"] = input_spec
        # Use spec shape as batch_size since it correctly includes both pool dimension
        # and child env batch dimensions (e.g., (4, 1) for 4 envs with batch_size=(1,))
        super().__init__(batch_size=input_spec.shape)
        self._busy = set()
        self._lock = threading.Lock()

    @property
    def env_batch_sizes(self) -> list[torch.Size]:
        """Returns the batch-sizes of every env."""
        raise NotImplementedError

    @abc.abstractmethod
    def _get_child_specs(self) -> list:
        """Returns the list of child env specs for stacking.

        For ThreadingAsyncEnvPool, returns [env.full_*_spec for env in self.envs].
        For ProcessorAsyncEnvPool, returns cached specs from setup.
        """
        raise NotImplementedError

    # Override spec properties to properly stack child env specs.
    # This bypasses the problematic StackedComposite.get() behavior that loses
    # nested keys like full_action_spec when cloning stacked specs.

    @property
    def full_action_spec(self):
        child_specs = self._get_child_specs()
        return torch.stack(
            [s["input_spec"]["full_action_spec"] for s in child_specs], dim=0
        )

    @property
    def full_observation_spec(self):
        child_specs = self._get_child_specs()
        return torch.stack(
            [s["output_spec"]["full_observation_spec"] for s in child_specs], dim=0
        )

    @property
    def full_reward_spec(self):
        child_specs = self._get_child_specs()
        return torch.stack(
            [s["output_spec"]["full_reward_spec"] for s in child_specs], dim=0
        )

    @property
    def full_done_spec(self):
        child_specs = self._get_child_specs()
        return torch.stack(
            [s["output_spec"]["full_done_spec"] for s in child_specs], dim=0
        )

    @property
    def full_state_spec(self):
        child_specs = self._get_child_specs()
        specs = torch.stack(
            [s["input_spec"]["full_state_spec"] for s in child_specs], dim=0
        )
        # Add env_index key for async tracking
        specs.set(self._env_idx_key, NonTensor(example_data=0, shape=specs.shape))
        return specs

    # TODO: _make_single_env_spec (used by *_unbatched properties) takes spec[0],
    # which assumes all child envs have identical specs. Should add validation
    # that child specs match, and error if they differ.

    def _reset(
        self,
        tensordict: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        if self._current_step > 0:
            raise RuntimeError("Some envs are still processing a step.")
        if tensordict is None:
            if self._stack_func in ("lazy_stack", "maybe_dense"):
                tensordict = LazyStackedTensorDict(
                    *[
                        TensorDict(batch_size=self.env_batch_sizes[i])
                        for i in range(self.num_envs)
                    ]
                )
            else:
                tensordict = TensorDict(
                    batch_size=(self.num_envs,) + self.env_batch_sizes[0]
                )
        env_idx_nt = NonTensorStack(*range(tensordict.shape[0]))
        while env_idx_nt.batch_dims < tensordict.batch_dims:
            env_idx_nt = expand_as_right(env_idx_nt, tensordict)
        tensordict[self._env_idx_key] = env_idx_nt
        self._async_private_reset_send(tensordict)
        tensordict = self._async_private_reset_recv(min_get=self.num_envs)
        return tensordict

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._current_step > 0:
            raise RuntimeError("Some envs are still processing a step.")
        tensordict.set(self._env_idx_key, torch.arange(tensordict.shape[0]))
        self._async_private_step_send(tensordict)
        tensordict = self._async_private_step_recv(min_get=self.num_envs)
        # Using pop instead of del to account for tensorclasses
        tensordict.pop(self._env_idx_key)
        return tensordict

    def step_and_maybe_reset(
        self, tensordict: TensorDictBase
    ) -> tuple[TensorDictBase, TensorDictBase]:
        if self._current_step_reset > 0:
            raise RuntimeError("Some envs are still processing a step.")
        tensordict.set(self._env_idx_key, torch.arange(tensordict.shape[0]))
        self.async_step_and_maybe_reset_send(tensordict)
        tensordict, tensordict_ = self.async_step_and_maybe_reset_recv(
            min_get=self.num_envs
        )
        return tensordict, tensordict_

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self._current_step > 0:
            raise RuntimeError("Some envs are still processing a step.")
        tensordict.set(self._env_idx_key, torch.arange(tensordict.shape[0]))
        self.async_step_send(tensordict)
        tensordict = self.async_step_recv(min_get=self.num_envs)
        return tensordict

    def reset(
        self,
        tensordict: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        if self._current_step > 0:
            raise RuntimeError("Some envs are still processing a step.")
        if tensordict is None:
            if self._stack_func in ("lazy_stack", "maybe_dense"):
                tensordict = LazyStackedTensorDict(
                    *[
                        TensorDict(batch_size=self.env_batch_sizes[i])
                        for i in range(self.num_envs)
                    ]
                )
            else:
                tensordict = TensorDict(
                    batch_size=(self.num_envs,) + self.env_batch_sizes[0]
                )
        indices = NonTensorStack(*range(tensordict.shape[0]))
        if indices.shape != tensordict.shape:
            indices = expand_as_right(indices, tensordict)
        tensordict[self._env_idx_key] = indices
        self.async_reset_send(tensordict)
        tensordict = self.async_reset_recv(min_get=self.num_envs)
        return tensordict

    def _sort_results(self, results, *other_results):
        # Extract env indices from results. When child envs have a batch dimension
        # (e.g., batch_size=(1,)), r[self._env_idx_key] may be a 1D sequence
        # instead of a scalar, so we need to handle both cases.
        idx = []
        for r in results:
            env_idx = r[self._env_idx_key]
            # Handle sequence types (NonTensorStack, etc.) by taking first element
            while hasattr(env_idx, "__len__") and not isinstance(env_idx, (str, bytes)):
                if len(env_idx) == 1:
                    env_idx = env_idx[0]
                else:
                    break
            idx.append(int(env_idx))
        argsort = torch.argsort(torch.tensor(idx)).tolist()
        results = [results[i] for i in argsort]
        if other_results:
            other_results = [
                [other_results[i] for i in argsort] for other_results in other_results
            ]
            return results, *other_results, idx
        return results, idx

    def _set_seed(self, seed: int | None) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    def _maybe_make_tensordict(self, tensordict, env_index, make_if_none):
        if env_index is None:
            env_idx = tensordict.view(-1)[self._env_idx_key]
            if isinstance(env_idx, torch.Tensor):
                env_idx = env_idx.tolist()
            if isinstance(env_idx, int):
                # If we squeezed a td with shape (1,) and got a NonTensorStack -> NonTensorData, then
                #  unsqueezed the NonTensorData, we'd still have a NonTensorData with shape (1,)
                #  This will give us an integer now, but we don't want to unsqueeze the full td because then
                #  we'd have a td with shape (1, 1)
                if tensordict.shape != (1, *self.env_batch_sizes[env_idx]):
                    tensordict = tensordict.unsqueeze(0)
                env_idx = [env_idx]
        elif isinstance(env_index, int):
            if make_if_none and tensordict is None:
                tensordict = TensorDict(
                    batch_size=self.env_batch_sizes[env_index], device=self.device
                )
            # Always add a leading dim so that unbind(0) works in send methods
            if self.stack in ("lazy_stack", "maybe_dense"):
                tensordict = tensordict.unsqueeze(0)
            else:
                tensordict = lazy_stack([tensordict])
            tensordict[self._env_idx_key] = NonTensorStack(env_index)
            env_idx = [env_index]
        else:
            if make_if_none and tensordict is None:
                if self.stack in ("lazy_stack", "maybe_dense"):
                    tensordict = lazy_stack(
                        [TensorDict(device=self.device) for _ in env_index]
                    )
                else:
                    tensordict = TensorDict(
                        batch_size=(len(env_index),), device=self.device
                    )
            tensordict[self._env_idx_key] = NonTensorStack(*env_index)
            env_idx = env_index
        return tensordict, env_idx

    @abc.abstractmethod
    def async_step_send(
        self, tensordict: TensorDictBase, env_index: int | list[int] | None = None
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def async_step_recv(
        self,
        min_get: int | None = None,
        env_index: int | None = None,
        *,
        max_get: int | None = None,
        timeout: float | None = None,
    ) -> TensorDictBase:
        """Collects step results from the pool.

        Args:
            min_get (int, optional): minimum number of results to collect
                before returning. Defaults to ``self.min_get``.
            env_index (int, optional): if provided, waits for the result of
                this specific env instead of forming a batch; ``min_get``,
                ``max_get`` and ``timeout`` are ignored.

        Keyword Args:
            max_get (int, optional): maximum number of results to return.
                ``None`` (default) drains everything available once
                ``min_get`` is satisfied.
            timeout (float, optional): deadline in seconds for the entire
                call, measured from call entry. If fewer than ``min_get``
                results are available when the deadline expires, a
                :class:`TimeoutError` is raised; results collected so far are
                put back and remain available to the next call. Once
                ``min_get`` is satisfied, the call returns at the deadline
                with whatever is ready (up to ``max_get``). ``timeout=0``
                polls without blocking. ``None`` (default) waits
                indefinitely.

        Returns:
            TensorDictBase: the stacked results, carrying ``"env_index"``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def async_step_and_maybe_reset_send(
        self, tensordict: TensorDictBase, env_index: int | list[int] | None = None
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def async_step_and_maybe_reset_recv(
        self,
        min_get: int | None = None,
        env_index: int | list[int] | None = None,
        *,
        max_get: int | None = None,
        timeout: float | None = None,
    ) -> tuple[TensorDictBase, TensorDictBase]:
        """Collects step-and-maybe-reset results from the pool.

        ``min_get``, ``env_index``, ``max_get`` and ``timeout`` behave as in
        :meth:`async_step_recv`; in particular ``timeout`` bounds the entire
        call and raises :class:`TimeoutError` (without losing results) if
        fewer than ``min_get`` results arrive in time.

        Returns:
            tuple of (TensorDictBase, TensorDictBase): the stacked step
            results and the stacked roots of the next steps (reset where
            needed), both carrying ``"env_index"``.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def async_reset_send(
        self,
        tensordict: TensorDictBase | None = None,
        env_index: int | list[int] | None = None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def async_reset_recv(
        self,
        min_get: int | None = None,
        env_index: int | None = None,
        *,
        max_get: int | None = None,
        timeout: float | None = None,
    ) -> TensorDictBase:
        """Collects reset results from the pool.

        ``min_get``, ``env_index``, ``max_get`` and ``timeout`` behave as in
        :meth:`async_step_recv`; in particular ``timeout`` bounds the entire
        call and raises :class:`TimeoutError` (without losing results) if
        fewer than ``min_get`` results arrive in time.

        Returns:
            TensorDictBase: the stacked reset results, carrying
            ``"env_index"``.
        """
        raise NotImplementedError

    @property
    def resolved_exchange(self) -> Literal["queue", "shm"]:
        """The data exchange actually in use, after resolving ``exchange="auto"``.

        ``"shm"`` when the shared-memory slot exchange is active, ``"queue"``
        otherwise (including the threading backend, which has no shared-memory
        exchange).
        """
        return "shm" if getattr(self, "_slot_exchange", None) is not None else "queue"

    @property
    def exchange_keys(self) -> tuple[NestedKey, ...]:
        """The tensor keys accepted by the active shared-memory exchange.

        Returns an empty tuple when the resolved exchange is ``"queue"``.
        """
        exchange = getattr(self, "_slot_exchange", None)
        if exchange is None:
            return ()
        return tuple(exchange._input_keys)

    def stats(self, *, reset: bool = False) -> dict[str, float | int]:
        """Return shared-memory exchange statistics.

        Args:
            reset: Whether to clear the counters after taking the snapshot.

        Returns:
            Batch counts, fill rates, and exchange latency statistics. Queue
            and threading exchanges return an empty dictionary.
        """
        exchange = getattr(self, "_slot_exchange", None)
        if exchange is None:
            return {}
        return exchange.stats(reset=reset)

    def __del__(self):
        self._maybe_shutdown()

    def _maybe_shutdown(self):
        try:
            self.shutdown()
        except Exception:
            pass

    @abc.abstractmethod
    def shutdown(self):
        raise NotImplementedError

    def close(self, *, raise_if_closed: bool = True):
        if raise_if_closed:
            self.shutdown()
        else:
            self._maybe_shutdown()


class ProcessorAsyncEnvPool(AsyncEnvPool):
    """An implementation of `AsyncEnvPool` using multiprocessing for parallel execution of environments.

    This class manages a pool of environments across worker processes and provides
    methods for asynchronous stepping and resetting using inter-process
    communication. A worker can host multiple environments in dedicated threads.

    Supports per-env ``recv`` via ``env_index`` for thread-safe concurrent access
    from multiple collector threads.

    .. note:: This class and its subclasses should work when nested in with :class:`~torchrl.envs.TransformedEnv` and
        batched environments, but users won't currently be able to use the async features of the base environment when
        it's nested in these classes. One should prefer nested transformed envs within an `AsyncEnvPool` instead.
        If this is not possible, please raise an issue.

    Methods:
        _setup(): Initializes multiprocessing queues and worker processes.
        async_step_send(tensordict): Sends a step command to the environments.
        async_step_recv(min_get): Receives the results of the step command.
        async_reset_send(tensordict): Sends a reset command to the environments.
        async_reset_recv(min_get): Receives the results of the reset command.
        shutdown(): Shuts down all environment processes.
    """

    def _setup(self) -> None:
        self._error_queue = Queue()
        self._worker_error = None
        self._worker_liveness_timer = timeit("async_env_worker_liveness").start()
        self.step_queue = Queue(maxsize=self.num_envs)
        self.reset_queue = Queue(maxsize=self.num_envs)
        self.step_reset_queue = Queue(maxsize=self.num_envs)
        # Per-env result queues for thread-safe per-env recv
        self._per_env_step_queues = [Queue(maxsize=1) for _ in range(self.num_envs)]
        self._per_env_reset_queues = [Queue(maxsize=1) for _ in range(self.num_envs)]
        self._per_env_step_reset_queues = [
            Queue(maxsize=1) for _ in range(self.num_envs)
        ]
        self._worker_env_indices = [
            tuple(range(start, min(start + self.envs_per_worker, self.num_envs)))
            for start in range(0, self.num_envs, self.envs_per_worker)
        ]
        self.num_workers = len(self._worker_env_indices)
        self._env_to_worker = [
            worker_index
            for worker_index, env_indices in enumerate(self._worker_env_indices)
            for _ in env_indices
        ]
        self.input_queue = [Queue(maxsize=1) for _ in range(self.num_workers)]
        self.output_queue = [
            Queue(maxsize=len(env_indices)) for env_indices in self._worker_env_indices
        ]
        self._current_reset = 0
        self._current_step = 0
        self._current_step_reset = 0
        self._slot_exchange = None

        self.threads = []
        try:
            for worker_index, env_indices in enumerate(self._worker_env_indices):
                if len(env_indices) == 1:
                    env_index = env_indices[0]
                    target = self._env_exec
                    worker_kwargs = {
                        "i": env_index,
                        "env_or_factory": self.env_makers[env_index],
                        "create_env_kwargs": self.create_env_kwargs[env_index],
                        "input_queue": self.input_queue[worker_index],
                        "output_queue": self.output_queue[worker_index],
                        "step_reset_queue": self.step_reset_queue,
                        "step_queue": self.step_queue,
                        "reset_queue": self.reset_queue,
                        "per_env_step_queue": self._per_env_step_queues[env_index],
                        "per_env_reset_queue": self._per_env_reset_queues[env_index],
                        "per_env_step_reset_queue": self._per_env_step_reset_queues[
                            env_index
                        ],
                        "grouped_input": True,
                    }
                else:
                    target = self._worker_exec
                    worker_kwargs = {
                        "env_indices": env_indices,
                        "env_makers": [self.env_makers[i] for i in env_indices],
                        "create_env_kwargs": [
                            self.create_env_kwargs[i] for i in env_indices
                        ],
                        "input_queue": self.input_queue[worker_index],
                        "output_queue": self.output_queue[worker_index],
                        "step_reset_queue": self.step_reset_queue,
                        "step_queue": self.step_queue,
                        "reset_queue": self.reset_queue,
                        "per_env_step_queues": [
                            self._per_env_step_queues[i] for i in env_indices
                        ],
                        "per_env_reset_queues": [
                            self._per_env_reset_queues[i] for i in env_indices
                        ],
                        "per_env_step_reset_queues": [
                            self._per_env_step_reset_queues[i] for i in env_indices
                        ],
                    }
                worker_kwargs["error_queue"] = self._error_queue
                worker_kwargs["cpu_affinity"] = (
                    None
                    if self._worker_affinity is None
                    else self._worker_affinity[worker_index]
                )
                thread = multiprocessing.Process(
                    target=target,
                    kwargs=worker_kwargs,
                )
                self.threads.append(thread)
                thread.start()
            if self._worker_affinity is not None:
                try:
                    for i in range(self.num_workers):
                        status, payload = _receive_batch(
                            self.output_queue[i],
                            1,
                            1,
                            None,
                            check_worker_errors=self._check_worker_errors,
                        )[0]
                        if status != "affinity_ready":
                            raise RuntimeError(
                                f"AsyncEnvPool worker {i} failed to set its CPU "
                                f"affinity: {payload}"
                            )
                except Exception:
                    for thread in self.threads:
                        if thread.is_alive():
                            thread.terminate()
                    for thread in self.threads:
                        thread.join()
                    raise
            # Get specs from each worker and cache them for _get_child_specs()
            for worker_index, env_indices in enumerate(self._worker_env_indices):
                self.input_queue[worker_index].put(
                    ("get_specs", [(env_index, None) for env_index in env_indices])
                )
            self._child_specs = [None] * self.num_envs
            for worker_index, env_indices in enumerate(self._worker_env_indices):
                for _ in env_indices:
                    env_index, spec = _receive_batch(
                        self.output_queue[worker_index],
                        1,
                        1,
                        None,
                        check_worker_errors=self._check_worker_errors,
                    )[0]
                    self._child_specs[env_index] = spec
            # Batch sizes are already available from the worker specs. Caching them
            # here avoids a later round trip over the input queues used for steps,
            # which may block behind in-flight env work.
            self._env_batch_sizes = [
                torch.Size(spec.shape) for spec in self._child_specs
            ]
            specs = torch.stack(list(self._child_specs))
            output_spec = specs["output_spec"]
            input_spec = specs["input_spec"]
            if self.exchange in ("shm", "auto"):
                for worker_index, env_indices in enumerate(self._worker_env_indices):
                    self.input_queue[worker_index].put(
                        (
                            "get_fake_tensordict",
                            [(env_index, None) for env_index in env_indices],
                        )
                    )
                fake_tensordicts = [None] * self.num_envs
                for worker_index, env_indices in enumerate(self._worker_env_indices):
                    for _ in env_indices:
                        env_index, fake = _receive_batch(
                            self.output_queue[worker_index],
                            1,
                            1,
                            None,
                            check_worker_errors=self._check_worker_errors,
                        )[0]
                        fake_tensordicts[env_index] = fake
                try:
                    self._slot_exchange = _SharedSlotExchange(fake_tensordicts)
                except (TypeError, ValueError, RuntimeError) as err:
                    if self.exchange == "shm":
                        raise
                    torchrl_logger.info(
                        "AsyncEnvPool(exchange='auto'): the env schema does not "
                        f"support the shared-memory exchange, falling back to "
                        f"the queue exchange. Reason: {err}"
                    )
                if self._slot_exchange is not None:
                    for worker_index, env_indices in enumerate(
                        self._worker_env_indices
                    ):
                        self.input_queue[worker_index].put(
                            (
                                "init_shm",
                                self._slot_exchange.worker_slots(
                                    env_indices[0], env_indices[-1] + 1
                                ),
                            )
                        )
                    for worker_index, env_indices in enumerate(
                        self._worker_env_indices
                    ):
                        for _ in env_indices:
                            _receive_batch(
                                self.output_queue[worker_index],
                                1,
                                1,
                                None,
                                check_worker_errors=self._check_worker_errors,
                            )[0]
            return output_spec, input_spec
        except Exception:
            for process in self.threads:
                if process.is_alive():
                    process.terminate()
            for process in self.threads:
                process.join()
            raise

    def _get_child_specs(self) -> list:
        """Returns the cached specs from each child environment process."""
        return self._child_specs

    @property
    def env_batch_sizes(self) -> list[torch.Size]:
        return self._env_batch_sizes

    def _send_worker_batches(
        self,
        msg: str,
        env_idx: list[int],
        local_tds: tuple[TensorDictBase, ...],
        *,
        per_env: bool,
        record_action: bool,
    ) -> None:
        requests = [[] for _ in range(self.num_workers)]
        for env_index, local_td in _zip_strict(env_idx, local_tds):
            data = self._prepare_worker_data(
                env_index, local_td, record_action=record_action
            )
            requests[self._env_to_worker[env_index]].append((env_index, data))
        for worker_index, worker_requests in enumerate(requests):
            if worker_requests:
                self.input_queue[worker_index].put((msg, worker_requests, per_env))

    def _prepare_worker_data(
        self,
        env_index: int,
        tensordict: TensorDictBase,
        *,
        record_action: bool,
    ):
        if self._slot_exchange is None:
            return tensordict
        return self._slot_exchange.write_input(
            env_index, tensordict, record_action=record_action
        )

    def _check_worker_errors(self) -> None:
        if self._worker_error is None:
            try:
                env_index, error = self._error_queue.get_nowait()
            except queue.Empty:
                if self._worker_liveness_timer.elapsed() >= 0.1:
                    self._worker_liveness_timer.start()
                    failed = [p for p in self.threads if p.exitcode is not None]
                    if failed:
                        self._worker_error = (
                            "AsyncEnvPool worker process exited unexpectedly."
                        )
            else:
                self._worker_error = (
                    f"AsyncEnvPool environment {env_index} failed:\n{error}"
                )
        if self._worker_error is not None:
            raise RuntimeError(self._worker_error)

    def _receive_items(
        self,
        result_queue,
        min_get: int,
        max_get: int | None,
        timeout: float | None,
        *,
        track_action: bool,
    ):
        if self._slot_exchange is None:
            return _receive_batch(
                result_queue,
                min_get,
                max_get,
                timeout,
                check_worker_errors=self._check_worker_errors,
            )
        return self._slot_exchange.receive(
            result_queue,
            min_get,
            max_get,
            timeout,
            track_action=track_action,
            check_worker_errors=self._check_worker_errors,
        )

    def _stack_queue_results(self, results) -> TensorDictBase:
        result = self._stack_func(results)
        if isinstance(result, LazyStackedTensorDict):
            # A lazy stack retains the individual TensorDicts reconstructed by
            # multiprocessing.Queue, and therefore one shared-memory mapping
            # per tensor. Clone its constituents into process-private storage;
            # dense stacks already own their newly allocated tensor storage.
            result = result.clone()
        return result

    def async_step_send(
        self, tensordict: TensorDictBase, env_index: int | list[int] | None = None
    ) -> None:
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, False)
        _per_env = isinstance(env_index, int)

        if not _per_env:
            if self._busy.intersection(env_idx):
                raise RuntimeError(
                    f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
                )
            self._busy.update(env_idx)

        self._send_worker_batches(
            "step",
            env_idx,
            tensordict.unbind(0),
            per_env=_per_env,
            record_action=True,
        )
        if not _per_env:
            self._current_step = self._current_step + len(env_idx)

    def async_step_recv(
        self,
        min_get: int = 1,
        env_index: int | None = None,
        *,
        max_get: int | None = None,
        timeout: float | None = None,
    ) -> TensorDictBase:
        if env_index is not None:
            if self._slot_exchange is None:
                return _receive_batch(
                    self._per_env_step_queues[env_index],
                    1,
                    1,
                    None,
                    check_worker_errors=self._check_worker_errors,
                )[0].clone()
            descriptor = self._slot_exchange.receive_one(
                self._per_env_step_queues[env_index],
                track_action=True,
                check_worker_errors=self._check_worker_errors,
            )
            return self._slot_exchange.read_one(descriptor)
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_step:
            raise RuntimeError(
                f"Cannot await {min_get} step when only {self._current_step} are being stepped."
            )
        r = self._receive_items(
            self.step_queue,
            min_get,
            max_get,
            timeout,
            track_action=True,
        )
        self._current_step = self._current_step - len(r)
        if self._slot_exchange is not None:
            idx = [item[0] for item in r]
            result = self._slot_exchange.read(r, self._stack_func)
            self._busy.difference_update(idx)
            return result
        r, idx = self._sort_results(r)
        self._busy.difference_update(idx)
        return self._stack_queue_results(r)

    def _async_private_step_send(
        self, tensordict: TensorDictBase, env_index: int | list[int] | None = None
    ) -> None:
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, False)

        if self._busy.intersection(env_idx):
            raise RuntimeError(
                f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
            )
        self._busy.update(env_idx)

        self._send_worker_batches(
            "_step",
            env_idx,
            tensordict.unbind(0),
            per_env=False,
            record_action=True,
        )
        self._current_step = self._current_step + len(env_idx)

    _async_private_step_recv = async_step_recv

    def async_step_and_maybe_reset_send(
        self, tensordict: TensorDictBase, env_index: int | list[int] | None = None
    ) -> None:
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, False)
        _per_env = isinstance(env_index, int)

        if not _per_env:
            if self._busy.intersection(env_idx):
                raise RuntimeError(
                    f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
                )
            self._busy.update(env_idx)
        if not _per_env:
            self._current_step_reset = self._current_step_reset + len(env_idx)
        self._send_worker_batches(
            "step_and_maybe_reset",
            env_idx,
            tensordict.unbind(0),
            per_env=_per_env,
            record_action=True,
        )

    def async_step_and_maybe_reset_recv(
        self,
        min_get: int = 1,
        env_index: int | None = None,
        *,
        max_get: int | None = None,
        timeout: float | None = None,
    ) -> tuple[TensorDictBase, TensorDictBase]:
        if env_index is not None:
            if self._slot_exchange is None:
                result, next_result = _receive_batch(
                    self._per_env_step_reset_queues[env_index],
                    1,
                    1,
                    None,
                    check_worker_errors=self._check_worker_errors,
                )[0]
                return result.clone(), next_result.clone()
            descriptor = self._slot_exchange.receive_one(
                self._per_env_step_reset_queues[env_index],
                track_action=True,
                check_worker_errors=self._check_worker_errors,
            )
            return self._slot_exchange.read_pair_one(descriptor)
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_step_reset:
            raise RuntimeError(
                f"Cannot await {min_get} step_and_maybe_reset when only {self._current_step_reset} are being stepped."
            )
        r = self._receive_items(
            self.step_reset_queue,
            min_get,
            max_get,
            timeout,
            track_action=True,
        )
        self._current_step_reset = self._current_step_reset - len(r)
        if self._slot_exchange is not None:
            idx = [item[0] for item in r]
            result = self._slot_exchange.read_pair(r, self._stack_func)
            self._busy.difference_update(idx)
            return result
        r, r_ = zip(*r)
        r, r_, idx = self._sort_results(r, r_)
        self._busy.difference_update(idx)
        return self._stack_queue_results(r), self._stack_queue_results(r_)

    def async_reset_send(
        self,
        tensordict: TensorDictBase | None = None,
        env_index: int | list[int] | None = None,
    ) -> None:
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, True)
        _per_env = isinstance(env_index, int)

        if not _per_env:
            if self._busy.intersection(env_idx):
                raise RuntimeError(
                    f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
                )
            self._busy.update(env_idx)
        if not _per_env:
            self._current_reset = self._current_reset + len(env_idx)
        self._send_worker_batches(
            "reset",
            env_idx,
            tensordict.unbind(0),
            per_env=_per_env,
            record_action=False,
        )

    def async_reset_recv(
        self,
        min_get: int | None = None,
        env_index: int | None = None,
        *,
        max_get: int | None = None,
        timeout: float | None = None,
    ) -> TensorDictBase:
        if env_index is not None:
            if self._slot_exchange is None:
                return _receive_batch(
                    self._per_env_reset_queues[env_index],
                    1,
                    1,
                    None,
                    check_worker_errors=self._check_worker_errors,
                )[0].clone()
            descriptor = self._slot_exchange.receive_one(
                self._per_env_reset_queues[env_index],
                track_action=False,
                check_worker_errors=self._check_worker_errors,
            )
            return self._slot_exchange.read_one(descriptor)
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_reset:
            raise RuntimeError(
                f"Cannot await {min_get} reset when only {self._current_reset} are being reset."
            )
        r = self._receive_items(
            self.reset_queue,
            min_get,
            max_get,
            timeout,
            track_action=False,
        )
        self._current_reset = self._current_reset - len(r)
        if self._slot_exchange is not None:
            idx = [item[0] for item in r]
            result = self._slot_exchange.read(r, self._stack_func)
            self._busy.difference_update(idx)
            return result
        r, idx = self._sort_results(r)
        self._busy.difference_update(idx)
        return self._stack_queue_results(r)

    def _async_private_reset_send(
        self,
        tensordict: TensorDictBase | None = None,
        env_index: int | list[int] | None = None,
    ) -> None:
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, True)

        if self._busy.intersection(env_idx):
            raise RuntimeError(
                f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
            )
        self._busy.update(env_idx)
        self._current_reset = self._current_reset + len(env_idx)
        self._send_worker_batches(
            "_reset",
            env_idx,
            tensordict.unbind(0),
            per_env=False,
            record_action=False,
        )

    _async_private_reset_recv = async_reset_recv

    _SHUTDOWN_TIMEOUT = 60.0

    def _drain_result_queues(self) -> None:
        for result_queue in (
            self.step_queue,
            self.reset_queue,
            self.step_reset_queue,
            *self._per_env_step_queues,
            *self._per_env_reset_queues,
            *self._per_env_step_reset_queues,
        ):
            while True:
                try:
                    result_queue.get_nowait()
                except queue.Empty:
                    break
                except (EOFError, OSError):
                    # The item has already been removed from the queue, but
                    # rebuilding a discarded tensor can fail once its worker's
                    # resource sharer has exited. Keep draining the remaining
                    # items so other workers can finish flushing their queues.
                    continue

    def shutdown(self):
        deadline = timeit("async_env_shutdown_deadline").start()
        pending = set(range(self.num_workers))
        while pending and deadline.elapsed() < self._SHUTDOWN_TIMEOUT:
            self._drain_result_queues()
            for worker_index in tuple(pending):
                if not self.threads[worker_index].is_alive():
                    pending.remove(worker_index)
                    continue
                try:
                    self.input_queue[worker_index].put(("shutdown", None), timeout=0.01)
                except queue.Full:
                    continue
                pending.remove(worker_index)

        # A worker whose unread results still sit in a result queue cannot
        # exit: its process teardown joins the queue's feeder thread, which
        # blocks writing into the full pipe that nothing reads any more.
        # Draining the result queues while joining unblocks those feeders so
        # the workers exit through the normal teardown path.
        for thread in self.threads:
            while thread.is_alive() and deadline.elapsed() < self._SHUTDOWN_TIMEOUT:
                self._drain_result_queues()
                thread.join(timeout=0.1)
        stragglers = [thread for thread in self.threads if thread.is_alive()]
        if stragglers:
            torchrl_logger.warning(
                f"AsyncEnvPool.shutdown: terminating {len(stragglers)} worker "
                f"process(es) that did not exit within "
                f"{self._SHUTDOWN_TIMEOUT}s."
            )
            for thread in stragglers:
                thread.terminate()
            for thread in stragglers:
                thread.join()

    @classmethod
    def _worker_exec(
        cls,
        env_indices,
        env_makers,
        create_env_kwargs,
        input_queue,
        output_queue,
        step_queue,
        step_reset_queue,
        reset_queue,
        per_env_step_queues=None,
        per_env_reset_queues=None,
        per_env_step_reset_queues=None,
        cpu_affinity=None,
        error_queue=None,
    ):
        if cpu_affinity is not None:
            try:
                os.sched_setaffinity(0, cpu_affinity)
            except Exception as err:
                output_queue.put(("affinity_error", repr(err)))
                return
            output_queue.put(("affinity_ready", None))
        local_input_queues = {env_index: queue.Queue(1) for env_index in env_indices}
        env_threads = []
        for (
            env_index,
            env_maker,
            kwargs,
            per_env_step_queue,
            per_env_reset_queue,
            per_env_step_reset_queue,
        ) in zip(
            env_indices,
            env_makers,
            create_env_kwargs,
            per_env_step_queues,
            per_env_reset_queues,
            per_env_step_reset_queues,
        ):
            env_thread = threading.Thread(
                target=cls._env_exec,
                daemon=True,
                kwargs={
                    "i": env_index,
                    "error_queue": error_queue,
                    "env_or_factory": env_maker,
                    "create_env_kwargs": kwargs,
                    "input_queue": local_input_queues[env_index],
                    "output_queue": output_queue,
                    "step_queue": step_queue,
                    "step_reset_queue": step_reset_queue,
                    "reset_queue": reset_queue,
                    "per_env_step_queue": per_env_step_queue,
                    "per_env_reset_queue": per_env_reset_queue,
                    "per_env_step_reset_queue": per_env_step_reset_queue,
                },
            )
            env_threads.append(env_thread)
            env_thread.start()

        while True:
            msg_data = input_queue.get()
            if len(msg_data) == 3:
                msg, requests, per_env = msg_data
            else:
                msg, requests = msg_data
                per_env = False
            if msg == "shutdown":
                for env_thread, local_queue in zip(
                    env_threads, local_input_queues.values()
                ):
                    if env_thread.is_alive():
                        local_queue.put(("shutdown", None))
                for env_thread in env_threads:
                    env_thread.join()
                break
            if msg == "init_shm":
                input_slots, result_slots, next_slots, clock = requests
                requests = [
                    (env_index, (input_slot, result_slot, next_slot, clock))
                    for env_index, input_slot, result_slot, next_slot in zip(
                        env_indices,
                        input_slots.unbind(0),
                        result_slots.unbind(0),
                        next_slots.unbind(0),
                    )
                ]
            for env_index, data in requests:
                local_input_queues[env_index].put((msg, data, per_env))

    @classmethod
    def _env_exec(
        cls,
        i,
        env_or_factory,
        create_env_kwargs,
        input_queue,
        output_queue,
        step_queue,
        step_reset_queue,
        reset_queue,
        per_env_step_queue=None,
        per_env_reset_queue=None,
        per_env_step_reset_queue=None,
        cpu_affinity=None,
        error_queue=None,
        grouped_input=False,
    ):
        try:
            if cpu_affinity is not None:
                try:
                    os.sched_setaffinity(0, cpu_affinity)
                except Exception as err:
                    output_queue.put(("affinity_error", repr(err)))
                    return
                output_queue.put(("affinity_ready", None))
            if not isinstance(env_or_factory, EnvBase):
                env = env_or_factory(**create_env_kwargs)
            else:
                env = env_or_factory
            shared_slots = None

            while True:
                msg_data = input_queue.get()
                if len(msg_data) == 3:
                    msg, data, per_env = msg_data
                else:
                    msg, data = msg_data
                    per_env = False
                if grouped_input and msg != "shutdown":
                    if msg == "init_shm":
                        input_slots, result_slots, next_slots, clock = data
                        data = (
                            input_slots[0],
                            result_slots[0],
                            next_slots[0],
                            clock,
                        )
                    else:
                        _, data = data[0]
                if msg == "get_specs":
                    output_queue.put((i, env.specs))
                elif msg == "get_fake_tensordict":
                    output_queue.put((i, env.fake_tensordict()))
                elif msg == "init_shm":
                    shared_slots = data
                    output_queue.put(True)
                elif msg == "reset":
                    if shared_slots is not None:
                        data = shared_slots[0].select(*data, strict=True)
                    data = env.reset(data)
                    target = per_env_reset_queue if per_env else reset_queue
                    if shared_slots is None:
                        data.set(cls._env_idx_key, NonTensorData(i))
                        target.put(data)
                    else:
                        keys, ready_s = _SharedSlotExchange.publish(
                            shared_slots[1], data, shared_slots[3]
                        )
                        target.put((i, keys, ready_s))
                elif msg == "_reset":
                    if shared_slots is not None:
                        data = shared_slots[0].select(*data, strict=True)
                    data = env._reset(data)
                    if shared_slots is None:
                        data.set(cls._env_idx_key, NonTensorData(i))
                        reset_queue.put(data)
                    else:
                        keys, ready_s = _SharedSlotExchange.publish(
                            shared_slots[1], data, shared_slots[3]
                        )
                        reset_queue.put((i, keys, ready_s))
                elif msg == "step_and_maybe_reset":
                    if shared_slots is not None:
                        data = shared_slots[0].select(*data, strict=True)
                    data, data_ = env.step_and_maybe_reset(data)
                    target = per_env_step_reset_queue if per_env else step_reset_queue
                    if shared_slots is None:
                        data.set(cls._env_idx_key, NonTensorData(i))
                        data_.set(cls._env_idx_key, NonTensorData(i))
                        target.put((data, data_))
                    else:
                        (
                            result_keys,
                            next_keys,
                            ready_s,
                        ) = _SharedSlotExchange.publish_pair(
                            shared_slots[1],
                            shared_slots[2],
                            data,
                            data_,
                            shared_slots[3],
                        )
                        target.put((i, result_keys, next_keys, ready_s))
                elif msg == "step":
                    if shared_slots is not None:
                        data = shared_slots[0].select(*data, strict=True)
                    data = env.step(data)
                    target = per_env_step_queue if per_env else step_queue
                    if shared_slots is None:
                        data.set(cls._env_idx_key, NonTensorData(i))
                        target.put(data)
                    else:
                        keys, ready_s = _SharedSlotExchange.publish(
                            shared_slots[1], data, shared_slots[3]
                        )
                        target.put((i, keys, ready_s))
                elif msg == "_step":
                    if shared_slots is not None:
                        data = shared_slots[0].select(*data, strict=True)
                    data = env._step(data)
                    if shared_slots is None:
                        data.set(cls._env_idx_key, NonTensorData(i))
                        step_queue.put(data)
                    else:
                        keys, ready_s = _SharedSlotExchange.publish(
                            shared_slots[1], data, shared_slots[3]
                        )
                        step_queue.put((i, keys, ready_s))
                elif msg == "shutdown":
                    env.close()
                    break
                else:
                    raise RuntimeError(f"Unknown msg {msg} for worker {i}")
        except Exception:
            if error_queue is None:
                raise
            error_queue.put((i, traceback.format_exc()))


class ThreadingAsyncEnvPool(AsyncEnvPool):
    """An implementation of `AsyncEnvPool` using threading for parallel execution of environments.

    This class manages a pool of environments, each running in its own thread, and
    provides methods for asynchronous stepping and resetting of environments using
    a thread pool executor.

    Supports per-env ``recv`` via ``env_index`` for thread-safe concurrent access
    from multiple collector threads.

    .. note:: This class and its subclasses should work when nested in with :class:`~torchrl.envs.TransformedEnv` and
        batched environments, but users won't currently be able to use the async features of the base environment when
        it's nested in these classes. One should prefer nested transformed envs within an `AsyncEnvPool` instead.
        If this is not possible, please raise an issue.

    Methods:
        _setup(): Initializes the thread pool and environment instances.
        async_step_send(tensordict): Sends a step command to the environments.
        async_step_recv(min_get): Receives the results of the step command.
        async_reset_send(tensordict): Sends a reset command to the environments.
        async_reset_recv(min_get): Receives the results of the reset command.
        shutdown(): Shuts down the thread pool.

    """

    def _setup(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=self.num_envs)
        self.envs = [
            env_factory(**create_env_kwargs)
            if not isinstance(env_factory, EnvBase)
            else env_factory
            for env_factory, create_env_kwargs in zip(
                self.env_makers, self.create_env_kwargs
            )
        ]
        self._reset_futures = []
        self._private_reset_futures = []
        self._step_futures = []
        self._private_step_futures = []
        self._step_and_maybe_reset_futures = []
        # Per-env future dicts for thread-safe per-env recv
        self._per_env_step_futures: dict[int, object] = {}
        self._per_env_step_reset_futures: dict[int, object] = {}
        self._per_env_reset_futures: dict[int, object] = {}
        self._current_step = 0
        self._current_step_reset = 0
        self._current_reset = 0

        # get specs
        specs = torch.stack([env.specs for env in self.envs])
        return specs["output_spec"].clone(), specs["input_spec"].clone()

    @property
    def env_batch_sizes(self) -> list[torch.Size]:
        return [env.batch_size for env in self.envs]

    def _get_child_specs(self) -> list:
        """Returns the specs from each child environment."""
        return [env.specs for env in self.envs]

    @classmethod
    def _get_specs(cls, env: EnvBase):
        return env.specs

    @classmethod
    def _step_func(cls, env_td: tuple[EnvBase, TensorDictBase, int]):
        env, td, idx = env_td
        return env.step(td).set(cls._env_idx_key, NonTensorData(idx))

    @classmethod
    def _private_step_func(cls, env_td: tuple[EnvBase, TensorDictBase, int]):
        env, td, idx = env_td
        return env._step(td).set(cls._env_idx_key, NonTensorData(idx))

    @classmethod
    def _reset_func(cls, env_td: tuple[EnvBase, TensorDictBase]):
        env, td, idx = env_td
        return env.reset(td).set(cls._env_idx_key, NonTensorData(idx))

    @classmethod
    def _private_reset_func(cls, env_td: tuple[EnvBase, TensorDictBase]):
        env, td, idx = env_td
        return env._reset(td).set(cls._env_idx_key, NonTensorData(idx))

    @classmethod
    def _step_and_maybe_reset_func(cls, env_td: tuple[EnvBase, TensorDictBase]):
        env, td, idx = env_td
        td, td_ = env.step_and_maybe_reset(td)
        idx = NonTensorData(idx)
        return td.set(cls._env_idx_key, idx), td_.set(cls._env_idx_key, idx)

    @staticmethod
    def _receive_futures(futures, min_get, max_get, timeout):
        if min_get < 1:
            raise ValueError(f"min_get must be positive, got {min_get}.")
        if max_get is not None and max_get < min_get:
            raise ValueError(
                f"max_get must be greater than or equal to min_get, got "
                f"min_get={min_get} and max_get={max_get}."
            )
        if timeout is not None and timeout < 0:
            raise ValueError(f"timeout must be non-negative, got {timeout}.")
        limit = len(futures) if max_get is None else max_get
        pending = set(futures)
        completed = []
        # The deadline is anchored at call entry and bounds the entire call,
        # including the wait for the first result.
        deadline_timer = (
            None
            if timeout is None
            else timeit("async_env_future_batch_deadline").start()
        )
        while pending and len(completed) < min_get:
            if deadline_timer is None:
                done, pending = wait(pending, return_when=FIRST_COMPLETED)
            else:
                remaining = timeout - deadline_timer.elapsed()
                # A non-positive remaining still polls: wait(timeout=0)
                # returns whatever is already done without blocking.
                done, pending = wait(
                    pending, timeout=max(remaining, 0.0), return_when=FIRST_COMPLETED
                )
                if not done:
                    # Completed futures stay in the pool's pending lists (the
                    # caller only removes returned ones), so no result is
                    # lost and state stays consistent.
                    raise TimeoutError(
                        f"async recv timed out: {len(completed)}/{min_get} "
                        f"results after {timeout}s; completed futures remain "
                        f"available to the next call."
                    )
            completed.extend(list(done)[: limit - len(completed)])
        while pending and len(completed) < limit:
            done = {future for future in pending if future.done()}
            if not done:
                if deadline_timer is None:
                    break
                remaining = timeout - deadline_timer.elapsed()
                if remaining <= 0:
                    break
                done, _ = wait(pending, timeout=remaining, return_when=FIRST_COMPLETED)
                if not done:
                    break
            room = limit - len(completed)
            selected = list(done)[:room]
            completed.extend(selected)
            pending.difference_update(selected)
        return completed

    def async_step_send(
        self, tensordict: TensorDictBase, env_index: int | list[int] | None = None
    ) -> None:
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, False)
        _per_env = isinstance(env_index, int)

        if not _per_env:
            if self._busy.intersection(env_idx):
                raise RuntimeError(
                    f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
                )
            self._busy.update(env_idx)

        tds = tensordict.unbind(0)
        envs = [self.envs[idx] for idx in env_idx]
        futures = [
            self._pool.submit(self._step_func, (env, td, idx))
            for env, td, idx in zip(envs, tds, env_idx)
        ]
        if _per_env:
            self._per_env_step_futures[env_index] = futures[0]
        else:
            self._step_futures.extend(futures)
            self._current_step = self._current_step + len(futures)

    def async_step_recv(
        self,
        min_get: int | None = None,
        env_index: int | None = None,
        *,
        max_get: int | None = None,
        timeout: float | None = None,
    ) -> TensorDictBase:
        if env_index is not None:
            future = self._per_env_step_futures.pop(env_index)
            return future.result()
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_step:
            raise RuntimeError(
                f"Cannot await {min_get} step when only {self._current_step_reset} are being stepped."
            )
        futures = self._step_futures
        completed_futures = self._receive_futures(futures, min_get, max_get, timeout)
        results = [future.result() for future in completed_futures]
        self._current_step -= len(completed_futures)
        self._step_futures = [
            f for f in self._step_futures if f not in completed_futures
        ]
        results, idx = self._sort_results(results)
        self._busy.difference_update(idx)
        return self._stack_func(results)

    def _async_private_step_send(
        self, tensordict: TensorDictBase, env_index: int | list[int] | None = None
    ) -> None:
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, False)

        if self._busy.intersection(env_idx):
            raise RuntimeError(
                f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
            )
        self._busy.update(env_idx)

        tds = tensordict.unbind(0)
        envs = [self.envs[idx] for idx in env_idx]
        futures = [
            self._pool.submit(self._private_step_func, (env, td, idx))
            for env, td, idx in zip(envs, tds, env_idx)
        ]
        self._private_step_futures.extend(futures)
        self._current_step = self._current_step + len(futures)

    def _async_private_step_recv(self, min_get: int | None = None) -> TensorDictBase:
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_step:
            raise RuntimeError(
                f"Cannot await {min_get} step when only {self._current_step_reset} are being stepped."
            )
        results = []
        futures = self._private_step_futures
        completed_futures = []
        for future in as_completed(futures):
            results.append(future.result())
            completed_futures.append(future)
            self._current_step = self._current_step - 1
            if len(results) >= min_get and sum([f.done() for f in futures]) == 0:
                break
        self._private_step_futures = [
            f for f in self._private_step_futures if f not in completed_futures
        ]
        results, idx = self._sort_results(results)
        self._busy.difference_update(idx)
        return self._stack_func(results)

    def async_step_and_maybe_reset_send(
        self, tensordict: TensorDictBase, env_index: int | list[int] | None = None
    ) -> None:
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, False)
        _per_env = isinstance(env_index, int)

        if not _per_env:
            if self._busy.intersection(env_idx):
                raise RuntimeError(
                    f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
                )
            self._busy.update(env_idx)

        tds = tensordict.unbind(0)
        envs = [self.envs[idx] for idx in env_idx]
        futures = [
            self._pool.submit(self._step_and_maybe_reset_func, (env, td, idx))
            for env, td, idx in zip(envs, tds, env_idx)
        ]
        if _per_env:
            self._per_env_step_reset_futures[env_index] = futures[0]
        else:
            self._step_and_maybe_reset_futures.extend(futures)
            self._current_step_reset = self._current_step_reset + len(futures)

    def async_step_and_maybe_reset_recv(
        self,
        min_get: int | None = None,
        env_index: int | None = None,
        *,
        max_get: int | None = None,
        timeout: float | None = None,
    ) -> tuple[TensorDictBase, TensorDictBase]:
        if env_index is not None:
            future = self._per_env_step_reset_futures.pop(env_index)
            return future.result()
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_step_reset:
            raise RuntimeError(
                f"Cannot await {min_get} step_and_maybe_reset when only {self._current_step_reset} are being stepped."
            )
        futures = self._step_and_maybe_reset_futures
        completed_futures = self._receive_futures(futures, min_get, max_get, timeout)
        results = [future.result() for future in completed_futures]
        self._current_step_reset -= len(completed_futures)
        self._step_and_maybe_reset_futures = [
            f for f in self._step_and_maybe_reset_futures if f not in completed_futures
        ]
        results, results_ = zip(*results)
        results, results_, idx = self._sort_results(results, results_)
        self._busy.difference_update(idx)
        return self._stack_func(results), self._stack_func(results_)

    def async_reset_send(
        self,
        tensordict: TensorDictBase | None = None,
        env_index: int | list[int] | None = None,
    ) -> None:
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, True)
        _per_env = isinstance(env_index, int)

        if not _per_env:
            if self._busy.intersection(env_idx):
                raise RuntimeError(
                    f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
                )
            self._busy.update(env_idx)

        tds = tensordict.unbind(0)
        envs = [self.envs[idx] for idx in env_idx]
        futures = [
            self._pool.submit(self._reset_func, (env, td, idx))
            for env, td, idx in zip(envs, tds, env_idx)
        ]
        if _per_env:
            self._per_env_reset_futures[env_index] = futures[0]
        else:
            self._current_reset = self._current_reset + len(futures)
            self._reset_futures.extend(futures)

    def async_reset_recv(
        self,
        min_get: int | None = None,
        env_index: int | None = None,
        *,
        max_get: int | None = None,
        timeout: float | None = None,
    ) -> TensorDictBase:
        if env_index is not None:
            future = self._per_env_reset_futures.pop(env_index)
            return future.result()
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_reset:
            raise RuntimeError(
                f"Cannot await {min_get} reset when only {self._current_step_reset} are being reset."
            )
        futures = self._reset_futures
        completed_futures = self._receive_futures(futures, min_get, max_get, timeout)
        results = [future.result() for future in completed_futures]
        self._current_reset -= len(completed_futures)
        self._reset_futures = [
            f for f in self._reset_futures if f not in completed_futures
        ]
        results, idx = self._sort_results(results)
        self._busy.difference_update(idx)
        return self._stack_func(results)

    def _async_private_reset_send(
        self,
        tensordict: TensorDictBase | None = None,
        env_index: int | list[int] | None = None,
    ) -> None:
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, True)

        if self._busy.intersection(env_idx):
            raise RuntimeError(
                f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
            )
        self._busy.update(env_idx)

        tds = tensordict.unbind(0)
        envs = [self.envs[idx] for idx in env_idx]
        futures = [
            self._pool.submit(self._private_reset_func, (env, td, idx))
            for env, td, idx in zip(envs, tds, env_idx)
        ]
        self._current_reset = self._current_reset + len(futures)
        self._private_reset_futures.extend(futures)

    def _async_private_reset_recv(self, min_get: int | None = None) -> TensorDictBase:
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_reset:
            raise RuntimeError(
                f"Cannot await {min_get} reset when only {self._current_step_reset} are being reset."
            )
        results = []
        futures = self._private_reset_futures
        completed_futures = []
        for future in as_completed(futures):
            results.append(future.result())
            completed_futures.append(future)
            self._current_reset = self._current_reset - 1
            if len(results) >= min_get and sum([f.done() for f in futures]) == 0:
                break
        self._private_reset_futures = [
            f for f in self._private_reset_futures if f not in completed_futures
        ]
        results, idx = self._sort_results(results)
        self._busy.difference_update(idx)
        return self._stack_func(results)

    def shutdown(self):
        self._pool.shutdown()
