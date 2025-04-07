# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc

import multiprocessing
from concurrent.futures import as_completed, ThreadPoolExecutor

# import queue
from multiprocessing import Queue
from queue import Empty
from typing import Callable, Literal, Sequence

import torch
from tensordict import (
    lazy_stack,
    LazyStackedTensorDict,
    maybe_dense_stack,
    TensorDict,
    TensorDictBase,
)

from tensordict.tensorclass import NonTensorData, NonTensorStack
from tensordict.utils import _zip_strict

from torchrl.data.tensor_specs import NonTensor
from torchrl.envs.common import _EnvPostInit, EnvBase


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

    Attributes:
        min_get (int): Minimum number of environments to process in a batch.
        env_makers (list): List of environment makers or environments.
        num_envs (int): Number of environments in the pool.
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
    ) -> None:
        if not isinstance(env_makers, Sequence):
            env_makers = [env_makers]
        self.min_get = 1
        self.env_makers = env_makers
        self.num_envs = len(env_makers)
        self.backend = backend

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
        super().__init__(batch_size=[self.num_envs])
        self._busy = set()

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
                    *[TensorDict() for _ in range(self.num_envs)]
                )
            else:
                tensordict = TensorDict(batch_size=self.num_envs)
        tensordict.set(self._env_idx_key, torch.arange(tensordict.shape[0]))
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
                    *[TensorDict() for _ in range(self.num_envs)]
                )
            else:
                tensordict = TensorDict(batch_size=self.num_envs)
        tensordict.set(self._env_idx_key, torch.arange(tensordict.shape[0]))
        self.async_reset_send(tensordict)
        tensordict = self.async_reset_recv(min_get=self.num_envs)
        return tensordict

    def _sort_results(self, results, *other_results):
        idx = [int(r[self._env_idx_key]) for r in results]
        argsort = torch.argsort(torch.tensor(idx)).tolist()
        results = [results[i] for i in argsort]
        if other_results:
            other_results = [
                [other_results[i] for i in argsort] for other_results in other_results
            ]
            return results, *other_results, idx
        return results, idx

    def _set_seed(self, seed: int | None):
        raise NotImplementedError

    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    def _maybe_make_tensordict(self, tensordict, env_index, make_if_none):
        if env_index is None:
            env_idx = tensordict[self._env_idx_key]
            if isinstance(env_idx, torch.Tensor):
                env_idx = env_idx.tolist()
            if isinstance(env_idx, int):
                env_idx = [env_idx]
                tensordict = tensordict.unsqueeze(0)
        elif isinstance(env_index, int):
            if make_if_none:
                if tensordict is None:
                    tensordict = TensorDict(batch_size=(), device=self.device)
                if self.stack in ("lazy_stack", "maybe_dense"):
                    tensordict = tensordict.unsqueeze(0)
                else:
                    tensordict = LazyStackedTensorDict(tensordict)
            tensordict[self._env_idx_key] = NonTensorStack(env_index)
            env_idx = [env_index]
        else:
            if make_if_none and tensordict is None:
                if self.stack in ("lazy_stack", "maybe_dense"):
                    tensordict = LazyStackedTensorDict(
                        *[TensorDict(device=self.device) for _ in env_index]
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
    def async_step_recv(self, min_get: int | None = None) -> TensorDictBase:
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
    ) -> tuple[TensorDictBase, TensorDictBase]:
        raise NotImplementedError

    @abc.abstractmethod
    def async_reset_send(
        self,
        tensordict: TensorDictBase | None = None,
        env_index: int | list[int] | None = None,
    ) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def async_reset_recv(self, min_get: int | None = None) -> TensorDictBase:
        raise NotImplementedError

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

    This class manages a pool of environments, each running in its own process, and
    provides methods for asynchronous stepping and resetting of environments using
    inter-process communication.

    .. note:: This class and its subclasses should work when nested in with :class:`~torchrl.envs.TransformedEnv` and
        batched environments, but users won't currently be able to use the async features of the base environment when
        it's nested in these classes. One should prefer nested transformed envs within an `AsyncEnvPool` instead.
        If this is not possible, please raise an issue.

    Methods:
        _setup(): Initializes the multiprocessing queues and processes for each
            environment.
        async_step_send(tensordict): Sends a step command to the environments.
        async_step_recv(min_get): Receives the results of the step command.
        async_reset_send(tensordict): Sends a reset command to the environments.
        async_reset_recv(min_get): Receives the results of the reset command.
        shutdown(): Shuts down all environment processes.
    """

    def _setup(self) -> None:
        self.step_queue = Queue(maxsize=self.num_envs)
        self.reset_queue = Queue(maxsize=self.num_envs)
        self.step_reset_queue = Queue(maxsize=self.num_envs)
        self.input_queue = [Queue(maxsize=1) for _ in range(self.num_envs)]
        self.output_queue = [Queue(maxsize=1) for _ in range(self.num_envs)]
        self._current_reset = 0
        self._current_step = 0
        self._current_step_reset = 0

        num_threads = self.num_envs
        assert num_threads > 0
        self.threads = []
        for i in range(num_threads):
            # thread = threading.Thread(target=_env_exec, kwargs={"i": i, "env_or_factory": self.env_maker[i], "input_queue": self.input_queue[i], "step_queue": self.step_queue, "reset_queue": self.reset_queue})
            thread = multiprocessing.Process(
                target=self._env_exec,
                kwargs={
                    "i": i,
                    "env_or_factory": self.env_makers[i],
                    "input_queue": self.input_queue[i],
                    "output_queue": self.output_queue[i],
                    "step_reset_queue": self.step_reset_queue,
                    "step_queue": self.step_queue,
                    "reset_queue": self.reset_queue,
                },
            )
            self.threads.append(thread)
            thread.start()
        # Get specs
        for i in range(num_threads):
            self.input_queue[i].put(("get_specs", None))
        specs = []
        for i in range(num_threads):
            specs.append(self.output_queue[i].get())
        specs = torch.stack(list(specs))
        output_spec = specs["output_spec"]
        input_spec = specs["input_spec"]
        return output_spec, input_spec

    def async_step_send(
        self, tensordict: TensorDictBase, env_index: int | list[int] | None = None
    ) -> None:
        # puts tds in a queue and ask for env.step
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, False)

        if self._busy.intersection(env_idx):
            raise RuntimeError(
                f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
            )
        self._busy.update(env_idx)

        local_tds = tensordict.unbind(0)
        for _env_idx, local_td in _zip_strict(env_idx, local_tds):
            self.input_queue[_env_idx].put(("step", local_td))
        self._current_step = self._current_step + len(env_idx)

    def async_step_recv(self, min_get: int = 1) -> TensorDictBase:
        # gets step results from the queue
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_step:
            raise RuntimeError(
                f"Cannot await {min_get} step when only {self._current_step} are being stepped."
            )
        r = self._wait_for_one_and_get(self.step_queue, min_get)
        self._current_step = self._current_step - len(r)
        assert self._current_step >= 0
        r, idx = self._sort_results(r)
        self._busy.difference_update(idx)
        return self._stack_func(r)

    def _async_private_step_send(
        self, tensordict: TensorDictBase, env_index: int | list[int] | None = None
    ) -> None:
        # puts tds in a queue and ask for env.step
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, False)

        if self._busy.intersection(env_idx):
            raise RuntimeError(
                f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
            )
        self._busy.update(env_idx)

        local_tds = tensordict.unbind(0)
        for _env_idx, local_td in _zip_strict(env_idx, local_tds):
            self.input_queue[_env_idx].put(("_step", local_td))
        self._current_step = self._current_step + len(env_idx)

    _async_private_step_recv = async_step_recv

    def async_step_and_maybe_reset_send(
        self, tensordict: TensorDictBase, env_index: int | list[int] | None = None
    ) -> None:
        # puts tds in a queue and ask for env.step
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, False)

        if self._busy.intersection(env_idx):
            raise RuntimeError(
                f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
            )
        self._busy.update(env_idx)
        local_tds = tensordict.unbind(0)
        for _env_idx, local_td in _zip_strict(env_idx, local_tds):
            self._current_step_reset = self._current_step_reset + 1
            self.input_queue[_env_idx].put(("step_and_maybe_reset", local_td))

    def async_step_and_maybe_reset_recv(self, min_get: int = 1) -> TensorDictBase:
        # gets step results from the queue
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_step_reset:
            raise RuntimeError(
                f"Cannot await {min_get} step_and_maybe_reset when only {self._current_step_reset} are being stepped."
            )
        r = self._wait_for_one_and_get(self.step_reset_queue, min_get)
        self._current_step_reset = self._current_step_reset - len(r)
        r, r_ = zip(*r)
        r, r_, idx = self._sort_results(r, r_)
        self._busy.difference_update(idx)
        return self._stack_func(r), self._stack_func(r_)

    def async_reset_send(
        self,
        tensordict: TensorDictBase | None = None,
        env_index: int | list[int] | None = None,
    ) -> None:
        # puts tds in a queue and ask for env.reset
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, True)

        if self._busy.intersection(env_idx):
            raise RuntimeError(
                f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
            )
        self._busy.update(env_idx)
        local_tds = tensordict.unbind(0)
        for _env_idx, local_td in _zip_strict(env_idx, local_tds):
            self._current_reset = self._current_reset + 1
            self.input_queue[_env_idx].put(("reset", local_td))

    def async_reset_recv(self, min_get: int | None = None) -> TensorDictBase:
        # gets reset results from the queue
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_reset:
            raise RuntimeError(
                f"Cannot await {min_get} reset when only {self._current_reset} are being reset."
            )
        r = self._wait_for_one_and_get(self.reset_queue, min_get)
        self._current_reset = self._current_reset - len(r)
        r, idx = self._sort_results(r)
        self._busy.difference_update(idx)
        return self._stack_func(r)

    def _async_private_reset_send(
        self,
        tensordict: TensorDictBase | None = None,
        env_index: int | list[int] | None = None,
    ) -> None:
        # puts tds in a queue and ask for env.reset
        tensordict, env_idx = self._maybe_make_tensordict(tensordict, env_index, True)

        if self._busy.intersection(env_idx):
            raise RuntimeError(
                f"Some envs are still processing a step: envs that are busy: {self._busy}, queried: {env_idx}."
            )
        self._busy.update(env_idx)
        local_tds = tensordict.unbind(0)
        for _env_idx, local_td in _zip_strict(env_idx, local_tds):
            self._current_reset = self._current_reset + 1
            self.input_queue[_env_idx].put(("_reset", local_td))

    _async_private_reset_recv = async_reset_recv

    def _wait_for_one_and_get(self, q, min_get):
        items = [q.get()]

        try:
            while True:
                item = q.get_nowait()
                items.append(item)
        except Empty:
            pass

        # Retrieve all other available items
        while len(items) < min_get:
            item = q.get()
            items.append(item)

        return items

    def shutdown(self):
        for env_id in range(self.num_envs):
            self.input_queue[env_id].put(("shutdown", None))

        for thread in self.threads:
            thread.join()

    @classmethod
    def _env_exec(
        cls,
        i,
        env_or_factory,
        input_queue,
        output_queue,
        step_queue,
        step_reset_queue,
        reset_queue,
    ):
        if not isinstance(env_or_factory, EnvBase):
            env = env_or_factory()
        else:
            env = env_or_factory

        while True:
            msg_data = input_queue.get()
            msg, data = msg_data
            if msg == "get_specs":
                output_queue.put(env.specs)
            elif msg == "reset":
                data = env.reset(data.copy())
                data.set(cls._env_idx_key, NonTensorData(i))
                reset_queue.put(data)
            elif msg == "_reset":
                data = env._reset(data.copy())
                data.set(cls._env_idx_key, NonTensorData(i))
                reset_queue.put(data)
            elif msg == "step_and_maybe_reset":
                data, data_ = env.step_and_maybe_reset(data.copy())
                data.set(cls._env_idx_key, NonTensorData(i))
                data_.set(cls._env_idx_key, NonTensorData(i))
                step_reset_queue.put((data, data_))
            elif msg == "step":
                data = env.step(data.copy())
                data.set(cls._env_idx_key, NonTensorData(i))
                step_queue.put(data)
            elif msg == "_step":
                data = env._step(data.copy())
                data.set(cls._env_idx_key, NonTensorData(i))
                step_queue.put(data)
            elif msg == "shutdown":
                env.close()
                break
            else:
                raise RuntimeError(f"Unknown msg {msg} for worker {i}")
        return


class ThreadingAsyncEnvPool(AsyncEnvPool):
    """An implementation of `AsyncEnvPool` using threading for parallel execution of environments.

    This class manages a pool of environments, each running in its own thread, and
    provides methods for asynchronous stepping and resetting of environments using
    a thread pool executor.

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
            env_factory() if not isinstance(env_factory, EnvBase) else env_factory
            for env_factory in self.env_makers
        ]
        self._reset_futures = []
        self._private_reset_futures = []
        self._step_futures = []
        self._private_step_futures = []
        self._step_and_maybe_reset_futures = []
        self._current_step = 0
        self._current_step_reset = 0
        self._current_reset = 0

        # get specs
        specs = torch.stack([env.specs for env in self.envs])
        return specs["output_spec"].clone(), specs["input_spec"].clone()

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

    def async_step_send(
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
            self._pool.submit(self._step_func, (env, td, idx))
            for env, td, idx in zip(envs, tds, env_idx)
        ]
        self._step_futures.extend(futures)
        self._current_step = self._current_step + len(futures)

    def async_step_recv(self, min_get: int | None = None) -> TensorDictBase:
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_step:
            raise RuntimeError(
                f"Cannot await {min_get} step when only {self._current_step_reset} are being stepped."
            )
        results = []
        futures = self._step_futures
        completed_futures = []
        for future in as_completed(futures):
            results.append(future.result())
            completed_futures.append(future)
            self._current_step = self._current_step - 1
            if len(results) >= min_get and sum([f.done() for f in futures]) == 0:
                break
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
        self._step_and_maybe_reset_futures.extend(futures)
        self._current_step_reset = self._current_step_reset + len(futures)

    def async_step_and_maybe_reset_recv(
        self, min_get: int | None = None
    ) -> TensorDictBase:
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_step_reset:
            raise RuntimeError(
                f"Cannot await {min_get} step_and_maybe_reset when only {self._current_step_reset} are being stepped."
            )
        results = []
        futures = self._step_and_maybe_reset_futures
        completed_futures = []
        for future in as_completed(futures):
            results.append(future.result())
            completed_futures.append(future)
            self._current_step_reset = self._current_step_reset - 1
            if len(results) >= min_get and sum([f.done() for f in futures]) == 0:
                break
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
        self._current_reset = self._current_reset + len(futures)
        self._reset_futures.extend(futures)

    def async_reset_recv(self, min_get: int | None = None) -> TensorDictBase:
        if min_get is None:
            min_get = self.min_get
        if min_get > self._current_reset:
            raise RuntimeError(
                f"Cannot await {min_get} reset when only {self._current_step_reset} are being reset."
            )
        results = []
        futures = self._reset_futures
        completed_futures = []
        for future in as_completed(futures):
            results.append(future.result())
            completed_futures.append(future)
            self._current_reset = self._current_reset - 1
            if len(results) >= min_get and sum([f.done() for f in futures]) == 0:
                break
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
