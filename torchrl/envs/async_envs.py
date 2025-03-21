# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import multiprocessing
import threading
from functools import partial

# import queue
from multiprocessing import Queue
from queue import Empty
from typing import Callable, Literal, Sequence

import torch
from tensordict import lazy_stack, maybe_dense_stack, TensorDict, TensorDictBase

from tensordict.tensorclass import NonTensorData

from torchrl.envs.libs import GymEnv
from torchrl.envs.common import _EnvPostInit, EnvBase
import queue
from concurrent.futures import ThreadPoolExecutor, as_completed
import abc

class _AsyncEnvMeta(_EnvPostInit):
    def __call__(cls, *args, **kwargs):
        backend = kwargs.get("backend", "threading")
        if cls is AsyncEnvPool:
            if backend == "threading":
                instance: MTAsyncEnvPool = MTAsyncEnvPool(*args, **kwargs)
            elif backend == "multiprocessing":
                instance: MPAsyncEnvPool = MPAsyncEnvPool(*args, **kwargs)
            elif backend == "asyncio":
                instance: MPAsyncEnvPool = MPAsyncEnvPool(*args, **kwargs)
            else:
                raise NotImplementedError
            return instance
        else:
            return super().__call__(*args, **kwargs)

class AsyncEnvPool(EnvBase, metaclass=_AsyncEnvMeta):
    _env_idx_key = "_env_idx"

    def __init__(
        self,
        env_makers: Callable[[], EnvBase]
                    | EnvBase
                    | list[EnvBase]
                    | list[Callable[[], EnvBase]],
        *,
        backend: Literal[
            "threading", "multiprocessing", "asyncio"
        ] = "threading",
        stack: Literal["dense", "maybe_dense", "lazy"] = "dense",
    ) -> None:
        if not isinstance(env_makers, Sequence):
            env_makers = [env_makers]
        self.min_get = 1
        self.env_makers = env_makers
        self.num_envs = len(env_makers)
        self.backend = backend

        if stack == "dense":
            self._stack_func = torch.stack
        elif stack == "maybe_dense":
            self._stack_func = maybe_dense_stack
        elif stack == "lazy":
            self._stack_func = lazy_stack
        else:
            raise NotImplementedError

        self._setup()

    def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        raise NotImplementedError
    def _step(
        self,
        tensordict: TensorDictBase,
    ) -> TensorDictBase:
        raise NotImplementedError
    def _set_seed(self, seed: int | None):
        raise NotImplementedError

    @abc.abstractmethod
    def _setup(self) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def async_step_send(self, tensordict: TensorDictBase) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def async_step_recv(self, min_get: int | None = None) -> TensorDictBase:
        raise NotImplementedError

    @abc.abstractmethod
    def async_step_and_maybe_reset_send(self, tensordict: TensorDictBase) -> None:
        raise NotImplementedError

    @abc.abstractmethod
    def async_step_and_maybe_reset_recv(self, min_get: int | None = None) -> TensorDictBase:
        raise NotImplementedError

    @abc.abstractmethod
    def async_reset_send(self, tensordict: TensorDictBase) -> None:
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


class MPAsyncEnvPool(AsyncEnvPool):
    def _setup(self) -> None:
        self.step_queue = Queue(maxsize=self.num_envs)
        self.reset_queue = Queue(maxsize=self.num_envs)
        self.step_reset_queue = Queue(maxsize=self.num_envs)
        self.input_queue = [Queue() for _ in range(self.num_envs)]
        self._current_reset = 0
        self._current_step = 0
        self._current_step_reset = 0

        num_threads = self.num_envs
        print("num_threads", num_threads)
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
                    "step_reset_queue": self.step_reset_queue,
                    "step_queue": self.step_queue,
                    "reset_queue": self.reset_queue,
                },
            )
            self.threads.append(thread)
            thread.start()


    def async_step_send(self, tensordict: TensorDictBase) -> None:
        # puts tds in a queue and ask for env.step
        # puts tds in a queue and ask for env.reset
        env_idx = tensordict[self._env_idx_key]
        if isinstance(env_idx, int):
            env_idx = [env_idx]
        local_tds = tensordict.unbind(0)
        for _env_idx, local_td in zip(env_idx, local_tds):
            self._current_step += 1
            self.input_queue[_env_idx].put(("step", local_td))

    def async_step_recv(self, min_get: int = 1) -> TensorDictBase:
        # gets step results from the queue
        if min_get is None:
            min_get = self.min_get
        if min_get < self._current_step:
            raise RuntimeError(
                f"Cannot await {min_get} step when only {self._current_step} are being stepped."
            )
        r = self._wait_for_one_and_get(self.step_queue, min_get)
        return self._stack_func(r)

    def async_step_and_maybe_reset_send(self, tensordict: TensorDictBase) -> None:
        # puts tds in a queue and ask for env.step
        # puts tds in a queue and ask for env.reset
        env_idx = tensordict[self._env_idx_key]
        if isinstance(env_idx, int):
            env_idx = [env_idx]
        local_tds = tensordict.unbind(0)
        for _env_idx, local_td in zip(env_idx, local_tds):
            self._current_step_reset += 1
            self.input_queue[_env_idx].put(("step_and_maybe_reset", local_td))

    def async_step_and_maybe_reset_recv(self, min_get: int = 1) -> TensorDictBase:
        # gets step results from the queue
        if min_get is None:
            min_get = self.min_get
        if min_get < self._current_step_reset:
            raise RuntimeError(
                f"Cannot await {min_get} step_and_maybe_reset when only {self._current_step_reset} are being stepped."
            )
        r = self._wait_for_one_and_get(self.step_reset_queue, min_get)
        return self._stack_func(r)

    def async_reset_send(self, tensordict: TensorDictBase) -> None:
        # puts tds in a queue and ask for env.reset
        env_idx = tensordict[self._env_idx_key]
        if isinstance(env_idx, int):
            env_idx = [env_idx]
        local_tds = tensordict.unbind(0)
        for _env_idx, local_td in zip(env_idx, local_tds):
            self._current_reset += 1
            self.input_queue[_env_idx].put(("reset", local_td))

    def async_reset_recv(self, min_get: int | None = None) -> TensorDictBase:
        # gets reset results from the queue
        if min_get is None:
            min_get = self.min_get
        if min_get < self._current_reset:
            raise RuntimeError(
                f"Cannot await {min_get} reset when only {self._current_reset} are being reset."
            )
        r = self._wait_for_one_and_get(self.reset_queue, min_get)
        return self._stack_func(r)

    def _wait_for_one_and_get(self, q, min_get):
        items = [q.get()]
        q.task_done()

        try:
            while True:
                item = q.get_nowait()
                items.append(item)
                q.task_done()
        except Empty:
            pass

        # Retrieve all other available items
        while len(items) < min_get:
            item = q.get()
            items.append(item)
            q.task_done()

        return items

    @abc.abstractmethod
    def shutdown(self):
        for env_id in range(self.num_envs):
            self.input_queue[env_id].put(("shutdown", None))

        for thread in self.threads:
            thread.join()

    @classmethod
    def _env_exec(
            cls, i, env_or_factory, input_queue, step_queue, step_reset_queue, reset_queue
    ):
        if not isinstance(env_or_factory, EnvBase):
            env = env_or_factory()
        else:
            env = env_or_factory

        while True:
            print("getting msg")
            msg, data = input_queue.get()
            print("msg", msg)
            if msg == "reset":
                data = env.reset(data.copy())
                data.set("_env_idx", NonTensorData(i))
                reset_queue.put(data)
            elif msg == "step_and_maybe_reset":
                data, data_ = env.step_and_maybe_reset(data.copy())
                data.set("_env_idx", NonTensorData(i))
                data_.set("_env_idx", NonTensorData(i))
                step_reset_queue.put((data, data_))
            elif msg == "step":
                data = env.step(data.copy())
                data.set("_env_idx", NonTensorData(i))
                step_queue.put(data)
            elif msg == "shutdown":
                env.close()
                break
            else:
                raise RuntimeError
            q.task_done()
        return


class MTAsyncEnvPool(AsyncEnvPool):
    def _setup(self) -> None:
        self._pool = ThreadPoolExecutor(max_workers=self.num_envs)
        self.envs = [
            env_factory() if not isinstance(env_factory, EnvBase) else env_factory for env_factory in self.env_makers
        ]
        self._reset_futures = []
        self._step_futures = []
        self._step_and_maybe_reset_futures = []

    @classmethod
    def _step_func(cls, env_td: Tuple[EnvBase, TensorDictBase]):
        env, td = env_td
        return env.step(td)
    @classmethod
    def _reset_func(cls, env_td: Tuple[EnvBase, TensorDictBase]):
        env, td = env_td
        return env.reset(td)
    @classmethod
    def _step_and_maybe_reset_func(cls, env_td: Tuple[EnvBase, TensorDictBase]):
        env, td = env_td
        return env.step_and_maybe_reset(td)

    def async_step_send(self, tensordict: TensorDictBase) -> None:
        idx = tensordict[self._env_idx_key]
        if isinstance(idx, int):
            idx = [idx]
        tds = tensordict.unbind(0)
        envs = [self.envs[idx] for idx in idx]
        futures = [self._pool.submit(self._step_func, (env, td)) for env, td in zip(envs, tds)]
        self._step_futures.extend(futures)

    def async_step_recv(self, min_get: int | None = None) -> TensorDictBase:
        if min_get is None:
            min_get = self.min_get
        results = []
        futures = self._step_futures
        completed_futures = []
        for future in as_completed(futures):
            results.append(future.result())
            completed_futures.append(future)
            if len(results) >= min_get and sum([f.done() for f in futures]) == 0:
                break
        self._step_futures = [f for f in self._step_futures if f not in completed_futures]
        return self._stack_func(results)

    def async_step_and_maybe_reset_send(self, tensordict: TensorDictBase) -> None:
        idx = tensordict[self._env_idx_key]
        if isinstance(idx, int):
            idx = [idx]
        tds = tensordict.unbind(0)
        envs = [self.envs[idx] for idx in idx]
        futures = [self._pool.submit(self._step_and_maybe_reset_func, (env, td)) for env, td in zip(envs, tds)]
        self._step_and_maybe_reset_futures.extend(futures)

    def async_step_and_maybe_reset_recv(self, min_get: int | None = None) -> TensorDictBase:
        if min_get is None:
            min_get = self.min_get
        results = []
        futures = self._step_and_maybe_reset_futures
        completed_futures = []
        for future in as_completed(futures):
            results.append(future.result())
            completed_futures.append(future)
            if len(results) >= min_get and sum([f.done() for f in futures]) == 0:
                break
        self._step_and_maybe_reset_futures = [f for f in self._step_and_maybe_reset_futures if f not in completed_futures]
        return self._stack_func(results)

    def async_reset_send(self, tensordict: TensorDictBase) -> None:
        idx = tensordict[self._env_idx_key]
        if isinstance(idx, int):
            idx = [idx]
        tds = tensordict.unbind(0)
        envs = [self.envs[idx] for idx in idx]
        futures = [self._pool.submit(self._reset_func, (env, td)) for env, td in zip(envs, tds)]
        self._reset_futures.extend(futures)

    def async_reset_recv(self, min_get: int | None = None) -> TensorDictBase:
        if min_get is None:
            min_get = self.min_get
        results = []
        futures = self._reset_futures
        completed_futures = []
        for future in as_completed(futures):
            results.append(future.result())
            completed_futures.append(future)
            if len(results) >= min_get and sum([f.done() for f in futures]) == 0:
                break
        self._reset_futures = [f for f in self._reset_futures if f not in completed_futures]
        print('results', results)
        return self._stack_func(results)

    def shutdown(self):
        self._pool.shutdown()

if __name__ == "__main__":
    torch.set_num_threads(10)
    envs = AsyncEnvPool(
        [partial(GymEnv, "Pendulum-v1"), partial(GymEnv, "Pendulum-v1")], stack="lazy", backend="threading",
    )
    print("send")
    envs.async_reset_send(TensorDict(_env_idx=torch.arange(2), batch_size=2))
    print("recv")
    r = envs.async_reset_recv(min_get=2)
    print(r)
    r["action"] = torch.randn(r.shape + (1,)).clamp(-1, 1)
    print("send", r)
    envs.async_step_send(r)
    print("recv")
    s = envs.async_step_recv(min_get=2)
    print(s)
    envs.shutdown()
