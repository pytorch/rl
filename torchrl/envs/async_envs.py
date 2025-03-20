# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import multiprocessing
from functools import partial

# import queue
from multiprocessing import Queue
from queue import Empty
from typing import Callable, Literal, Sequence

import torch
from tensordict import lazy_stack, maybe_dense_stack, TensorDict, TensorDictBase

from tensordict.tensorclass import NonTensorData

from torchrl.envs import EnvBase, GymEnv


class AsyncEnv(EnvBase):
    _env_idx_key = "_env_idx"

    def __init__(
        self,
        env_maker: Callable[[], EnvBase]
        | EnvBase
        | list[EnvBase]
        | list[Callable[[], EnvBase]],
        backend: Literal[
            "multithreading", "multiprocessing", "asyncio"
        ] = "multithreading",
        stack: Literal["dense", "maybe_dense", "lazy"] = "dense",
    ) -> None:
        if not isinstance(env_maker, Sequence):
            env_maker = [env_maker]
        self.min_get = 1
        self.env_maker = env_maker
        self.num_envs = len(env_maker)
        self.backend = backend
        if backend != "multithreading":
            raise NotImplementedError
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
                target=_env_exec,
                kwargs={
                    "i": i,
                    "env_or_factory": self.env_maker[i],
                    "input_queue": self.input_queue[i],
                    "step_reset_queue": self.step_reset_queue,
                    "step_queue": self.step_queue,
                    "reset_queue": self.reset_queue,
                },
            )
            self.threads.append(thread)
            thread.start()

        if stack == "dense":
            self._stack_func = torch.stack
        elif stack == "maybe_dense":
            self._stack_func = maybe_dense_stack
        elif stack == "lazy":
            self._stack_func = lazy_stack
        else:
            raise NotImplementedError

    def step(self, tensordict):
        raise NotImplementedError

    def _step(self, tensordict):
        raise NotImplementedError

    def _reset(self, tensordict):
        raise NotImplementedError

    def _set_seed(self, seed: int | None):
        raise NotImplementedError

    def reset(
        self,
        tensordict: TensorDictBase | None = None,
        **kwargs,
    ) -> TensorDictBase:
        raise NotImplementedError

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

        print("no wait")
        try:
            while True:
                item = q.get_nowait()
                items.append(item)
        except Empty:
            pass
        print("get", items)
        # Retrieve all other available items
        while len(items) < min_get:
            item = q.get()
            items.append(item)

        return items

    def __del__(self):
        self._maybe_shutdown()

    def _maybe_shutdown(self):
        try:
            self.shutdown()
        except Exception:
            pass

    def shutdown(self):
        for env_id in range(self.num_envs):
            self.input_queue[env_id].put(("shutdown", None))

        for thread in self.threads:
            thread.join()

    def close(self, *, raise_if_closed: bool = True):
        if raise_if_closed:
            self.shutdown()
        else:
            self._maybe_shutdown()


def _env_exec(
    i, env_or_factory, input_queue, step_queue, step_reset_queue, reset_queue
):
    if not isinstance(env_or_factory, EnvBase):
        env = env_or_factory()
    else:
        env = env_or_factory
    print("env", env)
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
            return
        else:
            raise RuntimeError


if __name__ == "__main__":
    envs = AsyncEnv(
        [partial(GymEnv, "Pendulum-v1"), partial(GymEnv, "Pendulum-v1")], stack="lazy"
    )
    print("send")
    envs.async_reset_send(TensorDict(_env_idx=range(2), batch_size=2))
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
