# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import importlib
from collections.abc import Callable, Iterator

from typing import Any

import torch
from torchrl._utils import logger as torchrl_logger
from torchrl.data.replay_buffers.distributed import DataParallelReplayBufferClient
from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.envs.transforms.transforms import Transform

RAY_ERR = None
_has_ray = importlib.util.find_spec("ray") is not None
if _has_ray:
    import ray
else:

    def ray():  # noqa: D103
        raise ImportError(
            "ray is not installed. Please install it with `pip install ray`."
        )


@classmethod
def as_remote(cls, remote_config=None):
    """Creates an instance of a remote ray class.

    Args:
        cls (Python Class): class to be remotely instantiated.
        remote_config (dict): the quantity of CPU cores to reserve for this class.
            Defaults to `torchrl.collectors.distributed.ray.DEFAULT_REMOTE_CLASS_CONFIG`.

    Returns:
        A function that creates ray remote class instances.
    """
    if remote_config is None:
        from torchrl.collectors.distributed.ray import DEFAULT_REMOTE_CLASS_CONFIG

        remote_config = DEFAULT_REMOTE_CLASS_CONFIG
    remote_collector = ray.remote(**remote_config)(cls)
    remote_collector.is_remote = True
    return remote_collector


ReplayBuffer.as_remote = as_remote


class _RayReplayBufferClient:
    """Picklable Ray replay-buffer client without lifecycle capabilities."""

    def __init__(self, actor, *, has_gpu: bool) -> None:
        self._actor = actor
        self.has_gpu = has_gpu

    @property
    def _replay_lock(self):
        return contextlib.nullcontext()

    @property
    def batch_size(self):
        return ray.get(self._actor._getattr.remote("_batch_size"))

    @property
    def write_count(self):
        return ray.get(self._actor._getattr.remote("write_count"))

    @property
    def dim_extend(self):
        return ray.get(self._actor._getattr.remote("dim_extend"))

    @dim_extend.setter
    def dim_extend(self, value):
        ray.get(self._actor._setattr.remote("dim_extend", value))

    def sample(self, *args, **kwargs):
        return ray.get(self._actor.sample.remote(*args, **kwargs))

    def _sample_data_parallel(self, *args, **kwargs):
        return ray.get(self._actor._sample_data_parallel.remote(*args, **kwargs))

    def data_parallel(
        self, *, rank: int, world_size: int
    ) -> DataParallelReplayBufferClient:
        """Create a lifecycle-free, rank-aware view of this client."""
        return DataParallelReplayBufferClient(self, rank=rank, world_size=world_size)

    def extend(self, *args, **kwargs):
        if not self.has_gpu:
            args = tuple(_to_cpu(arg) for arg in args)
            kwargs = {key: _to_cpu(value) for key, value in kwargs.items()}
        return ray.get(self._actor.extend.remote(*args, **kwargs))

    def add(self, *args, **kwargs):
        if not self.has_gpu:
            args = tuple(_to_cpu(arg) for arg in args)
            kwargs = {key: _to_cpu(value) for key, value in kwargs.items()}
        return ray.get(self._actor.add.remote(*args, **kwargs))

    def update_priority(self, *args, **kwargs):
        if not self.has_gpu:
            args = tuple(_to_cpu(arg) for arg in args)
            kwargs = {key: _to_cpu(value) for key, value in kwargs.items()}
        return ray.get(self._actor.update_priority.remote(*args, **kwargs))

    def __len__(self):
        return ray.get(self._actor.__len__.remote())

    def __getitem__(self, index):
        if not self.has_gpu:
            index = _to_cpu(index)
        return ray.get(self._actor.__getitem__.remote(index))

    def __setitem__(self, index, value) -> None:
        if not self.has_gpu:
            index = _to_cpu(index)
            value = _to_cpu(value)
        ray.get(self._actor.__setitem__.remote(index, value))

    def next(self):
        return ray.get(self._actor.next.remote())

    def __iter__(self) -> Iterator[Any]:
        while True:
            data = self.next()
            if data is None:
                return
            yield data

    def __getattr__(self, name: str):
        if name.startswith("_") or name in {"client", "close", "shutdown", "start"}:
            raise AttributeError(
                f"{type(self).__name__} has no lifecycle capability {name!r}."
            )

        def remote_call(*args, **kwargs):
            return ray.get(getattr(self._actor, name).remote(*args, **kwargs))

        return remote_call


def _to_cpu(value: Any) -> Any:
    if hasattr(value, "to"):
        return value.to("cpu")
    return value


class RayReplayBuffer(ReplayBuffer):
    """A Ray implementation of the Replay Buffer that can be extended and sampled remotely.

    Keyword Args:
        replay_buffer_cls (type[ReplayBuffer], optional): the class to use for the replay buffer.
            Defaults to :class:`~torchrl.data.ReplayBuffer`.
        ray_init_config (dict[str, Any], optional): keyword arguments to pass to `ray.init()`.
        remote_config (dict[str, Any], optional): keyword arguments to pass to `cls.as_remote()`.
            Defaults to `torchrl.collectors.distributed.ray.DEFAULT_REMOTE_CLASS_CONFIG`.
        **kwargs: keyword arguments to pass to the replay buffer class.

    .. seealso:: :class:`~torchrl.data.ReplayBuffer` for a list of other keyword arguments.

    The writer, sampler and storage should be passed as constructors to prevent serialization issues.
    Transforms constructors should be passed through the `transform_factory` argument.

    Example:
        >>> import asyncio
        >>> from tensordict.nn import TensorDictModule
        >>> from torch import nn
        >>> from torchrl.collectors.distributed.ray import RayCollector
        >>> from torchrl.data.replay_buffers.ray_buffer import RayReplayBuffer
        >>> from torchrl.envs.libs.gym import GymEnv
        >>>
        >>> async def main():
        ...     # 1. Create environment factory
        ...     def env_maker():
        ...         return GymEnv("Pendulum-v1", device="cpu")
        ...
        ...     policy = TensorDictModule(
        ...         nn.Linear(3, 1), in_keys=["observation"], out_keys=["action"]
        ...     )
        ...
        ...     buffer = RayReplayBuffer()
        ...
        ...     # 2. Define distributed collector
        ...     remote_config = {
        ...         "num_cpus": 1,
        ...         "num_gpus": 0,
        ...         "memory": 5 * 1024**3,
        ...         "object_store_memory": 2 * 1024**3,
        ...     }
        ...     distributed_collector = RayCollector(
        ...         [env_maker],
        ...         policy,
        ...         total_frames=600,
        ...         frames_per_batch=200,
        ...         remote_configs=remote_config,
        ...         replay_buffer=buffer,
        ...     )
        ...
        ...     print("start")
        ...     distributed_collector.start()
        ...
        ...     while True:
        ...         while not len(buffer):
        ...             print("waiting")
        ...             await asyncio.sleep(1)  # Use asyncio.sleep instead of time.sleep
        ...         print("sample", buffer.sample(32))
        ...         # break at some point
        ...         break
        ...
        ...     await distributed_collector.async_shutdown(shutdown_ray=False)
        ...     buffer.close()  # Close buffer after collector
        >>>
        >>> if __name__ == "__main__":
        ...     asyncio.run(main())

    """

    def __init__(
        self,
        *args,
        replay_buffer_cls: type[ReplayBuffer] = ReplayBuffer,
        ray_init_config: dict[str, Any] | None = None,
        remote_config: dict[str, Any] | None = None,
        delayed_init: bool = False,
        **kwargs,
    ) -> None:
        if not _has_ray:
            raise RuntimeError(
                "ray library not found, unable to create a RayReplayBuffer. "
            ) from RAY_ERR
        if not ray.is_initialized():
            if ray_init_config is None:
                from torchrl.collectors.distributed.ray import DEFAULT_RAY_INIT_CONFIG

                ray_init_config = DEFAULT_RAY_INIT_CONFIG
            ray.init(**ray_init_config)

        self._service_cls = replay_buffer_cls
        remote_cls = replay_buffer_cls.as_remote(remote_config).remote
        # We can detect if the buffer has a GPU allocated, if not
        #  we'll make sure that the data is sent to CPU when needed.
        if remote_config is not None:
            self.has_gpu = remote_config.get("num_gpus", 0) > 0
        else:
            self.has_gpu = False
        self._rb = remote_cls(*args, delayed_init=delayed_init, **kwargs)
        self._delayed_init = False
        self._client = _RayReplayBufferClient(self._rb, has_gpu=self.has_gpu)

    def start(self) -> RayReplayBuffer:
        """Return this already-started Ray replay-buffer owner."""
        if not self.is_alive:
            raise RuntimeError("A closed RayReplayBuffer cannot be restarted.")
        return self

    @property
    def is_alive(self) -> bool:
        """Whether the owned Ray replay-buffer actor is available."""
        return hasattr(self, "_rb")

    @property
    def service_backend(self) -> str:
        """The canonical deployment backend for this replay buffer."""
        return "ray"

    def client(self) -> _RayReplayBufferClient:
        """Return a picklable client without actor shutdown rights."""
        if not self.is_alive:
            raise RuntimeError("RayReplayBuffer is closed.")
        return _RayReplayBufferClient(self._rb, has_gpu=self.has_gpu)

    def data_parallel(
        self, *, rank: int, world_size: int
    ) -> DataParallelReplayBufferClient:
        """Return a rank-aware view of a new lifecycle-free client."""
        return self.client().data_parallel(rank=rank, world_size=world_size)

    def shutdown(self, timeout: float | None = None) -> None:
        """Terminate the owned Ray actor."""
        del timeout
        self.close()

    def close(self) -> None:
        """Terminates the Ray actor associated with this replay buffer."""
        if hasattr(self, "_rb"):
            try:
                torchrl_logger.info("Killing Ray actor.")
                ray.kill(self._rb, no_restart=True)  # Forcefully terminate the actor
                torchrl_logger.info("Ray actor killed.")
            except (ValueError, RuntimeError) as e:
                # Actor may already be dead if ray.shutdown() was called
                torchrl_logger.debug(
                    f"Failed to kill Ray actor (may already be terminated): {e}"
                )
            finally:
                delattr(self, "_rb")  # Remove the reference to the terminated actor

    @property
    def _replay_lock(self):
        """Placeholder for the replay lock.

        Replay-lock is not supported yet by RayReplayBuffer.
        """
        return contextlib.nullcontext()

    @property
    def batch_size(self):
        return self._client.batch_size

    def sample(self, *args, **kwargs):
        return self._client.sample(*args, **kwargs)

    def extend(self, *args, **kwargs):
        return self._client.extend(*args, **kwargs)

    def add(self, *args, **kwargs):
        return self._client.add(*args, **kwargs)

    def update_priority(self, *args, **kwargs):
        return self._client.update_priority(*args, **kwargs)

    def append_transform(self, *args, **kwargs):
        return ray.get(self._rb.append_transform.remote(*args, **kwargs))

    def dumps(self, path):
        return ray.get(self._rb.dumps.remote(path))

    def dump(self, path):
        return ray.get(self._rb.dump.remote(path))

    def loads(self, path):
        return ray.get(self._rb.loads.remote(path))

    def load(self, *args, **kwargs):
        return ray.get(self._rb.load.remote(*args, **kwargs))

    def empty(self, empty_write_count: bool = True):
        return ray.get(self._rb.empty.remote(empty_write_count=empty_write_count))

    def __getitem__(self, index):
        return self._client[index]

    def next(self):
        return self._client.next()

    def __iter__(self) -> Iterator[Any]:
        """Returns an iterator that yields None as the collector writes directly to the replay buffer."""
        while True:
            data = self.next()
            if data is not None:
                yield data
            else:
                break

    def insert_transform(
        self,
        index: int,
        transform: Transform,  # noqa-F821
        *,
        invert: bool = False,
    ) -> ReplayBuffer:
        return ray.get(
            self._rb.insert_transform.remote(index, transform, invert=invert)
        )

    def mark_update(self, index: int | torch.Tensor) -> None:
        return ray.get(self._rb.mark_update.remote(index))

    def register_load_hook(self, hook: Callable[[Any], Any]):
        return ray.get(self._rb.register_load_hook.remote(hook))

    def register_save_hook(self, hook: Callable[[Any], Any]):
        return ray.get(self._rb.register_save_hook.remote(hook))

    def save(self, path: str):
        return ray.get(self._rb.save.remote(path))

    def set_rng(self, generator):
        return ray.get(self._rb.set_rng.remote(generator))

    def set_sampler(self, sampler):
        return ray.get(self._rb.set_sampler.remote(sampler))

    def set_storage(self, storage):
        return ray.get(self._rb.set_storage.remote(storage))

    def set_writer(self, writer):
        return ray.get(self._rb.set_writer.remote(writer))

    def share(self, shared: bool = True):
        return ray.get(self._rb.share.remote(shared))

    def state_dict(self):
        return ray.get(self._rb.state_dict.remote())

    def __len__(self):
        return len(self._client)

    @property
    def write_count(self):
        return self._client.write_count

    @property
    def dim_extend(self):
        return self._client.dim_extend

    @dim_extend.setter
    def dim_extend(self, value):
        self._client.dim_extend = value

    def __setitem__(self, index, value) -> None:
        self._client[index] = value

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return ray.get(self._rb.load_state_dict.remote(state_dict))
