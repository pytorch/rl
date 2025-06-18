# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib
import importlib

from typing import Any, Callable, Iterator

import torch
from torchrl._utils import logger as torchrl_logger
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


class RayReplayBuffer(ReplayBuffer):
    """A Ray implementation of the Replay Buffer that can be extended and sampled remotely.

    Keyword Args:
        replay_buffer_cls (type[ReplayBuffer], optional): the class to use for the replay buffer.
            Defaults to :class:`~torchrl.data.ReplayBuffer`.
        ray_init_config (dict[str, Any], optiona): keyword arguments to pass to `ray.init()`.
        remote_config (dict[str, Any], optiona): keyword arguments to pass to `cls.as_remote()`.
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
        ...     await distributed_collector.async_shutdown()
        >>>
        >>> if __name__ == "__main__":
        ...     asyncio.run(main())

    """

    def __init__(
        self,
        *args,
        replay_buffer_cls: type[ReplayBuffer] | None = ReplayBuffer,
        ray_init_config: dict[str, Any] | None = None,
        remote_config: dict[str, Any] | None = None,
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

        remote_cls = replay_buffer_cls.as_remote(remote_config).remote
        # We can detect if the buffer has a GPU allocated, if not
        #  we'll make sure that the data is sent to CPU when needed.
        if remote_config is not None:
            self.has_gpu = remote_config.get("num_gpus", 0) > 0
        else:
            self.has_gpu = False
        self._rb = remote_cls(*args, **kwargs)

    def close(self):
        """Terminates the Ray actor associated with this replay buffer."""
        if hasattr(self, "_rb"):
            torchrl_logger.info("Killing Ray actor.")
            ray.kill(self._rb)  # Forcefully terminate the actor
            delattr(self, "_rb")  # Remove the reference to the terminated actor
            torchrl_logger.info("Ray actor killed.")

    @property
    def _replay_lock(self):
        """Placeholder for the replay lock.

        Replay-lock is not supported yet by RayReplayBuffer.
        """
        return contextlib.nullcontext()

    def sample(self, *args, **kwargs):
        pending_task = self._rb.sample.remote(*args, **kwargs)
        return ray.get(pending_task)

    def extend(self, *args, **kwargs):
        if not self.has_gpu:
            # Move the data to GPU
            args = [arg.to("cpu") for arg in args if hasattr(arg, "to")]
            kwargs = {k: v.to("cpu") for k, v in kwargs.items() if hasattr(v, "to")}
        pending_task = self._rb.extend.remote(*args, **kwargs)
        return ray.get(pending_task)

    def add(self, *args, **kwargs):
        return ray.get(self._rb.add.remote(*args, **kwargs))

    def update_priority(self, *args, **kwargs):
        return ray.get(self._rb.update_priority.remote(*args, **kwargs))

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

    def empty(self):
        return ray.get(self._rb.empty.remote())

    def __getitem__(self, index):
        return ray.get(self._rb.__getitem__.remote(index))

    def next(self):
        return ray.get(self._rb.next.remote())

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
        return ray.get(self._rb.__len__.remote())

    @property
    def write_count(self):
        return ray.get(self._rb._getattr.remote("write_count"))

    @property
    def dim_extend(self):
        return ray.get(self._rb._getattr.remote("dim_extend"))

    @dim_extend.setter
    def dim_extend(self, value):
        return ray.get(self._rb._setattr.remote("dim_extend", value))

    def __setitem__(self, index, value) -> None:
        return ray.get(self._rb.__setitem__.remote(index, value))

    def __repr__(self) -> str:
        rb_repr = ray.get(self._rb.__repr__.remote())
        return f"RayReplayBuffer(\n    remote_buffer={rb_repr}\n)"

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        return ray.get(self._rb.load_state_dict.remote(state_dict))
