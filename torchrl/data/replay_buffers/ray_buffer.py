# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any, Callable

import torch

from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.envs.transforms.transforms import Transform

RAY_ERR = None
try:
    import ray

    _has_ray = True
except ImportError as err:
    _has_ray = False
    RAY_ERR = err


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
        ray_init_config (dict[str, Any], optiona): keyword arguments to pass to `ray.init()`.
        remote_config (dict[str, Any], optiona): keyword arguments to pass to `cls.as_remote()`.
            Defaults to `torchrl.collectors.distributed.ray.DEFAULT_REMOTE_CLASS_CONFIG`.

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

        remote_cls = ReplayBuffer.as_remote(remote_config).remote
        self._rb = remote_cls(*args, **kwargs)

    def sample(self, *args, **kwargs):
        pending_task = self._rb.sample.remote(*args, **kwargs)
        return ray.get(pending_task)

    def extend(self, *args, **kwargs):
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
