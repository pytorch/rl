# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from torchrl.data.replay_buffers import ReplayBuffer

RAY_ERR = None
try:
    import ray

    _has_ray = True
except ImportError as err:
    _has_ray = False
    RAY_ERR = err

DEFAULT_REMOTE_CLASS_CONFIG = {
    "num_cpus": 1,
    "num_gpus": 0.0,
    "memory": 2 * 1024**3,
}


@classmethod
def as_remote(cls, remote_config=DEFAULT_REMOTE_CLASS_CONFIG):
    """Creates an instance of a remote ray class.

    Args:
        cls (Python Class): class to be remotely instantiated.
        remote_config (dict): the quantity of CPU cores to reserve for this class.

    Returns:
        A function that creates ray remote class instances.
    """
    remote_collector = ray.remote(**remote_config)(cls)
    remote_collector.is_remote = True
    return remote_collector


ReplayBuffer.as_remote = as_remote

remote_cls = ReplayBuffer.as_remote(DEFAULT_REMOTE_CLASS_CONFIG).remote


class RayReplayBuffer(ReplayBuffer):
    """A Ray implementation of the Replay Buffer that can be extended and sampled remotely.

    .. seealso:: :class:`~torchrl.data.ReplayBuffer` for a list of keyword arguments.

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

    def __init__(self, *args, ray_init_kwargs=None, **kwargs) -> None:
        if not _has_ray:
            raise RuntimeError(
                "ray library not found, unable to create a RayReplayBuffer. "
            ) from RAY_ERR
        if not ray.is_initialized():
            if ray_init_kwargs is None:
                from torchrl.collectors.distributed.ray import DEFAULT_RAY_INIT_CONFIG

                ray_init_kwargs = DEFAULT_RAY_INIT_CONFIG
            ray.init(**ray_init_kwargs)
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

    def __len__(self):
        return ray.get(self._rb.__len__.remote())
