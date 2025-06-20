# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copy

import warnings
from typing import Any, Callable, Iterator

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl.collectors.llm import LLMCollector
from torchrl.collectors.weight_update import WeightUpdaterBase
from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.envs import EnvBase
from torchrl.envs.llm.transforms.policy_version import PolicyVersion

RAY_ERR = None
try:
    import ray

    _has_ray = True
except ImportError as err:
    _has_ray = False
    RAY_ERR = err


class RayLLMCollector(LLMCollector):
    """A lightweight Ray implementation of the LLM Collector that can be extended and sampled remotely.

    Args:
        env (EnvBase or EnvBase constructor): the environment to be used for data collection.

    Keyword Args:
        policy (Callable[[TensorDictBase], TensorDictBase]): the policy to be used for data collection.
        policy_factory (Callable[[], Callable], optional): a callable that returns
            a policy instance. This is exclusive with the `policy` argument.
        dialog_turns_per_batch (int): A keyword-only argument representing the total
            number of elements in a batch.
        total_dialog_turns (int): A keyword-only argument representing the total
            number of dialog turns returned by the collector during its lifespan.
        yield_only_last_steps (bool, optional): whether to yield every step of a trajectory, or only the
            last (done) steps.
        yield_completed_trajectories (bool, optional): whether to yield batches of rollouts with a given number of steps
            or single, completed trajectories.
        postproc (Callable, optional): A post-processing transform.
        async_envs (bool, optional): if True, the environment will be run asynchronously.
        replay_buffer (ReplayBuffer, optional): if provided, the collector will not yield tensordicts
            but populate the buffer instead.
        reset_at_each_iter (bool, optional): if True, the environment will be reset at each iteration.
        flatten_data (bool, optional): if True, the collector will flatten the collected data
            before returning it.
        weight_updater (WeightUpdaterBase or constructor, optional): An instance of WeightUpdaterBase
            or its subclass, responsible for updating the policy weights on remote inference workers.
        ray_init_config (dict[str, Any], optional): keyword arguments to pass to ray.init().
        remote_config (dict[str, Any], optional): keyword arguments to pass to cls.as_remote().
        sync_iter (bool, optional): if `True`, items yeilded by the collector will be synced to the local process.
            If `False`, the collector will collect the next batch of data in between yielding.
            This has no effect when data is collected through the :meth:`start` method.
            For example:

               >>> collector = RayLLMCollector(..., sync_iter=True)
               >>> for data in collector:  # blocking
               ...     # expensive operation - collector is idle
               >>> collector = RayLLMCollector(..., sync_iter=False)
               >>> for data in collector:  # non-blocking
               ...     # expensive operation - collector is collecting data

            This is somehwat equivalent to using :class:`~torchrl.collectors.MultiSyncDataCollector` (`sync_iter=True`) or
            :class:`~torchrl.collectors.MultiAsyncDataCollector` (`sync_iter=False`).
            Defaults to `True`.
        verbose (bool, optional): if ``True``, the collector will print progress information.
            Defaults to `False`.
    """

    def __init__(
        self,
        env: EnvBase | Callable[[], EnvBase],
        *,
        policy: Callable[[TensorDictBase], TensorDictBase] | None = None,
        policy_factory: Callable[[], Callable[[TensorDictBase], TensorDictBase]]
        | None = None,
        dialog_turns_per_batch: int,
        total_dialog_turns: int = -1,
        yield_only_last_steps: bool | None = None,
        yield_completed_trajectories: bool | None = None,
        postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
        async_envs: bool | None = None,
        replay_buffer: ReplayBuffer | None = None,
        reset_at_each_iter: bool = False,
        flatten_data: bool | None = None,
        weight_updater: WeightUpdaterBase
        | Callable[[], WeightUpdaterBase]
        | None = None,
        ray_init_config: dict[str, Any] | None = None,
        remote_config: dict[str, Any] | None = None,
        track_policy_version: bool | PolicyVersion = False,
        sync_iter: bool = True,
        verbose: bool = False,
    ) -> None:
        if not _has_ray:
            raise RuntimeError(
                "ray library not found, unable to create a RayLLMCollector. "
            ) from RAY_ERR
        if not ray.is_initialized():
            if ray_init_config is None:
                from torchrl.collectors.distributed.ray import DEFAULT_RAY_INIT_CONFIG

                ray_init_config = DEFAULT_RAY_INIT_CONFIG
            ray.init(**ray_init_config)
        if not sync_iter:
            remote_config = copy.copy(remote_config)
            remote_config.setdefault("max_concurrency", 2)
        remote_cls = LLMCollector.as_remote(remote_config).remote
        self.sync_iter = sync_iter
        self._collector = remote_cls(
            env=env,
            policy=policy,
            policy_factory=policy_factory,
            dialog_turns_per_batch=dialog_turns_per_batch,
            total_dialog_turns=total_dialog_turns,
            yield_only_last_steps=yield_only_last_steps,
            yield_completed_trajectories=yield_completed_trajectories,
            postproc=postproc,
            async_envs=async_envs,
            replay_buffer=replay_buffer,
            reset_at_each_iter=reset_at_each_iter,
            flatten_data=flatten_data,
            weight_updater=weight_updater,
            track_policy_version=track_policy_version,
            verbose=verbose,
        )

    def set_postproc(self, postproc: Callable[[TensorDictBase], TensorDictBase]):
        return ray.get(self._collector.set_postproc.remote(postproc))

    def _next_remote(self) -> None:
        return self._collector.next.remote()

    def next(self) -> None:
        """Get the next batch of data from the collector.

        Returns:
            None as the data is written directly to the replay buffer.
        """
        return ray.get(self._next_remote())

    def __iter__(self) -> Iterator[None]:
        """Returns an iterator that yields None as the collector writes directly to the replay buffer."""
        if not self.sync_iter:
            future = self._next_remote()
        else:
            future = None
        while True:
            try:
                if self.sync_iter:
                    yield self.next()
                else:
                    result = ray.get(future)
                    future = self._next_remote()
                    yield result
            except StopIteration:
                break

    def start(self):
        """Starts the collector in a background thread."""
        pending_task = self._collector.start.remote()
        return ray.get(pending_task)

    def shutdown(self):
        """Shuts down the collector."""
        pending_task = self._collector.shutdown.remote()
        return ray.get(pending_task)

    def async_shutdown(self, timeout=None):
        """Shuts down the collector asynchronously."""
        pending_task = self._collector.async_shutdown.remote(timeout=timeout)
        return ray.get(pending_task)

    def update_policy_weights_(
        self,
        policy_or_weights: TensorDictBase | TensorDictModuleBase | dict | None = None,
        *,
        worker_ids: torch.device | int | list[int] | list[torch.device] | None = None,
        **kwargs,
    ):
        """Updates the policy weights on remote workers.

        Args:
            policy_or_weights: The weights to update with. Can be:
                - TensorDictModuleBase: A policy module whose weights will be extracted
                - TensorDictBase: A TensorDict containing weights
                - dict: A regular dict containing weights
                - None: Will try to get weights from server using _get_server_weights()
            worker_ids: The workers to update. If None, updates all workers.
        """
        if "policy_weights" in kwargs:
            warnings.warn(
                "`policy_weights` is deprecated. Use `policy_or_weights` instead.",
                DeprecationWarning,
            )
            policy_or_weights = kwargs.pop("policy_weights")

        pending_task = self._collector.update_policy_weights_.remote(
            policy_or_weights=policy_or_weights, worker_ids=worker_ids
        )
        return ray.get(pending_task)

    @property
    def total_dialog_turns(self):
        """Total number of dialog turns to collect."""
        return ray.get(self._collector.total_dialog_turns.remote)

    @property
    def dialog_turns_per_batch(self) -> int:
        """Number of dialog turns per batch."""
        return ray.get(self._collector.dialog_turns_per_batch.remote)

    @property
    def rollout(self) -> Callable[[], TensorDictBase]:
        """Returns the rollout function."""
        return ray.get(self._collector.rollout.remote())

    def init_updater(self, *args, **kwargs):
        """Initialize the weight updater with custom arguments.

        This method calls init_updater on the remote collector.

        Args:
            *args: Positional arguments for weight updater initialization
            **kwargs: Keyword arguments for weight updater initialization
        """
        ray.get(self._collector.init_updater.remote(*args, **kwargs))

    @property
    def policy_version(self) -> str | int | None:
        """The current version of the policy.

        Returns:
            The current version number (int) or UUID (str), or None if version tracking is disabled.
        """
        return ray.get(self._collector.get_policy_version.remote())

    @property
    def weight_updater(self) -> WeightUpdaterBase:
        """The weight updater instance.

        We can pass the weight updater because it's stateless, hence serializable.
        """
        return ray.get(self._collector.weight_updater.remote)

    @weight_updater.setter
    def weight_updater(self, weight_updater: WeightUpdaterBase):
        """Set the weight updater instance."""
        ray.get(self._collector.set_weight_updater.remote(weight_updater))
        weight_updater.register_collector(self)

    def increment_version(self):
        """Increment the policy version."""
        return ray.get(self._collector.increment_version.remote())
