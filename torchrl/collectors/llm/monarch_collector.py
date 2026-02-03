# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any
from collections.abc import Callable

from tensordict import TensorDictBase
from torchrl.collectors.llm import LLMCollector
from torchrl.collectors.weight_update import WeightUpdaterBase
from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.envs import EnvBase

MONARCH_ERR = None
try:
    import monarch

    _has_monarch = True
except ImportError as err:
    _has_monarch = False
    MONARCH_ERR = err


class MonarchLLMCollector(LLMCollector):
    """A Monarch implementation of the LLM Collector that can be extended and sampled remotely.

    This collector uses Monarch's actor model and RDMA buffer support for efficient distributed data collection.

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
        monarch_init_config (dict[str, Any], optional): keyword arguments to pass to monarch.init().
        remote_config (dict[str, Any], optional): keyword arguments to pass to cls.as_remote().
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
        monarch_init_config: dict[str, Any] | None = None,
        remote_config: dict[str, Any] | None = None,
    ) -> None:
        if not _has_monarch:
            raise RuntimeError(
                "monarch library not found, unable to create a MonarchLLMCollector. "
            ) from MONARCH_ERR
        if not monarch.is_initialized():
            if monarch_init_config is None:
                monarch_init_config = {
                    "mesh_config": {
                        "num_processes": 1,
                        "num_gpus_per_process": 1,
                        "use_rdma": True,
                    }
                }
            monarch.init(**monarch_init_config)

        # Create a Monarch actor that wraps the LLMCollector
        remote_cls = LLMCollector.as_remote(remote_config).remote
        self._collector = monarch.create_actor(
            remote_cls,
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
        )

    def __iter__(self):
        """Returns an iterator over the collected data."""
        pending_task = self._collector.async_iter()
        return monarch.await_result(pending_task)

    def start(self):
        """Starts the collector in a background thread."""
        pending_task = self._collector.async_start()
        return monarch.await_result(pending_task)

    def shutdown(self, timeout=None, *, close_env=True):
        """Shuts down the collector.

        Args:
            timeout (float, optional): shutdown timeout in seconds
            close_env (bool): if True, closes the environment
        """
        pending_task = self._collector.async_shutdown(
            timeout=timeout, close_env=close_env
        )
        return monarch.await_result(pending_task)

    def async_shutdown(self, timeout=None, *, close_env=True):
        """Shuts down the collector asynchronously.

        Args:
            timeout (float, optional): shutdown timeout in seconds
            close_env (bool): if True, closes the environment
        """
        pending_task = self._collector.async_shutdown(
            timeout=timeout, close_env=close_env
        )
        return monarch.await_result(pending_task)

    def update_policy_weights_(self, policy_or_weights, worker_ids=None, **kwargs):
        """Updates the policy weights on remote workers using RDMA buffers.

        Args:
            policy_or_weights: The new policy weights to update.
            worker_ids (list[int], optional): List of worker IDs to update
        """
        if "policy_weights" in kwargs:
            warnings.warn(
                "`policy_weights` is deprecated. Use `policy_or_weights` instead.",
                DeprecationWarning,
            )
            policy_or_weights = kwargs.pop("policy_weights")

        # Create RDMA buffer for efficient weight transfer
        rdma_buffer = monarch.create_rdma_buffer(policy_or_weights)
        pending_task = self._collector.async_update_policy_weights(
            rdma_buffer, worker_ids=worker_ids
        )
        return monarch.await_result(pending_task)

    @property
    def total_dialog_turns(self):
        """Total number of dialog turns to collect."""
        return monarch.await_result(
            self._collector.async_get_attr("total_dialog_turns")
        )

    @property
    def dialog_turns_per_batch(self) -> int:
        """Number of dialog turns per batch."""
        return monarch.await_result(
            self._collector.async_get_attr("dialog_turns_per_batch")
        )

    @property
    def rollout(self) -> Callable[[], TensorDictBase]:
        """Returns the rollout function."""
        return monarch.await_result(self._collector.async_get_attr("rollout"))

    def init_updater(self, *args, **kwargs):
        """Initialize the weight updater with custom arguments.

        This method calls init_updater on the remote collector.

        Args:
            *args: Positional arguments for weight updater initialization
            **kwargs: Keyword arguments for weight updater initialization
        """
        pending_task = self._collector.async_init_updater(*args, **kwargs)
        return monarch.await_result(pending_task)

    @property
    def env(self) -> EnvBase:
        """Returns the environment instance."""
        return monarch.await_result(self._collector.async_get_attr("env"))

    @property
    def policy(self) -> Callable[[TensorDictBase], TensorDictBase]:
        """Returns the policy instance."""
        return monarch.await_result(self._collector.async_get_attr("policy"))

    @property
    def frames_per_batch(self) -> int:
        """Number of frames per batch."""
        return self.dialog_turns_per_batch

    @property
    def total_frames(self) -> int:
        """Total number of frames to collect."""
        return self.total_dialog_turns

    @property
    def reset_at_each_iter(self) -> bool:
        """Whether to reset at each iteration."""
        return monarch.await_result(
            self._collector.async_get_attr("reset_at_each_iter")
        )

    @property
    def flatten_data(self) -> bool:
        """Whether to flatten the data."""
        return monarch.await_result(self._collector.async_get_attr("flatten_data"))

    @property
    def yield_completed_trajectories(self) -> bool:
        """Whether to yield completed trajectories."""
        return monarch.await_result(
            self._collector.async_get_attr("yield_completed_trajectories")
        )

    @property
    def yield_only_last_steps(self) -> bool:
        """Whether to yield only last steps."""
        return monarch.await_result(
            self._collector.async_get_attr("yield_only_last_steps")
        )

    @property
    def async_envs(self) -> bool:
        """Whether to run environments asynchronously."""
        return monarch.await_result(self._collector.async_get_attr("async_envs"))

    @property
    def postproc(self):
        """Returns the post-processing transform."""
        return monarch.await_result(self._collector.async_get_attr("postproc"))

    @property
    def replay_buffer(self):
        """Returns the replay buffer instance."""
        return monarch.await_result(self._collector.async_get_attr("replay_buffer"))

    @property
    def weight_updater(self):
        """Returns the weight updater instance."""
        return monarch.await_result(self._collector.async_get_attr("weight_updater"))

    @property
    def device(self):
        """Returns the device of the collector."""
        return monarch.await_result(self._collector.async_get_attr("device"))

    @property
    def storing_device(self):
        """Returns the storing device of the collector."""
        return monarch.await_result(self._collector.async_get_attr("storing_device"))

    @property
    def policy_device(self):
        """Returns the policy device of the collector."""
        return monarch.await_result(self._collector.async_get_attr("policy_device"))

    @property
    def env_device(self):
        """Returns the environment device of the collector."""
        return monarch.await_result(self._collector.async_get_attr("env_device"))

    @property
    def max_frames_per_traj(self):
        """Returns the maximum frames per trajectory."""
        return monarch.await_result(
            self._collector.async_get_attr("max_frames_per_traj")
        )

    @property
    def init_random_frames(self):
        """Returns the number of initial random frames."""
        return monarch.await_result(
            self._collector.async_get_attr("init_random_frames")
        )

    @property
    def split_trajs(self):
        """Returns whether to split trajectories."""
        return monarch.await_result(self._collector.async_get_attr("split_trajs"))

    @property
    def exploration_type(self):
        """Returns the exploration type."""
        return monarch.await_result(self._collector.async_get_attr("exploration_type"))

    @property
    def return_same_td(self):
        """Returns whether to return the same tensordict."""
        return monarch.await_result(self._collector.async_get_attr("return_same_td"))

    @property
    def set_truncated(self):
        """Returns whether to set truncated flag."""
        return monarch.await_result(self._collector.async_get_attr("set_truncated"))

    @property
    def use_buffers(self):
        """Returns whether to use buffers."""
        return monarch.await_result(self._collector.async_get_attr("use_buffers"))

    @property
    def extend_buffer(self):
        """Returns whether to extend buffer."""
        return monarch.await_result(self._collector.async_get_attr("extend_buffer"))

    @property
    def trust_policy(self):
        """Returns whether to trust the policy."""
        return monarch.await_result(self._collector.async_get_attr("trust_policy"))

    @property
    def no_cuda_sync(self):
        """Returns whether to skip CUDA synchronization."""
        return monarch.await_result(self._collector.async_get_attr("no_cuda_sync"))

    @property
    def started(self):
        """Returns whether the collector has started."""
        return monarch.await_result(self._collector.async_get_attr("started"))

    @property
    def closed(self):
        """Returns whether the collector is closed."""
        return monarch.await_result(self._collector.async_get_attr("closed"))
