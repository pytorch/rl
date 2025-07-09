# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections import deque
from typing import Any, Callable

import torch

from tensordict import lazy_stack, TensorDictBase

from torchrl._utils import as_remote, logger as torchrl_logger

from torchrl.collectors import SyncDataCollector
from torchrl.collectors.llm.utils import _QueueAsRB
from torchrl.collectors.weight_update import WeightUpdaterBase
from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.envs import AsyncEnvPool
from torchrl.envs.common import EnvBase
from torchrl.envs.llm.transforms.policy_version import PolicyVersion


class LLMCollector(SyncDataCollector):
    """A simplified version of SyncDataCollector for LLM inference.

    Args:
        env (EnvBase or EnvBase constructor): the environment to be used for data collection.

    Keyword Args:
        policy (Callable[[TensorDictBase], TensorDictBase]): the policy to be used for data collection.
        policy_factory (Callable[[], Callable], optional): a callable that returns
            a policy instance. This is exclusive with the `policy` argument.

            .. note:: `policy_factory` comes in handy whenever the policy cannot be serialized.

        dialog_turns_per_batch (int, optional): A keyword-only argument representing the total
            number of elements in a batch. It is always required except when `yield_completed_trajectories=True`.
        total_dialog_turns (int): A keyword-only argument representing the total
            number of steps returned by the collector during its lifespan. -1 is never ending (until shutdown).
            Defaults to -1.
        yield_completed_trajectories (bool, optional): whether to yield batches of rollouts with a given number of steps
            (`yield_completed_trajectories=False`, default) or single, completed trajectories
            (`yield_completed_trajectories=True`).
            Defaults to `False` unless `yield_only_last_steps=True`, where it cannot be `False`.

            .. warning:: If the `done` state of the environment is not properly set, this may lead to a collector
                that never leads any data.

        yield_only_last_steps (bool, optional): whether to yield every step of a trajectory, or only the
            last (done) steps.
            If `True`, a single trajectory is yielded (or written in the buffer) at a time.

            .. warning:: If the `done` state of the environment is not properly set, this may lead to a collector
                that never leads any data.

        postproc (Callable, optional): A post-processing transform, such as
            a :class:`~torchrl.envs.Transform` or a :class:`~torchrl.data.postprocs.MultiStep`
            instance.
            Defaults to ``None``.
        async_envs (bool, optional): if ``True``, the environment will be run asynchronously. Defaults to `True` if the
            environment is a :class:`~torchrl.envs.AsyncEnvPool` instance.
        replay_buffer (ReplayBuffer, optional): if provided, the collector will not yield tensordicts
            but populate the buffer instead. Defaults to ``None``.
        reset_at_each_iter (bool, optional): if ``True``, the environment will be reset at each iteration.
        flatten_data (bool, optional): if ``True``, the collector will flatten the collected data
            before returning it. In practice, this means that if an environment of batch-size `(B,)` is used
            and run for `T` steps, `flatten_data=True` will present data of shape `(B*T,)`, whereas
            `flatten_data=False` will not present data of shape `(B, T)`.
            Defaults to `True` when `replay_buffer` is provided, `False` otherwise.
        weight_updater (WeightUpdaterBase or constructor, optional): An instance of :class:`~torchrl.collectors.WeightUpdaterBase`
            or its subclass, responsible for updating the policy weights on remote inference workers.
            This is typically not used in :class:`~torchrl.collectors.SyncDataCollector` as it operates in a single-process environment.
            Consider using a constructor if the updater needs to be serialized.
        track_policy_version (bool or PolicyVersion, optional): if ``True``, the collector will track the version of the policy.
            This will be mediated by the :class:`~torchrl.envs.llm.transforms.policy_version.PolicyVersion` transform, which will be added to the environment.
            Alternatively, a :class:`~torchrl.envs.llm.transforms.policy_version.PolicyVersion` instance can be passed, which will be used to track
            the policy version.
            Defaults to `False`.
        verbose (bool, optional): if ``True``, the collector will print progress information.
            Defaults to `False`.

    Examples:
        >>> import vllm
        >>> from torchrl.modules import vLLMWrapper
        >>> from pytorch.rl.test.mocking_classes import DummyStrDataLoader
        >>> from torchrl.envs import LLMEnv
        >>> llm_model = vllm.LLM("gpt2")
        >>> tokenizer = llm_model.get_tokenizer()
        >>> tokenizer.pad_token = tokenizer.eos_token
        >>> policy = vLLMWrapper(llm_model)
        >>> dataloader = DummyStrDataLoader(1)
        >>> env = LLMEnv.from_dataloader(
        ...    dataloader=dataloader,
        ...    tokenizer=tokenizer,
        ...    from_text=True,
        ...    batch_size=1,
        ...    group_repeats=True,
        ... )
        >>> collector = LLMCollector(
        ...    env=env,
        ...    policy_factory=lambda: policy,
        ...    dialog_turns_per_batch=env.batch_size[0],
        ...    total_dialog_turns=3,
        ... )
        >>> for i, data in enumerate(collector):
        ...     if i == 2:
        ...         print(data)
        ...         break
        LazyStackedTensorDict(
        fields={
            attention_mask: Tensor(shape=torch.Size([1, 1, 22]), device=cpu, dtype=torch.int64, is_shared=False),
            collector: LazyStackedTensorDict(
                fields={
                    traj_ids: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False)},
                exclusive_fields={
                },
                batch_size=torch.Size([1, 1]),
                device=None,
                is_shared=False,
                stack_dim=1),
            done: Tensor(shape=torch.Size([1, 1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
            terminated: Tensor(shape=torch.Size([1, 1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
            text: NonTensorStack(
                [['plsgqejeyd']],
                batch_size=torch.Size([1, 1]),
                device=None),
            text_response: NonTensorStack(
                [['ec.n.n.n.tjbjz3perwhz']],
                batch_size=torch.Size([1, 1]),
                device=None),
            tokens: Tensor(shape=torch.Size([1, 1, 22]), device=cpu, dtype=torch.int64, is_shared=False),
            tokens_response: Tensor(shape=torch.Size([1, 1, 16]), device=cpu, dtype=torch.int64, is_shared=False)},
        exclusive_fields={
        },
        batch_size=torch.Size([1, 1]),
        device=None,
        is_shared=False,
        stack_dim=1)
        >>> del collector

    """

    def __init__(
        self,
        env: EnvBase | Callable[[], EnvBase],
        *,
        policy: Callable[[TensorDictBase], TensorDictBase] | None = None,
        policy_factory: Callable[[], Callable[[TensorDictBase], TensorDictBase]]
        | None = None,
        dialog_turns_per_batch: int | None = None,
        yield_only_last_steps: bool | None = None,
        yield_completed_trajectories: bool | None = None,
        postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
        total_dialog_turns: int = -1,
        async_envs: bool | None = None,
        replay_buffer: ReplayBuffer | None = None,
        reset_at_each_iter: bool = False,
        flatten_data: bool | None = None,
        weight_updater: WeightUpdaterBase
        | Callable[[], WeightUpdaterBase]
        | None = None,
        queue: Any | None = None,
        track_policy_version: bool | PolicyVersion = False,
        verbose: bool = False,
    ):
        if queue is not None and replay_buffer is not None:
            raise RuntimeError(
                "Handling both a buffer and a queue is not possible at the moment."
            )
        elif queue is not None:
            # disguise the queue as a replay buffer
            replay_buffer = _QueueAsRB(queue)
        if dialog_turns_per_batch is None and yield_completed_trajectories:
            dialog_turns_per_batch = 1
        super().__init__(
            create_env_fn=env,
            policy=policy,
            policy_factory=policy_factory,
            frames_per_batch=dialog_turns_per_batch,
            replay_buffer=replay_buffer,
            total_frames=total_dialog_turns,
            weight_updater=weight_updater,
            reset_at_each_iter=reset_at_each_iter,
            trust_policy=True,
            use_buffers=False,
            no_cuda_sync=True,
            extend_buffer=True,
            postproc=postproc,
        )
        if hasattr(self.policy, "register_collector"):
            self.policy.register_collector(self)

        if yield_only_last_steps is None:
            yield_only_last_steps = False

        if yield_completed_trajectories is None:
            yield_completed_trajectories = yield_only_last_steps
        elif yield_only_last_steps and not yield_completed_trajectories:
            raise TypeError(
                "yield_only_last_steps=True requires yield_completed_trajectories=True (or None)"
            )

        if yield_only_last_steps:
            if flatten_data is not None:
                raise TypeError(
                    "`yield_only_last_steps` cannot be `True` when `flatten_data` is passed."
                )
            if self.reset_at_each_iter:
                raise TypeError(
                    "`yield_only_last_steps` cannot be `True` when `reset_at_each_iter=True`."
                )
        if flatten_data is None:
            flatten_data = replay_buffer is not None
        self.flatten_data = flatten_data
        self.yield_completed_trajectories = yield_completed_trajectories
        self.yield_only_last_steps = yield_only_last_steps
        self.verbose = verbose
        if self.yield_completed_trajectories:
            if len(self.env.batch_size) != 1:
                raise ValueError(
                    "`yield_only_last_steps` only works with envs that have a single batch dimension. Got "
                    f"env.batch_size={self.env.batch_size}."
                )
            self._yield_queues = [deque() for _ in range(self.env.batch_size[0])]
            self._trajectory_queue = deque()
        self.async_envs = bool(async_envs) | isinstance(self.env, AsyncEnvPool)
        if self.async_envs and not isinstance(self.env, AsyncEnvPool):
            # This basically means that `async_envs` is automatically set and passing is it useless as of today,
            #  except for the following error.
            raise RuntimeError(
                "async_envs requires the environment to be an AsyncEnvPool instance."
            )
        self.policy_version_tracker = track_policy_version
        if isinstance(track_policy_version, bool) and track_policy_version:
            if isinstance(self.env, AsyncEnvPool):
                raise RuntimeError(
                    "AsyncEnvPool is not supported for policy version tracking. Please add the PolicyVersion transform to the environment manually, "
                    "and pass that transform to the collector."
                )
            self.policy_version_tracker = PolicyVersion()
            self.env = self.env.append_transform(self.policy_version_tracker)  # type: ignore
        elif isinstance(track_policy_version, PolicyVersion):
            self.policy_version_tracker = track_policy_version
            self.env = self.env.append_transform(self.policy_version_tracker)  # type: ignore
        else:
            self.policy_version_tracker = None

    def set_postproc(self, postproc: Callable[[TensorDictBase], TensorDictBase]):
        if self.postproc is not None:
            raise RuntimeError("Postproc already set")
        self.postproc = postproc

    def increment_version(self):
        """Increment the policy version."""
        if self.policy_version_tracker is not None:
            if not isinstance(self.policy_version_tracker, PolicyVersion):
                raise RuntimeError(
                    "Policy version tracker is not a PolicyVersion instance. Please pass a PolicyVersion instance to the collector."
                )
            self.policy_version_tracker.increment_version()

    @property
    def policy_version(self) -> str | int | None:
        """The current policy version."""
        if not isinstance(self.policy_version_tracker, PolicyVersion):
            return None
        return self.policy_version_tracker.version

    def get_policy_version(self) -> str | int | None:
        """Get the current policy version.

        This method exists to support remote calls in Ray actors, since properties
        cannot be accessed directly through Ray's RPC mechanism.

        Returns:
            The current version number (int) or UUID (str), or None if version tracking is disabled.
        """
        return self.policy_version

    @property
    def total_dialog_turns(self):
        return self.total_frames

    @property
    def dialog_turns_per_batch(self) -> int:
        """Alias to `frames_per_batch`."""
        return self.requested_frames_per_batch

    @property
    def rollout(self) -> Callable[[], TensorDictBase]:
        if self.yield_completed_trajectories:
            if self.async_envs:
                return self._rollout_yield_trajs_async
            else:
                return self._rollout_yield_trajs
        else:
            return self._rollout_all

    def _rollout_all(self) -> TensorDictBase:  # A simplified version of rollout
        if self.reset_at_each_iter or self._shuttle is None:
            self._shuttle = self.env.reset()

        trajectory = []
        collected_steps = 0
        policy_input = self._shuttle
        while collected_steps < self.dialog_turns_per_batch:
            if self.verbose:
                torchrl_logger.info(
                    f"LLMCollector: Collected {collected_steps} steps over {self.dialog_turns_per_batch} requested."
                )
            env_input = self.policy(policy_input)
            env_output, env_next_output = self.env.step_and_maybe_reset(env_input)

            # carry over collector data without messing up devices
            collector_data = env_output.get("collector").copy()
            env_next_output.set("collector", collector_data)
            self._update_traj_ids(env_output)
            trajectory.append(env_output.clone())
            collected_steps += env_output.numel()
            policy_input = self._shuttle = env_next_output
        trajectory = lazy_stack(trajectory, -1)
        if self.flatten_data:
            return trajectory.view(-1)
        return trajectory

    _result_numel = 0

    def _rollout_yield_trajs(self) -> TensorDictBase:  # A simplified version of rollout
        if self._shuttle is None:
            raise RuntimeError("Data shuttle not found")
            # next_output = self.env.reset()
        else:
            next_output = self._shuttle

        collected_steps = 0
        dones = torch.zeros(self.env.batch_size, dtype=torch.bool)
        while True:
            if self._result_numel >= self.dialog_turns_per_batch:
                break
            elif self.verbose:
                torchrl_logger.info(
                    f"LLMCollector: Collected {collected_steps} steps with {self._result_numel} elements in the resulting batch, over {self.dialog_turns_per_batch} requested."
                )
            env_input = self.policy(next_output)
            cur_output, next_output = self.env.step_and_maybe_reset(env_input)
            # for i in range(cur_output.numel()):
            #     print(len(cur_output[i]["text"]) < len(cur_output[i]["next", "text"]))

            # carry over collector data without messing up devices
            self._update_traj_ids(cur_output)

            collector_data = cur_output.get("collector").copy()
            next_output.set("collector", collector_data)

            # if the loop is interrupted
            self._shuttle = next_output
            collected_steps += next_output.numel()
            for i, (_data, queue) in enumerate(
                zip(cur_output.unbind(0), self._yield_queues)
            ):
                queue.append(_data)
                dones[i] = _data["next", "done"].any()
            if dones.any():
                for idx in dones.nonzero(as_tuple=True)[0].tolist():
                    if not self.yield_only_last_steps:
                        _result = lazy_stack(self._yield_queues[idx], -1)
                        self._trajectory_queue.append(_result)
                    else:
                        # FIXME: We need to increment the step count here because iterator() won't
                        #  see the extra steps
                        # We use lazy-stack because unsqueeze doesn't nest the strings in lists
                        _result = lazy_stack([self._yield_queues[idx][-1]])
                        self._trajectory_queue.append(_result)
                    self._result_numel += _result.numel()
                    self._yield_queues[idx].clear()
        result = [self._trajectory_queue.popleft()]
        elt = result[0].numel()
        self._result_numel -= result[0].numel()
        while elt < self.dialog_turns_per_batch:
            result.append(self._trajectory_queue.popleft())
            elt += result[-1].numel()
            self._result_numel -= result[-1].numel()
        result = torch.cat(result, -1)
        if self.verbose:
            torchrl_logger.info(
                f"LLMCollector: Yielding completed trajectory with shape {result.shape}."
            )
        return result

    started = False

    def _rollout_yield_trajs_async(
        self,
    ) -> TensorDictBase:  # A simplified version of rollout
        if not self.started:
            next_output = self._shuttle
            env_input = self.policy(next_output)
            self.env.async_step_and_maybe_reset_send(env_input)
        self.started = True

        collected_steps = 0
        dones = torch.zeros(self.env.batch_size, dtype=torch.bool)
        while True:
            if self._trajectory_queue:
                break

            cur_output, next_output = self.env.async_step_and_maybe_reset_recv()

            # Get the env ids
            env_ids = cur_output.get(self.env._env_idx_key).tolist()

            # carry over collector data without messing up devices
            self._update_traj_ids(cur_output)

            collector_data = cur_output.get("collector").copy()
            next_output.set("collector", collector_data)

            collected_steps += next_output.numel()
            dones.fill_(False)
            for i, _data in zip(env_ids, cur_output.unbind(0)):
                queue = self._yield_queues[i]
                queue.append(_data)
                dones[i] = _data["next", "done"].any()
            if dones.any():
                for idx in dones.nonzero(as_tuple=True)[0].tolist():
                    if not self.yield_only_last_steps:
                        self._trajectory_queue.append(
                            lazy_stack(self._yield_queues[idx], -1)
                        )
                    else:
                        # FIXME: We need to increment the step count here because iterator() won't
                        #  see the extra steps
                        # We use lazy-stack because unsqueeze doesn't nest the strings in lists
                        self._trajectory_queue.append(
                            lazy_stack([self._yield_queues[idx][-1]])
                        )
                    self._yield_queues[idx].clear()

            # Launch the next batch:
            # FIXME: Add a condition RE number of frames here
            if True:
                env_input = self.policy(next_output)
                self.env.async_step_and_maybe_reset_send(env_input)

        result = self._trajectory_queue.popleft()
        if self.verbose:
            torchrl_logger.info(
                f"LLMCollector: Yielding completed trajectory with shape {result.shape}."
            )
        return result

    as_remote = as_remote

    def get_policy_model(self):
        """Get the policy model.

        This method is used by RayLLMCollector to get the remote LLM instance
        for weight updates.

        Returns:
            The policy model instance
        """
        return self.policy.model

    def is_initialized(self) -> bool:
        """Check if the collector is initialized and ready.

        Returns:
            bool: True if the collector is initialized and ready to collect data.
        """
        # The collector is initialized if it has a valid environment and policy
        return hasattr(self, "_env") and hasattr(self, "_policy")

    def set_weight_updater(self, weight_updater: WeightUpdaterBase):
        self.weight_updater = weight_updater
        return True
