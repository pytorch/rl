# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections import deque
from typing import Callable

import torch

from tensordict import lazy_stack, TensorDictBase

from torchrl.collectors import (
    SyncDataCollector,
    WeightUpdateReceiverBase,
    WeightUpdateSenderBase,
)
from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
from torchrl.envs import AsyncEnvPool
from torchrl.envs.common import EnvBase


class LLMCollector(SyncDataCollector):
    """A simplified version of SyncDataCollector for LLM inference.

    Args:
        env (EnvBase or EnvBase constructor): the environment to be used for data collection.

    Keyword Args:
        policy (Callable[[TensorDictBase], TensorDictBase]): the policy to be used for data collection.
        policy_factory (Callable[[], Callable], optional): a callable that returns
            a policy instance. This is exclusive with the `policy` argument.

            .. note:: `policy_factory` comes in handy whenever the policy cannot be serialized.

        steps_per_batch (int): A keyword-only argument representing the total
            number of elements in a batch; -1 is never ending (until shutdown).
        total_steps (int): A keyword-only argument representing the total
            number of steps returned by the collector
            during its lifespan.
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
        weight_update_receiver (WeightUpdateReceiverBase or constructor, optional): An instance of :class:`~torchrl.collectors.WeightUpdateReceiverBase`
            or its subclass, responsible for updating the policy weights on the local inference worker.
            If not provided, a :class:`~torchrl.collectors.VanillaLocalWeightUpdater` will be used by default,
            which directly fetches and applies the weights from the server.
            Consider using a constructor if the updater needs to be serialized.
        weight_update_sender (WeightUpdateSenderBase or constructor, optional): An instance of :class:`~torchrl.collectors.WeightUpdateSenderBase`
            or its subclass, responsible for updating the policy weights on remote inference workers.
            This is typically not used in :class:`~torchrl.collectors.SyncDataCollector` as it operates in a single-process environment.
            Consider using a constructor if the updater needs to be serialized.

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
        ...    steps_per_batch=env.batch_size[0],
        ...    total_steps=3,
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
        steps_per_batch: int,
        yield_only_last_steps: bool | None = None,
        yield_completed_trajectories: bool | None = None,
        postproc: Callable[[TensorDictBase], TensorDictBase] | None = None,
        total_steps: int = -1,
        async_envs: bool | None = None,
        replay_buffer: ReplayBuffer | None = None,
        reset_at_each_iter: bool = False,
        flatten_data: bool | None = None,
        weight_update_receiver: WeightUpdateReceiverBase
        | Callable[[], WeightUpdateReceiverBase]
        | None = None,
        weight_update_sender: WeightUpdateSenderBase
        | Callable[[], WeightUpdateSenderBase]
        | None = None,
    ):
        super().__init__(
            create_env_fn=env,
            policy=policy,
            policy_factory=policy_factory,
            frames_per_batch=steps_per_batch,
            replay_buffer=replay_buffer,
            total_frames=total_steps,
            weight_update_receiver=weight_update_receiver,
            weight_update_sender=weight_update_sender,
            reset_at_each_iter=reset_at_each_iter,
            trust_policy=True,
            use_buffers=False,
            no_cuda_sync=True,
            extend_buffer=True,
        )
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

    @property
    def steps_per_batch(self) -> int:
        """Alias to `frames_per_batch`."""
        return self.frames_per_batch

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
            data = self.env.reset()
        else:
            data = self._shuttle

        trajectory = []
        collected_steps = 0
        while collected_steps < self.steps_per_batch:
            policy_input = data
            env_input = self.policy(policy_input)
            env_output, env_next_output = self.env.step_and_maybe_reset(env_input)

            # carry over collector data without messing up devices
            collector_data = env_output.get("collector").copy()
            env_next_output.set("collector", collector_data)
            self._shuttle = env_next_output
            self._update_traj_ids(env_output)
            data = env_output
            trajectory.append(data)
            collected_steps += data.numel()
        trajectory = lazy_stack(trajectory, -1)
        if self.flatten_data:
            return trajectory.view(-1)
        return trajectory

    def _rollout_yield_trajs(self) -> TensorDictBase:  # A simplified version of rollout
        if self._shuttle is None:
            raise RuntimeError("Data shuttle not found")
            # next_output = self.env.reset()
        else:
            next_output = self._shuttle

        collected_steps = 0
        dones = torch.zeros(self.env.batch_size, dtype=torch.bool)
        while True:
            if self._trajectory_queue:
                break
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
                for idx in dones.nonzero()[0].tolist():
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

        result = self._trajectory_queue.popleft()
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
                for idx in dones.nonzero()[0].tolist():
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
        return result
