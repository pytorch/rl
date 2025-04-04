# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Callable

from tensordict import lazy_stack, TensorDictBase

from torchrl.collectors import (
    LocalWeightUpdaterBase,
    RemoteWeightUpdaterBase,
    SyncDataCollector,
)
from torchrl.data.replay_buffers.replay_buffers import ReplayBuffer
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
        async_envs (bool, optional): if ``True``, the environment will be run synchronously.
        replay_buffer (ReplayBuffer, optional): if provided, the collector will not yield tensordicts
            but populate the buffer instead. Defaults to ``None``.
        reset_at_each_iter (bool, optional): if ``True``, the environment will be reset at each iteration.
        local_weight_updater (LocalWeightUpdaterBase or constructor, optional): An instance of :class:`~torchrl.collectors.LocalWeightUpdaterBase`
            or its subclass, responsible for updating the policy weights on the local inference worker.
            If not provided, a :class:`~torchrl.collectors.VanillaLocalWeightUpdater` will be used by default,
            which directly fetches and applies the weights from the server.
            Consider using a constructor if the updater needs to be serialized.
        remote_weight_updater (RemoteWeightUpdaterBase or constructor, optional): An instance of :class:`~torchrl.collectors.RemoteWeightUpdaterBase`
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
        ...    str2str=True,
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
        total_steps: int = -1,
        async_envs: bool = False,
        replay_buffer: ReplayBuffer | None = None,
        reset_at_each_iter: bool = False,
        local_weight_updater: LocalWeightUpdaterBase
        | Callable[[], LocalWeightUpdaterBase]
        | None = None,
        remote_weight_updater: RemoteWeightUpdaterBase
        | Callable[[], RemoteWeightUpdaterBase]
        | None = None,
    ):
        if async_envs:
            raise NotImplementedError
        super().__init__(
            create_env_fn=env,
            policy=policy,
            policy_factory=policy_factory,
            frames_per_batch=steps_per_batch,
            replay_buffer=replay_buffer,
            total_frames=total_steps,
            local_weight_updater=local_weight_updater,
            remote_weight_updater=remote_weight_updater,
            reset_at_each_iter=reset_at_each_iter,
            trust_policy=True,
            use_buffers=False,
            no_cuda_sync=True,
        )

    @property
    def steps_per_batch(self) -> int:
        """Alias to `frames_per_batch`."""
        return self.frames_per_batch

    def rollout(self) -> TensorDictBase:  # A simplified version of rollout
        if self.reset_at_each_iter or self._shuttle is None:
            data = self.env.reset()
        else:
            data = self._shuttle

        tensordicts = []
        collected_steps = 0
        while collected_steps < self.steps_per_batch:
            policy_input = data
            env_input = self.policy(policy_input)
            env_output, env_next_output = self.env.step_and_maybe_reset(env_input)

            # carry over collector data without messing up devices
            collector_data = env_output.get("collector").copy()
            env_next_output.set("collector", collector_data)
            self._shuttle = env_next_output
            self._shuttle.set("collector", collector_data)
            self._update_traj_ids(env_output)
            data = self._shuttle
            tensordicts.append(data)
            collected_steps += data.numel()

        data = lazy_stack(tensordicts, -1)

        if self.replay_buffer is not None:
            self.replay_buffer.extend(data)
            if self._increment_frames(data.numel()):
                return
            return None
        return data
