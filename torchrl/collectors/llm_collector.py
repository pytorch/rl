# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from tensordict import lazy_stack, TensorDictBase

from torchrl.collectors import (
    LocalWeightUpdaterBase,
    RemoteWeightUpdaterBase,
    SyncDataCollector,
)
from torchrl.data import ReplayBuffer


class LLMCollector(SyncDataCollector):
    """A simplified version of SyncDataCollector for LLM inference."""

    def __init__(
        self,
        env,
        policy_factory,
        *,
        steps_per_batch: int = -1,
        # -1 is never ending (until shutdown)
        total_steps: int = -1,
        async_envs: bool = False,
        replay_buffer: ReplayBuffer | None = None,
        reset_at_each_iter: bool = False,
        local_weight_updater: LocalWeightUpdaterBase | None = None,
        remote_weight_updater: RemoteWeightUpdaterBase | None = None,
    ):
        if async_envs:
            raise NotImplementedError
        super().__init__(
            create_env_fn=env,
            policy_factory=policy_factory,
            frames_per_batch=steps_per_batch,
            replay_buffer=replay_buffer,
            total_frames=total_steps,
            local_weight_updater=local_weight_updater,
            remote_weight_updater=remote_weight_updater,
            reset_at_each_iter=reset_at_each_iter,
            use_buffers=False,
        )

    def rollout(self) -> TensorDictBase:  # A simplified version of rollout
        if self.reset_at_each_iter or self._shuttle is None:
            data = self.env.reset()
        else:
            data = self._shuttle

        tensordicts = []
        collected_frames = 0
        while collected_frames < self.frames_per_batch:
            policy_input = data
            env_input = self.policy(policy_input)
            env_output, env_next_output = self.env.step_and_maybe_reset(env_input)

            # carry over collector data without messing up devices
            collector_data = env_output.get("collector").copy()
            env_next_output.set("collector", collector_data)
            self._shuttle = env_next_output
            if self._shuttle_has_no_device:
                self._shuttle.clear_device_()
            self._shuttle.set("collector", collector_data)
            self._update_traj_ids(env_output)
            data = self._shuttle
            tensordicts.append(data)
            collected_frames += data.numel()

        data = lazy_stack(tensordicts, -1)

        if self.replay_buffer is not None:
            self.replay_buffer.extend(data)
            return None
        return data
