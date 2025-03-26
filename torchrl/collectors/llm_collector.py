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
from torchrl.envs import LLMEnv, StepCounter
from torchrl.modules import vLLMWrapper


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



if __name__ == "__main__":
    import random
    import string
    from argparse import ArgumentParser

    from torchrl._utils import logger
    from vllm import LLM

    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--steps_per_batch", type=int, default=16)
    parser.add_argument("--optim_batch_size", type=int, default=4)

    class _DummyStrDataLoader:
        def __init__(self, batch_size=0):
            self.batch_size = batch_size

        def generate_random_string(self, length=10):
            """Generate a random string of a given length."""
            return "".join(random.choice(string.ascii_lowercase) for _ in range(length))

        def __iter__(self):
            return self

        def __next__(self):
            if self.batch_size == 0:
                return self.generate_random_string()
            else:
                return [self.generate_random_string() for _ in range(self.batch_size)]

    args = parser.parse_args()
    # NOTE: if VLLM fails with CUDA multiprocessing, try setting
    # `export VLLM_WORKER_MULTIPROC_METHOD=spawn`
    inference_model = LLM("gpt2")
    tokenizer = inference_model.get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    logger.info("Model loaded.")

    # Env
    dataloader = _DummyStrDataLoader(args.batch_size)
    env = LLMEnv.from_dataloader(
        dataloader=dataloader,
        tokenizer=tokenizer,
        str2str=True,
        batch_size=(args.batch_size,),
        repeats=args.repeats,
        group_repeats=True,
    )

    # Finally, we want the env to stop after the first step
    env.append_transform(StepCounter(max_steps=1))
    logger.info(f"Env: {env}")
    policy = vLLMWrapper(inference_model, tokenizer=tokenizer)
    logger.info(f"Policy: {policy}")
    collector = LLMCollector(
        env=env, policy_factory=lambda: policy, steps_per_batch=env.batch_size[0]
    )
    for data in collector:
        logger.info(data)
