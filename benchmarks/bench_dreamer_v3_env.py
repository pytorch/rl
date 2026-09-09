# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Fixed fake image-and-vector workload for the DreamerV3 training benchmark.

The environment is importable by name from environment worker processes and by
the example's ``env.factory`` option (``bench_dreamer_v3_env:make_env``). Keep
its observation layout and step cost stable: the continuous benchmark compares
revisions on this exact workload (see ASYNC_BENCHMARKS.md).
"""
from __future__ import annotations

import time

import torch
from tensordict import TensorDict

from torchrl.data import Bounded, Composite, OneHot, Unbounded
from torchrl.envs import EnvBase

PIXEL_SHAPE = (3, 64, 64)
VECTOR_DIM = 8
NUM_MILESTONES = 3


class FakePixelEnv(EnvBase):
    """Uint8 pixels, a float vector and boolean milestones with a fixed step cost.

    Episode lengths are staggered by ``env_index`` so resets are not
    synchronized across environments. Rewards depend on the chosen action, so
    the learner sees a non-degenerate target.
    """

    def __init__(
        self,
        *,
        seed: int | None = None,
        env_index: int = 0,
        num_envs: int = 1,
        episode_length: int = 200,
        num_actions: int = 6,
        step_latency_s: float = 0.001,
    ):
        super().__init__(device="cpu", batch_size=torch.Size([]))
        self.observation_spec = Composite(
            pixels=Bounded(0, 255, PIXEL_SHAPE, dtype=torch.uint8),
            vector=Unbounded((VECTOR_DIM,)),
            obtained=Unbounded((NUM_MILESTONES,), dtype=torch.bool),
            shape=(),
        )
        self.action_spec = OneHot(num_actions, dtype=torch.float32)
        self.reward_spec = Unbounded((1,))
        self.episode_length = episode_length + 7 * (env_index % max(num_envs, 1))
        self.step_latency_s = step_latency_s
        self._t = 0
        self._set_seed(seed)

    def _set_seed(self, seed: int | None):
        self.rng = torch.Generator().manual_seed(0 if seed is None else int(seed))

    def _observation(self) -> TensorDict:
        return TensorDict(
            {
                "pixels": torch.randint(
                    0, 256, PIXEL_SHAPE, generator=self.rng, dtype=torch.uint8
                ),
                "vector": torch.randn(VECTOR_DIM, generator=self.rng),
                "obtained": torch.tensor(
                    [
                        self._t > self.episode_length // 4,
                        self._t > self.episode_length // 2,
                        self._t > 3 * self.episode_length // 4,
                    ]
                ),
            },
            [],
        )

    def _reset(self, tensordict=None, **kwargs) -> TensorDict:
        self._t = 0
        observation = self._observation()
        observation.set("done", torch.zeros(1, dtype=torch.bool))
        observation.set("terminated", torch.zeros(1, dtype=torch.bool))
        return observation

    def _step(self, tensordict: TensorDict) -> TensorDict:
        if self.step_latency_s > 0:
            time.sleep(self.step_latency_s)
        self._t += 1
        observation = self._observation()
        action = tensordict["action"].float().argmax(-1, keepdim=True)
        observation.set("reward", 0.1 * action.float())
        done = torch.tensor([self._t >= self.episode_length])
        observation.set("done", done)
        observation.set("terminated", done)
        return observation


def make_env(*, seed, env_index, num_envs, **kwargs) -> FakePixelEnv:
    """``env.factory`` entry point of the DreamerV3 example."""
    return FakePixelEnv(seed=seed, env_index=env_index, num_envs=num_envs, **kwargs)
