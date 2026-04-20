# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
This script gives some examples of gym environment conversion with Dict, Tuple and Sequence spaces.
"""

import gymnasium as gym
from gymnasium import spaces

from torchrl.envs import GymWrapper

action_space = spaces.Discrete(2)


class BaseEnv(gym.Env):
    def step(self, action):
        return self.observation_space.sample(), 1, False, False, {}

    def reset(self, **kwargs):
        return self.observation_space.sample(), {}


class SimpleEnv(BaseEnv):
    def __init__(self):
        self.observation_space = spaces.Box(-1, 1, (2,))
        self.action_space = action_space


gym.register("SimpleEnv-v0", entry_point=SimpleEnv)


class SimpleEnvWithDict(BaseEnv):
    def __init__(self):
        self.observation_space = spaces.Dict(
            obs0=spaces.Box(-1, 1, (2,)), obs1=spaces.Box(-1, 1, (3,))
        )
        self.action_space = action_space


gym.register("SimpleEnvWithDict-v0", entry_point=SimpleEnvWithDict)


class SimpleEnvWithTuple(BaseEnv):
    def __init__(self):
        self.observation_space = spaces.Tuple(
            (spaces.Box(-1, 1, (2,)), spaces.Box(-1, 1, (3,)))
        )
        self.action_space = action_space


gym.register("SimpleEnvWithTuple-v0", entry_point=SimpleEnvWithTuple)


class SimpleEnvWithSequence(BaseEnv):
    def __init__(self):
        self.observation_space = spaces.Sequence(
            spaces.Box(-1, 1, (2,)),
            # Only stack=True is currently allowed
            stack=True,
        )
        self.action_space = action_space


gym.register("SimpleEnvWithSequence-v0", entry_point=SimpleEnvWithSequence)


class SimpleEnvWithSequenceOfTuple(BaseEnv):
    def __init__(self):
        self.observation_space = spaces.Sequence(
            spaces.Tuple(
                (
                    spaces.Box(-1, 1, (2,)),
                    spaces.Box(-1, 1, (3,)),
                )
            ),
            # Only stack=True is currently allowed
            stack=True,
        )
        self.action_space = action_space


gym.register(
    "SimpleEnvWithSequenceOfTuple-v0", entry_point=SimpleEnvWithSequenceOfTuple
)


class SimpleEnvWithTupleOfSequences(BaseEnv):
    def __init__(self):
        self.observation_space = spaces.Tuple(
            (
                spaces.Sequence(
                    spaces.Box(-1, 1, (2,)),
                    # Only stack=True is currently allowed
                    stack=True,
                ),
                spaces.Sequence(
                    spaces.Box(-1, 1, (3,)),
                    # Only stack=True is currently allowed
                    stack=True,
                ),
            )
        )
        self.action_space = action_space


gym.register(
    "SimpleEnvWithTupleOfSequences-v0", entry_point=SimpleEnvWithTupleOfSequences
)

if __name__ == "__main__":
    for envname in [
        "SimpleEnv",
        "SimpleEnvWithDict",
        "SimpleEnvWithTuple",
        "SimpleEnvWithSequence",
        "SimpleEnvWithSequenceOfTuple",
        "SimpleEnvWithTupleOfSequences",
    ]:
        print("\n\nEnv =", envname)
        env = gym.make(envname + "-v0")
        env_torchrl = GymWrapper(env)
        print(env_torchrl.rollout(10, return_contiguous=False))
