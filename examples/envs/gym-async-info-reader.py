# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
A toy example of executing a Gym environment asynchronously and gathering the info properly.
"""
import argparse

import gymnasium as gym
import numpy as np
from gymnasium import spaces

parser = argparse.ArgumentParser()
parser.add_argument("--use_wrapper", action="store_true")

# Create the dummy environment
class CustomEnv(gym.Env):
    def __init__(self, render_mode=None):
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(3,))
        self.action_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

    def _get_info(self):
        return {"field1": self.state**2}

    def _get_obs(self):
        return self.state.copy()

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        self.state = np.zeros(self.observation_space.shape)
        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        self.state += action.item()
        truncated = False
        terminated = False
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()
        return observation, reward, terminated, truncated, info


if __name__ == "__main__":
    import torch
    from torchrl.data.tensor_specs import UnboundedContinuousTensorSpec
    from torchrl.envs import check_env_specs, GymEnv, GymWrapper

    args = parser.parse_args()

    num_envs = 10

    if args.use_wrapper:
        # Option 1: using GymWrapper
        env = gym.vector.AsyncVectorEnv([lambda: CustomEnv() for _ in range(num_envs)])
        env = GymWrapper(env, device="cpu")
    else:
        # Option 2: using GymEnv directly, no need to call AsyncVectorEnv
        gym.register("Custom-v0", CustomEnv)
        env = GymEnv("Custom-v0", num_envs=num_envs)

    keys = ["field1"]
    specs = [
        UnboundedContinuousTensorSpec(shape=(num_envs, 3), dtype=torch.float64),
    ]

    # Create an info reader: this object will read the info and write its content to the tensordict
    reader = lambda info, tensordict: tensordict.set("field1", np.stack(info["field1"]))
    env.set_info_dict_reader(info_dict_reader=reader)

    # Print the info readers (there should be 2: one to read the terminal states and another to read the 'field1')
    print("readers", env.info_dict_reader)

    # We need to unlock the specs to make them writable
    env.observation_spec.unlock_()
    env.observation_spec["field1"] = specs[0]
    env.observation_spec.lock_()

    # Check that we did a good job
    check_env_specs(env)

    td = env.reset()
    print("reset data", td)
    print("content of field1 (should be a 10x3 tensor)", td["field1"])
