# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import gymnasium as gym
import numpy as np
import torch.nn
import torch.optim

from torchrl.data import CompositeSpec
from torchrl.envs import (
    CatFrames,
    default_info_dict_reader,
    DoubleToFloat,
    EnvCreator,
    GrayScale,
    NoopResetEnv,
    ParallelEnv,
    Resize,
    RewardClipping,
    RewardSum,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import ConvNet, MLP, QValueActor


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. It helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0

    def step(self, action):
        obs, rew, done, truncated, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        info["end_of_life"] = False
        if (lives < self.lives) or done:
            info["end_of_life"] = True
        self.lives = lives
        return obs, rew, done, truncated, info

    def reset(self, **kwargs):
        reset_data = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return reset_data


def make_base_env(env_name, frame_skip, device, is_test=False):
    env = gym.make(env_name)
    if not is_test:
        env = EpisodicLifeEnv(env)
    env = GymWrapper(
        env, frame_skip=frame_skip, from_pixels=True, pixels_only=False, device=device
    )
    env = TransformedEnv(env)
    if not is_test:
        env.append_transform(NoopResetEnv(noops=30, random=True))
        reader = default_info_dict_reader(["end_of_life"])
    env.set_info_dict_reader(reader)
    return env


def make_env(env_name, frame_skip, device, is_test=False):
    num_envs = 1
    env = ParallelEnv(
        num_envs,
        EnvCreator(
            lambda: make_base_env(env_name, frame_skip, device=device, is_test=is_test)
        ),
    )
    env = TransformedEnv(env)
    env.append_transform(ToTensorImage())
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=4, dim=-3))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter(max_steps=4500))
    if not is_test:
        env.append_transform(RewardClipping(-1, 1))
    env.append_transform(DoubleToFloat())
    # env.append_transform(VecNorm(in_keys=["pixels"]))
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_dqn_modules_pixels(proof_environment):

    # Define input shape
    input_shape = proof_environment.observation_spec["pixels"].shape
    env_specs = proof_environment.specs
    num_actions = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]

    # Define Q-Value Module
    cnn = ConvNet(
        activation_class=torch.nn.ReLU,
        num_cells=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[4, 2, 1],
    )
    cnn_output = cnn(torch.ones(input_shape))
    mlp = MLP(
        in_features=cnn_output.shape[-1],
        activation_class=torch.nn.ReLU,
        out_features=num_actions,
        num_cells=[512],
    )
    qvalue_module = QValueActor(
        module=torch.nn.Sequential(cnn, mlp),
        spec=CompositeSpec(action=action_spec),
        in_keys=["pixels"],
    )
    return qvalue_module


def make_dqn_model(env_name, frame_skip):
    proof_environment = make_env(env_name, frame_skip, device="cpu")
    qvalue_module = make_dqn_modules_pixels(proof_environment)
    del proof_environment
    return qvalue_module


# ====================================================================
# Evaluation utils
# --------------------------------------------------------------------


def eval_model(actor, test_env, num_episodes=3):
    test_rewards = []
    for _ in range(num_episodes):
        td_test = test_env.rollout(
            policy=actor,
            auto_reset=True,
            auto_cast_to_device=True,
            break_when_any_done=True,
            max_steps=10_000_000,
        )
        reward = td_test["next", "episode_reward"][td_test["next", "done"]]
        test_rewards = np.append(test_rewards, reward.cpu().numpy())
    del td_test
    return test_rewards.mean()
