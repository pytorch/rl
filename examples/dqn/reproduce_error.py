"""
DQN Benchmarks: Reproducing Experiments from Mnih et al. 2015
Deep Q-Learning Algorithm on Atari Environments.
"""

import random
import gym
import tqdm
import hydra
import numpy as np
import torch.nn
import torch.optim

from torchrl.collectors import SyncDataCollector
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
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
    RewardSum,
    RewardClipping
)
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import ConvNet, EGreedyWrapper, MLP, QValueActor

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
        obs, rew, done, info = self.env.step(action)
        lives = self.env.unwrapped.ale.lives()
        info["end_of_life"] = False
        if (lives < self.lives) or done:
            info["end_of_life"] = True
        self.lives = lives
        return obs, rew, done, info

    def reset(self, **kwargs):
        reset_data = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return reset_data


# TODO: this function makes collection crash
def make_env1(env_name, frame_skip, device, is_test=False):
    env = gym.make(env_name)
    if not is_test:
        env = EpisodicLifeEnv(env)
    env = GymWrapper(
        env, frame_skip=frame_skip, from_pixels=True, pixels_only=False, device=device
    )
    env = TransformedEnv(env)
    reader = default_info_dict_reader(["end_of_life"])
    env.set_info_dict_reader(reader)
    env.append_transform(ToTensorImage())
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=4, dim=-3))
    env.append_transform(DoubleToFloat())
    return env


def make_base_env(env_name, frame_skip, device, is_test=False):
    env = gym.make(env_name)
    env = GymWrapper(
        env, frame_skip=frame_skip, from_pixels=True, pixels_only=False, device=device
    )
    env = TransformedEnv(env)
    reader = default_info_dict_reader(["end_of_life"])
    env.set_info_dict_reader(reader)
    return env


def make_env2(env_name, frame_skip, device, is_test=False):
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
    env.append_transform(DoubleToFloat())
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
    proof_environment = make_env1(env_name, frame_skip, device="cpu")
    qvalue_module = make_dqn_modules_pixels(proof_environment)
    del proof_environment
    return qvalue_module


if __name__ == "__main__":

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Correct for frame_skip
    frame_skip = 4
    total_frames = 10_000_000
    frames_per_batch = 4
    init_random_frames = 1_000

    # Make the components
    model = make_dqn_model("PongNoFrameskip-v4", frame_skip)
    model_explore = EGreedyWrapper(
        model,
        annealing_num_steps=250_000,
        eps_end=0.01,
    ).to(device)

    # Create the collector
    collector = SyncDataCollector(
        create_env_fn=make_env1("PongNoFrameskip-v4", frame_skip, device),
        policy=model_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
        init_random_frames=init_random_frames
    )
    collector.set_seed(seed)

    pbar = tqdm.tqdm(total=total_frames)
    for data in collector:
        pbar.update(data.numel())

    collector.shutdown()

