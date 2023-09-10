"""
DQN Benchmarks: Reproducing Experiments from Mnih et al. 2015
Deep Q-Learning Algorithm on Atari Environments.
"""

import random
import time

import gym
import hydra
import numpy as np
import torch.nn
import torch.optim
import tqdm
from tensordict import TensorDict

from torchrl.collectors import SyncDataCollector
from torchrl.data import CompositeSpec, LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs import (
    CatFrames,
    default_info_dict_reader,
    DoubleToFloat,
    EnvCreator,
    ExplorationType,
    GrayScale,
    NoopResetEnv,
    ParallelEnv,
    Resize,
    RewardClipping,
    RewardSum,
    set_exploration_type,
    StepCounter,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.libs.gym import GymWrapper
from torchrl.modules import ConvNet, EGreedyWrapper, MLP, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.record.loggers import generate_exp_name, get_logger

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
# def make_env(env_name, device, is_test=False):
#     env = gym.make(env_name)
#     if not is_test:
#         env = EpisodicLifeEnv(env)
#     env = GymWrapper(
#         env, frame_skip=frame_skip, from_pixels=True, pixels_only=False, device=device
#     )
#     env = TransformedEnv(env)
#     reader = default_info_dict_reader(["end_of_life"])
#     env.set_info_dict_reader(reader)
#     env.append_transform(NoopResetEnv(noops=8))
#     env.append_transform(ToTensorImage())
#     env.append_transform(GrayScale())
#     env.append_transform(Resize(84, 84))
#     env.append_transform(CatFrames(N=4, dim=-3))
#     env.append_transform(RewardSum())
#     env.append_transform(StepCounter(max_steps=4500))
#     if not is_test:
#         env.append_transform(RewardClipping(-1, 1))
#     env.append_transform(DoubleToFloat())
#     # env.append_transform(VecNorm(in_keys=["pixels"]))
#     return env


def make_base_env(env_name, frame_skip, device, is_test=False):
    env = gym.make(env_name)
    if not is_test:
        env = EpisodicLifeEnv(env)
    env = GymWrapper(
        env, frame_skip=frame_skip, from_pixels=True, pixels_only=False, device=device
    )
    env = TransformedEnv(env)
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
    env.append_transform(VecNorm(in_keys=["pixels"]))
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


@hydra.main(config_path=".", config_name="config_atari", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Correct for frame_skip
    frame_skip = 4
    total_frames = cfg.collector.total_frames // frame_skip
    frames_per_batch = cfg.collector.frames_per_batch // frame_skip
    test_interval = cfg.logger.test_interval // frame_skip
    init_random_frames = cfg.collector.init_random_frames // frame_skip

    # Make the components
    model = make_dqn_model(cfg.env.env_name, frame_skip)
    model_explore = EGreedyWrapper(
        model,
        annealing_num_steps=cfg.collector.annealing_frames,
        eps_end=cfg.collector.end_e,
    ).to(device)

    # Create the collector
    collector = SyncDataCollector(
        create_env_fn=make_env(cfg.env.env_name, frame_skip, device),
        policy=model_explore,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )
    collector.set_seed(seed)

    # Create the replay buffer
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=3,
        storage=LazyMemmapStorage(
            max_size=cfg.buffer.buffer_size,
            scratch_dir="/tmp/",
            device=device,
        ),
        batch_size=cfg.buffer.batch_size,
    )

    # Create the loss module
    loss_module = DQNLoss(
        value_network=model,
        gamma=cfg.loss.gamma,
        loss_function="l2",
        delay_value=True,
    )
    loss_module.make_value_estimator(gamma=cfg.loss.gamma)
    target_net_updater = HardUpdate(
        loss_module, value_network_update_interval=cfg.loss.hard_update_freq
    )

    # Create the optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    # Create the logger
    exp_name = generate_exp_name("DQN", f"Atari_mnih15_{cfg.env.env_name}")
    logger = get_logger(cfg.logger.backend, logger_name="dqn", experiment_name=exp_name)

    # Create the test environment
    test_env = make_env(cfg.env.env_name, frame_skip, device, is_test=True)
    test_env.eval()

    # Main loop
    collected_frames = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    for data in collector:

        # Train loging
        logger.log_scalar(
            "q_values",
            (data["action_value"] * data["action"]).sum().item() / frames_per_batch,
            collected_frames,
        )
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            logger.log_scalar(
                "reward_train", episode_rewards.mean().item(), collected_frames
            )
            logger.log_scalar(
                "episode_length_train",
                episode_length.sum().item() / len(episode_length),
                collected_frames,
            )

        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel() * frame_skip
        collected_frames += current_frames
        model_explore.step(current_frames)
        replay_buffer.extend(data.to(device))

        # optimization steps
        if collected_frames > init_random_frames:

            q_losses = TensorDict({}, batch_size=[cfg.loss.num_updates])
            for j in range(cfg.loss.num_updates):

                sampled_tensordict = replay_buffer.sample()

                loss_td = loss_module(sampled_tensordict)
                q_loss = loss_td["loss"]
                optimizer.zero_grad()
                q_loss.backward()
                # grad_norm = torch.nn.utils.clip_grad_norm_(list(loss_module.parameters()), max_norm=0.5)
                optimizer.step()
                target_net_updater.step()
                q_losses[j] = loss_td.select("loss").detach()

            q_losses = q_losses.apply(lambda x: x.float().mean(), batch_size=[])
            for key, value in q_losses.items():
                logger.log_scalar(key, value.item(), collected_frames)
            logger.log_scalar("epsilon", model_explore.eps, collected_frames)

            # Test logging
            with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
                if (collected_frames - frames_per_batch) // test_interval < (
                    collected_frames // test_interval
                ):
                    model.eval()
                    test_rewards = []
                    for _ in range(cfg.logger.num_test_episodes):
                        td_test = test_env.rollout(
                            policy=model,
                            auto_reset=True,
                            auto_cast_to_device=True,
                            break_when_any_done=True,
                            max_steps=10_000_000,
                        )
                        reward = td_test["next", "episode_reward"][
                            td_test["next", "done"]
                        ]
                        test_rewards = np.append(test_rewards, reward.cpu().numpy())
                        del td_test
                    logger.log_scalar(
                        "reward_test", test_rewards.mean(), collected_frames
                    )
                    model.train()

        # update weights of the inference policy
        collector.update_policy_weights_()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
