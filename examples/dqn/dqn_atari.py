"""
DQN Benchmarks: Reproducing Experiments from Mnih et al. 2015
Deep Q-Learning Algorithm on Atari Environments.
"""

import gym
import tqdm
import time
import random
import torch.nn
import torch.optim
import numpy as np
from tensordict import TensorDict

from torchrl.collectors import SyncDataCollector
from torchrl.data import CompositeSpec, LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import (
    default_info_dict_reader,
    Resize,
    VecNorm,
    GrayScale,
    RewardSum,
    CatFrames,
    StepCounter,
    ToTensorImage,
    DoubleToFloat,
    RewardClipping,
    TransformedEnv,
    NoopResetEnv,
    ExplorationType,
    set_exploration_type,
)
from torchrl.modules import (
    MLP,
    ConvNet,
    QValueActor,
    EGreedyWrapper,
)
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


def make_env(env_name, device, is_test=False):
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


def make_dqn_model(env_name):
    proof_environment = make_env(env_name, device="cpu")
    qvalue_module = make_dqn_modules_pixels(proof_environment)
    del proof_environment
    return qvalue_module


# ====================================================================
# Collector utils
# --------------------------------------------------------------------

def make_collector(env_name, policy, device):
    collector_class = SyncDataCollector
    collector = collector_class(
        make_env(env_name, device),
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )
    collector.set_seed(seed)
    return collector

# ====================================================================
# Collector and replay buffer utils
# --------------------------------------------------------------------


def make_replay_buffer(
        batch_size,
        buffer_scratch_dir="/tmp/",
        prefetch=3,
):
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=prefetch,
        storage=LazyMemmapStorage(
            max_size=buffer_size,
            scratch_dir=buffer_scratch_dir,
            device=device,
        ),
        batch_size=batch_size,
    )
    return replay_buffer

# ====================================================================
# Discrete DQN Loss
# --------------------------------------------------------------------


def make_loss_module(value_network):
    """Make loss module and target network updater."""
    dqn_loss = DQNLoss(
        value_network=value_network,
        gamma=gamma,
        loss_function="l2",
        delay_value=True,
    )
    dqn_loss.make_value_estimator(gamma=gamma)
    targ_net_updater = HardUpdate(dqn_loss, value_network_update_interval=hard_update_freq)
    return dqn_loss, targ_net_updater

# ====================================================================
# Other component utils
# --------------------------------------------------------------------


def make_optimizer(dqn_loss):
    optimizer = torch.optim.Adam(dqn_loss.parameters(), lr=lr)
    return optimizer


def make_logger(backend="csv"):
    exp_name = generate_exp_name("DQN", f"Atari_mnih15_{env_name}")
    logger = get_logger(backend, logger_name="dqn", experiment_name=exp_name)
    return logger


if __name__ == "__main__":

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    env_name = "PongNoFrameskip-v4"
    frame_skip = 4
    total_frames = 10_000_000
    record_interval = 10_000_000
    frames_per_batch = 4
    num_updates = 1
    buffer_size = 1_000_000
    init_random_frames = 80_000
    annealing_frames = 1_000_000
    gamma = 0.99
    lr = 1e-4
    batch_size = 32
    hard_update_freq = 250
    start_e = 1.0
    end_e = 0.05
    logger_backend = "wandb"

    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    # Make the components
    model = make_dqn_model(env_name)
    model_explore = EGreedyWrapper(model, annealing_num_steps=annealing_frames, eps_end=end_e).to(device)
    collector = make_collector(env_name, model_explore, device)
    replay_buffer = make_replay_buffer(batch_size)
    loss_module, target_net_updater = make_loss_module(model)
    optimizer_actor = make_optimizer(loss_module)
    logger = make_logger(logger_backend)
    test_env = make_env(env_name, device, is_test=True)
    test_env.eval()

    # Main loop
    collected_frames = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=total_frames)

    for i, data in enumerate(collector):

        # Train loging
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            logger.log_scalar("reward_train", episode_rewards.mean().item(), collected_frames)
            logger.log_scalar("episode_length_train", episode_length.sum().item() / len(episode_length), collected_frames)

        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel() * frame_skip
        collected_frames += current_frames
        model_explore.step(current_frames)
        replay_buffer.extend(data.to(device))

        # optimization steps
        if collected_frames > init_random_frames:

            q_losses = TensorDict({}, batch_size=[num_updates])
            for j in range(num_updates):

                sampled_tensordict = replay_buffer.sample(batch_size).to(device)

                loss_td = loss_module(sampled_tensordict)
                q_loss = loss_td["loss"]
                optimizer_actor.zero_grad()
                q_loss.backward()
                # grad_norm = torch.nn.utils.clip_grad_norm_(list(loss_module.parameters()), max_norm=0.5)
                optimizer_actor.step()
                target_net_updater.step()
                q_losses[j] = loss_td.select("loss").detach()

            q_losses = q_losses.apply(lambda x: x.float().mean(), batch_size=[])
            for key, value in q_losses.items():
                logger.log_scalar(key, value.item(), collected_frames)
            logger.log_scalar("epsilon", model_explore.eps, collected_frames)

        # update weights of the inference policy
        collector.update_policy_weights_()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")
