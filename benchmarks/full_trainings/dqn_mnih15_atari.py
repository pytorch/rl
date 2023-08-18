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
    EnvCreator,
    StepCounter,
    ParallelEnv,
    ToTensorImage,
    DoubleToFloat,
    RewardClipping,
    TransformedEnv,
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

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset."""
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        self.noop_action = 0  # No-op is assumed to be action 0.
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self, **kwargs):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset(**kwargs)
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, *other = self.env.step(self.noop_action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
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


def make_base_env(env_name="BreakoutNoFrameskip-v4", device="cpu", is_test=False):
    env = gym.make(env_name)
    if not is_test:
        # env = NoopResetEnv(env, noop_max=30)
        env = EpisodicLifeEnv(env)
    env = GymWrapper(env, frame_skip=frame_skip, from_pixels=True, pixels_only=False, device=device)
    reader = default_info_dict_reader(["end_of_life"])
    env.set_info_dict_reader(reader)
    return env


def make_parallel_env(env_name, device, is_test=False):
    env = ParallelEnv(1, EnvCreator(lambda: make_base_env(env_name, device)))
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
    if not is_test:
        env.append_transform(VecNorm())
    return env

# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_dqn_modules_pixels(proof_environment):

    # Define input shape
    input_shape = proof_environment.observation_spec["pixels"].shape
    env_specs = proof_environment.specs
    num_outputs = env_specs["input_spec", "_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "_action_spec", "action"]

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
        activate_last_layer=True,
        out_features=num_outputs,
        num_cells=[512],
    )

    for layer in cnn.modules():
        if isinstance(layer, torch.nn.Conv2d):
            torch.nn.init.orthogonal_(layer.weight, gain=torch.nn.init.calculate_gain("relu"))
            layer.bias.data.zero_()

    for layer in mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    qvalue_module = QValueActor(
        module=torch.nn.Sequential(cnn, mlp),
        spec=CompositeSpec(action=action_spec),
        in_keys=["pixels"],
        safe=True,
        action_space="one-hot",
    )

    return qvalue_module


def make_dqn_model(env_name):

    proof_environment = make_parallel_env(env_name, device="cpu")
    qvalue_module = make_dqn_modules_pixels(proof_environment)

    with torch.no_grad():
        td = proof_environment.rollout(max_steps=100, break_when_any_done=False)
        td = qvalue_module(td)
        del td

    del proof_environment

    return qvalue_module


# ====================================================================
# Collector utils
# --------------------------------------------------------------------

def make_collector(env_name, policy, device):
    collector_class = SyncDataCollector
    collector = collector_class(
        make_parallel_env(env_name, device),
        policy,
        frames_per_batch=frames_per_batch,
        total_frames=total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )
    # collector.set_seed(seed)
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
    targ_net_updater = HardUpdate(
        dqn_loss, value_network_update_interval=hard_update_freq
    )
    return dqn_loss, targ_net_updater

# ====================================================================
# Other component utils
# --------------------------------------------------------------------


def make_optimizer(dqn_loss):
    # optimizer = torch.optim.RMSprop(dqn_loss.parameters(), lr=lr, alpha=0.95, eps=0.01)
    optimizer = torch.optim.Adam(dqn_loss.parameters(), lr=lr, eps=1e-6)
    return optimizer


def make_logger(backend="csv"):
    exp_name = generate_exp_name("DQN", f"Atari_mnih15_{env_name}")
    logger = get_logger(backend, logger_name="dqn", experiment_name=exp_name)
    return logger


if __name__ == "__main__":

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    env_name = "PongNoFrameskip-v4"
    frame_skip = 4
    total_frames = 40_000_000 // frame_skip
    record_interval = 40_000_000 // frame_skip  # Check final performance
    frames_per_batch = 4
    num_updates = 1
    buffer_size = 1_000_000 // frame_skip
    init_random_frames = 50_000
    annealing_frames = 100_000  # 1_000_000 // frame_skip
    gamma = 0.99
    lr = 2.5e-4
    batch_size = 32
    hard_update_freq = 10_000
    logger_backend = "wandb"

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Make the components
    model = make_dqn_model(env_name)
    model_explore = EGreedyWrapper(model, annealing_num_steps=annealing_frames).to(device)
    collector = make_collector(env_name, model_explore, device)
    replay_buffer = make_replay_buffer(batch_size)
    loss_module, target_net_updater = make_loss_module(model)
    optimizer_actor = make_optimizer(loss_module)
    logger = make_logger(logger_backend)
    test_env = make_parallel_env(env_name, device, is_test=True)
    test_env.eval()

    # Main loop
    collected_frames = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=total_frames)

    for i, data in enumerate(collector):

        # Train loging
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            logger.log_scalar("reward_train", episode_rewards.mean().item(), collected_frames)

        # Apply episodic end of life
        data["done"].copy_(data["end_of_life"])
        data["next", "done"].copy_(data["next", "end_of_life"])

        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel()
        replay_buffer.extend(data.to(device))
        collected_frames += current_frames * frame_skip

        # optimization steps
        if collected_frames >= init_random_frames:

            model_explore.step(current_frames)

            q_losses = TensorDict({}, batch_size=[num_updates])
            for j in range(num_updates):

                sampled_tensordict = replay_buffer.sample(batch_size).to(device)

                loss_td = loss_module(sampled_tensordict.to(device))
                q_loss = loss_td["loss"]
                optimizer_actor.zero_grad()
                q_loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(list(loss_module.parameters()), max_norm=0.5)
                optimizer_actor.step()
                target_net_updater.step()
                q_losses[j] = loss_td.select("loss").detach()

            q_losses = q_losses.apply(lambda x: x.float().mean(), batch_size=[])
            for key, value in q_losses.items():
                logger.log_scalar(key, value.item(), collected_frames)
            logger.log_scalar("epsilon", model_explore.eps, collected_frames)

        # Test logging
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if (collected_frames - frames_per_batch) // record_interval < (collected_frames // record_interval):
                model.eval()
                test_rewards = []
                for i in range(30):
                    td_test = test_env.rollout(
                        policy=model,
                        auto_reset=True,
                        auto_cast_to_device=True,
                        break_when_any_done=True,
                        max_steps=10_000_000,
                    )
                    reward = td_test["next", "episode_reward"][td_test["next", "done"]]
                    test_rewards = np.append(test_rewards, reward.cpu().numpy())
                    del td_test
                logger.log_scalar("reward_test", test_rewards.mean(), collected_frames)
                model.train()

        # update weights of the inference policy
        collector.update_policy_weights_()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")
