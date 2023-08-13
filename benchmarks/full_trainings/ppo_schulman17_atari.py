"""
PPO Benchmarks: Reproducing Experiments from Schulman et al. 2017
Proximal Policy Optimization (PPO) Algorithm on Atari Environments.
"""

import gym
import tqdm
import time
import random
import torch.nn
import torch.optim
import numpy as np

from tensordict import TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import CompositeSpec, LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.tensor_specs import DiscreteBox
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
    TanhNormal,
    ValueOperator,
    OneHotCategorical,
    ProbabilisticActor,
    ActorValueOperator,
)
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value.advantages import GAE
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
        self.lives = self.env.unwrapped.ale.lives()
        return self.env.reset(**kwargs)


def make_base_env(env_name="BreakoutNoFrameskip-v4", device="cpu", is_test=False):
    env = gym.make(env_name)
    if not is_test:
        env = NoopResetEnv(env, noop_max=30)
        env = EpisodicLifeEnv(env)
    env = GymWrapper(env, frame_skip=frame_skip, from_pixels=True, pixels_only=False, device=device)
    reader = default_info_dict_reader(["end_of_life"])
    env.set_info_dict_reader(reader)
    return env


def make_parallel_env(env_name, device, is_test=False):
    num_envs = 8
    env = ParallelEnv(num_envs, EnvCreator(lambda: make_base_env(env_name, device)))
    env = TransformedEnv(env)
    env.append_transform(ToTensorImage())
    env.append_transform(GrayScale())
    env.append_transform(Resize(84, 84))
    env.append_transform(CatFrames(N=4, dim=-3))
    env.append_transform(RewardSum())
    if not is_test:
        env.append_transform(StepCounter(max_steps=4500))
        env.append_transform(RewardClipping(-1, 1))
    env.append_transform(DoubleToFloat())
    env.append_transform(VecNorm(in_keys=["pixels"]))
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------

def make_ppo_modules_pixels(proof_environment):

    # Define input shape
    input_shape = proof_environment.observation_spec["pixels"].shape

    # Define distribution class and kwargs
    if isinstance(proof_environment.action_spec.space, DiscreteBox):
        num_outputs = proof_environment.action_spec.space.n
        distribution_class = OneHotCategorical
        distribution_kwargs = {}
    else:  # is ContinuousBox
        num_outputs = proof_environment.action_spec.shape
        distribution_class = TanhNormal
        distribution_kwargs = {
            "min": proof_environment.action_spec.space.minimum,
            "max": proof_environment.action_spec.space.maximum,
        }

    # Define input keys
    in_keys = ["pixels"]

    # Define a shared Module and TensorDictModule (CNN + MLP)
    common_cnn = ConvNet(
        activation_class=torch.nn.ReLU,
        num_cells=[32, 64, 64],
        kernel_sizes=[8, 4, 3],
        strides=[4, 2, 1],
    )
    common_cnn_output = common_cnn(torch.ones(input_shape))
    common_mlp = MLP(
        in_features=common_cnn_output.shape[-1],
        activation_class=torch.nn.ReLU,
        activate_last_layer=True,
        out_features=512,
        num_cells=[],
    )
    common_mlp_output = common_mlp(common_cnn_output)

    # Define shared net as TensorDictModule
    common_module = TensorDictModule(
        module=torch.nn.Sequential(common_cnn, common_mlp),
        in_keys=in_keys,
        out_keys=["common_features"],
    )

    # Define on head for the policy
    policy_net = MLP(
        in_features=common_mlp_output.shape[-1],
        out_features=num_outputs,
        activation_class=torch.nn.ReLU,
        num_cells=[],
    )
    policy_module = TensorDictModule(
        module=policy_net,
        in_keys=["common_features"],
        out_keys=["logits"],
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        policy_module,
        in_keys=["logits"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # Define another head for the value
    value_net = MLP(
        activation_class=torch.nn.ReLU,
        in_features=common_mlp_output.shape[-1],
        out_features=1,
        num_cells=[],
    )
    value_module = ValueOperator(
        value_net,
        in_keys=["common_features"],
    )

    return common_module, policy_module, value_module


def make_ppo_models(env_name):

    proof_environment = make_parallel_env(env_name, device="cpu")
    common_module, policy_module, value_module = make_ppo_modules_pixels(
        proof_environment
    )

    # Wrap modules in a single ActorCritic operator
    actor_critic = ActorValueOperator(
        common_operator=common_module,
        policy_operator=policy_module,
        value_operator=value_module,
    )

    with torch.no_grad():
        td = proof_environment.rollout(max_steps=100, break_when_any_done=False)
        td = actor_critic(td)
        del td

    actor = actor_critic.get_policy_operator()
    critic = actor_critic.get_value_operator()
    critic_head = actor_critic.get_value_head()

    del proof_environment

    return actor, critic, critic_head


# ====================================================================
# Other component utils
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
    return collector


def make_data_buffer():
    sampler = SamplerWithoutReplacement()
    return TensorDictReplayBuffer(
        storage=LazyMemmapStorage(frames_per_batch),
        sampler=sampler,
        batch_size=mini_batch_size,
    )


def make_advantage_module(value_network):
    advantage_module = GAE(
        gamma=gamma,
        lmbda=gae_lambda,
        value_network=value_network,
        average_gae=False,
    )
    return advantage_module


def make_loss(actor_network, value_network, value_head):
    advantage_module = make_advantage_module(value_network)
    loss_module = ClipPPOLoss(
        actor=actor_network,
        critic=value_head,
        clip_epsilon=clip_epsilon,
        loss_critic_type=loss_critic_type,
        entropy_coef=entropy_coef,
        critic_coef=critic_coef,
        normalize_advantage=True,
    )
    return loss_module, advantage_module


def make_optim(loss_module):
    optim = torch.optim.Adam(
        loss_module.parameters(),
        lr=lr,
        weight_decay=0.0,
        eps=1e-6,
    )
    return optim


def make_logger(backend="csv"):
    exp_name = generate_exp_name("PPO", f"Atari_Schulman17_{env_name}")
    logger = get_logger(backend, logger_name="ppo", experiment_name=exp_name)
    return logger


if __name__ == "__main__":

    # Define paper hyperparameters
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    env_name = "BreakoutNoFrameskip-v4"
    frame_skip = 4
    frames_per_batch = 4096 // frame_skip
    mini_batch_size = 1024 // frame_skip
    total_frames = 40_000_000 // frame_skip
    gamma = 0.99
    gae_lambda = 0.95
    lr = 2.5e-4
    ppo_epochs = 3
    critic_coef = 1.0
    entropy_coef = 0.01
    clip_epsilon = 0.1
    loss_critic_type = "l2"
    logger_backend = "wandb"
    num_mini_batches = frames_per_batch // mini_batch_size
    total_network_updates = (total_frames // frames_per_batch) * ppo_epochs * num_mini_batches

    # Make the components
    actor, critic, critic_head = make_ppo_models(env_name)
    actor, critic, critic_head = actor.to(device), critic.to(device), critic_head.to(device)
    collector = make_collector(env_name, actor, device)
    data_buffer = make_data_buffer()
    loss_module, adv_module = make_loss(actor, critic, critic_head)
    optim = make_optim(loss_module)
    logger = make_logger(logger_backend)
    test_env = make_parallel_env(env_name, device, is_test=True)
    test_env.eval()

    # Main loop
    collected_frames = 0
    num_network_updates = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=total_frames)

    for data in collector:

        # Apply episodic end of life
        data["done"].copy_(data["end_of_life"])
        data["next", "done"].copy_(data["next", "end_of_life"])

        frames_in_batch = data.numel()
        collected_frames += frames_in_batch * frame_skip
        pbar.update(data.numel())

        # Train loging
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            logger.log_scalar("reward_train", episode_rewards.mean().item(), collected_frames)

        losses = TensorDict({}, batch_size=[ppo_epochs, num_mini_batches])
        for j in range(ppo_epochs):

            # Compute GAE
            with torch.no_grad():
                data = adv_module(data)
            data_reshape = data.reshape(-1)

            # Update the data buffer
            data_buffer.extend(data_reshape)

            for i, batch in enumerate(data_buffer):

                # Linearly decrease the learning rate and clip epsilon
                alpha = 1 - (num_network_updates / total_network_updates)
                for g in optim.param_groups:
                    g['lr'] = lr * alpha
                loss_module.clip_epsilon.copy_(clip_epsilon * alpha)
                num_network_updates += 1

                # Get a data batch
                batch = batch.to(device)

                # Forward pass PPO loss
                loss = loss_module(batch)
                losses[j, i] = loss.select("loss_critic", "loss_entropy", "loss_objective").detach()
                loss_sum = loss["loss_critic"] + loss["loss_objective"] + loss["loss_entropy"]

                # Backward pass
                loss_sum.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    list(loss_module.parameters()), max_norm=0.5
                )

                # Update the networks
                optim.step()
                optim.zero_grad()

        losses = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses.items():
            logger.log_scalar(key, value.item(), collected_frames)
        logger.log_scalar("lr", alpha * lr, collected_frames)
        logger.log_scalar("clip_epsilon", alpha * clip_epsilon, collected_frames)

        # Test logging
        record_interval = 1_000_000
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if (collected_frames - frames_in_batch) // record_interval < collected_frames // record_interval:
                actor.eval()
                test_rewards = []
                for i in range(3):
                    td_test = test_env.rollout(
                        policy=actor,
                        auto_cast_to_device=True,
                        max_steps=10_000_000,
                    )
                    reward = td_test["next", "episode_reward"][td_test["next", "done"]]
                    test_rewards = np.append(test_rewards, reward.cpu().numpy())
                    del td_test
                logger.log_scalar("reward_test", test_rewards.mean(), collected_frames)
                actor.train()

        collector.update_policy_weights_()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")
