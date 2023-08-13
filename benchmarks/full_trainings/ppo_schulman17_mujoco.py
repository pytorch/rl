"""
PPO Benchmarks: Reproducing Experiments from Schulman et al. 2017
Proximal Policy Optimization (PPO) Algorithm on MuJoCo Environments.
"""

import gym
import tqdm
import time
import torch.nn
import torch.optim
import numpy as np

from tensordict import TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import CompositeSpec, LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import (
    VecNorm,
    RewardSum,
    CatTensors,
    StepCounter,
    RewardScaling,
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


def make_env(env_name="HalfCheetah-v4", device="cpu", state_dict=None, is_test=False):
    env = gym.make(env_name)
    env = GymWrapper(env, device=device)  # TODO: testing
    env = TransformedEnv(env)
    # env.append_transform(RewardScaling(0.0, reward_scaling))
    selected_keys = [key for key in env.observation_spec.keys(True, True)]
    env.append_transform(CatTensors(in_keys=selected_keys, out_key="observation_vector"))
    env.append_transform(RewardSum())
    env.append_transform(StepCounter())
    env.append_transform(VecNorm(in_keys=["observation_vector"]))  # TODO: testing
    env.append_transform(DoubleToFloat(in_keys=["observation_vector"]))
    env.append_transform(RewardScaling(loc=1.0, scale=1.0))
    if not is_test:
        env.append_transform(RewardClipping(-10.0, 10.0))
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------

def make_ppo_modules_state(proof_environment):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation_vector"].shape

    # Define distribution class and kwargs
    num_outputs = proof_environment.action_spec.shape[-1] * 2
    distribution_class = TanhNormal
    distribution_kwargs = {
        "min": proof_environment.action_spec.space.minimum,
        "max": proof_environment.action_spec.space.maximum,
        "tanh_loc": False,
    }

    policy_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=num_outputs,
        num_cells=[64, 64],
    )
    policy_mlp = torch.nn.Sequential(
        policy_mlp, NormalParamExtractor(scale_lb=1e-2)
    )
    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation_vector"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        spec=CompositeSpec(action=proof_environment.action_spec),
        safe=True,
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    value_mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.Tanh,
        out_features=1,
        num_cells=[64, 64],
    )
    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation_vector"],
    )

    with torch.no_grad():
        td = proof_environment.rollout(max_steps=100, break_when_any_done=False)
        td = policy_module(td)
        td = value_module(td)
        del td

    return policy_module, value_module


def make_ppo_models(env_name):
    proof_environment = make_env(env_name, device="cpu")
    actor, critic = make_ppo_modules_state(proof_environment)
    return actor, critic


# ====================================================================
# Other component utils
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


def make_loss(actor_network, value_network):
    advantage_module = make_advantage_module(value_network)
    loss_module = ClipPPOLoss(
        actor=actor_network,
        critic=value_network,
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
        eps=1e-5,
    )
    return optim


def make_logger(backend="csv"):
    exp_name = generate_exp_name("PPO", f"Atari_Schulman17_{env_name}")
    logger = get_logger(backend, logger_name="ppo", experiment_name=exp_name)
    return logger


if __name__ == "__main__":

    # Define paper hyperparameters
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    env_name = "Swimmer-v2"
    frames_per_batch = 2048
    mini_batch_size = 64
    total_frames = 1_000_000
    gamma = 0.99
    gae_lambda = 0.95
    lr = 3e-4
    ppo_epochs = 10
    critic_coef = 0.5
    entropy_coef = 0.0
    clip_epsilon = 0.2
    loss_critic_type = "l2"
    logger_backend = "wandb"
    num_mini_batches = frames_per_batch // mini_batch_size
    total_network_updates = (total_frames // frames_per_batch) * ppo_epochs * num_mini_batches

    # Make the components
    actor, critic = make_ppo_models(env_name)
    actor, critic = actor.to(device), critic.to(device)
    collector = make_collector(env_name, actor, device)
    data_buffer = make_data_buffer()
    loss_module, adv_module = make_loss(actor, critic)
    optim = make_optim(loss_module)
    logger = make_logger(logger_backend)
    test_env = make_env(env_name, device, is_test=True)
    test_env.eval()

    # Main loop
    collected_frames = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=total_frames)

    for data in collector:

        frames_in_batch = data.numel()
        collected_frames += frames_in_batch
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

        # Test logging
        record_interval = 100_000
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if (collected_frames - frames_in_batch) // record_interval < collected_frames // record_interval:
                actor.eval()
                test_rewards = []
                for i in range(10):
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
