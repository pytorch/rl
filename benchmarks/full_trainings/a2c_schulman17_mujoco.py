"""
A2C Benchmarks: Reproducing Experiments from Schulman et al. 2017
Asynchronous Actor Critic (A2C) Algorithm on MuJoCo Environments.
"""

import gym
import tqdm
import time
import torch.nn
import torch.optim
import numpy as np

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torchrl.collectors import SyncDataCollector
from torchrl.data import CompositeSpec, LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.envs.libs.gym import GymWrapper
from torchrl.envs import (
    RewardSum,
    DoubleToFloat,
    TransformedEnv,
    ExplorationType,
    set_exploration_type,
)
from torchrl.modules import (
    MLP,
    TanhNormal,
    ValueOperator,
    ProbabilisticActor,
)
from torchrl.objectives import A2CLoss
from torchrl.objectives.value.advantages import GAE
from torchrl.record.loggers import generate_exp_name, get_logger


# ====================================================================
# Environment utils
# --------------------------------------------------------------------


def make_env(env_name="HalfCheetah-v4", device="cpu"):
    env = gym.make(env_name)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
    env = GymWrapper(env, device=device)
    env = TransformedEnv(env)
    env.append_transform(RewardSum())
    env.append_transform(DoubleToFloat(in_keys=["observation"]))
    return env


# ====================================================================
# Model utils
# --------------------------------------------------------------------


class AddStateIndependentStd(torch.nn.Module):
    def __init__(self, num_outputs) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.zeros(num_outputs))

    def forward(self, *tensors: torch.Tensor) -> tuple[torch.Tensor, ...]:
        loc, *others = tensors

        # Get the number of dimensions in loc tensor
        num_dimensions = loc.dim()

        # Reshape logstd to match the number of dimensions in loc
        logstd = self.scale.view((1,) * (num_dimensions - self.scale.dim()) + self.scale.shape)

        # Clip logstd and convert to std
        scale = torch.zeros(loc.size()).to(loc.device) + logstd
        scale = scale.exp()

        return (loc, scale, *others)


def make_a2c_modules_state(proof_environment):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape

    # Define distribution class and kwargs
    num_outputs = proof_environment.action_spec.shape[-1]
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

    for layer in policy_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 1.0)
            layer.bias.data.zero_()

    policy_mlp = torch.nn.Sequential(
        policy_mlp,
        AddStateIndependentStd(proof_environment.action_spec.shape[-1])
    )

    # Add probabilistic sampling of the actions
    policy_module = ProbabilisticActor(
        TensorDictModule(
            module=policy_mlp,
            in_keys=["observation"],
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

    for layer in value_mlp.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.orthogonal_(layer.weight, 0.01)
            layer.bias.data.zero_()

    value_module = ValueOperator(
        value_mlp,
        in_keys=["observation"],
    )

    return policy_module, value_module


def make_a2c_models(env_name):
    proof_environment = make_env(env_name, device="cpu")
    actor, critic = make_a2c_modules_state(proof_environment)
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
        average_gae=True,
    )
    return advantage_module


def make_loss(actor_network, value_network):
    advantage_module = make_advantage_module(value_network)
    loss_module = A2CLoss(
        actor=actor_network,
        critic=value_network,
        loss_critic_type=loss_critic_type,
        entropy_coef=entropy_coef,
        critic_coef=critic_coef,
    )

    return loss_module, advantage_module


def make_logger(backend="csv"):
    exp_name = generate_exp_name("A2C", f"Atari_Schulman17_{env_name}")
    logger = get_logger(backend, logger_name="a2c", experiment_name=exp_name)
    return logger


if __name__ == "__main__":

    # Define paper hyperparameters
    device = "cpu" if not torch.cuda.is_available() else "cuda"
    env_name = "Ant-v3"
    frames_per_batch = 64
    mini_batch_size = 64
    total_frames = 1_000_000
    record_interval = 1_000_000  # check final performance
    gamma = 0.99
    gae_lambda = 0.95
    lr = 3e-4
    critic_coef = 0.25
    entropy_coef = 0.0
    loss_critic_type = "l2"
    logger_backend = "wandb"
    num_mini_batches = frames_per_batch // mini_batch_size
    total_network_updates = (total_frames // frames_per_batch)

    # Make the components
    actor, critic = make_a2c_models(env_name)
    actor, critic = actor.to(device), critic.to(device)
    collector = make_collector(env_name, actor, device)
    data_buffer = make_data_buffer()
    loss_module, adv_module = make_loss(actor, critic)
    logger = make_logger(logger_backend)
    test_env = make_env(env_name, device)
    test_env.eval()

    actor_optim = torch.optim.Adam(actor.parameters(), lr=lr, eps=1e-5)
    critic_optim = torch.optim.Adam(critic.parameters(), lr=lr, eps=1e-5)

    # Main loop
    collected_frames = 0
    num_network_updates = 0
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

        losses = TensorDict({}, batch_size=[num_mini_batches])

        # Compute GAE
        with torch.no_grad():
            data = adv_module(data)
        data_reshape = data.reshape(-1)

        # Update the data buffer
        data_buffer.extend(data_reshape)

        for i, batch in enumerate(data_buffer):

            # Linearly decrease the learning rate and clip epsilon
            alpha = 1 - (num_network_updates / total_network_updates)
            for g in actor_optim.param_groups:
                g['lr'] = lr * alpha
            for g in critic_optim.param_groups:
                g['lr'] = lr * alpha
            num_network_updates += 1

            # Get a data batch
            batch = batch.to(device)

            # Forward pass A2C loss
            loss = loss_module(batch)
            losses[i] = loss.select("loss_critic", "loss_objective").detach()
            critic_loss = loss["loss_critic"]
            actor_loss = loss["loss_objective"]

            # Backward pass
            actor_loss.backward()
            critic_loss.backward()

            # Update the networks
            actor_optim.step()
            critic_optim.step()
            actor_optim.zero_grad()
            critic_optim.zero_grad()

        losses = losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in losses.items():
            logger.log_scalar(key, value.item(), collected_frames)

        # Test logging
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if (collected_frames - frames_in_batch) // (record_interval < collected_frames // record_interval):
                actor.eval()
                test_rewards = []
                for i in range(10):
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
                logger.log_scalar("reward_test", test_rewards.mean(), collected_frames)
                actor.train()

        collector.update_policy_weights_()

    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")
