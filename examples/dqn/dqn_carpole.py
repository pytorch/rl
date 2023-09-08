"""
DQN Benchmarks: CartPole-v1
"""

import tqdm
import time
import torch.nn
import torch.optim
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import CompositeSpec, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import RewardSum, DoubleToFloat, TransformedEnv
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.modules import MLP, QValueActor, EGreedyWrapper
from torchrl.record.loggers import generate_exp_name, get_logger


# ====================================================================
# Environment utils
# --------------------------------------------------------------------

def make_env(env_name="CartPole-v1", device="cpu"):
    env = GymEnv(env_name, device=device)
    env = TransformedEnv(env)
    env.append_transform(RewardSum())
    env.append_transform(DoubleToFloat())
    return env

# ====================================================================
# Model utils
# --------------------------------------------------------------------


def make_dqn_modules(proof_environment):

    # Define input shape
    input_shape = proof_environment.observation_spec["observation"].shape
    env_specs = proof_environment.specs
    num_outputs = env_specs["input_spec", "full_action_spec", "action"].space.n
    action_spec = env_specs["input_spec", "full_action_spec", "action"]

    # Define Q-Value Module
    mlp = MLP(
        in_features=input_shape[-1],
        activation_class=torch.nn.ReLU,
        out_features=num_outputs,
        num_cells=[120, 84],
    )

    qvalue_module = QValueActor(
        module=mlp,
        spec=CompositeSpec(action=action_spec),
        in_keys=["observation"],
    )
    return qvalue_module


def make_dqn_model(env_name):
    proof_environment = make_env(env_name, device="cpu")
    qvalue_module = make_dqn_modules(proof_environment)
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
        prefetch=3,
):
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=prefetch,
        storage=LazyTensorStorage(
            max_size=buffer_size,
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
    exp_name = generate_exp_name("DQN", f"CartPole_{env_name}")
    logger = get_logger(backend, logger_name="dqn", experiment_name=exp_name)
    return logger


if __name__ == "__main__":

    device = "cpu" if not torch.cuda.is_available() else "cuda"
    env_name = "CartPole-v1"
    total_frames = 500_000
    record_interval = 500_000
    frames_per_batch = 10
    num_updates = 1
    buffer_size = 10_000
    init_random_frames = 10_000
    annealing_frames = 250_000
    gamma = 0.99
    lr = 2.5e-4
    batch_size = 128
    hard_update_freq = 125
    eps_end = 0.05
    logger_backend = "wandb"

    seed = 42
    torch.manual_seed(seed)

    # Make the components
    model = make_dqn_model(env_name)
    model_explore = EGreedyWrapper(model, annealing_num_steps=annealing_frames, eps_end=eps_end).to(device)
    collector = make_collector(env_name, model_explore, device)
    replay_buffer = make_replay_buffer(batch_size)
    loss_module, target_net_updater = make_loss_module(model)
    optimizer_actor = make_optimizer(loss_module)
    logger = make_logger(logger_backend)

    # Main loop
    collected_frames = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=total_frames)

    for i, data in enumerate(collector):

        # Train loging
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            logger.log_scalar("reward_train", episode_rewards.mean().item(), collected_frames)

        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel()
        replay_buffer.extend(data.to(device))
        collected_frames += current_frames
        model_explore.step(current_frames)

        # optimization steps
        if collected_frames >= init_random_frames:
            q_losses = TensorDict({}, batch_size=[num_updates])
            for j in range(num_updates):
                sampled_tensordict = replay_buffer.sample(batch_size).to(device)
                loss_td = loss_module(sampled_tensordict)
                q_loss = loss_td["loss"]
                optimizer_actor.zero_grad()
                q_loss.backward()
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
