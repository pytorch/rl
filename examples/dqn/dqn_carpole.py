"""
DQN Benchmarks: CartPole-v1
"""

import hydra
import tqdm
import time
import torch.nn
import torch.optim
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import CompositeSpec, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs import RewardSum, DoubleToFloat, TransformedEnv, StepCounter
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
    env.append_transform(StepCounter())
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


@hydra.main(config_path=".", config_name="config_cartpole", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    device = "cpu" if not torch.cuda.is_available() else "cuda"

    # Make the components
    model = make_dqn_model(cfg.env.env_name)
    model_explore = EGreedyWrapper(
        policy=model,
        annealing_num_steps=cfg.collector.annealing_frames,
        eps_init=cfg.collector.eps_start,
        eps_end=cfg.collector.eps_end,
    ).to(device)

    # Create the collector
    collector = SyncDataCollector(
        make_env(cfg.env.env_name, device),
        policy=model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
    )

    # Create the replay buffer
    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=3,
        storage=LazyTensorStorage(
            max_size=cfg.buffer.buffer_size,
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
    target_net_updater = HardUpdate(loss_module, value_network_update_interval=cfg.loss.hard_update_freq)

    # Create the optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    # Create the logger
    exp_name = generate_exp_name("DQN", f"CartPole_{cfg.env.env_name}")
    logger = get_logger(cfg.logger.backend, logger_name="dqn", experiment_name=exp_name)

    # Main loop
    collected_frames = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    for i, data in enumerate(collector):

        # Train loging
        logger.log_scalar("q_values", (data["action_value"]*data["action"]).sum().item() / cfg.collector.frames_per_batch, collected_frames)
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            logger.log_scalar("reward_train", episode_rewards.mean().item(), collected_frames)
            logger.log_scalar("episode_length_train", episode_length.sum().item() / len(episode_length), collected_frames)

        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel()
        replay_buffer.extend(data.to(device))
        collected_frames += current_frames
        model_explore.step(current_frames)

        # optimization steps
        if collected_frames >= cfg.collector.init_random_frames:
            q_losses = TensorDict({}, batch_size=[cfg.collector.num_updates])
            for j in range(cfg.collector.num_updates):
                sampled_tensordict = replay_buffer.sample(cfg.buffer.batch_size)
                loss_td = loss_module(sampled_tensordict)
                q_loss = loss_td["loss"]
                optimizer.zero_grad()
                q_loss.backward()
                optimizer.step()
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


if __name__ == "__main__":
    main()