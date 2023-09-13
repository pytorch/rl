"""
DQN Benchmarks: CartPole-v1
"""

import time

import hydra
import numpy as np
import torch.nn
import torch.optim
import tqdm
from tensordict import TensorDict
from torchrl.collectors import SyncDataCollector
from torchrl.data import CompositeSpec, LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import (
    DoubleToFloat,
    ExplorationType,
    RewardSum,
    set_exploration_type,
    StepCounter,
    TransformedEnv,
)
from torchrl.envs.libs.gym import GymEnv
from torchrl.modules import EGreedyWrapper, MLP, QValueActor
from torchrl.objectives import DQNLoss, HardUpdate
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
        create_env_fn=make_env(cfg.env.env_name, device),
        policy=model_explore,
        frames_per_batch=cfg.collector.frames_per_batch,
        total_frames=cfg.collector.total_frames,
        device=device,
        storing_device=device,
        max_frames_per_traj=-1,
        init_random_frames=cfg.collector.init_random_frames,
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
    target_net_updater = HardUpdate(
        loss_module, value_network_update_interval=cfg.loss.hard_update_freq
    )

    # Create the optimizer
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    # Create the logger
    exp_name = generate_exp_name("DQN", f"CartPole_{cfg.env.env_name}")
    logger = get_logger(cfg.logger.backend, logger_name="dqn", experiment_name=exp_name)

    # Create the test environment
    test_env = make_env(cfg.env.env_name, device)

    # Main loop
    collected_frames = 0
    start_time = time.time()
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)
    sampling_start = time.time()

    for data in collector:

        sampling_time = time.time() - sampling_start
        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel()
        replay_buffer.extend(data.to(device))
        collected_frames += current_frames
        model_explore.step(current_frames)

        # Log training rewards, episode lengths and q-values
        logger.log_scalar(
            "train/q_values",
            (data["action_value"] * data["action"]).sum().item()
            / cfg.collector.frames_per_batch,
            collected_frames,
        )
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            logger.log_scalar(
                "train/reward", episode_rewards.mean().item(), collected_frames
            )
            logger.log_scalar(
                "train/episode_length",
                episode_length.sum().item() / len(episode_length),
                collected_frames,
            )

        # optimization steps
        q_losses = TensorDict({}, batch_size=[cfg.loss.num_updates])
        training_start = time.time()
        for j in range(cfg.loss.num_updates):
            sampled_tensordict = replay_buffer.sample(cfg.buffer.batch_size)
            loss_td = loss_module(sampled_tensordict)
            q_loss = loss_td["loss"]
            optimizer.zero_grad()
            q_loss.backward()
            optimizer.step()
            target_net_updater.step()
            q_losses[j] = loss_td.select("loss").detach()

        training_time = time.time() - training_start
        q_losses = q_losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in q_losses.items():
            logger.log_scalar("train/" + key, value.item(), collected_frames)
        logger.log_scalar("train/epsilon", model_explore.eps, collected_frames)
        logger.log_scalar("train/sampling_time", sampling_time, collected_frames)
        logger.log_scalar("train/training_time", training_time, collected_frames)

        # Test logging
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if (
                collected_frames - cfg.collector.frames_per_batch
            ) // cfg.logger.test_interval < (
                collected_frames // cfg.logger.test_interval
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
                    reward = td_test["next", "episode_reward"][td_test["next", "done"]]
                    test_rewards = np.append(test_rewards, reward.cpu().numpy())
                    del td_test
                logger.log_scalar("eval/reward", test_rewards.mean(), collected_frames)
                model.train()

        # update weights of the inference policy
        collector.update_policy_weights_()
        sampling_start = time.time()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
