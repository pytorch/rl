# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

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
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.envs import ExplorationType, set_exploration_type
from torchrl.modules import EGreedyWrapper
from torchrl.objectives import DQNLoss, HardUpdate
from torchrl.record.loggers import generate_exp_name, get_logger
from utils_cartpole import eval_model, make_dqn_model, make_env


@hydra.main(config_path=".", config_name="config_cartpole", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821

    device = "cpu" if not torch.cuda.device_count() else "cuda"

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

        log_info = {}
        sampling_time = time.time() - sampling_start
        pbar.update(data.numel())
        data = data.reshape(-1)
        current_frames = data.numel()
        replay_buffer.extend(data.to(device))
        collected_frames += current_frames
        model_explore.step(current_frames)

        # Get training rewards, episode lengths and q-values
        log_info.update(
            {
                "train/q_values": (data["action_value"] * data["action"]).sum().item()
                / cfg.collector.frames_per_batch,
            }
        )
        episode_rewards = data["next", "episode_reward"][data["next", "done"]]
        if len(episode_rewards) > 0:
            episode_length = data["next", "step_count"][data["next", "done"]]
            log_info.update(
                {
                    "train/episode_reward": episode_rewards.mean().item(),
                    "train/episode_length": episode_length.sum().item()
                                            / len(episode_length),
                }
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

        # Get training losses, epsilon and times
        training_time = time.time() - training_start
        q_losses = q_losses.apply(lambda x: x.float().mean(), batch_size=[])
        for key, value in q_losses.items():
            log_info.update({f"train/{key}": value.item()})
        log_info.update(
            {
                "train/epsilon": model_explore.eps,
                "train/sampling_time": sampling_time,
                "train/training_time": training_time,
            }
        )

        # Get evaluation rewards and eval time
        with torch.no_grad(), set_exploration_type(ExplorationType.MODE):
            if (
                collected_frames - cfg.collector.frames_per_batch
            ) // cfg.logger.test_interval < (
                collected_frames // cfg.logger.test_interval
            ):
                model.eval()
                eval_start = time.time()
                test_rewards = eval_model(model, test_env, cfg.logger.num_test_episodes)
                eval_time = time.time() - eval_start
                log_info.update(
                    {
                        "eval/reward": np.mean(test_rewards),
                        "eval/eval_time": eval_time,
                    }
                )
                model.train()

        # Log all the information
        if logger:
            for key, value in log_info.items():
                logger.log_scalar(key, value, collected_frames)

        # update weights of the inference policy
        collector.update_policy_weights_()
        sampling_start = time.time()

    collector.shutdown()
    end_time = time.time()
    execution_time = end_time - start_time
    print(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
