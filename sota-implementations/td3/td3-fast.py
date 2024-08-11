# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""TD3 Example.

This is a simple self-contained example of a TD3 training script.

It supports state environments like MuJoCo.

The helper functions are coded in the utils.py associated with this script.
"""
import time

import hydra
import numpy as np
import torch
import torch.cuda
import tqdm
from torchrl._utils import logger as torchrl_logger
from torchrl.data.utils import CloudpickleWrapper

from torchrl.envs.utils import ExplorationType, set_exploration_type

from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    log_metrics,
    make_async_collector,
    make_environment,
    make_loss_module,
    make_optimizer,
    make_replay_buffer,
    make_simple_environment,
    make_td3_agent,
)


@hydra.main(version_base="1.1", config_path="", config_name="config-fast")
def main(cfg: "DictConfig"):  # noqa: F821
    device = cfg.network.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    # Create logger
    exp_name = generate_exp_name("TD3", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="td3_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Set seeds
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg, logger=logger)

    # Create agent
    model, exploration_policy = make_td3_agent(cfg, train_env, eval_env, device)

    # Create TD3 loss
    loss_module, target_net_updater = make_loss_module(cfg, model)

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=cfg.replay_buffer.device if cfg.replay_buffer.device else device,
        prefetch=0,
        mmap=False,
    )
    reshape = CloudpickleWrapper(lambda td: td.reshape(-1))
    replay_buffer.append_transform(reshape, invert=True)

    # Create off-policy collector
    envname = cfg.env.name
    task = cfg.env.task
    library = cfg.env.library
    seed = cfg.env.seed
    max_episode_steps = cfg.env.max_episode_steps
    collector = make_async_collector(
        cfg,
        lambda: make_simple_environment(
            envname, task, library, seed, max_episode_steps
        ),
        exploration_policy,
        replay_buffer,
    )

    # Create optimizers
    optimizer_actor, optimizer_critic = make_optimizer(cfg, loss_module)

    # Main loop
    start_time = time.time()
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(
        max(1, cfg.collector.env_per_collector)
        * cfg.collector.frames_per_batch
        * cfg.optim.utd_ratio
    )
    delayed_updates = cfg.optim.policy_update_delay
    prb = cfg.replay_buffer.prb
    update_counter = 0

    sampling_start = time.time()
    current_frames = cfg.collector.frames_per_batch
    update_actor = False

    test_env = make_simple_environment(envname, task, library, seed, max_episode_steps)
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        reward = test_env.rollout(10_000, exploration_policy)["next", "reward"].mean()
        print(f"reward before training: {reward: 4.4f}")

    # loss_module.value_loss = torch.compile(
    #     loss_module.value_loss, mode="reduce-overhead"
    # )
    # loss_module.actor_loss = torch.compile(
    #     loss_module.actor_loss, mode="reduce-overhead"
    # )

    def train_update(sampled_tensordict):
        # Compute loss
        q_loss, *_ = loss_module.value_loss(sampled_tensordict)

        # Update critic
        optimizer_critic.zero_grad()
        q_loss.backward()
        optimizer_critic.step()
        q_losses.append(q_loss.item())

        # Update actor
        if update_actor:
            actor_loss, *_ = loss_module.actor_loss(sampled_tensordict)
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            actor_losses.append(actor_loss.item())

            # Update target params
            target_net_updater.step()

    train_update_cuda = None
    g = torch.cuda.CUDAGraph()

    for _ in collector:
        sampling_time = time.time() - sampling_start
        exploration_policy[1].step(current_frames)

        # Update weights of the inference policy
        collector.update_policy_weights_()

        pbar.update(current_frames)

        # Add to replay buffer
        collected_frames += current_frames

        # Optimization steps
        training_start = time.time()

        if collected_frames >= init_random_frames:
            (
                actor_losses,
                q_losses,
            ) = ([], [])
            for _ in range(num_updates):

                # Update actor every delayed_updates
                update_counter += 1
                update_actor = update_counter % delayed_updates == 0

                # Sample from replay buffer
                sampled_tensordict = replay_buffer.sample()
                if sampled_tensordict.device != device:
                    sampled_tensordict = sampled_tensordict.to(
                        device, non_blocking=True
                    )
                else:
                    sampled_tensordict = sampled_tensordict.clone()

                if train_update_cuda is None:
                    static_sample = sampled_tensordict
                    with torch.cuda.graph(g):
                        train_update(static_sample)

                    def train_update_cuda(x):
                        static_sample.copy_(x)
                        g.replay()
                else:
                    train_update_cuda(sampled_tensordict)

                # Update priority
                if prb:
                    replay_buffer.update_priority(sampled_tensordict)

        training_time = time.time() - training_start

        # Logging
        metrics_to_log = {}
        if collected_frames >= init_random_frames:
            metrics_to_log["train/q_loss"] = np.mean(q_losses)
            if update_actor:
                metrics_to_log["train/a_loss"] = np.mean(actor_losses)
            metrics_to_log["train/sampling_time"] = sampling_time
            metrics_to_log["train/training_time"] = training_time

        if logger is not None:
            log_metrics(logger, metrics_to_log, collected_frames)
        sampling_start = time.time()

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()

    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        reward = test_env.rollout(10_000, exploration_policy)["next", "reward"].mean()
        print(f"reward before training: {reward: 4.4f}")
    test_env.close()

    end_time = time.time()
    execution_time = end_time - start_time
    torchrl_logger.info(f"Training took {execution_time:.2f} seconds to finish")


if __name__ == "__main__":
    main()
