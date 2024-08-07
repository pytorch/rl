# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""TD3+BC Example.

This is a self-contained example of an offline RL TD3+BC training script.

The helper functions are coded in the utils.py associated with this script.

"""
import time

import hydra
import numpy as np
import torch
import tqdm
from torchrl._utils import logger as torchrl_logger

from torchrl.envs import set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger

from utils import (
    dump_video,
    log_metrics,
    make_environment,
    make_loss_module,
    make_offline_replay_buffer,
    make_optimizer,
    make_td3_agent,
)


@hydra.main(config_path="", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821
    set_gym_backend(cfg.env.library).set()

    # Create logger
    exp_name = generate_exp_name("TD3BC-offline", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="td3bc_logging",
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
    device = cfg.network.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # Creante env
    eval_env = make_environment(
        cfg,
        logger=logger,
    )

    # Create replay buffer
    replay_buffer = make_offline_replay_buffer(cfg.replay_buffer)

    # Create agent
    model, _ = make_td3_agent(cfg, eval_env, device)

    # Create loss
    loss_module, target_net_updater = make_loss_module(cfg.optim, model)

    # Create optimizer
    optimizer_actor, optimizer_critic = make_optimizer(cfg.optim, loss_module)

    gradient_steps = cfg.optim.gradient_steps
    evaluation_interval = cfg.logger.eval_iter
    eval_steps = cfg.logger.eval_steps
    delayed_updates = cfg.optim.policy_update_delay
    update_counter = 0
    pbar = tqdm.tqdm(range(gradient_steps))
    # Training loop
    start_time = time.time()
    for i in pbar:
        pbar.update(1)
        # Update actor every delayed_updates
        update_counter += 1
        update_actor = update_counter % delayed_updates == 0

        # Sample from replay buffer
        sampled_tensordict = replay_buffer.sample()
        if sampled_tensordict.device != device:
            sampled_tensordict = sampled_tensordict.to(device)
        else:
            sampled_tensordict = sampled_tensordict.clone()

        # Compute loss
        q_loss, *_ = loss_module.qvalue_loss(sampled_tensordict)

        # Update critic
        optimizer_critic.zero_grad()
        q_loss.backward()
        optimizer_critic.step()
        q_loss.item()

        to_log = {"q_loss": q_loss.item()}

        # Update actor
        if update_actor:
            actor_loss, actorloss_metadata = loss_module.actor_loss(sampled_tensordict)
            optimizer_actor.zero_grad()
            actor_loss.backward()
            optimizer_actor.step()

            # Update target params
            target_net_updater.step()

            to_log["actor_loss"] = actor_loss.item()
            to_log.update(actorloss_metadata)

        # evaluation
        if i % evaluation_interval == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_td = eval_env.rollout(
                    max_steps=eval_steps, policy=model[0], auto_cast_to_device=True
                )
                eval_env.apply(dump_video)
            eval_reward = eval_td["next", "reward"].sum(1).mean().item()
            to_log["evaluation_reward"] = eval_reward
        if logger is not None:
            log_metrics(logger, to_log, i)

    if not eval_env.is_closed:
        eval_env.close()
    pbar.close()
    torchrl_logger.info(f"Training time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
