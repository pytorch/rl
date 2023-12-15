# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""IQL Example.

This is a self-contained example of an offline IQL training script.

The helper functions are coded in the utils.py associated with this script.

"""
import time

import hydra
import numpy as np
import torch
import tqdm

from torchrl.envs import set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger

from utils import (
    log_metrics,
    make_environment,
    make_iql_model,
    make_iql_optimizer,
    make_loss,
    make_offline_replay_buffer,
)


@set_gym_backend("gym")
@hydra.main(config_path=".", config_name="offline_config")
def main(cfg: "DictConfig"):  # noqa: F821

    # Create logger
    exp_name = generate_exp_name("IQL-offline", cfg.env.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="iql_logging",
            experiment_name=exp_name,
            wandb_kwargs={"mode": cfg.logger.mode, "config": cfg},
        )

    # Set seeds
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    device = torch.device(cfg.optim.device)

    # Creante env
    train_env, eval_env = make_environment(cfg, cfg.logger.eval_envs)

    # Create replay buffer
    replay_buffer = make_offline_replay_buffer(cfg.replay_buffer)

    # Create agent
    model = make_iql_model(cfg, train_env, eval_env, device)

    # Create loss
    loss_module, target_net_updater = make_loss(cfg.loss, model)

    # Create optimizer
    optimizer = make_iql_optimizer(cfg.optim, loss_module)

    pbar = tqdm.tqdm(total=cfg.optim.gradient_steps)

    gradient_steps = cfg.optim.gradient_steps
    evaluation_interval = cfg.logger.eval_iter
    eval_steps = cfg.logger.eval_steps

    # Training loop
    start_time = time.time()
    for i in range(gradient_steps):
        pbar.update(i)
        # sample data
        data = replay_buffer.sample()
        # compute loss
        loss_vals = loss_module(data.clone().to(device))

        actor_loss = loss_vals["loss_actor"]
        q_loss = loss_vals["loss_qvalue"]
        value_loss = loss_vals["loss_value"]
        loss_val = actor_loss + q_loss + value_loss

        # update model
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()
        target_net_updater.step()

        # log metrics
        to_log = {
            "loss_actor": actor_loss.item(),
            "loss_qvalue": q_loss.item(),
            "loss_value": value_loss.item(),
        }

        # evaluation
        if i % evaluation_interval == 0:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_td = eval_env.rollout(
                    max_steps=eval_steps, policy=model[0], auto_cast_to_device=True
                )
            eval_reward = eval_td["next", "reward"].sum(1).mean().item()
            to_log["evaluation_reward"] = eval_reward
        if logger is not None:
            log_metrics(logger, to_log, i)

    pbar.close()
    print(f"Training time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
