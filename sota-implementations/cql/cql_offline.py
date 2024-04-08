# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CQL Example.

This is a self-contained example of an offline CQL training script.

The helper functions are coded in the utils.py associated with this script.

"""
import time

import hydra
import numpy as np
import torch
import tqdm
from torchrl._utils import logger as torchrl_logger
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger

from utils import (
    log_metrics,
    make_continuous_cql_optimizer,
    make_continuous_loss,
    make_cql_model,
    make_environment,
    make_offline_replay_buffer,
)


@hydra.main(config_path="", config_name="offline_config", version_base="1.1")
def main(cfg: "DictConfig"):  # noqa: F821
    # Create logger
    exp_name = generate_exp_name("CQL-offline", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="cql_logging",
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
    device = torch.device(cfg.optim.device)

    # Create env
    train_env, eval_env = make_environment(cfg, cfg.logger.eval_envs)

    # Create replay buffer
    replay_buffer = make_offline_replay_buffer(cfg.replay_buffer)

    # Create agent
    model = make_cql_model(cfg, train_env, eval_env, device)

    # Create loss
    loss_module, target_net_updater = make_continuous_loss(cfg.loss, model)

    # Create Optimizer
    (
        policy_optim,
        critic_optim,
        alpha_optim,
        alpha_prime_optim,
    ) = make_continuous_cql_optimizer(cfg, loss_module)

    pbar = tqdm.tqdm(total=cfg.optim.gradient_steps)

    gradient_steps = cfg.optim.gradient_steps
    policy_eval_start = cfg.optim.policy_eval_start
    evaluation_interval = cfg.logger.eval_iter
    eval_steps = cfg.logger.eval_steps

    # Training loop
    start_time = time.time()
    for i in range(gradient_steps):
        pbar.update(1)
        # sample data
        data = replay_buffer.sample()
        # compute loss
        loss_vals = loss_module(data.clone().to(device))

        # official cql implementation uses behavior cloning loss for first few updating steps as it helps for some tasks
        if i >= policy_eval_start:
            actor_loss = loss_vals["loss_actor"]
        else:
            actor_loss = loss_vals["loss_actor_bc"]
        q_loss = loss_vals["loss_qvalue"]
        cql_loss = loss_vals["loss_cql"]

        q_loss = q_loss + cql_loss

        alpha_loss = loss_vals["loss_alpha"]
        alpha_prime_loss = loss_vals["loss_alpha_prime"]

        # update model
        alpha_loss = loss_vals["loss_alpha"]
        alpha_prime_loss = loss_vals["loss_alpha_prime"]

        alpha_optim.zero_grad()
        alpha_loss.backward()
        alpha_optim.step()

        policy_optim.zero_grad()
        actor_loss.backward()
        policy_optim.step()

        if alpha_prime_optim is not None:
            alpha_prime_optim.zero_grad()
            alpha_prime_loss.backward(retain_graph=True)
            alpha_prime_optim.step()

        critic_optim.zero_grad()
        # TODO: we have the option to compute losses independently retain is not needed?
        q_loss.backward(retain_graph=False)
        critic_optim.step()

        loss = actor_loss + q_loss + alpha_loss + alpha_prime_loss

        # log metrics
        to_log = {
            "loss": loss.item(),
            "loss_actor_bc": loss_vals["loss_actor_bc"].item(),
            "loss_actor": loss_vals["loss_actor"].item(),
            "loss_qvalue": q_loss.item(),
            "loss_cql": cql_loss.item(),
            "loss_alpha": alpha_loss.item(),
            "loss_alpha_prime": alpha_prime_loss.item(),
        }

        # update qnet_target params
        target_net_updater.step()

        # evaluation
        if i % evaluation_interval == 0:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_td = eval_env.rollout(
                    max_steps=eval_steps, policy=model[0], auto_cast_to_device=True
                )
            eval_reward = eval_td["next", "reward"].sum(1).mean().item()
            to_log["evaluation_reward"] = eval_reward

        log_metrics(logger, to_log, i)

    pbar.close()
    torchrl_logger.info(f"Training time: {time.time() - start_time}")


if __name__ == "__main__":
    main()
