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
from tensordict import TensorDict

from torchrl._utils import logger as torchrl_logger
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger

from utils import (
    dump_video,
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
    device = cfg.optim.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = "cuda:0"
        else:
            device = "cpu"
    device = torch.device(device)

    # Create replay buffer
    replay_buffer = make_offline_replay_buffer(cfg.replay_buffer)

    # Create env
    train_env, eval_env = make_environment(
        cfg, train_num_envs=1, eval_num_envs=cfg.logger.eval_envs, logger=logger
    )

    # Create agent
    model = make_cql_model(cfg, train_env, eval_env, device)
    del train_env

    # Create loss
    loss_module, target_net_updater = make_continuous_loss(cfg.loss, model)

    # Create Optimizer
    (
        policy_optim,
        critic_optim,
        alpha_optim,
        alpha_prime_optim,
    ) = make_continuous_cql_optimizer(cfg, loss_module)

    gradient_steps = cfg.optim.gradient_steps
    policy_eval_start = cfg.optim.policy_eval_start
    evaluation_interval = cfg.logger.eval_iter
    eval_steps = cfg.logger.eval_steps

    def update(data, i):
        critic_optim.zero_grad()
        q_loss, metadata = loss_module.q_loss(data)
        cql_loss, cql_metadata = loss_module.cql_loss(data)
        q_loss = q_loss + cql_loss
        q_loss.backward()
        critic_optim.step()
        metadata.update(cql_metadata)

        policy_optim.zero_grad()
        if i >= policy_eval_start:
            actor_loss, actor_metadata = loss_module.actor_loss(data)
        else:
            actor_loss, actor_metadata = loss_module.actor_bc_loss(data)
        actor_loss.backward()
        policy_optim.step()
        metadata.update(actor_metadata)

        alpha_optim.zero_grad()
        alpha_loss, alpha_metadata = loss_module.alpha_loss(actor_metadata)
        alpha_loss.backward()
        alpha_optim.step()
        metadata.update(alpha_metadata)

        if alpha_prime_optim is not None:
            alpha_prime_optim.zero_grad()
            alpha_prime_loss, alpha_prime_metadata = loss_module.alpha_prime_loss(data)
            alpha_prime_loss.backward()
            alpha_prime_optim.step()
            metadata.update(alpha_prime_metadata)

        loss_vals = TensorDict(metadata)
        loss_vals["loss_qvalue"] = q_loss
        loss_vals["loss_cql"] = cql_loss
        loss_vals["loss_alpha"] = alpha_loss
        loss = actor_loss + q_loss + alpha_loss
        if alpha_prime_optim is not None:
            loss_vals["loss_alpha_prime"] = alpha_prime_loss
            loss = loss + alpha_prime_loss
        loss_vals["loss"] = loss

        return loss_vals.detach()

    if cfg.loss.compile:
        update = torch.compile(update, mode=cfg.loss.compile_mode)

    # Training loop
    start_time = time.time()
    pbar = tqdm.tqdm(range(gradient_steps))
    for i in pbar:
        # sample data
        data = replay_buffer.sample().to(device)
        loss_vals = update(data, i)

        # log metrics
        to_log = loss_vals.mean().to_dict()

        # update qnet_target params
        target_net_updater.step()

        # evaluation
        if i % evaluation_interval == 0:
            with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
                eval_td = eval_env.rollout(
                    max_steps=eval_steps, policy=model[0], auto_cast_to_device=True
                )
                eval_env.apply(dump_video)
            eval_reward = eval_td["next", "reward"].sum(1).mean().item()
            to_log["evaluation_reward"] = eval_reward

        log_metrics(logger, to_log, i)

    pbar.close()
    torchrl_logger.info(f"Training time: {time.time() - start_time}")
    if not eval_env.is_closed:
        eval_env.close()


if __name__ == "__main__":
    main()
