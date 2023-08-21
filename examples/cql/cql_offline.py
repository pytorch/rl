# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CQL Example.

This is a self-contained example of an offline CQL training script.

The helper functions are coded in the utils.py associated with this script.

"""

import hydra
import numpy as np
import torch
import tqdm
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger

from utils import (
    make_cql_model,
    make_cql_optimizer,
    make_environment,
    make_loss,
    make_offline_replay_buffer,
)


@hydra.main(config_path=".", config_name="offline_config")
def main(cfg: "DictConfig"):  # noqa: F821
    exp_name = generate_exp_name("CQL-offline", cfg.env.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="cql_logging",
            experiment_name=exp_name,
            wandb_kwargs={"mode": cfg.logger.mode, "config": cfg},
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    device = torch.device(cfg.optim.device)

    # Make Env
    train_env, eval_env = make_environment(cfg, cfg.logger.eval_envs)

    # Make Buffer
    replay_buffer = make_offline_replay_buffer(cfg.replay_buffer)

    # Make Model
    model = make_cql_model(cfg, train_env, eval_env, device)

    # Make Loss
    loss_module, target_net_updater = make_loss(cfg.loss, model)

    # Make Optimizer
    policy_optim, critic_optim, alpha_optim, alpha_prime_optim = make_cql_optimizer(
        cfg.optim, loss_module
    )

    pbar = tqdm.tqdm(total=cfg.optim.gradient_steps)

    r0 = None
    l0 = None

    gradient_steps = cfg.optim.gradient_steps
    evaluation_interval = cfg.logger.eval_iter
    eval_steps = cfg.logger.eval_steps

    for i in range(gradient_steps):
        pbar.update(i)
        data = replay_buffer.sample()
        # loss
        loss_vals = loss_module(data)
        # backprop
        actor_loss = loss_vals["loss_actor"]
        q_loss = loss_vals["loss_qvalue"]
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
        q_loss.backward(retain_graph=False)
        critic_optim.step()

        loss = actor_loss + q_loss + alpha_loss + alpha_prime_loss

        target_net_updater.step()

        # evaluation
        if i % evaluation_interval == 0:
            with set_exploration_type(ExplorationType.MODE), torch.no_grad():
                eval_td = eval_env.rollout(
                    max_steps=eval_steps, policy=model[0], auto_cast_to_device=True
                )

        if r0 is None:
            r0 = eval_td["next", "reward"].sum(1).mean().item()
        if l0 is None:
            l0 = loss.item()

        for key, value in loss_vals.items():
            logger.log_scalar(key, value.item(), i)
        eval_reward = eval_td["next", "reward"].sum(1).mean().item()
        logger.log_scalar("evaluation_reward", eval_reward, i)

        pbar.set_description(
            f"loss: {loss.item(): 4.4f} (init: {l0: 4.4f}), evaluation_reward: {eval_reward: 4.4f} (init={r0: 4.4f})"
        )


if __name__ == "__main__":
    main()
