# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""CQL Example.

This is a self-contained example of a discrete offline CQL training script.

The helper functions are coded in the utils.py associated with this script.
"""
from __future__ import annotations

import warnings

import hydra
import numpy as np
import torch
import tqdm
from tensordict.nn import CudaGraphModule
from torchrl._utils import timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    dump_video,
    log_metrics,
    make_discrete_cql_optimizer,
    make_discrete_loss,
    make_discretecql_model,
    make_environment,
    make_offline_discrete_replay_buffer,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.1", config_path="", config_name="discrete_offline_config")
def main(cfg):  # noqa: F821
    device = cfg.optim.device
    if device in ("", None):
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    # Create logger
    exp_name = generate_exp_name("DiscreteCQL", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="discretecql_logging",
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
    if cfg.env.seed is not None:
        warnings.warn(
            "The seed in the environment config is deprecated. "
            "Please set the seed in the optim config instead."
        )

    # Create replay buffer
    replay_buffer = make_offline_discrete_replay_buffer(cfg.replay_buffer)

    # Create env
    train_env, eval_env = make_environment(
        cfg, train_num_envs=1, eval_num_envs=cfg.logger.eval_envs, logger=logger
    )

    # Create agent
    model, explore_policy = make_discretecql_model(cfg, train_env, eval_env, device)

    del train_env

    # Create loss
    loss_module, target_net_updater = make_discrete_loss(cfg.loss, model, device)

    # Create optimizers
    optimizer = make_discrete_cql_optimizer(cfg, loss_module)  # optimizer for CQL loss

    def update(data):

        # Compute loss components
        loss_vals = loss_module(data)

        q_loss = loss_vals["loss_qvalue"]
        cql_loss = loss_vals["loss_cql"]

        # Total loss = Q-learning loss + CQL regularization
        loss = q_loss + cql_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Soft update of target Q-network
        target_net_updater.step()

        # Detach to avoid keeping computation graph in logging
        return loss.detach(), loss_vals.detach()

    compile_mode = None
    if cfg.compile.compile:
        if cfg.compile.compile_mode not in (None, ""):
            compile_mode = cfg.compile.compile_mode
        elif cfg.compile.cudagraphs:
            compile_mode = "default"
        else:
            compile_mode = "reduce-overhead"
        update = torch.compile(update, mode=compile_mode)
    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule es experimental y puede llevar a resultados incorrectos silenciosamente. Úsalo con precaución.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, warmup=50)

    pbar = tqdm.tqdm(total=cfg.optim.gradient_steps)

    gradient_steps = cfg.optim.gradient_steps
    policy_eval_start = cfg.optim.policy_eval_start
    evaluation_interval = cfg.logger.eval_iter
    eval_steps = cfg.logger.eval_steps

    # Training loop
    policy_eval_start = torch.tensor(policy_eval_start, device=device)
    for i in range(gradient_steps):
        timeit.printevery(1000, gradient_steps, erase=True)
        pbar.update(1)
        # sample data
        with timeit("sample"):
            data = replay_buffer.sample()

        with timeit("update"):
            torch.compiler.cudagraph_mark_step_begin()
            loss, loss_vals = update(data.to(device))

        # log metrics
        metrics_to_log = {
            "loss": loss.cpu(),
            **loss_vals.cpu(),
        }

        # evaluation
        with timeit("log/eval"):
            if i % evaluation_interval == 0:
                with set_exploration_type(
                    ExplorationType.DETERMINISTIC
                ), torch.no_grad():
                    eval_td = eval_env.rollout(
                        max_steps=eval_steps,
                        policy=explore_policy,
                        auto_cast_to_device=True,
                    )
                    eval_env.apply(dump_video)

                # eval_td: matrix of shape: [num_episodes, max_steps, ...]
                eval_reward = (
                    eval_td["next", "reward"].sum(1).mean().item()
                )  # mean computed over the sum of rewards for each episode
                metrics_to_log["evaluation_reward"] = eval_reward

        with timeit("log"):
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, i)

    pbar.close()
    if not eval_env.is_closed:
        eval_env.close()


if __name__ == "__main__":
    main()
