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
from tensordict.nn import CudaGraphModule

from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import group_optimizers
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
    if hasattr(eval_env, "start"):
        # To set the number of threads to the definitive value
        eval_env.start()

    # Create loss
    loss_module, target_net_updater = make_continuous_loss(cfg.loss, model)

    # Create Optimizer
    (
        policy_optim,
        critic_optim,
        alpha_optim,
        alpha_prime_optim,
    ) = make_continuous_cql_optimizer(cfg, loss_module)

    # Group optimizers
    optimizer = group_optimizers(
        policy_optim, critic_optim, alpha_optim, alpha_prime_optim
    )

    def update(data, policy_eval_start, iteration):
        loss_vals = loss_module(data.to(device))

        # official cql implementation uses behavior cloning loss for first few updating steps as it helps for some tasks
        actor_loss = torch.where(
            iteration >= policy_eval_start,
            loss_vals["loss_actor"],
            loss_vals["loss_actor_bc"],
        )
        q_loss = loss_vals["loss_qvalue"]
        cql_loss = loss_vals["loss_cql"]

        q_loss = q_loss + cql_loss
        loss_vals["q_loss"] = q_loss

        # update model
        alpha_loss = loss_vals["loss_alpha"]
        alpha_prime_loss = loss_vals["loss_alpha_prime"]
        if alpha_prime_loss is None:
            alpha_prime_loss = 0

        loss = actor_loss + q_loss + alpha_loss + alpha_prime_loss

        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # update qnet_target params
        target_net_updater.step()

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
        update = CudaGraphModule(update, warmup=50)

    pbar = tqdm.tqdm(total=cfg.optim.gradient_steps)

    gradient_steps = cfg.optim.gradient_steps
    policy_eval_start = cfg.optim.policy_eval_start
    evaluation_interval = cfg.logger.eval_iter
    eval_steps = cfg.logger.eval_steps

    # Training loop
    start_time = time.time()
    policy_eval_start = torch.tensor(policy_eval_start, device=device)
    for i in range(gradient_steps):
        pbar.update(1)
        # sample data
        with timeit("sample"):
            data = replay_buffer.sample()

        with timeit("update"):
            # compute loss
            i_device = torch.tensor(i, device=device)
            loss, loss_vals = update(
                data.to(device), policy_eval_start=policy_eval_start, iteration=i_device
            )

        # log metrics
        to_log = {
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
                        max_steps=eval_steps, policy=model[0], auto_cast_to_device=True
                    )
                    eval_env.apply(dump_video)
                eval_reward = eval_td["next", "reward"].sum(1).mean().item()
                to_log["evaluation_reward"] = eval_reward

        with timeit("log"):
            if i % 200 == 0:
                to_log.update(timeit.todict(prefix="time"))
            log_metrics(logger, to_log, i)
        if i % 200 == 0:
            timeit.print()
            timeit.erase()

    pbar.close()
    torchrl_logger.info(f"Training time: {time.time() - start_time}")
    if not eval_env.is_closed:
        eval_env.close()


if __name__ == "__main__":
    main()
