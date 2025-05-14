# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""TD3+BC Example.

This is a self-contained example of an offline RL TD3+BC training script.

The helper functions are coded in the utils.py associated with this script.

"""
from __future__ import annotations

import warnings

import hydra
import numpy as np
import torch
import tqdm
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, timeit
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
def main(cfg: DictConfig):  # noqa: F821
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
    replay_buffer = make_offline_replay_buffer(cfg.replay_buffer, device=device)

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    # Create agent
    model, _ = make_td3_agent(cfg, eval_env, device)

    # Create loss
    loss_module, target_net_updater = make_loss_module(cfg.optim, model)

    # Create optimizer
    optimizer_actor, optimizer_critic = make_optimizer(cfg.optim, loss_module)

    def update(sampled_tensordict, update_actor):
        # Compute loss
        q_loss, *_ = loss_module.qvalue_loss(sampled_tensordict)

        # Update critic
        q_loss.backward()
        optimizer_critic.step()
        optimizer_critic.zero_grad(set_to_none=True)

        # Update actor
        if update_actor:
            actor_loss, actorloss_metadata = loss_module.actor_loss(sampled_tensordict)
            actor_loss.backward()
            optimizer_actor.step()
            optimizer_actor.zero_grad(set_to_none=True)

            # Update target params
            target_net_updater.step()
        else:
            actorloss_metadata = {}
            actor_loss = q_loss.new_zeros(())
        metadata = TensorDict(actorloss_metadata)
        metadata.set("q_loss", q_loss.detach())
        metadata.set("actor_loss", actor_loss.detach())
        return metadata

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)

    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)

    gradient_steps = cfg.optim.gradient_steps
    evaluation_interval = cfg.logger.eval_iter
    eval_steps = cfg.logger.eval_steps
    delayed_updates = cfg.optim.policy_update_delay
    pbar = tqdm.tqdm(range(gradient_steps))
    # Training loop
    for update_counter in pbar:
        timeit.printevery(num_prints=1000, total_count=gradient_steps, erase=True)

        # Update actor every delayed_updates
        update_actor = update_counter % delayed_updates == 0

        with timeit("rb - sample"):
            # Sample from replay buffer
            sampled_tensordict = replay_buffer.sample()

        with timeit("update"):
            torch.compiler.cudagraph_mark_step_begin()
            metadata = update(sampled_tensordict, update_actor).clone()

        metrics_to_log = {}
        if update_actor:
            metrics_to_log.update(metadata.to_dict())
        else:
            metrics_to_log.update(metadata.exclude("actor_loss").to_dict())

        # evaluation
        if update_counter % evaluation_interval == 0:
            with set_exploration_type(
                ExplorationType.DETERMINISTIC
            ), torch.no_grad(), timeit("eval"):
                eval_td = eval_env.rollout(
                    max_steps=eval_steps, policy=model[0], auto_cast_to_device=True
                )
                eval_env.apply(dump_video)
            eval_reward = eval_td["next", "reward"].sum(1).mean().item()
            metrics_to_log["evaluation_reward"] = eval_reward
        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, update_counter)

    if not eval_env.is_closed:
        eval_env.close()
    pbar.close()


if __name__ == "__main__":
    main()
