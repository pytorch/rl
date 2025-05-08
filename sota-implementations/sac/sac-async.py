# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""SAC Example.

This is a simple self-contained example of a SAC training script.

It supports state environments like MuJoCo.

The helper functions are coded in the utils.py associated with this script.
"""
from __future__ import annotations

import time

import warnings
from functools import partial

import hydra
import numpy as np
import torch
import torch.cuda
import tqdm
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    dump_video,
    log_metrics,
    make_collector_async,
    make_environment,
    make_loss_module,
    make_replay_buffer,
    make_sac_agent,
    make_sac_optimizer,
    make_train_environment,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.1", config_path="", config_name="config-async")
def main(cfg: DictConfig):  # noqa: F821
    device = cfg.network.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    # Create logger
    exp_name = generate_exp_name("SAC", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="sac_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    _, eval_env = make_environment(cfg, logger=logger)

    # Create agent
    model, exploration_policy = make_sac_agent(
        cfg, make_train_environment(cfg), eval_env, device
    )

    # Create SAC loss
    loss_module, target_net_updater = make_loss_module(cfg, model)

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"
        compile_mode_collector = "reduce-overhead"

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=device,
        shared=True,
        prefetch=0,
    )

    # TODO: Simplify this - ideally we'd like to share the uninitialized lazy tensor storage and fetch it once
    #  it's initialized
    replay_buffer.extend(make_train_environment(cfg).rollout(1).view(-1))
    replay_buffer.empty()

    # Create off-policy collector
    collector = make_collector_async(
        cfg,
        partial(make_train_environment, cfg),
        exploration_policy,
        compile_mode=compile_mode_collector,
        replay_buffer=replay_buffer,
    )

    # Create optimizers
    (
        optimizer_actor,
        optimizer_critic,
        optimizer_alpha,
    ) = make_sac_optimizer(cfg, loss_module)
    optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
    del optimizer_actor, optimizer_critic, optimizer_alpha

    def update(sampled_tensordict):
        # Compute loss
        loss_td = loss_module(sampled_tensordict)

        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        alpha_loss = loss_td["loss_alpha"]

        (actor_loss + q_loss + alpha_loss).sum().backward()
        optimizer.step()

        # Update qnet_target params
        target_net_updater.step()

        optimizer.zero_grad(set_to_none=True)
        return loss_td.detach()

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=2)

    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=10)

    # Main loop
    collected_frames = 0

    init_random_frames = cfg.collector.init_random_frames
    assert init_random_frames == 0

    prb = cfg.replay_buffer.prb
    update_freq = cfg.collector.update_freq

    eval_rollout_steps = cfg.env.max_episode_steps
    # TODO: customize this
    num_updates = 1000
    total_iter = 1_000_000
    pbar = tqdm.tqdm(total=total_iter * num_updates)

    while not replay_buffer.write_count:
        time.sleep(0.01)

    losses = TensorDict(batch_size=[num_updates])
    for i in range(total_iter * num_updates):
        timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)

        if i % update_freq == update_freq - 1:
            # Update weights of the inference policy
            collector.update_policy_weights_()

        pbar.update(1)

        collected_frames = replay_buffer.write_count

        # Optimization steps
        with timeit("train"):
            with timeit("rb - sample"):
                # Sample from replay buffer
                sampled_tensordict = replay_buffer.sample()

            with timeit("update"):
                torch.compiler.cudagraph_mark_step_begin()
                loss_td = update(sampled_tensordict).clone()
            losses[i % num_updates] = loss_td.select(
                "loss_actor", "loss_qvalue", "loss_alpha"
            )

            # Update priority
            if prb:
                replay_buffer.update_priority(sampled_tensordict)

        # Logging
        if i % num_updates == num_updates - 1:
            metrics_to_log = {}
            if collected_frames >= init_random_frames:
                losses_m = losses.mean()
                metrics_to_log["train/q_loss"] = losses_m.get("loss_qvalue")
                metrics_to_log["train/actor_loss"] = losses_m.get("loss_actor")
                metrics_to_log["train/alpha_loss"] = losses_m.get("loss_alpha")
                metrics_to_log["train/alpha"] = loss_td["alpha"]
                metrics_to_log["train/entropy"] = loss_td["entropy"]
                metrics_to_log["train/collected_frames"] = int(replay_buffer.write_count)
            # Log rewards in the buffer

            # Evaluation
            with set_exploration_type(
                ExplorationType.DETERMINISTIC
            ), torch.no_grad(), timeit("eval"):
                eval_rollout = eval_env.rollout(
                    eval_rollout_steps,
                    model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_env.apply(dump_video)
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                metrics_to_log["eval/reward"] = eval_reward
            if logger is not None:
                metrics_to_log.update(timeit.todict(prefix="time"))
                metrics_to_log["time/speed"] = pbar.format_dict["rate"]
                log_metrics(logger, metrics_to_log, collected_frames)

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()


if __name__ == "__main__":
    main()
