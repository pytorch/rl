# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Async SAC Example.

WARNING: This isn't a SOTA implementation but a rudimentary implementation of SAC where inference
and training are entirely decoupled. It can achieve a 20x speedup if compile and cudagraph are used.
Two GPUs are required for this script to run.
The API is currently being perfected, and contributions are welcome (as usual!) - see the TODOs in this script.

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
import tensordict
import torch
import torch.cuda
import tqdm
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, logger as torchrl_logger, timeit
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
tensordict.nn.functional_modules._exclude_td_from_pytree().set()


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
            logger_name="async_sac_logging",
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

    # TODO: This should be simplified. We need to create the policy on cuda:1 directly because of the bounds
    #  of the TanhDistribution which cannot be sent to cuda:1 within the distribution construction (ie, the
    #  distribution kwargs need to have access to the low / high values on the right device for compile and
    #  cudagraph to work).
    # Create agent
    dummy_train_env = make_train_environment(cfg)
    model, _ = make_sac_agent(cfg, dummy_train_env, eval_env, device)
    _, exploration_policy = make_sac_agent(cfg, dummy_train_env, eval_env, "cuda:1")
    dummy_train_env.close(raise_if_closed=False)
    del dummy_train_env
    exploration_policy.load_state_dict(model[0].state_dict())

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
        compile_mode_collector = compile_mode  # "reduce-overhead"

    # TODO: enabling prefetch for mp RBs would speed up sampling which is currently responsible for
    #  half of the compute time on the trainer side.
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

    # Create off-policy collector and start it
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

    cfg.compile.cudagraphs
    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=10)

    # Main loop
    init_random_frames = cfg.collector.init_random_frames

    prb = cfg.replay_buffer.prb
    update_freq = cfg.collector.update_freq

    eval_rollout_steps = cfg.env.max_episode_steps
    log_freq = cfg.logger.log_freq

    # TODO: customize this
    num_updates = 1000
    total_iter = 1000
    pbar = tqdm.tqdm(total=total_iter * num_updates)
    params = TensorDict.from_module(model[0]).data

    # Wait till we have enough data to start training
    while replay_buffer.write_count <= init_random_frames:
        time.sleep(0.01)

    losses = []
    for i in range(total_iter * num_updates):
        timeit.printevery(
            num_prints=total_iter * num_updates // log_freq,
            total_count=total_iter * num_updates,
            erase=True,
        )

        if (i % update_freq) == 0:
            # Update weights of the inference policy
            torchrl_logger.info("Updating weights")
            collector.update_policy_weights_(params)

        pbar.update(1)

        # Optimization steps
        with timeit("train"):
            with timeit("train - rb - sample"):
                # Sample from replay buffer
                sampled_tensordict = replay_buffer.sample()

            with timeit("train - update"):
                torch.compiler.cudagraph_mark_step_begin()
                loss_td = update(sampled_tensordict).clone()
            losses.append(loss_td.select("loss_actor", "loss_qvalue", "loss_alpha"))

            # Update priority
            if prb:
                replay_buffer.update_priority(sampled_tensordict)

        # Logging
        if (i % log_freq) == (log_freq - 1):
            torchrl_logger.info("Logging")
            collected_frames = replay_buffer.write_count
            metrics_to_log = {}
            if collected_frames >= init_random_frames:
                losses_m = torch.stack(losses).mean()
                losses = []
                metrics_to_log["train/q_loss"] = losses_m.get("loss_qvalue")
                metrics_to_log["train/actor_loss"] = losses_m.get("loss_actor")
                metrics_to_log["train/alpha_loss"] = losses_m.get("loss_alpha")
                metrics_to_log["train/alpha"] = loss_td["alpha"]
                metrics_to_log["train/entropy"] = loss_td["entropy"]
                metrics_to_log["train/collected_frames"] = int(collected_frames)

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
            torchrl_logger.info(f"Logs: {metrics_to_log}")
            if logger is not None:
                metrics_to_log.update(timeit.todict(prefix="time"))
                metrics_to_log["time/speed"] = pbar.format_dict["rate"]
                log_metrics(logger, metrics_to_log, collected_frames)

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()


if __name__ == "__main__":
    main()
