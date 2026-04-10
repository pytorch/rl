# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Diffusion BC Example.

This is a self-contained example of an offline Diffusion Behavioural Cloning
training script.

It trains a DiffusionActor using the ε-prediction (noise-prediction)
denoising loss from Diffusion Policy (Chi et al., RSS 2023) on offline
demonstration data from D4RL.

The helper functions are coded in the utils.py associated with this script.

"""
from __future__ import annotations

import warnings

import hydra
import numpy as np
import torch
import tqdm
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, get_available_device, timeit
from torchrl.envs import set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    dump_video,
    log_metrics,
    make_diffusion_actor,
    make_environment,
    make_loss_module,
    make_offline_replay_buffer,
    make_optimizer,
)


@hydra.main(config_path="", config_name="config", version_base="1.1")
def main(cfg: DictConfig):  # noqa: F821
    set_gym_backend(cfg.env.library).set()

    # Create logger
    exp_name = generate_exp_name("DiffusionBC-offline", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="diffusion_bc_logging",
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
    device = (
        torch.device(cfg.network.device)
        if cfg.network.device
        else get_available_device()
    )

    # Create eval env
    eval_env = make_environment(cfg, logger=logger)

    # Create offline replay buffer
    with timeit("setup/replay_buffer"):
        replay_buffer = make_offline_replay_buffer(
            cfg.replay_buffer, cfg, device=device
        )

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    # Create model
    with timeit("setup/model"):
        actor = make_diffusion_actor(cfg, eval_env, device)

    # Create loss
    loss_module = make_loss_module(actor)

    # Create optimizer
    optimizer = make_optimizer(cfg, loss_module)

    n_params = sum(p.numel() for p in actor.parameters())
    n_trainable = sum(p.numel() for p in actor.parameters() if p.requires_grad)

    clip_grad = cfg.optim.clip_grad

    def update(sampled_tensordict):
        optimizer.zero_grad(set_to_none=True)

        with timeit("training/forward"):
            loss_td = loss_module(sampled_tensordict)
            loss = loss_td["loss_diffusion_bc"]

        with timeit("training/backward"):
            loss.backward()

        grad_norm = torch.nn.utils.clip_grad_norm_(actor.parameters(), clip_grad)

        with timeit("training/optim_step"):
            optimizer.step()

        return loss_td.detach(), grad_norm

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
    pbar = tqdm.tqdm(range(gradient_steps))

    # Log setup info
    if logger is not None:
        log_metrics(
            logger,
            {
                "setup/n_params": n_params,
                "setup/n_trainable_params": n_trainable,
                "setup/num_diffusion_steps": cfg.network.num_steps,
                "setup/replay_buffer_size": len(replay_buffer),
            },
            0,
        )

    # Training loop
    for update_counter in pbar:
        timeit.printevery(num_prints=1000, total_count=gradient_steps, erase=True)

        with timeit("training/rb_sample"):
            sampled_tensordict = replay_buffer.sample()

        with timeit("training/update"):
            torch.compiler.cudagraph_mark_step_begin()
            loss_td, grad_norm = update(sampled_tensordict)

        loss_val = loss_td["loss_diffusion_bc"].item()

        metrics_to_log = {
            "training/loss": loss_val,
            "training/grad_norm": grad_norm.item()
            if isinstance(grad_norm, torch.Tensor)
            else grad_norm,
            "training/lr": optimizer.param_groups[0]["lr"],
        }

        pbar.set_postfix(loss=f"{loss_val:.4f}", grad=f"{grad_norm:.3f}")

        # Evaluation
        if update_counter % evaluation_interval == 0:
            with set_exploration_type(
                ExplorationType.DETERMINISTIC
            ), torch.no_grad(), timeit("eval/rollout"):
                eval_td = eval_env.rollout(
                    max_steps=eval_steps, policy=actor, auto_cast_to_device=True
                )
                eval_env.apply(dump_video)
            # Per-step reward
            eval_reward = eval_td["next", "reward"].sum(1).mean().item()
            # Episode return (from RewardSum transform)
            episode_return = eval_td["next", "episode_reward"][..., -1, :].mean().item()
            # Episode length
            episode_len = (
                eval_td["next", "step_count"][..., -1, :].float().mean().item()
            )

            metrics_to_log["eval/episode_return"] = episode_return
            metrics_to_log["eval/reward_sum"] = eval_reward
            metrics_to_log["eval/episode_length"] = episode_len

        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed_iters_per_sec"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, update_counter)

    if not eval_env.is_closed:
        eval_env.close()
    pbar.close()


if __name__ == "__main__":
    main()
