# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""IQL Example.

This is a self-contained example of an online discrete IQL training script.

It works across Gym and MuJoCo over a variety of tasks.

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
from torchrl._utils import timeit
from torchrl.envs import set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    dump_video,
    log_metrics,
    make_collector,
    make_discrete_iql_model,
    make_discrete_loss,
    make_environment,
    make_iql_optimizer,
    make_replay_buffer,
)

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="", config_name="discrete_iql")
def main(cfg: DictConfig):  # noqa: F821
    set_gym_backend(cfg.env.backend).set()

    # Create logger
    exp_name = generate_exp_name("Discrete-IQL-online", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="iql_logging",
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

    # Create environments
    train_env, eval_env = make_environment(
        cfg,
        cfg.env.train_num_envs,
        cfg.env.eval_num_envs,
        logger=logger,
    )

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        device="cpu",
    )

    # Create model
    model = make_discrete_iql_model(cfg, train_env, eval_env, device)

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            if cfg.compile.cudagraphs:
                compile_mode = "default"
            else:
                compile_mode = "reduce-overhead"

    # Create collector
    collector = make_collector(
        cfg, train_env, actor_model_explore=model[0], compile_mode=compile_mode
    )

    # Create loss
    loss_module, target_net_updater = make_discrete_loss(cfg.loss, model, device=device)

    # Create optimizer
    optimizer_actor, optimizer_critic, optimizer_value = make_iql_optimizer(
        cfg.optim, loss_module
    )
    optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_value)
    del optimizer_actor, optimizer_critic, optimizer_value

    def update(sampled_tensordict):
        optimizer.zero_grad(set_to_none=True)
        # compute losses
        actor_loss, _ = loss_module.actor_loss(sampled_tensordict)
        value_loss, _ = loss_module.value_loss(sampled_tensordict)
        q_loss, metadata = loss_module.qvalue_loss(sampled_tensordict)
        (actor_loss + value_loss + q_loss).backward()
        optimizer.step()

        # update qnet_target params
        target_net_updater.step()
        metadata.update(
            {"actor_loss": actor_loss, "value_loss": value_loss, "q_loss": q_loss}
        )
        return TensorDict(metadata).detach()

    if cfg.compile.compile:
        update = torch.compile(update, mode=compile_mode)
    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, warmup=50)

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(cfg.collector.frames_per_batch * cfg.optim.utd_ratio)
    prb = cfg.replay_buffer.prb
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    eval_rollout_steps = cfg.collector.max_frames_per_traj

    collector_iter = iter(collector)
    total_iter = len(collector)
    for _ in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)

        with timeit("collection"):
            tensordict = next(collector_iter)
        current_frames = tensordict.numel()
        pbar.update(current_frames)

        # update weights of the inference policy
        collector.update_policy_weights_()

        with timeit("buffer - extend"):
            tensordict = tensordict.reshape(-1)

            # add to replay buffer
            replay_buffer.extend(tensordict)
        collected_frames += current_frames

        # optimization steps
        with timeit("training"):
            if collected_frames >= init_random_frames:
                for _ in range(num_updates):
                    # sample from replay buffer
                    with timeit("buffer - sample"):
                        sampled_tensordict = replay_buffer.sample().to(device)

                    with timeit("training - update"):
                        torch.compiler.cudagraph_mark_step_begin()
                        metadata = update(sampled_tensordict)
                    # update priority
                    if prb:
                        sampled_tensordict.set(
                            loss_module.tensor_keys.priority,
                            metadata.pop("td_error").detach().max(0).values,
                        )
                        replay_buffer.update_priority(sampled_tensordict)

        episode_rewards = tensordict["next", "episode_reward"][
            tensordict["next", "done"]
        ]

        metrics_to_log = {}
        # Evaluation
        if abs(collected_frames % eval_iter) < frames_per_batch:
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

        # Logging
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][
                tensordict["next", "done"]
            ]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                episode_length
            )
        if collected_frames >= init_random_frames:
            metrics_to_log["train/q_loss"] = metadata["q_loss"]
            metrics_to_log["train/actor_loss"] = metadata["actor_loss"]
            metrics_to_log["train/value_loss"] = metadata["value_loss"]
        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, collected_frames)

    collector.shutdown()


if __name__ == "__main__":
    main()
