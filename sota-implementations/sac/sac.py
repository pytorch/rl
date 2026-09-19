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

import warnings

import hydra
import numpy as np
import torch
import torch.cuda
import tqdm
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import compile_with_warmup, get_available_device, timeit
from torchrl.checkpoint import (
    Checkpoint,
    GlobalRNGState,
    resolve_checkpoint_path,
    resume_config,
    RunCheckpointer,
    StopOnSignal,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    dump_video,
    make_collector,
    make_environment,
    make_loss_module,
    make_replay_buffer,
    make_sac_agent,
    make_sac_optimizer,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig):
    # Resume: the saved configuration is the base, CLI overrides apply on top.
    resume_path = None
    if cfg.resume:
        resume_path = resolve_checkpoint_path(to_absolute_path(cfg.resume))
        cfg = resume_config(cfg, resume_path)

    device = (
        torch.device(cfg.network.device)
        if cfg.network.device
        else get_available_device()
    )

    # Create logger, reattached to the saved run when resuming
    exp_name = generate_exp_name("SAC", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="sac_logging",
            experiment_name=exp_name,
            state_dict=(
                Checkpoint.read_component(resume_path, "logger", default=None)
                if resume_path
                else None
            ),
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )
        training_logger = logger.with_prefix("training")
        evaluation_logger = logger.with_prefix("evaluation")
        timing_logger = logger.with_prefix("timing")

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg, logger=logger)

    # Create agent
    model, exploration_policy = make_sac_agent(cfg, train_env, eval_env, device)

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

    # Create off-policy collector
    collector = make_collector(
        cfg, train_env, exploration_policy, compile_mode=compile_mode
    )

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=device,
    )

    # Create optimizers
    (
        optimizer_actor,
        optimizer_critic,
        optimizer_alpha,
    ) = make_sac_optimizer(cfg, loss_module)
    optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
    del optimizer_actor, optimizer_critic, optimizer_alpha

    # Checkpointing: the loss holds the online and target networks, so the
    # policy is not saved a second time; the collector is resynchronized
    # from the restored loss parameters.
    run_state = {"collected_frames": 0}
    checkpoint = Checkpoint(
        loss_module=loss_module,
        optimizer=optimizer,
        target_updater=target_net_updater,
        collector=collector,
        replay_buffer=replay_buffer,
        run_state=run_state,
        rng=GlobalRNGState(),
        config=OmegaConf.to_container(cfg, resolve=False),
    )
    if logger is not None:
        checkpoint.register("logger", logger)
    checkpointer = RunCheckpointer(
        checkpoint,
        directory=cfg.checkpoint.dir,
        interval=cfg.checkpoint.interval,
        keep_last=cfg.checkpoint.keep_last,
        exclude=() if cfg.checkpoint.include_replay_buffer else ("replay_buffer",),
        resume_path=resume_path,
    )
    if checkpointer.restore(map_location=device):
        collector.update_policy_weights_()

    def update(sampled_tensordict):
        # Compute loss
        loss_td = loss_module(sampled_tensordict)

        actor_loss = loss_td["loss_actor"]
        q_loss = loss_td["loss_qvalue"]
        alpha_loss = loss_td["loss_alpha"]

        (actor_loss + q_loss + alpha_loss).sum().backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # Update qnet_target params
        target_net_updater.step()
        return loss_td.detach()

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)

    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)

    # Main loop
    collected_frames = run_state["collected_frames"]
    pbar = tqdm.tqdm(total=cfg.collector.total_frames, initial=collected_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(cfg.collector.frames_per_batch * cfg.optim.utd_ratio)
    prb = cfg.replay_buffer.prb
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    eval_rollout_steps = cfg.env.max_episode_steps

    collector_iter = iter(collector)
    total_iter = len(collector)

    # SIGINT/SIGTERM stop after the current batch; a second signal interrupts.
    stop = StopOnSignal()
    try:
        with stop:
            while True:
                timeit.printevery(num_prints=1000, total_count=total_iter, erase=True)

                with timeit("collect"):
                    tensordict = next(collector_iter, None)
                if tensordict is None:
                    break

                # Update weights of the inference policy
                collector.update_policy_weights_()

                current_frames = tensordict.numel()
                pbar.update(current_frames)

                with timeit("rb - extend"):
                    # Add to replay buffer
                    tensordict = tensordict.reshape(-1)
                    replay_buffer.extend(tensordict)

                collected_frames += current_frames

                # Optimization steps
                with timeit("train"):
                    if collected_frames >= init_random_frames:
                        losses = TensorDict(batch_size=[num_updates])
                        for i in range(num_updates):
                            with timeit("rb - sample"):
                                # Sample from replay buffer
                                sampled_tensordict = replay_buffer.sample()

                            with timeit("update"):
                                torch.compiler.cudagraph_mark_step_begin()
                                loss_td = update(sampled_tensordict).clone()
                            losses[i] = loss_td.select(
                                "loss_actor", "loss_qvalue", "loss_alpha"
                            )

                            # Update priority
                            if prb:
                                replay_buffer.update_priority(sampled_tensordict)

                episode_end = (
                    tensordict["next", "done"]
                    if tensordict["next", "done"].any()
                    else tensordict["next", "truncated"]
                )
                episode_rewards = tensordict["next", "episode_reward"][episode_end]

                # Logging
                training_metrics = {}
                evaluation_metrics = {}
                if len(episode_rewards) > 0:
                    episode_length = tensordict["next", "step_count"][episode_end]
                    training_metrics["reward"] = episode_rewards
                    training_metrics["episode_length"] = episode_length.sum() / len(
                        episode_length
                    )
                if collected_frames >= init_random_frames:
                    losses = losses.mean()
                    training_metrics["q_loss"] = losses.get("loss_qvalue")
                    training_metrics["actor_loss"] = losses.get("loss_actor")
                    training_metrics["alpha_loss"] = losses.get("loss_alpha")
                    training_metrics["alpha"] = loss_td["alpha"]
                    training_metrics["entropy"] = loss_td["entropy"]

                # Evaluation
                if abs(collected_frames % eval_iter) < frames_per_batch:
                    with (
                        set_exploration_type(ExplorationType.DETERMINISTIC),
                        torch.no_grad(),
                        timeit("eval"),
                    ):
                        eval_rollout = eval_env.rollout(
                            eval_rollout_steps,
                            model[0],
                            auto_cast_to_device=True,
                            break_when_any_done=True,
                        )
                        eval_env.apply(dump_video)
                        eval_reward = (
                            eval_rollout["next", "reward"].sum(-2).mean().item()
                        )
                        evaluation_metrics["reward"] = eval_reward
                if logger is not None:
                    if training_metrics:
                        training_logger.log_metrics(training_metrics, collected_frames)
                    if evaluation_metrics:
                        evaluation_logger.log_metrics(
                            evaluation_metrics, collected_frames
                        )
                    timing_metrics = timeit.todict()
                    speed = pbar.format_dict["rate"]
                    if speed is not None:
                        timing_metrics["speed"] = speed
                    if timing_metrics:
                        timing_logger.log_metrics(timing_metrics, collected_frames)

                run_state["collected_frames"] = collected_frames
                checkpointer.save(collected_frames)
                if stop.requested:
                    break
        checkpointer.save(collected_frames, force=True)
    finally:
        collector.shutdown()
        if not eval_env.is_closed:
            eval_env.close()
        if not train_env.is_closed:
            train_env.close()


if __name__ == "__main__":
    main()
