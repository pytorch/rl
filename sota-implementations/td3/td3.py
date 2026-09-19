# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""TD3 Example.

This is a simple self-contained example of a TD3 training script.

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
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    dump_video,
    make_collector,
    make_environment,
    make_loss_module,
    make_optimizer,
    make_replay_buffer,
    make_td3_agent,
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
    exp_name = generate_exp_name("TD3", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="td3_logging",
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

    # Set seeds
    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)

    # Create environments
    train_env, eval_env = make_environment(cfg, logger=logger, device=device)

    # Create agent
    model, exploration_policy = make_td3_agent(cfg, train_env, eval_env, device)

    # Create TD3 loss
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
        cfg,
        train_env,
        exploration_policy,
        compile_mode=compile_mode,
        device=device,
    )

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device=device,
        compile=bool(compile_mode),
    )

    # Create optimizers
    optimizer_actor, optimizer_critic = make_optimizer(cfg, loss_module)

    # Checkpointing: the loss holds the online and target networks, so the
    # policy is not saved a second time; the collector is resynchronized
    # from the restored loss parameters.
    run_state = {"collected_frames": 0, "update_counter": 0}
    checkpoint = Checkpoint(
        loss_module=loss_module,
        optimizer_actor=optimizer_actor,
        optimizer_critic=optimizer_critic,
        target_updater=target_net_updater,
        exploration=exploration_policy[1],
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

    prb = cfg.replay_buffer.prb

    def update(sampled_tensordict, update_actor, prb=prb):

        # Compute loss
        q_loss, *_ = loss_module.value_loss(sampled_tensordict)

        # Update critic
        q_loss.backward()
        optimizer_critic.step()
        optimizer_critic.zero_grad(set_to_none=True)

        # Update actor
        if update_actor:
            actor_loss, *_ = loss_module.actor_loss(sampled_tensordict)

            actor_loss.backward()
            optimizer_actor.step()
            optimizer_actor.zero_grad(set_to_none=True)

            # Update target params
            target_net_updater.step()
        else:
            actor_loss = q_loss.new_zeros(())

        return q_loss.detach(), actor_loss.detach()

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
    delayed_updates = cfg.optim.policy_update_delay
    eval_rollout_steps = cfg.env.max_episode_steps
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    update_counter = run_state["update_counter"]

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

                with timeit("train"):
                    # Optimization steps
                    if collected_frames >= init_random_frames:
                        (
                            actor_losses,
                            q_losses,
                        ) = ([], [])
                        for _ in range(num_updates):
                            # Update actor every delayed_updates
                            update_counter += 1
                            update_actor = update_counter % delayed_updates == 0

                            with timeit("rb - sample"):
                                sampled_tensordict = replay_buffer.sample()
                            with timeit("update"):
                                torch.compiler.cudagraph_mark_step_begin()
                                q_loss, actor_loss = update(
                                    sampled_tensordict, update_actor
                                )

                            # Update priority
                            if prb:
                                with timeit("rb - priority"):
                                    replay_buffer.update_priority(sampled_tensordict)

                            q_losses.append(q_loss.clone())
                            if update_actor:
                                actor_losses.append(actor_loss.clone())

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
                    training_metrics["reward"] = episode_rewards.mean()
                    training_metrics["episode_length"] = episode_length.sum() / len(
                        episode_length
                    )

                if collected_frames >= init_random_frames:
                    training_metrics["q_loss"] = torch.stack(q_losses).mean()
                    if update_actor:
                        training_metrics["a_loss"] = torch.stack(actor_losses).mean()

                # Evaluation
                if abs(collected_frames % eval_iter) < frames_per_batch:
                    with (
                        set_exploration_type(ExplorationType.DETERMINISTIC),
                        torch.no_grad(),
                        timeit("eval"),
                    ):
                        eval_rollout = eval_env.rollout(
                            eval_rollout_steps,
                            exploration_policy,
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
                    timing_metrics["speed"] = pbar.format_dict["rate"]
                    if timing_metrics:
                        timing_logger.log_metrics(timing_metrics, collected_frames)

                run_state.update(
                    collected_frames=collected_frames, update_counter=update_counter
                )
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
