# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""CrossQ Example.

This is a simple self-contained example of a CrossQ training script.

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
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule
from torchrl._utils import timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    log_metrics,
    make_collector,
    make_crossQ_agent,
    make_crossQ_optimizer,
    make_environment,
    make_loss_module,
    make_replay_buffer,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg: DictConfig):  # noqa: F821
    device = cfg.network.device
    if device in ("", None):
        if torch.cuda.is_available():
            device = torch.device("cuda:0")
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    # Create logger
    exp_name = generate_exp_name("CrossQ", cfg.logger.exp_name)
    logger = None
    if cfg.logger.backend:
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="crossq_logging",
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
    train_env, eval_env = make_environment(cfg)

    # Create agent
    model, exploration_policy = make_crossQ_agent(cfg, train_env, device)

    # Create CrossQ loss
    loss_module = make_loss_module(cfg, model, device=device)

    compile_mode = None
    if cfg.compile.compile:
        if cfg.compile.compile_mode not in (None, ""):
            compile_mode = cfg.compile.compile_mode
        elif cfg.compile.cudagraphs:
            compile_mode = "default"
        else:
            compile_mode = "reduce-overhead"

    # Create off-policy collector
    collector = make_collector(
        cfg,
        train_env,
        exploration_policy.eval(),
        device=device,
        compile=cfg.compile.compile,
        compile_mode=compile_mode,
        cudagraph=cfg.compile.cudagraphs,
    )

    # Create replay buffer
    replay_buffer = make_replay_buffer(
        batch_size=cfg.optim.batch_size,
        prb=cfg.replay_buffer.prb,
        buffer_size=cfg.replay_buffer.size,
        scratch_dir=cfg.replay_buffer.scratch_dir,
        device="cpu",
    )

    # Create optimizers
    (
        optimizer_actor,
        optimizer_critic,
        optimizer_alpha,
    ) = make_crossQ_optimizer(cfg, loss_module)
    optimizer = group_optimizers(optimizer_actor, optimizer_critic, optimizer_alpha)
    del optimizer_actor, optimizer_critic, optimizer_alpha

    def update_qloss(sampled_tensordict):
        optimizer.zero_grad(set_to_none=True)
        td_loss = {}
        q_loss, value_meta = loss_module.qvalue_loss(sampled_tensordict)
        sampled_tensordict.set(loss_module.tensor_keys.priority, value_meta["td_error"])
        q_loss = q_loss.mean()

        # Update critic
        q_loss.backward()
        optimizer.step()
        td_loss["loss_qvalue"] = q_loss
        td_loss["loss_actor"] = float("nan")
        td_loss["loss_alpha"] = float("nan")
        return TensorDict(td_loss, device=device).detach()

    def update_all(sampled_tensordict: TensorDict):
        optimizer.zero_grad(set_to_none=True)

        td_loss = {}
        q_loss, value_meta = loss_module.qvalue_loss(sampled_tensordict)
        sampled_tensordict.set(loss_module.tensor_keys.priority, value_meta["td_error"])
        q_loss = q_loss.mean()

        actor_loss, metadata_actor = loss_module.actor_loss(sampled_tensordict)
        actor_loss = actor_loss.mean()
        alpha_loss = loss_module.alpha_loss(
            log_prob=metadata_actor["log_prob"].detach()
        ).mean()

        # Updates
        (q_loss + actor_loss + actor_loss).backward()
        optimizer.step()

        # Update critic
        td_loss["loss_qvalue"] = q_loss
        td_loss["loss_actor"] = actor_loss
        td_loss["loss_alpha"] = alpha_loss

        return TensorDict(td_loss, device=device).detach()

    if compile_mode:
        update_all = torch.compile(update_all, mode=compile_mode)
        update_qloss = torch.compile(update_qloss, mode=compile_mode)
    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results. Use with caution.",
            category=UserWarning,
        )
        update_all = CudaGraphModule(update_all, warmup=50)
        update_qloss = CudaGraphModule(update_qloss, warmup=50)

    def update(sampled_tensordict: TensorDict, update_actor: bool):
        if update_actor:
            return update_all(sampled_tensordict)
        return update_qloss(sampled_tensordict)

    # Main loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.collector.total_frames)

    init_random_frames = cfg.collector.init_random_frames
    num_updates = int(cfg.collector.frames_per_batch * cfg.optim.utd_ratio)
    prb = cfg.replay_buffer.prb
    eval_iter = cfg.logger.eval_iter
    frames_per_batch = cfg.collector.frames_per_batch
    eval_rollout_steps = cfg.env.max_episode_steps

    update_counter = 0
    delayed_updates = cfg.optim.policy_update_delay
    c_iter = iter(collector)
    total_iter = len(collector)
    for _ in range(total_iter):
        timeit.printevery(1000, total_iter, erase=True)
        with timeit("collecting"):
            torch.compiler.cudagraph_mark_step_begin()
            tensordict = next(c_iter)

        # Update weights of the inference policy
        collector.update_policy_weights_()

        current_frames = tensordict.numel()
        pbar.update(current_frames)
        tensordict = tensordict.reshape(-1)

        with timeit("rb - extend"):
            # Add to replay buffer
            replay_buffer.extend(tensordict)
        collected_frames += current_frames

        # Optimization steps
        if collected_frames >= init_random_frames:
            tds = []
            for _ in range(num_updates):
                # Update actor every delayed_updates
                update_counter += 1
                update_actor = update_counter % delayed_updates == 0
                # Sample from replay buffer
                with timeit("rb - sample"):
                    sampled_tensordict = replay_buffer.sample().to(device)
                with timeit("update"):
                    torch.compiler.cudagraph_mark_step_begin()
                    td_loss = update(sampled_tensordict, update_actor=update_actor)
                tds.append(td_loss.clone())
                # Update priority
                if prb:
                    replay_buffer.update_priority(sampled_tensordict)

            tds = TensorDict.stack(tds).nanmean()
        episode_end = (
            tensordict["next", "done"]
            if tensordict["next", "done"].any()
            else tensordict["next", "truncated"]
        )
        episode_rewards = tensordict["next", "episode_reward"][episode_end]

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
                eval_reward = eval_rollout["next", "reward"].sum(-2).mean().item()
                metrics_to_log["eval/reward"] = eval_reward

        # Logging
        if len(episode_rewards) > 0:
            episode_length = tensordict["next", "step_count"][episode_end]
            metrics_to_log["train/reward"] = episode_rewards.mean().item()
            metrics_to_log["train/episode_length"] = episode_length.sum().item() / len(
                episode_length
            )
        if collected_frames >= init_random_frames:
            metrics_to_log["train/q_loss"] = tds["loss_qvalue"]
            metrics_to_log["train/actor_loss"] = tds["loss_actor"]
            metrics_to_log["train/alpha_loss"] = tds["loss_alpha"]

        if logger is not None:
            metrics_to_log.update(timeit.todict(prefix="time"))
            metrics_to_log["time/speed"] = pbar.format_dict["rate"]
            log_metrics(logger, metrics_to_log, collected_frames)

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()


if __name__ == "__main__":
    main()
