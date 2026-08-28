# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import warnings

import hydra
import numpy as np
import torch
import tqdm
from omegaconf import DictConfig
from tensordict import TensorDict
from tensordict.nn import CudaGraphModule

from torchrl._utils import compile_with_warmup, get_available_device, timeit
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import group_optimizers
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    dump_video,
    make_collector,
    make_environment,
    make_loss_module,
    make_optimizers,
    make_replay_buffer,
    make_tqc_agent,
)

torch.set_float32_matmul_precision("high")


@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig):
    device = (
        torch.device(cfg.network.device)
        if cfg.network.device
        else get_available_device()
    )
    logger = None
    if cfg.logger.backend:
        experiment_name = generate_exp_name("TQC", cfg.logger.exp_name)
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="tqc_logging",
            experiment_name=experiment_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    torch.manual_seed(cfg.env.seed)
    np.random.seed(cfg.env.seed)
    train_env, eval_env = make_environment(cfg, logger=logger)
    model, exploration_policy = make_tqc_agent(cfg, train_env, eval_env, device)
    loss_module, target_updater = make_loss_module(cfg, model)

    compile_mode = None
    if cfg.compile.compile:
        compile_mode = cfg.compile.compile_mode
        if compile_mode in ("", None):
            compile_mode = "default" if cfg.compile.cudagraphs else "reduce-overhead"
    collector = make_collector(cfg, train_env, exploration_policy, compile_mode)
    replay_buffer = make_replay_buffer(cfg, device)
    actor_optimizer, critic_optimizer, alpha_optimizer = make_optimizers(
        cfg, loss_module
    )
    optimizer = group_optimizers(actor_optimizer, critic_optimizer, alpha_optimizer)

    def update(sampled_tensordict):
        loss_tensordict = loss_module(sampled_tensordict)
        if cfg.replay_buffer.prb:
            loss_tensordict.set(
                loss_module.tensor_keys.priority,
                sampled_tensordict.get(loss_module.tensor_keys.priority),
            )
        total_loss = (
            loss_tensordict.get("loss_actor")
            + loss_tensordict.get("loss_qvalue")
            + loss_tensordict.get("loss_alpha")
        )
        total_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        target_updater.step()
        return loss_tensordict.detach()

    if cfg.compile.compile:
        update = compile_with_warmup(update, mode=compile_mode, warmup=1)
    if cfg.compile.cudagraphs:
        warnings.warn(
            "CudaGraphModule is experimental and may lead to silently wrong results.",
            category=UserWarning,
        )
        update = CudaGraphModule(update, in_keys=[], out_keys=[], warmup=5)

    collected_frames = 0
    progress = tqdm.tqdm(total=cfg.collector.total_frames)
    num_updates = int(cfg.collector.frames_per_batch * cfg.optim.utd_ratio)
    total_iterations = len(collector)
    collector_iterator = iter(collector)

    while collected_frames < cfg.collector.total_frames:
        timeit.printevery(num_prints=1000, total_count=total_iterations, erase=True)
        with timeit("collect"):
            data = next(collector_iterator)
        collector.update_policy_weights_()
        current_frames = data.numel()
        progress.update(current_frames)
        data = data.reshape(-1)
        with timeit("rb - extend"):
            replay_buffer.extend(data)
        collected_frames += current_frames

        if collected_frames >= cfg.collector.init_random_frames:
            losses = TensorDict(batch_size=[num_updates])
            with timeit("train"):
                for update_index in range(num_updates):
                    with timeit("rb - sample"):
                        sampled_data = replay_buffer.sample()
                    with timeit("update"):
                        torch.compiler.cudagraph_mark_step_begin()
                        loss_tensordict = update(sampled_data).clone()
                    losses[update_index] = loss_tensordict.select(
                        "loss_actor", "loss_qvalue", "loss_alpha"
                    )
                    if cfg.replay_buffer.prb:
                        sampled_data.set(
                            loss_module.tensor_keys.priority,
                            loss_tensordict.get(loss_module.tensor_keys.priority),
                        )
                        replay_buffer.update_tensordict_priority(sampled_data)

        episode_end = data.get(("next", "done")) | data.get(("next", "truncated"))
        episode_rewards = data.get(("next", "episode_reward"))[episode_end]
        metrics = {}
        if episode_rewards.numel():
            episode_lengths = data.get(("next", "step_count"))[episode_end]
            metrics["train/reward"] = episode_rewards.float().mean()
            metrics["train/episode_length"] = episode_lengths.float().mean()
        if collected_frames >= cfg.collector.init_random_frames:
            losses = losses.mean()
            metrics["train/q_loss"] = losses.get("loss_qvalue")
            metrics["train/actor_loss"] = losses.get("loss_actor")
            metrics["train/alpha_loss"] = losses.get("loss_alpha")
            metrics["train/alpha"] = loss_tensordict.get("alpha")
            metrics["train/entropy"] = loss_tensordict.get("entropy")

        if collected_frames % cfg.logger.eval_iter < cfg.collector.frames_per_batch:
            with (
                set_exploration_type(ExplorationType.DETERMINISTIC),
                torch.no_grad(),
                timeit("eval"),
            ):
                rollout = eval_env.rollout(
                    cfg.env.max_episode_steps,
                    model[0],
                    auto_cast_to_device=True,
                    break_when_any_done=True,
                )
                eval_env.apply(dump_video)
                metrics["eval/reward"] = (
                    rollout.get(("next", "reward")).sum(-2).mean().item()
                )
        if logger is not None:
            metrics.update(timeit.todict(prefix="time"))
            metrics["time/speed"] = progress.format_dict["rate"]
            logger.log_metrics(metrics, collected_frames)

    collector.shutdown()
    if not eval_env.is_closed:
        eval_env.close()
    if not train_env.is_closed:
        train_env.close()


if __name__ == "__main__":
    main()
