# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""GRPO-style RL fine-tuning of a token-head VLA policy.

This is the SimpleVLA-RL recipe
(`arXiv:2509.09674 <https://arxiv.org/abs/2509.09674>`_): a token-head VLA
policy emits a whole action chunk per forward (parallel decoding),
trajectories are collected in groups sharing the same initial state, the
advantage is the group-normalized binary success return broadcast to every
chunk decision, degenerate groups are dropped (dynamic sampling), and the
policy is updated with an asymmetric-clip PPO objective (no critic, no
KL-to-reference, no entropy bonus).

The training-sample unit is the *decision* (one outer step of the
``MultiAction``-transformed environment = one chunk). Per-iteration
accounting: ``groups_per_iter`` initial states x ``group_size`` rollouts per
state, each contributing up to ``max_outer_steps`` decisions; the dynamic
sampling filter drops groups whose rollouts all failed or all succeeded, so
the effective batch is variable.

Two configurations ship:

- ``vla_grpo_toy.yaml`` (default): TinyVLA on the ToyVLAEnv tracking task,
  single device, no simulator dependencies; exercised in the sota CI.
- ``vla_grpo_libero.yaml``: OpenVLA-OFT (token variant, 7B) on LIBERO with
  the full SimpleVLA-RL hyper-parameters; parallel MuJoCo workers feed a
  single training device. Multi-GPU sharded training (FSDP) of the 7B model
  is the documented next step and should be sized on the target hardware;
  the LoRA fallback (``policy.lora_rank``) fits a single device.
"""
from __future__ import annotations

import os
import warnings

import hydra
import torch
import tqdm
from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    evaluate,
    log_metrics,
    make_action_tokenizer,
    make_collector,
    make_env,
    make_loss_module,
    make_optimizer,
    make_policy,
    make_replay_buffer,
)

warnings.filterwarnings("ignore", category=UserWarning, module="tensordict")


def save_checkpoint(path, policy, optim, scheduler, iteration):
    torch.save(
        {
            "policy": policy.state_dict(),
            "optim": optim.state_dict(),
            "scheduler": scheduler.state_dict(),
            "iteration": iteration,
            "torch_rng_state": torch.get_rng_state(),
        },
        path,
    )


def load_checkpoint(path, policy, optim, scheduler) -> int:
    checkpoint = torch.load(path, weights_only=False)
    policy.load_state_dict(checkpoint["policy"])
    optim.load_state_dict(checkpoint["optim"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    return checkpoint["iteration"] + 1


@hydra.main(config_path="config", config_name="vla_grpo_toy", version_base="1.1")
def main(cfg):  # noqa: F821
    torch.manual_seed(cfg.env.seed)
    device = torch.device(
        cfg.policy.device
        if cfg.policy.device
        else ("cuda:0" if torch.cuda.is_available() else "cpu")
    )

    # Logger
    logger = None
    if cfg.logger.backend:
        exp_name = generate_exp_name("VLA-GRPO", cfg.logger.exp_name)
        logger = get_logger(
            logger_type=cfg.logger.backend,
            logger_name="vla_grpo_logging",
            experiment_name=exp_name,
            wandb_kwargs={
                "mode": cfg.logger.mode,
                "config": dict(cfg),
                "project": cfg.logger.project_name,
                "group": cfg.logger.group_name,
            },
        )

    # Policy first (the LIBERO action codec lives in the checkpoint), then
    # the environments: training rollouts are grouped (the same initial state
    # is replayed group_size times and stamped with a group_id); evaluation
    # uses fresh (cycled) initial states and greedy decoding.
    policy = make_policy(cfg, device)
    tokenizer = make_action_tokenizer(cfg, policy)
    train_env = make_env(
        cfg,
        tokenizer,
        group_repeats=cfg.collector.group_size,
        seed=cfg.env.seed,
        device=device if cfg.env.backend == "toy" else None,
    )
    eval_env = make_env(
        cfg,
        tokenizer,
        seed=cfg.env.seed + 1,
        device=device if cfg.env.backend == "toy" else None,
        eval_mode=True,
    )
    # materialize lazy layers (TinyVLA) on a spec-shaped fake observation: a
    # real reset would consume a grouped init (train env) or a cycled
    # evaluation init state (eval env)
    with torch.no_grad():
        policy(eval_env.fake_tensordict().to(device))

    buffer_device = torch.device(cfg.buffer.device) if cfg.buffer.device else device
    replay_buffer, advantage_transform = make_replay_buffer(cfg, buffer_device)
    loss_module = make_loss_module(cfg, policy)
    optim, scheduler = make_optimizer(cfg, loss_module)
    collector = make_collector(cfg, train_env, policy, device)
    collector_iter = iter(collector)

    start_iter = 0
    if cfg.checkpoint.resume:
        start_iter = load_checkpoint(cfg.checkpoint.resume, policy, optim, scheduler)
        torchrl_logger.info(
            f"Resumed from {cfg.checkpoint.resume} at iteration {start_iter}."
        )

    episodes_per_iter = cfg.collector.groups_per_iter * cfg.collector.group_size
    accumulate = max(int(cfg.loss.accumulate_batches), 1)
    pbar = tqdm.tqdm(total=cfg.collector.total_iters, initial=start_iter)
    total_episodes = start_iter * episodes_per_iter

    for iteration in range(start_iter, cfg.collector.total_iters):
        # Collect one iteration of grouped rollouts: one collector batch =
        # episodes_per_iter complete trajectories, zero-padded along time
        # with a validity mask. Each trajectory is written whole (unpadded)
        # to the replay buffer, whose write path groups them by group_id,
        # computes the group-relative advantage and applies the dynamic
        # sampling filter.
        policy.mode = "sample"
        with timeit("collect"):
            trajectories = next(collector_iter)
            mask = trajectories["collector", "mask"]
            episode_successes = (
                (trajectories["next", "success"].squeeze(-1) & mask).any(-1).float()
            )
            episode_lengths = mask.sum(-1)
            for row, length in zip(trajectories.unbind(0), episode_lengths.tolist()):
                # the padding is a suffix: slice rather than bool-mask (plain
                # slices keep NonTensor entries intact)
                replay_buffer.extend(row[:length].exclude("collector"))
        # synchronous iteration semantics, as in the paper: episodes in
        # flight and incomplete groups do not straddle the policy update (a
        # group baseline must estimate a single policy's returns). The
        # dropped work is bounded by one episode per worker plus their
        # incomplete groups.
        collector.reset()
        advantage_transform.queues.clear()
        total_episodes += episode_successes.numel()

        # PPO update over the decisions that survived dynamic sampling, with
        # gradient accumulation (micro-batches of mini_batch_size decisions)
        num_decisions = len(replay_buffer)
        losses = []
        clip_fractions = []
        ess = []
        grad_norms = []

        def optimizer_step():
            grad_norms.append(
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.optim.max_grad_norm
                )
            )
            optim.step()
            optim.zero_grad(set_to_none=True)
            scheduler.step()

        micro_batches = 0
        with timeit("train"):
            for _ in range(cfg.loss.ppo_epochs):
                if not num_decisions:
                    break
                for batch in replay_buffer:
                    batch = batch.to(device)
                    if cfg.loss.ratio_level == "token":
                        # one importance ratio per action token: broadcast
                        # the decision's advantage over the token dims
                        tokens = batch["action_tokens"]
                        batch["advantage"] = (
                            batch["advantage"]
                            .view(-1, 1, 1, 1)
                            .expand(*tokens.shape, 1)
                        )
                    loss_vals = loss_module(batch)
                    loss = loss_vals["loss_objective"] / accumulate
                    loss.backward()
                    micro_batches += 1
                    losses.append(loss_vals["loss_objective"].detach())
                    clip_fractions.append(loss_vals["clip_fraction"])
                    ess.append(loss_vals["ESS"].detach())
                    if micro_batches % accumulate == 0:
                        optimizer_step()
            if micro_batches % accumulate:
                optimizer_step()
        replay_buffer.empty()

        eval_success = None
        if iteration % cfg.logger.eval_iter == 0:
            with timeit("eval"):
                eval_success = evaluate(eval_env, policy, cfg)

        timings = timeit.todict(prefix="time")
        timeit.erase()
        collect_time = max(timings.get("time/collect", 0.0), 1e-9)
        num_decisions_collected = int(episode_lengths.sum())
        env_steps = num_decisions_collected * cfg.env.chunk_size
        metrics = {
            "train/success_rate": episode_successes.mean().item(),
            "train/episode_decisions": episode_lengths.float().mean().item(),
            "train/episodes_total": total_episodes,
            "train/lr": scheduler.get_last_lr()[0],
            "buffer/decisions": num_decisions,
            "buffer/kept_fraction": num_decisions / max(1, num_decisions_collected),
            "throughput/env_steps_per_s": env_steps / collect_time,
            "throughput/decisions_per_s": num_decisions_collected / collect_time,
        }
        metrics.update(timings)
        if eval_success is not None:
            metrics["eval/success_rate"] = eval_success
        if losses:
            metrics.update(
                {
                    "train/loss_objective": torch.stack(losses).mean().item(),
                    "train/clip_fraction": torch.stack(clip_fractions).mean().item(),
                    "train/ESS": torch.stack(ess).mean().item(),
                    "train/grad_norm": torch.stack(grad_norms).mean().item(),
                }
            )
        log_metrics(logger, metrics, iteration)
        pbar.update(1)
        pbar.set_description(
            f"success {metrics['train/success_rate']:.2f} decisions {num_decisions}"
        )

        if cfg.checkpoint.save_iter and (iteration + 1) % cfg.checkpoint.save_iter == 0:
            save_checkpoint(
                os.path.join(os.getcwd(), "checkpoint_latest.pt"),
                policy,
                optim,
                scheduler,
                iteration,
            )

    if cfg.checkpoint.save_iter:
        save_checkpoint(
            os.path.join(os.getcwd(), "checkpoint_latest.pt"),
            policy,
            optim,
            scheduler,
            cfg.collector.total_iters - 1,
        )
    pbar.close()
    if logger is not None:
        final_success = evaluate(eval_env, policy, cfg)
        log_metrics(
            logger, {"eval/success_rate": final_success}, cfg.collector.total_iters
        )
        torchrl_logger.info(f"Final greedy success rate: {final_success:.3f}")
    collector.shutdown()
    train_env.close(raise_if_closed=False)
    eval_env.close(raise_if_closed=False)


if __name__ == "__main__":
    main()
