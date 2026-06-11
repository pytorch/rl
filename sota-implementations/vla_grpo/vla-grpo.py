# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""GRPO-style RL fine-tuning of a token-head VLA policy (toy scale).

This is the single-process v0 of the SimpleVLA-RL recipe
(`arXiv:2509.09674 <https://arxiv.org/abs/2509.09674>`_): a token-head VLA
policy emits a whole action chunk per forward, trajectories are collected in
groups sharing the same initial state, the advantage is the group-normalized
binary success return broadcast to every chunk decision, and the policy is
updated with an asymmetric-clip PPO objective (no critic, no KL-to-reference,
no entropy bonus).

The training-sample unit is the *decision* (one outer step of the
``MultiAction``-transformed environment = one chunk). Per-iteration
accounting: ``groups_per_iter`` initial states x ``group_size`` rollouts per
state, each contributing up to ``max_outer_steps`` decisions; the dynamic
sampling filter drops groups whose rollouts all failed or all succeeded, so
the effective batch is variable.

This toy variant trains TinyVLA on the ToyVLAEnv tracking task end-to-end on
a single device, with no Ray and no simulator dependencies, and is exercised
in the sota CI at a tiny budget.
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
    make_env,
    make_loss_module,
    make_policy,
    make_replay_buffer,
    make_tokenizer,
)

warnings.filterwarnings("ignore", category=UserWarning, module="tensordict")


def save_checkpoint(path, policy, optim, iteration):
    torch.save(
        {
            "policy": policy.state_dict(),
            "optim": optim.state_dict(),
            "iteration": iteration,
            "torch_rng_state": torch.get_rng_state(),
        },
        path,
    )


def load_checkpoint(path, policy, optim) -> int:
    checkpoint = torch.load(path, weights_only=False)
    policy.load_state_dict(checkpoint["policy"])
    optim.load_state_dict(checkpoint["optim"])
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

    # Environments: training rollouts are grouped (the same initial state is
    # replayed group_size times and stamped with a group_id); evaluation uses
    # fresh initial states and greedy decoding.
    tokenizer = make_tokenizer(cfg)
    train_env = make_env(
        cfg,
        tokenizer,
        group_repeats=cfg.collector.group_size,
        seed=cfg.env.seed,
        device=device,
    )
    eval_env = make_env(cfg, tokenizer, seed=cfg.env.seed + 1, device=device)

    # Policy (token head, one chunk per forward); materialize the lazy layers
    # on the eval env so the train env's grouped reset accounting stays
    # aligned with the iteration boundaries
    policy = make_policy(cfg, device)
    with torch.no_grad():
        policy(eval_env.reset())

    replay_buffer = make_replay_buffer(cfg, device)
    loss_module = make_loss_module(cfg, policy)
    optim = torch.optim.Adam(loss_module.parameters(), lr=cfg.optim.lr)

    start_iter = 0
    if cfg.checkpoint.resume:
        start_iter = load_checkpoint(cfg.checkpoint.resume, policy, optim)
        torchrl_logger.info(
            f"Resumed from {cfg.checkpoint.resume} at iteration {start_iter}."
        )

    episodes_per_iter = cfg.collector.groups_per_iter * cfg.collector.group_size
    pbar = tqdm.tqdm(total=cfg.collector.total_iters, initial=start_iter)
    total_episodes = start_iter * episodes_per_iter

    for iteration in range(start_iter, cfg.collector.total_iters):
        # Collect one iteration of grouped rollouts. The replay-buffer write
        # path groups whole trajectories by group_id, computes the
        # group-relative advantage and applies the dynamic sampling filter.
        episode_successes = []
        episode_lengths = []
        policy.mode = "sample"
        with torch.no_grad(), timeit("collect"):
            for _ in range(episodes_per_iter):
                episode = train_env.rollout(cfg.env.max_outer_steps, policy)
                episode_successes.append(float(episode["next", "success"].any()))
                episode_lengths.append(episode.numel())
                replay_buffer.extend(episode.reshape(-1))
        total_episodes += episodes_per_iter

        # PPO update over the decisions that survived dynamic sampling
        num_decisions = len(replay_buffer)
        losses = []
        clip_fractions = []
        ess = []
        grad_norms = []
        with timeit("train"):
            for _ in range(cfg.loss.ppo_epochs):
                if not num_decisions:
                    break
                for batch in replay_buffer:
                    loss_vals = loss_module(batch)
                    loss = loss_vals["loss_objective"]
                    loss.backward()
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        loss_module.parameters(), cfg.optim.max_grad_norm
                    )
                    optim.step()
                    optim.zero_grad(set_to_none=True)
                    losses.append(loss.detach())
                    clip_fractions.append(loss_vals["clip_fraction"])
                    ess.append(loss_vals["ESS"].detach())
                    grad_norms.append(grad_norm)
        replay_buffer.empty()

        metrics = {
            "train/success_rate": sum(episode_successes) / len(episode_successes),
            "train/episode_decisions": sum(episode_lengths) / len(episode_lengths),
            "train/episodes_total": total_episodes,
            "buffer/decisions": num_decisions,
            "buffer/kept_fraction": num_decisions / max(1, sum(episode_lengths)),
        }
        if losses:
            metrics.update(
                {
                    "train/loss_objective": torch.stack(losses).mean().item(),
                    "train/clip_fraction": torch.stack(clip_fractions).mean().item(),
                    "train/ESS": torch.stack(ess).mean().item(),
                    "train/grad_norm": torch.stack(grad_norms).mean().item(),
                }
            )
        if iteration % cfg.logger.eval_iter == 0:
            with timeit("eval"):
                metrics["eval/success_rate"] = evaluate(eval_env, policy, cfg)
        metrics.update(timeit.todict(prefix="time"))
        timeit.erase()
        log_metrics(logger, metrics, iteration)
        pbar.update(1)
        pbar.set_description(
            f"success {metrics['train/success_rate']:.2f} " f"decisions {num_decisions}"
        )

        if cfg.checkpoint.save_iter and (iteration + 1) % cfg.checkpoint.save_iter == 0:
            save_checkpoint(
                os.path.join(os.getcwd(), "checkpoint_latest.pt"),
                policy,
                optim,
                iteration,
            )

    if cfg.checkpoint.save_iter:
        save_checkpoint(
            os.path.join(os.getcwd(), "checkpoint_latest.pt"),
            policy,
            optim,
            cfg.collector.total_iters - 1,
        )
    pbar.close()
    if logger is not None:
        final_success = evaluate(eval_env, policy, cfg)
        log_metrics(
            logger, {"eval/success_rate": final_success}, cfg.collector.total_iters
        )
        torchrl_logger.info(f"Final greedy success rate: {final_success:.3f}")


if __name__ == "__main__":
    main()
