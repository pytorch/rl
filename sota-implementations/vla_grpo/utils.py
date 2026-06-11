# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Factories for the VLA GRPO training script.

The environment factory wires the chunk-decision data path: one outer step of
the transformed environment is one policy (chunk) decision. The policy emits
``action_tokens`` for a whole chunk in a single forward; the tokenizer
transform decodes them into the continuous chunk on the inverse path;
``MultiAction`` unbinds the chunk and steps the base environment once per
action; ``SuccessReward`` and ``StepCounter`` run once per outer step, so the
decision reward is the binary success flag and episodes are truncated in
decisions.
"""
from __future__ import annotations

import torch

from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.vla import UniformActionTokenizer
from torchrl.envs import (
    ActionTokenizerTransform,
    Compose,
    MultiAction,
    StepCounter,
    SuccessReward,
    ToyVLAEnv,
    TransformedEnv,
)
from torchrl.modules.vla import TinyVLA
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.llm import MCAdvantage


def make_tokenizer(cfg) -> UniformActionTokenizer:
    return UniformActionTokenizer(cfg.tokenizer.vocab_size, low=-1.0, high=1.0)


def make_env(
    cfg,
    tokenizer: UniformActionTokenizer,
    *,
    group_repeats: int | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
) -> TransformedEnv:
    base = ToyVLAEnv(
        action_dim=cfg.env.action_dim,
        state_dim=cfg.env.state_dim,
        image_shape=tuple(cfg.env.image_shape),
        success_steps=cfg.env.success_steps,
        success_tol=cfg.env.success_tol,
        group_repeats=group_repeats,
        batch_size=[1],
        seed=seed,
        device=device,
    )
    # The compose order is load-bearing: the inverse (action-input) path runs
    # in reverse, so the tokenizer decode happens before MultiAction unbinds
    # the chunk; on the step path SuccessReward and StepCounter run after
    # MultiAction, i.e. once per outer (decision) step. stack_rewards=False
    # keeps the outer transition dense when an episode ends inside a chunk
    # (the decision reward comes from the outer success flag instead).
    transform = Compose(
        MultiAction(stack_rewards=False),
        ActionTokenizerTransform(tokenizer),
        SuccessReward(),
        StepCounter(max_steps=cfg.env.max_outer_steps),
    )
    return TransformedEnv(base, transform)


def make_policy(cfg, device: torch.device) -> TinyVLA:
    return TinyVLA(
        action_dim=cfg.env.action_dim,
        chunk_size=cfg.env.chunk_size,
        action_head="tokens",
        vocab_size=cfg.tokenizer.vocab_size,
        hidden_dim=cfg.policy.hidden_dim,
        mode="sample",
        device=device,
    )


def make_replay_buffer(cfg, device: torch.device) -> TensorDictReplayBuffer:
    # The buffer holds one iteration's decisions; the write path computes the
    # group-relative advantage (and drops degenerate groups) on whole
    # trajectories, the read path samples decisions without replacement.
    capacity = (
        cfg.collector.groups_per_iter
        * cfg.collector.group_size
        * cfg.env.max_outer_steps
    )
    keep_return_bounds = cfg.advantage.keep_return_bounds
    if keep_return_bounds is not None:
        keep_return_bounds = tuple(keep_return_bounds)
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(capacity, device=device),
        sampler=SamplerWithoutReplacement(drop_last=False),
        batch_size=cfg.loss.mini_batch_size,
    )
    rb.append_transform(
        MCAdvantage(
            grpo_size=cfg.collector.group_size,
            prompt_key="group_id",
            trajectory_return=cfg.advantage.trajectory_return,
            keep_return_bounds=keep_return_bounds,
        )
    )
    return rb


def make_loss_module(cfg, policy: TinyVLA) -> ClipPPOLoss:
    clip_epsilon = cfg.loss.clip_epsilon
    if not isinstance(clip_epsilon, float):
        clip_epsilon = tuple(clip_epsilon)
    loss_module = ClipPPOLoss(
        actor_network=policy,
        critic_network=None,
        clip_epsilon=clip_epsilon,
        entropy_bonus=False,
    )
    loss_module.set_keys(
        action="action_tokens", sample_log_prob="log_probs", advantage="advantage"
    )
    return loss_module


def evaluate(env: TransformedEnv, policy: TinyVLA, cfg) -> float:
    """Greedy success rate over ``cfg.logger.eval_episodes`` episodes."""
    mode = policy.mode
    policy.mode = "greedy"
    successes = 0
    with torch.no_grad():
        for _ in range(cfg.logger.eval_episodes):
            rollout = env.rollout(cfg.env.max_outer_steps, policy)
            successes += int(rollout["next", "success"].any())
    policy.mode = mode
    return successes / cfg.logger.eval_episodes


def log_metrics(logger, metrics: dict, step: int) -> None:
    # One log_metrics call per step (not a per-key log_scalar loop): log_scalar
    # defaults to commit=False, so looping it never advances WandB's step and
    # every iteration collapses into a single history record. log_metrics
    # commits the whole step at once -- the pattern every other sota script uses.
    if logger is not None:
        logger.log_metrics(metrics, step)
