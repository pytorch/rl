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
accounting: ``groups_per_iter`` initial states x ``candidate_group_size``
rollout candidates per state, each contributing up to ``max_outer_steps``
decisions; the dynamic sampling filter and candidate selector decide how many
trajectories are useful for optimization, so the effective batch is variable.

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

import math
import multiprocessing as mp
import os
import queue
import tempfile
import warnings

import hydra
import torch
import tqdm
from tensordict import TensorDictBase
from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.record.loggers import generate_exp_name, get_logger
from utils import (
    candidate_group_size,
    evaluate,
    log_metrics,
    make_action_tokenizer,
    make_collector,
    make_env,
    make_loss_module,
    make_optimizer,
    make_policy,
    make_record_env,
    make_replay_buffer,
    record_eval_video,
)

warnings.filterwarnings("ignore", category=UserWarning, module="tensordict")


def _trainable_policy_state_dict(policy) -> dict[str, torch.Tensor]:
    """CPU state dict containing the trainable policy weights."""
    trainable_names = {
        name for name, parameter in policy.named_parameters() if parameter.requires_grad
    }
    state_dict = policy.state_dict()
    if not trainable_names:
        trainable_names = set(state_dict)
    return {
        name: tensor.detach().cpu()
        for name, tensor in state_dict.items()
        if name in trainable_names
    }


def _load_trainable_policy_state_dict(policy, state_dict: dict[str, torch.Tensor]):
    missing, unexpected = policy.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"Unexpected policy checkpoint keys: {', '.join(unexpected)}"
        )
    if len(missing) == len(policy.state_dict()):
        raise RuntimeError("Policy checkpoint did not match policy.")
    return missing


def _eval_device(cfg) -> torch.device:
    return torch.device(
        cfg.logger.get("eval_device", None)
        or cfg.policy.device
        or ("cuda:0" if torch.cuda.is_available() else "cpu")
    )


def _evaluator_worker(request_queue, response_queue, cfg) -> None:
    eval_env = None
    try:
        torch.manual_seed(cfg.env.seed + 1)
        device = _eval_device(cfg)
        policy = make_policy(cfg, device)
        tokenizer = make_action_tokenizer(cfg, policy)
        eval_env = make_env(
            cfg,
            tokenizer,
            seed=cfg.env.seed + 1,
            device=device if cfg.env.backend == "toy" else None,
            eval_mode=True,
        )
        with torch.no_grad():
            policy(eval_env.fake_tensordict().to(device))
        response_queue.put(("ready", None, None))
        while True:
            request = request_queue.get()
            if request is None:
                break
            iteration, state_path = request
            try:
                state_dict = torch.load(state_path, map_location="cpu")
                _load_trainable_policy_state_dict(policy, state_dict)
                success = evaluate(eval_env, policy, cfg)
                response_queue.put((iteration, success, None))
            except Exception as err:
                response_queue.put((iteration, None, repr(err)))
            finally:
                if os.path.exists(state_path):
                    os.unlink(state_path)
    except Exception as err:
        response_queue.put((-1, None, repr(err)))
    finally:
        if eval_env is not None:
            eval_env.close(raise_if_closed=False)


class _EvaluatorProcess:
    def __init__(self, cfg) -> None:
        self._ctx = mp.get_context("spawn")
        self._requests = self._ctx.Queue(maxsize=1)
        self._responses = self._ctx.Queue(maxsize=1)
        self._state_dir = (
            cfg.logger.get("eval_state_dir", None) or tempfile.gettempdir()
        )
        self._pending_iteration = None
        self._pending_state_path = None
        os.makedirs(self._state_dir, exist_ok=True)
        self._process = self._ctx.Process(
            target=_evaluator_worker,
            args=(self._requests, self._responses, cfg),
        )
        self._process.start()
        ready, _, error = self._recv()
        if error is not None:
            raise RuntimeError(f"Evaluator failed to initialize: {error}")
        if ready != "ready":
            raise RuntimeError(f"Unexpected evaluator init message: {ready!r}.")

    @property
    def has_pending(self) -> bool:
        return self._pending_iteration is not None

    @property
    def pending_iteration(self) -> int | None:
        return self._pending_iteration

    def submit(self, policy, iteration: int) -> bool:
        if self.has_pending:
            self._raise_if_failed()
            return False
        if not self._process.is_alive():
            self._raise_if_failed()
        state_path = self._make_state_path(iteration)
        torch.save(_trainable_policy_state_dict(policy), state_path)
        self._requests.put((iteration, state_path))
        self._pending_iteration = iteration
        self._pending_state_path = state_path
        return True

    def poll(self) -> tuple[int, float] | None:
        if not self.has_pending:
            self._raise_if_failed()
            return None
        try:
            response = self._responses.get_nowait()
        except queue.Empty:
            self._raise_if_failed()
            return None
        return self._handle_response(response)

    def wait(self) -> tuple[int, float]:
        if not self.has_pending:
            raise RuntimeError("No evaluator request is pending.")
        return self._handle_response(self._recv())

    def evaluate(self, policy, iteration: int) -> float:
        if not self.submit(policy, iteration):
            raise RuntimeError(
                f"Evaluator already has a pending request for iteration "
                f"{self._pending_iteration}."
            )
        _, success = self.wait()
        return success

    def close(self) -> None:
        if self._pending_state_path is not None and os.path.exists(
            self._pending_state_path
        ):
            os.unlink(self._pending_state_path)
        self._pending_iteration = None
        self._pending_state_path = None
        if self._process.is_alive():
            self._requests.put(None)
            self._process.join(timeout=30)
        if self._process.is_alive():
            self._process.terminate()
            self._process.join(timeout=30)

    def _make_state_path(self, iteration: int) -> str:
        return os.path.join(
            self._state_dir,
            f"vla_grpo_eval_state_{os.getpid()}_{iteration}.pt",
        )

    def _handle_response(self, response) -> tuple[int, float]:
        eval_iteration, success, error = response
        state_path = self._pending_state_path
        pending_iteration = self._pending_iteration
        self._pending_iteration = None
        self._pending_state_path = None
        if state_path is not None and os.path.exists(state_path):
            os.unlink(state_path)
        if error is not None:
            raise RuntimeError(
                f"Evaluator failed at iteration {eval_iteration}: {error}"
            )
        if eval_iteration != pending_iteration:
            raise RuntimeError(
                f"Evaluator returned iteration {eval_iteration}, "
                f"expected {pending_iteration}."
            )
        return eval_iteration, success

    def _recv(self):
        while True:
            try:
                return self._responses.get(timeout=1.0)
            except queue.Empty:
                self._raise_if_failed()

    def _raise_if_failed(self) -> None:
        if self._process.is_alive():
            return
        try:
            _, _, error = self._responses.get_nowait()
        except queue.Empty as err:
            raise RuntimeError("Evaluator process exited without a result.") from err
        raise RuntimeError(f"Evaluator process failed to initialize: {error}")


class _CollectorStats:
    """Accumulate raw rollout diagnostics from collector writer polls."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.raw_decisions = 0
        self.raw_trajectories = 0
        self._episode_lengths = []
        self._episode_successes = []
        self._episode_returns = []

    def __call__(self, batch: TensorDictBase) -> None:
        self.raw_decisions += int(batch.numel())
        try:
            flat = batch.reshape(-1)
            done = flat["next", "done"].squeeze(-1).bool()
        except KeyError:
            return
        if not done.any():
            return
        traj_ids = flat.get(("collector", "traj_ids"), None)
        if traj_ids is None:
            self._record_done_delimited(flat, done)
            return
        traj_ids = traj_ids.reshape(-1)
        valid_done = done & traj_ids.ge(0)
        if not valid_done.any():
            return
        for traj_id in torch.unique(traj_ids[valid_done]).detach().cpu().tolist():
            traj_mask = traj_ids == traj_id
            self._record_trajectory(flat, traj_mask)

    @property
    def episode_lengths(self) -> torch.Tensor:
        if not self._episode_lengths:
            return torch.zeros(0, dtype=torch.long)
        return torch.stack(self._episode_lengths)

    @property
    def episode_successes(self) -> torch.Tensor:
        if not self._episode_successes:
            return torch.zeros(0)
        return torch.stack(self._episode_successes)

    @property
    def episode_returns(self) -> torch.Tensor:
        if not self._episode_returns:
            return torch.zeros(0)
        return torch.stack(self._episode_returns)

    def _record_done_delimited(self, flat: TensorDictBase, done: torch.Tensor) -> None:
        episode_idx = done.long().cumsum(0) - done.long()
        episodes = int(done.sum())
        for episode in range(episodes):
            self._record_trajectory(flat, episode_idx == episode)

    def _record_trajectory(self, flat: TensorDictBase, traj_mask: torch.Tensor) -> None:
        if not bool(traj_mask.any()):
            return
        success = flat.get(("next", "success"), None)
        reward = flat.get(("next", "reward"), None)
        self.raw_trajectories += 1
        self._episode_lengths.append(traj_mask.sum().detach().cpu())
        if success is None:
            self._episode_successes.append(torch.zeros(()))
        else:
            self._episode_successes.append(
                success.reshape(-1)[traj_mask].float().max().detach().cpu()
            )
        if reward is None:
            self._episode_returns.append(torch.zeros(()))
        else:
            self._episode_returns.append(
                reward.reshape(-1)[traj_mask].float().sum().detach().cpu()
            )


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
    collection_group_size = candidate_group_size(cfg)
    env_get = getattr(
        cfg.env,
        "get",
        lambda key, default=None: getattr(cfg.env, key, default),
    )
    train_group_repeats = (
        cfg.collector.group_size
        if env_get("parallel_group_repeats", False)
        else collection_group_size
    )
    train_env = make_env(
        cfg,
        tokenizer,
        group_repeats=train_group_repeats,
        seed=cfg.env.seed,
        device=device if cfg.env.backend == "toy" else None,
    )
    eval_process = bool(cfg.logger.get("eval_process", False))
    eval_env = None
    if not eval_process:
        eval_env = make_env(
            cfg,
            tokenizer,
            seed=cfg.env.seed + 1,
            device=device if cfg.env.backend == "toy" else None,
            eval_mode=True,
        )
    # materialize lazy layers (TinyVLA) on a spec-shaped fake observation: a
    # real reset would consume a grouped init (train env) or a cycled eval
    # state (eval env)
    with torch.no_grad():
        fake_env = eval_env if eval_env is not None else train_env
        fake_td = fake_env.fake_tensordict()
        policy(fake_td.to(device))

    rollout_device = torch.device(cfg.collector.get("policy_device", None) or device)
    if rollout_device == device:
        rollout_policy = policy
    else:
        rollout_policy = make_policy(cfg, rollout_device)
        with torch.no_grad():
            rollout_policy(fake_td.to(rollout_device))
        _load_trainable_policy_state_dict(
            rollout_policy, _trainable_policy_state_dict(policy)
        )

    # optional eval-video recorder: a separate single-env rollout rendered to
    # pixels and written through a torchrl VideoRecorder (logged to wandb /
    # tensorboard / csv). Off by default and only built when a logger exists.
    record_env = recorder = None
    if cfg.logger.record_video and eval_process:
        warnings.warn(
            "logger.record_video is disabled when logger.eval_process=true; "
            "the evaluator process returns scalar success metrics only."
        )
    elif cfg.logger.record_video and logger is not None:
        record_env, recorder = make_record_env(cfg, tokenizer, logger, device)
    evaluator = _EvaluatorProcess(cfg) if eval_process else None
    async_eval = bool(eval_process and cfg.logger.get("eval_async", False))

    buffer_device = torch.device(cfg.buffer.device) if cfg.buffer.device else device
    replay_buffer, advantage_transform = make_replay_buffer(cfg, buffer_device)
    loss_module = make_loss_module(cfg, policy)
    optim, scheduler = make_optimizer(cfg, loss_module)
    collector_stats = _CollectorStats()
    collector = make_collector(
        cfg,
        train_env,
        rollout_policy,
        rollout_device,
        replay_buffer=replay_buffer,
        post_collect_hook=collector_stats,
    )
    collector_iter = iter(collector)

    start_iter = 0
    if cfg.checkpoint.resume:
        start_iter = load_checkpoint(cfg.checkpoint.resume, policy, optim, scheduler)
        if rollout_policy is not policy:
            _load_trainable_policy_state_dict(
                rollout_policy, _trainable_policy_state_dict(policy)
            )
        torchrl_logger.info(
            f"Resumed from {cfg.checkpoint.resume} at iteration {start_iter}."
        )

    episodes_per_iter = cfg.collector.groups_per_iter * collection_group_size
    max_collect_batches_per_iter = max(
        int(cfg.collector.get("max_collect_batches_per_iter", 1)), 1
    )
    max_same_policy_collect_attempts = max(
        int(cfg.collector.get("max_same_policy_collect_attempts", 1)), 1
    )
    min_replay_decisions = int(cfg.collector.get("min_replay_decisions", 0) or 0)
    num_envs = train_env.batch_size[0] if train_env.batch_size else 1
    collector_frames_per_poll = max(
        int(getattr(collector, "requested_frames_per_batch", num_envs)), 1
    )
    collect_polls_per_group_wave = max(
        math.ceil(
            episodes_per_iter * int(cfg.env.max_outer_steps) / collector_frames_per_poll
        ),
        1,
    )
    max_collect_polls_per_iter = (
        collect_polls_per_group_wave * max_collect_batches_per_iter
    )
    accumulate = max(int(cfg.loss.accumulate_batches), 1)
    pbar = tqdm.tqdm(total=cfg.collector.total_iters, initial=start_iter)
    total_episodes = start_iter * episodes_per_iter

    iteration = start_iter
    retry_same_policy = False
    same_policy_collect_polls = 0
    same_policy_collect_time = 0.0
    same_policy_collect_attempts = 0
    same_policy_safety_cap_hits = 0
    while iteration < cfg.collector.total_iters:
        # Collect under a fixed rollout policy until enough useful decisions
        # have reached the replay buffer (or the safety cap is hit). The
        # collector writer pushes complete trajectories directly to the replay
        # buffer as internal rollout polls finish; MCAdvantage is the replay
        # buffer transform, so incomplete GRPO groups stay queued across these
        # same-policy polls instead of being discarded at a collector-batch
        # boundary. Rollouts sample (rather than argmax) because the collector
        # runs under exploration_type=RANDOM (set in make_collector); the policy
        # reads that context, so the script never mutates it.
        with timeit("collect") as collect_timer:
            if not retry_same_policy:
                collector_stats.reset()
                advantage_transform.reset_stats()
                same_policy_collect_polls = 0
                same_policy_collect_time = 0.0
                same_policy_collect_attempts = 0
                same_policy_safety_cap_hits = 0
            retry_same_policy = False
            same_policy_collect_attempts += 1
            collect_polls = 0
            safety_cap_hit = False
            progress_log_interval = max(collect_polls_per_group_wave // 4, 1)
            while collect_polls < max_collect_polls_per_iter:
                next(collector_iter)
                collect_polls += 1
                replay_decisions = len(replay_buffer)
                reached_replay_target = (
                    min_replay_decisions > 0
                    and replay_decisions >= min_replay_decisions
                )
                should_log_progress = (
                    collect_polls % progress_log_interval == 0
                    or reached_replay_target
                    or collect_polls == max_collect_polls_per_iter
                )
                if should_log_progress:
                    torchrl_logger.info(
                        "collection progress iteration %d polls %d/%d "
                        "waves %.2f replay_decisions %d/%d raw_decisions %d "
                        "completed_trajs %d kept_groups %d skipped_groups %d "
                        "rescued_groups %d queued_trajs %d max_queue %d "
                        "elapsed_s %.1f total_waves %.2f attempts %d",
                        iteration,
                        collect_polls,
                        max_collect_polls_per_iter,
                        collect_polls / collect_polls_per_group_wave,
                        replay_decisions,
                        min_replay_decisions,
                        collector_stats.raw_decisions,
                        advantage_transform.completed_trajectories,
                        advantage_transform.written_groups,
                        advantage_transform.dropped_groups,
                        advantage_transform.rescued_groups,
                        advantage_transform.queued_trajectories,
                        advantage_transform.max_queued_trajectories_per_group,
                        collect_timer.elapsed(),
                        (same_policy_collect_polls + collect_polls)
                        / collect_polls_per_group_wave,
                        same_policy_collect_attempts,
                    )
                if min_replay_decisions > 0:
                    if reached_replay_target:
                        break
                elif collect_polls >= collect_polls_per_group_wave:
                    break
            else:
                safety_cap_hit = True

            if min_replay_decisions > 0 and len(replay_buffer) < min_replay_decisions:
                safety_cap_hit = True

            attempt_collect_time = collect_timer.elapsed()
            same_policy_collect_time += attempt_collect_time
            same_policy_collect_polls += collect_polls
            same_policy_safety_cap_hits += int(safety_cap_hit)
            collect_group_waves = (
                same_policy_collect_polls / collect_polls_per_group_wave
            )
            completed_trajectories = advantage_transform.completed_trajectories
            completed_decisions = advantage_transform.completed_decisions
            group_metrics = {
                "buffer/collect_batches": collect_group_waves,
                "buffer/collect_group_waves": collect_group_waves,
                "buffer/collect_polls": same_policy_collect_polls,
                "buffer/current_collect_polls": collect_polls,
                "buffer/collect_attempts": same_policy_collect_attempts,
                "buffer/collect_safety_cap_hit": float(same_policy_safety_cap_hits > 0),
                "buffer/collect_safety_cap_hits": same_policy_safety_cap_hits,
                "buffer/current_collect_safety_cap_hit": float(safety_cap_hit),
                "buffer/max_same_policy_collect_attempts": (
                    max_same_policy_collect_attempts
                ),
                "buffer/min_replay_decisions": min_replay_decisions,
                "buffer/grpo_group_size": cfg.collector.group_size,
                "buffer/candidate_group_size": collection_group_size,
                "buffer/complete_groups": advantage_transform.completed_groups,
                "buffer/complete_groups_written": advantage_transform.written_groups,
                "buffer/kept_groups": advantage_transform.written_groups,
                "buffer/dropped_dynamic_sampling_groups": (
                    advantage_transform.dropped_groups
                ),
                "buffer/dropped_complete_groups": advantage_transform.dropped_groups,
                "buffer/skipped_groups": advantage_transform.dropped_groups,
                "buffer/rescued_oversampled_groups": (
                    advantage_transform.rescued_groups
                ),
                "buffer/selected_trajectories": (
                    advantage_transform.selected_trajectories
                ),
                "buffer/unselected_candidate_trajectories": (
                    advantage_transform.unselected_trajectories
                ),
                "buffer/partial_groups": advantage_transform.queued_groups,
                "buffer/queued_groups": advantage_transform.queued_groups,
                "buffer/queued_incomplete_groups": advantage_transform.queued_groups,
                "buffer/queued_trajectories": (advantage_transform.queued_trajectories),
                "buffer/queued_incomplete_trajectories": (
                    advantage_transform.queued_trajectories
                ),
                "buffer/max_queued_trajectories_per_group": (
                    advantage_transform.max_queued_trajectories_per_group
                ),
                "collector/raw_decisions": collector_stats.raw_decisions,
                "collector/completed_decisions": completed_decisions,
                "collector/raw_trajectories": completed_trajectories,
                "collector/completed_trajectories": completed_trajectories,
                "collector/successful_trajectories": (
                    advantage_transform.successful_trajectories
                ),
                "collector/trajectory_return_sum": (
                    advantage_transform.trajectory_return_sum
                ),
                "collector/trajectory_return_max": (
                    advantage_transform.trajectory_return_max
                    if completed_trajectories
                    else 0.0
                ),
            }
        # PPO update over the decisions that survived dynamic sampling, with
        # gradient accumulation (micro-batches of mini_batch_size decisions)
        num_decisions = len(replay_buffer)
        insufficient_replay = (
            min_replay_decisions > 0 and num_decisions < min_replay_decisions
        )
        if (
            insufficient_replay
            and advantage_transform.queued_trajectories
            and same_policy_collect_attempts < max_same_policy_collect_attempts
        ):
            timeit.erase()
            torchrl_logger.warning(
                "iteration %d hit the collection safety cap with only %d/%d "
                "replay decisions and still has %d queued trajectories across "
                "%d groups (max_queue=%d/%d). Keeping the replay buffer and "
                "MCAdvantage queues, then continuing collection under the same "
                "policy instead of treating this as a policy-update boundary. "
                "attempt %d/%d collect_s %.1f",
                iteration,
                num_decisions,
                min_replay_decisions,
                advantage_transform.queued_trajectories,
                advantage_transform.queued_groups,
                advantage_transform.max_queued_trajectories_per_group,
                collection_group_size,
                same_policy_collect_attempts,
                max_same_policy_collect_attempts,
                same_policy_collect_time,
            )
            retry_same_policy = True
            continue
        if insufficient_replay and advantage_transform.queued_trajectories:
            torchrl_logger.warning(
                "iteration %d reached the same-policy collection attempt cap "
                "with only %d/%d replay decisions and %d queued trajectories "
                "across %d groups (max_queue=%d/%d). Proceeding with an "
                "undersized policy update and clearing in-flight groups at the "
                "policy boundary. attempts %d/%d collect_s %.1f",
                iteration,
                num_decisions,
                min_replay_decisions,
                advantage_transform.queued_trajectories,
                advantage_transform.queued_groups,
                advantage_transform.max_queued_trajectories_per_group,
                collection_group_size,
                same_policy_collect_attempts,
                max_same_policy_collect_attempts,
                same_policy_collect_time,
            )
        total_episodes += completed_trajectories
        losses = []
        clip_fractions = []
        ess = []
        grad_norms = []
        optim_steps = 0
        train_decisions = 0

        def optimizer_step(grad_norms=grad_norms):
            nonlocal optim_steps
            grad_norms.append(
                torch.nn.utils.clip_grad_norm_(
                    loss_module.parameters(), cfg.optim.max_grad_norm
                )
            )
            optim.step()
            optim.zero_grad(set_to_none=True)
            scheduler.step()
            optim_steps += 1

        micro_batches = 0
        with timeit("train"):
            for _ in range(cfg.loss.ppo_epochs):
                if not num_decisions:
                    break
                for batch in replay_buffer:
                    batch = batch.to(device)
                    train_decisions += batch.shape[0]
                    # ratio_level="token" gives per-token importance ratios;
                    # ClipPPOLoss broadcasts the per-decision advantage over the
                    # token dims itself, so the script passes it through as-is.
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
        # Synchronous policy boundary: old-policy in-flight episodes and
        # incomplete GRPO groups must not leak into the next rollout policy.
        advantage_transform.queues.clear()
        collector.reset()
        if rollout_policy is not policy:
            _load_trainable_policy_state_dict(
                rollout_policy, _trainable_policy_state_dict(policy)
            )

        eval_success = None
        eval_source_iteration = None
        if async_eval and evaluator is not None:
            eval_result = evaluator.poll()
            if eval_result is not None:
                eval_source_iteration, eval_success = eval_result
                torchrl_logger.info(
                    "async eval completed for iteration %d success %.3f",
                    eval_source_iteration,
                    eval_success,
                )
        if iteration % cfg.logger.eval_iter == 0:
            if evaluator is None:
                with timeit("eval"):
                    eval_success = evaluate(eval_env, policy, cfg)
                    eval_source_iteration = iteration
            elif async_eval:
                with timeit("eval_submit"):
                    eval_result = evaluator.poll()
                    if eval_result is not None:
                        eval_source_iteration, eval_success = eval_result
                        torchrl_logger.info(
                            "async eval completed for iteration %d success %.3f",
                            eval_source_iteration,
                            eval_success,
                        )
                    if evaluator.submit(policy, iteration):
                        torchrl_logger.info(
                            "submitted async eval for iteration %d", iteration
                        )
                    else:
                        torchrl_logger.warning(
                            "skipped async eval for iteration %d because "
                            "iteration %d is still pending",
                            iteration,
                            evaluator.pending_iteration,
                        )
            else:
                with timeit("eval"):
                    eval_success = evaluator.evaluate(policy, iteration)
                    eval_source_iteration = iteration
            if recorder is not None:
                record_eval_video(record_env, recorder, policy, cfg, iteration)

        timings = timeit.todict(prefix="time")
        timings["time/collect_current_attempt"] = timings.get("time/collect", 0.0)
        timings["time/collect"] = same_policy_collect_time
        timeit.erase()
        collect_time = max(timings.get("time/collect", 0.0), 1e-9)
        train_time = max(timings.get("time/train", 0.0), 1e-9)
        num_decisions_collected = int(group_metrics["collector/raw_decisions"])
        completed_episode_decisions = int(
            group_metrics["collector/completed_decisions"]
        )
        completed_trajectories = int(group_metrics["collector/completed_trajectories"])
        env_steps = num_decisions_collected * cfg.env.chunk_size
        success_rate = float(group_metrics["collector/successful_trajectories"]) / max(
            completed_trajectories, 1
        )
        episode_decisions = completed_episode_decisions / max(completed_trajectories, 1)
        reward_mean = float(group_metrics["collector/trajectory_return_sum"]) / max(
            completed_trajectories, 1
        )
        reward_max = float(group_metrics["collector/trajectory_return_max"])
        metrics = {
            "train/success_rate": success_rate,
            "train/episode_decisions": episode_decisions,
            "train/episodes_total": total_episodes,
            "train/lr": scheduler.get_last_lr()[0],
            # reward curves: per-episode return (binary-success reward summed
            # over the episode), averaged and best-of-batch
            "train/reward_mean": reward_mean,
            "train/reward_max": reward_max,
            "buffer/decisions": num_decisions,
            "buffer/useful_replay_decisions": num_decisions,
            "buffer/kept_fraction": num_decisions / max(1, num_decisions_collected),
            "collector/completed_episode_decisions": completed_episode_decisions,
            "throughput/replay_decisions_per_s": num_decisions / collect_time,
            # inference throughput: env steps / decisions generated per second
            # during collection (policy rollout)
            "throughput/inference_env_steps_per_s": env_steps / collect_time,
            "throughput/inference_decisions_per_s": num_decisions_collected
            / collect_time,
            # training throughput: decisions consumed / optimizer steps taken
            # per second during the PPO update
            "throughput/train_decisions_per_s": train_decisions / train_time,
            "throughput/optim_steps_per_s": optim_steps / train_time,
        }
        metrics.update(group_metrics)
        metrics.update(timings)
        if eval_success is not None:
            metrics["eval/success_rate"] = eval_success
            metrics["eval/source_iteration"] = eval_source_iteration
        if losses:
            metrics.update(
                {
                    "train/loss_objective": torch.stack(losses).mean().item(),
                    "train/clip_fraction": torch.stack(clip_fractions).mean().item(),
                    "train/ESS": torch.stack(ess).mean().item(),
                    "train/grad_norm": torch.stack(grad_norms).mean().item(),
                }
            )
        torchrl_logger.info(
            "iteration %d success %.3f decisions %d raw_decisions %d "
            "raw_trajs %d collect_waves %.2f polls %d kept_groups %d "
            "skipped_groups %d rescued_groups %d queued_trajs %d "
            "max_queue %d collect_s %.1f train_s %.1f",
            iteration,
            metrics["train/success_rate"],
            num_decisions,
            num_decisions_collected,
            int(metrics.get("collector/raw_trajectories", 0)),
            float(metrics.get("buffer/collect_group_waves", 0.0)),
            int(metrics.get("buffer/collect_polls", 0)),
            int(metrics.get("buffer/kept_groups", 0)),
            int(metrics.get("buffer/dropped_dynamic_sampling_groups", 0)),
            int(metrics.get("buffer/rescued_oversampled_groups", 0)),
            int(metrics.get("buffer/queued_trajectories", 0)),
            int(metrics.get("buffer/max_queued_trajectories_per_group", 0)),
            timings.get("time/collect", 0.0),
            timings.get("time/train", 0.0),
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
        iteration += 1

    if cfg.checkpoint.save_iter:
        save_checkpoint(
            os.path.join(os.getcwd(), "checkpoint_latest.pt"),
            policy,
            optim,
            scheduler,
            cfg.collector.total_iters - 1,
        )
    if async_eval and evaluator is not None and evaluator.has_pending:
        eval_source_iteration, final_success = evaluator.wait()
        torchrl_logger.info(
            "async eval completed for iteration %d success %.3f",
            eval_source_iteration,
            final_success,
        )
        if logger is not None:
            log_metrics(
                logger,
                {
                    "eval/success_rate": final_success,
                    "eval/source_iteration": eval_source_iteration,
                },
                cfg.collector.total_iters,
            )
    pbar.close()
    if logger is not None:
        if evaluator is None:
            final_success = evaluate(eval_env, policy, cfg)
        else:
            final_success = evaluator.evaluate(policy, cfg.collector.total_iters)
        log_metrics(
            logger, {"eval/success_rate": final_success}, cfg.collector.total_iters
        )
        torchrl_logger.info(f"Final greedy success rate: {final_success:.3f}")
        if recorder is not None:
            record_eval_video(
                record_env, recorder, policy, cfg, cfg.collector.total_iters
            )
    collector.shutdown()
    train_env.close(raise_if_closed=False)
    if eval_env is not None:
        eval_env.close(raise_if_closed=False)
    if record_env is not None:
        record_env.close(raise_if_closed=False)
    if evaluator is not None:
        evaluator.close()


if __name__ == "__main__":
    main()
