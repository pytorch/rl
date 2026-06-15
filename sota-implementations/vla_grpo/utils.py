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

Two backends share the loop: the dependency-free toy scale (``env.backend:
toy`` + ``policy.backend: tiny``) and the SimpleVLA-RL LIBERO scale
(``env.backend: libero`` + ``policy.backend: openvla``, one MuJoCo process
per parallel worker, grouped initial states with per-worker group-id
offsets).
"""
from __future__ import annotations

import warnings
from functools import partial

import torch

from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.vla import ActionTokenizerBase, UniformActionTokenizer
from torchrl.envs import (
    ActionTokenizerTransform,
    Compose,
    EnvBase,
    LiberoEnv,
    MultiAction,
    ParallelEnv,
    StepCounter,
    SuccessReward,
    ToyVLAEnv,
    TransformedEnv,
)
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules.vla import TinyVLA, VLAWrapperBase
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.llm import MCAdvantage
from torchrl.record import VideoRecorder

# group ids must be unique across parallel workers: each worker gets a
# disjoint offset block
GROUP_ID_OFFSET = 10**6


def make_policy(cfg, device: torch.device) -> VLAWrapperBase:
    # ratio_level="token" gives one importance ratio per action token (the
    # SimpleVLA-RL / DAPO semantics: the clip thresholds are per-token);
    # "sequence" sums the chunk's log-probs into a single ratio per decision.
    log_probs_mode = "token" if cfg.loss.ratio_level == "token" else "sequence"
    if cfg.policy.backend == "tiny":
        return TinyVLA(
            action_dim=cfg.env.action_dim,
            chunk_size=cfg.env.chunk_size,
            action_head="tokens",
            vocab_size=cfg.tokenizer.vocab_size,
            hidden_dim=cfg.policy.hidden_dim,
            log_probs_mode=log_probs_mode,
            device=device,
        )
    if cfg.policy.backend == "openvla":
        # local import: pulls in transformers/timm via the vendored modeling
        from openvla import OpenVLAOFTWrapper

        policy = OpenVLAOFTWrapper.from_pretrained(
            cfg.policy.checkpoint,
            torch_dtype=getattr(torch, cfg.policy.dtype),
            device=device,
            unnorm_key=cfg.policy.unnorm_key,
            dataset_statistics=cfg.policy.get("dataset_statistics", None),
            temperature=cfg.policy.temperature,
            log_probs_mode=log_probs_mode,
            use_wrist_image=cfg.policy.use_wrist_image,
            center_crop=cfg.policy.center_crop,
        )
        if cfg.policy.lora_rank:
            # de-risk fallback to full fine-tuning (RL4VLA shows LoRA r=32
            # works); validate on the target hardware
            from peft import get_peft_model, LoraConfig

            policy.model = get_peft_model(
                policy.model,
                LoraConfig(
                    r=cfg.policy.lora_rank,
                    lora_alpha=2 * cfg.policy.lora_rank,
                    target_modules=list(cfg.policy.lora_target_modules),
                ),
            )
        return policy
    raise ValueError(f"Unknown policy backend {cfg.policy.backend!r}.")


def make_action_tokenizer(cfg, policy: VLAWrapperBase) -> ActionTokenizerBase:
    if cfg.policy.backend == "openvla":
        # the codec lives in the checkpoint (vocab-tail mapping + norm_stats)
        return policy.action_tokenizer()
    return UniformActionTokenizer(cfg.tokenizer.vocab_size, low=-1.0, high=1.0)


def _chunk_transform(cfg, tokenizer: ActionTokenizerBase) -> Compose:
    # The compose order is load-bearing: the inverse (action-input) path runs
    # in reverse, so the tokenizer decode happens before MultiAction unbinds
    # the chunk; on the step path SuccessReward and StepCounter run after
    # MultiAction, i.e. once per outer (decision) step. stack_rewards=False
    # keeps the outer transition dense when an episode ends inside a chunk
    # (the decision reward comes from the outer success flag instead).
    return Compose(
        MultiAction(stack_rewards=False),
        ActionTokenizerTransform(tokenizer),
        SuccessReward(),
        StepCounter(max_steps=cfg.env.max_outer_steps),
    )


def _make_libero_worker(
    cfg, worker_idx: int, *, group_repeats=None, eval_mode=False, from_pixels=False
):
    task_ids = list(cfg.env.task_ids)
    task_id = task_ids[worker_idx % len(task_ids)]
    return LiberoEnv(
        cfg.env.task_suite,
        task_id=task_id,
        camera_height=cfg.env.camera_height,
        camera_width=cfg.env.camera_width,
        wrist_camera="robot0_eye_in_hand" if cfg.policy.use_wrist_image else None,
        from_pixels=from_pixels,
        max_episode_steps=cfg.env.max_env_steps,
        init_state_mode="cycle" if eval_mode else "random",
        group_repeats=group_repeats,
        group_id_offset=worker_idx * GROUP_ID_OFFSET,
    )


def make_env(
    cfg,
    tokenizer: ActionTokenizerBase,
    *,
    group_repeats: int | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
    eval_mode: bool = False,
    from_pixels: bool = False,
    num_envs: int | None = None,
) -> TransformedEnv:
    if cfg.env.backend == "toy":
        base = ToyVLAEnv(
            action_dim=cfg.env.action_dim,
            state_dim=cfg.env.state_dim,
            image_shape=tuple(cfg.env.image_shape),
            from_pixels=from_pixels,
            render_size=cfg.env.get("render_size", 64),
            success_steps=cfg.env.success_steps,
            success_tol=cfg.env.success_tol,
            group_repeats=group_repeats,
            batch_size=[1],
            seed=seed,
            device=device,
        )
    elif cfg.env.backend == "libero":
        # an explicit num_envs (e.g. a 1-env video recorder) intentionally
        # samples a subset of tasks, so the task-coverage guard only applies
        # to the train/eval envs sized from the config
        override = num_envs is not None
        if num_envs is None:
            num_envs = cfg.env.eval_num_envs if eval_mode else cfg.env.num_envs
        task_ids = list(cfg.env.task_ids)
        # each worker hosts ONE MuJoCo task for its whole lifetime: fewer
        # workers than tasks would silently drop tasks from the run
        if not override and num_envs < len(task_ids):
            raise ValueError(
                f"{'eval_num_envs' if eval_mode else 'num_envs'} ({num_envs}) "
                f"must cover task_ids ({len(task_ids)} tasks): each worker is "
                "bound to one task; fewer workers would silently drop tasks."
            )
        if not override and num_envs % len(task_ids):
            warnings.warn(
                f"num_envs ({num_envs}) is not a multiple of the number of "
                f"tasks ({len(task_ids)}): tasks will be sampled unevenly."
            )
        base = ParallelEnv(
            num_envs,
            [
                partial(
                    _make_libero_worker,
                    cfg,
                    worker_idx,
                    group_repeats=group_repeats,
                    eval_mode=eval_mode,
                    from_pixels=from_pixels,
                )
                for worker_idx in range(num_envs)
            ],
            mp_start_method="spawn",
            # MuJoCo runs on CPU; pin the env device so the collector/rollout
            # cast the GPU policy's action back to CPU before stepping (else a
            # cuda action reaches the CPU transforms -> mixed-device error)
            device="cpu",
        )
        if seed is not None:
            base.set_seed(seed)
    else:
        raise ValueError(f"Unknown env backend {cfg.env.backend!r}.")
    return TransformedEnv(base, _chunk_transform(cfg, tokenizer))


def make_replay_buffer(
    cfg, device: torch.device
) -> tuple[TensorDictReplayBuffer, MCAdvantage]:
    # The buffer holds one iteration's decisions; the write path computes the
    # group-relative advantage (and drops degenerate groups) on whole
    # trajectories, the read path samples decisions without replacement. The
    # advantage transform is returned too so the training loop can flush its
    # incomplete-group queues at iteration boundaries.
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
    advantage = MCAdvantage(
        grpo_size=cfg.collector.group_size,
        prompt_key="group_id",
        trajectory_return=cfg.advantage.trajectory_return,
        keep_return_bounds=keep_return_bounds,
    )
    rb.append_transform(advantage)
    return rb, advantage


def make_loss_module(cfg, policy: VLAWrapperBase) -> ClipPPOLoss:
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


def make_optimizer(cfg, loss_module: ClipPPOLoss):
    optim = torch.optim.AdamW(
        loss_module.parameters(), lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )
    warmup = max(int(cfg.optim.warmup_updates), 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda step: min((step + 1) / warmup, 1.0)
    )
    return optim, scheduler


def make_collector(cfg, env: EnvBase, policy: VLAWrapperBase, device) -> Collector:
    """Endless synchronous collector yielding whole-trajectory batches.

    Each yielded batch holds exactly one iteration's worth of complete,
    done-terminated trajectories, concatenated along time
    (``trajs_per_batch`` with ``traj_format="cat"``: flat and unpadded,
    episodes delimited by the done flags -- no padding frames for the
    image-heavy VLA observations; episodes spanning internal collection
    steps are reassembled by the collector, in-flight episodes are held
    back). The policy is held by reference (in-place optimizer updates apply
    immediately) and observations/actions are cast between the env's and the
    policy's devices by the collector. ``exploration_type=RANDOM`` makes the
    collector roll out under a sampling context, which the token policy reads
    via :func:`~torchrl.envs.utils.exploration_type` -- no policy mutation, so
    this works with any collector (including multi-process workers).
    """
    num_envs = env.batch_size[0] if env.batch_size else 1
    return Collector(
        env,
        policy,
        frames_per_batch=num_envs * cfg.env.max_outer_steps,
        total_frames=-1,
        trajs_per_batch=cfg.collector.groups_per_iter * cfg.collector.group_size,
        traj_format="cat",
        exploration_type=ExplorationType.RANDOM,
        policy_device=device,
        reset_at_each_iter=False,
    )


def evaluate(env: TransformedEnv, policy: VLAWrapperBase, cfg) -> float:
    """Greedy success rate over (at least) ``cfg.logger.eval_episodes`` episodes.

    One evaluation round = one reset per env row + one episode per row, with
    no auto-resets in between (``break_when_all_done`` freezes finished rows
    and stops once every row is done). Each LIBERO reset therefore consumes
    exactly one cycled initial state, keeping the fixed-trials evaluation
    protocol exact, and no collected episode is discarded. Greedy decoding is
    requested through the exploration context, not by mutating the policy.
    """
    successes = 0.0
    episodes = 0
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        while episodes < cfg.logger.eval_episodes:
            reset_td = env.reset()
            rollout = env.rollout(
                cfg.env.max_outer_steps,
                policy,
                break_when_any_done=False,
                break_when_all_done=True,
                auto_reset=False,
                tensordict=reset_td,
                auto_cast_to_device=True,
            )
            # one episode per row: success anywhere along the (frozen-once-
            # done) row
            row_success = rollout["next", "success"].any(-2)
            successes += float(row_success.sum())
            episodes += int(row_success.numel())
    return successes / max(episodes, 1)


def make_record_env(cfg, tokenizer: ActionTokenizerBase, logger, device):
    """Single-environment eval recorder feeding a torchrl ``VideoRecorder``.

    Built with ``from_pixels=True`` so the base env emits a root ``pixels``
    frame (``ToyVLAEnv`` renders the tracking scene; ``LiberoEnv`` exposes the
    camera). A :class:`~torchrl.record.VideoRecorder` transform appended last
    collects those frames; :func:`record_eval_video` rolls out greedily and
    flushes them to ``logger``. One environment keeps the video a single clean
    stream (rather than a tiled grid of workers).
    """
    env = make_env(
        cfg,
        tokenizer,
        seed=cfg.env.seed + 2,
        device=device if cfg.env.backend == "toy" else None,
        eval_mode=True,
        from_pixels=True,
        num_envs=1,
    )
    recorder = VideoRecorder(
        logger,
        tag="eval/video",
        in_keys=["pixels"],
        skip=1,
        # single env -> a single video stream, no torchvision grid needed
        make_grid=False,
        fps=cfg.logger.video_fps,
    )
    env.append_transform(recorder)
    return env, recorder


def record_eval_video(env, recorder, policy: VLAWrapperBase, cfg, step: int) -> None:
    """Roll out ``cfg.logger.video_episodes`` greedy episodes and log one video.

    Each reset consumes one cycled initial state and ``break_when_all_done``
    stops at the episode's natural end -- the same protocol as :func:`evaluate`,
    so the recorded episodes mirror the measured ones. ``recorder.dump`` stacks
    every frame seen since the last dump into a single clip and writes it to the
    logger at ``step``. Greedy decoding is requested through the exploration
    context, matching :func:`evaluate`.
    """
    with set_exploration_type(ExplorationType.DETERMINISTIC), torch.no_grad():
        for _ in range(max(int(cfg.logger.video_episodes), 1)):
            reset_td = env.reset()
            env.rollout(
                cfg.env.max_outer_steps,
                policy,
                break_when_any_done=False,
                break_when_all_done=True,
                auto_reset=False,
                tensordict=reset_td,
                auto_cast_to_device=True,
            )
    recorder.dump(step=step)


def log_metrics(logger, metrics: dict, step: int) -> None:
    # One log_metrics call per step (not a per-key log_scalar loop): log_scalar
    # defaults to commit=False, so looping it never advances WandB's step and
    # every iteration collapses into a single history record. log_metrics
    # commits the whole step at once -- the pattern every other sota script uses.
    if logger is not None:
        logger.log_metrics(metrics, step)
