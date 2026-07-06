# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Factories for the VLA GRPO training script.

The environment factory wires the chunk-decision data path: one outer step of
the transformed environment is one policy (chunk) decision. The policy emits
``("vla_action", "tokens")`` for a whole chunk in a single forward; the
tokenizer transform decodes them into the continuous chunk on the inverse path;
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

import os
import warnings
from collections.abc import Callable
from functools import partial

import torch

from tensordict import lazy_stack, TensorDictBase
from torchrl.collectors import AsyncBatchedCollector, Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SamplerWithoutReplacement
from torchrl.data.vla import (
    ACTION_TOKENS_KEY,
    ActionTokenizerBase,
    UniformActionTokenizer,
)
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
from torchrl.modules.inference_server import (
    InferenceServer,
    InferenceServerConfig,
    PolicyClientModule,
    ThreadingTransport,
)
from torchrl.modules.vla import TinyVLA, VLAWrapperBase
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.llm import MCAdvantage, MCAdvantageSelector
from torchrl.record import VideoRecorder

# group ids must be unique across parallel workers: each worker gets a
# disjoint offset block
GROUP_ID_OFFSET = 10**6
LOG_PROBS_KEY = ("vla_action", "log_probs")


def _cfg_get(section, key: str, default=None):
    get = getattr(section, "get", None)
    if get is not None:
        return get(key, default)
    return getattr(section, key, default)


def candidate_group_size(cfg) -> int:
    """Number of rollout candidates collected for each GRPO group."""
    collector_get = getattr(
        cfg.collector,
        "get",
        lambda key, default=None: getattr(cfg.collector, key, default),
    )
    return int(collector_get("candidate_group_size", None) or cfg.collector.group_size)


def _configure_mujoco_rendering(cfg) -> None:
    render_backend = cfg.env.get("render_backend", None)
    if render_backend is None:
        return
    render_backend = str(render_backend)
    os.environ["MUJOCO_GL"] = render_backend
    if render_backend in ("egl", "osmesa"):
        os.environ["PYOPENGL_PLATFORM"] = render_backend


def _worker_render_gpu_device_id(
    cfg, worker_idx: int, *, eval_mode: bool = False
) -> int | None:
    render_backend = cfg.env.get("render_backend", None)
    if render_backend is not None and str(render_backend) != "egl":
        return None
    render_gpu_ids = cfg.env.get("eval_render_gpu_ids", None) if eval_mode else None
    if render_gpu_ids is None:
        render_gpu_ids = cfg.env.get("render_gpu_ids", None)
    if render_gpu_ids is None:
        return None
    render_gpu_ids = [int(device_id) for device_id in render_gpu_ids]
    if not render_gpu_ids:
        return None
    return render_gpu_ids[worker_idx % len(render_gpu_ids)]


def _libero_worker_assignment(
    cfg, worker_idx: int, *, group_repeats=None, eval_mode=False
) -> tuple[int, int | None, int]:
    task_ids = list(cfg.env.task_ids)
    if (
        eval_mode
        or group_repeats is None
        or not cfg.env.get("parallel_group_repeats", False)
    ):
        return (
            task_ids[worker_idx % len(task_ids)],
            group_repeats,
            worker_idx * GROUP_ID_OFFSET,
        )
    parallel_repeats = int(cfg.collector.group_size)
    candidate_repeats = candidate_group_size(cfg)
    if candidate_repeats % parallel_repeats:
        raise ValueError(
            "collector.candidate_group_size must be a multiple of "
            "collector.group_size when env.parallel_group_repeats=true "
            f"({candidate_repeats=} and {parallel_repeats=})."
        )
    worker_group_repeats = candidate_repeats // parallel_repeats
    logical_worker_idx = worker_idx // parallel_repeats
    return (
        task_ids[logical_worker_idx % len(task_ids)],
        worker_group_repeats,
        (logical_worker_idx * GROUP_ID_OFFSET),
    )


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
            top_k=cfg.policy.get("top_k", None),
            log_probs_mode=log_probs_mode,
            use_wrist_image=cfg.policy.use_wrist_image,
            center_crop=cfg.policy.center_crop,
            image_backend=cfg.policy.get("image_backend", "torchvision"),
            gripper_binarize=cfg.policy.get("gripper_binarize", False),
            gripper_binarize_threshold=cfg.policy.get(
                "gripper_binarize_threshold", 0.0
            ),
            gripper_invert=cfg.policy.get("gripper_invert", False),
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
        if policy.action_tokenizer is None:
            raise RuntimeError(
                "The OpenVLA policy did not expose an action tokenizer. Check "
                "that the checkpoint carries dataset action statistics and "
                "that policy.unnorm_key selects one of them."
            )
        return policy.action_tokenizer
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
    _configure_mujoco_rendering(cfg)
    task_id, worker_group_repeats, group_id_offset = _libero_worker_assignment(
        cfg, worker_idx, group_repeats=group_repeats, eval_mode=eval_mode
    )
    parallel_group_repeats = (
        not eval_mode
        and group_repeats is not None
        and bool(cfg.env.get("parallel_group_repeats", False))
    )
    env_kwargs = dict(cfg.env.get("env_kwargs", None) or {})
    render_gpu_device_id = _worker_render_gpu_device_id(
        cfg, worker_idx, eval_mode=eval_mode
    )
    if render_gpu_device_id is not None:
        env_kwargs["render_gpu_device_id"] = render_gpu_device_id
    return LiberoEnv(
        cfg.env.task_suite,
        task_id=task_id,
        camera_height=cfg.env.camera_height,
        camera_width=cfg.env.camera_width,
        env_kwargs=env_kwargs,
        wrist_camera="robot0_eye_in_hand" if cfg.policy.use_wrist_image else None,
        from_pixels=from_pixels,
        max_episode_steps=cfg.env.max_env_steps,
        init_state_mode=(
            "cycle" if eval_mode else cfg.env.get("train_init_state_mode", "random")
        ),
        group_repeats=worker_group_repeats,
        group_id_offset=group_id_offset,
        group_id_mode="init_state" if parallel_group_repeats else "episode",
    )


def _num_envs_from_cfg(cfg, *, eval_mode: bool = False) -> int:
    if cfg.env.backend == "toy":
        default = 1
    else:
        default = cfg.env.eval_num_envs if eval_mode else cfg.env.num_envs
    key = "eval_num_envs" if eval_mode else "num_envs"
    return int(_cfg_get(cfg.env, key, default))


def _validate_libero_env_count(
    cfg, num_envs: int, *, group_repeats=None, eval_mode: bool = False, override=False
) -> None:
    task_ids = list(cfg.env.task_ids)
    parallel_group_repeats = (
        not eval_mode
        and group_repeats is not None
        and bool(_cfg_get(cfg.env, "parallel_group_repeats", False))
    )
    task_coverage_envs = num_envs
    if parallel_group_repeats:
        group_repeats = int(group_repeats)
        candidate_repeats = candidate_group_size(cfg)
        if candidate_repeats % group_repeats:
            raise ValueError(
                "collector.candidate_group_size must be a multiple of "
                "collector.group_size when env.parallel_group_repeats=true "
                f"({candidate_repeats=} and {group_repeats=})."
            )
        if num_envs % group_repeats:
            raise ValueError(
                "env.num_envs must be a multiple of collector.group_size "
                "when env.parallel_group_repeats=true so every parallel "
                f"group has exactly {group_repeats} workers ({num_envs=})."
            )
        if _cfg_get(cfg.env, "train_init_state_mode", "random") == "random":
            raise ValueError(
                "env.parallel_group_repeats=true requires "
                "env.train_init_state_mode='cycle' or 'fixed'. Random "
                "init-state sampling is local to each worker, so workers "
                "sharing a group id would not necessarily share the same "
                "initial state."
            )
        task_coverage_envs = num_envs // group_repeats
    if not override and task_coverage_envs < len(task_ids):
        raise ValueError(
            f"{'eval_num_envs' if eval_mode else 'num_envs'} ({num_envs}) "
            f"must cover task_ids ({len(task_ids)} tasks): each worker is "
            "bound to one task; fewer workers would silently drop tasks. "
            "With env.parallel_group_repeats=true, task coverage is "
            f"num_envs / collector.group_size ({task_coverage_envs})."
        )
    if not override and task_coverage_envs % len(task_ids):
        warnings.warn(
            f"effective task workers ({task_coverage_envs}) is not a "
            f"multiple of the number of tasks ({len(task_ids)}): tasks "
            "will be sampled unevenly."
        )


def _make_env_worker(
    cfg,
    tokenizer: ActionTokenizerBase,
    worker_idx: int,
    *,
    group_repeats: int | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
    eval_mode: bool = False,
    from_pixels: bool = False,
) -> TransformedEnv:
    worker_seed = None if seed is None else int(seed) + int(worker_idx)
    if cfg.env.backend == "toy":
        base = ToyVLAEnv(
            action_dim=cfg.env.action_dim,
            state_dim=cfg.env.state_dim,
            image_shape=tuple(cfg.env.image_shape),
            from_pixels=from_pixels,
            render_size=_cfg_get(cfg.env, "render_size", 64),
            success_steps=cfg.env.success_steps,
            success_tol=cfg.env.success_tol,
            group_repeats=group_repeats,
            group_id_offset=worker_idx * GROUP_ID_OFFSET,
            batch_size=[],
            seed=worker_seed,
            device=device,
        )
    elif cfg.env.backend == "libero":
        base = _make_libero_worker(
            cfg,
            worker_idx,
            group_repeats=group_repeats,
            eval_mode=eval_mode,
            from_pixels=from_pixels,
        )
        if worker_seed is not None:
            base.set_seed(worker_seed)
    else:
        raise ValueError(f"Unknown env backend {cfg.env.backend!r}.")
    return TransformedEnv(base, _chunk_transform(cfg, tokenizer))


def make_async_env_factories(
    cfg,
    tokenizer: ActionTokenizerBase,
    *,
    group_repeats: int | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
    eval_mode: bool = False,
    from_pixels: bool = False,
    num_envs: int | None = None,
) -> list[Callable[[], TransformedEnv]]:
    """Build one transformed VLA env factory per async collection slot."""
    override = num_envs is not None
    if num_envs is None:
        num_envs = _num_envs_from_cfg(cfg, eval_mode=eval_mode)
    if cfg.env.backend == "libero":
        _validate_libero_env_count(
            cfg,
            num_envs,
            group_repeats=group_repeats,
            eval_mode=eval_mode,
            override=override,
        )
    return [
        partial(
            _make_env_worker,
            cfg,
            tokenizer,
            worker_idx,
            group_repeats=group_repeats,
            seed=seed,
            device=device,
            eval_mode=eval_mode,
            from_pixels=from_pixels,
        )
        for worker_idx in range(num_envs)
    ]


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
        parallel_group_repeats = (
            not eval_mode
            and group_repeats is not None
            and bool(cfg.env.get("parallel_group_repeats", False))
        )
        # each worker hosts ONE MuJoCo task for its whole lifetime: fewer
        # workers than tasks would silently drop tasks from the run
        task_coverage_envs = num_envs
        if parallel_group_repeats:
            group_repeats = int(group_repeats)
            candidate_repeats = candidate_group_size(cfg)
            if candidate_repeats % group_repeats:
                raise ValueError(
                    "collector.candidate_group_size must be a multiple of "
                    "collector.group_size when env.parallel_group_repeats=true "
                    f"({candidate_repeats=} and {group_repeats=})."
                )
            if num_envs % group_repeats:
                raise ValueError(
                    "env.num_envs must be a multiple of collector.group_size "
                    "when env.parallel_group_repeats=true so every parallel "
                    f"group has exactly {group_repeats} workers "
                    f"({num_envs=})."
                )
            if cfg.env.get("train_init_state_mode", "random") == "random":
                raise ValueError(
                    "env.parallel_group_repeats=true requires "
                    "env.train_init_state_mode='cycle' or 'fixed'. Random "
                    "init-state sampling is local to each worker, so workers "
                    "sharing a group id would not necessarily share the same "
                    "initial state."
                )
            task_coverage_envs = num_envs // group_repeats
        if not override and task_coverage_envs < len(task_ids):
            raise ValueError(
                f"{'eval_num_envs' if eval_mode else 'num_envs'} ({num_envs}) "
                f"must cover task_ids ({len(task_ids)} tasks): each worker is "
                "bound to one task; fewer workers would silently drop tasks. "
                "With env.parallel_group_repeats=true, task coverage is "
                f"num_envs / collector.group_size ({task_coverage_envs})."
            )
        if not override and task_coverage_envs % len(task_ids):
            warnings.warn(
                f"effective task workers ({task_coverage_envs}) is not a "
                f"multiple of the number of tasks ({len(task_ids)}): tasks "
                "will be sampled unevenly."
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
    collector_get = getattr(
        cfg.collector,
        "get",
        lambda key, default=None: getattr(cfg.collector, key, default),
    )
    max_collect_batches_per_iter = max(
        int(collector_get("max_collect_batches_per_iter", 1)), 1
    )
    candidate_size = candidate_group_size(cfg)
    capacity = (
        cfg.collector.groups_per_iter
        * candidate_size
        * cfg.env.max_outer_steps
        * max_collect_batches_per_iter
    )
    keep_return_bounds = cfg.advantage.keep_return_bounds
    if keep_return_bounds is not None:
        keep_return_bounds = tuple(keep_return_bounds)
    advantage_get = getattr(
        cfg.advantage,
        "get",
        lambda key, default=None: getattr(cfg.advantage, key, default),
    )
    selector_strategy = advantage_get("candidate_selection", "balanced")
    selector_max_combinations = int(
        advantage_get("candidate_selection_max_combinations", 100_000)
    )
    candidate_selection_min_size = advantage_get("candidate_selection_min_size", None)
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
        candidate_group_size=candidate_size,
        candidate_selection_min_size=candidate_selection_min_size,
        candidate_selector=MCAdvantageSelector(
            selector_strategy,
            max_combinations=selector_max_combinations,
        ),
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
        action=ACTION_TOKENS_KEY,
        sample_log_prob=LOG_PROBS_KEY,
        advantage="advantage",
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


class _ServerBackedCollector:
    """Collector wrapper that owns a thread-backed inference server."""

    def __init__(self, collector: Collector, server: InferenceServer) -> None:
        self.collector = collector
        self.server = server
        self.requested_frames_per_batch = collector.requested_frames_per_batch

    def __iter__(self):
        return iter(self.collector)

    def reset(self, *args, **kwargs) -> None:
        self.collector.reset(*args, **kwargs)

    def shutdown(self, *args, **kwargs) -> None:
        try:
            self.collector.shutdown(*args, **kwargs)
        finally:
            self.server.shutdown()

    def server_stats(self, *, reset: bool = False) -> dict[str, float | int]:
        return self.server.stats(reset=reset)

    def __getattr__(self, name):
        return getattr(self.collector, name)


class _BatchedPolicyClientModule(PolicyClientModule):
    """Split a synchronous batched env observation into server requests."""

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if tensordict.ndim == 0:
            return super().forward(tensordict)
        batch_size = tensordict.batch_size
        flat_tensordict = tensordict.reshape(-1)
        futures = [self.submit(td) for td in flat_tensordict.unbind(0)]
        result = lazy_stack([future.result() for future in futures], 0)
        result = result.reshape(batch_size)
        self._check_policy_lag(result)
        return result


class _AsyncReplayCollector:
    """AsyncBatchedCollector wrapper that writes complete trajectories to replay."""

    yields_complete_trajectories = True
    requested_frames_per_batch = 1

    def __init__(
        self,
        *,
        create_env_fn: list[Callable[[], EnvBase]],
        policy: VLAWrapperBase,
        replay_buffer: TensorDictReplayBuffer | None,
        collector_kwargs: dict,
    ) -> None:
        self._create_env_fn = create_env_fn
        self._policy = policy
        self._replay_buffer = replay_buffer
        self._collector_kwargs = collector_kwargs
        self._collector = None
        self._iterator = None
        self._last_server_stats: dict[str, float | int] = {}
        self.num_envs = len(create_env_fn)

    def _ensure_collector(self):
        if self._collector is None:
            self._collector = AsyncBatchedCollector(
                create_env_fn=self._create_env_fn,
                policy=self._policy,
                **self._collector_kwargs,
            )
            self._iterator = iter(self._collector)
        return self._collector

    def __iter__(self):
        return self

    def __next__(self):
        self._ensure_collector()
        traj = next(self._iterator)
        if self._replay_buffer is not None:
            self._replay_buffer.extend(traj)
        return traj

    def pause_collection(self) -> None:
        if self._collector is None:
            return
        self._last_server_stats = self._collector.server_stats(reset=True)
        self._collector.shutdown()
        self._collector = None
        self._iterator = None

    def reset(self, *args, **kwargs) -> None:
        self.pause_collection()

    def shutdown(self, *args, **kwargs) -> None:
        self.pause_collection()

    def server_stats(self, *, reset: bool = False) -> dict[str, float | int]:
        if self._collector is not None:
            return self._collector.server_stats(reset=reset)
        result = dict(self._last_server_stats)
        if reset:
            self._last_server_stats = {}
        return result


def _training_group_repeats(cfg) -> int:
    env_get = getattr(
        cfg.env,
        "get",
        lambda key, default=None: getattr(cfg.env, key, default),
    )
    return (
        cfg.collector.group_size
        if env_get("parallel_group_repeats", False)
        else candidate_group_size(cfg)
    )


def _server_config_from_collector(cfg, *, num_envs: int) -> InferenceServerConfig:
    collector_get = getattr(
        cfg.collector,
        "get",
        lambda key, default=None: getattr(cfg.collector, key, default),
    )
    async_policy = bool(collector_get("async_policy", False))
    if async_policy:
        max_batch_size = int(collector_get("server_max_batch_size", None) or num_envs)
        min_batch_size = int(collector_get("server_min_batch_size", 1))
        timeout = float(collector_get("server_timeout", 0.01))
    else:
        max_batch_size = 1
        min_batch_size = 1
        timeout = 0.0
    return InferenceServerConfig(
        max_batch_size=max_batch_size,
        min_batch_size=min_batch_size,
        timeout=timeout,
        collect_stats=bool(collector_get("server_collect_stats", True)),
        stats_window_size=int(collector_get("server_stats_window_size", 1024)),
    )


def make_collector(
    cfg,
    env: EnvBase,
    policy: VLAWrapperBase,
    device: torch.device,
    *,
    tokenizer: ActionTokenizerBase | None = None,
    replay_buffer: TensorDictReplayBuffer | None = None,
    post_collect_hook: Callable[[TensorDictBase], None] | None = None,
) -> Collector:
    """Build a VLA rollout collector.

    The default path is the synchronous TorchRL ``Collector`` used by the
    original recipe. Set ``collector.async_env=true`` to use
    ``AsyncBatchedCollector`` env slots; set ``collector.async_policy=true`` to
    route policy calls through an inference server with configurable
    auto-batching. The async-env path yields complete trajectories and writes
    them to the replay buffer directly, so the training loop can use the same
    replay/advantage machinery across execution modes.

    With no replay buffer on the synchronous path, each yielded batch holds
    complete, done-terminated trajectories, concatenated along time
    (``trajs_per_batch`` with ``traj_format="cat"``). With a replay buffer,
    TorchRL's collector writer path pushes complete trajectories to storage as
    each internal rollout batch finishes.

    The policy is held by reference (in-place optimizer updates apply
    immediately) and observations/actions are cast between the env's and the
    policy's devices by the collector. ``exploration_type=RANDOM`` makes the
    collector roll out under a sampling context, which the token policy reads
    via :func:`~torchrl.envs.utils.exploration_type` -- no policy mutation, so
    this works with any collector (including multi-process workers).
    """
    collector_get = getattr(
        cfg.collector,
        "get",
        lambda key, default=None: getattr(cfg.collector, key, default),
    )
    async_env = bool(collector_get("async_env", False))
    async_policy = bool(collector_get("async_policy", False))
    if async_env:
        num_envs = _num_envs_from_cfg(cfg)
    else:
        num_envs = env.batch_size[0] if env.batch_size else 1
    groups_per_iter = int(cfg.collector.groups_per_iter)
    group_size = int(cfg.collector.group_size)
    candidate_size = candidate_group_size(cfg)
    env_get = getattr(
        cfg.env,
        "get",
        lambda key, default=None: getattr(cfg.env, key, default),
    )
    parallel_group_repeats = bool(env_get("parallel_group_repeats", False))
    group_workers = num_envs
    if parallel_group_repeats:
        if candidate_size % group_size:
            raise ValueError(
                "collector.candidate_group_size must be a multiple of "
                "collector.group_size when env.parallel_group_repeats=true "
                f"({candidate_size=} and {group_size=})."
            )
        if num_envs % group_size:
            raise ValueError(
                "env.num_envs must be a multiple of collector.group_size when "
                "env.parallel_group_repeats=true "
                f"({num_envs=} and {group_size=})."
            )
        group_workers = num_envs // group_size
    if groups_per_iter < group_workers:
        raise ValueError(
            "collector.groups_per_iter must be at least the number of parallel "
            "group workers. Each worker emits repeated rollouts for its "
            "own group ids (or, with env.parallel_group_repeats=true, each "
            "logical worker group emits parallel rollouts for its group ids); "
            "with fewer groups than workers, no worker can "
            f"complete a full GRPO group in one iteration ({groups_per_iter=} "
            f"< {group_workers=}), so the dynamic-sampling replay buffer would stay "
            "empty. Reduce env.num_envs or increase collector.groups_per_iter."
        )
    if groups_per_iter % group_workers:
        warnings.warn(
            "collector.groups_per_iter is not a multiple of the number of "
            "parallel group workers. Some workers will start a partial GRPO "
            "group near the iteration boundary; Collector.reset() drops those "
            "incomplete groups before the policy update. For best throughput, "
            "set the group-worker count to a divisor of collector.groups_per_iter, "
            "ideally the same value."
        )
    if parallel_group_repeats and groups_per_iter != group_workers:
        warnings.warn(
            "With env.parallel_group_repeats=true, collector.groups_per_iter "
            "should usually match the number of logical parallel group workers "
            "(env.num_envs / collector.group_size). If each logical worker must "
            "advance through multiple group ids inside one collector batch, "
            "variable episode lengths can create partial groups that are dropped "
            "at the policy-update boundary. To overcollect, prefer additional "
            "collector batches via collector.max_collect_batches_per_iter and "
            "collector.min_replay_decisions, but note that multiple consecutive "
            "batches can still drift without a group barrier; one aligned group "
            "wave per update gives the lowest boundary waste."
        )
    # Some transformed batched envs (notably LIBERO's ParallelEnv wrapped in
    # TransformedEnv) deliberately expose a device-less outer TensorDict even
    # though the simulator action path is CPU-only. If only ``policy_device``
    # is passed, Collector keeps CUDA policy outputs on the carrier and hands
    # CUDA ``("vla_action", "tokens")`` to the env inverse transforms. The
    # rollout still runs, but the decoded CPU simulator actions can silently
    # diverge from
    # env.rollout(auto_cast_to_device=True). Force the env side to CPU when the
    # env does not advertise a device so sampled tokens are copied back before
    # MuJoCo/ParallelEnv stepping.
    env_device = env.device if env.device is not None else torch.device("cpu")
    # With a replay buffer, use one outer step per internal collector poll so
    # complete trajectories are handed to the replay-buffer transform as soon
    # as they finish instead of waiting for a full max-length rollout from
    # every worker. Without a replay buffer, keep the historical full-episode
    # polling granularity so direct iteration yields whole group waves.
    frames_per_batch = (
        num_envs if replay_buffer is not None else (num_envs * cfg.env.max_outer_steps)
    )
    server_config = _server_config_from_collector(cfg, num_envs=num_envs)
    if async_env:
        if tokenizer is None:
            raise ValueError(
                "tokenizer is required when collector.async_env=true so async "
                "environment factories can decode action tokens."
            )
        create_env_fn = make_async_env_factories(
            cfg,
            tokenizer,
            group_repeats=_training_group_repeats(cfg),
            seed=cfg.env.seed,
            device=env_device if cfg.env.backend == "toy" else None,
        )
        return _AsyncReplayCollector(
            create_env_fn=create_env_fn,
            policy=policy,
            replay_buffer=replay_buffer,
            collector_kwargs={
                "frames_per_batch": 1,
                "total_frames": -1,
                "yield_completed_trajectories": True,
                "env_backend": collector_get("env_backend", "threading"),
                "policy_backend": collector_get("policy_backend", "threading"),
                "server_backend": collector_get("server_backend", "thread"),
                "server_config": server_config,
                "policy_device": device,
                "output_device": env_device,
                "env_device": env_device,
                "storing_device": collector_get("storing_device", None),
                "max_inflight_per_env": collector_get("max_inflight_per_env", 1),
                "verbose": bool(collector_get("verbose", False)),
            },
        )

    collector_policy = policy
    collector_policy_device = device
    server = None
    if async_policy:
        transport = ThreadingTransport()
        server = InferenceServer(
            policy,
            transport,
            server_config=server_config,
            policy_device=device,
            output_device=env_device,
        ).start()
        collector_policy = _BatchedPolicyClientModule(
            transport,
            in_keys=getattr(policy, "in_keys", None),
            out_keys=getattr(policy, "out_keys", None),
            max_inflight=None,
        )
        collector_policy_device = env_device

    collector = Collector(
        env,
        collector_policy,
        frames_per_batch=frames_per_batch,
        total_frames=-1,
        trajs_per_batch=groups_per_iter * candidate_size,
        traj_format="cat",
        exploration_type=ExplorationType.RANDOM,
        policy_device=collector_policy_device,
        env_device=env_device,
        reset_at_each_iter=False,
        replay_buffer=replay_buffer,
        post_collect_hook=post_collect_hook,
        trust_policy=True if async_policy else None,
    )
    if server is not None:
        return _ServerBackedCollector(collector, server)
    return collector


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
