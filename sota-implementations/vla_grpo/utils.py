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

import importlib.util
import multiprocessing as mp
import os
import time
import warnings
from collections.abc import Callable
from contextlib import contextmanager
from functools import partial

import torch

from tensordict import NonTensorData, TensorDict, TensorDictBase
from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.collectors import Evaluator, MultiCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.vla import (
    ACTION_TOKENS_KEY,
    ActionTokenizerBase,
    UniformActionTokenizer,
)
from torchrl.envs import (
    ActionTokenizerTransform,
    Compose,
    LiberoEnv,
    MultiAction,
    ParallelEnv,
    StepCounter,
    SuccessReward,
    ToyVLAEnv,
    TransformedEnv,
)
from torchrl.envs.utils import ExplorationType
from torchrl.modules.inference_server import (
    InferenceServerConfig,
    MPTransport,
    PolicyClientModule,
    ProcessInferenceServer,
)
from torchrl.modules.vla import TinyVLA, VLAWrapperBase
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.llm import MCAdvantage, MCAdvantageSelector
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.weight_update import WeightStrategy

# group ids must be unique across parallel workers: each worker gets a
# disjoint offset block
GROUP_ID_OFFSET = 10**6
_has_robosuite = importlib.util.find_spec("robosuite") is not None
_has_openvla = importlib.util.find_spec("openvla") is not None
_has_peft = importlib.util.find_spec("peft") is not None
_ROBOSUITE_EGL_DEVICE_COUNT: int | None = None
_OpenVLAOFTWrapper = None
_OpenVLAOFTL1Wrapper = None
_GripperPostProcessTransform = None
_get_peft_model = None
_LoraConfig = None
LOG_PROBS_KEY = ("vla_action", "log_probs")

_TORCH_DTYPES = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


def _openvla_wrapper_cls(mode: str):
    """Lazy optional import for the OpenVLA backend."""
    global _OpenVLAOFTL1Wrapper, _OpenVLAOFTWrapper
    if not _has_openvla:
        raise ImportError("The openvla backend requires the local openvla module.")
    if mode == "tokens":
        if _OpenVLAOFTWrapper is None:
            from openvla import OpenVLAOFTWrapper

            _OpenVLAOFTWrapper = OpenVLAOFTWrapper
        return _OpenVLAOFTWrapper
    if mode == "l1":
        if _OpenVLAOFTL1Wrapper is None:
            from openvla import OpenVLAOFTL1Wrapper

            _OpenVLAOFTL1Wrapper = OpenVLAOFTL1Wrapper
        return _OpenVLAOFTL1Wrapper
    raise ValueError(f"policy.mode must be 'tokens' or 'l1', got {mode!r}.")


def _gripper_postprocess_transform_cls():
    """Lazy optional import for OpenVLA gripper post-processing."""
    global _GripperPostProcessTransform
    if not _has_openvla:
        raise ImportError("The openvla backend requires the local openvla module.")
    if _GripperPostProcessTransform is None:
        from openvla import GripperPostProcessTransform

        _GripperPostProcessTransform = GripperPostProcessTransform
    return _GripperPostProcessTransform


def _peft_lora_tools():
    """Lazy optional import for LoRA fine-tuning."""
    global _LoraConfig, _get_peft_model
    if not _has_peft:
        raise ImportError("policy.lora_rank requires the peft package.")
    if _get_peft_model is None or _LoraConfig is None:
        from peft import get_peft_model, LoraConfig

        _get_peft_model = get_peft_model
        _LoraConfig = LoraConfig
    return _get_peft_model, _LoraConfig


def _cfg_get(section, key: str, default=None):
    """Read an optional key from a config section.

    Works for mappings (``DictConfig``, ``dict``) through ``.get`` and for
    attribute-style sections (``SimpleNamespace``, dataclasses) through
    ``getattr`` with a default.
    """
    getter = getattr(section, "get", None)
    if callable(getter):
        return getter(key, default)
    return getattr(section, key, default)


def candidate_group_size(cfg) -> int:
    """Number of rollout candidates collected for each GRPO group."""
    return int(cfg.collector.candidate_group_size or cfg.collector.group_size)


def auto_device(device_spec) -> torch.device:
    """Resolve an explicit device or default to cuda:0 when available."""
    if device_spec:
        return torch.device(device_spec)
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def make_logger(cfg):
    """Build the configured experiment logger."""
    if not cfg.logger.backend:
        return None
    exp_name = generate_exp_name("VLA-GRPO", cfg.logger.exp_name)
    return get_logger(
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


def _configure_mujoco_rendering(cfg) -> None:
    render_backend = cfg.env.render_backend
    if render_backend is None:
        return
    render_backend = str(render_backend)
    os.environ["MUJOCO_GL"] = render_backend
    if render_backend in ("egl", "osmesa"):
        os.environ["PYOPENGL_PLATFORM"] = render_backend


def _worker_render_gpu_device_id(
    cfg,
    worker_idx: int,
    *,
    eval_mode: bool = False,
    render_gpu_device_id: int | None = None,
) -> int | None:
    if render_gpu_device_id is not None:
        return int(render_gpu_device_id)
    render_backend = cfg.env.render_backend
    if render_backend is not None and str(render_backend) != "egl":
        return None
    render_gpu_ids = cfg.env.eval_render_gpu_ids if eval_mode else None
    if render_gpu_ids is None:
        render_gpu_ids = cfg.env.render_gpu_ids
    if render_gpu_ids is None:
        return None
    render_gpu_ids = [int(device_id) for device_id in render_gpu_ids]
    if not render_gpu_ids:
        return None
    return render_gpu_ids[worker_idx % len(render_gpu_ids)]


def _robosuite_egl_device_count() -> int | None:
    global _ROBOSUITE_EGL_DEVICE_COUNT
    if not _has_robosuite:
        return None
    if _ROBOSUITE_EGL_DEVICE_COUNT is None:
        try:
            import robosuite.renderers.context.egl_context as egl_context
        except Exception:
            _ROBOSUITE_EGL_DEVICE_COUNT = -1
        else:
            try:
                _ROBOSUITE_EGL_DEVICE_COUNT = len(egl_context.EGL.eglQueryDevicesEXT())
            except Exception:
                _ROBOSUITE_EGL_DEVICE_COUNT = -1
    if _ROBOSUITE_EGL_DEVICE_COUNT < 0:
        return None
    return _ROBOSUITE_EGL_DEVICE_COUNT


@contextmanager
def _worker_render_gpu_context(cfg, render_gpu_device_id: int | None):
    """Map the requested render GPU to EGL device 0 if EGL is CVD-scoped."""
    if render_gpu_device_id is None or not bool(
        cfg.env.render_gpu_device_zero_fallback
    ):
        yield render_gpu_device_id
        return

    egl_device_count = _robosuite_egl_device_count()
    if egl_device_count is None or render_gpu_device_id < egl_device_count:
        yield render_gpu_device_id
        return

    old_mujoco_egl_device_id = os.environ.get("MUJOCO_EGL_DEVICE_ID")
    old_cuda_visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    try:
        # Some pyxis images expose one EGL device relative to the current
        # CUDA_VISIBLE_DEVICES value.  Narrowing CVD in the env worker maps the
        # requested physical render GPU to local EGL id 0, instead of sending
        # every worker to physical GPU 0.
        #
        # robosuite 1.4 also asserts that MUJOCO_EGL_DEVICE_ID appears as a
        # literal entry in CUDA_VISIBLE_DEVICES, even though MuJoCo interprets
        # it as the local EGL ordinal.  Keep physical GPU 0 visible as a dummy
        # second entry for nonzero render GPUs so the assertion accepts local
        # EGL id 0 while CUDA still maps local device 0 to render_gpu_device_id.
        cuda_visible_devices = str(render_gpu_device_id)
        if render_gpu_device_id != 0:
            cuda_visible_devices = f"{render_gpu_device_id},0"
        os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
        os.environ["MUJOCO_EGL_DEVICE_ID"] = "0"
        yield 0
    finally:
        if old_mujoco_egl_device_id is None:
            os.environ.pop("MUJOCO_EGL_DEVICE_ID", None)
        else:
            os.environ["MUJOCO_EGL_DEVICE_ID"] = old_mujoco_egl_device_id
        if old_cuda_visible_devices is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda_visible_devices


def _libero_worker_assignment(
    cfg, worker_idx: int, *, group_repeats=None, eval_mode=False
) -> tuple[int, int | None, int]:
    task_ids = list(cfg.env.task_ids)
    if eval_mode or group_repeats is None or not cfg.env.parallel_group_repeats:
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


def make_policy(
    cfg,
    device: torch.device,
    *,
    policy_micro_batch_size: int | None = None,
) -> VLAWrapperBase:
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
            return_vla_action_container=False,
            device=device,
        )
    if cfg.policy.backend == "openvla":
        OpenVLAOFTWrapper = _openvla_wrapper_cls(cfg.policy.mode)

        kwargs = {
            "torch_dtype": _TORCH_DTYPES[cfg.policy.dtype],
            "device": device,
            "unnorm_key": cfg.policy.unnorm_key,
            "dataset_statistics": cfg.policy.dataset_statistics,
            "use_wrist_image": cfg.policy.use_wrist_image,
            "center_crop": cfg.policy.center_crop,
            "image_backend": cfg.policy.image_backend,
            "gripper_binarize": cfg.policy.gripper_binarize,
            "gripper_binarize_threshold": cfg.policy.gripper_binarize_threshold,
            "gripper_invert": cfg.policy.gripper_invert,
            "return_vla_action_container": False,
        }
        if cfg.policy.mode == "tokens":
            kwargs.update(
                temperature=cfg.policy.temperature,
                top_k=cfg.policy.top_k,
                micro_batch_size=policy_micro_batch_size,
                log_probs_mode=log_probs_mode,
            )
        else:
            kwargs.update(
                action_head_file=cfg.policy.action_head_file,
                proprio_projector_file=cfg.policy.proprio_projector_file,
                use_proprio=cfg.policy.use_proprio,
                num_images_in_input=cfg.policy.num_images_in_input,
            )
        policy = OpenVLAOFTWrapper.from_pretrained(cfg.policy.checkpoint, **kwargs)
        if cfg.policy.lora_rank:
            # de-risk fallback to full fine-tuning (RL4VLA shows LoRA r=32
            # works); validate on the target hardware
            get_peft_model, LoraConfig = _peft_lora_tools()

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


def make_action_tokenizer(cfg, policy: VLAWrapperBase) -> ActionTokenizerBase | None:
    if cfg.policy.backend == "openvla":
        if cfg.policy.mode == "l1":
            return None
        # the codec lives in the checkpoint (vocab-tail mapping + norm_stats)
        if policy.action_tokenizer is None:
            raise RuntimeError(
                "The OpenVLA policy did not expose an action tokenizer. Check "
                "that the checkpoint carries dataset action statistics and "
                "that policy.unnorm_key selects one of them."
            )
        return policy.action_tokenizer
    return UniformActionTokenizer(cfg.tokenizer.vocab_size, low=-1.0, high=1.0)


def _chunk_transform(
    cfg,
    tokenizer: ActionTokenizerBase | None,
    *,
    decode_actions_in_env: bool = True,
) -> Compose:
    # The compose order is load-bearing: when token decoding lives in the env,
    # the inverse (action-input) path runs in reverse, so tokenizer decode
    # happens before MultiAction unbinds the chunk. When token decoding lives
    # in the policy, the env receives the continuous chunk directly and
    # MultiAction is still responsible for unbinding it. On the step path,
    # SuccessReward and StepCounter run after MultiAction, i.e. once per outer
    # (decision) step. stack_rewards=False keeps the outer transition dense
    # when an episode ends inside a chunk (the decision reward comes from the
    # outer success flag instead).
    if decode_actions_in_env and tokenizer is not None:
        transforms = [MultiAction(stack_rewards=False)]
        if cfg.policy.backend == "openvla" and cfg.policy.mode == "tokens":
            GripperPostProcessTransform = _gripper_postprocess_transform_cls()
            transforms.append(
                GripperPostProcessTransform(
                    action_key="action",
                    rescale=True,
                    binarize=cfg.policy.gripper_binarize,
                    threshold=cfg.policy.gripper_binarize_threshold,
                    invert=cfg.policy.gripper_invert,
                )
            )
        transforms.append(ActionTokenizerTransform(tokenizer))
    else:
        transforms = [MultiAction.from_vla(stack_rewards=False)]
    transforms.extend(
        [
            SuccessReward(),
            StepCounter(max_steps=cfg.env.max_outer_steps),
        ]
    )
    return Compose(*transforms)


def _make_libero_worker(
    cfg,
    worker_idx: int,
    *,
    group_repeats=None,
    eval_mode=False,
    from_pixels=False,
    worker_idx_offset: int = 0,
    render_gpu_device_id: int | None = None,
):
    _configure_mujoco_rendering(cfg)
    worker_idx = int(worker_idx) + int(worker_idx_offset)
    task_id, worker_group_repeats, group_id_offset = _libero_worker_assignment(
        cfg, worker_idx, group_repeats=group_repeats, eval_mode=eval_mode
    )
    parallel_group_repeats = (
        not eval_mode
        and group_repeats is not None
        and bool(cfg.env.parallel_group_repeats)
    )
    env_kwargs = dict(cfg.env.env_kwargs or {})
    render_gpu_device_id = _worker_render_gpu_device_id(
        cfg,
        worker_idx,
        eval_mode=eval_mode,
        render_gpu_device_id=render_gpu_device_id,
    )
    with _worker_render_gpu_context(cfg, render_gpu_device_id) as egl_device_id:
        if egl_device_id is not None:
            env_kwargs["render_gpu_device_id"] = egl_device_id
        return LiberoEnv(
            cfg.env.task_suite,
            task_id=task_id,
            camera_height=cfg.env.camera_height,
            camera_width=cfg.env.camera_width,
            env_kwargs=env_kwargs,
            wrist_camera="robot0_eye_in_hand" if cfg.policy.use_wrist_image else None,
            from_pixels=from_pixels,
            max_episode_steps=cfg.env.max_env_steps,
            init_state_mode=("cycle" if eval_mode else cfg.env.train_init_state_mode),
            init_state_id=int(_cfg_get(cfg.env, "train_init_state_id", 0)),
            group_repeats=worker_group_repeats,
            group_id_offset=group_id_offset,
            group_id_mode="init_state" if parallel_group_repeats else "episode",
        )


def _num_envs_from_cfg(cfg, *, eval_mode: bool = False) -> int:
    if cfg.env.backend == "toy":
        return 1
    return int(_cfg_get(cfg.env, "eval_num_envs" if eval_mode else "num_envs", 1))


def _validate_libero_env_count(
    cfg, num_envs: int, *, group_repeats=None, eval_mode: bool = False, override=False
) -> None:
    task_ids = list(cfg.env.task_ids)
    parallel_group_repeats = (
        not eval_mode
        and group_repeats is not None
        and bool(cfg.env.parallel_group_repeats)
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
        if cfg.env.train_init_state_mode == "random":
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
    tokenizer: ActionTokenizerBase | None,
    worker_idx: int,
    *,
    group_repeats: int | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
    eval_mode: bool = False,
    from_pixels: bool = False,
    decode_actions_in_env: bool = True,
    worker_idx_offset: int = 0,
    render_gpu_device_id: int | None = None,
) -> TransformedEnv:
    worker_idx_with_offset = int(worker_idx) + int(worker_idx_offset)
    worker_seed = None if seed is None else int(seed) + worker_idx_with_offset
    if cfg.env.backend == "toy":
        base = ToyVLAEnv(
            action_dim=cfg.env.action_dim,
            state_dim=cfg.env.state_dim,
            image_shape=tuple(cfg.env.image_shape),
            from_pixels=from_pixels,
            render_size=cfg.env.render_size,
            success_steps=cfg.env.success_steps,
            success_tol=cfg.env.success_tol,
            group_repeats=group_repeats,
            group_id_offset=worker_idx_with_offset * GROUP_ID_OFFSET,
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
            worker_idx_offset=worker_idx_offset,
            render_gpu_device_id=render_gpu_device_id,
        )
        if worker_seed is not None:
            base.set_seed(worker_seed)
    else:
        raise ValueError(f"Unknown env backend {cfg.env.backend!r}.")
    return TransformedEnv(
        base,
        _chunk_transform(
            cfg,
            tokenizer,
            decode_actions_in_env=decode_actions_in_env,
        ),
    )


def make_env(
    cfg,
    tokenizer: ActionTokenizerBase | None,
    *,
    group_repeats: int | None = None,
    seed: int | None = None,
    device: torch.device | None = None,
    eval_mode: bool = False,
    from_pixels: bool = False,
    num_envs: int | None = None,
    decode_actions_in_env: bool = True,
    worker_idx_offset: int = 0,
    render_gpu_device_id: int | None = None,
) -> TransformedEnv:
    if cfg.env.backend == "toy":
        base = ToyVLAEnv(
            action_dim=cfg.env.action_dim,
            state_dim=cfg.env.state_dim,
            image_shape=tuple(cfg.env.image_shape),
            from_pixels=from_pixels,
            render_size=cfg.env.render_size,
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
            num_envs = _num_envs_from_cfg(cfg, eval_mode=eval_mode)
        _validate_libero_env_count(
            cfg,
            num_envs,
            group_repeats=group_repeats,
            eval_mode=eval_mode,
            override=override,
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
                    worker_idx_offset=worker_idx_offset,
                    render_gpu_device_id=render_gpu_device_id,
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
    return TransformedEnv(
        base,
        _chunk_transform(
            cfg,
            tokenizer,
            decode_actions_in_env=decode_actions_in_env,
        ),
    )


def _make_collector_env(
    cfg,
    tokenizer: ActionTokenizerBase | None,
    *,
    num_envs: int,
    group_repeats: int,
    seed: int,
    device: torch.device,
    worker_idx_offset: int,
    render_gpu_device_id: int | None,
) -> ParallelEnv:
    return ParallelEnv(
        num_envs,
        [
            partial(
                _make_env_worker,
                cfg,
                tokenizer,
                worker_idx,
                group_repeats=group_repeats,
                seed=seed,
                device=device if cfg.env.backend == "toy" else None,
                worker_idx_offset=worker_idx_offset,
                render_gpu_device_id=render_gpu_device_id,
            )
            for worker_idx in range(num_envs)
        ],
        mp_start_method="spawn",
        device=device,
    )


def make_replay_buffer(
    cfg, device: torch.device
) -> tuple[TensorDictReplayBuffer, MCAdvantage]:
    # The buffer holds one iteration's decisions; the write path computes the
    # group-relative advantage (and drops degenerate groups) on whole
    # trajectories, the read path samples decisions without replacement. The
    # advantage transform is returned too so the training loop can flush its
    # incomplete-group queues at iteration boundaries.
    capacity_group_waves = max(int(cfg.buffer.capacity_group_waves), 1)
    candidate_size = candidate_group_size(cfg)
    capacity = (
        cfg.collector.groups_per_iter
        * candidate_size
        * cfg.env.max_outer_steps
        * capacity_group_waves
    )
    keep_return_bounds = cfg.advantage.keep_return_bounds
    if keep_return_bounds is not None:
        keep_return_bounds = tuple(keep_return_bounds)
    selector_strategy = cfg.advantage.candidate_selection
    selector_max_combinations = int(cfg.advantage.candidate_selection_max_combinations)
    candidate_selection_min_size = cfg.advantage.candidate_selection_min_size
    shared_init = bool(cfg.buffer.shared_init)
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(
            capacity,
            device=device,
            shared_init=shared_init,
        ),
        batch_size=cfg.loss.mini_batch_size,
        consume_after_n_samples=cfg.buffer.consume_after_n_samples,
        shared=shared_init,
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
    if policy.action_head != "tokens":
        raise NotImplementedError(
            "VLA GRPO training currently expects a token policy with stored "
            "action log-probabilities. policy.mode='l1' is available for "
            "reference/evaluation rollouts; add a continuous-action loss "
            "before using it for training."
        )
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


_WEIGHT_STRATEGY = WeightStrategy(extract_as="tensordict")


def policy_weights(
    policy: torch.nn.Module, *, trainable_only: bool = False
) -> TensorDictBase:
    """Detached CPU TensorDict snapshot of a policy's parameters and buffers.

    With ``trainable_only=True`` the snapshot keeps only the tensors with
    ``requires_grad=True`` (e.g. the LoRA adapters), which is enough for
    weight syncs to a server policy built from the same checkpoint: the
    receiving :class:`~torchrl.weight_update.WeightStrategy` applies a partial
    TensorDict in place, leaving frozen weights and buffers untouched.
    """
    weights = _WEIGHT_STRATEGY.extract_weights(policy)
    if trainable_only:
        weights = weights.apply(
            lambda tensor: tensor if tensor.requires_grad else None,
            filter_empty=True,
        )
    return weights.data.detach().clone().cpu()


def sync_policy_server(
    policy_server: ProcessInferenceServer,
    policy: torch.nn.Module,
    *,
    trainable_only: bool = True,
):
    """Push trainer policy weights to the shared process policy server.

    ``trainable_only=True`` (see ``train.weight_sync_trainable_only``) ships
    only the ``requires_grad=True`` parameters through the control channel;
    with a 7B LoRA policy that is a few hundred MB instead of the ~14 GB full
    bf16 parameter set.
    """
    return policy_server.update_model_weights(
        policy_weights(policy, trainable_only=trainable_only)
    )


def apply_policy_weights(
    policy: torch.nn.Module, weights: TensorDictBase, device: torch.device
) -> None:
    """Apply a TensorDict parameter snapshot to ``policy`` on ``device``."""
    _WEIGHT_STRATEGY.apply_weights(policy, weights.to(device))


def _training_group_repeats(cfg) -> int:
    return (
        cfg.collector.group_size
        if cfg.env.parallel_group_repeats
        else candidate_group_size(cfg)
    )


def _render_gpu_for_subcollector(cfg, collector_idx: int) -> int | None:
    render_gpu_ids = cfg.env.render_gpu_ids
    if render_gpu_ids is None:
        return None
    render_gpu_ids = [int(device_id) for device_id in render_gpu_ids]
    if not render_gpu_ids:
        return None
    return render_gpu_ids[int(collector_idx) % len(render_gpu_ids)]


def _server_config_from_collector(cfg) -> InferenceServerConfig:
    return InferenceServerConfig(
        max_batch_size=int(cfg.collector.server_max_batch_size),
        min_batch_size=int(cfg.collector.server_min_batch_size),
        timeout=float(cfg.collector.server_timeout),
        collect_stats=bool(cfg.collector.server_collect_stats),
        stats_window_size=int(cfg.collector.server_stats_window_size),
    )


def make_collector(
    cfg,
    policy: VLAWrapperBase,
    device: torch.device,
    *,
    tokenizer: ActionTokenizerBase | None,
    replay_buffer: TensorDictReplayBuffer,
    post_collect_hook: Callable[[TensorDictBase], None] | None = None,
) -> tuple[MultiCollector, ProcessInferenceServer, PolicyClientModule]:
    """Build the TorchRL-native VLA rollout stack.

    The stack is always a ``MultiCollector`` whose workers own batched
    ``ParallelEnv`` instances and call a shared process policy server through
    ``PolicyClientModule``. The collector writes complete trajectories directly
    into the replay buffer.
    """
    num_collectors = int(cfg.collector.num_collectors)
    envs_per_collector = int(cfg.collector.envs_per_collector)
    group_size = int(cfg.collector.group_size)
    candidate_size = candidate_group_size(cfg)
    total_envs = num_collectors * envs_per_collector
    if cfg.env.parallel_group_repeats:
        if candidate_size % group_size:
            raise ValueError(
                "collector.candidate_group_size must be a multiple of "
                "collector.group_size when env.parallel_group_repeats=true "
                f"({candidate_size=} and {group_size=})."
            )
        if envs_per_collector % group_size and not replay_buffer.shared:
            raise ValueError(
                "collector.envs_per_collector must be a multiple of "
                "collector.group_size when env.parallel_group_repeats=true "
                "and the replay buffer does not share grouped write state "
                f"({envs_per_collector=} and {group_size=})."
            )

    group_workers = (
        total_envs // group_size if cfg.env.parallel_group_repeats else total_envs
    )
    if cfg.env.backend == "libero":
        _validate_libero_env_count(
            cfg,
            total_envs,
            group_repeats=_training_group_repeats(cfg),
        )
    groups_per_iter = int(cfg.collector.groups_per_iter)
    if groups_per_iter < group_workers:
        raise ValueError(
            "collector.groups_per_iter must be at least the number of "
            "shared policy-server group workers "
            f"({groups_per_iter=} < {group_workers=})."
        )
    if groups_per_iter % group_workers:
        warnings.warn(
            "collector.groups_per_iter is not a multiple of the shared "
            "policy-server group-worker count. Some same-policy partial "
            "groups can be dropped at the update boundary."
        )

    env_device = torch.device("cpu")
    ctx = mp.get_context("spawn")
    transport = MPTransport(ctx=ctx, use_manager=True)
    eval_client = transport.client()
    rollout_clients = [transport.client() for _ in range(num_collectors)]
    policy_micro_batch_size = _cfg_get(cfg.collector, "policy_micro_batch_size", None)
    server = ProcessInferenceServer(
        policy_factory=partial(
            make_policy,
            cfg,
            device,
            policy_micro_batch_size=policy_micro_batch_size,
        ),
        transport=transport,
        server_config=_server_config_from_collector(cfg),
        policy_device=device,
        output_device=env_device,
        mp_context=ctx,
    ).start()

    policy_in_keys = policy.in_keys
    policy_out_keys = [*policy.out_keys, "policy_version"]
    env_factories = [
        partial(
            _make_collector_env,
            cfg,
            tokenizer,
            num_envs=envs_per_collector,
            group_repeats=_training_group_repeats(cfg),
            seed=cfg.env.seed,
            device=env_device,
            worker_idx_offset=collector_idx * envs_per_collector,
            render_gpu_device_id=_render_gpu_for_subcollector(cfg, collector_idx),
        )
        for collector_idx in range(num_collectors)
    ]
    policy_client_factories = [
        partial(
            PolicyClientModule,
            client,
            in_keys=policy_in_keys,
            out_keys=policy_out_keys,
            max_inflight=cfg.collector.max_inflight_per_env,
        )
        for client in rollout_clients
    ]
    eval_policy = PolicyClientModule(
        eval_client,
        in_keys=policy_in_keys,
        out_keys=policy_out_keys,
    )
    collector = MultiCollector(
        env_factories,
        policy=None,
        policy_factory=policy_client_factories,
        frames_per_batch=envs_per_collector,
        total_frames=-1,
        reset_at_each_iter=False,
        replay_buffer=replay_buffer,
        trajs_per_batch=1,
        traj_format="cat",
        exploration_type=ExplorationType.RANDOM,
        policy_device=env_device,
        env_device=env_device,
        storing_device=torch.device(cfg.collector.storing_device),
        trust_policy=True,
        use_buffers=cfg.collector.use_buffers,
        sync=False,
        num_threads=int(cfg.collector.num_threads),
        num_sub_threads=int(cfg.collector.env_sub_threads),
        post_collect_hook=post_collect_hook,
    )
    return collector, server, eval_policy


def make_record_env(cfg, tokenizer: ActionTokenizerBase | None, logger, device):
    """Single-environment eval recorder feeding a torchrl ``VideoRecorder``.

    Built with ``from_pixels=True`` so the base env emits a root ``pixels``
    frame (``ToyVLAEnv`` renders the tracking scene; ``LiberoEnv`` exposes the
    camera). A :class:`~torchrl.record.VideoRecorder` transform appended last
    collects those frames during evaluator rollouts. One environment keeps the
    video a single clean stream rather than a tiled grid of workers.
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


def _make_eval_env(
    cfg,
    tokenizer: ActionTokenizerBase | None,
    logger,
    device: torch.device,
) -> TransformedEnv:
    if logger is not None:
        env, _ = make_record_env(cfg, tokenizer, logger, device)
        return env
    return make_env(
        cfg,
        tokenizer,
        seed=cfg.env.seed + 1,
        device=device if cfg.env.backend == "toy" else None,
        eval_mode=True,
    )


def _policy_module_factory(policy, *args, **kwargs):
    return policy


def _eval_success_metrics(rollout: TensorDictBase) -> dict[str, float]:
    success = rollout.get(("next", "success"), None)
    if success is None:
        return {"success_rate": float("nan")}
    mask = rollout.get(("collector", "mask"), None)
    success = success.bool()
    if mask is not None:
        success = success & mask.unsqueeze(-1).expand_as(success)
        success = success.reshape(success.shape[0], -1).any(-1)
    else:
        traj_ids = rollout.get(("collector", "traj_ids"), None)
        if traj_ids is None:
            success = success.reshape(1, -1).any(-1)
        else:
            traj_ids = traj_ids.reshape(-1)
            success = torch.stack(
                [
                    success.reshape(success.shape[0], -1)[traj_ids == traj_id].any()
                    for traj_id in traj_ids.unique(sorted=True)
                ]
            )
    return {"success_rate": float(success.float().mean())}


def make_evaluator(
    cfg,
    tokenizer: ActionTokenizerBase | None,
    policy,
    logger,
    device: torch.device,
) -> Evaluator:
    """Build the TorchRL evaluator used by the VLA GRPO recipe."""
    record_video = logger is not None and cfg.logger.eval_backend == "thread"
    if record_video and cfg.env.backend == "libero":
        task_ids = list(cfg.env.task_ids)
        if len(task_ids) > 1 and not bool(
            _cfg_get(cfg.logger, "record_video_single_task", False)
        ):
            raise ValueError(
                "logger.eval_backend='thread' with a logger replaces the eval "
                "env with a single-env video recorder bound to task "
                f"{task_ids[0]}, so eval/success_rate would cover 1 of the "
                f"{len(task_ids)} configured env.task_ids (and "
                "env.eval_num_envs would be ignored). Set "
                "logger.record_video_single_task=true to opt in to "
                "single-task eval video, or keep logger.eval_backend="
                "'process' for suite-wide evaluation without video."
            )
    env_factory = partial(
        _make_eval_env,
        cfg,
        tokenizer,
        logger if record_video else None,
        device,
    )
    return Evaluator(
        env_factory,
        policy=None,
        policy_factory=partial(_policy_module_factory, policy),
        num_trajectories=cfg.logger.eval_episodes,
        max_steps=0,
        frames_per_batch=cfg.env.max_outer_steps,
        collector_kwargs={"traj_format": "cat"},
        log_prefix="eval",
        reward_keys=("next", "reward"),
        done_keys=("next", "done"),
        device=device,
        exploration_type=ExplorationType.DETERMINISTIC,
        metrics_fn=_eval_success_metrics,
        dump_video=record_video,
        busy_policy=cfg.logger.eval_busy_policy,
        backend=cfg.logger.eval_backend,
    )


def _sync_replay_sampler_writes(replay_buffer: TensorDictReplayBuffer) -> None:
    """Mirror shared writer progress into the local consuming sampler state."""
    write_count = int(replay_buffer.write_count)
    try:
        previous_write_count = int(replay_buffer.__dict__["_vla_synced_write_count"])
    except KeyError:
        previous_write_count = 0
    if write_count <= previous_write_count:
        return
    storage = replay_buffer._storage
    storage_capacity = int(storage.max_size)
    num_new_writes = min(write_count - previous_write_count, storage_capacity)
    start = write_count - num_new_writes
    indices = torch.arange(start, write_count, dtype=torch.long)
    replay_buffer.mark_update(indices.remainder(storage_capacity))
    replay_buffer.__dict__["_vla_synced_write_count"] = write_count


def wait_for_replay(
    replay_buffer: TensorDictReplayBuffer,
    *,
    min_replay_decisions: int,
    poll_interval_s: float,
    log_interval_s: float,
    iteration: int,
) -> dict[str, float | int]:
    """Wait until the replay buffer has enough sampleable decisions."""
    polls = 0
    next_log_s = 0.0
    with timeit("replay_wait") as wait_timer:
        _sync_replay_sampler_writes(replay_buffer)
        while len(replay_buffer) < min_replay_decisions:
            time.sleep(poll_interval_s)
            polls += 1
            _sync_replay_sampler_writes(replay_buffer)
            elapsed = wait_timer.elapsed()
            if elapsed >= next_log_s:
                torchrl_logger.info(
                    "waiting for replay iteration %d decisions %d/%d " "elapsed_s %.1f",
                    iteration,
                    len(replay_buffer),
                    min_replay_decisions,
                    elapsed,
                )
                next_log_s = elapsed + log_interval_s
        elapsed = wait_timer.elapsed()
    return {
        "buffer/wait_polls": polls,
        "buffer/wait_s": elapsed,
        "buffer/decisions_before_update": len(replay_buffer),
    }


def replay_ready_target(cfg) -> int:
    """Number of sampleable decisions required before an update starts."""
    if cfg.collector.min_replay_decisions:
        return int(cfg.collector.min_replay_decisions)
    return (
        int(cfg.collector.groups_per_iter)
        * candidate_group_size(cfg)
        * int(cfg.env.max_outer_steps)
    )


def update(
    replay_buffer: TensorDictReplayBuffer,
    loss_module: ClipPPOLoss,
    optim: torch.optim.Optimizer,
    scheduler,
    cfg,
    device: torch.device,
    *,
    logger=None,
    iteration: int,
) -> dict[str, float | int]:
    """Run one PPO update over currently sampleable replay-buffer decisions."""
    num_decisions = len(replay_buffer)
    target_samples = max(
        num_decisions * int(cfg.buffer.consume_after_n_samples),
        num_decisions,
    )
    accumulate = max(int(cfg.loss.accumulate_batches), 1)
    losses = []
    clip_fractions = []
    ess = []
    grad_norms = []
    optim_steps = 0
    trained_decisions = 0
    micro_batches = 0

    def optimizer_step() -> None:
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

    with timeit("train") as train_timer:
        while trained_decisions < target_samples and len(replay_buffer):
            batch = replay_buffer.sample()
            batch = batch.to(device)
            trained_decisions += batch.shape[0]
            loss_vals = loss_module(batch)
            loss = loss_vals["loss_objective"] / accumulate
            loss.backward()
            micro_batches += 1
            losses.append(loss_vals["loss_objective"].detach())
            clip_fractions.append(loss_vals["clip_fraction"].detach())
            # Token-level ESS retains action feature dimensions. Reduce each
            # minibatch before aggregation so partial batches do not make the
            # diagnostic shapes heterogeneous.
            ess.append(loss_vals["ESS"].detach().mean())
            if micro_batches % accumulate == 0:
                optimizer_step()
        tail_micro_batches = micro_batches % accumulate
        if tail_micro_batches:
            # Each micro-batch loss was divided by the full `accumulate`, but
            # the tail optimizer step accumulated fewer micro-batches. Rescale
            # the accumulated gradients so the tail step averages over the
            # micro-batches that actually contributed to it.
            for param in loss_module.parameters():
                if param.grad is not None:
                    param.grad.mul_(accumulate / tail_micro_batches)
            optimizer_step()

    train_time = max(train_timer.elapsed(), 1e-9)
    metrics: dict[str, float | int] = {
        "train/decisions": trained_decisions,
        "train/micro_batches": micro_batches,
        "train/optim_steps": optim_steps,
        "throughput/train_decisions_per_s": trained_decisions / train_time,
        "throughput/optim_steps_per_s": optim_steps / train_time,
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
    log_metrics(logger, metrics, iteration)
    return metrics


_WORKER_ADVANTAGE_PATH = "replay_buffer._transform[0]"


def _advantage_stats(
    advantage: MCAdvantage, collector: MultiCollector | None = None
) -> list[dict[str, float | int]]:
    if collector is not None and not advantage.is_shared:
        return collector.map_fn(f"{_WORKER_ADVANTAGE_PATH}.get_stats")
    return [advantage.get_stats()]


def reset_advantage_state(
    advantage: MCAdvantage, collector: MultiCollector | None = None
) -> None:
    """Clear incomplete groups and reset counters at a policy boundary."""
    advantage.clear_queues()
    advantage.reset_stats()
    if collector is not None and not advantage.is_shared:
        collector.map_fn(f"{_WORKER_ADVANTAGE_PATH}.clear_queues")
        collector.map_fn(f"{_WORKER_ADVANTAGE_PATH}.reset_stats")


def reset_collection_state(advantage: MCAdvantage, collector: MultiCollector) -> None:
    """Drop partial trajectories before advancing the behavior policy."""
    reset_advantage_state(advantage, collector)
    collector.map_fn("reset")


def advantage_metrics(
    advantage: MCAdvantage, collector: MultiCollector | None = None
) -> dict[str, float | int]:
    """Compact metrics snapshot from GRPO replay-transform writer states."""
    stats = _advantage_stats(advantage, collector)
    completed_trajectories = sum(
        int(worker["completed_trajectories"]) for worker in stats
    )
    return {
        "buffer/complete_groups": sum(
            int(worker["completed_groups"]) for worker in stats
        ),
        "buffer/kept_groups": sum(int(worker["written_groups"]) for worker in stats),
        "buffer/skipped_groups": sum(int(worker["dropped_groups"]) for worker in stats),
        "buffer/rescued_groups": sum(int(worker["rescued_groups"]) for worker in stats),
        "buffer/queued_groups": sum(int(worker["queued_groups"]) for worker in stats),
        "buffer/queued_trajectories": sum(
            int(worker["queued_trajectories"]) for worker in stats
        ),
        "buffer/max_queued_trajectories_per_group": (
            max(int(worker["max_queued_trajectories_per_group"]) for worker in stats)
        ),
        "collector/completed_decisions": sum(
            int(worker["completed_decisions"]) for worker in stats
        ),
        "collector/completed_trajectories": completed_trajectories,
        "collector/successful_trajectories": sum(
            int(worker["successful_trajectories"]) for worker in stats
        ),
        "collector/trajectory_return_sum": sum(
            float(worker["trajectory_return_sum"]) for worker in stats
        ),
        "collector/trajectory_return_max": (
            max(float(worker["trajectory_return_max"]) for worker in stats)
            if completed_trajectories
            else 0.0
        ),
    }


def iteration_metrics(
    cfg,
    *,
    num_decisions: int,
    total_episodes: int,
    collect_metrics: dict[str, float | int],
    group_metrics: dict[str, float | int],
    train_metrics: dict[str, float | int],
    eval_metrics: dict[str, float | int],
    timings: dict[str, float],
) -> tuple[dict[str, float | int], float]:
    """Merge per-iteration VLA GRPO metrics into the logger payload."""
    completed_trajectories = int(group_metrics["collector/completed_trajectories"])
    train_success = float(group_metrics["collector/successful_trajectories"]) / max(
        completed_trajectories, 1
    )
    collect_time = max(timings["time/collect"], 1e-9)
    completed_decisions = int(group_metrics["collector/completed_decisions"])
    metrics: dict[str, float | int] = {
        "train/success_rate": train_success,
        "train/episodes_total": total_episodes,
        "buffer/decisions": num_decisions,
        "throughput/inference_env_steps_per_s": (
            completed_decisions * cfg.env.chunk_size / collect_time
        ),
        "throughput/inference_decisions_per_s": completed_decisions / collect_time,
    }
    metrics.update(collect_metrics)
    metrics.update(group_metrics)
    metrics.update(train_metrics)
    metrics.update(eval_metrics)
    metrics.update(timings)
    return metrics, train_success


def log_iteration_summary(
    iteration: int,
    *,
    train_success: float,
    num_decisions: int,
    group_metrics: dict[str, float | int],
    timings: dict[str, float],
) -> None:
    """One-line training progress summary for stdout logs."""
    torchrl_logger.info(
        "iteration %d success %.3f decisions %d kept_groups %d "
        "skipped_groups %d queued_trajs %d collect_s %.1f train_s %.1f",
        iteration,
        train_success,
        num_decisions,
        int(group_metrics["buffer/kept_groups"]),
        int(group_metrics["buffer/skipped_groups"]),
        int(group_metrics["buffer/queued_trajectories"]),
        timings["time/collect"],
        timings["time/train"],
    )


def checkpoint_tensordict(
    policy: torch.nn.Module, optim, scheduler, iteration: int
) -> TensorDict:
    """TensorDict checkpoint payload for the VLA recipe."""
    return TensorDict(
        {
            "policy": policy_weights(policy),
            "iteration": torch.tensor(iteration, dtype=torch.long),
            "torch_rng_state": torch.get_rng_state(),
            "optim": NonTensorData(optim.state_dict()),
            "scheduler": NonTensorData(scheduler.state_dict()),
        },
        batch_size=[],
    )


def save_checkpoint(path, policy: torch.nn.Module, optim, scheduler, iteration: int):
    checkpoint_tensordict(policy, optim, scheduler, iteration).save(path)


def load_checkpoint(path, policy: torch.nn.Module, optim, scheduler, device) -> int:
    checkpoint = TensorDict.load(path)
    apply_policy_weights(policy, checkpoint["policy"], device)
    optim.load_state_dict(checkpoint["optim"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    return int(checkpoint["iteration"].item()) + 1


def log_metrics(logger, metrics: dict, step: int) -> None:
    # One log_metrics call per step (not a per-key log_scalar loop): log_scalar
    # defaults to commit=False, so looping it never advances WandB's step and
    # every iteration collapses into a single history record. log_metrics
    # commits the whole step at once -- the pattern every other sota script uses.
    if logger is not None:
        logger.log_metrics(metrics, step)
