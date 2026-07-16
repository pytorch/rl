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

from tensordict import NonTensorData, NonTensorStack, TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torchrl._utils import logger as torchrl_logger, timeit
from torchrl.collectors import Evaluator, MultiCollector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.vla import (
    ACTION_CHUNK_KEY,
    ACTION_TOKENS_KEY,
    ActionTokenizerBase,
    IMAGE_KEY,
    INSTRUCTION_KEY,
    STATE_KEY,
    UniformActionTokenizer,
    VocabTailActionTokenizer,
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


def video_eval_episodes(cfg) -> int:
    """Number of trajectories to run in the bounded video evaluator."""
    value = _cfg_get(cfg.logger, "video_episodes", 1)
    if value is None:
        value = 1
    value = float(value)
    if 0.0 < value < 1.0:
        return max(1, int(round(float(cfg.logger.eval_episodes) * value)))
    return max(0, int(value))


def candidate_group_size(cfg) -> int:
    """Number of rollout candidates collected for each GRPO group."""
    return int(cfg.collector.candidate_group_size or cfg.collector.group_size)


def collector_group_workers(cfg) -> int:
    """Number of independently advancing logical GRPO group workers."""
    groups_per_iter = int(cfg.collector.groups_per_iter)
    num_collectors = _cfg_get(cfg.collector, "num_collectors", None)
    envs_per_collector = _cfg_get(cfg.collector, "envs_per_collector", None)
    if num_collectors is None or envs_per_collector is None:
        # Small factory tests and standalone replay users need not describe a
        # collector topology. Falling back to the update target preserves the
        # historical capacity calculation for those callers.
        return groups_per_iter
    total_envs = int(num_collectors) * int(envs_per_collector)
    if not _cfg_get(cfg.env, "parallel_group_repeats", False):
        return total_envs
    group_size = int(cfg.collector.group_size)
    return (total_envs + group_size - 1) // group_size


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
        service_backend=_cfg_get(cfg.logger, "service_backend", "direct"),
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
        policy = TinyVLA(
            action_dim=cfg.env.action_dim,
            chunk_size=cfg.env.chunk_size,
            action_head="tokens",
            vocab_size=cfg.tokenizer.vocab_size,
            hidden_dim=cfg.policy.hidden_dim,
            log_probs_mode=log_probs_mode,
            return_vla_action_container=False,
            device=device,
        )
        # TinyVLA contains lazy layers. Materialize them before the policy is
        # cloned into the inference process so the first weight sync copies
        # real parameters instead of independently initialized placeholders.
        init_td = TensorDict(
            {
                "observation": {
                    "image": torch.zeros(
                        1, *cfg.env.image_shape, dtype=torch.uint8, device=device
                    ),
                    "state": torch.zeros(
                        1, cfg.env.state_dim, dtype=torch.float32, device=device
                    ),
                },
                "language_instruction": NonTensorStack("initialize policy"),
            },
            batch_size=[1],
            device=device,
        )
        with torch.no_grad():
            policy(init_td)
        return policy
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


def make_inference_policy(
    cfg,
    device: torch.device,
    *,
    policy_micro_batch_size: int | None = None,
) -> VLAWrapperBase:
    """Build the rollout policy in inference mode.

    The learner returned by :func:`make_policy` must remain in training mode,
    whereas the independently constructed inference-server copy must disable
    dropout and other training-only state before serving rollout and
    evaluation requests.
    """
    policy = make_policy(
        cfg,
        device,
        policy_micro_batch_size=policy_micro_batch_size,
    )
    policy.eval()
    return policy


def _load_openvla_dataset_statistics(spec: str) -> dict:
    """Load action statistics without instantiating an OpenVLA model."""
    if not _has_openvla:
        raise ImportError("The openvla backend requires the local openvla module.")
    from openvla import _load_dataset_statistics

    return _load_dataset_statistics(spec)


def make_action_tokenizer(cfg) -> ActionTokenizerBase | None:
    """Build the environment-side action codec independently of a policy."""
    if cfg.policy.backend == "openvla":
        if cfg.policy.mode == "l1":
            return None
        # The token-to-action mapping only needs the normalization statistics;
        # do not load a 7B learner model merely to construct the CPU env codec.
        statistics_source = cfg.policy.dataset_statistics or cfg.policy.checkpoint
        norm_stats = _load_openvla_dataset_statistics(statistics_source)
        return VocabTailActionTokenizer.from_norm_stats(
            norm_stats,
            cfg.policy.unnorm_key,
            num_bins=int(cfg.tokenizer.vocab_size),
        )
    return UniformActionTokenizer(cfg.tokenizer.vocab_size, low=-1.0, high=1.0)


def policy_io_keys(cfg) -> tuple[list[NestedKey], list[NestedKey]]:
    """Return the VLA TensorDict contract without constructing the model."""
    if cfg.policy.backend == "tiny":
        return (
            [IMAGE_KEY, STATE_KEY, INSTRUCTION_KEY],
            [ACTION_TOKENS_KEY, LOG_PROBS_KEY],
        )
    if cfg.policy.backend != "openvla":
        raise ValueError(f"Unknown policy backend {cfg.policy.backend!r}.")

    in_keys: list[NestedKey] = [IMAGE_KEY]
    if cfg.policy.mode == "l1" and cfg.policy.use_proprio:
        in_keys.append(STATE_KEY)
    in_keys.append(INSTRUCTION_KEY)
    if cfg.policy.use_wrist_image:
        in_keys.append(("observation", "wrist_image"))

    if cfg.policy.mode == "tokens":
        out_keys: list[NestedKey] = [ACTION_TOKENS_KEY, LOG_PROBS_KEY]
    elif cfg.policy.mode == "l1":
        out_keys = [ACTION_CHUNK_KEY]
    else:
        raise ValueError(
            f"policy.mode must be 'tokens' or 'l1', got {cfg.policy.mode!r}."
        )
    return in_keys, out_keys


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
        env_factory = partial(
            _make_libero_worker,
            cfg,
            group_repeats=group_repeats,
            eval_mode=eval_mode,
            from_pixels=from_pixels,
            worker_idx_offset=worker_idx_offset,
            render_gpu_device_id=render_gpu_device_id,
        )
        base = ParallelEnv(
            num_envs,
            env_factory,
            create_env_kwargs=[
                {"worker_idx": worker_idx} for worker_idx in range(num_envs)
            ],
            mp_start_method="spawn",
            # MuJoCo runs on CPU; pin the env device so the collector/rollout
            # cast the GPU policy's action back to CPU before stepping (else a
            # cuda action reaches the CPU transforms -> mixed-device error)
            device="cpu",
            metadata_from_workers=bool(
                _cfg_get(cfg.env, "metadata_from_workers", False)
            ),
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
    env_factory = partial(
        _make_env_worker,
        cfg,
        tokenizer,
        group_repeats=group_repeats,
        seed=seed,
        device=device if cfg.env.backend == "toy" else None,
        worker_idx_offset=worker_idx_offset,
        render_gpu_device_id=render_gpu_device_id,
    )
    return ParallelEnv(
        num_envs,
        env_factory,
        create_env_kwargs=[
            {"worker_idx": worker_idx} for worker_idx in range(num_envs)
        ],
        mp_start_method="spawn",
        device=device,
        metadata_from_workers=bool(_cfg_get(cfg.env, "metadata_from_workers", False)),
    )


def make_replay_buffer(
    cfg, device: torch.device
) -> tuple[TensorDictReplayBuffer, MCAdvantage]:
    # Storage holds one update's accepted decisions; the transform may carry
    # incomplete groups across updates. The write path computes group-relative
    # advantage (and drops degenerate groups) on whole trajectories, while the
    # read path samples accepted decisions without replacement. The advantage
    # transform is returned too so the training loop can read and reset its
    # per-update counters without clearing incomplete groups.
    capacity_group_waves = max(int(cfg.buffer.capacity_group_waves), 1)
    candidate_size = candidate_group_size(cfg)
    groups_per_iter = int(cfg.collector.groups_per_iter)
    if groups_per_iter < 1:
        raise ValueError(
            "collector.groups_per_iter must be positive, got " f"{groups_per_iter}."
        )
    # groups_per_iter is an update trigger, not a collector-wave shape. Size
    # storage for either a full target batch or a full logical worker wave so
    # small update targets remain safe under asynchronous overcollection.
    capacity_groups = max(groups_per_iter, collector_group_workers(cfg))
    capacity = (
        capacity_groups
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
        # The update samples one complete policy batch, then forms
        # trajectory-aware optimizer batches itself. ``mini_batch_size`` only
        # limits the number of decisions sent through the VLA at once.
        batch_size=None,
        consume_after_n_samples=cfg.buffer.consume_after_n_samples,
        shared=shared_init,
    )
    advantage = MCAdvantage(
        grpo_size=cfg.collector.group_size,
        prompt_key="group_id",
        rewards_key=("next", "reward"),
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
        entropy_bonus=True,
        entropy_coeff=0.0,
        # SimpleVLA averages decisions within a trajectory, then averages
        # trajectories. Keep the token loss unreduced so the update can
        # reproduce that weighting even when trajectories have different
        # lengths.
        reduction="none",
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
    device: torch.device,
    *,
    policy_factory: Callable[[], VLAWrapperBase],
    tokenizer: ActionTokenizerBase | None,
    replay_buffer: TensorDictReplayBuffer,
    post_collect_hook: Callable[[TensorDictBase], None] | None = None,
) -> tuple[MultiCollector, ProcessInferenceServer, PolicyClientModule]:
    """Build the TorchRL-native VLA rollout stack.

    The stack is always a ``MultiCollector`` whose workers own batched
    ``ParallelEnv`` instances and call a shared process policy server through
    ``PolicyClientModule``. Only the inference-server child invokes
    ``policy_factory``; the collector builder never receives or constructs the
    learner policy. The collector writes complete trajectories directly into
    the replay buffer.
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

    if cfg.env.backend == "libero":
        _validate_libero_env_count(
            cfg,
            total_envs,
            group_repeats=_training_group_repeats(cfg),
        )
    env_device = torch.device("cpu")
    ctx = mp.get_context("spawn")
    torchrl_logger.info(
        "Collector setup: creating shared transport for %d collectors.",
        num_collectors,
    )
    transport = MPTransport(ctx=ctx, use_manager=True)
    eval_client = transport.client()
    rollout_clients = [transport.client() for _ in range(num_collectors)]
    torchrl_logger.info(
        "Collector setup: starting inference-server child on %s; the child "
        "will instantiate the rollout policy.",
        device,
    )
    server = ProcessInferenceServer(
        policy_factory=policy_factory,
        transport=transport,
        server_config=_server_config_from_collector(cfg),
        # OpenVLA wrappers own their input placement: raw uint8 images must stay
        # on CPU through the reference preprocessing path, then the processed
        # model inputs are moved to the model device.  Let the policy factory
        # construct the model on ``device`` but do not have the generic
        # inference server pre-move the whole request TensorDict to CUDA.
        policy_device=None if cfg.policy.backend == "openvla" else device,
        output_device=env_device,
        mp_context=ctx,
    ).start()
    torchrl_logger.info(
        "Collector setup: inference server is ready at policy version %d.",
        server.policy_version,
    )

    policy_in_keys, policy_out_keys = policy_io_keys(cfg)
    policy_out_keys = [*policy_out_keys, "policy_version"]
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
    torchrl_logger.info(
        "Collector setup: constructing MultiCollector with %d subprocesses "
        "and %d environment workers each.",
        num_collectors,
        envs_per_collector,
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
    torchrl_logger.info("Collector setup: MultiCollector is ready.")
    return collector, server, eval_policy


def _make_eval_env(
    cfg,
    tokenizer: ActionTokenizerBase | None,
    logger,
    device: torch.device,
    num_envs: int | None = None,
) -> TransformedEnv:
    env = make_env(
        cfg,
        tokenizer,
        seed=cfg.env.seed + 1,
        device=device if cfg.env.backend == "toy" else None,
        eval_mode=True,
        from_pixels=logger is not None,
        num_envs=num_envs,
    )
    if logger is not None:
        env.append_transform(
            VideoRecorder(logger, tag="eval/video", fps=cfg.logger.video_fps)
        )
    return env


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
    *,
    record_video: bool = False,
    num_trajectories: int | None = None,
    num_envs: int | None = None,
) -> Evaluator:
    """Build the TorchRL evaluator used by the VLA GRPO recipe."""
    record_video = bool(record_video) and logger is not None
    if (
        record_video
        and cfg.logger.eval_backend == "process"
        and getattr(logger, "service_backend", "direct") == "direct"
    ):
        raise ValueError(
            "logger.eval_backend='process' requires a process- or Ray-backed "
            "logger so VideoRecorder receives a transferable client. Set "
            "logger.service_backend='process'."
        )
    video_logger = logger.client() if record_video else None
    env_factory = partial(
        _make_eval_env,
        cfg,
        tokenizer,
        video_logger,
        device,
        num_envs,
    )
    if num_trajectories is None:
        num_trajectories = int(cfg.logger.eval_episodes)
    return Evaluator(
        env_factory,
        policy=None,
        policy_factory=partial(_policy_module_factory, policy),
        num_trajectories=num_trajectories,
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


def make_video_evaluator(
    cfg,
    tokenizer: ActionTokenizerBase | None,
    policy,
    logger,
    device: torch.device,
) -> Evaluator | None:
    """Build the bounded visual evaluator, or ``None`` when video is disabled."""
    if not bool(_cfg_get(cfg.logger, "record_video", True)) or logger is None:
        return None
    num_trajectories = video_eval_episodes(cfg)
    if num_trajectories < 1:
        return None
    num_envs = _cfg_get(cfg.logger, "video_num_envs", 1)
    if num_envs is not None:
        num_envs = int(num_envs)
    return make_evaluator(
        cfg,
        tokenizer,
        policy,
        logger,
        device,
        record_video=True,
        num_trajectories=num_trajectories,
        num_envs=num_envs,
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
    advantage: MCAdvantage,
    collector: MultiCollector,
    *,
    min_replay_groups: int,
    min_replay_decisions: int,
    poll_interval_s: float,
    log_interval_s: float,
    iteration: int,
    max_completed_trajectories: int | None = None,
) -> dict[str, float | int]:
    """Wait for complete useful GRPO groups, plus an optional decision floor."""
    polls = 0
    next_log_s = 0.0
    with timeit("replay_wait") as wait_timer:
        _sync_replay_sampler_writes(replay_buffer)
        group_metrics = advantage_metrics(advantage, collector)
        while True:
            ready = _replay_ready(
                sampleable_decisions=len(replay_buffer),
                kept_groups=int(group_metrics["buffer/kept_groups"]),
                min_replay_groups=min_replay_groups,
                min_replay_decisions=min_replay_decisions,
            )
            trajectory_budget_reached = (
                max_completed_trajectories is not None
                and int(group_metrics["collector/completed_trajectories"])
                >= max_completed_trajectories
            )
            if ready or trajectory_budget_reached:
                break
            time.sleep(poll_interval_s)
            polls += 1
            _sync_replay_sampler_writes(replay_buffer)
            group_metrics = advantage_metrics(advantage, collector)
            elapsed = wait_timer.elapsed()
            if elapsed >= next_log_s:
                torchrl_logger.info(
                    "waiting for replay iteration %d kept_groups %d/%d "
                    "complete_groups %d skipped_groups %d queued_trajectories %d "
                    "completed_trajectories %d successes %d decisions %d/%d "
                    "elapsed_s %.1f",
                    iteration,
                    int(group_metrics["buffer/kept_groups"]),
                    min_replay_groups,
                    int(group_metrics["buffer/complete_groups"]),
                    int(group_metrics["buffer/skipped_groups"]),
                    int(group_metrics["buffer/queued_trajectories"]),
                    int(group_metrics["collector/completed_trajectories"]),
                    int(group_metrics["collector/successful_trajectories"]),
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
        "buffer/kept_groups_before_update": int(group_metrics["buffer/kept_groups"]),
        "buffer/trajectory_budget_reached": int(trajectory_budget_reached),
        "collector/completed_trajectories_before_update": int(
            group_metrics["collector/completed_trajectories"]
        ),
    }


def _replay_ready(
    *,
    sampleable_decisions: int,
    kept_groups: int,
    min_replay_groups: int,
    min_replay_decisions: int,
) -> bool:
    """Return whether a complete, useful rollout batch is ready to train."""
    return (
        kept_groups >= min_replay_groups
        and sampleable_decisions > 0
        and sampleable_decisions >= min_replay_decisions
    )


def replay_ready_targets(cfg) -> tuple[int, int]:
    """Useful-group and optional decision thresholds for an optimizer update."""
    min_replay_decisions = _cfg_get(cfg.collector, "min_replay_decisions", None)
    return (
        int(cfg.collector.groups_per_iter),
        0 if min_replay_decisions is None else int(min_replay_decisions),
    )


def _decision_loss(loss: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Average token-level PPO losses into one scalar per decision."""
    if loss.numel() % batch_size:
        raise RuntimeError(
            "The unreduced PPO loss cannot be partitioned by decision: "
            f"got shape {tuple(loss.shape)} for batch size {batch_size}."
        )
    return loss.reshape(batch_size, -1).mean(-1)


def _decision_scalar_tensor(
    value: torch.Tensor,
    *,
    num_decisions: int,
    name: str,
) -> torch.Tensor:
    """Collapse a possibly broadcast metadata leaf to one value per decision."""
    if value.shape[0] != num_decisions:
        raise RuntimeError(
            f"Expected one {name} row per decision, got shape "
            f"{tuple(value.shape)} for {num_decisions} decisions."
        )
    rows = value.reshape(num_decisions, -1)
    if not rows.shape[1]:
        raise RuntimeError(f"{name} has an empty per-decision shape.")
    scalar = rows[:, 0]
    if rows.shape[1] > 1 and not torch.equal(
        rows, scalar.unsqueeze(-1).expand_as(rows)
    ):
        raise RuntimeError(f"Each decision must carry exactly one {name} value.")
    return scalar


def _policy_version_span_metrics(
    versions: torch.Tensor,
    ids: torch.Tensor | None,
    *,
    prefix: str,
) -> dict[str, float | int]:
    """Measure behavior-policy mixing within trajectories or GRPO groups."""
    if ids is None or not isinstance(ids, torch.Tensor):
        return {}
    ids = (
        _decision_scalar_tensor(
            ids,
            num_decisions=versions.numel(),
            name=f"{prefix} id",
        )
        .detach()
        .cpu()
    )
    versions = versions.detach().cpu()
    spans = torch.stack(
        [
            versions[ids == unit_id].max() - versions[ids == unit_id].min()
            for unit_id in ids.unique()
        ]
    )
    return {
        f"train/mixed_policy_{prefix}_fraction": float((spans > 0).float().mean()),
        f"train/{prefix}_policy_version_span_max": int(spans.max()),
    }


def policy_staleness_metrics(
    data: TensorDictBase,
    current_policy_version: int,
) -> dict[str, float | int]:
    """Summarize behavior-policy age without filtering replay data."""
    version = data.get("policy_version", None)
    if version is None:
        raise KeyError(
            "VLA-GRPO staleness accounting requires the inference server's "
            "'policy_version' annotation."
        )
    if not isinstance(version, torch.Tensor):
        raise TypeError(
            "Expected tensor policy-version metadata, got " f"{type(version).__name__}."
        )
    versions = _decision_scalar_tensor(
        version,
        num_decisions=data.shape[0],
        name="policy version",
    ).to(dtype=torch.long)
    current_policy_version = int(current_policy_version)
    staleness = current_policy_version - versions
    if bool((staleness < 0).any()):
        raise RuntimeError(
            "Replay contains a behavior-policy version newer than the trainer: "
            f"trainer={current_policy_version}, newest={int(versions.max())}."
        )
    staleness_float = staleness.float()
    metrics: dict[str, float | int] = {
        "train/current_policy_version": current_policy_version,
        "train/behavior_policy_version_min": int(versions.min()),
        "train/behavior_policy_version_max": int(versions.max()),
        "train/policy_staleness_min": int(staleness.min()),
        "train/policy_staleness_mean": float(staleness_float.mean()),
        "train/policy_staleness_p95": float(torch.quantile(staleness_float, 0.95)),
        "train/policy_staleness_max": int(staleness.max()),
        "train/stale_decision_fraction": float((staleness > 0).float().mean()),
    }
    metrics.update(
        _policy_version_span_metrics(
            versions,
            data.get(("collector", "traj_ids"), None),
            prefix="trajectory",
        )
    )
    metrics.update(
        _policy_version_span_metrics(
            versions,
            data.get("group_id", None),
            prefix="group",
        )
    )
    return metrics


def _check_pre_update_policy_match(metrics: dict[str, float], cfg) -> None:
    """Fail early when rollout and training policies disagree before PPO."""
    min_ess = _cfg_get(cfg.train, "pre_update_min_ess", None)
    max_clip_fraction = _cfg_get(cfg.train, "pre_update_max_clip_fraction", None)
    if min_ess is not None and metrics["train/pre_update_ESS"] < float(min_ess):
        raise RuntimeError(
            "The rollout and training policies disagree before the first "
            "optimizer step: pre-update ESS is "
            f"{metrics['train/pre_update_ESS']:.6f}, below {float(min_ess):.6f}; "
            f"clip_fraction={metrics['train/pre_update_clip_fraction']:.6f}, "
            f"mean_ratio={metrics.get('train/pre_update_mean_ratio', float('nan')):.6f}, "
            f"max_ratio={metrics.get('train/pre_update_max_ratio', float('nan')):.6f}."
        )
    if max_clip_fraction is not None and metrics[
        "train/pre_update_clip_fraction"
    ] > float(max_clip_fraction):
        raise RuntimeError(
            "The rollout and training policies disagree before the first "
            "optimizer step: pre-update clip fraction is "
            f"{metrics['train/pre_update_clip_fraction']:.6f}, above "
            f"{float(max_clip_fraction):.6f}; "
            f"ESS={metrics['train/pre_update_ESS']:.6f}, "
            f"mean_ratio={metrics.get('train/pre_update_mean_ratio', float('nan')):.6f}, "
            f"max_ratio={metrics.get('train/pre_update_max_ratio', float('nan')):.6f}."
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
    current_policy_version: int | None = None,
) -> dict[str, float | int]:
    """Run a trajectory-weighted PPO update over one rollout-policy batch."""
    num_decisions = len(replay_buffer)
    if not num_decisions:
        raise RuntimeError("Cannot update from an empty replay buffer.")
    epochs = max(int(cfg.buffer.consume_after_n_samples), 1)
    forward_batch_size = max(int(cfg.loss.mini_batch_size), 1)
    trajectories_per_step = max(int(cfg.loss.trajectories_per_optimizer_step), 1)
    loss_sum = torch.zeros((), device=device)
    diagnostic_sums = {
        "clip_fraction": torch.zeros((), device=device),
        "ESS": torch.zeros((), device=device),
        "kl_approx": torch.zeros((), device=device),
        "mean_ratio": torch.zeros((), device=device),
        "entropy": torch.zeros((), device=device),
        "loss_entropy": torch.zeros((), device=device),
    }
    diagnostic_decisions = 0
    grad_norms = []
    optim_steps = 0
    trained_decisions = 0
    trained_trajectories = 0
    micro_batches = 0
    pre_update_metrics: dict[str, float] | None = None
    staleness_metrics: dict[str, float | int] = {}

    def optimizer_step() -> None:
        nonlocal optim_steps
        grad_norm = torch.nn.utils.clip_grad_norm_(
            loss_module.parameters(), cfg.optim.max_grad_norm
        )
        if not torch.isfinite(grad_norm):
            raise FloatingPointError(
                f"Non-finite gradient norm before optimizer step: {grad_norm}."
            )
        grad_norms.append(grad_norm.detach())
        optim.step()
        optim.zero_grad(set_to_none=True)
        scheduler.step()
        optim_steps += 1

    optim.zero_grad(set_to_none=True)
    with timeit("train") as train_timer:
        for _ in range(epochs):
            data = replay_buffer.sample(batch_size=num_decisions)
            if current_policy_version is not None and not staleness_metrics:
                staleness_metrics = policy_staleness_metrics(
                    data, current_policy_version
                )
            traj_ids = data.get(("collector", "traj_ids"), None)
            if traj_ids is None:
                raise KeyError(
                    "Trajectory-aware VLA PPO requires "
                    "('collector', 'traj_ids') in replay data."
                )
            traj_ids = traj_ids.reshape(-1).cpu()
            if traj_ids.numel() != data.shape[0]:
                raise RuntimeError(
                    "Expected one trajectory id per replay decision, got "
                    f"{traj_ids.numel()} ids for {data.shape[0]} decisions."
                )
            _, decision_to_traj = traj_ids.unique(sorted=False, return_inverse=True)
            num_trajectories = int(decision_to_traj.max().item()) + 1
            trajectory_lengths = torch.bincount(
                decision_to_traj, minlength=num_trajectories
            )
            trajectory_order = torch.randperm(num_trajectories)

            for traj_start in range(0, num_trajectories, trajectories_per_step):
                step_trajectories = trajectory_order[
                    traj_start : traj_start + trajectories_per_step
                ]
                step_trajectory_count = int(step_trajectories.numel())
                decision_indices = (
                    torch.isin(decision_to_traj, step_trajectories)
                    .nonzero(as_tuple=False)
                    .squeeze(-1)
                )
                step_diagnostic_sums = {
                    key: torch.zeros((), device=device) for key in diagnostic_sums
                }
                step_max_ratio = torch.zeros((), device=device)
                step_decisions = 0
                step_loss = torch.zeros((), device=device)

                for decision_start in range(
                    0, decision_indices.numel(), forward_batch_size
                ):
                    indices = decision_indices[
                        decision_start : decision_start + forward_batch_size
                    ]
                    batch_size = int(indices.numel())
                    batch = data[indices].to(device)
                    loss_vals = loss_module(batch)
                    per_decision_loss = _decision_loss(
                        loss_vals["loss_objective"], batch_size
                    )
                    batch_traj = decision_to_traj[indices]
                    weights = (
                        trajectory_lengths[batch_traj]
                        .to(device=device, dtype=per_decision_loss.dtype)
                        .reciprocal_()
                    )
                    weights.div_(step_trajectory_count)
                    weighted_loss = (per_decision_loss * weights).sum()
                    if not torch.isfinite(weighted_loss):
                        raise FloatingPointError(
                            "Non-finite trajectory-weighted PPO loss."
                        )
                    weighted_loss.backward()

                    step_loss += weighted_loss.detach()
                    trained_decisions += batch_size
                    micro_batches += 1
                    step_decisions += batch_size
                    for key in step_diagnostic_sums:
                        value = loss_vals[key].detach().mean()
                        step_diagnostic_sums[key] += value * batch_size
                        diagnostic_sums[key] += value * batch_size
                    step_max_ratio = torch.maximum(
                        step_max_ratio, loss_vals["max_ratio"].detach().max()
                    )
                    diagnostic_decisions += batch_size

                if pre_update_metrics is None:
                    pre_update_metrics = {
                        "train/pre_update_clip_fraction": float(
                            step_diagnostic_sums["clip_fraction"] / step_decisions
                        ),
                        "train/pre_update_ESS": float(
                            step_diagnostic_sums["ESS"] / step_decisions
                        ),
                        "train/pre_update_kl_approx": float(
                            step_diagnostic_sums["kl_approx"] / step_decisions
                        ),
                        "train/pre_update_mean_ratio": float(
                            step_diagnostic_sums["mean_ratio"] / step_decisions
                        ),
                        "train/pre_update_max_ratio": float(step_max_ratio),
                    }
                    _check_pre_update_policy_match(pre_update_metrics, cfg)

                loss_sum += step_loss * step_trajectory_count
                trained_trajectories += step_trajectory_count
                optimizer_step()

    train_time = max(train_timer.elapsed(), 1e-9)
    metrics: dict[str, float | int] = {
        "train/decisions": trained_decisions,
        "train/trajectories": trained_trajectories,
        "train/micro_batches": micro_batches,
        "train/optim_steps": optim_steps,
        "train/decisions_per_optim_step": trained_decisions / optim_steps,
        "train/trajectories_per_optim_step": trained_trajectories / optim_steps,
        "throughput/train_decisions_per_s": trained_decisions / train_time,
        "throughput/optim_steps_per_s": optim_steps / train_time,
    }
    metrics.update(
        {
            "train/loss_objective": float(loss_sum / trained_trajectories),
            "train/clip_fraction": float(
                diagnostic_sums["clip_fraction"] / diagnostic_decisions
            ),
            "train/ESS": float(diagnostic_sums["ESS"] / diagnostic_decisions),
            "train/kl_approx": float(
                diagnostic_sums["kl_approx"] / diagnostic_decisions
            ),
            "train/mean_ratio": float(
                diagnostic_sums["mean_ratio"] / diagnostic_decisions
            ),
            "train/entropy": float(diagnostic_sums["entropy"] / diagnostic_decisions),
            "train/loss_entropy": float(
                diagnostic_sums["loss_entropy"] / diagnostic_decisions
            ),
            "train/grad_norm": float(torch.stack(grad_norms).mean()),
        }
    )
    metrics.update(pre_update_metrics or {})
    metrics.update(staleness_metrics)
    log_metrics(logger, metrics, iteration)
    return metrics


_WORKER_ADVANTAGE_PATH = "replay_buffer._transform[0]"


def _advantage_stats(
    advantage: MCAdvantage, collector: MultiCollector | None = None
) -> list[dict[str, float | int]]:
    if collector is not None and not advantage.is_shared:
        return collector.map_fn(f"{_WORKER_ADVANTAGE_PATH}.get_stats")
    return [advantage.get_stats()]


def reset_advantage_stats(
    advantage: MCAdvantage, collector: MultiCollector | None = None
) -> None:
    """Reset per-update counters without discarding incomplete groups."""
    advantage.reset_stats()
    if collector is not None and not advantage.is_shared:
        collector.map_fn(f"{_WORKER_ADVANTAGE_PATH}.reset_stats")


def reset_collection_state(advantage: MCAdvantage, collector: MultiCollector) -> None:
    """Advance metric accounting while preserving all collector work.

    Completed trajectories waiting for the rest of their GRPO group remain in
    ``MCAdvantage`` and environments continue any in-flight trajectory after
    the inference server publishes new weights. Every decision is stamped with
    its actual behavior-policy version, so the resulting lag can be measured
    when the completed group is trained.
    """
    reset_advantage_stats(advantage, collector)


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
    policy: torch.nn.Module,
    optim,
    scheduler,
    iteration: int,
    total_episodes: int | None = None,
) -> TensorDict:
    """TensorDict checkpoint payload for the VLA recipe."""
    checkpoint = TensorDict(
        {
            "policy": policy_weights(policy),
            "iteration": torch.tensor(iteration, dtype=torch.long),
            "torch_rng_state": torch.get_rng_state(),
            "optim": NonTensorData(optim.state_dict()),
            "scheduler": NonTensorData(scheduler.state_dict()),
        },
        batch_size=[],
    )
    if total_episodes is not None:
        checkpoint["total_episodes"] = torch.tensor(total_episodes, dtype=torch.long)
    return checkpoint


def save_checkpoint(
    path,
    policy: torch.nn.Module,
    optim,
    scheduler,
    iteration: int,
    total_episodes: int | None = None,
):
    checkpoint_tensordict(
        policy,
        optim,
        scheduler,
        iteration,
        total_episodes=total_episodes,
    ).save(path)


def load_checkpoint(
    path, policy: torch.nn.Module, optim, scheduler, device
) -> tuple[int, int | None]:
    checkpoint = TensorDict.load(path)
    apply_policy_weights(policy, checkpoint["policy"], device)
    optim.load_state_dict(checkpoint["optim"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    torch.set_rng_state(checkpoint["torch_rng_state"])
    total_episodes = checkpoint.get("total_episodes", None)
    return (
        int(checkpoint["iteration"].item()) + 1,
        None if total_episodes is None else int(total_episodes.item()),
    )


def log_metrics(logger, metrics: dict, step: int) -> None:
    # One log_metrics call per step (not a per-key log_scalar loop): log_scalar
    # defaults to commit=False, so looping it never advances WandB's step and
    # every iteration collapses into a single history record. log_metrics
    # commits the whole step at once -- the pattern every other sota script uses.
    if logger is not None:
        logger.log_metrics(metrics, step)
