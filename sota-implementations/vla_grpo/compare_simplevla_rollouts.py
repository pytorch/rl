#!/usr/bin/env python
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Compare TorchRL VLA rollouts against the SimpleVLA-RL/VeRL rollout.

The script loads one OpenVLA-OFT token policy and evaluates the same
``(task_id, LIBERO init-state id)`` trajectories through two rollout stacks:

* TorchRL: ``LiberoEnv`` + VLA GRPO transforms + TorchRL policy wrapper.
* SimpleVLA-RL: the reference ``verl.workers.rollout.RobHFRollout`` trajectory
  generator from the paper codebase.

The output is a JSON file with per-trajectory records and side-by-side success
rates. Run it from a checkout with both LIBERO and SimpleVLA-RL available, for
example:

.. code-block:: bash

    export SIMPLEVLA_RL_ROOT=/path/to/SimpleVLA-RL
    export LIBERO_ROOT=/path/to/LIBERO
    export PYTHONPATH="${SIMPLEVLA_RL_ROOT}:${LIBERO_ROOT}:${PYTHONPATH:-}"
    python sota-implementations/vla_grpo/compare_simplevla_rollouts.py \
        --task-ids 0 --trial-ids 0 --policy-device cuda:0 --render-gpu 0
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import random
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from datetime import datetime, UTC
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("LIBERO_CONFIG_PATH", os.path.expanduser("~/.libero"))
os.environ.setdefault("ROBOT_PLATFORM", "LIBERO")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("WANDB_MODE", "disabled")

_SIMPLEVLA_ROOT_ENV = os.environ.get("SIMPLEVLA_RL_ROOT")
_LIBERO_ROOT_ENV = os.environ.get("LIBERO_ROOT")
for _path in (_SIMPLEVLA_ROOT_ENV, _LIBERO_ROOT_ENV):
    if _path and _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch

import utils as vla_utils
from omegaconf import DictConfig, OmegaConf
from tensordict import TensorDict
from tensordict.nn import InteractionType, set_interaction_type
from torchrl._utils import logger as torchrl_logger
from torchrl.data.vla import ACTION_TOKENS_KEY
from torchrl.envs import LiberoEnv, TransformedEnv
from torchrl.envs.utils import step_mdp

_has_verl = importlib.util.find_spec("verl") is not None
_DataProto = None
_RobHFRollout = None
_ORIGINAL_TORCH_LOAD = torch.load


def _torch_load_trusted_init_state(*args, **kwargs):
    """Compatibility shim for LIBERO init-state files under torch>=2.6."""
    kwargs.setdefault("weights_only", False)
    return _ORIGINAL_TORCH_LOAD(*args, **kwargs)


@dataclass
class _TrajectorySpec:
    """One LIBERO task / init-state pair to evaluate."""

    task_id: int
    trial_id: int


@dataclass
class _TrajectoryResult:
    """Per-trajectory rollout summary."""

    stack: Literal["torchrl", "simplevla"]
    task_id: int
    trial_id: int
    success: bool
    outer_steps: int | None
    env_steps: int
    reward_sum: float | None = None
    first_tokens: list[int] | None = None
    first_action: list[float] | None = None
    tokens: list[list[int]] | None = None
    actions: list[list[float]] | None = None


@dataclass
class _StackSummary:
    """Success-rate summary for one rollout stack."""

    stack: Literal["torchrl", "simplevla"]
    trajectories: int
    successes: int
    success_rate: float
    env_steps: int


@dataclass
class _ParityResult:
    """Per-trajectory parity metrics between the two rollout stacks."""

    task_id: int
    trial_id: int
    success_agreement: bool
    torchrl_success: bool
    simplevla_success: bool
    first_token_exact_match: bool | None
    first_tokens_exact_match: bool | None
    first_token_compare_count: int
    first_token_mismatch_index: int | None
    torchrl_first_token_at_mismatch: int | None
    simplevla_first_token_at_mismatch: int | None
    first_action_compare_count: int
    first_action_max_abs_diff: float | None
    first_action_mismatch_index: int | None


@dataclass
class _ParitySummary:
    """Aggregate parity metrics for the report header."""

    trajectories: int
    success_agreements: int
    success_agreement_rate: float
    first_token_exact_matches: int
    first_token_exact_match_rate: float
    first_tokens_exact_matches: int
    first_tokens_exact_match_rate: float
    first_action_max_abs_diff: float | None


def _insert_optional_root(root: str | None) -> None:
    if root and root not in sys.path:
        sys.path.insert(0, root)


def _load_verl_rollout(simplevla_root: str | None):
    """Load SimpleVLA-RL's VeRL rollout class as an optional dependency."""
    global _DataProto, _RobHFRollout, _has_verl
    _insert_optional_root(simplevla_root)
    if not _has_verl:
        _has_verl = importlib.util.find_spec("verl") is not None
    if not _has_verl:
        raise ImportError(
            "Could not import 'verl'. Set SIMPLEVLA_RL_ROOT or pass "
            "--simplevla-root pointing to the SimpleVLA-RL checkout."
        )
    if _DataProto is None or _RobHFRollout is None:
        from verl import DataProto
        from verl.workers.rollout import RobHFRollout

        _DataProto = DataProto
        _RobHFRollout = RobHFRollout
    return _DataProto, _RobHFRollout


def _git_commit(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except Exception:
        return None


def _jsonable(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        data = value.detach().cpu()
        if data.numel() == 1:
            return data.reshape(()).item()
        return data.tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _parse_int_list(value: str) -> list[int]:
    values: list[int] = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            values.extend(range(int(start), int(end) + 1))
        else:
            values.append(int(part))
    return values


def _trajectory_specs(
    task_ids: Iterable[int],
    trial_ids: Iterable[int],
) -> list[_TrajectorySpec]:
    return [
        _TrajectorySpec(task_id=int(task_id), trial_id=int(trial_id))
        for task_id in task_ids
        for trial_id in trial_ids
    ]


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_cfg(args: argparse.Namespace, task_id: int) -> DictConfig:
    cfg = OmegaConf.load(Path(args.config))
    cfg.env.backend = "libero"
    cfg.env.task_suite = args.task_suite
    cfg.env.task_ids = [int(task_id)]
    cfg.env.num_envs = 1
    cfg.env.eval_num_envs = 1
    cfg.env.parallel_group_repeats = False
    cfg.env.render_gpu_ids = [int(args.render_gpu)]
    cfg.env.eval_render_gpu_ids = [int(args.render_gpu)]
    # honor the CLI rollout bounds: the chunk transform's StepCounter truncates
    # at cfg.env.max_outer_steps, which would silently cap a larger
    # --max-outer-steps at the yaml default otherwise
    cfg.env.max_outer_steps = int(args.max_outer_steps)
    cfg.env.max_env_steps = int(args.max_env_steps)
    cfg.env.seed = int(args.seed)
    cfg.policy.backend = "openvla"
    cfg.policy.mode = "tokens"
    cfg.policy.checkpoint = args.checkpoint
    cfg.policy.unnorm_key = args.unnorm_key
    cfg.policy.dataset_statistics = args.dataset_statistics
    cfg.policy.device = str(args.policy_device)
    cfg.policy.temperature = float(args.temperature)
    cfg.policy.top_k = args.top_k
    cfg.policy.use_wrist_image = False
    cfg.policy.use_proprio = False
    cfg.policy.num_images_in_input = 1
    cfg.policy.center_crop = bool(args.center_crop)
    cfg.policy.image_backend = args.torchrl_image_backend
    cfg.policy.gripper_binarize = True
    cfg.policy.gripper_binarize_threshold = float(args.gripper_binarize_threshold)
    cfg.policy.gripper_invert = bool(args.gripper_invert)
    cfg.policy.lora_rank = int(args.lora_rank)
    cfg.logger.backend = "none"
    cfg.logger.mode = "disabled"
    return cfg


def _simplevla_cfg(args: argparse.Namespace) -> DictConfig:
    return OmegaConf.create(
        {
            "vla": "openvla-oft",
            "pretrained_checkpoint": args.checkpoint,
            "unnorm_key": args.unnorm_key,
            "model_family": "openvla",
            "task_suite_name": args.task_suite,
            "num_steps_wait": int(args.settle_steps),
            "center_crop": bool(args.center_crop),
            "num_images_in_input": 1,
            "use_proprio": False,
            "action_chunks_len": int(args.chunk_size),
            "temperature": float(args.temperature),
            "do_sample": bool(args.sample_actions),
            "micro_batch_size": int(args.simplevla_batch_size),
            "val_micro_batch_size": int(args.simplevla_batch_size),
            "max_prompt_length": int(args.max_prompt_length),
            "experiment_name": args.experiment_name,
        }
    )


def _policy_and_tokenizer(args: argparse.Namespace):
    device = torch.device(args.policy_device)
    cfg = _load_cfg(args, int(args.task_ids[0]))
    policy = vla_utils.make_policy(cfg, device)
    policy.eval()
    tokenizer = vla_utils.make_action_tokenizer(cfg, policy)
    return cfg, policy, tokenizer


def _make_torchrl_env(
    cfg: DictConfig,
    tokenizer,
    spec: _TrajectorySpec,
    args: argparse.Namespace,
) -> TransformedEnv:
    env_kwargs = dict(cfg.env.env_kwargs or {})
    env_kwargs["render_gpu_device_id"] = int(args.render_gpu)
    base = LiberoEnv(
        args.task_suite,
        task_id=int(spec.task_id),
        camera_height=int(cfg.env.camera_height),
        camera_width=int(cfg.env.camera_width),
        env_kwargs=env_kwargs,
        wrist_camera=None,
        from_pixels=False,
        max_episode_steps=int(args.max_env_steps),
        settle_steps=int(args.settle_steps),
        init_state_mode="fixed",
        init_state_id=int(spec.trial_id),
    )
    return TransformedEnv(base, vla_utils._chunk_transform(cfg, tokenizer))


def _tensor_bool(value: torch.Tensor | None) -> bool:
    if value is None:
        return False
    return bool(torch.as_tensor(value).reshape(-1).any().item())


def _first_flat(value: torch.Tensor, limit: int) -> list[int] | list[float]:
    data = value.detach().cpu()
    while data.ndim > 2:
        data = data[0]
    return data.reshape(-1)[: min(data.numel(), limit)].tolist()


def _decode_env_actions(policy, tokenizer, tokens: torch.Tensor) -> torch.Tensor:
    decoded = tokenizer.decode(tokens)
    gripper_postprocess = getattr(policy, "gripper_postprocess", None)
    if gripper_postprocess is not None:
        decoded = gripper_postprocess.postprocess(decoded)
    return decoded


def _torchrl_rollout_one(
    cfg: DictConfig,
    policy,
    tokenizer,
    spec: _TrajectorySpec,
    args: argparse.Namespace,
) -> _TrajectoryResult:
    env = _make_torchrl_env(cfg, tokenizer, spec, args)
    reward_sum = 0.0
    first_tokens: list[int] | None = None
    first_action: list[float] | None = None
    tokens: list[list[int]] = []
    actions: list[list[float]] = []
    env_steps = 0
    outer_steps = 0
    success = False
    try:
        env.set_seed(int(args.seed))
        td = env.reset()
        interaction = (
            InteractionType.RANDOM
            if bool(args.sample_actions)
            else InteractionType.DETERMINISTIC
        )
        with torch.no_grad(), set_interaction_type(interaction):
            for _ in range(int(args.max_outer_steps)):
                outer_steps += 1
                policy_td = policy(td)
                action_tokens = policy_td.get(ACTION_TOKENS_KEY).detach().cpu()
                decoded = _decode_env_actions(policy, tokenizer, action_tokens)
                tokens.append(_first_flat(action_tokens, action_tokens.numel()))
                actions.append(_first_flat(decoded.float(), decoded.numel()))
                if first_tokens is None:
                    first_tokens = tokens[-1][:32]
                    first_action = _first_flat(decoded.float(), 16)
                next_td = env.step(policy_td)
                reward = next_td.get(("next", "reward"), None)
                if reward is not None:
                    reward_sum += float(torch.as_tensor(reward).sum().item())
                env_steps += int(args.chunk_size)
                success = success or _tensor_bool(
                    next_td.get(("next", "success"), None)
                )
                done = _tensor_bool(next_td.get(("next", "done"), None))
                if done or success:
                    break
                td = step_mdp(next_td)
    finally:
        env.close()
    return _TrajectoryResult(
        stack="torchrl",
        task_id=int(spec.task_id),
        trial_id=int(spec.trial_id),
        success=bool(success),
        outer_steps=outer_steps,
        env_steps=int(min(env_steps, int(args.max_env_steps))),
        reward_sum=float(reward_sum),
        first_tokens=first_tokens,
        first_action=first_action,
        tokens=tokens,
        actions=actions,
    )


def _run_torchrl_stack(
    args: argparse.Namespace,
    specs: list[_TrajectorySpec],
    policy,
    tokenizer,
) -> list[_TrajectoryResult]:
    results = []
    cfg_cache: dict[int, DictConfig] = {}
    for spec in specs:
        cfg = cfg_cache.setdefault(spec.task_id, _load_cfg(args, spec.task_id))
        result = _torchrl_rollout_one(cfg, policy, tokenizer, spec, args)
        torchrl_logger.info(
            "TorchRL rollout task=%s trial=%s success=%s env_steps=%s",
            spec.task_id,
            spec.trial_id,
            result.success,
            result.env_steps,
        )
        results.append(result)
    return results


def _make_simplevla_prompts(specs: list[_TrajectorySpec], meta_info: dict[str, Any]):
    DataProto, _ = _load_verl_rollout(None)
    batch = TensorDict(
        {
            "task_id": torch.tensor(
                [[spec.task_id] for spec in specs], dtype=torch.int64
            ),
            "trial_id": torch.tensor(
                [[spec.trial_id] for spec in specs], dtype=torch.int64
            ),
            "trial_seed": torch.full((len(specs), 1), -1, dtype=torch.int64),
        },
        batch_size=[len(specs)],
    )
    non_tensor_batch = {
        "task_suite_name": np.array(
            [meta_info["task_suite_name"] for _ in specs], dtype=object
        )
    }
    return DataProto(
        batch=batch,
        non_tensor_batch=non_tensor_batch,
        meta_info={"n_samples": 1},
    )


def _run_simplevla_batch(
    rollout,
    specs: list[_TrajectorySpec],
    task_suite_name: str,
    policy,
    tokenizer,
) -> list[_TrajectoryResult]:
    prompts = _make_simplevla_prompts(
        specs,
        {"task_suite_name": task_suite_name},
    )
    output = rollout.generate_sequences(prompts)
    complete = output.batch["complete"].detach().cpu().bool()
    finish_step = output.batch["finish_step"].detach().cpu().long()
    responses = output.batch["responses"].detach().cpu().long()
    vocab_size = getattr(rollout.module, "vocab_size", None)
    num_bins = getattr(tokenizer, "num_bins", 256)
    results = []
    for index, spec in enumerate(specs):
        trajectory_tokens = responses[index].detach().cpu().long()
        if vocab_size is not None:
            trajectory_tokens = trajectory_tokens - (int(vocab_size) - int(num_bins))
        trajectory_actions = [
            _decode_env_actions(
                policy,
                tokenizer,
                step_tokens.reshape(int(step_tokens.numel() // 7), 7),
            )
            for step_tokens in trajectory_tokens
        ]
        results.append(
            _TrajectoryResult(
                stack="simplevla",
                task_id=int(spec.task_id),
                trial_id=int(spec.trial_id),
                success=bool(complete[index].item()),
                outer_steps=None,
                env_steps=int(finish_step[index].item()),
                first_tokens=trajectory_tokens[0].reshape(-1)[:32].tolist(),
                first_action=trajectory_actions[0]
                .detach()
                .cpu()
                .float()
                .reshape(-1)[:16]
                .tolist(),
                tokens=[step.reshape(-1).tolist() for step in trajectory_tokens],
                actions=[
                    step.detach().cpu().float().reshape(-1).tolist()
                    for step in trajectory_actions
                ],
            )
        )
    return results


def _run_simplevla_stack(
    args: argparse.Namespace,
    specs: list[_TrajectorySpec],
    policy,
) -> list[_TrajectoryResult]:
    _, RobHFRollout = _load_verl_rollout(args.simplevla_root)
    rollout = RobHFRollout(policy.model, _simplevla_cfg(args))
    results: list[_TrajectoryResult] = []
    batch_size = int(args.simplevla_batch_size)
    # SimpleVLA-RL calls LIBERO's ``get_task_init_states``, which still uses
    # ``torch.load(path)``.  Torch>=2.6 defaults ``weights_only=True`` and
    # rejects these trusted benchmark numpy arrays.  Use PyTorch's process-wide
    # compatibility knob so SimpleVLA's worker process also sees it.
    old_force_no_weights_only = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
    os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    torch.load = _torch_load_trusted_init_state
    try:
        for start in range(0, len(specs), batch_size):
            batch_specs = specs[start : start + batch_size]
            batch_results = _run_simplevla_batch(
                rollout,
                batch_specs,
                args.task_suite,
                policy,
                policy.action_tokenizer,
            )
            for result in batch_results:
                torchrl_logger.info(
                    "SimpleVLA rollout task=%s trial=%s success=%s env_steps=%s",
                    result.task_id,
                    result.trial_id,
                    result.success,
                    result.env_steps,
                )
            results.extend(batch_results)
    finally:
        torch.load = _ORIGINAL_TORCH_LOAD
        if old_force_no_weights_only is None:
            os.environ.pop("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", None)
        else:
            os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = old_force_no_weights_only
    return results


def _summary(
    stack: Literal["torchrl", "simplevla"],
    results: list[_TrajectoryResult],
) -> _StackSummary:
    successes = sum(int(result.success) for result in results)
    trajectories = len(results)
    return _StackSummary(
        stack=stack,
        trajectories=trajectories,
        successes=successes,
        success_rate=float(successes / trajectories) if trajectories else float("nan"),
        env_steps=sum(result.env_steps for result in results),
    )


def _first_mismatch(
    torchrl_values: list[int] | list[float] | None,
    simplevla_values: list[int] | list[float] | None,
    *,
    atol: float = 0.0,
) -> tuple[int, int | None]:
    if torchrl_values is None or simplevla_values is None:
        return 0, 0 if torchrl_values != simplevla_values else None
    compare_count = min(len(torchrl_values), len(simplevla_values))
    for index in range(compare_count):
        if abs(float(torchrl_values[index]) - float(simplevla_values[index])) > atol:
            return compare_count, index
    if len(torchrl_values) != len(simplevla_values):
        return compare_count, compare_count
    return compare_count, None


def _value_at(values: list[int] | None, index: int | None) -> int | None:
    if values is None or index is None or index >= len(values):
        return None
    return int(values[index])


def _max_abs_diff(
    torchrl_values: list[float] | None,
    simplevla_values: list[float] | None,
) -> float | None:
    if torchrl_values is None or simplevla_values is None:
        return None
    count = min(len(torchrl_values), len(simplevla_values))
    if count == 0:
        return None
    torchrl_array = np.asarray(torchrl_values[:count], dtype=np.float64)
    simplevla_array = np.asarray(simplevla_values[:count], dtype=np.float64)
    return float(np.abs(torchrl_array - simplevla_array).max())


def _parity_results(
    torchrl_results: list[_TrajectoryResult],
    simplevla_results: list[_TrajectoryResult],
    *,
    action_diff_atol: float,
) -> list[_ParityResult]:
    parity = []
    for torchrl_result, simplevla_result in zip(
        torchrl_results, simplevla_results, strict=True
    ):
        if (torchrl_result.task_id, torchrl_result.trial_id) != (
            simplevla_result.task_id,
            simplevla_result.trial_id,
        ):
            raise RuntimeError(
                "TorchRL and SimpleVLA results are not aligned: "
                f"{torchrl_result.task_id}/{torchrl_result.trial_id} vs "
                f"{simplevla_result.task_id}/{simplevla_result.trial_id}."
            )
        token_count, token_mismatch = _first_mismatch(
            torchrl_result.first_tokens,
            simplevla_result.first_tokens,
        )
        action_count, action_mismatch = _first_mismatch(
            torchrl_result.first_action,
            simplevla_result.first_action,
            atol=float(action_diff_atol),
        )
        first_token_exact_match = None
        if (
            torchrl_result.first_tokens is not None
            and simplevla_result.first_tokens is not None
        ):
            first_token_exact_match = (
                bool(torchrl_result.first_tokens)
                and bool(simplevla_result.first_tokens)
                and torchrl_result.first_tokens[0] == simplevla_result.first_tokens[0]
            )
        parity.append(
            _ParityResult(
                task_id=int(torchrl_result.task_id),
                trial_id=int(torchrl_result.trial_id),
                success_agreement=(
                    bool(torchrl_result.success) == bool(simplevla_result.success)
                ),
                torchrl_success=bool(torchrl_result.success),
                simplevla_success=bool(simplevla_result.success),
                first_token_exact_match=first_token_exact_match,
                first_tokens_exact_match=token_mismatch is None,
                first_token_compare_count=int(token_count),
                first_token_mismatch_index=token_mismatch,
                torchrl_first_token_at_mismatch=_value_at(
                    torchrl_result.first_tokens, token_mismatch
                ),
                simplevla_first_token_at_mismatch=_value_at(
                    simplevla_result.first_tokens, token_mismatch
                ),
                first_action_compare_count=int(action_count),
                first_action_max_abs_diff=_max_abs_diff(
                    torchrl_result.first_action,
                    simplevla_result.first_action,
                ),
                first_action_mismatch_index=action_mismatch,
            )
        )
    return parity


def _parity_summary(parity: list[_ParityResult]) -> _ParitySummary:
    trajectories = len(parity)
    success_agreements = sum(int(item.success_agreement) for item in parity)
    first_token_exact_matches = sum(
        int(bool(item.first_token_exact_match)) for item in parity
    )
    first_tokens_exact_matches = sum(
        int(bool(item.first_tokens_exact_match)) for item in parity
    )
    first_action_diffs = [
        item.first_action_max_abs_diff
        for item in parity
        if item.first_action_max_abs_diff is not None
    ]
    return _ParitySummary(
        trajectories=trajectories,
        success_agreements=success_agreements,
        success_agreement_rate=(
            float(success_agreements / trajectories) if trajectories else float("nan")
        ),
        first_token_exact_matches=first_token_exact_matches,
        first_token_exact_match_rate=(
            float(first_token_exact_matches / trajectories)
            if trajectories
            else float("nan")
        ),
        first_tokens_exact_matches=first_tokens_exact_matches,
        first_tokens_exact_match_rate=(
            float(first_tokens_exact_matches / trajectories)
            if trajectories
            else float("nan")
        ),
        first_action_max_abs_diff=(
            float(max(first_action_diffs)) if first_action_diffs else None
        ),
    )


def _first_mismatch_boundary(parity: list[_ParityResult]) -> dict[str, Any] | None:
    for item in parity:
        if item.first_token_mismatch_index is not None:
            return {
                "kind": "first_tokens",
                "task_id": item.task_id,
                "trial_id": item.trial_id,
                "index": item.first_token_mismatch_index,
                "torchrl": item.torchrl_first_token_at_mismatch,
                "simplevla": item.simplevla_first_token_at_mismatch,
            }
        if item.first_action_mismatch_index is not None:
            return {
                "kind": "first_action",
                "task_id": item.task_id,
                "trial_id": item.trial_id,
                "index": item.first_action_mismatch_index,
                "max_abs_diff": item.first_action_max_abs_diff,
            }
        if not item.success_agreement:
            return {
                "kind": "success",
                "task_id": item.task_id,
                "trial_id": item.trial_id,
                "torchrl": item.torchrl_success,
                "simplevla": item.simplevla_success,
            }
    return None


def _write_report(
    args: argparse.Namespace,
    specs: list[_TrajectorySpec],
    torchrl_results: list[_TrajectoryResult],
    simplevla_results: list[_TrajectoryResult],
) -> Path:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"simplevla_torchrl_rollout_compare_{stamp}.json"
    simplevla_root = Path(args.simplevla_root) if args.simplevla_root else None
    parity = _parity_results(
        torchrl_results,
        simplevla_results,
        action_diff_atol=float(args.action_diff_atol),
    )
    report = {
        "script": str(Path(__file__).resolve()),
        "created_utc": datetime.now(UTC).isoformat(),
        "args": vars(args),
        "torch": torch.__version__,
        "torchrl": getattr(__import__("torchrl"), "__version__", None),
        "simplevla_root": str(simplevla_root) if simplevla_root is not None else None,
        "simplevla_commit": (
            _git_commit(simplevla_root) if simplevla_root is not None else None
        ),
        "specs": [asdict(spec) for spec in specs],
        "summaries": {
            "torchrl": asdict(_summary("torchrl", torchrl_results)),
            "simplevla": asdict(_summary("simplevla", simplevla_results)),
            "parity": asdict(_parity_summary(parity)),
        },
        "first_mismatch_boundary": _first_mismatch_boundary(parity),
        "parity": [asdict(result) for result in parity],
        "trajectories": {
            "torchrl": [asdict(result) for result in torchrl_results],
            "simplevla": [asdict(result) for result in simplevla_results],
        },
    }
    out_path.write_text(json.dumps(_jsonable(report), indent=2), encoding="utf-8")
    torchrl_logger.info("Wrote rollout comparison report to %s", out_path)
    return out_path


def main() -> None:
    default_config = Path(__file__).parent / "config" / "vla_grpo_libero.yaml"
    parser = argparse.ArgumentParser(
        description="Compare TorchRL VLA and SimpleVLA-RL/VeRL LIBERO rollouts."
    )
    parser.add_argument("--config", default=str(default_config))
    parser.add_argument("--simplevla-root", default=os.environ.get("SIMPLEVLA_RL_ROOT"))
    parser.add_argument("--task-suite", default="libero_spatial")
    parser.add_argument("--task-ids", type=_parse_int_list, default=[0])
    parser.add_argument("--trial-ids", type=_parse_int_list, default=[0])
    parser.add_argument(
        "--checkpoint", default="Haozhan72/Openvla-oft-SFT-libero-spatial-traj1"
    )
    parser.add_argument(
        "--dataset-statistics",
        default="moojink/openvla-7b-oft-finetuned-libero-spatial",
    )
    parser.add_argument("--unnorm-key", default="libero_spatial_no_noops")
    parser.add_argument("--policy-device", default="cuda:0")
    parser.add_argument("--render-gpu", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument(
        "--sample-actions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use sampled token actions in both TorchRL and SimpleVLA rollouts.",
    )
    parser.add_argument(
        "--action-diff-atol",
        type=float,
        default=1e-6,
        help="Tolerance for reporting the first decoded-action mismatch.",
    )
    parser.add_argument("--gripper-binarize-threshold", type=float, default=0.0)
    parser.add_argument(
        "--gripper-invert",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument(
        "--center-crop",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--torchrl-image-backend",
        choices=["torchvision", "pil", "tensorflow"],
        default="torchvision",
    )
    parser.add_argument("--settle-steps", type=int, default=10)
    parser.add_argument("--chunk-size", type=int, default=8)
    parser.add_argument("--max-env-steps", type=int, default=512)
    parser.add_argument("--max-outer-steps", type=int, default=64)
    parser.add_argument("--max-prompt-length", type=int, default=512)
    parser.add_argument("--simplevla-batch-size", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", default="/tmp")
    parser.add_argument("--experiment-name", default="simplevla_torchrl_compare")
    args = parser.parse_args()

    if args.simplevla_root is None:
        raise ValueError(
            "Pass --simplevla-root or set SIMPLEVLA_RL_ROOT to the "
            "SimpleVLA-RL checkout."
        )
    _set_seed(int(args.seed))
    specs = _trajectory_specs(args.task_ids, args.trial_ids)
    _, policy, tokenizer = _policy_and_tokenizer(args)
    try:
        _set_seed(int(args.seed))
        torchrl_results = _run_torchrl_stack(args, specs, policy, tokenizer)
        _set_seed(int(args.seed))
        simplevla_results = _run_simplevla_stack(args, specs, policy)
    finally:
        del policy
        torch.cuda.empty_cache()
    report = _write_report(args, specs, torchrl_results, simplevla_results)
    parity = _parity_results(
        torchrl_results,
        simplevla_results,
        action_diff_atol=float(args.action_diff_atol),
    )
    summaries = {
        "torchrl": asdict(_summary("torchrl", torchrl_results)),
        "simplevla": asdict(_summary("simplevla", simplevla_results)),
        "parity": asdict(_parity_summary(parity)),
        "first_mismatch_boundary": _first_mismatch_boundary(parity),
        "report": str(report),
    }
    sys.stdout.write(json.dumps(_jsonable(summaries), indent=2) + "\n")


if __name__ == "__main__":
    main()
