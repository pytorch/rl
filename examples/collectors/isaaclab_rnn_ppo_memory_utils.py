# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Utilities for the Isaac Lab recurrent PPO example.

Isaac Lab is only imported lazily inside :func:`make_env`, so the
top-level main script can run without ``isaaclab`` on the Python path
and only the worker subprocesses (where ``make_env`` is called) pay the
import cost.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Any, Literal, Sequence

import torch
import torch.nn as nn
from tensordict import TensorDictBase
from tensordict.nn import (
    AddStateIndependentNormalScale,
    TensorDictModule,
    TensorDictSequential,
)
from torchrl.envs import ExplorationType, RewardSum, TransformedEnv
from torchrl.modules import (
    LSTMModule,
    MLP,
    ProbabilisticActor,
    TanhNormal,
    ValueOperator,
)

RnnBackend = Literal["cudnn", "pad", "scan", "triton"]


def _to_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, TensorDictBase):
        return value.detach().to("cpu")
    if isinstance(value, dict):
        return {key: _to_cpu(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_cpu(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_to_cpu(item) for item in value)
    return value


def _finite_stats(prefix: str, value: torch.Tensor) -> dict[str, float | int | bool]:
    value = value.detach()
    finite = torch.isfinite(value)
    stats: dict[str, float | int | bool] = {
        f"{prefix}/numel": value.numel(),
        f"{prefix}/finite": bool(finite.all().cpu()),
        f"{prefix}/nonfinite": int((~finite).sum().cpu()),
    }
    if finite.any():
        finite_value = value[finite].float()
        stats.update(
            {
                f"{prefix}/mean": float(finite_value.mean().cpu()),
                f"{prefix}/std": float(finite_value.std(unbiased=False).cpu()),
                f"{prefix}/min": float(finite_value.min().cpu()),
                f"{prefix}/max": float(finite_value.max().cpu()),
            }
        )
    return stats


class NonFiniteChecker:
    """Check tensors for non-finite values and save a replay bundle on failure."""

    def __init__(
        self,
        *,
        enabled: bool,
        save_dir: Path | None,
        run_args: dict[str, Any] | None,
        actor: nn.Module | None = None,
        critic: nn.Module | None = None,
        optimizer: torch.optim.Optimizer | None = None,
    ) -> None:
        self.enabled = enabled
        self.save_dir = save_dir
        self.run_args = run_args
        self.actor = actor
        self.critic = critic
        self.optimizer = optimizer

    @staticmethod
    def _filename(phase: str, name: str, context: dict[str, int]) -> str:
        raw = (
            f"nonfinite_{phase}_{name}"
            f"_i{context['iteration']}_e{context['epoch']}"
            f"_mb{context['minibatch']}_u{context['update']}.pt"
        )
        return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)

    @staticmethod
    def _first_bad_index(value: torch.Tensor) -> tuple[int, ...] | None:
        bad = (~torch.isfinite(value)).nonzero()
        if bad.numel() == 0:
            return None
        return tuple(int(item) for item in bad[0].cpu())

    def _state_dict(self, module: nn.Module | None) -> dict[str, Any] | None:
        if module is None:
            return None
        return {key: _to_cpu(val) for key, val in module.state_dict().items()}

    def _grads(self, module: nn.Module | None) -> dict[str, Any] | None:
        if module is None:
            return None
        return {
            key: _to_cpu(param.grad)
            for key, param in module.named_parameters()
            if param.grad is not None
        }

    def check_tensor(
        self,
        phase: str,
        name: str,
        value: torch.Tensor,
        *,
        context: dict[str, int],
        batch: TensorDictBase | None = None,
        loss: TensorDictBase | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled or torch.isfinite(value).all():
            return
        stats = _finite_stats(f"debug_nonfinite/{phase}/{name}", value)
        first_bad_index = self._first_bad_index(value)
        saved_path = None
        if self.save_dir is not None:
            self.save_dir.mkdir(parents=True, exist_ok=True)
            saved_path = self.save_dir / self._filename(phase, name, context)
            bundle = {
                "phase": phase,
                "name": name,
                "args": self.run_args,
                "context": dict(context),
                "stats": stats,
                "first_bad_index": first_bad_index,
                "offending_value": _to_cpu(value),
                "batch": _to_cpu(batch) if batch is not None else None,
                "actor_state_dict": self._state_dict(self.actor),
                "critic_state_dict": self._state_dict(self.critic),
                "optimizer_state_dict": (
                    _to_cpu(self.optimizer.state_dict())
                    if self.optimizer is not None
                    else None
                ),
                "loss": _to_cpu(loss) if loss is not None else None,
                "actor_grads": self._grads(self.actor),
                "critic_grads": self._grads(self.critic),
                "extra": extra,
                "torch_rng_state": torch.random.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all()
                    if torch.cuda.is_available()
                    else None
                ),
            }
            torch.save(bundle, saved_path)
        raise RuntimeError(
            f"Non-finite tensor during {phase}: {name}; "
            f"context={context}; first_bad_index={first_bad_index}; "
            f"stats={stats}; saved_path={saved_path}"
        )

    def check_tensordict(
        self,
        phase: str,
        tensordict: TensorDictBase,
        *,
        context: dict[str, int],
        batch: TensorDictBase | None = None,
        loss: TensorDictBase | None = None,
        extra: dict[str, Any] | None = None,
    ) -> None:
        if not self.enabled:
            return
        batch = tensordict if batch is None else batch
        for key, value in tensordict.items(include_nested=True, leaves_only=True):
            if not isinstance(value, torch.Tensor) or not value.is_floating_point():
                continue
            self.check_tensor(
                phase,
                str(key),
                value,
                context=context,
                batch=batch,
                loss=loss,
                extra=extra,
            )

    def check_gradients(
        self,
        phase: str,
        named_modules: Sequence[tuple[str, nn.Module]],
        *,
        context: dict[str, int],
        batch: TensorDictBase | None = None,
        loss: TensorDictBase | None = None,
    ) -> None:
        if not self.enabled:
            return
        for module_name, module in named_modules:
            for name, parameter in module.named_parameters():
                if parameter.grad is None:
                    continue
                self.check_tensor(
                    phase,
                    f"{module_name}.{name}",
                    parameter.grad,
                    context=context,
                    batch=batch,
                    loss=loss,
                )


def _init_isaac_app(device: str | None = None) -> None:
    """Start Isaac Lab's AppLauncher in headless mode inside a worker."""
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="TorchRL Isaac Lab env launcher.")
    AppLauncher.add_app_launcher_args(parser)
    launch_args = ["--headless"]
    if device is not None:
        launch_args.extend(["--device", device])
    args_cli, _ = parser.parse_known_args(launch_args)
    AppLauncher(args_cli)


def make_env(task: str, num_envs: int, max_episode_steps: int, device: str):
    """Build an Isaac Lab env. Imports ``isaaclab`` lazily (worker-only)."""
    import gymnasium as gym
    import isaaclab_tasks  # noqa: F401
    from isaaclab_tasks.manager_based.classic.ant.ant_env_cfg import AntEnvCfg
    from torchrl.envs.libs.isaac_lab import IsaacLabWrapper

    if task == "Isaac-Ant-v0":
        cfg = AntEnvCfg()
        if hasattr(cfg, "scene") and hasattr(cfg.scene, "num_envs"):
            cfg.scene.num_envs = num_envs
        if hasattr(cfg, "sim") and hasattr(cfg.sim, "device"):
            cfg.sim.device = device
        if hasattr(cfg, "device"):
            cfg.device = device
        if (
            hasattr(cfg, "episode_length_s")
            and hasattr(cfg, "sim")
            and hasattr(cfg.sim, "dt")
        ):
            cfg.episode_length_s = max_episode_steps * cfg.sim.dt
        env = gym.make(task, cfg=cfg)
    else:
        env = gym.make(task)
    return TransformedEnv(
        IsaacLabWrapper(env, device=torch.device(device)),
        RewardSum(),
    )


def make_models(
    *,
    obs_dim: int,
    action_dim: int,
    hidden_size: int,
    rnn_backend: RnnBackend,
    device: torch.device,
) -> tuple[ProbabilisticActor, ValueOperator, TensorDictSequential]:
    """Build the actor, critic head, and the full value module sharing the backbone.

    The actor is a :class:`~torchrl.modules.ProbabilisticActor` wrapping a
    shared embed + LSTM backbone and a per-action Gaussian head. The value
    module reuses the same backbone (so a single GAE pass amortises the
    recurrent compute across the policy and the value head).
    """
    embed = TensorDictModule(
        nn.Linear(obs_dim, hidden_size, device=device),
        in_keys=["policy"],
        out_keys=["embed"],
    )
    lstm_backend = "pad" if rnn_backend == "cudnn" else rnn_backend
    lstm = LSTMModule(
        input_size=hidden_size,
        hidden_size=hidden_size,
        num_layers=1,
        in_keys=["embed", "recurrent_state_h", "recurrent_state_c"],
        out_keys=[
            "lstm_out",
            ("next", "recurrent_state_h"),
            ("next", "recurrent_state_c"),
        ],
        recurrent_backend=lstm_backend,
        device=device,
    )
    backbone = TensorDictSequential(embed, lstm)

    actor_head = TensorDictModule(
        nn.Sequential(
            MLP(
                in_features=hidden_size,
                out_features=action_dim,
                num_cells=[],
                activation_class=nn.Tanh,
                device=device,
            ),
            AddStateIndependentNormalScale(action_dim, scale_lb=1e-4).to(device),
        ),
        in_keys=["lstm_out"],
        out_keys=["loc", "scale"],
    )
    actor = ProbabilisticActor(
        module=TensorDictSequential(backbone, actor_head),
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": -torch.ones(action_dim, device=device),
            "high": torch.ones(action_dim, device=device),
            "tanh_loc": False,
        },
        return_log_prob=True,
        default_interaction_type=ExplorationType.RANDOM,
    )

    # The value module reuses the backbone (shared params, single LSTM call
    # per GAE/update pass). An identity TDM caches the lstm_out under a
    # distinct key so the critic head and the actor head don't fight over
    # write semantics on a shared key.
    value_feature = TensorDictModule(
        nn.Identity(), in_keys=["lstm_out"], out_keys=["value_lstm_out"]
    )
    critic = ValueOperator(
        nn.Linear(hidden_size, 1, device=device),
        in_keys=["value_lstm_out"],
    ).to(device)
    full_value = TensorDictSequential(backbone, value_feature, critic)
    return actor, critic, full_value
