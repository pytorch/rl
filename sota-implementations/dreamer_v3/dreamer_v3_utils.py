# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Run logging, RNG streams and evaluation of the DreamerV3 example."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import NamedTuple

import numpy as np
import torch
from omegaconf import DictConfig
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl._utils import logger as torchrl_logger
from torchrl.envs import EnvBase
from torchrl.envs.utils import ExplorationType, set_exploration_type

_has_matplotlib = importlib.util.find_spec("matplotlib") is not None


# --- RNG streams -------------------------------------------------------------


POLICY_RNG_STREAM = 0
LEARNER_RNG_STREAM = 1
REPLAY_RNG_STREAM = 2


def stream_seed(seed: int, counter: int, stream: int) -> int:
    """Make one deterministic Torch seed from a seed, a counter and a stream.

    ``stream`` keeps its users independent: a change in the number of draws of
    one stream does not change the sequences of the others.
    """
    rng = np.random.default_rng(seed=[seed, counter, stream])
    words = rng.integers(0, np.iinfo(np.uint32).max, (2,), np.uint32)
    return (int(words[0]) << 32) | int(words[1])


# --- Run logging and episode bookkeeping -------------------------------------


def append_jsonl(path: Path | None, record: dict[str, object]) -> None:
    if path is None:
        return
    with path.open("a") as file:
        file.write(json.dumps(record) + "\n")


def latent_state_dim(cfg: DictConfig) -> int:
    return cfg.networks.num_categoricals * cfg.networks.num_classes


class CompileSettings(NamedTuple):
    """The resolved compile decisions of one run."""

    strategy: str
    train_step: bool
    rssm: str | None
    scan_unroll: int
    cudagraph: bool
    mode: str

    @property
    def enabled(self) -> bool:
        return bool(self.train_step or self.rssm or self.cudagraph)


def resolve_compile_settings(cfg: DictConfig, device: torch.device) -> CompileSettings:
    """Turn ``optimization.compile`` and its overrides into concrete decisions.

    ``auto`` takes the fastest supported path for ``device``: on CUDA the
    complete learner step is compiled around a higher-order scan of the
    recurrence and captured in a CUDA graph, elsewhere everything runs eagerly.
    ``off`` disables all of it. ``compile_train_step``, ``compile_rssm`` and
    ``cudagraph_train_step`` default to ``null``, which follows the strategy;
    an explicit value overrides that single decision.
    """
    strategy = cfg.optimization.get("compile", "auto")
    if strategy not in ("auto", "off"):
        raise ValueError(
            f"optimization.compile must be 'auto' or 'off', got {strategy!r}."
        )
    fast = strategy == "auto" and device.type == "cuda"
    train_step = cfg.optimization.compile_train_step
    train_step = fast if train_step is None else bool(train_step)
    rssm = cfg.optimization.compile_rssm
    if rssm is None and fast:
        rssm = "scan"
    if rssm not in (None, "step", "scan"):
        raise ValueError(
            f"optimization.compile_rssm must be null, 'step' or 'scan', got {rssm!r}."
        )
    cudagraph = cfg.optimization.cudagraph_train_step
    cudagraph = fast if cudagraph is None else bool(cudagraph)
    if cudagraph and device.type != "cuda":
        raise ValueError(
            "optimization.cudagraph_train_step requires a CUDA training device."
        )
    return CompileSettings(
        strategy=strategy,
        train_step=train_step,
        rssm=rssm,
        scan_unroll=cfg.optimization.rssm_scan_unroll,
        cudagraph=cudagraph,
        mode=cfg.optimization.compile_train_step_mode,
    )


def training_episode_returns(
    data: TensorDictBase,
    running_return: torch.Tensor,
    num_envs: int,
) -> list[tuple[int, int, float]]:
    reward = data.get(("next", "reward")).squeeze(-1)
    done = data.get(("next", "done")).squeeze(-1)
    env_index = data.get("env_index", default=None)
    if env_index is not None:
        completed = []
        for position, (env, step_reward, step_done) in enumerate(
            zip(
                env_index.reshape(-1).cpu(),
                reward.reshape(-1).cpu(),
                done.reshape(-1).cpu(),
            )
        ):
            env = int(env)
            running_return[env].add_(step_reward)
            if step_done:
                completed.append((position, env, float(running_return[env])))
                running_return[env] = 0
        return completed
    if num_envs == 1:
        reward = reward.reshape(1, -1)
        done = done.reshape(1, -1)
    completed = []
    for time_index in range(reward.shape[-1]):
        running_return.add_(reward[..., time_index].cpu())
        finished = done[..., time_index].cpu()
        completed.extend(
            (time_index, int(env_index), float(running_return[env_index]))
            for env_index in finished.nonzero().flatten()
        )
        running_return.masked_fill_(finished, 0)
    return completed


# --- Evaluation and plotting -------------------------------------------------


@torch.no_grad()
def eval_episode_reward(
    env: EnvBase,
    actor: TensorDictModuleBase,
    num_episodes: int,
    max_episode_steps: int,
) -> torch.Tensor:
    totals = []
    with set_exploration_type(ExplorationType.DETERMINISTIC):
        for _ in range(num_episodes):
            td = env.rollout(
                max_steps=max_episode_steps,
                policy=actor,
                break_when_any_done=True,
                auto_cast_to_device=True,
            )
            totals.append(td.get(("next", "reward")).sum())
    return torch.stack(totals).mean()


def plot_enabled(cfg: DictConfig) -> bool:
    """Return True if the run must record the per-update losses."""
    return bool(cfg.logger.output_plot) and _has_matplotlib


def save_run_plot(
    cfg: DictConfig,
    eval_steps: list[int],
    eval_returns: list[torch.Tensor],
    loss_history: list[torch.Tensor],
) -> None:
    if not _has_matplotlib:
        torchrl_logger.warning(
            "matplotlib is not installed; skipping plot %s", cfg.logger.output_plot
        )
        return
    import matplotlib.pyplot as plt  # noqa: PLC0415

    returns = (
        (torch.stack(eval_returns) if eval_returns else torch.empty(0)).cpu().numpy()
    )
    losses = (torch.cat(loss_history) if loss_history else torch.empty(0, 6)).numpy()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(eval_steps, returns, marker="o")
    axes[0].set_title(f"{cfg.env.name} eval reward (real env)")
    axes[0].set_xlabel("env_step")
    axes[0].set_ylabel("avg episode return")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(losses[:, 1], label="reco", alpha=0.8)
    axes[1].plot(losses[:, 2], label="reward", alpha=0.8)
    axes[1].plot(losses[:, 0], label="kl", alpha=0.8)
    axes[1].set_title("World-model losses (update step)")
    axes[1].set_xlabel("update step")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"DreamerV3 on {cfg.env.name} - {cfg.collector.total_frames} env steps"
    )
    fig.tight_layout()
    fig.savefig(cfg.logger.output_plot, dpi=120)
    torchrl_logger.info("Saved plot to %s", cfg.logger.output_plot)
