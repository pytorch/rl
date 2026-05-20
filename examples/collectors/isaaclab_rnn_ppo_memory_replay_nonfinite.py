# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Replay a saved non-finite PPO update from ``isaaclab_rnn_ppo_memory.py``.

The training script's ``--debug-nonfinite-update`` mode writes a ``.pt`` bundle
at the first non-finite loss or gradient. This utility reloads that bundle,
rebuilds the actor / critic / value modules, reruns the same PPO loss and
backward pass, and prints finite stats for each output and gradient.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from isaaclab_rnn_ppo_memory_utils import make_models
from torchrl.modules import set_recurrent_mode
from torchrl.objectives import ClipPPOLoss


def _finite_stats(name: str, value: torch.Tensor) -> dict[str, float | int | bool]:
    value = value.detach()
    finite = torch.isfinite(value)
    stats: dict[str, float | int | bool] = {
        "name": name,
        "shape": tuple(value.shape),
        "numel": value.numel(),
        "finite": bool(finite.all().cpu()),
        "nonfinite": int((~finite).sum().cpu()),
    }
    if finite.any():
        finite_value = value[finite].float()
        stats.update(
            {
                "mean": float(finite_value.mean().cpu()),
                "std": float(finite_value.std(unbiased=False).cpu()),
                "min": float(finite_value.min().cpu()),
                "max": float(finite_value.max().cpu()),
            }
        )
    return stats


def _grad_diff_stats(
    name: str, replay: torch.Tensor, saved: torch.Tensor
) -> dict[str, float | str]:
    diff = replay.detach().cpu() - saved.detach().cpu()
    saved_norm = saved.detach().float().norm().item()
    return {
        "name": name,
        "max_abs": float(diff.abs().max()),
        "rel_norm": float(diff.float().norm() / max(saved_norm, 1e-12)),
        "saved_norm": float(saved_norm),
        "replay_norm": float(replay.detach().float().norm().cpu()),
    }


def _print_stats(header: str, stats: dict) -> None:
    print(header, stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("bundle", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--rnn-backend",
        choices=["cudnn", "pad", "scan", "triton"],
        default=None,
        help="Override the backend saved in the bundle args.",
    )
    return parser.parse_args()


def main() -> None:
    cli_args = parse_args()
    device = torch.device(cli_args.device)
    bundle = torch.load(cli_args.bundle, map_location="cpu", weights_only=False)
    run_args = bundle["args"]
    rnn_backend = cli_args.rnn_backend or run_args["rnn_backend"]

    print("bundle", cli_args.bundle)
    print("saved_phase", bundle["phase"])
    print("saved_name", bundle["name"])
    print("saved_context", bundle["context"])
    print("saved_stats", bundle["stats"])
    print("replay_backend", rnn_backend)
    print("device", device)

    actor, critic, full_value = make_models(
        obs_dim=run_args["obs_dim"],
        action_dim=run_args["action_dim"],
        hidden_size=run_args["hidden_size"],
        rnn_backend=rnn_backend,
        device=device,
    )
    actor.load_state_dict(bundle["actor_state_dict"])
    critic.load_state_dict(bundle["critic_state_dict"])
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=full_value,
        clip_epsilon=run_args["clip_epsilon"],
        entropy_coeff=run_args["entropy_coeff"],
        critic_coeff=run_args["critic_coeff"],
        normalize_advantage=True,
    )

    batch = bundle["batch"].to(device)
    actor.zero_grad(set_to_none=True)
    critic.zero_grad(set_to_none=True)
    with set_recurrent_mode(True):
        loss = loss_module(batch)
    total = loss["loss_objective"] + loss["loss_entropy"] + loss["loss_critic"]
    loss.set("loss_total", total.detach())

    for key, value in loss.items():
        _print_stats(f"loss {key}", _finite_stats(str(key), value))
    total.backward()

    saved_actor_grads = bundle["actor_grads"] or {}
    saved_critic_grads = bundle["critic_grads"] or {}
    for prefix, module, saved_grads in [
        ("actor", actor, saved_actor_grads),
        ("critic", critic, saved_critic_grads),
    ]:
        for name, parameter in module.named_parameters():
            if parameter.grad is None:
                continue
            stat_name = f"{prefix}.{name}"
            _print_stats("grad", _finite_stats(stat_name, parameter.grad))
            if name in saved_grads:
                _print_stats(
                    "grad_diff",
                    _grad_diff_stats(stat_name, parameter.grad, saved_grads[name]),
                )


if __name__ == "__main__":
    main()
