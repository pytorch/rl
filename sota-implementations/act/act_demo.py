# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""ACT demo: learn to imitate a circular trajectory.

Synthetic task
--------------
Observation : [cos θ, sin θ]  — current angle on a unit circle
Action chunk : next CHUNK_SIZE (cos, sin) positions  — where to go

The expert policy is deterministic: step forward by dt each tick.
ACT's CVAE should learn to encode this and reproduce it.

Produces ``act_demo.png`` with three panels:
  1. Training losses (total, reconstruction, KL)
  2. Predicted vs ground-truth action chunk for a held-out obs
  3. Latent-space scatter: z samples coloured by θ

Usage
-----
    python sota-implementations/act/act_demo.py
"""
from __future__ import annotations

import os
import sys

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib.pyplot as plt
import numpy as np
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule

from torchrl.modules.models import ACTModel
from torchrl.objectives import ACTLoss

# ── Config ──────────────────────────────────────────────────────────────────
CHUNK_SIZE = 20       # predict 20 future steps at once
OBS_DIM    = 2        # (cos θ, sin θ)
ACTION_DIM = 2        # (cos θ, sin θ) — same space
DT         = 0.15     # radians per expert step
BATCH_SIZE = 128
N_STEPS    = 600
LR         = 3e-4
KL_WEIGHT  = 1.0
SEED       = 42
OUT_PATH   = os.path.join(os.path.dirname(__file__), "act_demo.png")


def make_batch(batch_size: int, dt: float = DT) -> TensorDict:
    """Sample random angles → build (obs, action_chunk) pairs."""
    theta = torch.rand(batch_size) * 2 * torch.pi
    obs = torch.stack([theta.cos(), theta.sin()], dim=-1)  # (B, 2)

    steps = torch.arange(1, CHUNK_SIZE + 1, dtype=torch.float32)  # (K,)
    future_theta = theta[:, None] + steps[None, :] * dt            # (B, K)
    action_chunk = torch.stack(
        [future_theta.cos(), future_theta.sin()], dim=-1
    )  # (B, K, 2)

    return TensorDict(
        {"observation": obs, "action_chunk": action_chunk},
        batch_size=[batch_size],
    )


def main() -> None:
    torch.manual_seed(SEED)

    model = ACTModel(
        obs_dim=OBS_DIM,
        action_dim=ACTION_DIM,
        chunk_size=CHUNK_SIZE,
        hidden_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=3,
        latent_dim=16,
    )
    actor = TensorDictModule(
        model,
        in_keys=["observation", "action_chunk"],
        out_keys=["action_pred", "mu", "log_var"],
    )
    loss_fn = ACTLoss(actor, kl_weight=KL_WEIGHT, reduction="mean")
    optimizer = torch.optim.Adam(loss_fn.parameters(), lr=LR)

    hist_total, hist_reco, hist_kl = [], [], []

    print("Training ACT on circular trajectory imitation …")
    for step in range(N_STEPS):
        td = make_batch(BATCH_SIZE)
        loss_td = loss_fn(td)

        optimizer.zero_grad(set_to_none=True)
        loss_td["loss_act"].backward()
        torch.nn.utils.clip_grad_norm_(loss_fn.parameters(), 1.0)
        optimizer.step()

        hist_total.append(loss_td["loss_act"].item())
        hist_reco.append(loss_td["loss_reconstruction"].item())
        hist_kl.append(loss_td["loss_kl"].item())

        if (step + 1) % 100 == 0:
            print(
                f"  step {step+1:4d}  total={hist_total[-1]:.4f}  "
                f"reco={hist_reco[-1]:.4f}  kl={hist_kl[-1]:.4f}"
            )

    # ── Evaluation ──────────────────────────────────────────────────────────
    model.eval()
    theta_test = torch.tensor([0.0, torch.pi / 4, torch.pi / 2, torch.pi])
    obs_test   = torch.stack([theta_test.cos(), theta_test.sin()], dim=-1)
    steps_f    = torch.arange(1, CHUNK_SIZE + 1, dtype=torch.float32)
    gt_chunks  = torch.stack(
        [
            (theta_test[:, None] + steps_f[None, :] * DT).cos(),
            (theta_test[:, None] + steps_f[None, :] * DT).sin(),
        ],
        dim=-1,
    )  # (4, K, 2)

    with torch.no_grad():
        # Inference: no action_chunk → z = 0 (prior mean)
        td_eval = TensorDict({"observation": obs_test}, batch_size=[4])
        actor(td_eval)
        pred_chunks = td_eval["action_pred"]  # (4, K, 2)

        # Latent encoding for scatter
        z_samples, labels = [], []
        for _ in range(200):
            td_s = make_batch(32)
            actor(td_s)
            z_samples.append(td_s["mu"].numpy())
            # angle label = atan2(sin θ, cos θ)
            labels.append(
                torch.atan2(td_s["observation"][:, 1], td_s["observation"][:, 0]).numpy()
            )
        z_all = np.concatenate(z_samples)[:, :2]  # first 2 latent dims
        l_all = np.concatenate(labels)

    # ── Plot ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(14, 4))
    colours = ["#e41a1c", "#377eb8", "#4daf4a", "#984ea3"]
    labels_test = [f"θ=0", "θ=π/4", "θ=π/2", "θ=π"]

    # Panel 1: loss curves
    ax1 = fig.add_subplot(1, 3, 1)
    xs = range(1, N_STEPS + 1)
    ax1.plot(xs, hist_total, label="total", lw=1.5)
    ax1.plot(xs, hist_reco,  label="reconstruction", lw=1.2, alpha=0.8)
    ax1.plot(xs, hist_kl,    label="KL", lw=1.2, alpha=0.8)
    ax1.set_title("Training losses")
    ax1.set_xlabel("step")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # Panel 2: predicted vs GT action chunk (cos component)
    ax2 = fig.add_subplot(1, 3, 2)
    k = np.arange(1, CHUNK_SIZE + 1)
    for i, (lbl, col) in enumerate(zip(labels_test, colours)):
        ax2.plot(k, gt_chunks[i, :, 0].numpy(),   color=col, lw=2,   label=f"GT {lbl}")
        ax2.plot(k, pred_chunks[i, :, 0].numpy(), color=col, lw=1.5, linestyle="--", label=f"Pred {lbl}")
    ax2.set_title("Predicted vs GT (cos component)")
    ax2.set_xlabel("chunk step")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    # Panel 3: latent scatter coloured by θ
    ax3 = fig.add_subplot(1, 3, 3)
    sc = ax3.scatter(z_all[:, 0], z_all[:, 1], c=l_all, cmap="hsv", s=8, alpha=0.6)
    plt.colorbar(sc, ax=ax3, label="θ (radians)")
    ax3.set_title("Latent z (dims 0-1) coloured by θ")
    ax3.set_xlabel("z₀")
    ax3.set_ylabel("z₁")
    ax3.grid(True, alpha=0.3)

    fig.suptitle(
        f"ACT — circular trajectory imitation  "
        f"(chunk={CHUNK_SIZE}, kl_weight={KL_WEIGHT}, {N_STEPS} steps)",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(OUT_PATH, dpi=130)
    print(f"\nSaved → {OUT_PATH}")


if __name__ == "__main__":
    main()
