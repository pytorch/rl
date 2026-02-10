#!/usr/bin/env bash
# run_sweep_sequential.sh — Run 4 Dreamer sweep experiments sequentially
# This script runs ON THE CLUSTER NODE inside the container.
set -euo pipefail

export WANDB_API_KEY="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"

cd /root/rl

echo "============================================================"
echo " SWEEP: rl0 — Higher WM LR (2e-3)"
echo "============================================================"
bash setup-and-run.sh \
    optimization.world_model_lr=2e-3 \
    logger.exp_name=rl0-wmlr2e3

echo "============================================================"
echo " SWEEP: rl1 — Grayscale + WM LR 1e-3"
echo "============================================================"
bash setup-and-run.sh \
    env.grayscale=True \
    optimization.world_model_lr=1e-3 \
    logger.exp_name=rl1-gray-wmlr1e3

echo "============================================================"
echo " SWEEP: rl2 — Wider CNN (48ch) + Larger RSSM (60/600)"
echo "============================================================"
bash setup-and-run.sh \
    networks.encoder_channels=48 \
    networks.state_dim=60 \
    networks.rssm_hidden_dim=600 \
    logger.exp_name=rl2-wide48-rssm60x600

echo "============================================================"
echo " SWEEP: rl3 — Grayscale + Wider CNN + Larger RSSM + Higher LR"
echo "============================================================"
bash setup-and-run.sh \
    env.grayscale=True \
    networks.encoder_channels=48 \
    networks.state_dim=45 \
    networks.rssm_hidden_dim=400 \
    optimization.world_model_lr=1.5e-3 \
    logger.exp_name=rl3-gray-wide48-rssm45x400-lr15e4

echo "============================================================"
echo " SWEEP COMPLETE — all 4 experiments finished"
echo "============================================================"
