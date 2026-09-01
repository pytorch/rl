#!/usr/bin/env bash
# Reproduce the DreamerV3 DMC Walker Walk result: the preset's benchmark
# seeds and acceptance gate through benchmark.py. --smoke validates the
# same pipeline with tiny settings in minutes; OUTPUT_DIR overrides the
# output directory.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${OUTPUT_DIR:-dmc_walker_runs}"

smoke=0
for arg in "$@"; do
  case "$arg" in
    --smoke) smoke=1 ;;
    *)
      echo "usage: reproduce_dmc_walker.sh [--smoke]" >&2
      exit 2
      ;;
  esac
done

if [ "$smoke" -eq 1 ]; then
  python "$script_dir/benchmark.py" \
    --output-dir "$output_dir" \
    benchmark.window_size=200 \
    benchmark.minimum_final_median_return=-100000 \
    collector.num_envs=1 \
    collector.count_reset_records=false \
    collector.total_frames=400 \
    collector.frames_per_batch=200 \
    env.max_episode_steps=100 \
    replay_buffer.batch_size=2 \
    replay_buffer.seq_len=4 \
    replay_buffer.warmup_factor=1 \
    optimization.updates_per_batch=1 \
    optimization.train_ratio=8 \
    optimization.mixed_precision=false \
    logger.eval_every=200 \
    logger.eval_episodes=1 \
    networks.hidden_dim=8 \
    networks.encoder_layers=1 \
    networks.decoder_layers=1 \
    networks.reward_layers=1 \
    networks.actor_layers=1 \
    networks.value_layers=1 \
    networks.num_categoricals=2 \
    networks.num_classes=2 \
    networks.num_reward_bins=11 \
    networks.num_value_bins=11 \
    networks.rnn_hidden_dim=8
else
  python "$script_dir/benchmark.py" --output-dir "$output_dir"
fi
