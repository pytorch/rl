#!/usr/bin/env bash
# Reproduce the DreamerV3 DMC Walker Walk result: the preset's benchmark
# seeds and acceptance gate through benchmark.py. --fast enables the
# compiled RSSM scan; --smoke validates the eager pipeline with tiny settings
# in minutes. OUTPUT_DIR overrides the output directory, and KEY=VALUE
# arguments are forwarded to Hydra.
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
output_dir="${OUTPUT_DIR:-}"

usage() {
  echo "usage: reproduce_dmc_walker.sh [--fast | --smoke] [KEY=VALUE ...]"
}

smoke=0
fast=0
hydra_overrides=()
for arg in "$@"; do
  case "$arg" in
    --smoke) smoke=1 ;;
    --fast) fast=1 ;;
    -h|--help)
      usage
      exit 0
      ;;
    *=*) hydra_overrides+=("$arg") ;;
    *)
      usage >&2
      exit 2
      ;;
  esac
done

if [ "$smoke" -eq 1 ] && [ "$fast" -eq 1 ]; then
  echo "--fast and --smoke are mutually exclusive." >&2
  usage >&2
  exit 2
fi

if [ -z "$output_dir" ]; then
  if [ "$smoke" -eq 1 ]; then
    output_dir="dmc_walker_smoke"
  else
    output_dir="dmc_walker_runs"
  fi
fi

benchmark_command=(python "$script_dir/benchmark.py" --output-dir "$output_dir")

if [ "$smoke" -eq 1 ]; then
  benchmark_command+=(
    benchmark.window_size=200
    benchmark.minimum_final_median_return=-100000
    collector.num_envs=1
    collector.count_reset_records=false
    collector.total_frames=400
    collector.frames_per_batch=200
    env.max_episode_steps=100
    replay_buffer.batch_size=2
    replay_buffer.buffer_size=400
    replay_buffer.seq_len=4
    replay_buffer.warmup_factor=1
    optimization.compile_rssm=null
    optimization.updates_per_batch=1
    optimization.train_ratio=null
    optimization.mixed_precision=false
    logger.eval_every=200
    logger.eval_episodes=1
    networks.hidden_dim=8
    networks.encoder_layers=1
    networks.decoder_layers=1
    networks.reward_layers=1
    networks.actor_layers=1
    networks.value_layers=1
    networks.num_categoricals=2
    networks.num_classes=2
    networks.num_reward_bins=11
    networks.num_value_bins=11
    networks.rnn_hidden_dim=8
  )
elif [ "$fast" -eq 1 ]; then
  benchmark_command+=(
    optimization.compile_rssm=scan
    optimization.rssm_scan_unroll=8
  )
fi

if [ "${#hydra_overrides[@]}" -gt 0 ]; then
  benchmark_command+=("${hydra_overrides[@]}")
fi
"${benchmark_command[@]}"
