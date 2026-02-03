#!/bin/bash
# A/B Test Script for torch.compile on Dreamer actor/value losses
#
# Usage:
#   # Run baseline (no compile)
#   ./scripts/profile_compile_ab.sh baseline
#
#   # Run with compile enabled
#   ./scripts/profile_compile_ab.sh compiled
#
# This script runs Dreamer with distributed profiling to measure the impact
# of torch.compile on actor and value loss modules.

set -e

MODE=${1:-"baseline"}
TRACE_DIR="/root/traces_${MODE}"

echo "=========================================="
echo "Dreamer Compile A/B Test: $MODE"
echo "=========================================="

mkdir -p "$TRACE_DIR"

if [ "$MODE" = "baseline" ]; then
    COMPILE_ENABLED="false"
    echo "Running WITHOUT torch.compile (baseline)"
elif [ "$MODE" = "compiled" ]; then
    COMPILE_ENABLED="true"
    echo "Running WITH torch.compile (actor, value losses)"
else
    echo "Unknown mode: $MODE"
    echo "Usage: $0 [baseline|compiled]"
    exit 1
fi

echo ""
echo "Configuration:"
echo "  optimization.compile.enabled=$COMPILE_ENABLED"
echo "  optimization.compile.losses=[actor,value]"
echo "  PROF_ITERATIONS=50-55"
echo "  Output: $TRACE_DIR"
echo ""

# Run dreamer with distributed profiling via prof
# Use WANDB_MODE=disabled to avoid login requirement for profiling
# MUJOCO_GL=egl for headless rendering on cluster
MUJOCO_GL=egl \
PROF_ENABLED=1 \
PROF_ITERATIONS=50-55 \
PROF_OUTPUT_DIR="$TRACE_DIR" \
PROF_MODE=lite \
WANDB_MODE=disabled \
python sota-implementations/dreamer/dreamer.py \
  optimization.autocast=true \
  optimization.compile.enabled="$COMPILE_ENABLED" \
  'optimization.compile.losses=[actor,value]' \
  profiling.enabled=true \
  profiling.distributed.enabled=true \
  profiling.collector.enabled=false \
  logger.video=false \
  logger.backend=csv \
2>&1 | tee "${TRACE_DIR}/dreamer.log"

echo ""
echo "=========================================="
echo "Merging trace files..."
echo "=========================================="

if ls "$TRACE_DIR"/trace_*.json 1> /dev/null 2>&1; then
    prof-merge "$TRACE_DIR"/trace_*.json -o "$TRACE_DIR/merged_trace.json"
    echo "Merged trace saved to $TRACE_DIR/merged_trace.json"
else
    echo "No trace files found in $TRACE_DIR"
fi

echo ""
echo "=========================================="
echo "Quick Analysis"
echo "=========================================="

# Extract key metrics
python3 -c "
import json

with open('$TRACE_DIR/merged_trace.json') as f:
    trace = json.load(f)

events = trace['traceEvents']

# Find actor and value loss times
actor_events = [e for e in events if 'actor_loss/forward' == e.get('name') and e.get('ph') == 'X']
value_events = [e for e in events if 'value_loss/forward' == e.get('name') and e.get('ph') == 'X']
sample_events = [e for e in events if '## train/sample ##' == e.get('name') and e.get('ph') == 'X']

if actor_events:
    actor_times = [e['dur']/1000 for e in actor_events]
    print(f'Actor loss forward: {sum(actor_times)/len(actor_times):.2f}ms avg ({len(actor_times)} calls)')
    print(f'  Range: {min(actor_times):.2f}ms - {max(actor_times):.2f}ms')

if value_events:
    value_times = [e['dur']/1000 for e in value_events]
    print(f'Value loss forward: {sum(value_times)/len(value_times):.2f}ms avg ({len(value_times)} calls)')
    print(f'  Range: {min(value_times):.2f}ms - {max(value_times):.2f}ms')

if sample_events:
    sample_times = [e['dur']/1000 for e in sample_events]
    print(f'Train/sample: {sum(sample_times)/len(sample_times):.2f}ms avg ({len(sample_times)} calls)')

# Check for Triton kernels (torch.compile indicator)
triton = [e for e in events if 'triton_' in e.get('name', '').lower()]
print(f'\\nTriton kernels: {len(triton)} (torch.compile indicator)')
"

echo ""
echo "Test complete! Download traces with:"
echo "  steve cp \$JOBID :$TRACE_DIR/merged_trace.json ./traces_$MODE.json"
