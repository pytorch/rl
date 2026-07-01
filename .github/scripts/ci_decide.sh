#!/usr/bin/env bash
# Decide the selective-CI matrix from event, ref, labels, and changed-file flags.
#
# Writes key=value lines on stdout. The caller appends the output to
# $GITHUB_OUTPUT so downstream jobs can read them via needs.prepare.outputs.*.
#
# Precedence (highest first):
#   1. push / workflow_call / workflow_dispatch        -> full matrix
#   2. pull_request with ciflow/full label              -> full matrix
#   3. pull_request: per-track changed-files flags,
#      with per-track ciflow/* labels as escape hatches
#
# Usage:
#   ci_decide.sh EVENT REF LABELS_JSON SHARD1 SHARD2 SHARD3 DISTRIBUTED
#
#   LABELS_JSON is the raw toJSON output of github.event.pull_request.labels.*.name,
#   e.g. ["Transforms","ciflow/full"]. On non-PR events pass [].
#   SHARD1/SHARD2/SHARD3/DISTRIBUTED are literal "true"/"false" from
#   tj-actions/changed-files <bucket>_any_changed outputs.

set -euo pipefail

event_name="${1:?event_name required}"
ref="${2:?ref required}"
labels_json="${3:-[]}"
shard1_changed="${4:-false}"
shard2_changed="${5:-false}"
shard3_changed="${6:-false}"
distributed_changed="${7:-false}"

has_label() {
  # labels_json example: ["Foo","ciflow/full"]. Match the quoted name exactly.
  printf '%s' "$labels_json" | grep -Fq "\"$1\""
}

is_full=false
case "$event_name" in
  push|workflow_call|workflow_dispatch) is_full=true ;;
esac
if [ "$event_name" = pull_request ] && has_label "ciflow/full"; then
  is_full=true
fi

if [ "$is_full" = true ]; then
  run_cpu_matrix=true
  run_shard1=true
  run_shard2=true
  run_shard3=true
  run_olddeps=true
  run_stable=true
  run_optdeps=true
  run_distributed=true
else
  if has_label "ciflow/cpu-matrix"; then run_cpu_matrix=true; else run_cpu_matrix=false; fi

  if has_label "ciflow/gpu"; then
    run_shard1=true; run_shard2=true; run_shard3=true
  else
    run_shard1="$shard1_changed"
    run_shard2="$shard2_changed"
    run_shard3="$shard3_changed"
  fi

  if has_label "ciflow/olddeps"; then run_olddeps=true; else run_olddeps=false; fi
  if has_label "ciflow/stable";  then run_stable=true;  else run_stable=false;  fi
  if has_label "ciflow/optdeps"; then run_optdeps=true; else run_optdeps=false; fi

  if [ "$distributed_changed" = true ] || has_label "ciflow/distributed"; then
    run_distributed=true
  else
    run_distributed=false
  fi
fi

shards=()
[ "$run_shard1" = true ] && shards+=('"1"')
[ "$run_shard2" = true ] && shards+=('"2"')
[ "$run_shard3" = true ] && shards+=('"3"')
# Safety net: paths-ignore should keep us from reaching here on docs-only PRs,
# but if no shard is selected we still want one to run so the job matrix is non-empty.
if [ "${#shards[@]}" -eq 0 ]; then
  shards+=('"3"')
  run_shard3=true
fi
shards_json="[$(IFS=,; printf '%s' "${shards[*]}")]"

cat <<OUT
full=$is_full
run_cpu_matrix=$run_cpu_matrix
run_shard1=$run_shard1
run_shard2=$run_shard2
run_shard3=$run_shard3
run_olddeps=$run_olddeps
run_stable=$run_stable
run_optdeps=$run_optdeps
run_distributed=$run_distributed
shards_json=$shards_json
OUT
