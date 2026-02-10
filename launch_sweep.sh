#!/usr/bin/env bash
# launch_sweep.sh - Poll until job starts, then set up and launch training
# Usage: bash launch_sweep.sh <JOBID> <HYDRA_OVERRIDES...>
set -euo pipefail

steve() {
    ( cd /Users/vmoens/repos/periodic-mono && VIRTUAL_ENV= uv run --no-sync --package pj pj "$@" )
}

JOBID="$1"
shift
OVERRIDES=("$@")

WANDB_KEY="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"

# Poll until job is RUNNING (check every 60s, up to 24h)
echo "[sweep] Job $JOBID: polling until RUNNING..."
for i in $(seq 1 1440); do
    STATE=$(steve squeue 2>&1 | grep "$JOBID" | awk '{print $4}' || true)
    if [[ "$STATE" == "R" ]]; then
        echo "[sweep] Job $JOBID: now RUNNING!"
        break
    fi
    if [[ -z "$STATE" ]]; then
        echo "[sweep] Job $JOBID: no longer in queue (cancelled?). Aborting."
        exit 1
    fi
    sleep 60
done

# Wait a bit for container initialization
echo "[sweep] Job $JOBID: waiting 60s for container init..."
sleep 60

echo "[sweep] Job $JOBID: installing rsync..."
steve step "$JOBID" 'apt-get update -qq && apt-get install -y -qq rsync > /dev/null 2>&1'

echo "[sweep] Job $JOBID: copying setup script..."
steve cp "$JOBID" ./setup-and-run.sh :/root/code/setup-and-run.sh

echo "[sweep] Job $JOBID: launching training with overrides: ${OVERRIDES[*]:-<none>}"
OVERRIDE_STR="${OVERRIDES[*]:-}"
steve step -d "$JOBID" "export WANDB_API_KEY=${WANDB_KEY} && cd /root/code && bash setup-and-run.sh ${OVERRIDE_STR}"

echo "[sweep] Job $JOBID: launched successfully!"
