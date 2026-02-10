# Using `steve` on the cluster (and writing a good `setup-and-run.sh`)

This is a short practical guide for running Dreamer (or any RL job) via `steve` on the cluster, based on what actually went wrong / what worked reliably.

## The mental model

- A `steve job` gives you **an allocated pod/container** (the “node”).
- `steve step $JOBID "<cmd>"` runs a **command inside that container**, and logs to a per-step log directory.
- `steve step -d $JOBID "<cmd>"` launches the step **detached** (good for long training).
- `steve cp` copies files **to/from** the job container via rsync.

## Minimal “happy path” workflow

### 1) Create a job

```bash
JOBID=$(steve job --partition h200-high --gpus-per-task 8 --ntasks 1 --time 840:00:00 --job-name "rl" --jobid-only)
```

### 2) Copy your setup script to the container

```bash
steve cp "$JOBID" ./setup-and-run.sh :/root/setup-and-run.sh
```

### 3) Run setup (build-only first)

This validates dependencies and avoids “mystery manual patches”.

```bash
steve step "$JOBID" 'GH_TOKEN=<YOUR_GH_TOKEN> bash /root/setup-and-run.sh --nightly --build-only'
```

### 4) Run training (no profiling)

```bash
steve step -d "$JOBID" 'source /root/torchrl/bin/activate && cd /root/rl && MUJOCO_GL=egl WANDB_MODE=online python sota-implementations/dreamer/dreamer.py profiling.enabled=false profiling.distributed.enabled=false profiling.collector.enabled=false logger.video=false'
```

## `steve cp` gotcha (very common)

Exactly **one** of SOURCE or DEST must be “remote”, indicated by a leading `:`.

- Copy **from** job to local:

```bash
steve cp "$JOBID" :/root/traces/merged_trace.json ./traces_new/merged_trace.json
```

- Copy **to** job:

```bash
steve cp "$JOBID" ./setup-and-run.sh :/root/setup-and-run.sh
```

If you see:
> “Exactly one of SOURCE or DESTINATION must start with ':'”

…it means both sides looked “local” to `steve`.

Also: ensure the local destination directory exists (e.g. `mkdir -p traces_new/`) before copying.

## Environment variables you’ll actually use

### Required for private repos

- `GH_TOKEN`: GitHub PAT with permission to read private repos.
  - Needed if your setup clones private repos (e.g. `vmoens/prof`).

### MuJoCo headless rendering

- `MUJOCO_GL=egl`: fixes headless OpenGL issues on nodes.

### Weights & Biases

- `WANDB_MODE=online|disabled|offline`
- Prefer to **login once** in the container (writes `/root/.netrc`) and then rely on that.
- Avoid printing `WANDB_API_KEY` in logs: it can end up in fused logs via `ps`/tracebacks.

### `prof` distributed profiling (when you do want profiling)

These should be set only for a profiling run:

- `PROF_ENABLED=1`
- `PROF_ITERATIONS=50-55` (or any window after warmup)
- `PROF_OUTPUT_DIR=/root/traces`
- `PROF_MODE=lite` (or `full`)

Notes:

- `prof` is coordinating the **PyTorch profiler** across processes; it’s not “instead of” PyTorch profiler, it’s a *launcher/collector* for it.
- For profiling runs, make sure `optimization.total_optim_steps` is high enough to include the `PROF_ITERATIONS` window.

## How to write a good `setup-and-run.sh` (like `setup-and-run.sh` in this repo)

The key properties that made it reliable:

- **Single source of truth**: all dependencies installed in one place, not ad hoc per run.
- **Fail early on missing credentials**:
  - If a private repo is required, check `GH_TOKEN` and exit with a clear message.
- **Idempotent**:
  - Only clone repos if missing.
  - Use “safe pull” logic to recover from diverged branches.
  - Create the venv only if absent.
- **Explicit verification**:
  - Print torch/cuda/cudnn status (and run a tiny CUDA conv) to catch “built without cuDNN” situations.
- **Headless graphics deps**:
  - Install EGL/GL libs once (and export `MUJOCO_GL=egl`).

What to avoid in setup scripts:

- **Hard-coding secrets** (GH/WandB keys) in the file.
  - Use environment variables instead.
- **Relying on system `python -m venv`** in minimal containers.
  - Some images don’t have `ensurepip` / `python3-venv`; use `uv venv` (worked reliably for us).
- **Manual patching after setup** (e.g. apt installs, random pip installs).
  - It becomes impossible to reproduce and you’ll “lose track” of the environment.

## Logging and monitoring

- Every `steve step` creates a step log dir under something like:
  - `/mnt/home/logs/slurm/steps/$SLURM_JOB_ID/step-$SLURM_STEP_ID/fused.log`
- For detached steps, prefer tailing logs rather than re-running commands.

Example:

```bash
steve step "$JOBID" 'tail -n 200 /mnt/home/logs/slurm/steps/$SLURM_JOB_ID/step-$SLURM_STEP_ID/fused.log'
```

## What went wrong (lessons learned)

- **Misusing `steve cp`**:
  - Fixed by always using `:$REMOTE_PATH` for the remote side and pre-creating local dirs.
- **Running in the wrong repo directory**:
  - Some containers have `/root/code` pointing to another repo; always `cd /root/rl` (or whatever your script sets up) before running.
- **Missing `GH_TOKEN`**:
  - Setup couldn’t clone private `prof` repo, breaking profiling.
- **Missing EGL libs**:
  - MuJoCo import failed with EGL errors until EGL/GL libs were installed (should be done in setup).
- **WandB not logged in**:
  - Training crashed early with “No API key configured” until login happened inside the container.
- **Process leaks / zombies between runs**:
  - Old multiprocess workers caused BrokenPipe/leaked semaphores and confusing failures.
  - Fix: `pkill -9 python` before relaunching (or create a fresh job).
- **Secrets ended up in logs**:
  - Avoid commands that print full `ps aux` lines or environment dumps when `WANDB_API_KEY` is set.
- **Replay buffer “experimental” optimizations broke training**:
  - Micro-optimizations (e.g. `stack_onto_`) can silently violate invariants and crash later.
  - Treat them as experimental until proven by end-to-end tests on real Dreamer async runs.
