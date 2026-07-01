# Selective CI on pull requests

This repo runs a large matrix of Linux/Windows/GPU/dependency variants on every
push to `main`, `nightly`, and `release/*`. On PRs, the matrix is **pruned to the
tests relevant to the diff**, with opt-in `ciflow/*` labels to pull individual
tracks back in without waiting for merge.

The goal is fast PR iteration, with the assurance that the full matrix still
runs at merge time.

## What runs on a PR by default

| Track | Trigger on PR | Full trigger |
|---|---|---|
| `test-linux.yml` → `tests-cpu` | Python 3.12 only | all 5 Python versions on push / `ciflow/cpu-matrix` / `ciflow/full` |
| `test-linux.yml` → `tests-gpu` | only the shard(s) whose paths changed; falls back to shard 3 | all 3 shards on push / `ciflow/gpu` / `ciflow/full` |
| `test-linux.yml` → `tests-gpu-distributed` | only if distributed code (`torchrl/collectors/distributed/**`, `torchrl/services/**`, `test/test_distributed.py`, `test/test_rb_distributed.py`) changed | on push / `ciflow/distributed` / `ciflow/full` |
| `test-linux.yml` → `tests-olddeps` | skipped | on push / `ciflow/olddeps` / `ciflow/full` |
| `test-linux.yml` → `tests-optdeps` | skipped | on push / `ciflow/optdeps` / `ciflow/full` |
| `test-linux.yml` → `tests-stable-gpu` and `tests-stable-gpu-distributed` | skipped | on push / `ciflow/stable` / `ciflow/full` |
| `test-linux-sota.yml` | only if `sota-implementations/**` or `torchrl/trainers/**` changed | on push / `ciflow/sota` / `ciflow/full` |
| `test-linux-tutorials.yml` | only if `tutorials/**` changed | on push / `ciflow/tutorials` / `ciflow/full` |
| `test-windows-optdepts.yml` | only if `torchrl/**`, `test/**`, `setup.py`, `pyproject.toml`, or `.github/unittest/windows_optdepts/**` changed | on push / `ciflow/windows` / `ciflow/full` |
| `test-linux-libs.yml`, `test-linux-llm.yml`, `test-linux-habitat.yml` | already label-gated on file-based labels (`Environments/*`, `llm/`, `Modules`, ...) applied by `.github/workflows/auto-labeler.yml` | on push or matching label |

Docs-only / `sota-implementations/`-only / `tutorials/`-only PRs skip the relevant
heavy workflows via `paths-ignore:` — the workflow doesn't even enqueue.

## GPU shard selection

The GPU test suite is split three ways:

- **Shard 1** — `test/transforms/`
- **Shard 2** — `test/envs/` and `test/test_collectors.py`
- **Shard 3** — everything else

The `prepare` job at the top of `test-linux.yml` uses
[`tj-actions/changed-files`](https://github.com/tj-actions/changed-files) to
compute which shards' paths intersect the PR diff and emits a JSON array
(`shards_json`) consumed by the `tests-gpu` matrix. A PR touching only
`torchrl/envs/transforms/vec_norm.py` runs shard 1 only; a PR touching
`torchrl/objectives/ppo.py` runs shard 3 only.

If no shard matches (shouldn't happen thanks to `paths-ignore`), the safety net
is to run shard 3.

## `ciflow/*` escape hatches

Apply any of these labels to a PR to force specific tracks back in:

- `ciflow/full` — run the whole matrix (everything below, plus all 5 CPU Python versions)
- `ciflow/cpu-matrix` — run all 5 Python versions on the CPU job
- `ciflow/gpu` — run all 3 GPU shards
- `ciflow/stable` — run `tests-stable-gpu` + `tests-stable-gpu-distributed`
- `ciflow/olddeps` — run `tests-olddeps`
- `ciflow/optdeps` — run `tests-optdeps`
- `ciflow/distributed` — run `tests-gpu-distributed`
- `ciflow/sota` — run `test-linux-sota`
- `ciflow/tutorials` — run `test-linux-tutorials`
- `ciflow/windows` — run `test-windows-optdepts`

Push events to `main`, `nightly`, or `release/*` implicitly behave as if
`ciflow/full` were set.

## How the decision is made

[`.github/scripts/ci_decide.sh`](scripts/ci_decide.sh) centralises the
precedence logic. It takes event name, ref, labels JSON, and the four
`<bucket>_any_changed` flags from `tj-actions/changed-files`, and writes
`key=value` lines suitable for `$GITHUB_OUTPUT`.

Precedence (highest first):

1. `push` / `workflow_call` / `workflow_dispatch` → full run.
2. PR with `ciflow/full` → full run.
3. PR: per-track file-change flags OR per-track `ciflow/*` labels.

Unit tests for the script: [`.github/scripts/test_ci_decide.sh`](scripts/test_ci_decide.sh).
Run locally:

```bash
bash .github/scripts/test_ci_decide.sh
```

## Running the same filtered subset locally

The test driver `.github/unittest/linux/scripts/run_all.sh` respects the same
sharding env vars used in CI:

```bash
# Run just the transforms shard (what a PR touching transforms would run on GPU CI)
TORCHRL_TEST_SHARD=1 bash .github/unittest/linux/scripts/run_all.sh

# Run just envs + collectors
TORCHRL_TEST_SHARD=2 bash .github/unittest/linux/scripts/run_all.sh

# Run everything else
TORCHRL_TEST_SHARD=3 bash .github/unittest/linux/scripts/run_all.sh
```

For even finer selection, the `TORCHRL_TEST_PATHS` env var is passed verbatim
to `pytest` and short-circuits the shard dispatch:

```bash
# Arbitrary pytest invocation, wrapped in the CI's coverage runner.
TORCHRL_TEST_PATHS="test/test_rb.py -k memmap" \
  bash .github/unittest/linux/scripts/run_all.sh
```

This is useful for `workflow_dispatch` runs and local reproduction of a
failing CI shard.

## Adding a new track

When adding a new workflow (or a new track in an existing one) that you want
to be selective on PRs:

1. Add a `paths-ignore:` to the PR trigger for obviously-unrelated paths.
2. Add (or extend the existing) `prepare`/`changes` job to emit an `in_scope`
   output via `tj-actions/changed-files`.
3. Gate the heavy job(s) with:
   ```yaml
   if: >-
     github.event_name != 'pull_request' ||
     needs.changes.outputs.in_scope == 'true' ||
     contains(github.event.pull_request.labels.*.name, 'ciflow/full') ||
     contains(github.event.pull_request.labels.*.name, 'ciflow/<track>')
   ```
4. Register `ciflow/<track>` in `.github/labels.yml` and document it above.

If the new track needs shard-style output coordination with `test-linux.yml`,
extend `ci_decide.sh` instead of bolting on another decision script.
