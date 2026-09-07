# Continuous async environment benchmarks

The [Continuous Benchmark workflow](https://github.com/pytorch/rl/actions/workflows/benchmarks.yml)
runs the short async suite after each main-branch merge touching TorchRL or
benchmarks. The full suite still runs through the nightly orchestrator.
The [async trend dashboard](https://pytorch.org/rl/dev/bench/async/) appears after
the first successful main run. Each point links to its source commit. Branch
runs produce the same summary and downloadable artifacts without publishing to
the main trend. This lets a benchmark-only PR establish the baseline before
merging collector changes.

## Workloads and series

Existing `test_async_env_pool_*` names and workloads remain the dispatch,
shared-memory versus queue, slow-reset, and affinity reference series. Aggregate
receives now cap the final harvest to the remaining frame budget, so throughput
never counts an oversized final batch as fewer transitions.

`test_async_collection_pixels[mode-regime]` measures complete collection through
the public `Collector` and `AsyncBatchedCollector` APIs. It reuses `PixelMockEnv`
and `PolicyFactory` from `bench_collectors.py`.

| Input | CPU | GPU |
| --- | --- | --- |
| Environments | 8 CPU processes | 32 CPU processes |
| Observation | 3 x 84 x 84 uint8 pixels | same |
| Policy | CNN + two 256-wide layers | CNN + two 1024-wide layers |
| Environment step delay | 1 ms | same |
| Output batch / measured round | 256 / 1,024 transitions | same |
| Warm-up / measured rounds | 2 / 5 | same |
| Fresh process repetitions | 3 | same |
| Torch threads / seed | 1 / 0 | same |
| Async batching | min 1, max environment count, timeout 1 ms | same |

Uniform episodes last 200 steps. The slow-reset workload has 16-step episodes
and adds 200 ms to environment zero's reset. This exposes barriers without
artificially delaying every stream. Setup, process startup, graph capture and
warm-up are excluded from measurements. Each round checks its exact emitted
transition budget. Collection and teardown failures fail the run.

The modes are `parallel` (synchronous reference), `async-queue`, `async-shm`,
`async-shm-static` (CUDA graphs), and `async-shm-grouped` (four environments per
worker). The last two explicitly skip on revisions before their public APIs
exist. They start new series when those features merge; absence is never
reported as zero time or as a speedup. CUDA-only cases carry the `gpu` marker.
Existing eager series continue unchanged through all merges.

## Reading results

The dashboard plots **transitions per second**, higher is better. Each point is
the median of three run medians; its range is the minimum and maximum run
median. The range is run-to-run variation, not a confidence interval. Compare
changes with that variation and repeat borderline measurements. There is no
promise that every mode or merge improves performance: synchronous batching can
win for uniform cheap environments, while asynchronous collection can absorb
slow resets.

The Actions summary includes batch p95 latency, final process-tree RSS and CUDA
peak allocation. Raw JSON includes every measured round, batch latency p50/p95,
server batching/queue/forward statistics, machine information, and dependency
versions. RSS includes shared pages once per process and is a final snapshot,
not peak unique memory. CUDA peak covers measured rounds in the policy process.
Artifacts are retained for 30 days and the dashboard keeps 250 points per series.

The short suite uses the existing `linux.g5.4xlarge.nvidia.gpu` runner class, pinned
container digests, Python 3.10.20, an exact dependency lock and a pinned TensorDict
source revision. CPU and GPU are separate series. Dependency or workload changes
must start a new trend version; do not present them as collector improvements.
The full nightly suite retains its moving dependency environment and its
[existing dashboard](https://pytorch.org/rl/dev/bench/).

## Running and merge order

Use **Run workflow**, choose the branch and `suite: async`. The CLI equivalent is
`gh workflow run benchmarks.yml --ref <branch> -f suite=async -f skip-upload=true`.
A default manual run on main publishes; set `skip-upload` to keep artifacts only.
To rerun the complete nightly workload, choose `suite: full`.

Locally, install `benchmarks/requirements.txt` and run from the repository:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 python -m pytest \
  benchmarks/test_envs_benchmark.py benchmarks/test_collectors_benchmark.py \
  -k 'async_env_pool or async_collection_pixels' --timeout=300 \
  --benchmark-only --benchmark-save-data --benchmark-json=async-1.json
```

Repeat in three separate processes with filenames `async-1.json` through
`async-3.json`, then run
`python .github/scripts/summarize_async_benchmarks.py <results-directory> --summary summary.md`.

Merge this benchmark/CI change before #4269 to record the current eager baseline.
Then review #4269, #4271 and #4272 in order. Their graph, grouping and coordination
changes appear against the continuing eager series. Replay integration follows
#4273, #4274 and #4275; it must retain these collection workloads and introduce
separately named replay-inclusive series if the measured operation changes.

## CI health and publication

The September 7 nightly run passed CPU benchmarks but timed out during two TQC
CUDA compilation warm-ups; the shared upload dependency suppressed all results.
Compiled TQC now has a bounded 15-minute setup budget without changing its timed
workload. CPU and GPU publication is independent, only successful suites enter
the trend, and partial output remains available as diagnostic artifacts.
Failures still mark the workflow failed. Manual branch runs cannot contaminate
main history.
