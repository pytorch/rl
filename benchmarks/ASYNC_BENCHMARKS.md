# Continuous async environment benchmarks

The [Continuous Benchmark workflow](https://github.com/pytorch/rl/actions/workflows/benchmarks.yml)
runs the short async suite after each main-branch merge touching TorchRL,
benchmarks or the DreamerV3 example. The full suite still runs through the nightly
orchestrator.
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
| Torch threads / parent seed | 1 / 0 | same |
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

The `async-process-slots` and CUDA-only `async-process-slots-static` series
exercise the direct worker-to-server transport when `ProcessSlotTransport`
becomes public in #4272. They keep one environment per worker because that
transport rejects grouped workers. The same workload and batching limits apply;
these remain separate series alongside the grouped integrated curve.

The fixed series pin what the collector now resolves on its own: the thread-server
modes pass `transport="thread"` and the process-slot modes pass an explicit
`transition_chunk_size` (`1` except for the chunked series), so a change of the
collector defaults cannot move them.

`async-process-slots-integrated` is the cumulative curve of the direct-transport
pipeline, the counterpart of `async-shm-integrated`. It keeps one environment per
worker (the transport rejects grouped workers), enables static inference batches
on GPU when that option is available, and sends 64 transitions per worker message
once `AsyncBatchedCollector` accepts `transition_chunk_size`; before that it
matches `async-process-slots`. Each point records its configuration in the
`execution` field, so a step in this series can be attributed to the option that
changed. Use the fixed `async-process-slots*` series to separate the individual
changes.

`async-process-slots-chunked` is the same direct-transport workload with
`transition_chunk_size=64`: each environment process sends 64 consecutive
transitions per message instead of one, so the driver concatenates whole chunks
into each 256-transition batch rather than unpickling and stacking transitions
one at a time. Compare it with `async-process-slots` to see what the driver's
per-transition work costs; the series skips on revisions before the option
exists. `bench_collectors.py --backends async-process-slot --replay-mode
background --transition-chunk-size N` measures the same choice as replay-buffer
write throughput.

`test_async_collection_pixels_64_envs[mode]` adds a separate CUDA acceptance
comparison for `async-shm` and `async-process-slots`: 64 environments, one per
worker, with the same CNN plus two 1024-wide layers, one-millisecond environment
steps, batches, warm-up and measured rounds described above. It runs in the
same three fresh-process repetitions. Compare those two series with each other;
they keep worker count fixed and do not replace the existing 32-environment
trends. The policy is a pixel-collection proxy, not the complete DreamerV3 actor.

Follow **`async-shm-integrated`** for the cumulative performance curve. It starts
with eager inference and one environment per worker, enables static inference
on GPU when that public option becomes available, and uses four environments
per worker when grouping becomes available. Its environment/policy workload and
measurement budget remain fixed. Each point records the execution configuration
in its tooltip, summary and raw JSON. These explicit configuration transitions
show the combined pipeline as features land; use the fixed modes to separate
individual changes and detect regressions.

## DreamerV3 training series

`test_dreamer_v3_async_training[backend-ratio]` (`test_dreamer_v3_benchmark.py`)
measures the `sota-implementations/dreamer_v3` example end to end. The example
runs unmodified in a child process using the fixed `dreamer_v3.yaml` benchmark
configuration, independent of the example defaults. Changes to its training loop,
replay write-back and inference wiring show up here. The workload is
`bench_dreamer_v3_env.py`: eight environment processes producing 3 x 64 x 64 uint8
pixels, an 8-float vector and three boolean
milestones, with one-millisecond steps, six discrete actions and episode lengths
staggered by environment index. Collection is asynchronous with shared-memory
exchange and one environment per worker; the learner runs eagerly on the GPU
(`optimization.compile=off`) with a small network configuration, replay batches of
16 sequences of 32 records and replay context write-back enabled. Inference is
limited to one request per batch: the current thread backend can mix exploration
contexts while the learner runs, causing larger batches to fail. Keep this limit
fixed after the bug is resolved; the collection-only series measures batched
inference.

| Series | Learner updates per collected batch | Emphasis |
| --- | --- | --- |
| `ratio2` | `train_ratio=2`: one update per 256-transition batch | collection path, driver overhead |
| `ratio16` | `train_ratio=16`: eight updates per batch | learner step, replay sampling and write-back |

`thread` serves the acting policy from the training process; `process` serves it
from a dedicated inference process and skips until the example exposes
`collector.inference_backend`. The series are CUDA-only.

Throughput is read from the example's own metrics log (`logger.metrics_jsonl`,
one `train` record per 256-transition batch): after the first learner update,
two warm-up rounds of 1,024 environment steps pass unmeasured, then five
measured rounds each wait for the next 1,024 steps to be logged. Setup, process
startup, replay warm-up and the final shutdown stay outside the measurement.
The summary reports frames per second like the other series; the raw JSON adds
learner updates per second, total measured updates and the process-tree RSS. The
pinned lock includes `hydra-core` and `omegaconf` for the example. CUDA-graph
inference batches stay disabled in this workload.

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
not peak unique memory. CUDA peak covers measured rounds when the policy runs in
the benchmark process.
Process-slot series omit that allocator metric because inference owns a separate
CUDA process; they still report complete process-tree RSS and server timings.
Artifacts are retained for 30 days and the dashboard keeps 250 points per series.

The short suite uses the existing `linux.g5.4xlarge.nvidia.gpu` runner class, pinned
container digests, a September 7 Ubuntu archive snapshot, Python 3.10.20, an
exact dependency lock (stable PyTorch 2.14.0 with CUDA 12.6) and a pinned
TensorDict source revision. Stable wheels keep cache eviction independent of
nightly wheel retention. The snapshot
uses the [Ubuntu snapshot service](https://snapshot.ubuntu.com/). System package
versions and the GPU model/driver are recorded alongside the Python environment. CPU and GPU are separate series. Dependency or workload changes
must start a new trend version; do not present them as collector improvements.
The source checkout intentionally tests TensorDict development code, whose base
version metadata can precede TorchRL's declared stable release floor. `pip check`
runs before installing the source checkout; source compatibility is exercised by
the benchmark suites, not claimed from that metadata check.
The full nightly suite uses a matching nightly PyTorch/torchvision pair, validates
the third-party environment with `pip check`, and retains its
[existing dashboard](https://pytorch.org/rl/dev/bench/).

## Running and merge order

Use **Run workflow**, choose the branch and `suite: async`. The CLI equivalent is
`gh workflow run benchmarks.yml --ref <branch> -f suite=async -f skip-upload=true`.
A default manual run on main publishes; set `skip-upload` to keep artifacts only.
To rerun the complete nightly workload, choose `suite: full`.

Merging a pull request that carries the `benchmarks/trigger` label dispatches
the full suite on main immediately (`benchmarks_post_merge.yml`), so the merge
gets a trend point without waiting for the nightly sample. The short async
suite also runs on main merges touching `torchrl/`, `benchmarks/` or
`sota-implementations/dreamer_v3/`.

Locally, install `benchmarks/requirements.txt` and run from the repository:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONHASHSEED=0 python -m pytest \
  benchmarks/test_envs_benchmark.py benchmarks/test_collectors_benchmark.py \
  benchmarks/test_dreamer_v3_benchmark.py \
  -k 'async_env_pool or async_collection_pixels or dreamer_v3_async' --timeout=300 \
  --benchmark-only --benchmark-save-data --benchmark-json=async-1.json
```

Repeat in three separate processes with filenames `async-1.json` through
`async-3.json`, then run
`python .github/scripts/summarize_async_benchmarks.py <results-directory> --summary summary.md`.

Merge the benchmark additions and record a successful main-branch run before
merging #4304, #4305, #4306, #4307 or #4308. Keep the workload fixed while those
changes land; the process-inference series starts when #4304 exposes its config
option, alongside the continuing thread-inference baseline.

## CI health and publication

The September 7 nightly run passed CPU benchmarks but timed out during two TQC
CUDA compilation warm-ups; the shared upload dependency suppressed all results.
Compiled TQC now has a bounded 15-minute setup budget without changing its timed
workload. CPU and GPU publication is independent, only successful suites enter
the trend, and partial output remains available as diagnostic artifacts.
Failures still mark the workflow failed. Manual branch runs cannot contaminate
main history. A performance alert (a series more than twice as slow as its
previous point) does not fail the upload job; the upload job summary compares
every series with its previous point.
