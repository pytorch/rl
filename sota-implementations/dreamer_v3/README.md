# DreamerV3

The maintained implementation includes a compact Pendulum smoke configuration
and a proprioceptive DeepMind Control Walker Walk reproduction configuration.

Run the small example with:

```bash
python sota-implementations/dreamer_v3/train.py
```

Run the full Walker Walk configuration with:

```bash
python sota-implementations/dreamer_v3/train.py \
  --config-name=config_dmc_walker
```

The Walker preset tracks the author-maintained JAX implementation at commit
`e3f02248693a79dc8b0ebd62c93683888ddaccfe`. It matches that implementation's
640,867-parameter `size1m` configuration, uses 16 environments, batches of 16
sequences of length 64, a replay ratio of 1024, and 1.1 million environment
steps. BF16 training is enabled on CUDA. It logs stochastic training-episode
returns against environment steps, matching the current JAX curve protocol
without relying on wall-clock-dependent training iterations.

The Walker task is seeded from `env.seed`, as every other TorchRL example is;
pass `env.use_seed=false` for the JAX implementation's unseeded DMC resets. The
step axis counts initial and reset-only driver records as that implementation
does. Those counts are reporting and update-scheduling semantics only: replay
stores the canonical transitions emitted by the collector and does not insert
synthetic reset records.

This is deliberately a reproduction of the pinned JAX `dmc_proprio` preset,
not of the paper's proprioceptive protocol. The two protocols differ:

| Setting | Pinned JAX `dmc_proprio` preset | DreamerV3 paper proprioceptive protocol |
| --- | --- | --- |
| Model size | `size1m` (640,867 parameters here) | 12M parameters |
| Environment steps | 1.1M | 500K |
| Action repeat | 1 | 2 |
| Replay ratio | 1024 | 512 |
| Optimizer | AGC, LaProp-style RMS scaling then momentum, 1,000-step warmup | Paper recipe |
| Reported aggregation | Three-seed median and interquartile range in this benchmark | Five-seed mean and standard deviation |

TorchRL's public DreamerV3 API documentation remains centered on the paper's
algorithmic semantics. This named SOTA preset documents later choices in the
evolving JAX codebase instead of silently treating them as paper requirements.

Real collection and evaluation environments run on CPU; `optimization.device`
selects where the models, losses and policy run and defaults to `null`, which
auto-selects an available accelerator. Pass `optimization.device=cpu` to force
CPU execution.

Replay is assembled entirely from reusable TorchRL components. One
`TensorDictReplayBuffer` is allocated per environment stream and the configured
`replay_buffer.buffer_size` is split across them. A routed
`ReplayBufferEnsemble` writes synchronous collector batches by their environment
dimension and asynchronous batches by `env_index`. Sampling chooses ready
streams according to their available windows. `replay_buffer.online=true` uses
`StreamingSliceSampler` to consume newly completed, non-overlapping windows
before uniform fallback; `false` uses regular uniform `SliceSampler` sampling.
Both modes sample `seq_len + 1` transitions so the learner can train on the
first `seq_len` and conditionally refresh the following records' latent state.

`collector.backend=auto` (the default) uses `AsyncBatchedCollector` with
multiprocessing environment workers on CPU and CUDA training devices, and the
synchronous `Collector` on other devices: MPS cannot run the acting policy in a
server thread or share its weights with another process. `sync` and `async`
force one of the two; an explicit `async` on MPS needs `collector.policy_device=cpu`.
With the asynchronous collector,
`collector.inference_backend=auto` (the default) serves the acting policy from a
dedicated process that the environment workers reach directly whenever
`collector.async_env_backend=multiprocessing` and `collector.envs_per_worker=1`,
and from a thread of the training process otherwise; `thread` and `process`
force one of the two. `collector.transition_chunk_size=auto` sends one
contiguous run per environment and batch from each worker process. In both
backends collection post-processing, episode reporting, device normalization,
and replay writes happen through the collector's standard replay integration.

For a three-seed median and interquartile reproduction run:

```bash
./sota-implementations/dreamer_v3/reproduce_dmc_walker.sh
```

For the fastest supported accelerator path, enable the compiled RSSM scan
(unrolled eight steps at a time) and CUDA-graph capture of the fixed-shape
learner forward/backward:

```bash
./sota-implementations/dreamer_v3/reproduce_dmc_walker.sh --fast
```

To measure the same fixed-shape learner update after compile and capture
warmup—including every loss, backward, optimizer, and slow-target update—run:

```bash
python benchmarks/ad_hoc/bench_dreamer_v3_learner.py
```

The timing excludes replay sampling and environment collection. Use the
benchmark arguments to change the batch size, sequence length, scan unroll,
warmup, or number of measured updates. Compilation and graph capture happen
during warmup and are excluded from the reported samples.

Pass `--replay-device cpu` or `--replay-device cuda` to benchmark the complete
learner with the same routed replay stack used by training. Replay uses its
built-in one-batch prefetch and ordered conditional updates, allowing sampling
and latent writeback to overlap learner work. CPU replay samples are pinned and
the complete contiguous sequence is transferred non-blockingly before it is
sliced on the learner device.

On one NVIDIA GB200 with PyTorch 2.12.0, CUDA 13.0, BF16, batch size 16,
sequence length 64, scan unroll 8, 10 warmup updates and 50 measured updates:

| Learner backend | Median update | Transitions/s | Speedup |
| --- | ---: | ---: | ---: |
| Compiled scan | 358.51 ms | 2,856 | 1.00x |
| Compiled scan + CUDA graph | 17.83 ms | 57,415 | 20.10x |

Compilation has an up-front cost, so the short validation remains eager:

```bash
./sota-implementations/dreamer_v3/reproduce_dmc_walker.sh --smoke
```

The benchmark writes one metrics file per seed plus `summary.json`, aggregates
the stochastic training returns into median and interquartile curves over fixed
windows, and fails when the final window median falls short. The seeds, the
window and the threshold come from the `benchmark` block of
`config_dmc_walker.yaml`, which ships three seeds, 50,000-step windows and a
minimum final median return of 900; `benchmark.*` Hydra overrides change them,
as in `benchmark.seeds=[0,1,2,3,4]`. `env.seed` and `logger.metrics_jsonl` are
set per run and are rejected as overrides, since either would collapse the
seeds onto one trajectory. Full learning-curve runs are intended for scheduled
or manual validation; pull-request CI uses short smoke overrides. Set
`OUTPUT_DIR` to change the output directory (the defaults are
`dmc_walker_runs` and `dmc_walker_smoke`), and append any other Hydra overrides
to the wrapper, for example `benchmark.seeds=[0]`. Each run logs the resolved
training device, replay device, RSSM backend, scan unroll, mixed-precision state
and learner CUDA-graph setting.

For a smaller ablation, shorten the run rather than the window:

```bash
python sota-implementations/dreamer_v3/benchmark.py --output-dir smoke \
  collector.total_frames=100000 \
  benchmark.minimum_final_median_return=0
```

Every worker runs to the same time limit, so episodes finish in bursts one
episode apart: `(env.max_episode_steps + 1) * collector.num_envs`, or 16,016
records for the preset. A window narrower than that holds no completed episode
over most of the run, so the script refuses one before launching anything. The
command above keeps the 50,000-step window and still fills two of them with
about 48 episodes each.

`optimization.compile=auto` (the default) takes the fastest supported path for
the training device without further input: on CUDA the complete learner forward
and backward is compiled with TorchInductor around a higher-order scan of the
RSSM recurrence and then captured in a CUDA graph; on other devices everything
runs eagerly, since a short CPU run never repays the build.
`optimization.compile=off` disables all of it. The three switches
`optimization.compile_train_step`, `optimization.compile_rssm` and
`optimization.cudagraph_train_step` default to `null`, which follows the
strategy; setting one overrides that single decision, for example
`optimization.compile_train_step=false` keeps the compiled scan and the CUDA
graph but not the whole-step compile.

`optimization.compile_rssm` selects the recurrence backend. `step` compiles the
deterministic work of the explicit loop and draws the same categories as an
eager run; `scan` uses the higher-order scan, unrolled by
`optimization.rssm_scan_unroll=8` steps (lower values reduce compilation time
and graph size, `1` disables manual unrolling), and is faster, but its draws
fall inside the compiled region, so a seeded run diverges from an eager one.
On its own the scan gets its own `torch.compile`, together with the imagination
prior. Inside a compiled step it is selected without a nested compile, so Dynamo
traces one scan of `rssm_scan_unroll` steps; the explicit loop would be unrolled
over the whole sequence there, which makes the compile of a long sequence very
slow (a 256-step batch was still compiling after 15 minutes).
`optimization.compile_train_step=true` compiles the complete learner forward
and backward, including the model, actor, value, and replay value losses. The
compile mode defaults to `optimization.compile_train_step_mode=default`;
autotuning modes are opt-in. Compilation and CUDA-graph warmup use a
fixed-shape synthetic batch before the collector is constructed, so async
collection is not live while Dynamo runs. When combined with
`optimization.cudagraph_train_step=true`, the Inductor-compiled function is
warmed up before CUDA graph capture. `optimization.cudagraph_train_step=true`
captures the learner forward and backward after five warmup calls. It requires
CUDA and fixed input shapes; optimizer and target-network steps remain outside
capture so their schedules continue to advance normally.

The public :class:`~torchrl.trainers.algorithms.DreamerV3OptimizationStepper`
follows the same defaults: `compile_train_step=None` and `cudagraph=None`
resolve from the sample device at `warmup`, and a compiled step switches every
`RSSMRolloutV3` of its loss to the scan (`rssm_scan_unroll=8`) unless a backend
was already selected.

Train-update timing remains asynchronous by default. Set
`optimization.sync_timers=true` when completed GPU timing is needed; this
synchronizes around each measured update and intentionally disables CPU/GPU
overlap.

To compare the existing CUDA-graph path with whole-step compilation, run:

```bash
python benchmarks/ad_hoc/bench_dreamer_v3_learner.py \
  --variants cuda_graph compiled_train_step_cuda_graph
```

Each variant reports the median synchronized update time plus a one-update
profiler sample with kernel count, summed GPU kernel time, and CPU launch time.


## Custom environments, observations and actions

The standard configuration retains its Pendulum environment, synchronous collection,
training defaults and optional JSONL output. A custom environment factory can select
nested observation keys without changing the learner or replay implementation:

```yaml
env:
  backend: custom
  factory: my_envs:make_env
  factory_kwargs: {}
  vector_key: [sensors, vector]
  pixels_key: [sensors, image]
  milestone_key: [episode, milestones]
  milestone_names: [started, completed]
collector:
  async_env_backend: multiprocessing
  envs_per_worker: 4
```

The importable factory receives `seed`, `env_index`, `num_envs` and the configured
keyword arguments, and returns one TorchRL environment. Multiprocessing factories
must be spawn-compatible. Set either observation key to `null` to disable that
path. Vector observations have one feature dimension; images use channels-first
`(C, H, W)` shape and either uint8 pixels or floating-point values scaled to `[0, 1]`.
Image dimensions must match the configured encoder/decoder downsampling stages.
The image decoder predicts unconstrained values against normalized pixel targets.
Continuous vector actions and discrete `OneHot` action specs are supported.

Discrete policies use the public `torchrl.modules.DreamerV3DiscreteActor`.
It owns the normalized network, DreamerV3 initialization, uniform probability
mixture and one-hot sampling. The recipe only supplies dimensions and configured
hyperparameters; standalone users can construct the actor directly or through
`DreamerV3DiscreteActorConfig` and use `get_dist()` for imagination.

Milestone flags are read from the configured key under `next` at each completed
episode. Their boolean vector must match `milestone_names`. Both synchronous and
asynchronous runs log these flags with episode returns. Imagination runs only in
latent state and does not require real sensor or milestone observations.

Asynchronous collection supports `env_exchange` (`auto` picks shared memory
whenever the environment schema allows), `envs_per_worker`, a separate
`policy_device`, and the existing inference batch size/timeout settings. Setting
`inference_static_batch_size` enables fixed-size CUDA-graph policy batches and
requires a CUDA policy device. The default maximum inference batch size remains
`num_envs`. Native replay performs all insertion in the parent process and checks
write generations before applying latent-context updates.

## Logging and time budgets

`logger.backend` selects an optional TorchRL logger, for example `csv`,
`tensorboard` or `wandb`; JSONL logging can remain enabled alongside it. Use
`logger.log_dir` and `logger.exp_name` for local output. W&B additionally accepts
`project`, `entity`, `group`, `tags`, `mode` and an explicit `base_url`. The latter
is passed to W&B settings and takes precedence over its environment-variable
server setting. Tracker metrics use the environment-step counter.

`optimization.max_time` is a positive wall-clock budget in seconds, checked after
completed collection batches. An unlimited frame budget (`collector.total_frames=-1`)
requires a time budget. A current batch or learner update is allowed to finish;
this is not a hard process deadline. `optimization.collection_warmup_seconds`
collects into native replay without training for that duration, measured from the
first completed collection batch. Both options preserve their disabled defaults.
Compilation and capture warm-up complete before collection starts, using the real
observation specs and the same learner input keys as native replay. Gradient
buffers remain attached after capture, and an optimizer step without parameter
gradients fails explicitly.


Checkpoint saving is opt-in. For example:

```bash
python sota-implementations/dreamer_v3/train.py \
  optimization.checkpoint_dir=checkpoints \
  optimization.checkpoint_every=1000 \
  optimization.checkpoint_include_replay=true
```

`checkpoint_every` counts completed learner updates; a save happens after the
current training batch. Set it to `null` to save only on completion or graceful
termination. `checkpoint_keep_last=2` controls retention. Checkpoint directory
names use the cumulative action count. The default `checkpoint_dir=null`
disables saving, and replay is excluded unless explicitly enabled.

Resume explicitly from a checkpoint or a rotation directory:

```bash
python sota-implementations/dreamer_v3/train.py \
  optimization.resume_from=checkpoints \
  optimization.checkpoint_dir=checkpoints \
  collector.total_frames=10000
```

The frame budget is cumulative, so increase it to continue a completed run.
Elapsed time also resumes from the saved value; increase `optimization.max_time`
when continuing a run that reached its time budget. The environment count,
replay capacity, batch/sequence sizes and sampling mode must match the saved
configuration. Model shapes must remain compatible with the saved state.

TorchRL checkpoint utilities save the learner (including normalization and
slow-value parameters), optimizer, target-update state, global and replay RNGs,
policy RNG counter, update-ratio remainder, reporting counters, unfinished
logging window, logger identity and resolved configuration. Optional native
replay serialization includes writer generations, streaming sampler state and
queued prefetched samples. Async collection pauses and pending replay operations
finish before the snapshot. CUDA work is synchronized before saving.

A resumed process performs compile/capture warm-up before loading training state,
then restores RNG state after constructing environments and loggers. JSONL and
CSV logging append to their saved paths. W&B resumes the saved run ID with
`resume="must"`; keep the same logger backend and service configuration.

Environments restart. Saved unfinished replay tails become truncation boundaries
before new transitions arrive, and incomplete episode returns restart at zero.
Initial reset records are counted again when reset-record reporting is enabled.
Queued collector results that were not emitted are not checkpointed. The acting
policy is rebuilt from the restored learner; when using a separate policy RNG,
its saved module state and counter are then restored. With process inference
(`collector.inference_backend=auto` where it applies, or `process`), the
inference process rebuilds the policy on its device and receives the restored
weights before it serves the first request, but the separate policy RNG counter
stays in the training process, so the served stream restarts from the seed.
Consequently resumed trajectories are not promised to match an uninterrupted run. Without a replay payload,
collection refills native replay before training continues, without accumulating
a catch-up update burst. Counters and logger identity still continue.

SIGINT and SIGTERM request a stop after the current collection/update batch.
The final checkpoint is written before replay, environments and loggers close.
Abrupt process termination cannot take a final snapshot; resume from the latest
completed checkpoint instead.


Checkpoint ownership is provided by core components: `DreamerV3Loss` stores
learner and normalization state; `DreamerV3OptimizationStepper` stores optimizer
and target-update progress. `DreamerV3SeededPolicy` and `DreamerV3UpdateRatio`
serialize their own RNG counter and fractional scheduling progress. The recipe
registers these objects with `Checkpoint` directly. Native replay closes restored
stream tails through `ReplayBufferEnsemble.end_streams()`, and
`get_logger(..., state_dict=saved_state)` owns logger identity and counter restoration.
