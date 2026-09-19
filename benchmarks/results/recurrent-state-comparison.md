# Recurrent state container comparison

Measurements collected 10 September 2026; draft integration revalidated
11 September 2026. The public state container is still a
decision gate. Existing GRU/LSTM defaults and the reference transformer's
module-owned caches retain their existing behavior.

## Recommendation

Use **TypedTensorDict** for the proposed opt-in structured state API, subject to
review of this comparison. It offers typed field access while remaining a
TensorDictBase mapping. Both typed candidates now work through the exercised
environment, collector, storage and training paths. Neither has a consistent
end-to-end throughput advantage in these CPU measurements. TensorClass remains
a viable choice; its generic access requires `.get()` rather than string
indexing in code shared with TensorDict.

Keep the compatibility fixes independent of the container decision. Keep the
current flat GRU/LSTM representation as the default: nested containers add
measurable overhead for small recurrent models.

For large transformer states, add an **opt-in whole-window replay contract** as
separate follow-up work. Dense storage can already represent one initial state
alongside a window of transitions. Restricting valid sequence starts makes the
storage reduction possible without reconstructing missing carries. Do not
silently change the existing SliceSampler contract.

## Dependency snapshot and implementation

The timing snapshot used these dependency revisions on 10 September:

- [TensorDict #1766](https://github.com/pytorch/tensordict/pull/1766), head
  `5270f21ee4cef936cf4d7132e0a26ff2937f7377`.
- [TorchRL #4193](https://github.com/pytorch/rl/pull/4193), head
  `75ef00c52caddada9cfbfb8ecf1bb211137b044c`, merged locally with TorchRL main
  `9a58d8450` for this experiment.

The draft stack was subsequently rebuilt on TorchRL main `a2acf99c7`, which
includes the merged #4193, and validated against TensorDict #1766 head
`b4fcf028e1b2901e22a26254e2ecc1bc7c0e996f` plus the compatibility fix in
[TensorDict #1789](https://github.com/pytorch/tensordict/pull/1789).
The timing tables retain the original measurements; they were not rerun during
PR preparation. TensorDict #1766 is still open, so recheck its final revision
before release integration.

All three candidates use identical models, tensors and leaf specs:

| Model | State leaves for environment batch `[*B]` |
|---|---|
| GRU | `carry: [*B, L, H]`, floating |
| LSTM | `hidden, cell: [*B, L, H]`, floating |
| GTrXL | `memory: [*B, L, M, D]`, floating; `valid: [*B, M]`, boolean |

The transformer candidate uses a validity mask, including support for holes in
valid memory, rather than a valid-length counter. Memory is ordered oldest to
newest and stores the inputs to each layer. `L`, `M` and `D` are feature
dimensions; environment batch and training time remain TensorDict batch
dimensions. This is a comparison schema, not a selected public API.

`Composite(data_cls=StateClass)` associates the schema with explicit shape,
dtype and device leaf specs. The GTrXL validity leaf uses a Binary spec. No new
spec subclass or global registry was introduced. Primer container paths are
prepared at setup; ordinary TensorDict operations do not gain schema validation.

Implemented compatibility changes:

- Preserve nested classes through primer initialization, partial resets and
  next-state propagation.
- Preserve batch metadata in eager and memoized Composite encoding; use
  tensorclass-compatible access and correctly recurse for dtype checks.
- Preserve the source class when step_mdp writes into an existing destination
  with a different nested container class.
- Support nested TypedTensorDict leaf traversal with custom leaf predicates.
  This is the only TensorDict implementation change, demonstrated by environment
  spec checking and a focused dense/lazy/memmap regression.
- Extend TransformerModule with an experimental explicit-state backbone path
  and primer factory. Existing cache-owning callers continue through their
  original path.

The private GTrXL backbone implements relative attention, reordered layer
normalization and GRU gates following
[Parisotto et al. (2020)](https://proceedings.mlr.press/v119/parisotto20a.html).
It uses fixed-capacity rolling layer memory. Step and window execution share the
same attention horizon. The window implementation is a **sequential reference**;
parallel masked window attention remains release-milestone work.

Root state is the carry before the current observation. The policy writes the
resulting state under `("next", state_key)`. Training starts from the first
stored carry even if its `is_init` is false. At subsequent `is_init` boundaries
it loads the stored carry, including boundaries inserted by SliceSampler.
True episode starts contain the primer's reset state. Initial/boundary memory
is detached while gradients flow through the current window. Parameter updates
do not invalidate or discard caller-owned state: stored activations can be stale
relative to current weights, as with other recurrent replay.

## Measurements

Apple M5 Max, 64 GiB, macOS 26.6.2, Python 3.12, PyTorch 2.11.0, CPU,
one PyTorch thread. The installed TorchRL C++ extension was unavailable.
These are local CPU results; GPU behavior and optimized GTrXL training
throughput are not established by this experiment.

Each of 22 model/container/size combinations was measured three times with
rotating container order. Each operation uses torch.utils.benchmark autorange
for at least 0.25 seconds. Tables report medians of the three run medians.
The raw file retains within-run IQR and sample counts. Collection ranges expose
substantial run-to-run variation; small throughput differences should not be
treated as a ranking.

Collection includes the same mock environment and linear action head for every
container. Training includes sequence cloning, forward and backward, but no
optimizer update. Construction wraps the same tensors without cloning them.
Python allocation peaks are traced separately and exclude native tensor
allocations. Tensor payload sizes are reported separately.

| Configuration | Batch | Layers | Width | Memory length | Training window |
|---|---:|---:|---:|---:|---:|
| Small | 1 | 2 | 16 | 8 for GTrXL | 8 |
| Larger | 32 | 2 | 64 | 64 for GTrXL | 16 |

### Collection and sequence training

| Model / batch | Container | Collect ms | Collect frames/s | Collect run range | Train ms | Train frames/s |
|---|---|---:|---:|---:|---:|---:|
| GRU / 1 | Flat | 2.995 | 2,671 | 2,560–2,867 | 0.671 | 11,931 |
| GRU / 1 | TensorDict | 3.632 | 2,202 | 1,998–2,280 | 0.691 | 11,571 |
| GRU / 1 | TensorClass | 4.065 | 1,968 | 1,908–2,025 | 0.710 | 11,260 |
| GRU / 1 | TypedTensorDict | 3.998 | 2,001 | 1,484–2,477 | 0.701 | 11,416 |
| GRU / 32 | Flat | 6.496 | 78,824 | 57,602–80,333 | 2.377 | 215,364 |
| GRU / 32 | TensorDict | 8.118 | 63,067 | 59,624–65,917 | 2.332 | 219,565 |
| GRU / 32 | TensorClass | 8.617 | 59,419 | 58,179–59,933 | 2.316 | 221,103 |
| GRU / 32 | TypedTensorDict | 8.177 | 62,615 | 55,342–63,209 | 2.355 | 217,423 |
| LSTM / 1 | Flat | 3.035 | 2,636 | 2,594–2,740 | 0.655 | 12,212 |
| LSTM / 1 | TensorDict | 3.785 | 2,114 | 2,062–2,124 | 0.632 | 12,668 |
| LSTM / 1 | TensorClass | 4.197 | 1,906 | 1,627–1,933 | 0.655 | 12,208 |
| LSTM / 1 | TypedTensorDict | 4.189 | 1,910 | 1,746–1,971 | 0.661 | 12,111 |
| LSTM / 32 | Flat | 7.433 | 68,884 | 66,204–69,917 | 2.681 | 190,971 |
| LSTM / 32 | TensorDict | 8.765 | 58,415 | 57,839–58,851 | 2.649 | 193,291 |
| LSTM / 32 | TensorClass | 9.891 | 51,762 | 42,886–52,675 | 2.715 | 188,592 |
| LSTM / 32 | TypedTensorDict | 9.431 | 54,291 | 45,445–54,310 | 2.738 | 187,028 |
| GTRXL / 1 | TensorDict | 6.448 | 1,241 | 661–1,377 | 6.354 | 1,259 |
| GTRXL / 1 | TensorClass | 6.811 | 1,175 | 1,097–1,307 | 6.373 | 1,255 |
| GTRXL / 1 | TypedTensorDict | 6.310 | 1,268 | 1,085–1,282 | 6.282 | 1,273 |
| GTRXL / 32 | TensorDict | 28.267 | 18,113 | 17,923–18,265 | 49.333 | 10,378 |
| GTRXL / 32 | TensorClass | 29.617 | 17,288 | 9,895–17,523 | 49.762 | 10,289 |
| GTRXL / 32 | TypedTensorDict | 29.641 | 17,274 | 14,570–18,615 | 48.739 | 10,505 |

### Construction, access and copying

All entries below are microseconds. Field access uses attributes for typed
containers and string indexing for plain containers; mapping access uses `.get()`.

| Model / batch | Container | Construct | Mapping access | Field access | Spec zero | Partial reset | step_mdp | Policy step |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GRU / 1 | Flat | 27.291 | 0.316 | 0.417 | 31.124 | 13.147 | 5.125 | 108.992 |
| GRU / 1 | TensorDict | 27.769 | 0.320 | 0.420 | 48.127 | 22.665 | 11.940 | 155.837 |
| GRU / 1 | TensorClass | 28.421 | 0.509 | 0.340 | 50.177 | 25.321 | 18.531 | 158.826 |
| GRU / 1 | TypedTensorDict | 29.061 | 0.436 | 0.136 | 52.999 | 23.899 | 15.047 | 177.128 |
| GRU / 32 | Flat | 27.204 | 0.306 | 0.425 | 30.571 | 14.934 | 5.092 | 152.845 |
| GRU / 32 | TensorDict | 27.653 | 0.327 | 0.430 | 46.377 | 22.183 | 11.486 | 182.589 |
| GRU / 32 | TensorClass | 28.590 | 0.496 | 0.334 | 49.560 | 26.103 | 17.360 | 198.960 |
| GRU / 32 | TypedTensorDict | 26.809 | 0.435 | 0.135 | 47.247 | 26.561 | 15.140 | 206.366 |
| LSTM / 1 | Flat | 50.567 | 0.317 | 0.423 | 56.592 | 20.137 | 5.745 | 121.495 |
| LSTM / 1 | TensorDict | 49.661 | 0.315 | 0.424 | 73.673 | 28.973 | 12.635 | 154.361 |
| LSTM / 1 | TensorClass | 51.570 | 0.505 | 0.327 | 77.286 | 33.502 | 19.523 | 169.167 |
| LSTM / 1 | TypedTensorDict | 51.455 | 0.441 | 0.138 | 76.926 | 32.192 | 17.070 | 180.662 |
| LSTM / 32 | Flat | 49.387 | 0.312 | 0.426 | 59.032 | 24.390 | 5.633 | 181.866 |
| LSTM / 32 | TensorDict | 50.437 | 0.315 | 0.424 | 73.188 | 32.487 | 12.508 | 218.019 |
| LSTM / 32 | TensorClass | 52.030 | 0.521 | 0.341 | 78.005 | 38.713 | 20.235 | 243.027 |
| LSTM / 32 | TypedTensorDict | 51.324 | 0.447 | 0.133 | 75.087 | 35.463 | 17.293 | 248.619 |
| GTRXL / 1 | TensorDict | 54.217 | 0.308 | 0.424 | 73.470 | 29.881 | 12.492 | 459.083 |
| GTRXL / 1 | TensorClass | 51.333 | 0.508 | 0.334 | 79.547 | 33.565 | 19.970 | 463.683 |
| GTRXL / 1 | TypedTensorDict | 51.182 | 0.443 | 0.131 | 78.179 | 32.316 | 16.600 | 444.012 |
| GTRXL / 32 | TensorDict | 49.328 | 0.314 | 0.420 | 86.468 | 124.172 | 12.601 | 1359.475 |
| GTRXL / 32 | TensorClass | 51.177 | 0.508 | 0.332 | 89.800 | 126.836 | 19.563 | 1367.542 |
| GTRXL / 32 | TypedTensorDict | 52.252 | 0.461 | 0.136 | 87.151 | 136.231 | 17.134 | 1394.717 |

### Memory

Payload bytes are identical across containers, including the flat RNN reference.
State includes all environments in the batch. Rollout state includes root and
next snapshots. KiB and MiB use powers of 1024.

| Model / batch | One state KiB | Rollout state MiB | Complete rollout MiB | Python peak bytes, flat / TD / TC / TTD |
|---|---:|---:|---:|---|
| GRU / 1 | 0.125 | 0.0020 | 0.0036 | 7,911 / 12,471 / 13,023 / 13,111 |
| GRU / 32 | 16.000 | 0.5000 | 0.7021 | 7,852 / 12,420 / 13,023 / 13,170 |
| LSTM / 1 | 0.250 | 0.0039 | 0.0056 | 8,490 / 13,045 / 13,564 / 13,711 |
| LSTM / 32 | 32.000 | 1.0000 | 1.2021 | 8,439 / 13,045 / 13,717 / 13,762 |
| GTRXL / 1 | 1.008 | 0.0157 | 0.0174 | — / 13,294 / 13,746 / 14,235 |
| GTRXL / 32 | 1,026.000 | 32.0625 | 32.2646 | — / 13,141 / 13,899 / 14,296 |

For the larger GTrXL case, just the root and next states consume **32.0625 MiB**
per 16-step rollout. A single batched state is **1.002 MiB**. Container choice
does not change those tensor allocations. In contrast, the Python allocation
difference during one policy step is around a kilobyte for this workload.


## Dense replay with one state per window

The feasibility probe stores an outer TensorDict with batch `[N]`:

```text
record [N]
  transitions [N, T]      observations, episode flags, per-step outputs
  initial_state [N]       memory [N, L, M, D], valid [N, M]
```

Nested TensorDicts may have additional batch dimensions. LazyTensorStorage and
LazyMemmapStorage therefore allocate a time dimension for transitions and no
time dimension for the initial state. The replay buffer samples outer records
as complete windows. No new storage backend is needed for this representation.

The executable probe uses a nonzero starting history and a real episode reset
inside an 8-step window. All six container/storage combinations preserve the
state class and reproduce the full-snapshot learner outputs with maximum
absolute error **0.0** on this run.

| Actual allocated tensor storage, capacity 4 windows | Full snapshots | One initial state |
|---|---:|---:|
| Root memory only | 32,768 bytes | 4,096 bytes |
| All stored leaves | 69,280 bytes | 7,136 bytes |

The root-memory reduction is exactly **8x**, matching the window length. The
total reduction also removes duplicate next-state snapshots, so it is not a
container-overhead measurement. All three containers use the same tensor bytes
in both tensor and memmap storage.

For this probe the learner broadcasts the starting memory across time as a
zero-stride view and materializes only the small validity mask. Genuine episode
resets invalidate that history. This is safe because whole-window sampling does
not insert artificial slice boundaries. Blindly expanding the first state and
then applying the ordinary slice-boundary loading rule at every reset would
incorrectly restore pre-window history.

| Replay policy | Stored state | Permitted sequence starts | Consequence |
|---|---|---|---|
| Existing trajectories | Every step | Arbitrary stored steps | Existing SliceSampler semantics |
| Whole-window records | Once per window | Stored window starts | Largest simple reduction; no missing-state reconstruction |
| Sparse checkpoints | Every K steps | Checkpoints, or other starts after prefix recomputation | Intermediate starts need extra computation; current-weight reconstruction can differ from the original collected carry |

Whole-window records can still be randomly sampled, shuffled and prioritized.
They need not dictate the optimization minibatch size or replay order. The
restriction is where a training sequence may begin; a shorter prefix may start
at the same saved boundary, but an arbitrary interior offset lacks its carry.
Supporting padded or variable-length records would additionally require lengths
and loss masks. Those production contracts are not implemented by this probe.

The probe compacts **after collection** and proves replay allocation savings
only. Reducing collector peak memory requires saving the initial state before
the rollout and omitting intermediate state leaves before stacking. Likewise,
the current reference backbone still returns full next-state snapshots during
training. Production support should avoid unnecessary materialization there.
Memmap moves storage off heap; it does not eliminate the repeated payload or
its I/O cost. Merely keeping a one-step view into an already stacked allocation
would also fail to release its backing storage.

## Validation and remaining decision

- Full TransformerModule suite on the refreshed stack: **89 passed**, including the existing
  module-owned cache tests and 54 GRU/LSTM/GTrXL lifecycle combinations across
  three containers, single/serial/parallel environments and tensor/memmap replay.
- Composite specs, TestStepMdp and TensorDictPrimer suites on the refreshed
  stack: **3,100 passed, 16 skipped** (CUDA and existing primer exclusions).
- TensorDict typed suite: **67 passed, 1 skipped** (mypy unavailable).
- Dense-window replay probe: **6 passed**, recorded in the JSON artifact.

Coverage includes partial resets, direct intermediate slices and SliceSampler
boundaries, nested keys, multidimensional batches, empty/partial/full memory,
invalid-slot masking, float64/bfloat16/autocast, detached initial memory and
finite gradients through the current window. A regression also checks that
weight synchronization preserves caller-owned history. Explicit-state windows compile
with fullgraph AOT eager; the existing transformer tests also exercise Inductor.
Saved trajectory recomputation checks per-step snapshot values. Collector output
buffers remain reusable: retain a clone or write to replay before advancing the
collector when a durable batch is needed.

An earlier broader environment run found an unrelated installed-gymnasium
mismatch: a pre-existing TestInfoDict test requests HalfCheetah-v5 but the local
environment provides only earlier versions. No dependency upgrade was made.
Formatting, flake8 and whitespace checks pass for the changed Python files.

The comparison candidates and measurements are ready for a container decision.
Public state exports, optimized window attention, production compact collection,
the PPO example, padded training losses and reproducible learning runs have not
been shipped in this milestone. Standard trajectory storage and existing
recurrent defaults remain the baseline.

## Reproduction and artifacts

Use a Python environment with the dependency revisions above, then put both
worktrees on PYTHONPATH. For example, set `TD_WORKTREE` and `RL_WORKTREE` to the
comparison checkouts and run from `RL_WORKTREE`:

```sh
export PYTHONPATH="$TD_WORKTREE:$RL_WORKTREE"
python benchmarks/ad_hoc/bench_recurrent_state.py \
  --repeats 3 --min-run-time 0.25 \
  --output benchmarks/results/recurrent-state-cpu.json
python benchmarks/ad_hoc/probe_recurrent_window_storage.py \
  --output benchmarks/results/recurrent-window-storage.json
python -m pytest test/modules/test_transformer.py -q
python -m pytest test/test_specs.py test/envs/test_step_mdp.py::TestStepMdp -q
python -m pytest test/transforms/test_compose_and_env.py::TestTensorDictPrimer -q
# Run in TD_WORKTREE:
python -m pytest test/tensorclass/test_typedtensordict.py -q
```

- [Raw benchmark results](recurrent-state-cpu.json)
- [Dense-window allocation and parity results](recurrent-window-storage.json)
- [Benchmark script](../ad_hoc/bench_recurrent_state.py)
- [Window storage probe](../ad_hoc/probe_recurrent_window_storage.py)
- [Private candidate fixtures](../../torchrl/testing/_state_candidates.py)
- [Private GTrXL backbone](../../torchrl/modules/models/_gtrxl.py)
