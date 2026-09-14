# GTrXL PPO with fixed-window replay

This reference trains PPO on CartPole with cart and pole velocities hidden. The
policy combines `TransformerModule`, a `GTrXL` backbone, a categorical actor and a
shared value head. It uses the recurrent-mode and primer interfaces of the
transformer stack. The existing module-owned cache API remains available.

Run from the repository root with PyTorch, TorchRL, TensorDict, Gymnasium and
Hydra installed:

```bash
python sota-implementations/gtrxl/gtrxl_ppo.py
bash sota-check/run_gtrxl_ppo.sh
```

The second command runs seeds 0, 1 and 2, recording configuration, evaluation
returns, losses, training timings and memory payloads in separate JSON files.
Evaluation reports the first episode on each of four fresh seeded streams, using
deterministic actions. These small runs check learning behavior, not paper-level
benchmark reproduction. See [measured results](../../benchmarks/results/gtrxl-learning.md).

For a quick smoke test that exercises padding, minibatches and disk storage:

```bash
python sota-implementations/gtrxl/gtrxl_ppo.py \
  env.num_envs=2 collector.steps_per_batch=9 collector.total_frames=36 \
  replay.window_length=4 replay.batch_size=3 replay.storage=memmap \
  loss.epochs=2 logger.eval_envs=2
```

The example runs on CPU to keep Gym environments, policy and replay colocated.
`network.container=td`, `tc` or `ttd` selects identical plain TensorDict,
TensorClass or TypedTensorDict state payloads. The example defaults to `ttd`;
this is an experimental configuration, not a change to GRU/LSTM defaults or a
final decision on the public recurrent-state container. The draft currently
requires the TensorDict prerequisites pinned by the stack's CI installers.

## Inputs, outputs and state

The collector's step TensorDict has batch `[N]` for `N` environments:

| Entry | Shape | Meaning |
|---|---|---|
| `observation` | `[N, 2]` | Cart position and pole angle |
| `state.memory` | `[N, L, M, D]` | Inputs to each layer at the preceding `M` steps |
| `state.valid` | `[N, M]`, boolean | Usable memory slots |
| `is_init` | `[N, 1]`, boolean | Real episode start |
| `features` | `[N, D]` | Transformer output for both heads |
| `next.state` | Same state shapes | Carry after the observation |

Memory is ordered oldest to newest. `L`, `M` and `D` are feature dimensions;
the state container's batch size is `[N]`. `InitTracker` and
`transformer.make_tensordict_primer()` register and reset that state. Its spec is
an ordinary `Composite(data_cls=...)` with explicit memory and validity leaf specs.

## Replay indexed by windows

Collection yields `[N, rollout_steps]`. Advantages and value targets are computed
on that rollout before it is divided into fixed-length windows. Each replay
record has outer batch `[B]` and two children:

```text
initial_state: batch [B]
    memory: [B, L, M, D]        positions -M, ..., -1
    valid:  [B, M]
transitions: batch [B, T]
    observation: [B, T, 2]     positions  0, ..., T-1
    action, reward, targets, is_init, collector.mask, ...
```

Here `B` counts windows, not transitions. Both `LazyTensorStorage` and
`LazyMemmapStorage` allocate along that outer batch axis. `SamplerWithoutReplacement`
selects whole records for each PPO epoch. There is no time flattening or
`SliceSampler` in this path. The replay buffer is emptied after each rollout,
so PPO trains only on the current batch.

For `S = B*T` transitions, memory payload is `B*L*M*D` floating elements instead
of `S*L*M*D`, a factor of `T` saving, plus omission of next-state duplicates.
With the default 16 streams, 128 rollout steps, 32-step windows, two layers,
16 memory positions and width 32 in float32, the root carry payload falls from
8 MiB to 256 KiB. Observations and targets still have their normal per-step cost.

Sampling flexibility pays for this saving: a new arbitrary time offset needs a
carry that was discarded. Window length and alignment are fixed when packing.
To use arbitrary contiguous slices, retain per-step states and use the stack's
ordinary recurrent path, which honors stored carries at `SliceSampler`'s
synthetic starts and directly sliced windows whose first `is_init` is false.

## Learner behavior and limitations

`TransformerModule` receives a `[B]` record with observations `[B,T,2]`, markers
`[B,T,1]` and one `[B]` state. Under `set_recurrent_mode(True)`, GTrXL computes
parallel relative attention with the same `M`-step causal horizon as collection,
and returns `[B,T,D]` features plus one final state. It does not create a state
snapshot for every training timestep. The features enter the standard PPO heads
and `ClipPPOLoss`; actor and critic gradients both train the shared backbone.

Every compact-window `is_init` is a real episode reset. Supplied memory is
detached; gradients flow through recomputed activations within the window.
Weight updates leave caller-owned state intact. Stored activations may therefore
be stale, as with recurrent PPO using saved GRU/LSTM carries.

Short final windows are padded after removing the large states. The collector
mask excludes padding from loss reductions, and advantage normalization precedes
padding. Do not resume an actor from a padded window's final state.

This version packs **after standard collection**: it saves replay allocation and
compact learner state, but the collector still holds per-step snapshots before
packing. This limitation matters for large `M`, `D` and rollout lengths. Training
attention/activations also have their own cost. The
[window benchmark](../../benchmarks/ad_hoc/bench_gtrxl_windows.py) measures training
latency separately from carry payload; container overhead remains covered by the
[comparison benchmark](../../benchmarks/ad_hoc/bench_recurrent_state.py).

See the [transformer tutorial](../../tutorials/sphinx-tutorials/transformer_policies.py)
for executable API examples and the
[GTrXL paper](https://proceedings.mlr.press/v119/parisotto20a.html) for the architecture.
