Note

Go to the end
to download the full example code.

# Memory-Efficient RL Training

**Author**: [Vincent Moens](https://github.com/vmoens)

 What you will learn

- The cost of keeping `("next", obs)` in rollouts and replay buffers
- Why TorchRL keeps it by default (bootstrap targets and MultiStep)
- Halving the observation footprint with
[`Collector`](../reference/generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector) `compact_obs=True`
- Rebuilding `("next", obs)` on the consumer side with
[`NextStateReconstructor`](../reference/generated/torchrl.envs.transforms.NextStateReconstructor.html#torchrl.envs.transforms.NextStateReconstructor)
- Why the resulting `NaN` at trajectory ends does not crash GAE / TD
- The lossy-delta variant with
[`NextObservationDelta`](../reference/generated/torchrl.envs.transforms.NextObservationDelta.html#torchrl.envs.transforms.NextObservationDelta) --
boundary-preserving but smaller saving
- Halving the value-net forward with the budgeted `shifted=True`
backend, which masks dropped samples instead of approximating them
- Capping peak value-net activation memory with the
`value_chunk_size` / `num_chunks` knobs
- Trading collection speed against the training memory budget with
the collector's `env_device` / `policy_device` /
`storing_device` placement
- When *not* to take the compact path (MultiStep DQN, truncated
transitions) -- and why the delta path keeps MultiStep available
- Other knobs: `LazyMemmapStorage`,
`SliceSampler`, the padding-free `"scan"` /
`"triton"` RNN backends, and
[`split_trajectories()`](../reference/generated/torchrl.collectors.utils.split_trajectories.html#torchrl.collectors.utils.split_trajectories) nested mode

 Prerequisites

- [TorchRL](https://github.com/pytorch/rl) and
[gymnasium](https://gymnasium.farama.org) installed
- Familiarity with [`Collector`](../reference/generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector) and
[`ReplayBuffer`](../reference/generated/torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer)
(see [the data-collection tutorial](getting-started-3.html#gs-storage) and
[the replay-buffer tutorial](rb_tutorial.html#rb-tuto))

```
import tempfile

import torch
from tensordict.nn import TensorDictModule
from torchrl.collectors import Collector
from torchrl.data import LazyMemmapStorage, LazyTensorStorage, ReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import GymEnv, TransformedEnv
from torchrl.envs.transforms import NextObservationDelta, NextStateReconstructor
from torchrl.objectives.value import GAE

torch.manual_seed(0)
```

```
<torch._C.Generator object at 0x7f4fb3087ef0>
```

## The problem

Two independent trends have made memory the bottleneck in modern RL:

- **Environments scaled up.** Highly vectorized simulators (MuJoCo XLA /
Warp / the native PyTorch backend, Isaac Lab, Genesis, ...) now produce
transitions at staggering rates, so the sheer *volume* of data flowing
through rollouts and replay buffers explodes.
- **Policies scaled up.** Models routinely run from hundreds of millions
to billions of parameters -- vision-language-action (VLA) policies,
large recurrent critics, transformer world models -- so each forward and
backward pass carries a heavy parameter and activation footprint.

Either trend alone strains memory; together they compound. If one is not
careful, the cost of *acquiring* the data piles up just as fast as the
cost of *training* on it.

There is a fundamental tension here. Fast data collection wants to keep
everything on the accelerator -- simulate on device, run the policy on
device, minimise host/device transfers. Efficient training at scale
wants the opposite: spill the replay buffer to cheaper memory, keep only
what the loss actually reads, and free activations as early as possible.

The pipeline makes this concrete. In a typical PPO loop the same data is
touched by three different consumers: the model runs as a *policy* during
collection, as a *critic* during GAE, and as the *whole loss* (including
the backward pass) during the update. Keeping each tensor on the right
device at the right time, reaching for accelerators only when they pay
off, and moving data quickly between the inference and training halves of
the loop are challenges that have to be addressed deliberately -- they do
not solve themselves.

This tutorial works through a set of independent *knobs* that each
attack one corner of this problem, from what gets stored to what gets
transferred to what gets transiently allocated. Adopt them à la carte.

## Where the memory goes

A typical RL rollout returns a tensordict with both the current
observation (`"observation"`) and the next observation
(`("next", "observation")`). The two overlap by `T - 1` entries
within a trajectory of length `T`: `data["observation"][1:]` is
bit-for-bit equal to `data[("next", "observation")][:-1]`. We are
storing roughly *two copies of every observation*.

Let's measure this directly on a tiny CartPole rollout.

```
env_maker = lambda: GymEnv("CartPole-v1") # noqa: E731
collector = Collector(
 create_env_fn=env_maker,
 frames_per_batch=200,
 total_frames=200,
)

data = next(iter(collector))
collector.shutdown()

total_bytes = data.bytes()
obs_bytes = data.get("observation").numel() * data.get("observation").element_size()
next_obs_bytes = (
 data.get(("next", "observation")).numel()
 * data.get(("next", "observation")).element_size()
)

print(f"Full rollout: {total_bytes:>6d} B")
print(f" observation share: {obs_bytes:>6d} B")
print(f" ('next','observation'): {next_obs_bytes:>6d} B")
print(
 f" duplicated obs: "
 f"{int(next_obs_bytes * (data.shape[-1] - 1) / data.shape[-1]):>6d} B "
 f"(≈ (T-1)/T of the next-obs share)"
)
```

```
Full rollout: 13200 B
 observation share: 3200 B
 ('next','observation'): 3200 B
 duplicated obs: 3184 B (≈ (T-1)/T of the next-obs share)
```

CartPole's 4-dim float observation is small, but the same pattern
applies to vision policies (84×84×3 frames), critic features
(hundreds of dimensions), or LLM hidden states (thousands).
Multiplied by a 10⁶-step replay buffer, the duplication is the
difference between fitting on a single GPU and not.

## Why we keep `("next", obs)` by default

Before we drop anything we should be explicit about what the
duplicated tensor is worth. There are two main consumers:

1. **Bootstrap target at trajectory ends.** TD(0), TD(λ) and GAE all
compute `target = r_t + γ (1 - done_t) V(next_obs_t)`. On *every*
transition we need the canonical next observation -- including the
very last frame of a *truncated* episode, where the bootstrap is
still applied because the trajectory was artificially cut.
2. **MultiStep n-step fallback.**
`MultiStepTransform` places
`data[t + n]` into `data[("next", obs)][t]`. For the last
`n - 1` frames of every trajectory it falls back to
`data[t + n - 1]`, `data[t + n - 2]`, ..., down to `data[t + 1]`
-- and it can only do that because the genuine
`("next", obs)` lives in storage.

Both of these consumers need *information that is not present in
``data["observation"][t + 1]``* once the trajectory ends. That is why
the default is to keep both copies.

## Knob 1 -- drop the duplicates at the collector

If your loss does not depend on a *bootable* terminal next-obs
(vanilla policy-gradient losses, on-policy GAE with terminated-only
transitions, ...), the trade-off flips. The
[`Collector`](../reference/generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector) exposes a
`compact_obs=True` flag that drops every observation / state key
under `("next", ...)` *before* stacking per-step data.
`("next", "reward")`, `("next", "done")` and
`("next", "truncated")` are preserved -- they cannot be reconstructed
from the root keys. The flag works for `MultiSyncCollector` and
`MultiAsyncCollector` too.

```
compact_collector = Collector(
 create_env_fn=env_maker,
 frames_per_batch=200,
 total_frames=200,
 compact_obs=True,
)
compact_data = next(iter(compact_collector))
compact_collector.shutdown()

print(f"Default rollout bytes: {data.bytes():>6d}")
print(f"compact_obs=True bytes: {compact_data.bytes():>6d}")
print(
 f"saving: {data.bytes() - compact_data.bytes():>6d} B "
 f"({100 * (data.bytes() - compact_data.bytes()) / data.bytes():.1f} %)"
)
print()
print("Keys dropped from the rollout:")
print(set(data.keys(True, True)) - set(compact_data.keys(True, True)))
```

```
Default rollout bytes: 13200
compact_obs=True bytes: 10000
saving: 3200 B (24.2 %)

Keys dropped from the rollout:
{('next', 'observation')}
```

The collector queries `env._observation_keys_step_mdp` and
`env._state_keys_step_mdp` to discover *which* keys are duplicated,
so nested obs (`("agents", "pos")`, dict-shaped vision obs, ...) are
handled automatically.

## Knob 2 -- rehydrate at sampling time

Many losses *do* read `("next", obs)` (notably GAE / TD). The
consumer-side counterpart of `compact_obs` is
[`NextStateReconstructor`](../reference/generated/torchrl.envs.transforms.NextStateReconstructor.html#torchrl.envs.transforms.NextStateReconstructor). The rule is
simple: for each sampled position `i`, the canonical next is
position `i + 1` of the same batch *iff* it belongs to the same
trajectory and the trajectory hasn't ended; otherwise the slot is
filled with `NaN` (configurable).

"Same trajectory" is decided from a trajectory id (default
`("collector", "traj_ids")`, which
[`Collector`](../reference/generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector) populates by default)
and `("next", "done")`. The transform is sampler-agnostic -- it does
not require `SliceSampler` -- but
`SliceSampler` is the natural pairing because
adjacent positions inside a slice are also adjacent in trajectory
time.

```
rb = ReplayBuffer(
 storage=LazyTensorStorage(200),
 sampler=SliceSampler(slice_len=20, traj_key=("collector", "traj_ids")),
 transform=NextStateReconstructor(),
 batch_size=40,
)
rb.extend(compact_data)
sample = rb.sample()

# ``("next", "observation")`` is back in the sample, even though it was
# absent from the storage.
print("sample keys:", sorted(str(k) for k in sample.keys(True, True))[:6])
print(
 "any NaN in ('next', observation')?",
 torch.isnan(sample[("next", "observation")]).any().item(),
)
```

```
sample keys: ["('collector', 'traj_ids')", "('next', 'done')", "('next', 'observation')", "('next', 'reward')", "('next', 'terminated')", "('next', 'truncated')"]
any NaN in ('next', observation')? True
```

The NaN entries land exactly where the *real* next observation is no
longer reconstructable -- slice boundaries that coincide with
trajectory ends. We can see them by looking at the rows where the
trajectory id changes (or where the trajectory ended):

```
traj = sample[("collector", "traj_ids")]
done = sample[("next", "done")].squeeze(-1)
boundary = torch.cat([(traj[1:] != traj[:-1]), torch.tensor([True])]) | done
print(
 "rows with NaN next-obs: ",
 torch.isnan(sample[("next", "observation")])
 .any(-1)
 .nonzero(as_tuple=True)[0]
 .tolist(),
)
print(
 "rows flagged as trajectory boundaries: ",
 boundary.nonzero(as_tuple=True)[0].tolist(),
)
```

```
rows with NaN next-obs: [19, 39]
rows flagged as trajectory boundaries: [19, 39]
```

## Knob 3 -- Lossy delta compression, boundary-preserving

`compact_obs` + [`NextStateReconstructor`](../reference/generated/torchrl.envs.transforms.NextStateReconstructor.html#torchrl.envs.transforms.NextStateReconstructor)
is *lossless within a trajectory* but loses information at trajectory
boundaries (the NaN positions above). For tasks that bootstrap on
truncated transitions, or for users who'd rather not propagate `NaN`
at all,
[`NextObservationDelta`](../reference/generated/torchrl.envs.transforms.NextObservationDelta.html#torchrl.envs.transforms.NextObservationDelta) provides a
different trade-off: keep `("next", obs)` information at every step,
but store it at low precision.

The env-side transform writes, for each step,
`("next", "delta", obs) = (next_obs - obs).to(delta_dtype)` (default
`torch.float16`) and drops the full-precision `("next", obs)` from
the post-step tensordict before the collector stacks it. The same
class attached to the replay buffer reconstructs
`("next", obs) = obs + ("next", "delta", obs)` at sample time.

```
env_maker_delta = lambda: TransformedEnv( # noqa: E731
 GymEnv("CartPole-v1"), NextObservationDelta()
)
delta_collector = Collector(
 create_env_fn=env_maker_delta,
 frames_per_batch=200,
 total_frames=200,
)
delta_data = next(iter(delta_collector))
delta_collector.shutdown()

print(f"Default rollout bytes: {data.bytes():>6d}")
print(f"compact_obs=True bytes: {compact_data.bytes():>6d}")
print(f"NextObservationDelta: {delta_data.bytes():>6d}")
print(
 f" delta vs default saving: {data.bytes() - delta_data.bytes():>6d} B "
 f"({100 * (data.bytes() - delta_data.bytes()) / data.bytes():.1f} %)"
)
print()
print("Delta key dtype:", delta_data[("next", "delta", "observation")].dtype)
print(
 "('next', 'observation') in rollout?",
 ("next", "observation") in delta_data.keys(True, True),
)
```

```
Default rollout bytes: 13200
compact_obs=True bytes: 10000
NextObservationDelta: 11600
 delta vs default saving: 1600 B (12.1 %)

Delta key dtype: torch.float16
('next', 'observation') in rollout? False
```

The collector batch carries `("next", "delta", "observation")` at
`float16` and the full-precision `("next", "observation")` is
gone. Root `"observation"` is untouched (full precision) so the
policy can still read it.

Attaching the same class to a replay buffer reconstructs
`("next", "observation")` at sample time. The reconstruction is
elementwise `obs + delta`, so it does *not* depend on the sampler
layout or on trajectory boundaries.

```
rb_delta = ReplayBuffer(
 storage=LazyTensorStorage(200),
 sampler=SliceSampler(slice_len=20, traj_key=("collector", "traj_ids")),
 transform=NextObservationDelta(in_keys=["observation"]),
 batch_size=40,
)
rb_delta.extend(delta_data)
delta_sample = rb_delta.sample()

print(
 "('next', 'observation') in sample?",
 ("next", "observation") in delta_sample.keys(True, True),
)
print(
 "delta key dropped from sample?",
 ("next", "delta", "observation") not in delta_sample.keys(True, True),
)
print(
 "any NaN in reconstructed ('next', observation')?",
 torch.isnan(delta_sample[("next", "observation")]).any().item(),
)
```

```
('next', 'observation') in sample? True
delta key dropped from sample? True
any NaN in reconstructed ('next', observation')? False
```

Compare with the compact-obs path above: there, the same sample
carried `NaN` at every position whose `i + 1` left the trajectory
(slice boundaries, `done` flags). With the delta path the
reconstructed next observation is finite *everywhere*, including the
trajectory ends -- at the cost of storing a half-precision delta per
step instead of nothing.

Trade-offs vs. Knob 1 + Knob 2:

- **Memory.** Smaller saving (~25% of obs bytes for float32→float16,
vs. ~50% for the compact-obs route -- root `obs` is untouched
either way; the half goes into a half-precision delta rather than
disappearing).
- **Boundaries.** The delta encodes the *actual* transition that
happened inside `env.step`, so end-of-trajectory positions
reconstruct correctly within the round-trip precision of
`delta_dtype`. No `NaN`, no need for the value-estimator
sanitizer at Knob 4.
- **Loss compatibility.** The reconstructed `("next", obs)` is
bit-close to the original (subject to `delta_dtype` precision).
Truncated-step bootstraps see the real next obs, not the
`V(obs[t]) ≈ V(real_next_obs)` approximation.
- **MultiStep.** Compatible, unlike the compact-obs path. The delta
keeps the full per-step transition: full-precision root `obs` is
retained at every step and `("next", obs)` reconstructs exactly
everywhere (rewards / dones are never dropped), so the n-step
observation `data[t + n]` is recoverable and n-step returns can be
rebuilt on top of it -- reconstruct `("next", obs)` from the delta
first, then apply
`MultiStepTransform`.
- **Precision.** Lossy. Round-trip error scales with `delta_dtype`
precision and observation magnitude -- best when observations are
roughly normalized.
- **Composition.** `NextObservationDelta` lives *outside* batched
envs (`TransformedEnv(ParallelEnv(N, ...), NextObservationDelta())`);
placing it inside a worker raises at construction time.

Pick this knob when you want bootstrap-correct next-obs at boundaries
without surrendering all the saving, or when `NaN`-propagation
concerns rule out the compact-obs route in your loss pipeline.

## Knob 4 -- value-estimator NaN safety

`NaN` propagating through GAE / TD would be catastrophic:
`V(NaN) = NaN` and the canonical `(1 - done) * V_next` masking
does *not* save us because IEEE 754 has `0 * NaN = NaN`. The
value-estimator pipeline therefore sanitises the input before calling
the value network -- see
`_sanitize_next_obs_nan()`
-- substituting the corresponding root observation at every NaN
position. At *terminated* steps the substitute is masked out
downstream by `(1 - done)`; at *truncated-only* steps it acts as
an approximate bootstrap `V(obs[t]) ≈ V(real_next_obs)`.

The upshot: `compact_obs` + `NextStateReconstructor` + GAE / TD
is numerically safe out of the box.

```
# Tiny value net for illustration: V(s) = ‖s‖₂.
value_net = TensorDictModule(
 lambda x: x.pow(2).sum(-1, keepdim=True).sqrt(),
 in_keys=["observation"],
 out_keys=["state_value"],
)
gae = GAE(gamma=0.99, lmbda=0.95, value_network=value_net, shifted=False)

# Reshape the flat slice batch into (num_slices, slice_len) for GAE.
gae_in = sample.reshape(-1, 20)
out = gae(gae_in.clone())
print("GAE advantage finite everywhere?", torch.isfinite(out["advantage"]).all().item())
print("any -inf or +inf?", torch.isinf(out["advantage"]).any().item())
```

```
GAE advantage finite everywhere? True
any -inf or +inf? False
```

## When *not* to rehydrate

Two situations call for keeping the canonical `("next", obs)`:

1. `MultiStepTransform` *on the
compact path*. The n-step next observation is the *original*
`data[t + n]`, not `data[t + 1]`, and the in-trajectory
fallback at the last `n - 1` frames depends on having every
`data[t + k]` written to `("next", obs)` at extend time.
[`NextStateReconstructor`](../reference/generated/torchrl.envs.transforms.NextStateReconstructor.html#torchrl.envs.transforms.NextStateReconstructor) only
rebuilds the *one-step* next, so it cannot reconstruct that. (The
delta path of Knob 3 *is* MultiStep-compatible -- it keeps the full
per-step transition; see its MultiStep note.)
2. Losses that bootstrap on *truncated* transitions and need the
real next observation, not the
`V(obs[t]) ≈ V(real_next_obs)` approximation that
`_sanitize_next_obs_nan()`
falls back to. The approximation is fine for many tasks (it's
consistent and finite) but it *is* an approximation. For these
cases, prefer
[`NextObservationDelta`](../reference/generated/torchrl.envs.transforms.NextObservationDelta.html#torchrl.envs.transforms.NextObservationDelta) (Knob 3):
it pays a smaller memory saving but reconstructs the real
transition at every position, including trajectory boundaries.

## Knob 5 -- the budgeted `shifted=True` value-net backend

`shifted=True` on the value estimators
([`GAE`](../reference/generated/torchrl.objectives.value.GAE.html#torchrl.objectives.value.GAE),
[`TD0Estimator`](../reference/generated/torchrl.objectives.value.TD0Estimator.html#torchrl.objectives.value.TD0Estimator),
[`TDLambdaEstimator`](../reference/generated/torchrl.objectives.value.TDLambdaEstimator.html#torchrl.objectives.value.TDLambdaEstimator), ...) folds the two
value-net forward passes (one on root obs, one on `("next", obs)`)
into a *single* call, roughly halving value-net compute and the
activation memory it allocates.

As of the budgeted-backend rework, `shifted=True` no longer
silently reuses `("next", obs)` at trajectory ends (the old
`V(obs[t]) ≈ V(real_next_obs)` approximation). Instead it runs the
value net once over a fixed-length `T + shifted_budget` sequence
(default `shifted_budget=1` → `T + 1`): it inserts the *true*
reset next-observation after every internal truncation
(`done & ~terminated`), shifts the following samples one slot to the
right, and marks the displaced suffix that no longer fits in the
budget as invalid via a `"shifted_valid"` mask. Retained samples use
exact next observations -- no approximation.

Two consequences make this a genuine memory knob, not just a compute
one:

- **It tolerates a compact rollout.** Missing `("next", <in_key>)`
entries are filled from the root observation under the same one-step
layout assumption that `compact_obs` relies on
(`("next", obs)[t] == obs[t + 1]` whenever `done[t]` is False),
so `compact_obs=True` + `shifted=True` composes without a
[`NextStateReconstructor`](../reference/generated/torchrl.envs.transforms.NextStateReconstructor.html#torchrl.envs.transforms.NextStateReconstructor) in between.
- **Dropped samples don't bias the loss.** The `"shifted_valid"`
mask is threaded through the loss reductions of
[`PPOLoss`](../reference/generated/torchrl.objectives.PPOLoss.html#torchrl.objectives.PPOLoss),
[`A2CLoss`](../reference/generated/torchrl.objectives.A2CLoss.html#torchrl.objectives.A2CLoss) and
[`ReinforceLoss`](../reference/generated/torchrl.objectives.ReinforceLoss.html#torchrl.objectives.ReinforceLoss), so the few
budget-displaced samples are excluded from the mean rather than
contaminating it.

Raise `shifted_budget` to retain more samples (`2` covers one
internal reset plus the rollout boundary without dropping anything);
the cost is a longer fused sequence. `shifted=True` still requires
identical parameters at `t` and `t + 1` (no distinct target
network) and is unsupported with multi-step returns.

```
gae_shifted = GAE(gamma=0.99, lmbda=0.95, value_network=value_net, shifted=True)
shifted_out = gae_shifted(sample.reshape(-1, 20).clone())
print(
 "shifted GAE advantage finite?",
 torch.isfinite(shifted_out["advantage"]).all().item(),
)
print(
 "shifted_valid mask present?",
 "shifted_valid" in shifted_out.keys(),
)
```

```
shifted GAE advantage finite? True
shifted_valid mask present? True
```

## Knob 6 -- cap peak value-net memory with chunked calls

The two knobs above shrink what is *stored*. The value estimators also
expose a knob for what is *transiently allocated*: the activations of
the value-network forward pass over a large batch. On a deep critic or
a long slice-sampled batch, that single forward can dominate peak GPU
memory even when the stored data is modest.

[`GAE`](../reference/generated/torchrl.objectives.value.GAE.html#torchrl.objectives.value.GAE) (and the other estimators)
accept `value_chunk_size` and `num_chunks` (alias `num_chunk`).
Either one splits the value-network input along the leading dimension
and evaluates the chunks sequentially, trading a little extra Python
overhead for a lower activation high-water mark. The two are mutually
exclusive: give `value_chunk_size` a fixed number of leading-dim
elements per chunk, or `num_chunks` a fixed number of chunks. The
advantage / value targets are identical to the unchunked computation --
only the forward is split.

```
gae_full = GAE(gamma=0.99, lmbda=0.95, value_network=value_net, shifted=False)
gae_chunked = GAE(
 gamma=0.99, lmbda=0.95, value_network=value_net, shifted=False, num_chunks=4
)
adv_full = gae_full(sample.reshape(-1, 20).clone())["advantage"]
adv_chunked = gae_chunked(sample.reshape(-1, 20).clone())["advantage"]
print(
 "chunked advantage matches unchunked?",
 torch.allclose(adv_full, adv_chunked, equal_nan=True),
)
```

```
chunked advantage matches unchunked? True
```

`num_chunks` is the convenient default -- it adapts the chunk size to
whatever batch arrives. Reach for `value_chunk_size` when you want a
fixed, hardware-tuned forward footprint regardless of batch size. The
knob composes with `shifted` and with everything below; it is purely
a peak-memory lever and leaves the numerics untouched.

## Knob 7 -- memory-mapped replay buffer storage

Even after halving the observation footprint, the replay buffer can
easily outgrow VRAM (and RAM). `LazyMemmapStorage`
is a drop-in replacement for `LazyTensorStorage`
that allocates each leaf tensor as a memory-mapped file on disk.
Reading is fast (the OS page cache keeps hot pages in memory), and
the buffer can be larger than physical memory.

```
with tempfile.TemporaryDirectory() as tmpdir:
 rb_mmap = ReplayBuffer(
 storage=LazyMemmapStorage(max_size=1_000, scratch_dir=tmpdir),
 sampler=SliceSampler(slice_len=20, traj_key=("collector", "traj_ids")),
 transform=NextStateReconstructor(),
 batch_size=40,
 )
 rb_mmap.extend(compact_data)
 mmap_sample = rb_mmap.sample()
 print("memmap sample shape:", mmap_sample.shape)
```

```
memmap sample shape: torch.Size([40])
```

The data went through disk, but the public API is identical to the
in-memory case. See the [replay buffer tutorial](rb_tutorial.html#rb-tuto) for
more on storage choices.

## Knob 8 -- keep collection off the training memory budget

This is the "fast collection vs. efficient training" tension from the
intro, made concrete. By default the collector keeps the rollout on
whatever device the policy and env run on -- ideal for *speed* (no
host/device transfers), but every batch then competes with the model's
own parameters and activations for the same VRAM.

[`Collector`](../reference/generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector) lets you place each stage of the
pipeline independently:

- `env_device` -- where the environment simulates. On-device
simulators (Isaac Lab, MuJoCo Warp / XLA, Genesis) want this on the
accelerator.
- `policy_device` -- where the policy network runs during collection.
- `storing_device` -- where the *collected batch* is materialised
before it is yielded or written to the buffer. Setting this to
`"cpu"` evicts each rollout off the accelerator as soon as it is
produced, freeing VRAM for the next rollout and for the trainer -- at
the cost of a device-to-host transfer.
- `device` -- a shorthand that sets all three at once.

```
collector = Collector(
 create_env_fn=env_maker,
 policy=policy,
 frames_per_batch=1024,
 total_frames=1_000_000,
 env_device="cuda", # simulate + act on the accelerator
 policy_device="cuda", # (fast collection)
 storing_device="cpu", # but spill the batch to host RAM
 no_cuda_sync=False, # keep explicit syncs unless transfers
) # are already correctly ordered
```

The rule of thumb: keep `env_device` / `policy_device` on the
accelerator for throughput, and move `storing_device` *down* the
memory hierarchy (accelerator → host RAM → memmap on disk, Knob 7) as
the buffer grows. `no_cuda_sync=True` drops the explicit
synchronisations the collector inserts around cross-device transfers --
safe only when those transfers are already ordered, or on pure CPU. The
exact cast and sync points are documented in
[the collector internals page](../reference/collectors_internals.html#ref-collectors-internals).

## Knob 9 -- sequence training without padding

RNN-based policies and value heads classically train on
zero-padded `(batch, max_T, feature)` tensors, with a mask telling
the loss which timesteps are real. Padding wastes both memory (every
trajectory pays for the longest one) and compute (the RNN unrolls
through the padding tokens).

Two recent additions sidestep both:

- `SliceSampler` returns *contiguous* slices of
pre-specified length. There is no padding; every entry is a real
transition. The trajectory-id key lets the sampler align slices to
trajectory boundaries.
- [`LSTMModule`](../reference/generated/torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule) and
[`GRUModule`](../reference/generated/torchrl.modules.GRUModule.html#torchrl.modules.GRUModule) accept a
`recurrent_backend` argument with three non-default values:

> - `"scan"` -- built on the `hoptorch` scan primitive
> (`pip install hoptorch>=0.1.4`; requires PyTorch ≥ 2.7).
> Resets the hidden state at each `is_init=True` frame inside
> the kernel, so trajectories of different lengths can be
> concatenated end to end with no padding. Designed for
> [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile) friendliness and uses less backward-pass
> activation memory.
> - `"triton"` -- same idea, implemented as a custom Triton
> kernel (requires CUDA and `triton >= 2.2`). Fastest of the
> three on GPU.
> - `"auto"` -- picks `"scan"` under `torch.compile` and
> falls back to the classical `"pad"` path otherwise.

Why this matters for memory: the classical cuDNN `LSTM` / `GRU`
kernels cannot reset the hidden state *in the middle* of a sequence, so
a batch that contains intermediate resets (the common case once you
concatenate trajectories) has to be split at every boundary and
zero-padded to the longest piece. The `"scan"` and `"triton"`
backends reset in place from `is_init` instead, so no split-and-pad
step is needed and no memory is spent on padding tokens. See
[the recurrent state lifecycle guide](../reference/recurrent_state_lifecycle.html#ref-recurrent-state-lifecycle)
for the full reset semantics and the
[recurrent DQN tutorial](dqn_with_rnn.html#rnn-tuto) for an end-to-end example.

A typical configuration looks like this:

```
from torchrl.modules import GRUModule

rnn = GRUModule(
 input_size=64,
 hidden_size=128,
 in_keys=["obs", "rhs"],
 out_keys=["features", ("next", "rhs")],
 recurrent_backend="scan", # or "triton" on CUDA
 default_recurrent_mode=True,
)
```

Combined with `SliceSampler`, the trained
sequence is exactly the concatenation of the slices -- no padding
allocated, no hidden states wasted on zero tokens. The value
estimators are aligned with this: [`GAE`](../reference/generated/torchrl.objectives.value.GAE.html#torchrl.objectives.value.GAE)
(and TD(λ)) consume the flat, contiguous slice layout directly and
never materialise a padded `(batch, max_T)` view, so the advantage
pass stays padding-free as well.

When you *do* need a per-trajectory `(batch, T, ...)` view -- for a
custom loss or analysis -- reach for
[`split_trajectories()`](../reference/generated/torchrl.collectors.utils.split_trajectories.html#torchrl.collectors.utils.split_trajectories) with
`as_nested=True`: it returns a *nested* tensor keyed by trajectory
instead of a zero-padded dense tensor, so ragged trajectory lengths
cost no padding memory. `split_trajectories(data, as_nested=True)`
and the padded form are interchangeable
(`.to_padded_tensor(...)` round-trips between them).

## Putting it together

A memory-conscious value-based pipeline (off-policy actor / critic,
GAE bootstraps, slice-sampled sequence training). Two end-to-end
recipes -- pick the one whose trade-offs match your loss.

**Recipe A -- compact_obs + NextStateReconstructor** (max saving,
`NaN` at boundaries, handled downstream by the value-estimator
sanitizer):

```
collector = Collector(
 create_env_fn=env_maker,
 policy=policy,
 frames_per_batch=1024,
 total_frames=1_000_000,
 compact_obs=True, # halve obs memory
)
rb = ReplayBuffer(
 storage=LazyMemmapStorage(1_000_000), # spill to disk
 sampler=SliceSampler( # no padding
 slice_len=64,
 traj_key=("collector", "traj_ids"),
 ),
 transform=NextStateReconstructor(), # rehydrate ('next', obs)
 batch_size=8 * 64,
)
loss = ClipPPOLoss(actor=actor, critic=critic)
advantage = GAE(
 gamma=0.99, lmbda=0.95,
 value_network=critic,
 shifted=True, # single budgeted V-net call; masks
 # displaced samples via "shifted_valid"
 num_chunks=4, # cap peak value-net activation memory
)
```

**Recipe B -- NextObservationDelta on both sides** (smaller saving,
boundary-preserving, no value-estimator workaround needed):

```
env_maker_delta = lambda: TransformedEnv(
 base_env_maker(), NextObservationDelta(),
)
collector = Collector(
 create_env_fn=env_maker_delta,
 policy=policy,
 frames_per_batch=1024,
 total_frames=1_000_000,
)
rb = ReplayBuffer(
 storage=LazyMemmapStorage(1_000_000),
 sampler=SliceSampler(
 slice_len=64,
 traj_key=("collector", "traj_ids"),
 ),
 transform=NextObservationDelta( # SAME class, RB side
 in_keys=["observation"],
 ),
 batch_size=8 * 64,
)
loss = ClipPPOLoss(actor=actor, critic=critic)
advantage = GAE(
 gamma=0.99, lmbda=0.95,
 value_network=critic, shifted=True,
)
```

Every knob is independent -- adopt them à la carte depending on what
your loss needs. The ones that *interact* are highlighted in the
*When not to rehydrate* section above.

## Conclusion

- `("next", obs)` is a duplicate of `obs[t + 1]` *within* a
trajectory, but it is *not* a duplicate at trajectory boundaries.
That is why TorchRL keeps it by default.
- [`Collector`](../reference/generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector)'s `compact_obs`
flag drops it at the producer side, halving the observation
footprint of rollouts and replay buffers.
- [`NextStateReconstructor`](../reference/generated/torchrl.envs.transforms.NextStateReconstructor.html#torchrl.envs.transforms.NextStateReconstructor) rebuilds
it on the consumer side, with `NaN` at the (genuinely missing)
trajectory ends.
- [`NextObservationDelta`](../reference/generated/torchrl.envs.transforms.NextObservationDelta.html#torchrl.envs.transforms.NextObservationDelta) offers an
alternative: store `("next", obs)` as a low-precision delta
(smaller saving but boundary-preserving and `NaN`-free).
- The value-estimator pipeline keeps GAE / TD targets numerically
defined via
`_sanitize_next_obs_nan()`
when `shifted=False`.
- `shifted=True` is now a budgeted single-call backend: it halves the
value-net forward, tolerates a compact rollout, and masks the few
budget-displaced samples via `"shifted_valid"` (threaded through the
PPO / A2C / Reinforce loss reductions) rather than approximating them.
`shifted_budget` trades sequence length for fewer dropped samples.
- `value_chunk_size` / `num_chunks` cap the *transient* activation
memory of the value-net forward without changing the numerics.
- The collector's `env_device` / `policy_device` / `storing_device`
resolve the collection-speed-vs-training-memory tension: simulate and
act on the accelerator, but spill the stored batch toward host RAM (and
then memmap on disk) as the buffer grows.
- `MultiStepTransform` is the main
loss-side reason to *not* take the *compact* path (Knob 1 + Knob 2);
the delta path (Knob 3) keeps MultiStep available.
- `LazyMemmapStorage`,
`SliceSampler`, the padding-free `"scan"` /
`"triton"` recurrent backends, and
[`split_trajectories()`](../reference/generated/torchrl.collectors.utils.split_trajectories.html#torchrl.collectors.utils.split_trajectories) nested mode
compose orthogonally for further memory wins.

### Useful next resources

- [Replay buffer tutorial](rb_tutorial.html#rb-tuto) -- storage and sampler
choices in depth.
- [Recurrent DQN tutorial](dqn_with_rnn.html#rnn-tuto) -- sequence training with
RNN policies; pair with the `"scan"` / `"triton"` backends for
padding-free training.
- [Recurrent state lifecycle](../reference/recurrent_state_lifecycle.html#ref-recurrent-state-lifecycle) --
what gets auto-wired for recurrent policies and what to check when
sampling sequences for the loss.
- [Trajectory assembly tutorial](collector_trajectory_assembly.html#collector-trajectory-assembly)
-- how collectors lay out trajectory ids, masks, and slices.
- [TorchRL documentation](https://pytorch.org/rl/)

**Total running time of the script:** (0 minutes 0.367 seconds)

[`Download Jupyter notebook: memory_efficient_rl.ipynb`](../_downloads/1dc9c90b893b412af56b0fa1674e3ac1/memory_efficient_rl.ipynb)

[`Download Python source code: memory_efficient_rl.py`](../_downloads/b363ef53772040edf80fe12ff68e0710/memory_efficient_rl.py)

[`Download zipped: memory_efficient_rl.zip`](../_downloads/fb25c8a4208142263170e88cc1bdeff4/memory_efficient_rl.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)