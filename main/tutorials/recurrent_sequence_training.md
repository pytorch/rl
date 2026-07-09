Note

Go to the end
to download the full example code.

# Recurrent training on sequence batches

**Author**: [Achintya Paningapalli](https://github.com/theap06)

 What you will learn

- How TorchRL auto-wires `InitTracker` and the
recurrent-state primer when a recurrent policy is detected
- How to sample multi-step slices from a replay buffer with
[`SliceSampler`](../reference/generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler)
- How to train a recurrent policy in *full-sequence* mode using
[`set_recurrent_mode`](../reference/generated/torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode)
- How [`set_recurrent_mode`](../reference/generated/torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode) is used in
collectors (sequential, `False`) vs. loss modules (recurrent, `True`),
and how to control it across processes and workers
- Why `is_init` at slice starts is what keeps hidden state from
leaking across episode boundaries inside a replay batch

 Prerequisites

- PyTorch v2.0.0
- gymnasium[classic_control]
- Familiarity with [Recurrent DQN](dqn_with_rnn.html#rnn-tuto) is helpful but not
required -- this tutorial is its multi-step complement

```
from __future__ import annotations
```

## Overview

There are two ways to train a recurrent policy in TorchRL, and choosing
between them is the first design decision in any recurrent project:

1. **Sequential mode** (one step per forward call). The policy reads the
previous step's hidden state from the TensorDict, runs the LSTM for one
step, and writes the new hidden state under the `("next", ...)` keys.
This is the natural mode during collection and is what
[Recurrent DQN](dqn_with_rnn.html#rnn-tuto) covers in depth.
2. **Recurrent mode** (full `[B, T]` sequence per forward call). The LSTM
processes a whole time dimension at once. This is used inside loss /
advantage code over replayed trajectory slices, where you want to
backprop through several timesteps in a single batched call. Rectangular
`[B, T]` tensors are one option, but TorchRL also accepts the flat,
ragged slices returned by [`SliceSampler`](../reference/generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler):
the flat dimension is treated as a packed time axis and `is_init`
partitions it into independent chunks.

This tutorial focuses on (2): how to collect data, sample multi-step
slices, and run a recurrent policy in full-sequence mode without leaking
hidden state across episode boundaries.

The key building blocks -- most of which the collector now auto-wires for
you -- are:

- [`LSTMModule`](../reference/generated/torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule) (the recurrent policy core),
- `InitTracker` (writes `is_init=True` at the start
of every trajectory),
- A [`TensorDictPrimer`](../reference/generated/torchrl.envs.transforms.TensorDictPrimer.html#torchrl.envs.transforms.TensorDictPrimer) that seeds the
initial recurrent state on reset,
- [`SliceSampler`](../reference/generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) for trajectory-aware
replay,
- [`set_recurrent_mode`](../reference/generated/torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode) for switching the LSTM
between single-step and full-sequence execution.

See [Recurrent state lifecycle](../reference/recurrent_state_lifecycle.html#ref-recurrent-state-lifecycle) for the full reference on how
hidden state flows through the pipeline.

If you are running this in Google Colab, install the dependencies first:

```
!pip3 install torchrl
!pip3 install gymnasium
```

```
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
from torch import nn

from torchrl.collectors import Collector
from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import SliceSampler
from torchrl.envs import GymEnv
from torchrl.modules import LSTMModule, set_recurrent_mode

torch.manual_seed(0)
device = torch.device("cpu")
```

## Environment and policy

We use `CartPole-v1` for a small, fast, fully-observed env. A recurrent
policy is overkill for CartPole, but that is precisely the point: it lets
us focus on the sequence-batching machinery without the env being the
bottleneck. The same pattern scales unchanged to partially-observed
environments where memory actually matters.

The policy is a tiny [`LSTMModule`](../reference/generated/torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule) followed by a
linear head that maps the LSTM output to the action logits. We pick
`hidden_size=16` so the tutorial finishes quickly on CPU.

```
OBS_DIM = 4 # CartPole observation: cart pos, cart vel, pole angle, pole vel
N_ACTIONS = 2 # left or right
HIDDEN = 16

def make_policy() -> Seq:
 """Construct the recurrent policy: LSTMModule + linear logits head."""
 lstm = LSTMModule(
 input_size=OBS_DIM,
 hidden_size=HIDDEN,
 in_keys=["observation", "rs_h", "rs_c"],
 out_keys=["features", ("next", "rs_h"), ("next", "rs_c")],
 python_based=True, # avoids cuDNN for vmap / torch.compile compatibility
 )
 head = Mod(
 nn.Linear(HIDDEN, N_ACTIONS),
 in_keys=["features"],
 out_keys=["logits"],
 )
 # Deterministic argmax action selection -- enough to demonstrate the
 # collection / replay / sequence-training plumbing.
 chooser = Mod(
 lambda logits: logits.argmax(-1, keepdim=False),
 in_keys=["logits"],
 out_keys=["action"],
 )
 return Seq(lstm, head, chooser)

policy = make_policy()
policy.eval()
```

```
TensorDictSequential(
 module=ModuleList(
 (0): LSTMModule()
 (1): TensorDictModule(
 module=Linear(in_features=16, out_features=2, bias=True),
 device=cpu,
 in_keys=['features'],
 out_keys=['logits'])
 (2): TensorDictModule(
 module=<function make_policy.<locals>.<lambda> at 0x7f52123d44a0>,
 device=cpu,
 in_keys=['logits'],
 out_keys=['action'])
 ),
 device=cpu,
 in_keys=['observation', 'rs_h', 'rs_c', 'is_init'],
 out_keys=['features', ('next', 'rs_h'), ('next', 'rs_c'), 'logits', 'action'])
```

## Auto-wiring recurrent env transforms

A recurrent policy needs two env-side transforms to behave correctly:

1. `InitTracker`, which writes `is_init=True` at
the first step after every reset.
2. A [`TensorDictPrimer`](../reference/generated/torchrl.envs.transforms.TensorDictPrimer.html#torchrl.envs.transforms.TensorDictPrimer) that zero-fills
the initial `("rs_h", "rs_c")` recurrent state slots on reset.

Forgetting either of these is the most common source of silent
hidden-state bugs. The collector now detects recurrent submodules in the
policy and auto-appends both transforms when `auto_register_policy_transforms=True`
is passed. As of v0.15 this becomes the default; for earlier versions you
need to opt in explicitly.

```
env = GymEnv("CartPole-v1", device=device)

collector = Collector(
 env,
 policy,
 frames_per_batch=64,
 total_frames=512,
 device=device,
 storing_device=device,
 auto_register_policy_transforms=True,
 reset_at_each_iter=False,
)
```

## Inspecting a single batch

Run one rollout and look at what comes back. Three things to notice:

- The batch shape is `[T]` where `T == frames_per_batch` (single env).
- `is_init` is `True` at the very first step and immediately after
each `done`. These mark trajectory boundaries.
- `("next", "rs_h")` / `("next", "rs_c")` carry the LSTM's next-step
hidden / cell state across timesteps.

```
data = next(iter(collector))

print("Batch shape:", data.shape)
print("Available keys:", sorted(k for k in data.keys()))
print("is_init shape:", data["is_init"].shape)
print("# trajectory boundaries in batch:", int(data["is_init"].sum().item()))
print(
 "Next-step hidden shape:",
 data["next", "rs_h"].shape,
 "(batch, num_layers, hidden_size)",
)
```

```
Batch shape: torch.Size([64])
Available keys: ['action', 'collector', 'done', 'features', 'is_init', 'logits', 'next', 'observation', 'rs_c', 'rs_h', 'terminated', 'truncated']
is_init shape: torch.Size([64, 1])
# trajectory boundaries in batch: 7
Next-step hidden shape: torch.Size([64, 1, 16]) (batch, num_layers, hidden_size)
```

## Replay with SliceSampler

We store the rollout in a replay buffer and sample fixed-length slices of
trajectories from it. [`SliceSampler`](../reference/generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler)
is trajectory-aware: it uses the boundary information already in the
batch (`("next", "done")` and the collector-written `("collector",
"traj_ids")`) to draw whole sub-trajectories.

See also

[Trajectory boundaries](../reference/data_layout.html#ref-traj-boundaries) documents how the
sampler recovers episode boundaries from these markers, including the
circular-storage subtleties (wraparound, write cursor) that appear once
the buffer is full.

Two design choices in this section:

1. **Slices remain *ragged* by default**. `SliceSampler` returns a flat
batch of concatenated variable-length slices, *not* a rectangular
`[B, T]` tensor. The trajectory boundaries inside that flat batch
are marked by `is_init=True`. This avoids padding waste and matches
how TorchRL's recurrent modules already consume input.
2. **``pad_output=True`` is available but discouraged**. It pads short
slices to `slice_len` and writes a `("collector", "mask")` key for
consumers (e.g. mask-aware losses), but the docstring on SliceSampler
explicitly steers users toward the ragged path when possible.

```
SLICE_LEN = 8
NUM_SLICES = 4

rb = TensorDictReplayBuffer(
 storage=LazyTensorStorage(max_size=1024, device=device),
 sampler=SliceSampler(slice_len=SLICE_LEN, strict_length=False),
 batch_size=SLICE_LEN * NUM_SLICES,
)
rb.extend(data)

sample = rb.sample()
is_init_positions = sample["is_init"].squeeze(-1).nonzero().squeeze(-1).tolist()
print("Sampled batch shape:", sample.shape)
print("is_init True positions (slice starts):", is_init_positions)
```

```
Sampled batch shape: torch.Size([32])
is_init True positions (slice starts): [0, 8, 16, 24]
```

### What `is_init` marks

Each `True` value in the sampled batch's `is_init` flags the first
timestep of a slice (or a trajectory boundary that fell inside a slice
because the underlying episode ended mid-slice). Downstream code -- the
recurrent module's full-sequence forward and GAE / advantage computation --
reads these markers to know where to reset hidden state. Padding masks are a
separate signal: when `pad_output=True` is used, `SliceSampler` writes a
`("collector", "mask")` key for consumers that need to ignore padded
timesteps.

## Sequential mode vs. recurrent mode

[`set_recurrent_mode`](../reference/generated/torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode) has two modes:

- `set_recurrent_mode(False)` (*sequential*): the LSTM processes **one
timestep at a time**, reading `rs_h` / `rs_c` from the TensorDict
and writing the updated state into `("next", "rs_h")` /
`("next", "rs_c")`. This is the natural mode during **collection**:
the collector calls the policy once per environment step.
- `set_recurrent_mode(True)` (*recurrent*): the LSTM processes a **whole
time batch** in a single `nn.LSTM` call, using `is_init` to reset
the hidden state at trajectory boundaries. This is the mode used during
**training** over replayed slices.

**How set_recurrent_mode is managed in collectors and loss modules:**

- During *collection*, the collector calls the policy inside sequential
mode. The default `default_recurrent_mode=False` on
[`LSTMModule`](../reference/generated/torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule) means no explicit context manager
is required -- the LSTM automatically runs step-by-step.
If you need to override this (e.g. in a custom collector), wrap the
policy call with `set_recurrent_mode(False)`.
- All built-in TorchRL **loss modules** wrap their `forward` with
`set_recurrent_mode(True)` automatically, so you never need to set
the mode manually when calling a loss. The same applies to advantage
estimators and value estimators.
- In **multi-process / distributed** settings (e.g.
`MultiaSyncDataCollector`), each worker
process carries its own thread-local recurrent mode. The mode set in
the main process does **not** propagate to worker processes. Workers
always start with the default (sequential, `False`); the training
process sets recurrent mode independently when computing the loss.

**Equivalence of sequential and recurrent mode:**

Running the LSTM step-by-step under `set_recurrent_mode(False)` and
running it in a single batch under `set_recurrent_mode(True)` produce
numerically identical outputs. We verify this now on a short sequence.

```
# Build a minimal hand-crafted 4-step trajectory so the equivalence is
# unambiguous: one trajectory, is_init only at t=0, zero initial hidden.
T_equiv = 4
demo = TensorDict(
 {
 "observation": torch.randn(T_equiv, OBS_DIM),
 "rs_h": torch.zeros(T_equiv, 1, HIDDEN),
 "rs_c": torch.zeros(T_equiv, 1, HIDDEN),
 "is_init": torch.tensor([True, False, False, False]),
 },
 batch_size=[T_equiv],
)

# --- set_recurrent_mode(False): one call per step ---
# The LSTM reads rs_h / rs_c, processes a single timestep, and writes the
# updated state into ("next", "rs_h") / ("next", "rs_c"). We thread the
# hidden state from one step to the next by hand.
outs_seq = []
h = torch.zeros(1, 1, HIDDEN)
c = torch.zeros(1, 1, HIDDEN)
for t in range(T_equiv):
 step = demo[t : t + 1].clone()
 step["rs_h"] = h
 step["rs_c"] = c
 with set_recurrent_mode(False):
 step_out = policy(step)
 outs_seq.append(step_out["features"])
 h = step_out["next", "rs_h"]
 c = step_out["next", "rs_c"]
features_seq = torch.cat(outs_seq, dim=0)

# --- set_recurrent_mode(True): one batched call ---
with set_recurrent_mode(True):
 out_rec = policy(demo.clone())

# They must be numerically identical.
torch.testing.assert_close(features_seq, out_rec["features"], rtol=1e-4, atol=1e-5)
print(
 "set_recurrent_mode(False) step-by-step == set_recurrent_mode(True) batched: verified."
)

# Now run the full replay sample in recurrent mode (the training-time pattern).
with set_recurrent_mode(True):
 out = policy(sample.clone())

print("Recurrent-mode output shape:", out["features"].shape, "(total_steps, hidden)")
print(
 "Recurrent-mode logits shape:",
 out["logits"].shape,
 "(total_steps, n_actions)",
)
```

```
set_recurrent_mode(False) step-by-step == set_recurrent_mode(True) batched: verified.
Recurrent-mode output shape: torch.Size([32, 16]) (total_steps, hidden)
Recurrent-mode logits shape: torch.Size([32, 2]) (total_steps, n_actions)
```

## Why the boundary handling matters: a controlled check

The forward above already ran the interior-split path on real sampled
data. To make the no-leakage guarantee *provable* rather than merely
exercised, we now build a small two-trajectory packed batch by hand, seed
the *first* trajectory with non-zero noise in its incoming hidden, and
check that the second trajectory's outputs match a standalone forward over
just that second trajectory.

If hidden state leaked across the boundary, the noisy first half would
pollute the second half's outputs and the comparison would fail.

```
# Two adjacent slices of length 4 each, packed end-to-end. is_init=True at
# index 0 (first traj start) and index 4 (second traj start).
T_A = T_B = 4
T = T_A + T_B
is_init_packed = torch.zeros(1, T, dtype=torch.bool)
is_init_packed[0, 0] = True
is_init_packed[0, T_A] = True

obs = torch.randn(1, T, OBS_DIM)
# Seed the incoming hidden with noise. The recurrent forward should
# *override* this at every is_init=True position.
noisy_h = torch.randn(1, T, 1, HIDDEN)
noisy_c = torch.randn(1, T, 1, HIDDEN)

packed = TensorDict(
 {
 "observation": obs,
 "rs_h": noisy_h,
 "rs_c": noisy_c,
 "is_init": is_init_packed,
 },
 batch_size=[1, T],
)

# Isolated forward over only the second trajectory, with is_init=True at
# step 0 (its real start). If the packed run handles boundaries correctly,
# the packed batch's second-half output must match this isolated output.
is_init_b = torch.zeros(1, T_B, dtype=torch.bool)
is_init_b[0, 0] = True
b_only = TensorDict(
 {
 "observation": obs[:, T_A:].clone(),
 "rs_h": noisy_h[:, T_A:].clone(), # same noise -- must be overridden
 "rs_c": noisy_c[:, T_A:].clone(),
 "is_init": is_init_b,
 },
 batch_size=[1, T_B],
)

with set_recurrent_mode(True):
 packed_out = policy(packed)
 b_out = policy(b_only)

# Trajectory B's features inside the packed batch == trajectory B alone.
torch.testing.assert_close(
 packed_out["features"][:, T_A:], b_out["features"], rtol=1e-5, atol=1e-6
)
print("Hidden-state isolation across is_init boundary: verified.")
```

```
Hidden-state isolation across is_init boundary: verified.
```

## A tiny gradient-path smoke test

We close with a minimal supervised update on top of the same
infrastructure. We use the simplest possible objective -- match the
action logits to a constant target -- purely to exercise the
`set_recurrent_mode(True)` + replay-buffer + LSTM gradient path. In a real
recurrent training job, this block is where you would compute returns or
advantages and call the objective module, still under the same recurrent
context manager.

```
trainable_policy = make_policy()
optimizer = torch.optim.Adam(trainable_policy.parameters(), lr=3e-4)
target_logits = torch.zeros(N_ACTIONS)
target_logits[0] = 1.0 # arbitrary supervised target

losses = []
for _step in range(4):
 sample = rb.sample()
 with set_recurrent_mode(True):
 out = trainable_policy(sample.clone())
 loss = (out["logits"] - target_logits.expand_as(out["logits"])).pow(2).mean()
 optimizer.zero_grad()
 loss.backward()
 optimizer.step()
 losses.append(loss.item())

collector.shutdown()

print("Training loss trajectory:", [round(v, 4) for v in losses])
```

```
Training loss trajectory: [0.4127, 0.4065, 0.403, 0.4076]
```

## Conclusion

You have built a recurrent training pipeline that:

- Lets the collector auto-wire `InitTracker` and the
recurrent-state primer for you, removing two manual steps that used to
be a frequent source of silent bugs.
- Samples multi-step trajectory slices from a replay buffer with
[`SliceSampler`](../reference/generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler).
- Runs the LSTM in full-sequence mode under
`set_recurrent_mode(True)` and verified by
construction that hidden state does not leak across trajectory
boundaries inside a sampled batch.

This is the canonical TorchRL pattern for recurrent / sequence-based RL.
It composes cleanly with every loss module, every advantage estimator,
and every replay-buffer extension in the library.

## Further reading

- [Recurrent DQN](dqn_with_rnn.html#rnn-tuto) -- the single-step / collection-time
complement to this tutorial.
- `examples/replay-buffers/recurrent_slice_sampler_pipeline.py` -- a
minimal, runnable end-to-end version of this pipeline (GRU policy, the
collector writing directly into the buffer in a background thread, and
[`SliceSampler`](../reference/generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) auto-detecting the
trajectory key).
- [Recurrent state lifecycle](../reference/recurrent_state_lifecycle.html#ref-recurrent-state-lifecycle) -- full reference on how hidden
state flows from collection through replay to the loss, and what
`is_init` controls along the way.
- [Collector Internals](../reference/collectors_internals.html#ref-collectors-internals) -- the per-step rollout flow,
`_carrier` semantics, and how the device-cast flags interact with
recurrent state.
- [Glossary](../reference/glossary.html#ref-glossary) -- short definitions of `is_init`,
`TensorDictPrimer`, `recurrent mode`, `set_keys`, and other
shorthand that appears throughout the recurrent code paths.

**Total running time of the script:** (0 minutes 0.169 seconds)

[`Download Jupyter notebook: recurrent_sequence_training.ipynb`](../_downloads/70bdaea647addd62eb8379f8c256e50b/recurrent_sequence_training.ipynb)

[`Download Python source code: recurrent_sequence_training.py`](../_downloads/c5f0529613de180df8279ab970ead224/recurrent_sequence_training.py)

[`Download zipped: recurrent_sequence_training.zip`](../_downloads/8f7c6500108a26ed094c1f7e410c85ee/recurrent_sequence_training.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)