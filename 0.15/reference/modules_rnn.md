# Recurrent modules

TorchRL recurrent modules wrap PyTorch RNNs in TensorDict-aware modules.
[`LSTMModule`](generated/torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule) and [`GRUModule`](generated/torchrl.modules.GRUModule.html#torchrl.modules.GRUModule) read observations and recurrent
state entries from a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), write features and
`("next", ...)` recurrent states back to it, and use the `is_init` key to
reset hidden states at trajectory boundaries.

The recurrent modules are designed for the contiguous trajectory layout
described in [Data layout: contiguous trajectories](data_layout.html#data-layout): replay buffers and samplers can return flat
1-D slices, and the modules recover the sequence boundaries from `is_init`
instead of requiring padded `[B, T]` tensors and masks.

## Execution modes

By default, recurrent modules run in single-step mode. This is the mode used
during environment interaction: the input TensorDict contains one step per
environment, the previous recurrent state is read from the TensorDict, and the
next recurrent state is written under `("next", ...)`.

During training, wrap the recurrent policy with
[`set_recurrent_mode`](generated/torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode) to process complete rollouts or replay-buffer
slices:

```
from torchrl.modules import GRUModule, set_recurrent_mode

gru = GRUModule(
 input_size=4,
 hidden_size=64,
 in_keys=["observation", "recurrent_state", "is_init"],
 out_keys=["features", ("next", "recurrent_state")],
)

with set_recurrent_mode("recurrent"):
 batch = gru(batch)
```

In recurrent mode, every `is_init=True` entry resets the hidden state to the
state stored at that same position. This lets a flat batch of concatenated
trajectory slices behave like independent sequences without materializing
padding.

## Backend selection

The `recurrent_backend` constructor argument controls how recurrent-mode
calls handle resets inside a batch.

`"pad"`

Splits trajectories on `is_init`, pads them to a common length, and uses
PyTorch's cuDNN-backed [`torch.nn.LSTM`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM) or [`torch.nn.GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU).
This is the default and the broadest compatibility path.

`"scan"`

Uses a scan over the time dimension and avoids padded trajectory chunks.
This is friendlier to [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile) for reset-heavy RL batches.
Supports unidirectional GRU/LSTM without dropout, and (for LSTM) without
projections. Unsupported configurations raise when the recurrent path is
executed.

`"triton"`

Uses TorchRL's fused Triton kernels for reset-aware GRU/LSTM recurrence.
This backend is CUDA-only and requires a recent Triton installation. It is
intended for reset-heavy recurrent RL training where split/pad overhead is
significant. Multilayer unidirectional modules (including dropout between
layers) are handled directly; unsupported variants -- bidirectional modules
and LSTM projections -- silently fall back to the pad semantics.

`"auto"`

Uses `"pad"` in eager mode and `"scan"` when called under
[`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile).

For long-running experiments, prefer choosing a backend explicitly once the
model shape and deployment target are known. `"pad"` is the safest baseline,
`"scan"` is the compile-friendly baseline, and `"triton"` is the
performance-oriented CUDA backend.

## Optimized GRU scan backward

For [`GRUModule`](generated/torchrl.modules.GRUModule.html#torchrl.modules.GRUModule), selecting `recurrent_backend="scan"` also selects a
specialized first-order backward pass when `recurrent_recompute="none"`
(the default). The input projection is evaluated over the flattened
batch-time dimensions, the reverse scan carries only the hidden-state
gradient, and input and parameter gradients are reduced outside the recurrent
loop. This avoids carrying the ordinary autograd graph through every timestep
while preserving the same module parameters and recurrent-state semantics.

No separate optimization flag is required:

```
gru = GRUModule(
 input_size=4,
 hidden_size=64,
 recurrent_backend="scan",
 in_keys=["observation", "recurrent_state", "is_init"],
 out_keys=["features", ("next", "recurrent_state")],
)
```

The optimized backward supports ordinary first-order training, including
autocast BF16 inputs with FP32 parameters. It intentionally does not support
double backward or [`torch.func`](https://docs.pytorch.org/docs/stable/func.api.html#module-torch.func) transforms such as `jacrev`, `vmap`,
and `grad`. Set `recurrent_recompute="full"` to use the checkpointed
reference loop when lower saved-activation memory is more important than the
specialized backward.

Backend performance depends on the device, batch size, rollout horizon, and
hidden width. From a TorchRL source checkout, use the
[recurrent backward benchmark](https://github.com/pytorch/rl/blob/main/benchmarks/bench_rnn_backward.py)
to compare the available implementations on the target hardware:

```
python benchmarks/bench_rnn_backward.py --rnn gru \
 --backends cudnn,scan,triton \
 --batches 256,1024 --seq-lens 64,512 --hiddens 128,512 \
 --warmup 10 --iters 30
```

The benchmark reports synchronized forward, backward, and total times plus
peak allocated CUDA memory for every requested shape.

## Triton precision controls

The Triton backend performs hidden-to-hidden recurrent matrix multiplications
inside Triton kernels and input-to-hidden projections through PyTorch/cuBLAS.
The `recurrent_matmul_precision` argument keeps those paths aligned.

Supported values are:

`"auto"`

Defer to the process-wide TorchRL setting, and fall back to
[`torch.get_float32_matmul_precision()`](https://docs.pytorch.org/docs/stable/generated/torch.get_float32_matmul_precision.html#torch.get_float32_matmul_precision) if the global is itself
`"auto"`. The `TORCHRL_RNN_PRECISION` environment variable seeds the
process-wide setting at import time. It is not consulted at every kernel
call; call [`set_recurrent_matmul_precision()`](generated/torchrl.modules.set_recurrent_matmul_precision.html#torchrl.modules.set_recurrent_matmul_precision) with `"auto"` or
`None` to re-read it after import.

`"ieee"`

Use IEEE FP32 matmuls (~23 bits of mantissa, CUDA cores, no tensor
cores). This is the most conservative setting and is useful for numerical
comparisons with the scan backend.

`"tf32"`

Use TF32 tensor cores on Ampere or newer NVIDIA GPUs (~10 bits of
mantissa, highest throughput).

`"tf32x3"`

Use Triton's three-product TF32 decomposition for the recurrent matmul
(~22 bits of mantissa on tensor cores). cuBLAS has no `tf32x3` mode, so
the input-to-hidden projection stays IEEE FP32. Useful when long rollouts
make recurrent precision drift visible.

`"fast"` and `"high-prec"`

GPU-aware presets. On TF32-capable NVIDIA GPUs, `"fast"` resolves to
`"tf32"` and `"high-prec"` resolves to `"tf32x3"`. On devices
without TF32 tensor cores, both resolve to `"ieee"`.

The process-wide default can be changed with
[`set_recurrent_matmul_precision()`](generated/torchrl.modules.set_recurrent_matmul_precision.html#torchrl.modules.set_recurrent_matmul_precision):

```
from torchrl.modules import set_recurrent_matmul_precision

set_recurrent_matmul_precision("high-prec")
gru = GRUModule(
 input_size=4,
 hidden_size=64,
 recurrent_backend="triton",
 recurrent_matmul_precision="auto",
 in_keys=["observation", "recurrent_state", "is_init"],
 out_keys=["features", ("next", "recurrent_state")],
)
```

A module-level `recurrent_matmul_precision=...` value takes precedence over
the process-wide setting. Use [`get_recurrent_matmul_precision()`](generated/torchrl.modules.get_recurrent_matmul_precision.html#torchrl.modules.get_recurrent_matmul_precision) to inspect
the resolved concrete mode for the current device.

## Transformer temporal policies

[`TransformerModule`](generated/torchrl.modules.TransformerModule.html#torchrl.modules.TransformerModule) extends the same contract to causal transformers:
observations are read from the TensorDict, features written back, and the
`is_init` key drives state resets. Collection runs one step at a time
against a key/value cache, while training processes `[B, T]` windows under
a block-diagonal causal mask so attention never crosses an episode boundary.
The two paths share parameters and produce matching outputs.

```
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
from torchrl.envs import GymEnv, InitTracker, TransformedEnv
from torchrl.modules import TransformerModule, set_recurrent_mode

env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
transformer = TransformerModule(
 input_size=3,
 hidden_size=64,
 num_layers=2,
 num_heads=4,
 max_seq_len=256,
 in_key="observation",
 out_key="features",
)
policy = TensorDictSequential(
 transformer,
 TensorDictModule(nn.Linear(64, 1), in_keys=["features"], out_keys=["action"]),
)

rollout = env.rollout(100, policy) # cached steps, no state in the rollout
with set_recurrent_mode(True):
 window = transformer(rollout.exclude("features")) # same features
```

Unlike the RNN modules, no state travels in the TensorDict. The key/value
cache is inference state owned by the module instance: the backbone allocates
it on the first cached step in the dtype of its projections, one stream per
batch position, and the module clears the streams flagged by `is_init`,
restarts every stream when the parameters change, and releases the cache on
[`reset_cache()`](generated/torchrl.modules.TransformerModule.html#torchrl.modules.TransformerModule.reset_cache); copies and pickled
instances start with an empty cache. TorchRL's weight-synchronization paths
(collectors and the inference server) notify the module through
[`mark_weight_update()`](generated/torchrl.modules.TransformerModule.html#torchrl.modules.TransformerModule.mark_weight_update) once new
weights are applied; call it yourself after updating parameters by other
means. Under autocast the cache is allocated in the compute dtype, so no
conversion happens on the hot path. Rollouts and replay buffers never carry
a cache, whatever the context length. Use one module instance per collector
(or per collector worker); batches whose composition changes between calls
are not supported yet.

Training windows must be episode-aligned: every row must start with
`is_init=True`, which complete-trajectory sampling provides, and a window
that starts mid-episode raises an error. The check is data-dependent, so it
costs one graph break under [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile); pass
`validate_windows=False` to compile the window path as a single graph and
take responsibility for alignment. Episode boundaries inside a window
are recovered from `is_init` through [`positions_from_is_init()`](generated/torchrl.modules.positions_from_is_init.html#torchrl.modules.positions_from_is_init) and
[`segment_causal_mask_from_is_init()`](generated/torchrl.modules.segment_causal_mask_from_is_init.html#torchrl.modules.segment_causal_mask_from_is_init). Any backbone honoring the
[`CausalTransformer`](generated/torchrl.modules.CausalTransformer.html#torchrl.modules.CausalTransformer) contract (`forward`, `new_kv_cache` and
`reset_kv_cache` plus the `num_layers`, `num_heads`, `head_dim` and
`max_seq_len` attributes) can be passed via the `transformer` argument;
the cache object is opaque to the module, so an adapter over an inference
engine can keep it in the engine's own representation.

## Choosing a layout and backend

For most recurrent RL pipelines:

- Use [`InitTracker`](generated/torchrl.envs.transforms.InitTracker.html#torchrl.envs.transforms.InitTracker) or pass the policy to the
env/collector so that TorchRL adds the `is_init` key and recurrent-state
primers automatically.
- Store replay data in the flat contiguous layout and sample with
[`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler).
- Run collection in single-step mode and training under
[`set_recurrent_mode`](generated/torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode).
- Start with `recurrent_backend="pad"` for correctness, then benchmark
`"scan"` or `"triton"` for the target hardware.

## See also

- [Data layout: contiguous trajectories](data_layout.html#data-layout) for the contiguous trajectory layout and replay-buffer
handoff.
- [Recurrent state lifecycle](recurrent_state_lifecycle.html#ref-recurrent-state-lifecycle) for
the primer / `auto_register_policy_transforms` collection path.
- [`LSTMModule`](generated/torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule) and [`GRUModule`](generated/torchrl.modules.GRUModule.html#torchrl.modules.GRUModule) for constructor arguments and
examples.
- [`set_recurrent_mode`](generated/torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode) for switching between single-step and recurrent
execution.
- [`set_recurrent_matmul_precision()`](generated/torchrl.modules.set_recurrent_matmul_precision.html#torchrl.modules.set_recurrent_matmul_precision) and
[`get_recurrent_matmul_precision()`](generated/torchrl.modules.get_recurrent_matmul_precision.html#torchrl.modules.get_recurrent_matmul_precision) for Triton precision control.