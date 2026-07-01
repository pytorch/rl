# LSTMModule

*class*torchrl.modules.LSTMModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#LSTMModule)

An embedder for an LSTM module.

This class adds the following functionality to [`torch.nn.LSTM`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM):

- Compatibility with TensorDict: the hidden states are reshaped to match
the tensordict batch size.
- Optional multi-step execution: with torch.nn, one has to choose between
[`torch.nn.LSTMCell`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTMCell.html#torch.nn.LSTMCell) and [`torch.nn.LSTM`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM), the former being
compatible with single step inputs and the latter being compatible with
multi-step. This class enables both usages.

After construction, the module is *not* set in recurrent mode, ie. it will
expect single steps inputs.

If in recurrent mode, it is expected that the last dimension of the tensordict
marks the number of steps. There is no constrain on the dimensionality of the
tensordict (except that it must be greater than one for temporal inputs).

Note

This class can handle multiple consecutive trajectories along the time dimension
*but* the final hidden values should not be trusted in those cases (ie. they
should not be reused for a consecutive trajectory).
The reason is that LSTM returns only the last hidden value, which for the
padded inputs we provide can correspond to a 0-filled input.

Parameters:

- **input_size** - The number of expected features in the input x
- **hidden_size** - The number of features in the hidden state h
- **num_layers** - Number of recurrent layers. E.g., setting `num_layers=2`
would mean stacking two LSTMs together to form a stacked LSTM,
with the second LSTM taking in outputs of the first LSTM and
computing the final results. Default: 1
- **bias** - If `False`, then the layer does not use bias weights b_ih and b_hh.
Default: `True`
- **dropout** - If non-zero, introduces a Dropout layer on the outputs of each
LSTM layer except the last layer, with dropout probability equal to
`dropout`. Default: 0
- **python_based** - If `True`, will use a full Python implementation of the LSTM cell. Default: `False`
- **recurrent_backend** - backend used in recurrent mode when trajectories reset
in the middle of a batch. `"pad"` keeps the existing split/pad
strategy. `"scan"` uses a scan loop over the time dimension and
avoids materializing padded trajectory chunks via `hoptorch`.
`"triton"`
(prototype, CUDA only) uses Triton kernels where available and
otherwise preserves pad-backend recurrent semantics for dropout,
projections and bidirectional layers. `"auto"` uses `"pad"`
in eager mode and `"scan"` when called under
[`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile). Default: `"pad"`.
- **recurrent_compute_dtype** - dtype used for the recurrent matmul inside the
`"triton"` backend (`torch.float32` -> TF32 on H100, default;
`torch.bfloat16` -> bigger SMEM margin, lower precision).
Ignored by the other backends. Default: `torch.float32`.
- **recurrent_recompute** - when set to `"full"`, trade extra compute for
lower backward activation memory. For `recurrent_backend="triton"`
this drops the per-step gate buffers (`save_i/f/g/o/save_tanhc`)
from the autograd save set and replays the forward kernel during
backward. For `recurrent_backend="scan"` this swaps the
`torch._higher_order_ops.scan` HOP for a python time-loop wrapped
with [`torch.utils.checkpoint.checkpoint()`](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint); gradients then
match the `"pad"` (cuDNN) backend to float precision. Only
`"none"` (default) and `"full"` are accepted today; the
`"pad"` backend rejects non-`"none"` values because cuDNN
manages its own backward workspace. Default: `"none"`.
- **recurrent_matmul_precision** - precision used by `tl.dot` inside the
`"triton"` backend's recurrent matmul (and the matching cuBLAS
calls in the autograd wrapper). Concrete modes: `"ieee"` (full
IEEE FP32, off tensor cores), `"tf32"` (matches cuDNN's
default, fastest on Ampere+), `"tf32x3"` (three-product
compensated TF32, ~22 bits of mantissa on tensor cores).
GPU-aware presets: `"fast"` (Ampere+ → `"tf32"`, else
`"ieee"`) and `"high-prec"` (Ampere+ → `"tf32x3"`, else
`"ieee"`). Or `"auto"` to derive from
[`torch.get_float32_matmul_precision()`](https://docs.pytorch.org/docs/stable/generated/torch.get_float32_matmul_precision.html#torch.get_float32_matmul_precision) and the
`TORCHRL_RNN_PRECISION` env var (`"highest"` → `"ieee"`,
`"high"` → `"high-prec"`, `"medium"` → `"fast"`). See
[`torchrl.modules.set_recurrent_matmul_precision()`](torchrl.modules.set_recurrent_matmul_precision.html#torchrl.modules.set_recurrent_matmul_precision). Ignored
by the other backends. Default: `"auto"`.

Keyword Arguments:

- **in_key** (*str**or**tuple**of**str*) - the input key of the module. Exclusive use
with `in_keys`. If provided, the recurrent keys are assumed to be
["recurrent_state_h", "recurrent_state_c"] and the `in_key` will be
appended before these.
- **in_keys** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**str*) - a triplet of strings corresponding to the input value,
first and second hidden key. Exclusive with `in_key`.
- **out_key** (*str**or**tuple**of**str*) - the output key of the module. Exclusive use
with `out_keys`. If provided, the recurrent keys are assumed to be
[("next", "recurrent_state_h"), ("next", "recurrent_state_c")]
and the `out_key` will be
appended before these.
- **out_keys** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**str*) -

a triplet of strings corresponding to the output value,
first and second hidden key.

Note

For a better integration with TorchRL's environments, the best naming
for the output hidden key is `("next", <custom_key>)`, such
that the hidden values are passed from step to step during a rollout.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**compatible*) - the device of the module.
- **lstm** ([*torch.nn.LSTM*](https://docs.pytorch.org/docs/stable/generated/torch.nn.LSTM.html#torch.nn.LSTM)*,**optional*) - an LSTM instance to be wrapped.
Exclusive with other nn.LSTM arguments.
- **default_recurrent_mode** (*bool**,**optional*) - if provided, the recurrent mode if it hasn't been overridden
by the [`set_recurrent_mode`](torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode) context manager / decorator.
Defaults to `False`.

Variables:

**recurrent_mode** - Returns the recurrent mode of the module.

set_recurrent_mode()[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#LSTMModule.set_recurrent_mode)

controls whether the module should be executed in
recurrent mode.

make_tensordict_primer()[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#LSTMModule.make_tensordict_primer)

creates the TensorDictPrimer transforms for the environment to be aware of the
recurrent states of the RNN.

Note

This module relies on specific `recurrent_state` keys being present in the input
TensorDicts. To generate a [`TensorDictPrimer`](torchrl.envs.transforms.TensorDictPrimer.html#torchrl.envs.transforms.TensorDictPrimer) transform that will automatically
add hidden states to the environment TensorDicts, use the method `make_tensordict_primer()`.
If this class is a submodule in a larger module, the method `get_primers_from_module()` can be called
on the parent module to automatically generate the primer transforms required for all submodules, including this one.

Examples

```
>>> from torchrl.envs import TransformedEnv, InitTracker
>>> from torchrl.envs import GymEnv
>>> from torchrl.modules import MLP, LSTMModule
>>> from torch import nn
>>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
>>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
>>> lstm_module = LSTMModule(
... input_size=env.observation_spec["observation"].shape[-1],
... hidden_size=64,
... in_keys=["observation", "rs_h", "rs_c"],
... out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")])
>>> mlp = MLP(num_cells=[64], out_features=1)
>>> policy = Seq(lstm_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
>>> policy(env.reset())
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 intermediate: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
 is_init: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 rs_c: Tensor(shape=torch.Size([1, 64]), device=cpu, dtype=torch.float32, is_shared=False),
 rs_h: Tensor(shape=torch.Size([1, 64]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)
```

*property*canonical_keys*: list[NestedKey]*

Return TensorDict keys whose canonical layout matters for this module.

The result is the union of `self.in_keys` and `self.out_keys` -
the minimal subset a caller needs to canonicalize before invoking the
module, so unrelated leaves (rewards, advantages, log-probs, ...) can
keep whatever layout the data pipeline produces.

See also

`canonicalize()`,
[`canonicalize_rnn_subset()`](torchrl.modules.canonicalize_rnn_subset.html#torchrl.modules.canonicalize_rnn_subset).

canonicalize(*data: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *inplace: bool = False*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#LSTMModule.canonicalize)

Canonicalize only the RNN-relevant leaves of `data`.

Equivalent to `data.contiguous(canonical=True)` restricted to
`canonical_keys`. Other leaves are left untouched, avoiding the
transient full-batch copy a top-level canonicalization would create.

Parameters:

- **data** - TensorDict to canonicalize. Missing keys in
`canonical_keys` are skipped silently.
- **inplace** - When `True`, mutates `data` in place and returns it.
Defaults to `False` (returns a shallow copy with the
canonicalized leaves replaced).

Returns:

A TensorDict with canonical layout on the RNN keys.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.modules import LSTMModule
>>> module = LSTMModule(input_size=3, hidden_size=4, in_key="obs",
... out_key="out")
>>> td = TensorDict(
... {"obs": torch.zeros(2, 5, 3),
... "reward": torch.zeros(2, 5, 1)},
... batch_size=[2, 5],
... )
>>> td_canon = module.canonicalize(td)
>>> td_canon["obs"].is_contiguous()
True
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*)[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#LSTMModule.forward)

Run the LSTM on a tensordict, honouring `is_init` for hidden-state resets.

Two execution paths, picked by `recurrent_mode`:

- **Sequential** (`recurrent_mode=False`): one step at a time, called
inside a collector rollout. Batch is flattened, a synthetic time dim
of size 1 is added, and `is_init` *zeros the incoming hidden* so
a fresh trajectory does not inherit the previous one's state
(see the `torch.where` block below).
- **Recurrent** (`recurrent_mode=True`): a full `(B, T, ...)`
batch is processed at once, called inside loss / GAE / training
code. If any `is_init[..., 1:]` is set we have multiple
trajectories packed into the time dim; we split-and-pad along
trajectory boundaries (via `_split_and_pad_sequence`) so each
chunk has a single trajectory, run the LSTM, then unpad. This is
what prevents hidden state from leaking *across* trajectories
within a single training batch.

`is_init` is sourced from `InitTracker` on the
env side; without that transform there is no signal for boundary
resets and hidden state will silently leak across episodes.

make_cudnn_based() → LSTMModule[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#LSTMModule.make_cudnn_based)

Transforms the LSTM layer in its CuDNN-based version.

Returns:

self

make_python_based() → LSTMModule[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#LSTMModule.make_python_based)

Transforms the LSTM layer in its python-based version.

Returns:

self

make_tensordict_primer()[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#LSTMModule.make_tensordict_primer)

Makes a tensordict primer for the environment.

A `TensorDictPrimer` object will ensure that the policy is aware of the supplementary
inputs and outputs (recurrent states) during rollout execution. That way, the data can be shared across
processes and dealt with properly.

When using batched environments such as [`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv), the transform can be used at the
single env instance level (i.e., a batch of transformed envs with tensordict primers set within) or at the
batched env instance level (i.e., a transformed batch of regular envs).

Not including a `TensorDictPrimer` in the environment may result in poorly defined behaviors, for instance
in parallel settings where a step involves copying the new recurrent state from `"next"` to the root
tensordict, which the meth:~torchrl.EnvBase.step_mdp method will not be able to do as the recurrent states
are not registered within the environment specs.

See `torchrl.modules.utils.get_primers_from_module()` for a method to generate all primers for a given
module.

Examples

```
>>> from torchrl.collectors import Collector
>>> from torchrl.envs import TransformedEnv, InitTracker
>>> from torchrl.envs import GymEnv
>>> from torchrl.modules import MLP, LSTMModule
>>> from torch import nn
>>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
>>>
>>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
>>> lstm_module = LSTMModule(
... input_size=env.observation_spec["observation"].shape[-1],
... hidden_size=64,
... in_keys=["observation", "rs_h", "rs_c"],
... out_keys=["intermediate", ("next", "rs_h"), ("next", "rs_c")])
>>> mlp = MLP(num_cells=[64], out_features=1)
>>> policy = Seq(lstm_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
>>> policy(env.reset())
>>> env = env.append_transform(lstm_module.make_tensordict_primer())
>>> data_collector = Collector(
... env,
... policy,
... frames_per_batch=10
... )
>>> for data in data_collector:
... print(data)
... break
```