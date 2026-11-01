# GRUModule

*class*torchrl.modules.GRUModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#GRUModule)

An embedder for an GRU module.

This class adds the following functionality to [`torch.nn.GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU):

- Compatibility with TensorDict: the hidden states are reshaped to match
the tensordict batch size.
- Optional multi-step execution: with torch.nn, one has to choose between
[`torch.nn.GRUCell`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRUCell.html#torch.nn.GRUCell) and [`torch.nn.GRU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU), the former being
compatible with single step inputs and the latter being compatible with
multi-step. This class enables both usages.

After construction, the module is *not* set in recurrent mode, ie. it will
expect single steps inputs.

If in recurrent mode, it is expected that the last dimension of the tensordict
marks the number of steps. There is no constrain on the dimensionality of the
tensordict (except that it must be greater than one for temporal inputs).

Parameters:

- **input_size** - The number of expected features in the input x
- **hidden_size** - The number of features in the hidden state h
- **num_layers** - Number of recurrent layers. E.g., setting `num_layers=2`
would mean stacking two GRUs together to form a stacked GRU,
with the second GRU taking in outputs of the first GRU and
computing the final results. Default: 1
- **bias** - If `False`, then the layer does not use bias weights.
Default: `True`
- **dropout** - If non-zero, introduces a Dropout layer on the outputs of each
GRU layer except the last layer, with dropout probability equal to
`dropout`. Default: 0
- **python_based** - If `True`, will use a full Python implementation of the GRU cell. Default: `False`
- **recurrent_backend** - backend used in recurrent mode when trajectories reset
in the middle of a batch. `"pad"` keeps the existing split/pad
strategy. `"scan"` uses a scan loop over the time dimension and
avoids materializing padded trajectory chunks via `hoptorch`.
`"triton"`
(prototype, CUDA only) uses Triton kernels where available and
otherwise preserves pad-backend recurrent semantics for dropout
and bidirectional layers.
`"auto"` uses `"pad"` in eager mode and `"scan"` when called
under [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile). Default: `"pad"`.
- **recurrent_compute_dtype** - dtype used for the recurrent matmul inside the
`"triton"` backend (`torch.float32` -> TF32 on H100, default;
`torch.bfloat16` -> bigger SMEM margin, lower precision).
Ignored by the other backends. Default: `torch.float32`.
- **recurrent_recompute** - when set to `"full"`, trade extra compute for
lower backward activation memory. For `recurrent_backend="triton"`
this drops the per-step gate buffers (`save_r/z/n/save_gh_n`)
from the autograd save set and replays the forward kernel during
backward. For `recurrent_backend="scan"` this swaps the
`torch._higher_order_ops.scan` HOP for a python time-loop wrapped
with [`torch.utils.checkpoint.checkpoint()`](https://docs.pytorch.org/docs/stable/checkpoint.html#torch.utils.checkpoint.checkpoint). Only `"none"`
(default) and `"full"` are accepted; the `"pad"` backend rejects
non-`"none"` values because cuDNN manages its own backward
workspace. Default: `"none"`.
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
["recurrent_state"] and the `in_key` will be
appended before this.
- **in_keys** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**str*) - a pair of strings corresponding to the input value and recurrent entry.
Exclusive with `in_key`.
- **out_key** (*str**or**tuple**of**str*) - the output key of the module. Exclusive use
with `out_keys`. If provided, the recurrent keys are assumed to be
[("recurrent_state")] and the `out_key` will be
appended before these.
- **out_keys** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**str*) -

a pair of strings corresponding to the output value,
first and second hidden key.

Note

For a better integration with TorchRL's environments, the best naming
for the output hidden key is `("next", <custom_key>)`, such
that the hidden values are passed from step to step during a rollout.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**compatible*) - the device of the module.
- **gru** ([*torch.nn.GRU*](https://docs.pytorch.org/docs/stable/generated/torch.nn.GRU.html#torch.nn.GRU)*,**optional*) - a GRU instance to be wrapped.
Exclusive with other nn.GRU arguments.
- **default_recurrent_mode** (*bool**,**optional*) - if provided, the recurrent mode if it hasn't been overridden
by the [`set_recurrent_mode`](torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode) context manager / decorator.
Defaults to `False`.

Variables:

**recurrent_mode** - Returns the recurrent mode of the module.

set_recurrent_mode()[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#GRUModule.set_recurrent_mode)

controls whether the module should be executed in
recurrent mode.

make_tensordict_primer()[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#GRUModule.make_tensordict_primer)

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
>>> from torchrl.modules import MLP
>>> from torch import nn
>>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
>>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
>>> gru_module = GRUModule(
... input_size=env.observation_spec["observation"].shape[-1],
... hidden_size=64,
... in_keys=["observation", "rs"],
... out_keys=["intermediate", ("next", "rs")])
>>> mlp = MLP(num_cells=[64], out_features=1)
>>> policy = Seq(gru_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
>>> policy(env.reset())
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 intermediate: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
 is_init: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 rs: Tensor(shape=torch.Size([1, 64]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)
>>> gru_module_training = gru_module.set_recurrent_mode()
>>> policy_training = Seq(gru_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
>>> traj_td = env.rollout(3) # some random temporal data
>>> traj_td = policy_training(traj_td)
>>> print(traj_td)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 intermediate: Tensor(shape=torch.Size([3, 64]), device=cpu, dtype=torch.float32, is_shared=False),
 is_init: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 is_init: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 rs: Tensor(shape=torch.Size([3, 1, 64]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3]),
 device=cpu,
 is_shared=False)
```

*property*canonical_keys*: list[NestedKey]*

Return TensorDict keys whose canonical layout matters for this module.

The result is the union of `self.in_keys` and `self.out_keys`.

See also

`canonicalize()`,
[`canonicalize_rnn_subset()`](torchrl.modules.canonicalize_rnn_subset.html#torchrl.modules.canonicalize_rnn_subset).

canonicalize(*data: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, ***, *inplace: bool = False*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#GRUModule.canonicalize)

Canonicalize only the RNN-relevant leaves of `data`.

See [`LSTMModule.canonicalize()`](torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule.canonicalize) for details.

Parameters:

- **data** - TensorDict to canonicalize.
- **inplace** - When `True`, mutates `data` in place.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.modules import GRUModule
>>> module = GRUModule(input_size=3, hidden_size=4, in_key="obs",
... out_key="out")
>>> td = TensorDict({"obs": torch.zeros(2, 5, 3)}, batch_size=[2, 5])
>>> module.canonicalize(td)["obs"].is_contiguous()
True
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*)[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#GRUModule.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

make_cudnn_based() → GRUModule[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#GRUModule.make_cudnn_based)

Transforms the GRU layer in its CuDNN-based version.

Returns:

self

make_python_based() → GRUModule[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#GRUModule.make_python_based)

Transforms the GRU layer in its python-based version.

Returns:

self

make_tensordict_primer()[[source]](../../_modules/torchrl/modules/tensordict_module/rnn.html#GRUModule.make_tensordict_primer)

Makes a tensordict primer for the environment.

A `TensorDictPrimer` object will ensure that the policy is aware of the supplementary
inputs and outputs (recurrent states) during rollout execution. That way, the data can be shared across
processes and dealt with properly.

Not including a `TensorDictPrimer` in the environment may result in poorly defined behaviors, for instance
in parallel settings where a step involves copying the new recurrent state from `"next"` to the root
tensordict, which the meth:~torchrl.EnvBase.step_mdp method will not be able to do as the recurrent states
are not registered within the environment specs.

When using batched environments such as [`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv), the transform can be used at the
single env instance level (i.e., a batch of transformed envs with tensordict primers set within) or at the
batched env instance level (i.e., a transformed batch of regular envs).

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
>>> gru_module = GRUModule(
... input_size=env.observation_spec["observation"].shape[-1],
... hidden_size=64,
... in_keys=["observation", "rs"],
... out_keys=["intermediate", ("next", "rs")])
>>> mlp = MLP(num_cells=[64], out_features=1)
>>> policy = Seq(gru_module, Mod(mlp, in_keys=["intermediate"], out_keys=["action"]))
>>> policy(env.reset())
>>> env = env.append_transform(gru_module.make_tensordict_primer())
>>> data_collector = Collector(
... env,
... policy,
... frames_per_batch=10
... )
>>> for data in data_collector:
... print(data)
... break
```