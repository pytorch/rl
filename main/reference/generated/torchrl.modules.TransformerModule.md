# TransformerModule

*class*torchrl.modules.TransformerModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/transformer.html#TransformerModule)

A TensorDict wrapper turning a causal transformer into a temporal policy module.

The transformer analogue of [`LSTMModule`](torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule): the same
network runs either over a full `[B, T]` window (training) or one step
at a time against a key/value cache (collection), with matching outputs.
The execution path is selected by the
[`set_recurrent_mode`](torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode) context manager, exactly as
for the recurrent modules.

Unlike the recurrent modules, no state travels in the tensordict. The
key/value cache is inference state owned by the module instance: it is
allocated by the backbone on the first cached step, indexed by batch
position (one stream per environment of the batch), cleared wherever
`is_init` is set (sourced from `InitTracker`),
invalidated when the parameters change (in place or swapped for other
tensors), and released by `reset_cache()`. Copies and pickled
instances start with an empty cache. Rollouts and replay buffers hold
observations and features only, never a cache; the training path reads
`is_init` to rebuild positions and a block-diagonal causal mask over the
window.

Parameters:

- **input_size** (*int**,**optional*) - number of input features. Unused if
`transformer` is passed.
- **hidden_size** (*int**,**optional*) - dimension of the transformer's residual
stream. Unused if `transformer` is passed.
- **num_layers** (*int**,**optional*) - number of transformer blocks. Defaults to
`1`. Unused if `transformer` is passed.

Keyword Arguments:

- **num_heads** (*int**,**optional*) - number of attention heads. Required unless
`transformer` is passed.
- **max_seq_len** (*int**,**optional*) - maximum episode length (positional table
and cache size). Required unless `transformer` is passed.
- **dim_feedforward** (*int**,**optional*) - per-block MLP width. Defaults to
`4 * hidden_size`.
- **dropout** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - dropout probability. Defaults to `0.0`.
- **transformer** (*nn.Module**,**optional*) - a pre-built backbone honoring the
contract described in [`CausalTransformer`](torchrl.modules.CausalTransformer.html#torchrl.modules.CausalTransformer)
(`forward`, `new_kv_cache` and `reset_kv_cache` plus the
`num_layers`, `num_heads`, `head_dim` and `max_seq_len`
attributes). Exclusive with the size arguments.
- **in_key** (*NestedKey**,**optional*) - the input value key. Exclusive with
`in_keys`.
- **in_keys** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**NestedKey**,**optional*) - the input value key, optionally
followed by `"is_init"`. Defaults to `[in_key, "is_init"]`.
- **out_key** (*NestedKey**,**optional*) - the output value key. Exclusive with
`out_keys`.
- **out_keys** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**NestedKey**,**optional*) - a one-element list with the
output value key. Defaults to `[out_key]`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device to build the parameters on.
- **default_recurrent_mode** (*bool**,**optional*) - the recurrent mode when not
overridden by the [`set_recurrent_mode`](torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode)
context manager. Defaults to `False`.
- **validate_windows** (*bool**,**optional*) - whether the window path checks that
every row starts with `is_init=True` and raises otherwise. The
check is data-dependent: under [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile) it costs one
graph break, and `fullgraph=True` rejects it at compile time.
Pass `False` to compile the window path as one graph, in which
case the caller is responsible for episode-aligned windows.
Defaults to `True`.

Note

The cache is discarded whenever the parameters change. Parameter
edits in place or swapped parameter tensors are detected on the next
eager step; TorchRL's weight-synchronization paths (collectors and
the inference server) call `mark_weight_update()` explicitly,
which also covers compiled modules and updates that write through
`.data`. Call `mark_weight_update()` (or `reset_cache()`)
yourself after updating the parameters by any other means.

Note

The batch position is the stream identity of the cached-step path:
a module instance must see the same environments in the same order
on every call, which is what a collector over a batched environment
provides. Use one instance per collector (or per collector worker)
and call `reset_cache()` before reusing an instance with another
environment. Batches whose composition changes between calls, such as
the partial batches of an asynchronous collector, need a stream-keyed
cache and are not supported by this module yet.

Note

Training windows must be episode-aligned: every row must start with
`is_init=True`, which is what complete-trajectory sampling
provides. A window that starts mid-episode raises a `ValueError`
rather than silently recomputing the prefix from position `0`.

Note

Episodes longer than `max_seq_len` raise an error; sliding-window
attention is deliberately out of scope.

Examples

```
>>> import torch
>>> from tensordict.nn import TensorDictModule, TensorDictSequential
>>> from torch import nn
>>> from torchrl.envs import GymEnv, InitTracker, TransformedEnv
>>> from torchrl.modules import TransformerModule, set_recurrent_mode
>>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
>>> module = TransformerModule(
... input_size=env.observation_spec["observation"].shape[-1],
... hidden_size=16,
... num_layers=2,
... num_heads=4,
... max_seq_len=200,
... in_key="observation",
... out_key="embed",
... )
>>> policy = TensorDictSequential(
... module,
... TensorDictModule(nn.Linear(16, 1), in_keys=["embed"], out_keys=["action"]),
... )
>>> rollout = env.rollout(10, policy)
>>> rollout["embed"].shape
torch.Size([10, 16])
>>> "transformer_state" in rollout.keys()
False
>>> with set_recurrent_mode(True):
... window = module(rollout.exclude("embed").clone())
>>> torch.allclose(window["embed"], rollout["embed"], atol=1e-5)
True
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*)[[source]](../../_modules/torchrl/modules/tensordict_module/transformer.html#TransformerModule.forward)

Run the transformer, honouring `is_init` for state resets.

With `recurrent_mode=False`, one step is processed against the
module's cache, whose rows are cleared where `is_init` is set; this
path is inference only and runs under `torch.no_grad()`. With
`recurrent_mode=True`, a full `(B, T)` window is processed under a
block-diagonal causal mask built from `is_init`; the cache is
neither read nor written, and gradients flow through the window.

mark_weight_update() → None[[source]](../../_modules/torchrl/modules/tensordict_module/transformer.html#TransformerModule.mark_weight_update)

Discard the cache after a weight update.

TorchRL's weight-synchronization paths (collectors and the inference
server) call this through `torchrl._utils.mark_weight_update()`
once new weights are applied, so every stream restarts instead of
attending to keys and values computed with the previous weights. Call
it yourself after updating the parameters by other means.

reset_cache() → None[[source]](../../_modules/torchrl/modules/tensordict_module/transformer.html#TransformerModule.reset_cache)

Release the key/value cache and the position counters.

The next cached step allocates a fresh cache for the batch it sees.
Call this before reusing the module with a different environment or
collector.