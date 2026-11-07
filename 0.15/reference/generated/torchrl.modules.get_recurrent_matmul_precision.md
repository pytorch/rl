# get_recurrent_matmul_precision

torchrl.modules.get_recurrent_matmul_precision() → Literal['ieee', 'tf32', 'tf32x3'][[source]](../../_modules/torchrl/modules/tensordict_module/_rnn_precision.html#get_recurrent_matmul_precision)

Resolve the currently effective precision to a concrete mode.

Always returns one of `"ieee"`, `"tf32"` or `"tf32x3"`. The result
is what the kernel actually runs at, including preset / GPU resolution.
Does not see per-module overrides; those are resolved at the call site
via `_resolve_precision()`.