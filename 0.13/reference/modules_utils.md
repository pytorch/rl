# Utilities and Helpers

Utility modules and helper functions for building RL networks.

| [`ActorValueOperator`](generated/torchrl.modules.ActorValueOperator.html#torchrl.modules.ActorValueOperator)(*args, **kwargs) | Actor-value operator. |
| --- | --- |
| [`ActorCriticOperator`](generated/torchrl.modules.ActorCriticOperator.html#torchrl.modules.ActorCriticOperator)(*args, **kwargs) | Actor-critic operator. |
| [`ActorCriticWrapper`](generated/torchrl.modules.ActorCriticWrapper.html#torchrl.modules.ActorCriticWrapper)(*args, **kwargs) | Actor-value operator without common module. |
| [`get_primers_from_module`](generated/torchrl.modules.get_primers_from_module.html#torchrl.modules.get_primers_from_module)(module[, warn, strict]) | Get all tensordict primers from all submodules of a module. |
| [`get_env_transforms_from_module`](generated/torchrl.modules.get_env_transforms_from_module.html#torchrl.modules.get_env_transforms_from_module)(module[, ...]) | Return all [`TransformedEnv`](generated/torchrl.envs.transforms.TransformedEnv.html#torchrl.envs.transforms.TransformedEnv) transforms needed for a recurrent module. |

| [`get_recurrent_matmul_precision`](generated/torchrl.modules.get_recurrent_matmul_precision.html#torchrl.modules.get_recurrent_matmul_precision)() | Resolve the currently effective precision to a concrete mode. |
| --- | --- |
| [`set_recurrent_matmul_precision`](generated/torchrl.modules.set_recurrent_matmul_precision.html#torchrl.modules.set_recurrent_matmul_precision)(mode) | Set the process-global precision for the triton RNN backend. |

torchrl.modules.RecurrentMatmulPrecision

alias of `Literal`['ieee', 'tf32', 'tf32x3']

torchrl.modules.RecurrentMatmulPrecisionUserMode

alias of `Literal`['auto', 'fast', 'high-prec', 'ieee', 'tf32', 'tf32x3']

| [`SquashDims`](generated/torchrl.modules.models.utils.SquashDims.html#torchrl.modules.models.utils.SquashDims)([ndims_in]) | A squashing layer. |
| --- | --- |