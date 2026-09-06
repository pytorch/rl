# get_env_transforms_from_module

*class*torchrl.modules.get_env_transforms_from_module(*module*, *init_key='is_init'*)[[source]](../../_modules/torchrl/modules/utils/utils.html#get_env_transforms_from_module)

Return all [`TransformedEnv`](torchrl.envs.transforms.TransformedEnv.html#torchrl.envs.transforms.TransformedEnv) transforms needed for a recurrent module.

Composes [`InitTracker`](torchrl.envs.transforms.InitTracker.html#torchrl.envs.transforms.InitTracker) (writes
`is_init=True` at episode resets) with
[`TensorDictPrimer`](torchrl.envs.transforms.TensorDictPrimer.html#torchrl.envs.transforms.TensorDictPrimer) (initialises hidden
states). Pass the result directly to
[`TransformedEnv`](torchrl.envs.transforms.TransformedEnv.html#torchrl.envs.transforms.TransformedEnv).

Parameters:

- **module** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) - A module that may contain recurrent
submodules (e.g. [`LSTMModule`](torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule) or
[`GRUModule`](torchrl.modules.GRUModule.html#torchrl.modules.GRUModule)).
- **init_key** (*str**,**optional*) - the key used by
[`InitTracker`](torchrl.envs.transforms.InitTracker.html#torchrl.envs.transforms.InitTracker) to mark episode
starts. Must match the `is_init` key expected by the recurrent
module. Defaults to `"is_init"`.

Returns:

A [`Compose`](torchrl.envs.transforms.Compose.html#torchrl.envs.transforms.Compose) of
`[InitTracker, TensorDictPrimer]` when the module contains recurrent
submodules, or a bare [`InitTracker`](torchrl.envs.transforms.InitTracker.html#torchrl.envs.transforms.InitTracker)
otherwise.

Example

```
>>> from torchrl.modules import GRUModule
>>> from torchrl.modules.utils import get_env_transforms_from_module
>>> gru = GRUModule(
... input_size=4, hidden_size=8, num_layers=1,
... in_keys=["obs", "recurrent_state", "is_init"],
... out_keys=["features", ("next", "recurrent_state")],
... )
>>> transforms = get_env_transforms_from_module(gru)
>>> # TransformedEnv(base_env, transforms)
```