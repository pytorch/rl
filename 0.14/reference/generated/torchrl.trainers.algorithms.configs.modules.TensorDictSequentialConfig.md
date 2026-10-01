# torchrl.trainers.algorithms.configs.modules.TensorDictSequentialConfig

*class*torchrl.trainers.algorithms.configs.modules.TensorDictSequentialConfig(*_partial_: bool = False*, *in_keys: Any = None*, *out_keys: Any = None*, *shared: bool = False*, *modules: Any | None = None*, *partial_tolerant: bool = False*, *selected_out_keys: Any | None = None*, *inplace: bool | str | None = None*, *_target_: str = 'torchrl.trainers.algorithms.configs.modules._make_tensordict_sequential'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#TensorDictSequentialConfig)

A class to configure a TensorDictSequential.

Example

```
>>> cfg = TensorDictSequentialConfig(
... modules=[
... TensorDictModuleConfig(module=MLPConfig(in_features=10, out_features=10, depth=2, num_cells=32), in_keys=["observation"], out_keys=["hidden"]),
... TensorDictModuleConfig(module=MLPConfig(in_features=10, out_features=5, depth=2, num_cells=32), in_keys=["hidden"], out_keys=["action"])
... ]
... )
>>> seq = instantiate(cfg)
>>> assert isinstance(seq, TensorDictSequential)
```

See also

[`tensordict.nn.TensorDictSequential`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential)