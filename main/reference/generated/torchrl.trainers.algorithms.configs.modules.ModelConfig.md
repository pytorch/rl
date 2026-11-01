# torchrl.trainers.algorithms.configs.modules.ModelConfig

*class*torchrl.trainers.algorithms.configs.modules.ModelConfig(*_partial_: bool = False*, *in_keys: Any = None*, *out_keys: Any = None*, *shared: bool = False*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/modules.html#ModelConfig)

Parent class to configure a model.

A model can be made of several networks. It is always a [`TensorDictModuleBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) instance.

See also

[`TanhNormalModelConfig`](torchrl.trainers.algorithms.configs.modules.TanhNormalModelConfig.html#torchrl.trainers.algorithms.configs.modules.TanhNormalModelConfig), [`ValueModelConfig`](torchrl.trainers.algorithms.configs.modules.ValueModelConfig.html#torchrl.trainers.algorithms.configs.modules.ValueModelConfig)