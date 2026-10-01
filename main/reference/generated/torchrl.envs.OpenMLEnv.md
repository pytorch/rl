# OpenMLEnv

torchrl.envs.OpenMLEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/openml.html#OpenMLEnv)

An environment interface to OpenML data to be used in bandits contexts.

Doc: [https://www.openml.org/search?type=data](https://www.openml.org/search?type=data)

Scikit-learn interface: [https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_openml.html)

Parameters:

- **dataset_name** (*str*) - the following datasets are supported:
`"adult_num"`, `"adult_onehot"`, `"mushroom_num"`, `"mushroom_onehot"`,
`"covertype"`, `"shuttle"` and `"magic"`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**compatible**,**optional*) - the device where the input
and output data is to be expected. Defaults to `"cpu"`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or**compatible**,**optional*) - the batch size of the environment,
ie. the number of elements samples and returned when a [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) is
called. Defaults to an empty batch size, ie. one element is sampled
at a time.

Variables:

**available_envs** (*List**[**str**]*) - list of envs to be built by this class.

Examples

```
>>> env = OpenMLEnv("adult_onehot", batch_size=[2, 3])
>>> print(env.reset())
TensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([2, 3, 106]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([2, 3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 y: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([2, 3]),
 device=cpu,
 is_shared=False)
```