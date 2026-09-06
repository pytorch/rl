# ModelBasedEnvBase

torchrl.envs.ModelBasedEnvBase(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/model_based/common.html#ModelBasedEnvBase)

Basic environment for Model Based RL sota-implementations.

Wrapper around the model of the MBRL algorithm.
It is meant to give an env framework to a world model (including but not limited to observations, reward, done state and safety constraints models).
and to behave as a classical environment.

This is a base class for other environments and it should not be used directly.

Example

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.data import Composite, Unbounded
>>> class MyMBEnv(ModelBasedEnvBase):
... def __init__(self, world_model, device="cpu", dtype=None, batch_size=None):
... super().__init__(world_model, device=device, dtype=dtype, batch_size=batch_size)
... self.observation_spec = Composite(
... hidden_observation=Unbounded((4,))
... )
... self.state_spec = Composite(
... hidden_observation=Unbounded((4,)),
... )
... self.action_spec = Unbounded((1,))
... self.reward_spec = Unbounded((1,))
...
... def _reset(self, tensordict: TensorDict) -> TensorDict:
... tensordict = TensorDict(
... batch_size=self.batch_size,
... device=self.device,
... )
... tensordict = tensordict.update(self.state_spec.rand())
... tensordict = tensordict.update(self.observation_spec.rand())
... return tensordict
>>> # This environment is used as follows:
>>> import torch.nn as nn
>>> from torchrl.modules import MLP, WorldModelWrapper
>>> world_model = WorldModelWrapper(
... TensorDictModule(
... MLP(out_features=4, activation_class=nn.ReLU, activate_last_layer=True, depth=0),
... in_keys=["hidden_observation", "action"],
... out_keys=["hidden_observation"],
... ),
... TensorDictModule(
... nn.Linear(4, 1),
... in_keys=["hidden_observation"],
... out_keys=["reward"],
... ),
... )
>>> env = MyMBEnv(world_model)
>>> tensordict = env.rollout(max_steps=10)
>>> print(tensordict)
TensorDict(
 fields={
 action: Tensor(torch.Size([10, 1]), dtype=torch.float32),
 done: Tensor(torch.Size([10, 1]), dtype=torch.bool),
 hidden_observation: Tensor(torch.Size([10, 4]), dtype=torch.float32),
 next: LazyStackedTensorDict(
 fields={
 hidden_observation: Tensor(torch.Size([10, 4]), dtype=torch.float32)},
 batch_size=torch.Size([10]),
 device=cpu,
 is_shared=False),
 reward: Tensor(torch.Size([10, 1]), dtype=torch.float32)},
 batch_size=torch.Size([10]),
 device=cpu,
 is_shared=False)
```

Properties:

observation_spec (Composite): sampling spec of the observations;
action_spec (TensorSpec): sampling spec of the actions;
reward_spec (TensorSpec): sampling spec of the rewards;
input_spec (Composite): sampling spec of the inputs;
batch_size (torch.Size): batch_size to be used by the env. If not set, the env accept tensordicts of all batch sizes.
device (torch.device): device where the env input and output are expected to live

Parameters:

- **world_model** (*nn.Module*) - model that generates world states and its corresponding rewards;
- **params** (*List**[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*]**,**optional*) - list of parameters of the world model;
- **buffers** (*List**[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*]**,**optional*) - list of buffers of the world model;
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device where the env input and output are expected to live
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - dtype of the env input and output
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - number of environments contained in the instance
- **run_type_check** (*bool**,**optional*) - whether to run type checks on the step of the env

torchrl.envs.step(*TensorDict -> TensorDict*)

step in the environment

torchrl.envs.reset(*TensorDict*, *optional -> TensorDict*)

reset the environment

torchrl.envs.set_seed(*int -> int*)

sets the seed of the environment

torchrl.envs.rand_step(*TensorDict*, *optional -> TensorDict*)

random step given the action spec

torchrl.envs.rollout(*Callable*, *... -> TensorDict*)

executes a rollout in the environment with the given policy (or random
steps if no policy is provided)