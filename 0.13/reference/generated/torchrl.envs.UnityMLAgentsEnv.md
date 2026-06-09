# UnityMLAgentsEnv

torchrl.envs.UnityMLAgentsEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/unity_mlagents.html#UnityMLAgentsEnv)

Unity ML-Agents environment wrapper.

GitHub: [Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents)

Documentation: [https://unity-technologies.github.io/ml-agents/Python-LLAPI/](https://unity-technologies.github.io/ml-agents/Python-LLAPI/)

This class can be provided any of the optional initialization arguments that
`mlagents_envs.environment.UnityEnvironment` class provides. For a
list of these arguments, see:
[https://unity-technologies.github.io/ml-agents/Python-LLAPI-Documentation/#__init__](https://unity-technologies.github.io/ml-agents/Python-LLAPI-Documentation/#__init__)

If both `file_name` and `registered_name` are given, an error is raised.

If neither `file_name` nor``registered_name`` are given, the environment
setup waits on a localhost port, and the user must execute a Unity ML-Agents
environment binary for to connect to it.

Parameters:

- **file_name** (*str**,**optional*) - if provided, the path to the Unity
environment binary. Defaults to `None`.
- **registered_name** (*str**,**optional*) - if provided, the Unity environment
binary is loaded from the default ML-Agents registry. The list of
registered environments is in `available_envs`. Defaults to
`None`.

Keyword Arguments:

- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - if provided, the device on which the data
is to be cast. Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - the batch size of the environment.
Defaults to `torch.Size([])`.
- **allow_done_after_reset** (*bool**,**optional*) - if `True`, it is tolerated
for envs to be `done` just after [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset) is called.
Defaults to `False`.
- **group_map** ([*MarlGroupMapType*](torchrl.envs.MarlGroupMapType.html#torchrl.envs.MarlGroupMapType)*or**Dict**[**str**,**List**[**str**]**]**]**,**optional*) - how to
group agents in tensordicts for input/output. See
`MarlGroupMapType` for more info. If not
specified, agents are grouped according to the group ID given by the
Unity environment. Defaults to `None`.
- **categorical_actions** (*bool**,**optional*) - if `True`, categorical specs
will be converted to the TorchRL equivalent
([`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)), otherwise a one-hot encoding
will be used ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot)). Defaults to `False`.

Variables:

**available_envs** - list of registered environments available to build

Examples

```
>>> from torchrl.envs import UnityMLAgentsEnv
>>> env = UnityMLAgentsEnv(registered_name='3DBall')
>>> td = env.reset()
>>> td = env.step(td.update(env.full_action_spec.rand()))
>>> td
TensorDict(
 fields={
 group_0: TensorDict(
 fields={
 agent_0: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_10: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_11: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_1: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_2: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_3: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_4: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_5: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_6: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_7: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_8: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_9: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 continuous_action: Tensor(shape=torch.Size([2]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 next: TensorDict(
 fields={
 group_0: TensorDict(
 fields={
 agent_0: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_10: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_11: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_1: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_2: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_3: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_4: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_5: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_6: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_7: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_8: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 agent_9: TensorDict(
 fields={
 VectorSensor_size8: Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 group_reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```