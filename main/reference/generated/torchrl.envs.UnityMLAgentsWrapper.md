# UnityMLAgentsWrapper

torchrl.envs.UnityMLAgentsWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/unity_mlagents.html#UnityMLAgentsWrapper)

Unity ML-Agents environment wrapper.

GitHub: [Unity-Technologies/ml-agents](https://github.com/Unity-Technologies/ml-agents)

Documentation: [https://unity-technologies.github.io/ml-agents/Python-LLAPI/](https://unity-technologies.github.io/ml-agents/Python-LLAPI/)

Parameters:

**env** (*mlagents_envs.environment.UnityEnvironment*) - the ML-Agents
environment to wrap.

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
>>> from mlagents_envs.environment import UnityEnvironment
>>> base_env = UnityEnvironment()
>>> from torchrl.envs import UnityMLAgentsWrapper
>>> env = UnityMLAgentsWrapper(base_env)
>>> td = env.reset()
>>> td = env.step(td.update(env.full_action_spec.rand()))
```