# VmasEnv

torchrl.envs.VmasEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/vmas.html#VmasEnv)

Vmas environment wrapper.

GitHub: [proroklab/VectorizedMultiAgentSimulator](https://github.com/proroklab/VectorizedMultiAgentSimulator)

Paper: [https://arxiv.org/abs/2207.03530](https://arxiv.org/abs/2207.03530)

Parameters:

**scenario** (*str**or**vmas.simulator.scenario.BaseScenario*) - the vmas scenario to build.
Must be one of `available_envs`. For a description and rendering of available scenarios see
[the README](https://github.com/proroklab/VectorizedMultiAgentSimulator/tree/VMAS-1.3.3?tab=readme-ov-file#main-scenarios).

Keyword Arguments:

- **num_envs** (*int*) - Number of vectorized simulation environments. VMAS performs vectorized simulations using PyTorch.
This argument indicates the number of vectorized environments that should be simulated in a batch. It will also
determine the batch size of the environment.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device for simulation. Defaults to the defaultt device. All the tensors created by VMAS
will be placed on this device.
- **continuous_actions** (*bool**,**optional*) - Whether to use continuous actions. Defaults to `True`. If `False`, actions
will be discrete. The number of actions and their size will depend on the chosen scenario.
See the VMAS repository for more info.
- **max_steps** (*int**,**optional*) - Horizon of the task. Defaults to `None` (infinite horizon). Each VMAS scenario can
be terminating or not. If `max_steps` is specified,
the scenario is also terminated (and the `"terminated"` flag is set) whenever this horizon is reached.
Unlike gym's `TimeLimit` transform or torchrl's [`StepCounter`](torchrl.envs.transforms.StepCounter.html#torchrl.envs.transforms.StepCounter),
this argument will not set the `"truncated"` entry in the tensordict.
- **categorical_actions** (*bool**,**optional*) - if the environment actions are discrete, whether to transform
them to categorical or one-hot. Defaults to `True`.
- **group_map** ([*MarlGroupMapType*](torchrl.envs.MarlGroupMapType.html#torchrl.envs.MarlGroupMapType)*or**Dict**[**str**,**List**[**str**]**]**,**optional*) - how to group agents in tensordicts for
input/output. By default, if the agent names follow the `"<name>_<int>"`
convention, they will be grouped by `"<name>"`. If they do not follow this convention, they will be all put
in one group named `"agents"`.
Otherwise, a group map can be specified or selected from some premade options.
See `MarlGroupMapType` for more info.
- **scenario_kwargs** (*Dict**,**optional*) - dictionary of additional arguments passed to the VMAS
scenario constructor (e.g., number of agents, reward sparsity).
This is convenient when scenario parameters are stored under a dedicated config field.
- ****kwargs** (*Dict**,**optional*) - Additional arguments passed to the VMAS scenario constructor.
This allows passing scenario arguments directly as keyword arguments.
If the same key is provided in both `scenario_kwargs` and `kwargs`, the value in
`kwargs` takes precedence.
The available arguments will vary based on the chosen scenario.
To see the available arguments for a specific scenario, see the constructor in its file from
[the scenario folder](https://github.com/proroklab/VectorizedMultiAgentSimulator/tree/VMAS-1.3.3/vmas/scenarios).

Variables:

- **group_map** (*Dict**[**str**,**List**[**str**]**]*) - how to group agents in tensordicts for
input/output. See `MarlGroupMapType` for more info.
- **agent_names** (*list**of**str*) - names of the agent in the environment
- **agent_names_to_indices_map** (*Dict**[**str**,**int**]*) - dictionary mapping agent names to their index in the environment
- **full_action_spec_unbatched** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - version of the spec without the vectorized dimension
- **full_observation_spec_unbatched** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - version of the spec without the vectorized dimension
- **full_reward_spec_unbatched** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - version of the spec without the vectorized dimension
- **full_done_spec_unbatched** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - version of the spec without the vectorized dimension
- **het_specs** (*bool*) - whether the environment has any lazy spec
- **het_specs_map** (*Dict**[**str**,**bool**]*) - dictionary mapping each group to a flag representing of the group has lazy specs
- **available_envs** (*List**[**str**]*) - the list of the scenarios available to build.

Warning

VMAS returns a single `done` flag which does not distinguish between
when the env reached `max_steps` and termination.
If you deem the `truncation` signal necessary, set `max_steps` to
`None` and use a [`StepCounter`](torchrl.envs.transforms.StepCounter.html#torchrl.envs.transforms.StepCounter) transform.

Examples

```
>>> env = VmasEnv(
... scenario="flocking",
... num_envs=32,
... continuous_actions=True,
... max_steps=200,
... device="cpu",
... seed=None,
... # Scenario kwargs
... n_agents=5,
... )
>>> print(env.rollout(10))
TensorDict(
 fields={
 agents: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([32, 10, 5, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 info: TensorDict(
 fields={
 agent_collision_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 agent_distance_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([32, 10, 5]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([32, 10, 5, 18]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([32, 10, 5]),
 device=cpu,
 is_shared=False),
 done: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 agents: TensorDict(
 fields={
 info: TensorDict(
 fields={
 agent_collision_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 agent_distance_rew: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([32, 10, 5]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([32, 10, 5, 18]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([32, 10, 5, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([32, 10, 5]),
 device=cpu,
 is_shared=False),
 done: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32, 10]),
 device=cpu,
 is_shared=False),
 terminated: Tensor(shape=torch.Size([32, 10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([32, 10]),
 device=cpu,
 is_shared=False)
```