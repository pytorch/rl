# MeltingpotWrapper

torchrl.envs.MeltingpotWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/meltingpot.html#MeltingpotWrapper)

Meltingpot environment wrapper.

GitHub: [google-deepmind/meltingpot](https://github.com/google-deepmind/meltingpot)

Paper: [https://arxiv.org/abs/2211.13746](https://arxiv.org/abs/2211.13746)

Melting Pot assesses generalization to novel social situations involving both familiar and unfamiliar individuals,
and has been designed to test a broad range of social interactions such as: cooperation, competition, deception,
reciprocation, trust, stubbornness and so on. Melting Pot offers researchers a set of over 50 multi-agent
reinforcement learning substrates (multi-agent games) on which to train agents, and over 256 unique test scenarios
on which to evaluate these trained agents.

Parameters:

**env** (`meltingpot.utils.substrates.substrate.Substrate`) - the meltingpot substrate to wrap.

Keyword Arguments:

- **max_steps** (*int**,**optional*) - Horizon of the task. Defaults to `None` (infinite horizon).
Each Meltingpot substrate can
be terminating or not. If `max_steps` is specified,
the scenario is also terminated (and the `"terminated"` flag is set) whenever this horizon is reached.
Unlike gym's `TimeLimit` transform or torchrl's [`StepCounter`](torchrl.envs.transforms.StepCounter.html#torchrl.envs.transforms.StepCounter),
this argument will not set the `"truncated"` entry in the tensordict.
- **categorical_actions** (*bool**,**optional*) - if the environment actions are discrete, whether to transform
them to categorical or one-hot. Defaults to `True`.
- **group_map** ([*MarlGroupMapType*](torchrl.envs.MarlGroupMapType.html#torchrl.envs.MarlGroupMapType)*or**Dict**[**str**,**List**[**str**]**]**,**optional*) - how to group agents in tensordicts for
input/output. By default, they will be all put
in one group named `"agents"`.
Otherwise, a group map can be specified or selected from some premade options.
See `MarlGroupMapType` for more info.

Variables:

- **group_map** (*Dict**[**str**,**List**[**str**]**]*) - how to group agents in tensordicts for
input/output. See `MarlGroupMapType` for more info.
- **agent_names** (*list**of**str*) - names of the agent in the environment
- **agent_names_to_indices_map** (*Dict**[**str**,**int**]*) - dictionary mapping agent names to their index in the environment
- **available_envs** (*List**[**str**]*) - the list of the scenarios available to build.

Warning

Meltingpot returns a single `done` flag which does not distinguish between
when the env reached `max_steps` and termination.
If you deem the `truncation` signal necessary, set `max_steps` to
`None` and use a [`StepCounter`](torchrl.envs.transforms.StepCounter.html#torchrl.envs.transforms.StepCounter) transform.

Examples

```
>>> from meltingpot import substrate
>>> from torchrl.envs.libs.meltingpot import MeltingpotWrapper
>>> substrate_config = substrate.get_config("commons_harvest__open")
>>> mp_env = substrate.build_from_config(
... substrate_config, roles=substrate_config.default_player_roles
... )
>>> env_torchrl = MeltingpotWrapper(env=mp_env)
>>> print(env_torchrl.rollout(max_steps=5))
TensorDict(
 fields={
 RGB: Tensor(shape=torch.Size([5, 144, 192, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
 agents: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([5, 7]), device=cpu, dtype=torch.int64, is_shared=False),
 observation: TensorDict(
 fields={
 COLLECTIVE_REWARD: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False),
 READY_TO_SHOOT: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False),
 RGB: Tensor(shape=torch.Size([5, 7, 88, 88, 3]), device=cpu, dtype=torch.uint8, is_shared=False)},
 batch_size=torch.Size([5, 7]),
 device=cpu,
 is_shared=False)},
 batch_size=torch.Size([5, 7]),
 device=cpu,
 is_shared=False),
 done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 RGB: Tensor(shape=torch.Size([5, 144, 192, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
 agents: TensorDict(
 fields={
 observation: TensorDict(
 fields={
 COLLECTIVE_REWARD: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False),
 READY_TO_SHOOT: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False),
 RGB: Tensor(shape=torch.Size([5, 7, 88, 88, 3]), device=cpu, dtype=torch.uint8, is_shared=False)},
 batch_size=torch.Size([5, 7]),
 device=cpu,
 is_shared=False),
 reward: Tensor(shape=torch.Size([5, 7, 1]), device=cpu, dtype=torch.float64, is_shared=False)},
 batch_size=torch.Size([5, 7]),
 device=cpu,
 is_shared=False),
 done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5]),
 device=cpu,
 is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5]),
 device=cpu,
 is_shared=False)
```