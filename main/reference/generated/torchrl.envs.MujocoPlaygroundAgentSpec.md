# MujocoPlaygroundAgentSpec

torchrl.envs.MujocoPlaygroundAgentSpec(*name: str*, *action_indices: list[int]*, *observation_indices: list[int] | dict[str, list[int]]*) → None[[source]](../../_modules/torchrl/envs/libs/mujoco_playground.html#MujocoPlaygroundAgentSpec)

Observation/action slice definition for one agent in a cooperative task.

Parameters:

- **name** (*str*) - group key used in output TensorDicts (e.g. `"agent_0"`).
- **action_indices** (*list**of**int*) - indices into the global action vector that
this agent controls. Must be non-overlapping across all agents and
together must cover `range(env.action_size)`.
- **observation_indices** (*list**of**int**or**dict**of**str to list**of**int*) - for
flat-obs environments, a list of ints selecting from the global
observation vector. For dict-obs environments, a `dict` mapping
each observation key to a list of ints selecting from that key's
sub-vector.