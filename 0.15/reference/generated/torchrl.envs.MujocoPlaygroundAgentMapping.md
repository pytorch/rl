# MujocoPlaygroundAgentMapping

torchrl.envs.MujocoPlaygroundAgentMapping(*agents: list[~torchrl.envs.libs.mujoco_playground.MujocoPlaygroundAgentSpec] = <factory>*, *homogenization_mode: ~typing.Literal['none'*, *'max'*, *'concat'] = 'none'*) → None[[source]](../../_modules/torchrl/envs/libs/mujoco_playground.html#MujocoPlaygroundAgentMapping)

Agent mapping for [`MujocoPlaygroundWrapper`](torchrl.envs.MujocoPlaygroundWrapper.html#torchrl.envs.MujocoPlaygroundWrapper).

Defines how to split a single-agent MuJoCo Playground environment into a
cooperative multi-agent task by partitioning the observation and action
vectors among named agents.

Parameters:

- **agents** (*list**of**MujocoPlaygroundAgentSpec*) - one entry per agent,
defining each agent's observation slice and the action indices it
controls.
- **homogenization_mode** (*str**,**optional*) -

strategy for unifying
heterogeneous observation/action shapes across agents so that a
single shared policy can be used.

- `"none"` (default): each agent receives exactly its own
observation/action slice; shapes may differ across agents.
- `"max"`: observations are padded to
`max_obs_size + n_agents` (a one-hot agent-ID prefix is
prepended), and actions are padded to `max_action_size`.
All agents share the same input/output shape.

**Policy contract (max):** each agent's observation layout is
`[one_hot_id (n_agents) | raw_obs (len(observation_indices)) | zero_pad]`,
for a total length of `max_obs_size + n_agents`. The action
vector emitted by the policy has length `max_action_size`; only
the first `len(action_indices)` entries are used and the rest
are silently dropped.
- `"concat"`: each agent receives the full global
observation/action vector with zeros at positions it does not
own. All agents share the same input/output shape equal to the
full environment dimensions.

**Policy contract (concat):** the observation has length
`env.observation_size` with zeros at positions not owned by
the agent. The policy must emit a full-length action vector;
only entries at the agent's own `action_indices` are applied
to the environment and entries at indices owned by other agents
are ignored.

Examples

```
>>> mapping = MujocoPlaygroundAgentMapping(
... agents=[
... MujocoPlaygroundAgentSpec(
... name="agent_0",
... action_indices=[0, 1, 2],
... observation_indices=[0, 1, 2, 3],
... ),
... MujocoPlaygroundAgentSpec(
... name="agent_1",
... action_indices=[3, 4, 5],
... observation_indices=[4, 5, 6, 7],
... ),
... ],
... homogenization_mode="none",
... )
```