# PettingZooEnv

torchrl.envs.PettingZooEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/pettingzoo.html#PettingZooEnv)

PettingZoo Environment.

To install petting zoo follow the guide here <https://github.com/Farama-Foundation/PettingZoo#installation>__.

This class is a general torchrl wrapper for all PettingZoo environments.
It is able to wrap both `pettingzoo.AECEnv` and `pettingzoo.ParallelEnv`.

Let's see how more in details:

For wrapping `pettingzoo.ParallelEnv` provide the name of your petting zoo task (in the `task` argument)
and specify `parallel=True`. This will construct the `pettingzoo.ParallelEnv` version of that task
(if it is supported in pettingzoo) and wrap it for torchrl.
In wrapped `pettingzoo.ParallelEnv` all agents will step at each environment step.
If the number of agents during the task varies, please set `use_mask=True`.
`"mask"` will be provided
as an output in each group and should be used to mask out dead agents.
The environment will be reset as soon as one agent is done (unless `done_on_any` is `False`).

For wrapping `pettingzoo.AECEnv` provide the name of your petting zoo task (in the `task` argument)
and specify `parallel=False`. This will construct the `pettingzoo.AECEnv` version of that task
and wrap it for torchrl.
In wrapped `pettingzoo.AECEnv`, at each step only one agent will act.
For this reason, it is compulsory to set `use_mask=True` for this type of environment.
`"mask"` will be provided as an output for each group and can be used to mask out non-acting agents.
The environment will be reset only when all agents are done (unless `done_on_any` is `True`).

If there are any unavailable actions for an agent,
the environment will also automatically update the mask of its `action_spec` and output an `"action_mask"`
for each group to reflect the latest available actions. This should be passed to a masked distribution during
training.

As a feature of torchrl multiagent, you are able to control the grouping of agents in your environment.
You can group agents together (stacking their tensors) to leverage vectorization when passing them through the same
neural network. You can split agents in different groups where they are heterogeneous or should be processed by
different neural networks. To group, you just need to pass a `group_map` at env constructiuon time.

By default, agents in pettingzoo will be grouped by name.
For example, with agents `["agent_0","agent_1","agent_2","adversary_0"]`, the tensordicts will look like:

```
>>> print(env.rand_action(env.reset()))
TensorDict(
 fields={
 agent: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 9]), device=cpu, dtype=torch.int64, is_shared=False),
 action_mask: Tensor(shape=torch.Size([3, 9]), device=cpu, dtype=torch.bool, is_shared=False),
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 3, 3, 2]), device=cpu, dtype=torch.int8, is_shared=False),
 terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3]))},
 adversary: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([1, 9]), device=cpu, dtype=torch.int64, is_shared=False),
 action_mask: Tensor(shape=torch.Size([1, 9]), device=cpu, dtype=torch.bool, is_shared=False),
 done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([1, 3, 3, 2]), device=cpu, dtype=torch.int8, is_shared=False),
 terminated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([1]))},
 batch_size=torch.Size([]))
>>> print(env.group_map)
{"agent": ["agent_0", "agent_1", "agent_2"], "adversary": ["adversary_0"]}
```

Otherwise, a group map can be specified or selected from some premade options.
See `torchrl.envs.utils.MarlGroupMapType` for more info.
For example, you can provide `MarlGroupMapType.ONE_GROUP_PER_AGENT`, telling that each agent should
have its own tensordict (similar to the pettingzoo parallel API).

Grouping is useful for leveraging vectorization among agents whose data goes through the same
neural network.

Parameters:

- **task** (*str*) - the name of the pettingzoo task to create in the "<env>/<task>" format (for example, "sisl/multiwalker_v9")
or "<task>" format (for example, "multiwalker_v9").
- **parallel** (*bool*) - if to construct the `pettingzoo.ParallelEnv` version of the task or the `pettingzoo.AECEnv`.
- **return_state** (*bool**,**optional*) - whether to return the global state from pettingzoo
(not available in all environments). Defaults to `False`.
- **group_map** (*MarlGroupMapType**or**Dict**[**str**,**List**[**str**]**]**]**,**optional*) - how to group agents in tensordicts for
input/output. By default, agents will be grouped by their name. Otherwise, a group map can be specified
or selected from some premade options. See `torchrl.envs.utils.MarlGroupMapType` for more info.
- **use_mask** (*bool**,**optional*) - whether the environment should output an `"mask"`. This is compulsory in
wrapped `pettingzoo.AECEnv` to mask out non-acting agents and should be also used
for `pettingzoo.ParallelEnv` when the number of agents can vary. Defaults to `False`.
- **categorical_actions** (*bool**,**optional*) - if the environments actions are discrete, whether to transform
them to categorical or one-hot.
- **seed** (*int**,**optional*) - the seed. Defaults to `None`.
- **done_on_any** (*bool**,**optional*) - whether the environment's done keys are set by aggregating the agent keys
using `any()` (when `True`) or `all()` (when `False`). Default (`None`) is to use `any()` for
parallel environments and `all()` for AEC ones.

Examples

```
>>> # Parallel env
>>> from torchrl.envs.libs.pettingzoo import PettingZooEnv
>>> kwargs = {"n_pistons": 21, "continuous": True}
>>> env = PettingZooEnv(
... task="pistonball_v6",
... parallel=True,
... return_state=True,
... group_map=None, # Use default (all pistons grouped together)
... **kwargs,
... )
>>> print(env.group_map)
... {'piston': ['piston_0', 'piston_1', ..., 'piston_20']}
>>> env.rollout(10)
>>> # AEC env
>>> from torchrl.envs.libs.pettingzoo import PettingZooEnv
>>> from torchrl.envs.utils import MarlGroupMapType
>>> env = PettingZooEnv(
... task="tictactoe_v3",
... parallel=False,
... use_mask=True, # Must use it since one player plays at a time
... group_map=None # # Use default for AEC (one group per player)
... )
>>> print(env.group_map)
... {'player_1': ['player_1'], 'player_2': ['player_2']}
>>> env.rollout(10)
```