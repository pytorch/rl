# MarlGroupMapType

torchrl.envs.MarlGroupMapType(*value*, *names=None*, ***, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[[source]](../../_modules/torchrl/envs/utils.html#MarlGroupMapType)

Marl Group Map Type.

As a feature of torchrl multiagent, you are able to control the grouping of agents in your environment.
You can group agents together (stacking their tensors) to leverage vectorization when passing them through the same
neural network. You can split agents in different groups where they are heterogeneous or should be processed by
different neural networks. To group, you just need to pass a `group_map` at env constructiuon time.

Otherwise, you can choose one of the premade grouping strategies from this class.

- With `group_map=MarlGroupMapType.ALL_IN_ONE_GROUP` and
agents `["agent_0", "agent_1", "agent_2", "agent_3"]`,
the tensordicts coming and going from your environment will look
something like:

```
>>> print(env.rand_action(env.reset()))
TensorDict(
 fields={
 agents: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([4, 9]), device=cpu, dtype=torch.int64, is_shared=False),
 done: Tensor(shape=torch.Size([4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([4, 3, 3, 2]), device=cpu, dtype=torch.int8, is_shared=False)},
 batch_size=torch.Size([4]))},
 batch_size=torch.Size([]))
>>> print(env.group_map)
{"agents": ["agent_0", "agent_1", "agent_2", "agent_3]}
```
- With `group_map=MarlGroupMapType.ONE_GROUP_PER_AGENT` and
agents `["agent_0", "agent_1", "agent_2", "agent_3"]`,
the tensordicts coming and going from your environment will look
something like:

```
>>> print(env.rand_action(env.reset()))
TensorDict(
 fields={
 agent_0: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([9]), device=cpu, dtype=torch.int64, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 3, 2]), device=cpu, dtype=torch.int8, is_shared=False)},
 batch_size=torch.Size([]))},
 agent_1: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([9]), device=cpu, dtype=torch.int64, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 3, 2]), device=cpu, dtype=torch.int8, is_shared=False)},
 batch_size=torch.Size([]))},
 agent_2: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([9]), device=cpu, dtype=torch.int64, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 3, 2]), device=cpu, dtype=torch.int8, is_shared=False)},
 batch_size=torch.Size([]))},
 agent_3: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([9]), device=cpu, dtype=torch.int64, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 3, 2]), device=cpu, dtype=torch.int8, is_shared=False)},
 batch_size=torch.Size([]))},
 batch_size=torch.Size([]))
>>> print(env.group_map)
{"agent_0": ["agent_0"], "agent_1": ["agent_1"], "agent_2": ["agent_2"], "agent_3": ["agent_3"]}
```