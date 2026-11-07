# LBForagingWrapper

torchrl.envs.LBForagingWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/lbforaging.html#LBForagingWrapper)

Level-Based Foraging environment wrapper.

[Level-Based Foraging](https://github.com/semitable/lb-foraging) is a
fully-cooperative, sparse-reward Gymnasium environment for evaluating
multi-agent credit assignment: a variable number of agents, each with a
randomly assigned level, must coordinate to collect food items that also
have levels, and a food item is only collected when the agents currently
adjacent to it have levels summing to at least its own.

`lbforaging` exposes a single `gymnasium.Env` whose observation,
action and reward spaces are `gymnasium.spaces.Tuple` instances, one
entry per agent, and whose `terminated`/`truncated` flags are shared
by the whole team (LBF episodes end when either all food is collected or
a step limit is reached, for every agent at once). This wrapper exposes
that structure the way every other TorchRL multi-agent wrapper does:
per-agent entries nested under a single `"agents"` group, and the
shared `done`/`terminated`/`truncated` at the root.
Reaching LBF's step limit with food still on the board is a truncation;
collecting the last food item is a termination, including at the step limit.

Parameters:

**env** (*gymnasium.Env*) - a Level-Based Foraging environment, i.e. the
result of `gymnasium.make("Foraging-<...>-v3")` after
`import lbforaging`.

Keyword Arguments:

- **categorical_actions** (*bool**,**optional*) - whether discrete actions
should be provided as categorical indices or one-hot encodings.
Defaults to `True`.
- **seed** (*int**,**optional*) - the seed to use to reset the environment on
the first call to [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset). Defaults to `None`.

Examples

```
>>> import gymnasium
>>> import lbforaging
>>> from torchrl.envs.libs.lbforaging import LBForagingWrapper
>>> base_env = gymnasium.make("Foraging-8x8-2p-3f-v3")
>>> env = LBForagingWrapper(base_env, categorical_actions=False)
>>> env.rollout(3)
TensorDict(
 fields={
 agents: TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 2, 6]), device=cpu, dtype=torch.int64, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 2, 15]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3, 2]),
 device=None,
 is_shared=False),
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 agents: TensorDict(
 fields={
 observation: Tensor(shape=torch.Size([3, 2, 15]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([3, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3, 2]),
 device=None,
 is_shared=False),
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False),
 terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```