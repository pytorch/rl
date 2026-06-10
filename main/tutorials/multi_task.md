Note

Go to the end
to download the full example code.

# Task-specific policy in multi-task environments

This tutorial details how multi-task policies and batched environments can be used.

At the end of this tutorial, you will be capable of writing policies that
can compute actions in diverse settings using a distinct set of weights.
You will also be able to execute diverse environments in parallel.

```
from tensordict import LazyStackedTensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential
from torch import nn
```

```
from torchrl.envs import CatTensors, Compose, DoubleToFloat, ParallelEnv, TransformedEnv
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.modules import MLP
```

We design two environments, one humanoid that must complete the stand task
and another that must learn to walk.

```
env1 = DMControlEnv("humanoid", "stand")
env1_obs_keys = list(env1.observation_spec.keys())
env1 = TransformedEnv(
 env1,
 Compose(
 CatTensors(env1_obs_keys, "observation_stand", del_keys=False),
 CatTensors(env1_obs_keys, "observation"),
 DoubleToFloat(
 in_keys=["observation_stand", "observation"],
 in_keys_inv=["action"],
 ),
 ),
)
env2 = DMControlEnv("humanoid", "walk")
env2_obs_keys = list(env2.observation_spec.keys())
env2 = TransformedEnv(
 env2,
 Compose(
 CatTensors(env2_obs_keys, "observation_walk", del_keys=False),
 CatTensors(env2_obs_keys, "observation"),
 DoubleToFloat(
 in_keys=["observation_walk", "observation"],
 in_keys_inv=["action"],
 ),
 ),
)
```

```
tdreset1 = env1.reset()
tdreset2 = env2.reset()

# With LazyStackedTensorDict, stacking is done in a lazy manner: the original tensordicts
# can still be recovered by indexing the main tensordict
tdreset = LazyStackedTensorDict.lazy_stack([tdreset1, tdreset2], 0)
assert tdreset[0] is tdreset1
```

```
print(tdreset[0])
```

```
TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_stand: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

## Policy

We will design a policy where a backbone reads the "observation" key.
Then specific sub-components will read the "observation_stand" and
"observation_walk" keys of the stacked tensordicts, if they are present,
and pass them through the dedicated sub-network.

```
action_dim = env1.action_spec.shape[-1]
```

```
policy_common = TensorDictModule(
 nn.Linear(67, 64), in_keys=["observation"], out_keys=["hidden"]
)
policy_stand = TensorDictModule(
 MLP(67 + 64, action_dim, depth=2),
 in_keys=["observation_stand", "hidden"],
 out_keys=["action"],
)
policy_walk = TensorDictModule(
 MLP(67 + 64, action_dim, depth=2),
 in_keys=["observation_walk", "hidden"],
 out_keys=["action"],
)
seq = TensorDictSequential(
 policy_common, policy_stand, policy_walk, partial_tolerant=True
)
```

Let's check that our sequence outputs actions for a single env (stand).

```
seq(env1.reset())
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([21]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 hidden: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_stand: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Let's check that our sequence outputs actions for a single env (walk).

```
seq(env2.reset())
```

```
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([21]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 hidden: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_walk: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This also works with the stack: now the stand and walk keys have
disappeared, because they're not shared by all tensordicts. But the
`TensorDictSequential` still performed the operations. Note that the
backbone was executed in a vectorized way - not in a loop - which is more efficient.

```
seq(tdreset)
```

```
LazyStackedTensorDict(
 fields={
 action: Tensor(shape=torch.Size([2, 21]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 hidden: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([2, 67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 exclusive_fields={
 0 ->
 observation_stand: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 1 ->
 observation_walk: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False,
 stack_dim=0)
```

## Executing diverse tasks in parallel

We can parallelize the operations if the common key-value pairs share the
same specs (in particular their shape and dtype must match: you can't do the
following if the observation shapes are different but are pointed to by the
same key).

If ParallelEnv receives a single env making function, it will assume that
a single task has to be performed. If a list of functions is provided, then
it will assume that we are in a multi-task setting.

```
def env1_maker():
 return TransformedEnv(
 DMControlEnv("humanoid", "stand"),
 Compose(
 CatTensors(env1_obs_keys, "observation_stand", del_keys=False),
 CatTensors(env1_obs_keys, "observation"),
 DoubleToFloat(
 in_keys=["observation_stand", "observation"],
 in_keys_inv=["action"],
 ),
 ),
 )

def env2_maker():
 return TransformedEnv(
 DMControlEnv("humanoid", "walk"),
 Compose(
 CatTensors(env2_obs_keys, "observation_walk", del_keys=False),
 CatTensors(env2_obs_keys, "observation"),
 DoubleToFloat(
 in_keys=["observation_walk", "observation"],
 in_keys_inv=["action"],
 ),
 ),
 )

env = ParallelEnv(2, [env1_maker, env2_maker], mp_start_method=mp_context)
assert not env._single_task

tdreset = env.reset()
print(tdreset)
print(tdreset[0])
print(tdreset[1]) # should be different
```

```
LazyStackedTensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([2, 67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 exclusive_fields={
 0 ->
 observation_stand: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 1 ->
 observation_walk: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False,
 stack_dim=0)
TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_stand: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_walk: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Let's pass the output through our network.

```
tdreset = seq(tdreset)
print(tdreset)
print(tdreset[0])
print(tdreset[1]) # should be different but all have an "action" key

env.step(tdreset) # computes actions and execute steps in parallel
print(tdreset)
print(tdreset[0])
print(tdreset[1]) # next_observation has now been written
```

```
LazyStackedTensorDict(
 fields={
 action: Tensor(shape=torch.Size([2, 21]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 hidden: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([2, 67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 exclusive_fields={
 0 ->
 observation_stand: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 1 ->
 observation_walk: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False,
 stack_dim=0)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([21]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 hidden: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_stand: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([21]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 hidden: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_walk: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
LazyStackedTensorDict(
 fields={
 action: Tensor(shape=torch.Size([2, 21]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 hidden: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.float32, is_shared=False),
 next: LazyStackedTensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([2, 67]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float64, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 exclusive_fields={
 0 ->
 observation_stand: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 1 ->
 observation_walk: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False,
 stack_dim=0),
 observation: Tensor(shape=torch.Size([2, 67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 exclusive_fields={
 0 ->
 observation_stand: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 1 ->
 observation_walk: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False,
 stack_dim=0)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([21]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 hidden: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_stand: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float64, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_stand: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([21]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 hidden: Tensor(shape=torch.Size([64]), device=cpu, dtype=torch.float32, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_walk: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float64, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_walk: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

## Rollout

```
td_rollout = env.rollout(100, policy=seq, return_contiguous=False)
```

```
td_rollout[:, 0] # tensordict of the first step: only the common keys are shown
```

```
LazyStackedTensorDict(
 fields={
 action: Tensor(shape=torch.Size([2, 21]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 hidden: Tensor(shape=torch.Size([2, 64]), device=cpu, dtype=torch.float32, is_shared=False),
 next: LazyStackedTensorDict(
 fields={
 done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([2, 67]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float64, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 exclusive_fields={
 0 ->
 observation_stand: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 1 ->
 observation_walk: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False,
 stack_dim=0),
 observation: Tensor(shape=torch.Size([2, 67]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 exclusive_fields={
 0 ->
 observation_stand: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False),
 1 ->
 observation_walk: Tensor(shape=torch.Size([67]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False,
 stack_dim=0)
```

```
td_rollout[0] # tensordict of the first env: the stand obs is present

env.close()
del env
```

**Total running time of the script:** (0 minutes 4.199 seconds)

[`Download Jupyter notebook: multi_task.ipynb`](../_downloads/9b5fe5a65e76c28e02edf59221b0dc86/multi_task.ipynb)

[`Download Python source code: multi_task.py`](../_downloads/fe94dedecc2afb14d9227db6afb81c3f/multi_task.py)

[`Download zipped: multi_task.zip`](../_downloads/53d45db6c317a5dacd743dd31c55d6e3/multi_task.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)