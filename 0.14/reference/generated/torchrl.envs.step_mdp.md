# step_mdp

torchrl.envs.step_mdp(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *next_tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*, *keep_other: bool = True*, *exclude_reward: bool = True*, *exclude_done: bool = False*, *exclude_action: bool = True*, *reward_keys: NestedKey | list[NestedKey] = 'reward'*, *done_keys: NestedKey | list[NestedKey] = 'done'*, *action_keys: NestedKey | list[NestedKey] = 'action'*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/utils.html#step_mdp)

Creates a new tensordict that reflects a step in time of the input tensordict.

Given a tensordict retrieved after a step, returns the `"next"` indexed-tensordict.
The arguments allow for precise control over what should be kept and what
should be copied from the `"next"` entry. The default behavior is:
move the observation entries, reward, and done states to the root, exclude
the current action, and keep all extra keys (non-action, non-done, non-reward).

Parameters:

- **tensordict** (*TensorDictBase*) - The tensordict with keys to be renamed.
- **next_tensordict** (*TensorDictBase**,**optional*) - The destination tensordict. If None, a new tensordict is created.
- **keep_other** (*bool**,**optional*) - If `True`, all keys that do not start with `'next_'` will be kept.
Default is `True`.
- **exclude_reward** (*bool**,**optional*) - If `True`, the `"reward"` key will be discarded
from the resulting tensordict. If `False`, it will be copied (and replaced)
from the `"next"` entry (if present). Default is `True`.
- **exclude_done** (*bool**,**optional*) - If `True`, the `"done"` key will be discarded
from the resulting tensordict. If `False`, it will be copied (and replaced)
from the `"next"` entry (if present). Default is `False`.
- **exclude_action** (*bool**,**optional*) - If `True`, the `"action"` key will
be discarded from the resulting tensordict. If `False`, it will
be kept in the root tensordict (since it should not be present in
the `"next"` entry). Default is `True`.
- **reward_keys** (*NestedKey**or**list**of**NestedKey**,**optional*) - The keys where the reward is written. Defaults
to "reward".
- **done_keys** (*NestedKey**or**list**of**NestedKey**,**optional*) - The keys where the done is written. Defaults
to "done".
- **action_keys** (*NestedKey**or**list**of**NestedKey**,**optional*) - The keys where the action is written. Defaults
to "action".

Returns:

A new tensordict (or next_tensordict if provided) containing the tensors of the t+1 step.

Return type:

TensorDictBase

See also

[`EnvBase.step_mdp()`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase.step_mdp) is the class-based version of this free function. It will attempt to cache the
key values to reduce the overhead of making a step in the MDP.

Examples

```
>>> from tensordict import TensorDict
>>> import torch
>>> td = TensorDict({
... "done": torch.zeros((), dtype=torch.bool),
... "reward": torch.zeros(()),
... "extra": torch.zeros(()),
... "next": TensorDict({
... "done": torch.zeros((), dtype=torch.bool),
... "reward": torch.zeros(()),
... "obs": torch.zeros(()),
... }, []),
... "obs": torch.zeros(()),
... "action": torch.zeros(()),
... }, [])
>>> print(step_mdp(td))
TensorDict(
 fields={
 done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
 extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> print(step_mdp(td, exclude_done=True)) # "done" is dropped
TensorDict(
 fields={
 extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> print(step_mdp(td, exclude_reward=False)) # "reward" is kept
TensorDict(
 fields={
 done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
 extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> print(step_mdp(td, exclude_action=False)) # "action" persists at the root
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
 extra: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> print(step_mdp(td, keep_other=False)) # "extra" is missing
TensorDict(
 fields={
 done: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.bool, is_shared=False),
 obs: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

Warning

This function will not work properly if the reward key is also part of the input key when
the reward keys are excluded. This is why the `RewardSum` transform registers
the episode reward in the observation and not the reward spec by default.
When using the fast, cached version of this function (`_StepMDP`), this issue should not
be observed.