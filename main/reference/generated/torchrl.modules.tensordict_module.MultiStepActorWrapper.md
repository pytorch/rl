# MultiStepActorWrapper

*class*torchrl.modules.tensordict_module.MultiStepActorWrapper(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#MultiStepActorWrapper)

A wrapper around a multi-action actor.

This class enables macros to be executed in an environment.
The actor action(s) entry must have an additional time dimension to
be consumed. It must be placed adjacent to the last dimension of the
input tensordict (i.e. at `tensordict.ndim`).

The action entry keys are retrieved automatically from the actor if
not provided using a simple heuristic (any nested key ending with the
`"action"` string).

An `"is_init"` entry must also be present in the input tensordict
to track which and when the current collection should be interrupted
because a "done" state has been encountered. Unlike `action_keys`,
this key must be unique.

Parameters:

- **actor** (*TensorDictModuleBase*) - An actor.
- **n_steps** (*int**,**optional*) - the number of actions the actor outputs at once
(lookahead window). Defaults to None.

Keyword Arguments:

- **action_keys** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**NestedKeys**,**optional*) - the action keys from
the environment. Can be retrieved from `env.action_keys`.
Defaults to all `out_keys` of the `actor` which end
with the `"action"` string.
- **init_key** (*NestedKey**,**optional*) - the key of the entry indicating
when the environment has gone through a reset.
Defaults to `"is_init"` which is the `out_key` from the
[`InitTracker`](torchrl.envs.transforms.InitTracker.html#torchrl.envs.transforms.InitTracker) transform.
- **keep_dim** (*bool**,**optional*) - whether to keep the time dimension of
the macro during indexing. Defaults to `False`.
- **replan_interval** (*int**,**optional*) - re-query the wrapped actor after this
many actions have been consumed from the cache (receding-horizon
execution; the actor call is skipped in between, which is the
point of action chunking for expensive policies such as VLAs).
Must be in `[1, n_steps]`; `replan_interval=1` re-plans at
every step (closed loop). Defaults to `None`, i.e. the whole
cache is consumed before re-querying (open loop). With
`n_steps=None` the bound is enforced at execution time against
the actual chunk length instead.

Examples

```
>>> import torch.nn
>>> from torchrl.modules.tensordict_module.actors import MultiStepActorWrapper, Actor
>>> from torchrl.envs import CatFrames, GymEnv, TransformedEnv, SerialEnv, InitTracker, Compose
>>> from tensordict.nn import TensorDictSequential as Seq, TensorDictModule as Mod
>>>
>>> time_steps = 6
>>> n_obs = 4
>>> n_action = 2
>>> batch = 5
>>>
>>> # Transforms a CatFrames in a stack of frames
>>> def reshape_cat(data: torch.Tensor):
... return data.unflatten(-1, (time_steps, n_obs))
>>> # an actor that reads `time_steps` frames and outputs one action per frame
>>> # (actions are conditioned on the observation of `time_steps` in the past)
>>> actor_base = Seq(
... Mod(reshape_cat, in_keys=["obs_cat"], out_keys=["obs_cat_reshape"]),
... Mod(torch.nn.Linear(n_obs, n_action), in_keys=["obs_cat_reshape"], out_keys=["action"])
... )
>>> # Wrap the actor to dispatch the actions
>>> actor = MultiStepActorWrapper(actor_base, n_steps=time_steps)
>>>
>>> env = TransformedEnv(
... SerialEnv(batch, lambda: GymEnv("CartPole-v1")),
... Compose(
... InitTracker(),
... CatFrames(N=time_steps, in_keys=["observation"], out_keys=["obs_cat"], dim=-1)
... )
... )
>>>
>>> print(env.rollout(100, policy=actor, break_when_any_done=False))
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([5, 100, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 action_orig: Tensor(shape=torch.Size([5, 100, 6, 2]), device=cpu, dtype=torch.float32, is_shared=False),
 counter: Tensor(shape=torch.Size([5, 100, 1]), device=cpu, dtype=torch.int32, is_shared=False),
 done: Tensor(shape=torch.Size([5, 100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 is_init: Tensor(shape=torch.Size([5, 100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([5, 100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 is_init: Tensor(shape=torch.Size([5, 100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 obs_cat: Tensor(shape=torch.Size([5, 100, 24]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([5, 100, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([5, 100, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([5, 100, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5, 100]),
 device=cpu,
 is_shared=False),
 obs_cat: Tensor(shape=torch.Size([5, 100, 24]), device=cpu, dtype=torch.float32, is_shared=False),
 observation: Tensor(shape=torch.Size([5, 100, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([5, 100, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([5, 100, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([5, 100]),
 device=cpu,
 is_shared=False)
```

See also

`torchrl.envs.MultiStepEnvWrapper` is the EnvBase alter-ego of this wrapper:
It wraps an environment and unbinds the action, executing it one element at a time.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#MultiStepActorWrapper.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

*property*init_key*: NestedKey*

The indicator of the initial step for a given element of the batch.