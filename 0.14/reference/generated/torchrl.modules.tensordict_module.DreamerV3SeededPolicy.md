# DreamerV3SeededPolicy

*class*torchrl.modules.tensordict_module.DreamerV3SeededPolicy(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#DreamerV3SeededPolicy)

Run a DreamerV3 policy with an independent, checkpointable random stream.

Each call derives a seed from the initial seed and call count, restoring the
caller's torch RNG state afterwards. The seed and count are included in the
module's `state_dict` alongside its parameters. Calls must be serialized;
Python seeding is not compatible with CUDA-graph capture of this wrapper.

Parameters:

- **module** (*TensorDictModuleBase*) - Policy to execute, with declared input keys.
- **seed** (*int*) - Initial non-negative seed for the policy stream.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.modules import DreamerV3DiscreteActor, DreamerV3SeededPolicy
>>> actor = DreamerV3DiscreteActor(6, 3, depth=1, num_cells=8)
>>> policy = DreamerV3SeededPolicy(actor, seed=7)
>>> data = TensorDict({"state": torch.zeros(2, 4), "belief": torch.zeros(2, 2)}, [2])
>>> _ = policy(data.clone())
>>> saved = policy.state_dict()
>>> expected = policy(data.clone())["action"]
>>> _ = policy.load_state_dict(saved)
>>> torch.equal(policy(data.clone())["action"], expected)
True
```

See also

`DreamerV3SeededPolicyConfig`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#DreamerV3SeededPolicy.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

get_extra_state() → dict[str, int][[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#DreamerV3SeededPolicy.get_extra_state)

Return the policy's seed and call count for module checkpointing.

reset_counter() → None[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#DreamerV3SeededPolicy.reset_counter)

Restart the counter, because a setup call can move it before step 0.

set_extra_state(*state: Mapping[str, int]*) → None[[source]](../../_modules/torchrl/modules/tensordict_module/actors.html#DreamerV3SeededPolicy.set_extra_state)

Restore the next policy draw without changing the caller's RNG.