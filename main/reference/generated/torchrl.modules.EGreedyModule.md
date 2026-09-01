# EGreedyModule

*class*torchrl.modules.EGreedyModule(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/tensordict_module/exploration.html#EGreedyModule)

Epsilon-Greedy exploration module.

This module randomly updates the action(s) in a tensordict given an epsilon greedy exploration strategy.
At each call, random draws (one per action) are executed given a certain probability threshold. If successful,
the corresponding actions are being replaced by random samples drawn from the action spec provided.
Others are left unchanged.

Parameters:

- **spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - the spec used for sampling actions.
- **eps_init** (*scalar**,**optional*) - initial epsilon value.
default: 1.0
- **eps_end** (*scalar**,**optional*) - final epsilon value.
default: 0.1
- **annealing_num_steps** (*int**,**optional*) - number of steps it will take for epsilon to reach
the `eps_end` value. Defaults to 1000.

Keyword Arguments:

- **action_key** (*NestedKey**,**optional*) - the key where the action can be found in the input tensordict.
Default is `"action"`.
- **action_mask_key** (*NestedKey**,**optional*) - the key where the action mask can be found in the input tensordict.
Default is `None` (corresponding to no mask).
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - the device of the exploration module.

Note

It is crucial to incorporate a call to `step()` in the training loop
to update the exploration factor.
Since it is not easy to capture this omission no warning or exception
will be raised if this is omitted!

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictSequential
>>> from torchrl.modules import EGreedyModule, Actor
>>> from torchrl.data import Bounded
>>> torch.manual_seed(0)
>>> spec = Bounded(-1, 1, torch.Size([4]))
>>> module = torch.nn.Linear(4, 4, bias=False)
>>> policy = Actor(spec=spec, module=module)
>>> explorative_policy = TensorDictSequential(policy, EGreedyModule(eps_init=0.2))
>>> td = TensorDict({"observation": torch.zeros(10, 4)}, batch_size=[10])
>>> print(explorative_policy(td).get("action"))
tensor([[ 0.0000, 0.0000, 0.0000, 0.0000],
 [ 0.0000, 0.0000, 0.0000, 0.0000],
 [ 0.9055, -0.9277, -0.6295, -0.2532],
 [ 0.0000, 0.0000, 0.0000, 0.0000],
 [ 0.0000, 0.0000, 0.0000, 0.0000],
 [ 0.0000, 0.0000, 0.0000, 0.0000],
 [ 0.0000, 0.0000, 0.0000, 0.0000],
 [ 0.0000, 0.0000, 0.0000, 0.0000],
 [ 0.0000, 0.0000, 0.0000, 0.0000],
 [ 0.0000, 0.0000, 0.0000, 0.0000]], grad_fn=<AddBackward0>)
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/tensordict_module/exploration.html#EGreedyModule.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

step(*frames: int = 1*) → None[[source]](../../_modules/torchrl/modules/tensordict_module/exploration.html#EGreedyModule.step)

A step of epsilon decay.

After self.annealing_num_steps calls to this method, calls result in no-op.

Parameters:

**frames** (*int**,**optional*) - number of frames since last step. Defaults to `1`.