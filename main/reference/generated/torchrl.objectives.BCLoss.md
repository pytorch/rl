# BCLoss

*class*torchrl.objectives.BCLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/bc.html#BCLoss)

Behavior Cloning Loss Module.

Implements behavior cloning loss for both stochastic and deterministic policies.
Minimizes the negative log-likelihood: -E[log π(a_expert | s)] where π is the
policy being trained and a_expert are the expert actions from the demonstration dataset.

Works with any actor network that implements `get_dist()`
method, including both
stochastic and deterministic policies.

Reference:

"Integrating Behavior Cloning and Reinforcement Learning for Improved
Performance in Dense and Sparse Reward Environments"
[https://arxiv.org/abs/1910.04281](https://arxiv.org/abs/1910.04281)

Parameters:

**actor_network** (*TensorDictModule*) - the actor network to be trained.

Keyword Arguments:

**reduction** (*str**,**optional*) - Specifies the reduction to apply to the output:
`"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied,
`"mean"`: the sum of the output will be divided by the number of
elements in the output, `"sum"`: the output will be summed. Default: `"mean"`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data.tensor_specs import Bounded
>>> from torchrl.modules.tensordict_module.actors import Actor
>>> from torchrl.objectives.bc import BCLoss
>>> from tensordict import TensorDict
>>> n_act, n_obs = 4, 3
>>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
>>> module = nn.Linear(n_obs, n_act)
>>> actor = Actor(module=module, spec=spec)
>>> loss = BCLoss(actor)
>>> batch = [2, ]
>>> data = TensorDict({
... "observation": torch.randn(*batch, n_obs),
... "action": spec.rand(batch),
... }, batch)
>>> loss(data)
TensorDict(
 fields={
 loss_bc: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This class is compatible with non-tensordict based modules too and can be
used without recurring to any tensordict-related primitive. In this case,
the expected keyword arguments are the actor's `in_keys` + `["action"]`.
The return value is a tensor corresponding to the loss.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data.tensor_specs import Bounded
>>> from torchrl.modules.tensordict_module.actors import Actor
>>> from torchrl.objectives.bc import BCLoss
>>> n_act, n_obs = 4, 3
>>> spec = Bounded(-torch.ones(n_act), torch.ones(n_act), (n_act,))
>>> module = nn.Linear(n_obs, n_act)
>>> actor = Actor(module=module, spec=spec)
>>> loss = BCLoss(actor)
>>> _ = loss.select_out_keys("loss_bc")
>>> batch = [2, ]
>>> loss_bc = loss(
... observation=torch.randn(*batch, n_obs),
... action=spec.rand(batch))
>>> loss_bc.backward()
```

Chunked (VLA-style) behavior cloning is the same loss with the action
chunk as the `action` and the padding mask excluded via `pad_mask`:

Examples

```
>>> from tensordict.nn import TensorDictModule
>>> chunk_actor = TensorDictModule(
... nn.Sequential(nn.Linear(n_obs, 8), nn.Unflatten(-1, (2, 4))),
... in_keys=["observation"],
... out_keys=["action_chunk"],
... )
>>> loss = BCLoss(chunk_actor, loss_function="l1")
>>> loss.set_keys(action="action_chunk", pad_mask="action_is_pad")
>>> data = TensorDict({
... "observation": torch.randn(2, n_obs),
... "action_chunk": torch.randn(2, 2, 4),
... "action_is_pad": torch.tensor([[False, False], [False, True]]),
... }, [2])
>>> loss(data)["loss_bc"].shape
torch.Size([])
```

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/objectives/bc.html#BCLoss.forward)

Compute the behavior cloning loss.

Parameters:

**tensordict** (*TensorDictBase*) - input data containing observations and expert actions.

Returns:

TensorDict with key "loss_bc".

set_keys(***kwargs*) → None[[source]](../../_modules/torchrl/objectives/bc.html#BCLoss.set_keys)

Set tensordict key names.

Examples

```
>>> from torchrl.objectives import DQNLoss
>>> # initialize the DQN loss
>>> actor = torch.nn.Linear(3, 4)
>>> dqn_loss = DQNLoss(actor, action_space="one-hot")
>>> dqn_loss.set_keys(priority_key="td_error", action_value_key="action_value")
```