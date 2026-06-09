# DiscreteIQLLoss

*class*torchrl.objectives.DiscreteIQLLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/iql.html#DiscreteIQLLoss)

TorchRL implementation of the discrete IQL loss.

Presented in "Offline Reinforcement Learning with Implicit Q-Learning" [https://arxiv.org/abs/2110.06169](https://arxiv.org/abs/2110.06169)

Parameters:

- **actor_network** (*ProbabilisticTensorDictSequential*) - stochastic actor
- **qvalue_network** (*TensorDictModule*) - Q(s, a) parametric model.
- **value_network** (*TensorDictModule**,**optional*) - V(s) parametric model.

Keyword Arguments:

- **action_space** (*str**or*[*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - Action space. Must be one of
`"one-hot"`, `"mult_one_hot"`, `"binary"` or `"categorical"`,
or an instance of the corresponding specs ([`torchrl.data.OneHot`](torchrl.data.OneHot.html#torchrl.data.OneHot),
[`torchrl.data.MultiOneHot`](torchrl.data.MultiOneHot.html#torchrl.data.MultiOneHot),
[`torchrl.data.Binary`](torchrl.data.Binary.html#torchrl.data.Binary) or [`torchrl.data.Categorical`](torchrl.data.Categorical.html#torchrl.data.Categorical)).
- **num_qvalue_nets** (*integer**,**optional*) - number of Q-Value networks used.
Defaults to `2`.
- **loss_function** (*str**,**optional*) - loss function to be used with
the value function loss. Default is "smooth_l1".
- **temperature** (`float`, optional) - Inverse temperature (beta).
For smaller hyperparameter values, the objective behaves similarly to
behavioral cloning, while for larger values, it attempts to recover the
maximum of the Q-function.
- **expectile** (`float`, optional) - expectile \(\tau\). A larger value of \(\tau\) is crucial
for antmaze tasks that require dynamical programming ("stichting").
- **priority_key** (*str**,**optional*) - [Deprecated, use .set_keys(priority_key=priority_key) instead]
tensordict key where to write the priority (for prioritized replay
buffer usage). Default is "td_error".
- **separate_losses** (*bool**,**optional*) - if `True`, shared parameters between
policy and critic will only be trained on the policy loss.
Defaults to `False`, i.e., gradients are propagated to shared
parameters for both policy and critic losses.
- **reduction** (*str**,**optional*) - Specifies the reduction to apply to the output:
`"none"` | `"mean"` | `"sum"`. `"none"`: no reduction will be applied,
`"mean"`: the sum of the output will be divided by the number of
elements in the output, `"sum"`: the output will be summed. Default: `"mean"`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from torchrl.data.tensor_specs import OneHot
>>> from torchrl.modules.distributions.discrete import OneHotCategorical
>>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives.iql import DiscreteIQLLoss
>>> from tensordict import TensorDict
>>> n_act, n_obs = 4, 3
>>> spec = OneHot(n_act)
>>> module = SafeModule(nn.Linear(n_obs, n_act), in_keys=["observation"], out_keys=["logits"])
>>> actor = ProbabilisticActor(
... module=module,
... in_keys=["logits"],
... out_keys=["action"],
... spec=spec,
... distribution_class=OneHotCategorical)
>>> qvalue = SafeModule(
... nn.Linear(n_obs, n_act),
... in_keys=["observation"],
... out_keys=["state_action_value"],
... )
>>> value = SafeModule(
... nn.Linear(n_obs, 1),
... in_keys=["observation"],
... out_keys=["state_value"],
... )
>>> loss = DiscreteIQLLoss(actor, qvalue, value)
>>> batch = [2, ]
>>> action = spec.rand(batch).long()
>>> data = TensorDict({
... "observation": torch.randn(*batch, n_obs),
... "action": action,
... ("next", "done"): torch.zeros(*batch, 1, dtype=torch.bool),
... ("next", "terminated"): torch.zeros(*batch, 1, dtype=torch.bool),
... ("next", "reward"): torch.randn(*batch, 1),
... ("next", "observation"): torch.randn(*batch, n_obs),
... }, batch)
>>> loss(data)
TensorDict(
 fields={
 entropy: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_actor: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_qvalue: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 loss_value: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

This class is compatible with non-tensordict based modules too and can be
used without recurring to any tensordict-related primitive. In this case,
the expected keyword arguments are:
`["action", "next_reward", "next_done", "next_terminated"]` + in_keys of the actor, value, and qvalue network
The return value is a tuple of tensors in the following order:
`["loss_actor", "loss_qvalue", "loss_value", "entropy"]`.

Examples

```
>>> import torch
>>> import torch
>>> from torch import nn
>>> from torchrl.data.tensor_specs import OneHot
>>> from torchrl.modules.distributions.discrete import OneHotCategorical
>>> from torchrl.modules.tensordict_module.actors import ProbabilisticActor
>>> from torchrl.modules.tensordict_module.common import SafeModule
>>> from torchrl.objectives.iql import DiscreteIQLLoss
>>> _ = torch.manual_seed(42)
>>> n_act, n_obs = 4, 3
>>> spec = OneHot(n_act)
>>> module = SafeModule(nn.Linear(n_obs, n_act), in_keys=["observation"], out_keys=["logits"])
>>> actor = ProbabilisticActor(
... module=module,
... in_keys=["logits"],
... out_keys=["action"],
... spec=spec,
... distribution_class=OneHotCategorical)
>>> qvalue = SafeModule(
... nn.Linear(n_obs, n_act),
... in_keys=["observation"],
... out_keys=["state_action_value"],
... )
>>> value = SafeModule(
... nn.Linear(n_obs, 1),
... in_keys=["observation"],
... out_keys=["state_value"],
... )
>>> loss = DiscreteIQLLoss(actor, qvalue, value)
>>> batch = [2, ]
>>> action = spec.rand(batch).long()
>>> loss_actor, loss_qvalue, loss_value, entropy = loss(
... observation=torch.randn(*batch, n_obs),
... action=action,
... next_done=torch.zeros(*batch, 1, dtype=torch.bool),
... next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
... next_observation=torch.zeros(*batch, n_obs),
... next_reward=torch.randn(*batch, 1))
>>> loss_actor.backward()
```

The output keys can also be filtered using the `DiscreteIQLLoss.select_out_keys()`
method.

Examples

```
>>> _ = loss.select_out_keys('loss_actor', 'loss_qvalue', 'loss_value')
>>> loss_actor, loss_qvalue, loss_value = loss(
... observation=torch.randn(*batch, n_obs),
... action=action,
... next_done=torch.zeros(*batch, 1, dtype=torch.bool),
... next_terminated=torch.zeros(*batch, 1, dtype=torch.bool),
... next_observation=torch.zeros(*batch, n_obs),
... next_reward=torch.randn(*batch, 1))
>>> loss_actor.backward()
```

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.