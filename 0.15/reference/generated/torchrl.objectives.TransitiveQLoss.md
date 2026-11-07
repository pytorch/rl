# TransitiveQLoss

*class*torchrl.objectives.TransitiveQLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/transitive_q.html#TransitiveQLoss)

Transitive RL loss for offline goal-conditioned RL.

Transitive RL learns a goal-conditioned critic
\(Q(s, a, g) \in (0, 1]\), read as \(\gamma^{d(s, g)}\) for the
temporal distance \(d\), with a divide-and-conquer update that
replaces the one-step temporal-difference backup with the triangle
inequality:

\[Q(s_i, a_i, s_j) \leftarrow
\bar{Q}(s_i, a_i, s_k)\,\bar{Q}(s_k, a_k, s_j),
\qquad i \le k < j,\]

where \(s_k\) is a state the dataset trajectory visited between
\(s_i\) and \(s_j\) and \(\bar{Q}\) is a target critic. A
factor whose horizon is at most one step is replaced by its exact value
\(\gamma^{k - i}\) (resp. \(\gamma^{j - k}\)). The critic is fit
with an expectile-weighted binary cross-entropy, optionally re-weighted by
\((1 + \log_\gamma \text{target})^{-\lambda}\) to favour short chunks.

Reference: Park, Oberai, Atreya, Levine. "Transitive RL: Value Learning
via Divide and Conquer." ICLR 2026. [https://arxiv.org/abs/2510.22512](https://arxiv.org/abs/2510.22512)

See also [`TransitiveQLossConfig`](torchrl.trainers.algorithms.configs.objectives.TransitiveQLossConfig.html#torchrl.trainers.algorithms.configs.objectives.TransitiveQLossConfig).

The loss reads batches of shape `[*B, 4]` whose last dimension holds,
in this order, the anchor record \(s_i\), the subgoal record
\(s_k\), the goal record \(s_j\) and a policy-goal record. The
flat `[4 * B]` batches sampled with
`SubgoalTriplet` are accepted as is: a
one-dimensional batch is read as consecutive groups of four records, so it
must follow the unit's layout. Each record carries its `"step_offset"`
from the anchor, and a batch without that entry is rejected.

The goal fed to the critic is read from the goal record at the
`achieved_goal` key (the full observation by default), so that the
triangle inequality holds in the critic's state space. When the policy is
conditioned on another goal representation (`policy_goal` key), an
`oracle_qvalue_network` reading that representation is distilled from
the critic and used for policy extraction.

The policy is extracted with reparameterized gradients: the actor
maximizes the (ensemble minimum of the) critic logits, normalized by their
mean absolute value, plus `alpha` times the behavior log-likelihood of
the dataset action.

Anchors whose [`loss_mask_key`](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule.loss_mask_key) entry
is `False` are left out of every term. It defaults to the
`"validity_mask"` written by
`SubgoalTriplet`. The loss assumes
uniform anchor sampling: it neither applies prioritized-replay importance
weights nor writes a priority entry.

Parameters:

- **actor_network** (*TensorDictModuleBase*) - a probabilistic actor reading the
`observation` and `goal` keys and supporting `get_dist`.
- **qvalue_network** (*TensorDictModule**or**list**of**TensorDictModule*) - the
critic, reading the `observation`, `goal` and `action` keys
and writing the `value` key as a logit. A list provides the
ensemble members (with their own initialization); a single module
is expanded `num_qvalue_nets` times.

Keyword Arguments:

- **oracle_qvalue_network** (*TensorDictModule**or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**TensorDictModule**,**optional*) - a critic with the same keys as `qvalue_network` whose `goal`
entry carries the `policy_goal` representation, with as many
members as the critic ensemble. Each member is trained to match the
critic member of the same index, and the oracle replaces the critic
for policy extraction. Defaults to
`None` (the policy is extracted from the critic, and
`policy_goal` must match `achieved_goal`).
- **num_qvalue_nets** (*int**,**optional*) - ensemble size used when a single
critic module is given. Defaults to `2`.
- **discount** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - the discount \(\gamma\) defining the
value scale. Defaults to `0.99`.
- **expectile** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - expectile \(\kappa\) of the critic
regression; `0.5` gives a behavioral critic. Defaults to `0.7`.
- **distance_weight** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - exponent \(\lambda\) of the
distance-based re-weighting; `0` disables it. Defaults to `0`.
- **alpha** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - weight of the behavior log-likelihood in the
actor loss. Defaults to `1.0`.
- **actor_action** (*Literal**[**"mode"**,**"rsample"**]**,**optional*) - the policy action
scored by the critic: the distribution mode or a reparameterized
sample. Actions are clamped to `[-1, 1]`. Defaults to `"mode"`.

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules import IndependentNormal, ProbabilisticActor
>>> from torchrl.objectives import TransitiveQLoss
>>> class Critic(nn.Module):
... def __init__(self):
... super().__init__()
... self.net = nn.Linear(3 + 3 + 2, 1)
... def forward(self, obs, goal, action):
... return self.net(torch.cat([obs, goal, action], -1))
>>> class Actor(nn.Module):
... def __init__(self):
... super().__init__()
... self.net = nn.Linear(3 + 3, 2)
... def forward(self, obs, goal):
... loc = self.net(torch.cat([obs, goal], -1))
... return loc, torch.ones_like(loc)
>>> actor = ProbabilisticActor(
... TensorDictModule(Actor(), ["observation", "goal"], ["loc", "scale"]),
... in_keys=["loc", "scale"],
... distribution_class=IndependentNormal,
... )
>>> critic = TensorDictModule(
... Critic(), ["observation", "goal", "action"], ["state_action_value"]
... )
>>> loss = TransitiveQLoss(actor, critic, discount=0.9)
>>> batch = TensorDict(
... {
... "observation": torch.randn(8, 4, 3),
... "action": torch.rand(8, 4, 2) * 2 - 1,
... "step_offset": torch.tensor([0, 1, 5, 3]).expand(8, 4),
... },
... batch_size=[8, 4],
... )
>>> out = loss(batch)
>>> sorted(k for k in out.keys() if k.startswith("loss"))
['loss_actor', 'loss_qvalue']
```

actor_loss(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), dict][[source]](../../_modules/torchrl/objectives/transitive_q.html#TransitiveQLoss.actor_loss)

Returns the reparameterized-gradient actor loss and metadata.

default_keys

alias of `_AcceptedKeys`

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)[[source]](../../_modules/torchrl/objectives/transitive_q.html#TransitiveQLoss.forward)

It is designed to read an input TensorDict and return another tensordict with loss keys named "loss*".

Splitting the loss in its component can then be used by the trainer to log the various loss values throughout
training. Other scalars present in the output tensordict will be logged too.

Parameters:

**tensordict** - an input tensordict with the values required to compute the loss.

Returns:

A new tensordict with no batch dimension containing various loss scalars which will be named "loss*". It
is essential that the losses are returned with this name as they will be read by the trainer before
backpropagation.

*property*in_keys

The keys that the module reads from the input tensordict.

qvalue_loss(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None, dict][[source]](../../_modules/torchrl/objectives/transitive_q.html#TransitiveQLoss.qvalue_loss)

Returns the critic loss, the oracle distillation loss and metadata.