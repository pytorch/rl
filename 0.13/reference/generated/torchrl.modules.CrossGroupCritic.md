# CrossGroupCritic

*class*torchrl.modules.CrossGroupCritic(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/models/cross_group_critic.html#CrossGroupCritic)

Centralised critic that conditions on observations from multiple agent groups.

Standard `MultiAgentMLP` centralises only within a
single group. `CrossGroupCritic` removes that restriction: it reads
observations from an arbitrary number of groups (each potentially with a
different observation dimensionality), encodes them to a shared embedding
space, processes the joint representation through a shared MLP trunk, and
writes a per-group value estimate back to the tensordict.

This enables two use-cases that single-group critics cannot handle:

- **Heterogeneous teams** -- agents in different groups have different
observation / action specs. Each group gets its own encoder
(`Linear(obs_dim_g → d_model)`), so no padding or obs-dim alignment
is required.
- **Ad-hoc teamwork** -- one group follows a fixed (non-training) policy
but its observations still inform the value baseline of the training
group. Pass the fixed group's name via `detach_groups` so its encoder
output is detached before building the team state: the critic sees the
full team state but gradients do not flow into the fixed group's
observations.

Because `CrossGroupCritic` is a plain [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule),
it plugs into [`MAPPOLoss`](torchrl.objectives.multiagent.MAPPOLoss.html#torchrl.objectives.multiagent.MAPPOLoss) and
[`IPPOLoss`](torchrl.objectives.multiagent.IPPOLoss.html#torchrl.objectives.multiagent.IPPOLoss) without any changes to
those classes.

Parameters:

**group_map** (*dict**[**str**,*[*CrossCriticGroupSpec*](torchrl.modules.CrossCriticGroupSpec.html#torchrl.modules.CrossCriticGroupSpec)*]*) - ordered mapping from a group name
to a [`CrossCriticGroupSpec`](torchrl.modules.CrossCriticGroupSpec.html#torchrl.modules.CrossCriticGroupSpec) that describes the group's observation
dimensionality, agent count, and tensordict keys.

Keyword Arguments:

- **d_model** (*int*) - common embedding dimension. All per-group encoders
project to this size. Defaults to `64`.
- **trunk_depth** (*int*) - number of hidden layers in the shared MLP trunk.
Defaults to `2`.
- **trunk_cells** (*int*) - width of each trunk hidden layer. Defaults to `256`.
- **activation_class** ([*type*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.type)*[**nn.Module**]*) - activation used in encoders and
trunk. Defaults to [`Tanh`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh).
- **share_params** (*bool*) - if `True` a single value head is shared across
all groups (useful when groups are homogeneous or have the same
role). If `False` each group gets its own head. Encoders are
always group-specific and the central trunk is always shared.
Defaults to `False`.
- **detach_groups** (*iterable**of**str**,**optional*) - names of groups whose encoder
outputs should be detached before the trunk. Use this to include
fixed-policy agents in the centralised state without propagating
gradients to their observations. Defaults to `None`.
- **device** (*DEVICE_TYPING**,**optional*) - device on which to allocate
parameters. Defaults to `None` (CPU).

Note

The order of keys in `group_map` determines the order of positional
inputs to the inner network. Python `dict` preserves insertion order
(Python 3.7+), so the mapping is stable.

Example

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.modules.models.cross_group_critic import CrossGroupCritic, CrossCriticGroupSpec
>>> group_map = {
... "soldiers": CrossCriticGroupSpec(obs_dim=12, n_agents=3,
... obs_key=("soldiers", "observation"),
... value_key=("soldiers", "state_value")),
... "medics": CrossCriticGroupSpec(obs_dim=8, n_agents=2,
... obs_key=("medics", "observation"),
... value_key=("medics", "state_value")),
... }
>>> critic = CrossGroupCritic(group_map, d_model=32, trunk_depth=1, trunk_cells=64)
>>> td = TensorDict(
... {
... "soldiers": {"observation": torch.zeros(4, 3, 12)},
... "medics": {"observation": torch.zeros(4, 2, 8)},
... },
... batch_size=[4],
... )
>>> td = critic(td)
>>> print(td["soldiers", "state_value"].shape)
torch.Size([4, 3, 1])
>>> print(td["medics", "state_value"].shape)
torch.Size([4, 2, 1])
```