# MAPPOLoss

*class*torchrl.objectives.multiagent.MAPPOLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/multiagent/mappo.html#MAPPOLoss)

Multi-Agent PPO loss with a centralised critic (Yu et al. 2022).

MAPPO trains a *decentralised actor* (each agent's policy conditions only
on its local observation) together with a *centralised critic* (single
value function that conditions on the full team state or concatenated
observations). The decentralised actor lets policies run independently at
execution time, while the centralised critic reduces variance during
training by giving every agent the same value baseline derived from full
state information.

This class is a thin specialisation of `ClipPPOLoss`. The
differences:

- The default value estimator is [`MultiAgentGAE`](torchrl.objectives.value.MultiAgentGAE.html#torchrl.objectives.value.MultiAgentGAE),
which broadcasts team-shared rewards / done flags along the agent
dimension before computing returns.
- `normalize_advantage_exclude_dims` defaults to `(-2,)` so the agent
dim is excluded when standardising advantages.
- An optional [`ValueNorm`](torchrl.modules.ValueNorm.html#torchrl.modules.ValueNorm) can be supplied via
`value_norm=PopArtValueNorm(shape=1)` to stabilise the critic loss;
the MAPPO paper reports this is load-bearing on SMAC (their Table 13).
[`RunningValueNorm`](torchrl.modules.RunningValueNorm.html#torchrl.modules.RunningValueNorm) is a no-decay alternative
for stationary reward scales.

Parameters:

- **actor_network** (*ProbabilisticTensorDictSequential*) - per-agent policy
operator. Conventionally built with
`MultiAgentMLP` using
`centralized=False, share_params=True` for cooperative
homogeneous teams.
- **critic_network** (*TensorDictModule*) - centralised value operator. Build
this with `MultiAgentMLP` and
`centralized=True, share_params=True`, or with any module that
consumes a global `"state"` key and returns
`("agents", "state_value")` of shape `[*B, n_agents, 1]`.

Keyword Arguments:

- **value_norm** ([*ValueNorm*](torchrl.modules.ValueNorm.html#torchrl.modules.ValueNorm)*,**optional*) - if supplied, the critic target and
prediction are normalised by this running normaliser before the
MSE / smooth-L1 distance. Composes correctly with `clip_value`
(the clip radius is applied in normalised space).
Defaults to `None` (no value norm).
- **clip_epsilon** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)) - PPO ratio clip. Defaults to `0.2`.
- **entropy_coeff** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)) - entropy bonus weight. Defaults to `0.01`
(MAPPO default).
- **critic_coef** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)*,**optional*) - critic loss weight. Defaults to `1.0`.
- **normalize_advantage** (*bool*) - whether to standardise the advantage.
Defaults to `True` (MAPPO default; differs from base
`ClipPPOLoss` which defaults to `False`).
- **normalize_advantage_exclude_dims** (*tuple**of**int*) - dimensions to
exclude from advantage standardisation. Defaults to `(-2,)`
(the agent dim).
- ****kwargs** - forwarded to `ClipPPOLoss`.

The expected tensordict layout follows the torchrl multi-agent convention
(see `VmasEnv`,
`PettingZooEnv`):

- `("agents", "observation")`: `[*B, T, n_agents, obs_dim]`
- `("agents", "action")`: `[*B, T, n_agents, action_dim]`
- Optional `"state"` at the root for centralised critics
- Team-shared `("next", "reward")`, `("next", "done")`,
`("next", "terminated")` of shape `[*B, T, 1]` (or per-agent under
`("next", "agents", "reward")` for competitive settings).

Example

```
>>> import torch
>>> from tensordict.nn import TensorDictModule
>>> from torchrl.modules import (
... MultiAgentMLP, PopArtValueNorm, ProbabilisticActor,
... )
>>> from torchrl.modules.distributions import NormalParamExtractor, TanhNormal
>>> from torchrl.objectives.multiagent import MAPPOLoss
>>> n_agents, obs_dim, action_dim = 3, 6, 2
>>> actor_net = torch.nn.Sequential(
... MultiAgentMLP(
... n_agent_inputs=obs_dim, n_agent_outputs=2 * action_dim,
... n_agents=n_agents, centralized=False, share_params=True,
... ),
... NormalParamExtractor(),
... )
>>> actor_module = TensorDictModule(
... actor_net,
... in_keys=[("agents", "observation")],
... out_keys=[("agents", "loc"), ("agents", "scale")],
... )
>>> actor = ProbabilisticActor(
... module=actor_module,
... in_keys=[("agents", "loc"), ("agents", "scale")],
... out_keys=[("agents", "action")],
... distribution_class=TanhNormal,
... )
>>> critic = TensorDictModule(
... MultiAgentMLP(
... n_agent_inputs=obs_dim, n_agent_outputs=1,
... n_agents=n_agents, centralized=True, share_params=True,
... ),
... in_keys=[("agents", "observation")],
... out_keys=[("agents", "state_value")],
... )
>>> loss = MAPPOLoss(actor, critic, value_norm=PopArtValueNorm(shape=1))
>>> loss.set_keys(value=("agents", "state_value"), action=("agents", "action"))
```

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