# IPPOLoss

*class*torchrl.objectives.multiagent.IPPOLoss(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/multiagent/mappo.html#IPPOLoss)

Independent PPO loss (de Witt et al. 2020).

IPPO is the decentralised counterpart of MAPPO: each agent has its *own*
value function that conditions only on its local observation. There is no
centralised critic and no global state required. Surprisingly competitive
with MAPPO on many SMAC scenarios (the de Witt et al. paper is titled
*Is Independent Learning All You Need...*).

Structurally this loss is identical to [`MAPPOLoss`](torchrl.objectives.multiagent.MAPPOLoss.html#torchrl.objectives.multiagent.MAPPOLoss); the difference
lives entirely in the critic the user passes in. We expose it as a
separate class so the API is self-documenting: when you import
`IPPOLoss` it is unambiguous which algorithm you are running, and the
docstring spells out the critic-construction recipe.

Parameters:

- **actor_network** (*ProbabilisticTensorDictSequential*) - per-agent policy.
Build with `MultiAgentMLP(centralized=False, share_params=True)`.
- **critic_network** (*TensorDictModule*) - per-agent value operator. Build
with `MultiAgentMLP(centralized=False, share_params=True)` so
each agent values its own observation.

Keyword Arguments:

- **value_norm** ([*ValueNorm*](torchrl.modules.ValueNorm.html#torchrl.modules.ValueNorm)*,**optional*) - rarely used with IPPO; defaults to
`None`.
- **entropy_coeff** ([*float*](torchrl.data.llm.TopKRewardSelector.html#torchrl.data.llm.TopKRewardSelector.float)) - defaults to `0.01`.
- **normalize_advantage** (*bool*) - defaults to `True`.
- **normalize_advantage_exclude_dims** (*tuple**of**int*) - defaults to `(-2,)`.
- ****kwargs** - forwarded to `ClipPPOLoss`.

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