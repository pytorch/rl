# MultiAgentGAE

*class*torchrl.objectives.value.MultiAgentGAE(**args*, ***kwargs*)[[source]](../../_modules/torchrl/objectives/value/advantages.html#MultiAgentGAE)

Multi-agent Generalized Advantage Estimator.

Drop-in replacement for [`GAE`](torchrl.objectives.value.GAE.html#torchrl.objectives.value.GAE) when the value network produces per-agent
state values (shape `[*B, T, n_agents, 1]`) but the reward / done /
terminated signals are shared across agents at the team level
(shape `[*B, T, 1]`) -- the standard cooperative-MARL layout in torchrl
(see e.g. `torchrl/envs/libs/vmas.py` and
`torchrl/envs/libs/pettingzoo.py`).

The estimator detects whether the reward/done/terminated tensors are missing
the agent dimension relative to the value tensor, and broadcasts them along
that dimension before running the standard vectorised GAE recursion. If the
reward is already per-agent (e.g. a competitive setting), it is passed
through unchanged.

The output `"advantage"` and `"value_target"` entries match the shape
of the value tensor (`[*B, T, n_agents, 1]`), which is what
[`MAPPOLoss`](torchrl.objectives.multiagent.MAPPOLoss.html#torchrl.objectives.multiagent.MAPPOLoss) expects.

Keyword Arguments:

**agent_dim** (*int**,**optional*) - the dimension that holds the agent index in
the value tensor. Negative dimensions are taken modulo
`value.ndim`. Defaults to `-2` (penultimate), matching the
convention used by `MultiAgentMLP`.

Other args/kwargs are forwarded to [`GAE`](torchrl.objectives.value.GAE.html#torchrl.objectives.value.GAE).