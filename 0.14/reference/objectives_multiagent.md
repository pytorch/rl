# Multi-Agent Objectives

Loss modules for multi-agent reinforcement learning algorithms. These losses
follow the torchrl multi-agent tensordict convention (per-agent tensors
nested under group keys such as `("agents", "observation")`; see
[`VmasEnv`](generated/torchrl.envs.VmasEnv.html#torchrl.envs.VmasEnv) and
[`PettingZooEnv`](generated/torchrl.envs.PettingZooEnv.html#torchrl.envs.PettingZooEnv)).

## MAPPO and IPPO

[`MAPPOLoss`](generated/torchrl.objectives.multiagent.MAPPOLoss.html#torchrl.objectives.multiagent.MAPPOLoss) implements Multi-Agent PPO (Yu et al. 2022) -- a
decentralised actor paired with a *centralised critic* that conditions on the
joint observation / state. [`IPPOLoss`](generated/torchrl.objectives.multiagent.IPPOLoss.html#torchrl.objectives.multiagent.IPPOLoss) is the independent-learner
counterpart from de Witt et al. 2020: each agent has its own local critic and
there is no centralised information at training time.

Both are thin specialisations of [`ClipPPOLoss`](generated/torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss)
that:

- default the value estimator to
[`MultiAgentGAE`](generated/torchrl.objectives.value.MultiAgentGAE.html#torchrl.objectives.value.MultiAgentGAE), which broadcasts
team-shared rewards / done flags across the agent dimension before
computing returns;
- default `normalize_advantage_exclude_dims` to `(-2,)` so the agent dim
is excluded from advantage standardisation;
- optionally accept a [`ValueNorm`](generated/torchrl.modules.ValueNorm.html#torchrl.modules.ValueNorm) subclass -- either
[`PopArtValueNorm`](generated/torchrl.modules.PopArtValueNorm.html#torchrl.modules.PopArtValueNorm) (EMA, recommended for drifting
reward scales) or [`RunningValueNorm`](generated/torchrl.modules.RunningValueNorm.html#torchrl.modules.RunningValueNorm) (exact
Welford running stats, recommended for stationary scales) -- to stabilise
the critic loss. The MAPPO paper credits this trick for its strong SMAC
results.

See `sota-implementations/multiagent/mappo_ippo.py` for a hydra-configured
recipe and `examples/multiagent/mappo_vmas.py` for a minimal one.

| [`MAPPOLoss`](generated/torchrl.objectives.multiagent.MAPPOLoss.html#torchrl.objectives.multiagent.MAPPOLoss)(*args, **kwargs) | Multi-Agent PPO loss with a centralised critic (Yu et al. 2022). |
| --- | --- |
| [`IPPOLoss`](generated/torchrl.objectives.multiagent.IPPOLoss.html#torchrl.objectives.multiagent.IPPOLoss)(*args, **kwargs) | Independent PPO loss (de Witt et al. 2020). |

## QMixer

[`QMixerLoss`](generated/torchrl.objectives.multiagent.QMixerLoss.html#torchrl.objectives.multiagent.QMixerLoss) mixes local per-agent Q values into a global team Q
value via a learnable mixing network, and trains them jointly with a DQN
update on the global value (Rashid et al. 2018).

| [`QMixerLoss`](generated/torchrl.objectives.multiagent.QMixerLoss.html#torchrl.objectives.multiagent.QMixerLoss)(*args, **kwargs) | The QMixer loss class. |
| --- | --- |