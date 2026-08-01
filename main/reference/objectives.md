# torchrl.objectives package

TorchRL provides a comprehensive collection of loss modules for reinforcement learning algorithms.
These losses are designed to be stateful, reusable, and follow the tensordict convention.

## Key Features

- **Stateful objects**: Contain trainable parameters accessible via `loss_module.parameters()`
- **TensorDict convention**: Input and output use TensorDict format
- **Structured output**: Loss values returned with `"loss_<name>"` keys
- **Value estimators**: Support for TD(0), TD(λ), GAE, and more
- **Vmap support**: Efficient batched operations with customizable randomness modes

## Quick Example

```
from torchrl.objectives import DDPGLoss
from torchrl.modules import Actor, ValueOperator

# Create loss module
loss = DDPGLoss(
 actor_network=actor,
 value_network=value,
 gamma=0.99,
)

# Compute loss
td = collector.rollout()
loss_vals = loss(td)

# Get total loss
total_loss = sum(v for k, v in loss_vals.items() if k.startswith("loss_"))
```

## Documentation Sections

- [Common Components](objectives_common.html)

- [LossModule](generated/torchrl.objectives.LossModule.html)
- [add_random_module](generated/torchrl.objectives.add_random_module.html)
- [Value Estimators](objectives_common.html#value-estimators)
- [Value-Based Methods](objectives_value.html)

- [DQNLoss](generated/torchrl.objectives.DQNLoss.html)
- [DistributionalDQNLoss](generated/torchrl.objectives.DistributionalDQNLoss.html)
- [IQLLoss](generated/torchrl.objectives.IQLLoss.html)
- [DiscreteIQLLoss](generated/torchrl.objectives.DiscreteIQLLoss.html)
- [CQLLoss](generated/torchrl.objectives.CQLLoss.html)
- [DiscreteCQLLoss](generated/torchrl.objectives.DiscreteCQLLoss.html)
- [Policy Gradient Methods](objectives_policy.html)

- [PPOLoss](generated/torchrl.objectives.PPOLoss.html)
- [ClipPPOLoss](generated/torchrl.objectives.ClipPPOLoss.html)
- [KLPENPPOLoss](generated/torchrl.objectives.KLPENPPOLoss.html)
- [A2CLoss](generated/torchrl.objectives.A2CLoss.html)
- [ReinforceLoss](generated/torchrl.objectives.ReinforceLoss.html)
- [Actor-Critic Methods](objectives_actorcritic.html)

- [DDPGLoss](generated/torchrl.objectives.DDPGLoss.html)
- [SACLoss](generated/torchrl.objectives.SACLoss.html)
- [DiscreteSACLoss](generated/torchrl.objectives.DiscreteSACLoss.html)
- [TD3Loss](generated/torchrl.objectives.TD3Loss.html)
- [REDQLoss](generated/torchrl.objectives.REDQLoss.html)
- [CrossQLoss](generated/torchrl.objectives.CrossQLoss.html)
- [Offline RL Methods](objectives_offline.html)

- [CQLLoss](generated/torchrl.objectives.CQLLoss.html)
- [DiscreteCQLLoss](generated/torchrl.objectives.DiscreteCQLLoss.html)
- [IQLLoss](generated/torchrl.objectives.IQLLoss.html)
- [DiscreteIQLLoss](generated/torchrl.objectives.DiscreteIQLLoss.html)
- [TD3BCLoss](generated/torchrl.objectives.TD3BCLoss.html)
- [Multi-Agent Objectives](objectives_multiagent.html)

- [MAPPO and IPPO](objectives_multiagent.html#mappo-and-ippo)
- [QMixer](objectives_multiagent.html#qmixer)
- [Other Loss Modules](objectives_other.html)

- [ACTLoss](generated/torchrl.objectives.ACTLoss.html)
- [BCLoss](generated/torchrl.objectives.BCLoss.html)
- [DiffusionBCLoss](generated/torchrl.objectives.DiffusionBCLoss.html)
- [GAILLoss](generated/torchrl.objectives.GAILLoss.html)
- [DTLoss](generated/torchrl.objectives.DTLoss.html)
- [OnlineDTLoss](generated/torchrl.objectives.OnlineDTLoss.html)
- [DreamerActorLoss](generated/torchrl.objectives.DreamerActorLoss.html)
- [DreamerModelLoss](generated/torchrl.objectives.DreamerModelLoss.html)
- [DreamerValueLoss](generated/torchrl.objectives.DreamerValueLoss.html)
- [WorldModelLoss](generated/torchrl.objectives.WorldModelLoss.html)
- [ExponentialQuadraticCost](generated/torchrl.objectives.ExponentialQuadraticCost.html)
- [RNDLoss](generated/torchrl.objectives.RNDLoss.html)
- [DreamerV3](objectives_other.html#dreamerv3)