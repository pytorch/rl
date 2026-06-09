# torchrl.modules package

TorchRL offers a comprehensive collection of RL-specific neural network modules built on top of
[`tensordict.nn.TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule). These modules are designed to work seamlessly with
tensordict data structures, making it easy to build and compose RL models.

## Key Features

- **Spec-based construction**: Automatically configure output layers based on action specs
- **Probabilistic modules**: Built-in support for stochastic policies
- **Exploration strategies**: Modular exploration wrappers (ε-greedy, Ornstein-Uhlenbeck, etc.)
- **Value networks**: Q-value, distributional, and dueling architectures
- **Safe modules**: Automatic projection to satisfy action constraints
- **Model-based RL**: World model and dynamics modules

## Quick Example

```
from torchrl.modules import ProbabilisticActor, TanhNormal
from torchrl.envs import GymEnv
from tensordict.nn import TensorDictModule
import torch.nn as nn

env = GymEnv("Pendulum-v1")

# Create a probabilistic actor
actor = ProbabilisticActor(
 module=TensorDictModule(
 nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 2)),
 in_keys=["observation"],
 out_keys=["loc", "scale"],
 ),
 in_keys=["loc", "scale"],
 distribution_class=TanhNormal,
 spec=env.action_spec,
)
```

## Documentation Sections

- [Actor Modules](modules_actors.html)

- [TensorDictModules and SafeModules](modules_actors.html#tensordictmodules-and-safemodules)
- [Probabilistic actors](modules_actors.html#probabilistic-actors)
- [Q-Value actors](modules_actors.html#q-value-actors)
- [Exploration Strategies](modules_exploration.html)

- [AdditiveGaussianModule](generated/torchrl.modules.AdditiveGaussianModule.html)
- [ConsistentDropoutModule](generated/torchrl.modules.ConsistentDropoutModule.html)
- [EGreedyModule](generated/torchrl.modules.EGreedyModule.html)
- [OrnsteinUhlenbeckProcessModule](generated/torchrl.modules.OrnsteinUhlenbeckProcessModule.html)
- [Helpers](modules_exploration.html#helpers)
- [Value Networks and Critics](modules_critics.html)

- [ValueOperator](generated/torchrl.modules.ValueOperator.html)
- [ValueNorm](generated/torchrl.modules.ValueNorm.html)
- [PopArtValueNorm](generated/torchrl.modules.PopArtValueNorm.html)
- [RunningValueNorm](generated/torchrl.modules.RunningValueNorm.html)
- [DuelingCnnDQNet](generated/torchrl.modules.DuelingCnnDQNet.html)
- [DistributionalDQNnet](generated/torchrl.modules.DistributionalDQNnet.html)
- [ConvNet](generated/torchrl.modules.ConvNet.html)
- [CrossCriticGroupSpec](generated/torchrl.modules.CrossCriticGroupSpec.html)
- [CrossGroupCritic](generated/torchrl.modules.CrossGroupCritic.html)
- [MLP](generated/torchrl.modules.MLP.html)
- [DdpgCnnActor](generated/torchrl.modules.DdpgCnnActor.html)
- [DdpgCnnQNet](generated/torchrl.modules.DdpgCnnQNet.html)
- [DdpgMlpActor](generated/torchrl.modules.DdpgMlpActor.html)
- [DdpgMlpQNet](generated/torchrl.modules.DdpgMlpQNet.html)
- [LSTMModule](generated/torchrl.modules.LSTMModule.html)
- [GRUModule](generated/torchrl.modules.GRUModule.html)
- [canonicalize_rnn_subset](generated/torchrl.modules.canonicalize_rnn_subset.html)
- [set_recurrent_mode](generated/torchrl.modules.set_recurrent_mode.html)
- [OnlineDTActor](generated/torchrl.modules.OnlineDTActor.html)
- [DTActor](generated/torchrl.modules.DTActor.html)
- [DecisionTransformer](generated/torchrl.modules.DecisionTransformer.html)
- [Recurrent modules](modules_rnn.html)

- [Execution modes](modules_rnn.html#execution-modes)
- [Backend selection](modules_rnn.html#backend-selection)
- [Triton precision controls](modules_rnn.html#triton-precision-controls)
- [Choosing a layout and backend](modules_rnn.html#choosing-a-layout-and-backend)
- [See also](modules_rnn.html#see-also)
- [torchrl.modules.mcts package](modules_mcts.html)

- [MCTS Scores](modules_mcts.html#mcts-scores)
- [Robot Learning](modules_models.html)
- [World Models and Model-Based RL](modules_models.html#world-models-and-model-based-rl)

- [WorldModelWrapper](generated/torchrl.modules.WorldModelWrapper.html)
- [DreamerActor](generated/torchrl.modules.DreamerActor.html)
- [ObsEncoder](generated/torchrl.modules.ObsEncoder.html)
- [ObsDecoder](generated/torchrl.modules.ObsDecoder.html)
- [RSSMPosterior](generated/torchrl.modules.RSSMPosterior.html)
- [RSSMPrior](generated/torchrl.modules.RSSMPrior.html)
- [RSSMRollout](generated/torchrl.modules.RSSMRollout.html)
- [PILCO](modules_models.html#pilco)
- [Distribution Classes](modules_distributions.html)

- [Delta](generated/torchrl.modules.Delta.html)
- [IndependentNormal](generated/torchrl.modules.IndependentNormal.html)
- [MaskedCategorical](generated/torchrl.modules.MaskedCategorical.html)
- [NormalParamExtractor](generated/torchrl.modules.NormalParamExtractor.html)
- [OneHotCategorical](generated/torchrl.modules.OneHotCategorical.html)
- [ReparamGradientStrategy](generated/torchrl.modules.ReparamGradientStrategy.html)
- [TanhDelta](generated/torchrl.modules.TanhDelta.html)
- [TanhNormal](generated/torchrl.modules.TanhNormal.html)
- [TruncatedNormal](generated/torchrl.modules.TruncatedNormal.html)
- [Inference Server](modules_inference_server.html)

- [Core API](modules_inference_server.html#core-api)
- [Transport Backends](modules_inference_server.html#transport-backends)
- [Usage](modules_inference_server.html#usage)
- [Utilities and Helpers](modules_utils.html)

- [ActorValueOperator](generated/torchrl.modules.ActorValueOperator.html)
- [ActorCriticOperator](generated/torchrl.modules.ActorCriticOperator.html)
- [ActorCriticWrapper](generated/torchrl.modules.ActorCriticWrapper.html)
- [get_primers_from_module](generated/torchrl.modules.get_primers_from_module.html)
- [get_env_transforms_from_module](generated/torchrl.modules.get_env_transforms_from_module.html)
- [get_recurrent_matmul_precision](generated/torchrl.modules.get_recurrent_matmul_precision.html)
- [set_recurrent_matmul_precision](generated/torchrl.modules.set_recurrent_matmul_precision.html)
- [`RecurrentMatmulPrecision`](modules_utils.html#torchrl.modules.RecurrentMatmulPrecision)
- [`RecurrentMatmulPrecisionUserMode`](modules_utils.html#torchrl.modules.RecurrentMatmulPrecisionUserMode)
- [SquashDims](generated/torchrl.modules.models.utils.SquashDims.html)
- [Recurrent state lifecycle](recurrent_state_lifecycle.html)

- [Minimal recurrent PPO wiring](recurrent_state_lifecycle.html#minimal-recurrent-ppo-wiring)
- [The path at a glance](recurrent_state_lifecycle.html#the-path-at-a-glance)
- [What `is_init` means](recurrent_state_lifecycle.html#what-is-init-means)
- [When hidden state resets vs. is carried forward](recurrent_state_lifecycle.html#when-hidden-state-resets-vs-is-carried-forward)
- [Mid-batch done](recurrent_state_lifecycle.html#mid-batch-done)
- [Hidden outputs and recurrent backends](recurrent_state_lifecycle.html#hidden-outputs-and-recurrent-backends)
- [Common debugging symptoms](recurrent_state_lifecycle.html#common-debugging-symptoms)
- [What to check, in order](recurrent_state_lifecycle.html#what-to-check-in-order)
- [See also](recurrent_state_lifecycle.html#see-also)