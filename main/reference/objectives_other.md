# Other Loss Modules

Additional loss modules for specialized algorithms.

| [`ACTLoss`](generated/torchrl.objectives.ACTLoss.html#torchrl.objectives.ACTLoss)(*args, **kwargs) | Loss module for Action Chunking with Transformers (ACT). |
| --- | --- |
| [`BCLoss`](generated/torchrl.objectives.BCLoss.html#torchrl.objectives.BCLoss)(*args, **kwargs) | Behavior Cloning Loss Module. |
| [`DiffusionBCLoss`](generated/torchrl.objectives.DiffusionBCLoss.html#torchrl.objectives.DiffusionBCLoss)(*args, **kwargs) | Behavioural Cloning loss for diffusion-based policies. |
| [`GAILLoss`](generated/torchrl.objectives.GAILLoss.html#torchrl.objectives.GAILLoss)(*args, **kwargs) | TorchRL implementation of the Generative Adversarial Imitation Learning (GAIL) loss. |
| [`DTLoss`](generated/torchrl.objectives.DTLoss.html#torchrl.objectives.DTLoss)(*args, **kwargs) | TorchRL implementation of the Online Decision Transformer loss. |
| [`OnlineDTLoss`](generated/torchrl.objectives.OnlineDTLoss.html#torchrl.objectives.OnlineDTLoss)(*args, **kwargs) | TorchRL implementation of the Online Decision Transformer loss. |
| [`DreamerActorLoss`](generated/torchrl.objectives.DreamerActorLoss.html#torchrl.objectives.DreamerActorLoss)(*args, **kwargs) | Dreamer Actor Loss. |
| [`DreamerModelLoss`](generated/torchrl.objectives.DreamerModelLoss.html#torchrl.objectives.DreamerModelLoss)(*args, **kwargs) | Dreamer Model Loss. |
| [`DreamerValueLoss`](generated/torchrl.objectives.DreamerValueLoss.html#torchrl.objectives.DreamerValueLoss)(*args, **kwargs) | Dreamer Value Loss. |
| [`WorldModelLoss`](generated/torchrl.objectives.WorldModelLoss.html#torchrl.objectives.WorldModelLoss)(*args, **kwargs) | A general loss module for model-based world models. |
| [`ExponentialQuadraticCost`](generated/torchrl.objectives.ExponentialQuadraticCost.html#torchrl.objectives.ExponentialQuadraticCost)(*args, **kwargs) | Computes the expected saturating cost for a Gaussian-distributed state. |
| [`RNDLoss`](generated/torchrl.objectives.RNDLoss.html#torchrl.objectives.RNDLoss)(*args, **kwargs) | Loss module for training the predictor network in Random Network Distillation. |

## DreamerV3

Loss modules for DreamerV3 ([Mastering Diverse Domains in World Models, Hafner et al. 2023](https://arxiv.org/abs/2301.04104)).
Key differences from V1: discrete categorical latent state, KL balancing, symlog transforms, and two-hot value distributions.

| [`DreamerV3ActorLoss`](generated/torchrl.objectives.DreamerV3ActorLoss.html#torchrl.objectives.DreamerV3ActorLoss)(*args, **kwargs) | DreamerV3 Actor Loss. |
| --- | --- |
| [`DreamerV3ModelLoss`](generated/torchrl.objectives.DreamerV3ModelLoss.html#torchrl.objectives.DreamerV3ModelLoss)(*args, **kwargs) | DreamerV3 World Model Loss. |
| [`DreamerV3ValueLoss`](generated/torchrl.objectives.DreamerV3ValueLoss.html#torchrl.objectives.DreamerV3ValueLoss)(*args, **kwargs) | DreamerV3 Value Loss. |

### DreamerV3 Utilities

| [`symlog`](generated/torchrl.objectives.symlog.html#torchrl.objectives.symlog)(x) | Symmetric logarithm: `sign(x) * log(\|x\| + 1)`. |
| --- | --- |
| [`symexp`](generated/torchrl.objectives.symexp.html#torchrl.objectives.symexp)(x) | Symmetric exponential: `sign(x) * (exp(\|x\|) - 1)`. |
| [`two_hot_encode`](generated/torchrl.objectives.two_hot_encode.html#torchrl.objectives.two_hot_encode)(x, bins) | Encode a scalar tensor as a two-hot distribution over `bins`. |
| [`two_hot_decode`](generated/torchrl.objectives.two_hot_decode.html#torchrl.objectives.two_hot_decode)(logits, bins) | Decode a distribution over `bins` to a scalar expectation. |