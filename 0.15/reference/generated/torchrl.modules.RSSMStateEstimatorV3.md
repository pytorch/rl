# RSSMStateEstimatorV3

*class*torchrl.modules.RSSMStateEstimatorV3(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMStateEstimatorV3)

Update the DreamerV3 acting state from an encoded observation.

The estimator shares the trained prior and posterior modules. It advances
the recurrent belief and samples the observation-conditioned posterior,
without evaluating or sampling the unused prior distribution. Reset entries
discard the preceding state, belief and action independently in each stream.
Compose this module between an observation encoder and a probabilistic actor
with [`TensorDictSequential`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictSequential.html#tensordict.nn.TensorDictSequential).

Reference: Hafner et al., "Mastering Diverse Domains through World Models"
(2023), [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104).

Parameters:

- **prior** ([*RSSMPriorV3*](torchrl.modules.RSSMPriorV3.html#torchrl.modules.RSSMPriorV3)) - Trained recurrent prior, shared with the world model.
- **posterior** ([*RSSMPosteriorV3*](torchrl.modules.RSSMPosteriorV3.html#torchrl.modules.RSSMPosteriorV3)) - Trained observation-conditioned posterior.

Keyword Arguments:

- **in_keys** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**NestedKey**,**optional*) - Five keys, in order: previous
stochastic state, previous belief, previous action, encoded current
observation and reset flag. Defaults to `["state", "belief",
"previous_action", "encoded_latents", "is_init"]`. Features have
one trailing dimension; reset flags may omit their singleton feature
dimension. Keys are fixed at construction.
- **out_keys** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**NestedKey**,**optional*) - Two keys receiving the posterior
state and updated belief, in that order. Defaults to `["state",
"belief"]`, replacing the input entries. Outputs are float32 for
recurrent collection, including under autocast.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.modules import RSSMPriorV3, RSSMPosteriorV3, RSSMStateEstimatorV3
>>> prior = RSSMPriorV3(
... action_shape=(2,), action_dim=2, hidden_dim=8,
... rnn_hidden_dim=8, num_categoricals=2, num_classes=4,
... )
>>> posterior = RSSMPosteriorV3(
... hidden_dim=8, rnn_hidden_dim=8, obs_embed_dim=6,
... num_categoricals=2, num_classes=4,
... )
>>> estimator = RSSMStateEstimatorV3(prior, posterior)
>>> data = TensorDict({
... "state": torch.randn(2, 8), "belief": torch.randn(2, 8),
... "previous_action": torch.randn(2, 2),
... "encoded_latents": torch.randn(2, 6),
... "is_init": torch.tensor([True, False]),
... }, [2])
>>> with torch.no_grad():
... result = estimator(data)
>>> assert result["state"].shape == (2, 8)
>>> assert torch.allclose(result["state"].reshape(2, 2, 4).sum(-1), torch.ones(2, 2))
```

See also `RSSMStateEstimatorV3Config`.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMStateEstimatorV3.forward)

Write the current posterior state and belief into the input TensorDict.