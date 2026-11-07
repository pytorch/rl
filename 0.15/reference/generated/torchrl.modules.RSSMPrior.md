# RSSMPrior

*class*torchrl.modules.RSSMPrior(*action_spec*, *hidden_dim=200*, *rnn_hidden_dim=200*, *state_dim=30*, *scale_lb=0.1*, *action_dim=None*, *device=None*)[[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMPrior)

The prior network of the RSSM.

This network takes as input the previous state and belief and the current action.
It returns the next prior state and belief, as well as the parameters of the prior state distribution.
State is by construction stochastic and belief is deterministic. In "Dream to control", these are called "deterministic state " and "stochastic state", respectively.

Reference: [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)

Parameters:

- **action_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - Action spec.
- **hidden_dim** (*int**,**optional*) - Number of hidden units in the linear network. Input size of the recurrent network.
Defaults to 200.
- **rnn_hidden_dim** (*int**,**optional*) - Number of hidden units in the recurrent network. Also size of the belief.
Defaults to 200.
- **state_dim** (*int**,**optional*) - Size of the state.
Defaults to 30.
- **scale_lb** (`float`, optional) - Lower bound of the scale of the state distribution.
Defaults to 0.1.
- **action_dim** (*int**,**optional*) - Dimension of the action. If provided along with state_dim,
uses explicit Linear instead of LazyLinear. Defaults to None for backward compatibility.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device to create the module on.
Defaults to None (uses default device).

forward(*state*, *belief*, *action*, *noise=None*)[[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMPrior.forward)

Forward pass through the prior network.

Parameters:

- **state** - Previous stochastic state.
- **belief** - Previous deterministic belief.
- **action** - Action to condition on.
- **noise** - Optional pre-sampled noise for the prior state.
If None, samples from standard normal. Used for deterministic testing.

Returns:

Tuple of (prior_mean, prior_std, state, belief).