# RSSMPosterior

*class*torchrl.modules.RSSMPosterior(*hidden_dim=200*, *state_dim=30*, *scale_lb=0.1*, *rnn_hidden_dim=None*, *obs_embed_dim=None*, *device=None*)[[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMPosterior)

The posterior network of the RSSM.

This network takes as input the belief and the associated encoded observation.
It returns the parameters of the posterior as well as a state sampled according to this distribution.

Reference: [https://arxiv.org/abs/1811.04551](https://arxiv.org/abs/1811.04551)

Parameters:

- **hidden_dim** (*int**,**optional*) - Number of hidden units in the linear network.
Defaults to 200.
- **state_dim** (*int**,**optional*) - Size of the state.
Defaults to 30.
- **scale_lb** (`float`, optional) - Lower bound of the scale of the state distribution.
Defaults to 0.1.
- **rnn_hidden_dim** (*int**,**optional*) - Dimension of the belief/rnn hidden state.
If provided along with obs_embed_dim, uses explicit Linear. Defaults to None.
- **obs_embed_dim** (*int**,**optional*) - Dimension of the observation embedding.
If provided along with rnn_hidden_dim, uses explicit Linear. Defaults to None.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device to create the module on.
Defaults to None (uses default device).

forward(*belief*, *obs_embedding*, *noise=None*)[[source]](../../_modules/torchrl/modules/models/model_based.html#RSSMPosterior.forward)

Forward pass through the posterior network.

Parameters:

- **belief** - Deterministic belief from the prior.
- **obs_embedding** - Encoded observation.
- **noise** - Optional pre-sampled noise for the posterior state.
If None, samples from standard normal. Used for deterministic testing.

Returns:

Tuple of (posterior_mean, posterior_std, state).