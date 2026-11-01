# Robot Learning

Policy architectures for robot manipulation and imitation learning.

# World Models and Model-Based RL

Modules for model-based reinforcement learning, including world models and dynamics models.

| [`WorldModelWrapper`](generated/torchrl.modules.WorldModelWrapper.html#torchrl.modules.WorldModelWrapper)(*args, **kwargs) | World model wrapper. |
| --- | --- |
| [`DreamerActor`](generated/torchrl.modules.DreamerActor.html#torchrl.modules.DreamerActor)(out_features[, depth, ...]) | Dreamer actor network. |
| [`ObsEncoder`](generated/torchrl.modules.ObsEncoder.html#torchrl.modules.ObsEncoder)([channels, num_layers, ...]) | Observation encoder network. |
| [`ObsDecoder`](generated/torchrl.modules.ObsDecoder.html#torchrl.modules.ObsDecoder)([channels, num_layers, ...]) | Observation decoder network. |
| [`RSSMPosterior`](generated/torchrl.modules.RSSMPosterior.html#torchrl.modules.RSSMPosterior)([hidden_dim, state_dim, ...]) | The posterior network of the RSSM. |
| [`RSSMPrior`](generated/torchrl.modules.RSSMPrior.html#torchrl.modules.RSSMPrior)(action_spec[, hidden_dim, ...]) | The prior network of the RSSM. |
| [`RSSMRollout`](generated/torchrl.modules.RSSMRollout.html#torchrl.modules.RSSMRollout)(*args, **kwargs) | Rollout the RSSM network. |

## PILCO

Components for moment-matching model-based policy search (PILCO).

| [`GPWorldModel`](generated/torchrl.modules.GPWorldModel.html#torchrl.modules.GPWorldModel)(obs_dim, action_dim[, in_keys, ...]) | Gaussian Process world model with moment-matching uncertainty propagation. |
| --- | --- |
| [`RBFController`](generated/torchrl.modules.RBFController.html#torchrl.modules.RBFController)(input_dim, output_dim, max_action) | Radial Basis Function controller for moment-matching policy search. |