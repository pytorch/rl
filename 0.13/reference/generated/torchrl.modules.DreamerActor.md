# DreamerActor

*class*torchrl.modules.DreamerActor(*out_features*, *depth=4*, *num_cells=200*, *activation_class=<class 'torch.nn.modules.activation.ELU'>*, *std_bias=5.0*, *std_min_val=0.0001*, *device=None*)[[source]](../../_modules/torchrl/modules/models/model_based.html#DreamerActor)

Dreamer actor network.

This network is used to predict the action distribution given the
the stochastic state and the deterministic belief at the current
time step.
It outputs the mean and the scale of the action distribution.

Reference: [https://arxiv.org/abs/1912.01603](https://arxiv.org/abs/1912.01603)

Parameters:

- **out_features** (*int*) - Number of output features.
- **depth** (*int**,**optional*) - Number of hidden layers.
Defaults to 4.
- **num_cells** (*int**,**optional*) - Number of hidden units per layer.
Defaults to 200.
- **activation_class** (*nn.Module**,**optional*) - Activation class.
Defaults to nn.ELU.
- **std_bias** (`float`, optional) - Bias of the softplus transform.
Defaults to 5.0.
- **std_min_val** (`float`, optional) - Minimum value of the standard deviation.
Defaults to 1e-4.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Device to create the module on.
Defaults to None (uses default device).

forward(*state*, *belief*)[[source]](../../_modules/torchrl/modules/models/model_based.html#DreamerActor.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.