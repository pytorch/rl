# WeightStrategy

*class*torchrl.weight_update.WeightStrategy(*extract_as: Literal['tensordict', 'state_dict'] = 'tensordict'*)[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#WeightStrategy)

Unified strategy for weight transmission.

This strategy handles both extraction and application of weights, supporting
both TensorDict and state_dict formats.

Parameters:

**extract_as** (*str*) - Format for extracting weights. Can be:
- "tensordict" (default): Extract weights as TensorDict
- "state_dict": Extract weights as PyTorch state_dict

The application format is automatically detected based on the type of weights
received (dict -> state_dict, TensorDict -> tensordict).

apply_weights(*destination: Any*, *weights: Any*, *inplace: bool = True*) → None[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#WeightStrategy.apply_weights)

Apply weights to destination model.

The format is automatically detected from the weights type:
- dict -> state_dict format
- TensorDictBase -> tensordict format

Parameters:

- **destination** - The model to apply weights to. Can be:
- nn.Module: PyTorch module
- TensorDictBase: TensorDict
- dict: State dictionary
- **weights** - The weights to apply (dict or TensorDictBase).
- **inplace** - Whether to apply weights in place.

extract_weights(*source: Any*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | dict | None[[source]](../../_modules/torchrl/weight_update/weight_sync_schemes.html#WeightStrategy.extract_weights)

Extract weights from source model in the specified format.

Parameters:

**source** - The model to extract weights from. Can be:
- nn.Module: PyTorch module
- TensorDictBase: TensorDict
- dict: State dictionary

Returns:

Weights in the format specified by extract_as constructor argument.