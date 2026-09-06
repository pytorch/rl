# DreamerV3BlockGRUCell

*class*torchrl.modules.DreamerV3BlockGRUCell(*input_size: int*, *hidden_size: int*, ***, *projection_size: int = 512*, *num_blocks: int = 8*, *num_layers: int = 1*, *activation_class: type[~torch.nn.modules.module.Module] | ~collections.abc.Callable = <class 'torch.nn.modules.activation.SiLU'>*, *norm_eps: float = 0.0001*, *update_bias: float = -1.0*, *device: ~torch.device | str | int | None = None*)[[source]](../../_modules/torchrl/modules/models/model_based.html#DreamerV3BlockGRUCell)

Single-step DreamerV3 block-diagonal GRU cell.

Parameters:

- **input_size** (*int*) - Input feature count.
- **hidden_size** (*int*) - Recurrent hidden-state width.
- **projection_size** (*int**,**optional*) - Input and hidden projection width.
Defaults to 512.
- **num_blocks** (*int**,**optional*) - Number of independent recurrent blocks.
Defaults to 8.
- **num_layers** (*int**,**optional*) - Number of block-linear dynamics layers.
Defaults to 1.
- **activation_class** (*type**[**nn.Module**] or**callable**,**optional*) - Parameter-free,
elementwise, shape-preserving activation. Defaults to `nn.SiLU`.
- **norm_eps** (*float**,**optional*) - RMS normalization epsilon. Defaults to `1e-4`.
- **update_bias** (*float**,**optional*) - Fixed update-gate logit offset. Defaults
to `-1.0`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Parameter device. Defaults to None.

Examples

```
>>> import torch
>>> from torchrl.modules import DreamerV3BlockGRUCell
>>> cell = DreamerV3BlockGRUCell(6, 8, projection_size=4, num_blocks=2)
>>> cell(torch.randn(3, 6), torch.zeros(3, 8)).shape
torch.Size([3, 8])
```

forward(*input: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *hidden: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/model_based.html#DreamerV3BlockGRUCell.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.