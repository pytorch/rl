# DreamerV3BlockGRU

*class*torchrl.modules.DreamerV3BlockGRU(*input_size: int*, *hidden_size: int*, ***, *projection_size: int = 512*, *num_blocks: int = 8*, *num_layers: int = 1*, *activation_class: type[~torch.nn.modules.module.Module] | ~collections.abc.Callable = <class 'torch.nn.modules.activation.SiLU'>*, *norm_eps: float = 0.0001*, *update_bias: float = -1.0*, *recurrent_backend: ~typing.Literal['reference'*, *'scan'*, *'triton'] = 'reference'*, *device: ~torch.device | str | int | None = None*)[[source]](../../_modules/torchrl/modules/models/model_based.html#DreamerV3BlockGRU)

Batch-major DreamerV3 block-diagonal GRU sequence module.

`is_init` marks entries whose carry is zeroed before that timestep.
The `"reference"` backend uses ordinary autograd and supports every
TorchRL-compatible PyTorch version. The opt-in `"scan"` backend uses a
specialized compiled reverse scan. The explicit `"triton"` backend
fuses each complete CUDA recurrence into one forward and one reverse-time
kernel on NVIDIA GPUs (Triton 3.3 or newer); it keeps parameters and
accumulation in `float32` and does not fall back to another backend.
Only the reference backend supports double backward: the optimized
backends raise on `create_graph=True` instead of returning wrong
second-order gradients. See [DreamerV3 in a nutshell](../dreamer_v3.html) for a full
backend comparison.

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
- **recurrent_backend** (*"reference"**,**"scan"**, or**"triton"**,**optional*) - Sequence backend. Defaults to `"reference"`. The `"triton"`
backend only supports [`SiLU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.SiLU.html#torch.nn.SiLU),
[`Tanh`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh), and [`ReLU`](https://docs.pytorch.org/docs/stable/generated/torch.nn.ReLU.html#torch.nn.ReLU) activations
and `torch.float32` / `torch.bfloat16` inputs (mixed input and
hidden dtypes are promoted like the reference backend).
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - Parameter device. Defaults to None.

Examples

```
>>> import torch
>>> from torchrl.modules import DreamerV3BlockGRU
>>> gru = DreamerV3BlockGRU(6, 8, projection_size=4, num_blocks=2)
>>> output, hidden = gru(torch.randn(3, 5, 6))
>>> output.shape, hidden.shape
(torch.Size([3, 5, 8]), torch.Size([3, 8]))
```

forward(*input: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *hidden: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*, *is_init: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None = None*) → tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)][[source]](../../_modules/torchrl/modules/models/model_based.html#DreamerV3BlockGRU.forward)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.