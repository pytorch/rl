# ConvNet

*class*torchrl.modules.ConvNet(*in_features: int | None = None*, *depth: int | None = None*, *num_cells: ~collections.abc.Sequence[int] | int = None*, *kernel_sizes: ~collections.abc.Sequence[int] | int = 3*, *strides: ~collections.abc.Sequence[int] | int = 1*, *paddings: ~collections.abc.Sequence[int] | int = 0*, *activation_class: type[~torch.nn.modules.module.Module] | ~collections.abc.Callable = <class 'torch.nn.modules.activation.ELU'>*, *activation_kwargs: dict | list[dict] | None = None*, *norm_class: type[~torch.nn.modules.module.Module] | ~collections.abc.Callable | None = None*, *norm_kwargs: dict | list[dict] | None = None*, *bias_last_layer: bool = True*, *aggregator_class: type[~torch.nn.modules.module.Module] | ~collections.abc.Callable | None = <class 'torchrl.modules.models.utils.SquashDims'>*, *aggregator_kwargs: dict | None = None*, *squeeze_output: bool = False*, *device: ~torch.device | str | int | None = None*)[[source]](../../_modules/torchrl/modules/models/models.html#ConvNet)

A convolutional neural network.

Parameters:

- **in_features** (*int**,**optional*) - number of input features. If `None`, a
[`LazyConv2d`](https://docs.pytorch.org/docs/stable/generated/torch.nn.LazyConv2d.html#torch.nn.LazyConv2d) module is used for the first layer.;
- **depth** (*int**,**optional*) - depth of the network. A depth of 1 will produce
a single linear layer network with the desired input size, and
with an output size equal to the last element of the num_cells
argument.
If no depth is indicated, the depth information should be contained
in the `num_cells` argument (see below).
If `num_cells` is an iterable and `depth` is indicated, both
should match: `len(num_cells)` must be equal to the `depth`.
- **num_cells** (*int**or**Sequence**of**int**,**optional*) - number of cells of
every layer in between the input and output. If an integer is
provided, every layer will have the same number of cells. If an
iterable is provided, the linear layers `out_features` will match
the content of num_cells. Defaults to `[32, 32, 32]`.
- **kernel_sizes** (*int**,**sequence**of**int**,**optional*) - Kernel size(s) of the
conv network. If iterable, the length must match the depth,
defined by the `num_cells` or depth arguments.
Defaults to `3`.
- **strides** (*int**or**sequence**of**int**,**optional*) - Stride(s) of the conv network. If
iterable, the length must match the depth, defined by the
`num_cells` or depth arguments. Defaults to `1`.
- **activation_class** (*Type**[**nn.Module**] or**callable**,**optional*) - activation
class or constructor to be used.
Defaults to [`Tanh`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh).
- **activation_kwargs** (*dict**or**list**of**dicts**,**optional*) - kwargs to be used
with the activation class. A list of kwargs of length `depth`
can also be passed, with one element per layer.
- **norm_class** (*Type**or**callable**,**optional*) - normalization class or
constructor, if any.
- **norm_kwargs** (*dict**or**list**of**dicts**,**optional*) - kwargs to be used with
the normalization layers. A list of kwargs of length `depth` can
also be passed, with one element per layer.
- **bias_last_layer** (*bool*) - if `True`, the last Linear layer will have a
bias parameter. Defaults to `True`.
- **aggregator_class** (*Type**[**nn.Module**] or**callable*) - aggregator class or
constructor to use at the end of the chain.
Defaults to `torchrl.modules.utils.models.SquashDims`;
- **aggregator_kwargs** (*dict**,**optional*) - kwargs for the
`aggregator_class`.
- **squeeze_output** (*bool*) - whether the output should be squeezed of its
singleton dimensions.
Defaults to `False`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device to create the module on.

Examples

```
>>> # All of the following examples provide valid, working MLPs
>>> cnet = ConvNet(in_features=3, depth=1, num_cells=[32,]) # MLP consisting of a single 3 x 6 linear layer
>>> print(cnet)
ConvNet(
 (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
 (1): ELU(alpha=1.0)
 (2): SquashDims()
)
>>> cnet = ConvNet(in_features=3, depth=4, num_cells=32)
>>> print(cnet)
ConvNet(
 (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
 (1): ELU(alpha=1.0)
 (2): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
 (3): ELU(alpha=1.0)
 (4): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
 (5): ELU(alpha=1.0)
 (6): Conv2d(32, 32, kernel_size=(3, 3), stride=(1, 1))
 (7): ELU(alpha=1.0)
 (8): SquashDims()
)
>>> cnet = ConvNet(in_features=3, num_cells=[32, 33, 34, 35]) # defines the depth by the num_cells arg
>>> print(cnet)
ConvNet(
 (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
 (1): ELU(alpha=1.0)
 (2): Conv2d(32, 33, kernel_size=(3, 3), stride=(1, 1))
 (3): ELU(alpha=1.0)
 (4): Conv2d(33, 34, kernel_size=(3, 3), stride=(1, 1))
 (5): ELU(alpha=1.0)
 (6): Conv2d(34, 35, kernel_size=(3, 3), stride=(1, 1))
 (7): ELU(alpha=1.0)
 (8): SquashDims()
)
>>> cnet = ConvNet(in_features=3, num_cells=[32, 33, 34, 35], kernel_sizes=[3, 4, 5, (2, 3)]) # defines kernels, possibly rectangular
>>> print(cnet)
ConvNet(
 (0): Conv2d(3, 32, kernel_size=(3, 3), stride=(1, 1))
 (1): ELU(alpha=1.0)
 (2): Conv2d(32, 33, kernel_size=(4, 4), stride=(1, 1))
 (3): ELU(alpha=1.0)
 (4): Conv2d(33, 34, kernel_size=(5, 5), stride=(1, 1))
 (5): ELU(alpha=1.0)
 (6): Conv2d(34, 35, kernel_size=(2, 3), stride=(1, 1))
 (7): ELU(alpha=1.0)
 (8): SquashDims()
)
```

*classmethod*default_atari_dqn(*num_actions: int*)[[source]](../../_modules/torchrl/modules/models/models.html#ConvNet.default_atari_dqn)

Returns the default DQN as presented in the seminal DQN paper.

Parameters:

**num_actions** (*int*) - the action space of the atari game.

forward(*inputs: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/models.html#ConvNet.forward)

Runs the forward pass.