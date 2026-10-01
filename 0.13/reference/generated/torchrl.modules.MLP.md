# MLP

*class*torchrl.modules.MLP(*in_features: int | None = None*, *out_features: int | ~torch.Size | None = None*, *depth: int | None = None*, *num_cells: ~collections.abc.Sequence[int] | int | None = None*, *activation_class: type[~torch.nn.modules.module.Module] | ~collections.abc.Callable = <class 'torch.nn.modules.activation.Tanh'>*, *activation_kwargs: dict | list[dict] | None = None*, *norm_class: type[~torch.nn.modules.module.Module] | ~collections.abc.Callable | None = None*, *norm_kwargs: dict | list[dict] | None = None*, *dropout: float | None = None*, *bias_last_layer: bool = True*, *single_bias_last_layer: bool = False*, *layer_class: type[~torch.nn.modules.module.Module] | ~collections.abc.Callable = <class 'torch.nn.modules.linear.Linear'>*, *layer_kwargs: dict | None = None*, *activate_last_layer: bool = False*, *device: ~torch.device | str | int | None = None*)[[source]](../../_modules/torchrl/modules/models/models.html#MLP)

A multi-layer perceptron.

If MLP receives more than one input, it concatenates them all along the last dimension before passing the
resulting tensor through the network. This is aimed at allowing for a seamless interface with calls of the type of

```
>>> model(state, action) # compute state-action value
```

In the future, this feature may be moved to the ProbabilisticTDModule, though it would require it to handle
different cases (vectors, images, ...)

Parameters:

- **in_features** (*int**,**optional*) - number of input features;
- **out_features** (*int**,*[*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or**equivalent*) - number of output
features. If iterable of integers, the output is reshaped to the
desired shape.
- **depth** (*int**,**optional*) - depth of the network. A depth of 0 will produce
a single linear layer network with the desired input and output size.
A length of 1 will create 2 linear layers etc. If no depth is indicated,
the depth information should be contained in the `num_cells`
argument (see below). If `num_cells` is an iterable and depth is
indicated, both should match: `len(num_cells)` must be equal to
`depth`.
Defaults to `0` (no depth - the network contains a single linear layer).
- **num_cells** (*int**or**sequence**of**int**,**optional*) - number of cells of every
layer in between the input and output. If an integer is provided,
every layer will have the same number of cells. If an iterable is provided,
the linear layers `out_features` will match the content of
`num_cells`. Defaults to `32`;
- **activation_class** (*Type**[**nn.Module**] or**callable**,**optional*) - activation
class or constructor to be used.
Defaults to [`Tanh`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Tanh.html#torch.nn.Tanh).
- **activation_kwargs** (*dict**or**list**of**dicts**,**optional*) - kwargs to be used
with the activation class. Also accepts a list of kwargs of length
`depth + int(activate_last_layer)`.
- **norm_class** (*Type**or**callable**,**optional*) - normalization class or
constructor, if any.
- **norm_kwargs** (*dict**or**list**of**dicts**,**optional*) - kwargs to be used with
the normalization layers. Also accepts a list of kwargs of length
`depth + int(activate_last_layer)`.
- **dropout** (`float`, optional) - dropout probability. Defaults to `None` (no
dropout);
- **bias_last_layer** (*bool*) - if `True`, the last Linear layer will have a bias parameter.
default: True;
- **single_bias_last_layer** (*bool*) - if `True`, the last dimension of the bias of the last layer will be a singleton
dimension.
default: True;
- **layer_class** (*Type**[**nn.Module**] or**callable**,**optional*) - class to be used
for the linear layers;
- **layer_kwargs** (*dict**or**list**of**dicts**,**optional*) - kwargs for the linear
layers. Also accepts a list of kwargs of length `depth + 1`.
- **activate_last_layer** (*bool*) - whether the MLP output should be activated. This is useful when the MLP output
is used as the input for another module.
default: False.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - device to create the module on.

Examples

```
>>> # All of the following examples provide valid, working MLPs
>>> mlp = MLP(in_features=3, out_features=6, depth=0) # MLP consisting of a single 3 x 6 linear layer
>>> print(mlp)
MLP(
 (0): Linear(in_features=3, out_features=6, bias=True)
)
>>> mlp = MLP(in_features=3, out_features=6, depth=4, num_cells=32)
>>> print(mlp)
MLP(
 (0): Linear(in_features=3, out_features=32, bias=True)
 (1): Tanh()
 (2): Linear(in_features=32, out_features=32, bias=True)
 (3): Tanh()
 (4): Linear(in_features=32, out_features=32, bias=True)
 (5): Tanh()
 (6): Linear(in_features=32, out_features=32, bias=True)
 (7): Tanh()
 (8): Linear(in_features=32, out_features=6, bias=True)
)
>>> mlp = MLP(out_features=6, depth=4, num_cells=32) # LazyLinear for the first layer
>>> print(mlp)
MLP(
 (0): LazyLinear(in_features=0, out_features=32, bias=True)
 (1): Tanh()
 (2): Linear(in_features=32, out_features=32, bias=True)
 (3): Tanh()
 (4): Linear(in_features=32, out_features=32, bias=True)
 (5): Tanh()
 (6): Linear(in_features=32, out_features=32, bias=True)
 (7): Tanh()
 (8): Linear(in_features=32, out_features=6, bias=True)
)
>>> mlp = MLP(out_features=6, num_cells=[32, 33, 34, 35]) # defines the depth by the num_cells arg
>>> print(mlp)
MLP(
 (0): LazyLinear(in_features=0, out_features=32, bias=True)
 (1): Tanh()
 (2): Linear(in_features=32, out_features=33, bias=True)
 (3): Tanh()
 (4): Linear(in_features=33, out_features=34, bias=True)
 (5): Tanh()
 (6): Linear(in_features=34, out_features=35, bias=True)
 (7): Tanh()
 (8): Linear(in_features=35, out_features=6, bias=True)
)
>>> mlp = MLP(out_features=(6, 7), num_cells=[32, 33, 34, 35]) # returns a view of the output tensor with shape [*, 6, 7]
>>> print(mlp)
MLP(
 (0): LazyLinear(in_features=0, out_features=32, bias=True)
 (1): Tanh()
 (2): Linear(in_features=32, out_features=33, bias=True)
 (3): Tanh()
 (4): Linear(in_features=33, out_features=34, bias=True)
 (5): Tanh()
 (6): Linear(in_features=34, out_features=35, bias=True)
 (7): Tanh()
 (8): Linear(in_features=35, out_features=42, bias=True)
)
>>> from torchrl.modules import NoisyLinear
>>> mlp = MLP(out_features=(6, 7), num_cells=[32, 33, 34, 35], layer_class=NoisyLinear) # uses NoisyLinear layers
>>> print(mlp)
MLP(
 (0): NoisyLazyLinear(in_features=0, out_features=32, bias=False)
 (1): Tanh()
 (2): NoisyLinear(in_features=32, out_features=33, bias=True)
 (3): Tanh()
 (4): NoisyLinear(in_features=33, out_features=34, bias=True)
 (5): Tanh()
 (6): NoisyLinear(in_features=34, out_features=35, bias=True)
 (7): Tanh()
 (8): NoisyLinear(in_features=35, out_features=42, bias=True)
)
```

forward(**inputs: tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)[[source]](../../_modules/torchrl/modules/models/models.html#MLP.forward)

Runs the forward pass.