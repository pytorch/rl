# TensorSpec System

TensorSpec classes define the shape, dtype, and domain of tensors in TorchRL.

| [`Binary`](generated/torchrl.data.Binary.html#torchrl.data.Binary)([n, shape, device, dtype]) | A binary discrete tensor spec. |
| --- | --- |
| [`Bounded`](generated/torchrl.data.Bounded.html#torchrl.data.Bounded)(*args, **kwargs) | A bounded tensor spec. |
| [`Categorical`](generated/torchrl.data.Categorical.html#torchrl.data.Categorical)(n[, shape, device, dtype, mask]) | A discrete tensor spec. |
| [`Composite`](generated/torchrl.data.Composite.html#torchrl.data.Composite)(*args, **kwargs) | A composition of TensorSpecs. |
| [`MultiCategorical`](generated/torchrl.data.MultiCategorical.html#torchrl.data.MultiCategorical)(nvec[, shape, device, ...]) | A concatenation of discrete tensor spec. |
| [`MultiOneHot`](generated/torchrl.data.MultiOneHot.html#torchrl.data.MultiOneHot)(nvec[, shape, device, dtype, ...]) | A concatenation of one-hot discrete tensor spec. |
| [`NonTensor`](generated/torchrl.data.NonTensor.html#torchrl.data.NonTensor)([shape, device, dtype, ...]) | A spec for non-tensor data. |
| [`OneHot`](generated/torchrl.data.OneHot.html#torchrl.data.OneHot)(n[, shape, device, dtype, ...]) | A unidimensional, one-hot discrete tensor spec. |
| [`Stacked`](generated/torchrl.data.Stacked.html#torchrl.data.Stacked)(*specs, dim) | A lazy representation of a stack of tensor specs. |
| [`StackedComposite`](generated/torchrl.data.StackedComposite.html#torchrl.data.StackedComposite)(*args, **kwargs) | A lazy representation of a stack of composite specs. |
| [`TensorSpec`](generated/torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)(shape, space, device, dtype, ...) | Parent class of the tensor meta-data containers. |
| [`Unbounded`](generated/torchrl.data.Unbounded.html#torchrl.data.Unbounded)(*args, **kwargs) | An unbounded tensor spec. |
| [`UnboundedContinuous`](generated/torchrl.data.UnboundedContinuous.html#torchrl.data.UnboundedContinuous)(*args, **kwargs) | A specialized version of [`torchrl.data.Unbounded`](generated/torchrl.data.Unbounded.html#torchrl.data.Unbounded) with continuous space. |
| [`UnboundedDiscrete`](generated/torchrl.data.UnboundedDiscrete.html#torchrl.data.UnboundedDiscrete)(*args, **kwargs) | A specialized version of [`torchrl.data.Unbounded`](generated/torchrl.data.Unbounded.html#torchrl.data.Unbounded) with discrete space. |

## Supported PyTorch Operations

TensorSpec classes support various PyTorch-like operations for manipulating their shape and structure.
These operations return new spec instances with the modified shape while preserving dtype, device, and domain information.

**PyTorch function overloads** (via `__torch_function__`):

These can be called using the standard PyTorch functional API:

```
import torch
from torchrl.data import Bounded, Composite

# torch.stack - stack multiple specs along a new dimension
spec1 = Bounded(low=0, high=1, shape=(3, 4))
spec2 = Bounded(low=0, high=1, shape=(3, 4))
stacked = torch.stack([spec1, spec2], dim=0) # shape: (2, 3, 4)

# torch.squeeze / torch.unsqueeze - remove or add singleton dimensions
spec = Bounded(low=0, high=1, shape=(1, 3, 4))
squeezed = torch.squeeze(spec, dim=0) # shape: (3, 4)
unsqueezed = torch.unsqueeze(squeezed, dim=0) # shape: (1, 3, 4)

# torch.index_select - select indices along a dimension
spec = Bounded(low=0, high=1, shape=(5, 4))
selected = torch.index_select(spec, dim=0, index=torch.tensor([0, 2, 4])) # shape: (3, 4)
```

**Instance methods**:

TensorSpec also provides instance methods that mirror common tensor operations:

- [`expand()`](generated/torchrl.data.TensorSpec.html#torchrl.data.TensorSpec.expand) - broadcast the spec to a larger shape
- [`squeeze()`](generated/torchrl.data.TensorSpec.html#torchrl.data.TensorSpec.squeeze) - remove singleton dimensions
- [`unsqueeze()`](generated/torchrl.data.TensorSpec.html#torchrl.data.TensorSpec.unsqueeze) - add a singleton dimension
- [`reshape()`](generated/torchrl.data.TensorSpec.html#torchrl.data.TensorSpec.reshape) - reshape the spec to a new shape
- [`flatten()`](generated/torchrl.data.TensorSpec.html#torchrl.data.TensorSpec.flatten) - flatten dimensions
- [`unflatten()`](generated/torchrl.data.TensorSpec.html#torchrl.data.TensorSpec.unflatten) - unflatten a dimension into multiple dimensions
- `unbind()` - split the spec along a dimension

```
from torchrl.data import Bounded

spec = Bounded(low=0, high=1, shape=(2, 3, 4))

# Reshape operations
reshaped = spec.reshape(6, 4) # shape: (6, 4)
flattened = spec.flatten(0, 1) # shape: (6, 4)
expanded = spec.expand(5, 2, 3, 4) # shape: (5, 2, 3, 4)

# Split operations
unbound = spec.unbind(dim=0) # tuple of 2 specs, each with shape (3, 4)
```

Note

Some operations have restrictions for discrete specs like [`OneHot`](generated/torchrl.data.OneHot.html#torchrl.data.OneHot), [`MultiOneHot`](generated/torchrl.data.MultiOneHot.html#torchrl.data.MultiOneHot),
and [`Binary`](generated/torchrl.data.Binary.html#torchrl.data.Binary), where the last dimension represents the domain and cannot be modified.
For example, `torch.index_select` along the last dimension of a [`OneHot`](generated/torchrl.data.OneHot.html#torchrl.data.OneHot) spec will raise
a `ValueError`.