.. currentmodule:: torchrl.data

.. _ref_specs:

TensorSpec System
=================

TensorSpec classes define the shape, dtype, and domain of tensors in TorchRL.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Binary
    Bounded
    Categorical
    Composite
    MultiCategorical
    MultiOneHot
    NonTensor
    OneHot
    Stacked
    StackedComposite
    TensorSpec
    Unbounded
    UnboundedContinuous
    UnboundedDiscrete

Supported PyTorch Operations
----------------------------

TensorSpec classes support various PyTorch-like operations for manipulating their shape and structure.
These operations return new spec instances with the modified shape while preserving dtype, device, and domain information.

**PyTorch function overloads** (via ``__torch_function__``):

These can be called using the standard PyTorch functional API:

.. code-block:: python

    import torch
    from torchrl.data import Bounded, Composite

    # torch.stack - stack multiple specs along a new dimension
    spec1 = Bounded(low=0, high=1, shape=(3, 4))
    spec2 = Bounded(low=0, high=1, shape=(3, 4))
    stacked = torch.stack([spec1, spec2], dim=0)  # shape: (2, 3, 4)

    # torch.squeeze / torch.unsqueeze - remove or add singleton dimensions
    spec = Bounded(low=0, high=1, shape=(1, 3, 4))
    squeezed = torch.squeeze(spec, dim=0)  # shape: (3, 4)
    unsqueezed = torch.unsqueeze(squeezed, dim=0)  # shape: (1, 3, 4)

    # torch.index_select - select indices along a dimension
    spec = Bounded(low=0, high=1, shape=(5, 4))
    selected = torch.index_select(spec, dim=0, index=torch.tensor([0, 2, 4]))  # shape: (3, 4)

**Instance methods**:

TensorSpec also provides instance methods that mirror common tensor operations:

- :meth:`~TensorSpec.expand` - broadcast the spec to a larger shape
- :meth:`~TensorSpec.squeeze` - remove singleton dimensions
- :meth:`~TensorSpec.unsqueeze` - add a singleton dimension
- :meth:`~TensorSpec.reshape` - reshape the spec to a new shape
- :meth:`~TensorSpec.flatten` - flatten dimensions
- :meth:`~TensorSpec.unflatten` - unflatten a dimension into multiple dimensions
- :meth:`~TensorSpec.unbind` - split the spec along a dimension

.. code-block:: python

    from torchrl.data import Bounded

    spec = Bounded(low=0, high=1, shape=(2, 3, 4))

    # Reshape operations
    reshaped = spec.reshape(6, 4)  # shape: (6, 4)
    flattened = spec.flatten(0, 1)  # shape: (6, 4)
    expanded = spec.expand(5, 2, 3, 4)  # shape: (5, 2, 3, 4)

    # Split operations
    unbound = spec.unbind(dim=0)  # tuple of 2 specs, each with shape (3, 4)

.. note::

    Some operations have restrictions for discrete specs like :class:`OneHot`, :class:`MultiOneHot`,
    and :class:`Binary`, where the last dimension represents the domain and cannot be modified.
    For example, ``torch.index_select`` along the last dimension of a :class:`OneHot` spec will raise
    a ``ValueError``.
