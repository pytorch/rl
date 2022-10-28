.. currentmodule:: torchrl.data

torchrl.data package
====================

Replay Buffers
--------------

Replay buffers are a central part of off-policy RL algorithms. TorchRL provides an efficient implementation of a few,
widely used replay buffers:


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ReplayBuffer
    PrioritizedReplayBuffer
    TensorDictReplayBuffer
    TensorDictPrioritizedReplayBuffer

Composable Replay Buffers (Prototype)
-------------------------------------

We also provide a prototyped composable replay buffer.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    .. currentmodule:: torchrl.data.replay_buffers

    torchrl.data.replay_buffers.rb_prototype.ReplayBuffer
    torchrl.data.replay_buffers.rb_prototype.TensorDictReplayBuffer
    torchrl.data.replay_buffers.samplers.Sampler
    torchrl.data.replay_buffers.samplers.RandomSampler
    torchrl.data.replay_buffers.samplers.PrioritizedSampler
    torchrl.data.replay_buffers.storages.Storage
    torchrl.data.replay_buffers.storages.ListStorage
    torchrl.data.replay_buffers.storages.LazyTensorStorage
    torchrl.data.replay_buffers.storages.LazyMemmapStorage
    torchrl.data.replay_buffers.writers.Writer
    torchrl.data.replay_buffers.writers.RoundRobinWriter



TensorDict
----------

Passing data across objects can become a burdensome task when designing high-level classes: for instance it can be
hard to design an actor class that can take an arbitrary number of inputs and return an arbitrary number of inputs. The
`TensorDict` class simplifies this process by packing together a bag of tensors in a dictionary-like object. This
class supports a set of basic operations on tensors to facilitate the manipulation of entire batch of data (e.g.
`torch.cat`, `torch.stack`, `.to(device)` etc.).


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    TensorDict
    SubTensorDict
    LazyStackedTensorDict

TensorSpec
----------

The `TensorSpec` parent class and subclasses define the basic properties of observations and actions in TorchRL, such
as shape, device, dtype and domain.


.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    TensorSpec
    BoundedTensorSpec
    OneHotDiscreteTensorSpec
    UnboundedContinuousTensorSpec
    NdBoundedTensorSpec
    NdUnboundedContinuousTensorSpec
    BinaryDiscreteTensorSpec
    MultOneHotDiscreteTensorSpec
    DiscreteTensorSpec
    CompositeSpec


Utils
-----

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    utils.expand_as_right
    utils.expand_right
    MultiStep
