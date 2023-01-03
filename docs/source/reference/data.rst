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
    torchrl.data.replay_buffers.rb_prototype.RemoteTensorDictReplayBuffer
    torchrl.data.replay_buffers.samplers.Sampler
    torchrl.data.replay_buffers.samplers.RandomSampler
    torchrl.data.replay_buffers.samplers.PrioritizedSampler
    torchrl.data.replay_buffers.storages.Storage
    torchrl.data.replay_buffers.storages.ListStorage
    torchrl.data.replay_buffers.storages.LazyTensorStorage
    torchrl.data.replay_buffers.storages.LazyMemmapStorage
    torchrl.data.replay_buffers.writers.Writer
    torchrl.data.replay_buffers.writers.RoundRobinWriter

Storage choice is very influential on replay buffer sampling latency, especially in distributed reinforcement learning settings with larger data volumes.
:class:`LazyMemmapStorage` is highly advised in distributed settings with shared storage due to the lower serialisation cost of MemmapTensors as well as the ability to specify file storage locations for improved node failure recovery.
The following mean sampling latency improvements over using ListStorage were found from rough benchmarking in https://github.com/pytorch/rl/tree/main/benchmarks/storage.

+-------------------------------+-----------+
| Storage Type                  | Speed up  |
|                               |           |
+===============================+===========+
| :class:`ListStorage`          | 1x        |
+-------------------------------+-----------+
| :class:`LazyTensorStorage`    | 1.83x     |
+-------------------------------+-----------+
| :class:`LazyMemmapStorage`    | 3.44x     |
+-------------------------------+-----------+


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
    BoundedTensorSpec
    UnboundedContinuousTensorSpec
    BinaryDiscreteTensorSpec
    MultOneHotDiscreteTensorSpec
    DiscreteTensorSpec
    CompositeSpec


Utils
-----

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    MultiStep
