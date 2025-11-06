.. currentmodule:: torchrl.data

Replay Buffers
==============

Replay buffers are a central part of off-policy RL algorithms. TorchRL provides an efficient implementation of a few,
widely used replay buffers:

Core Replay Buffer Classes
--------------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ReplayBuffer
    PrioritizedReplayBuffer
    TensorDictReplayBuffer
    TensorDictPrioritizedReplayBuffer
    RayReplayBuffer
    RemoteTensorDictReplayBuffer

Composable Replay Buffers
-------------------------

.. _ref_buffers:

We also give users the ability to compose a replay buffer.
We provide a wide panel of solutions for replay buffer usage, including support for
almost any data type; storage in memory, on device or on physical memory;
several sampling strategies; usage of transforms etc.

Supported data types and choosing a storage
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In theory, replay buffers support any data type but we can't guarantee that each
component will support any data type. The most crude replay buffer implementation
is made of a :class:`~torchrl.data.replay_buffers.ReplayBuffer` base with a
:class:`~torchrl.data.replay_buffers.ListStorage` storage. This is very inefficient
but it will allow you to store complex data structures with non-tensor data.
Storages in contiguous memory include :class:`~torchrl.data.replay_buffers.TensorStorage`,
:class:`~torchrl.data.replay_buffers.LazyTensorStorage` and
:class:`~torchrl.data.replay_buffers.LazyMemmapStorage`.

Sampling and indexing
~~~~~~~~~~~~~~~~~~~~~

Replay buffers can be indexed and sampled.
Indexing and sampling collect data at given indices in the storage and then process them
through a series of transforms and ``collate_fn`` that can be passed to the `__init__`
function of the replay buffer.
