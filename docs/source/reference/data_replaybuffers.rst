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
    ReplayBufferEnsemble
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
is made of a :class:`~torchrl.data.ReplayBuffer` base with a
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

The full physical storage can be read with ``rb[:]``. This is useful when all
stored items must be processed in storage order, for example to recompute value
targets after collection. :meth:`~torchrl.data.ReplayBuffer.read_all_in_order`
is an explicit equivalent to ``rb[:]``, and
:meth:`~torchrl.data.ReplayBuffer.write_all` is an explicit equivalent to
``rb[:] = data``. Passing ``end=...`` to these helpers updates only the leading
storage entries.

    >>> from tensordict import TensorDict
    >>> import torch
    >>> from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    >>> rb = TensorDictReplayBuffer(storage=LazyTensorStorage(10))
    >>> rb.extend(TensorDict({"obs": torch.arange(3)}, [3]))
    tensor([0, 1, 2])
    >>> data = rb.read_all_in_order()
    >>> assert (data == rb[:]).all()
    >>> data["target"] = data["obs"] + 1
    >>> rb.write_all(data)
    >>> assert (rb[:] == data).all()

TED-format conversion
~~~~~~~~~~~~~~~~~~~~~

The following helpers convert between the TorchRL Episode Data (TED) layout and
a flat, storage-friendly representation when serializing or restoring a buffer:

.. currentmodule:: torchrl.data

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    TED2Flat
    Flat2TED

Video-backed replay buffers
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Video-backed datasets are dominated by frames; materializing every decoded frame
as a dense tensor throws away the video codec's compression. :class:`VideoClipRef`
is a lightweight, picklable reference to frames inside an encoded video (mp4, ...):
it stores only *where* the frames are (a source path/URI plus a ``frame_index``),
so indexing the whole buffer stays cheap. Frames are decoded on-demand with
torchcodec by :class:`~torchrl.envs.transforms.DecodeVideoTransform`, appended on
the replay-buffer sample path, so ``rb.sample()`` returns decoded frames aligned to
the sampled steps. It composes with :class:`SliceSampler`: a contiguous window of
sampled steps maps to consecutive frame indices and decodes as a single ranged
read. Decoders are opened lazily and cached per worker process (see
:func:`set_video_decoder_cache_size` and :func:`clear_video_decoder_cache`); the
references stored in the buffer never hold an open decoder.

.. currentmodule:: torchrl.data

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    VideoClipRef
    clear_video_decoder_cache
    set_video_decoder_cache_size
