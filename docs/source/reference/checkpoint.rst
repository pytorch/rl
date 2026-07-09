.. currentmodule:: torchrl.checkpoint

Checkpointing
=============

TorchRL checkpoints use one manifest-driven format for standalone scripts,
trainers, and policy-only consumers. Components are registered independently,
so a checkpoint may contain only a policy or a complete training state.

The directory and archive containers share the same logical layout. Directory
checkpoints are the default and are best suited to large replay buffers;
archives are convenient single-file artifacts. Loading either container is
automatic.

Format v1 targets local filesystems. URI paths and coordinated distributed
rank checkpoints are rejected rather than importing an optional remote-storage
stack implicitly.

Basic usage
-----------

.. code-block:: python

    from torchrl.checkpoint import Checkpoint, GlobalRNGState

    checkpoint = Checkpoint(
        policy=policy,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        rng=GlobalRNGState(),
    )
    checkpoint.save("run/checkpoint")
    checkpoint.load(
        "run/checkpoint",
        components={"policy", "optimizer", "rng"},
        map_location="cpu",
    )

Replay buffers use their ``dump`` and ``load`` implementations, including the
configured storage checkpointer and compression. Other TorchRL and PyTorch
objects normally use ``state_dict`` and ``load_state_dict``. JSON-compatible
configuration, metrics, and metadata are stored without pickle.

Custom components
-----------------

Objects exposing ``dump(path, ...)`` and ``load(path, ...)`` are detected before
objects exposing ``state_dict`` and ``load_state_dict``. A custom
:class:`CheckpointAdapter` can instead be supplied to
:meth:`Checkpoint.register`, or registered by type on one checkpoint with
:meth:`Checkpoint.register_adapter`.

Use :class:`CheckpointOptions` to preserve component-specific arguments. Options
registered with a component are the baseline; operation-level keyword arguments
override matching entries and explicitly supplied positional arguments replace
the baseline tuple.

Compatibility
-------------

The manifest records the checkpoint format version, adapter versions, component
files, and TorchRL, TensorDict, and PyTorch versions. Newer unsupported formats
and incompatible adapters fail clearly. Partial restoration reports loaded,
missing, incompatible, and unrequested components through
:class:`CheckpointLoadResult`.

Trainer's legacy ``CKPT_BACKEND`` path remains available during the migration
window. Passing ``checkpoint=Checkpoint(...)`` to a trainer opts into the
unified format. Existing torch, torchsnapshot, and memmap trainer checkpoints
remain readable.

API
---

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Checkpoint
    CheckpointAdapter
    CheckpointError
    CheckpointLoadResult
    CheckpointOptions
    CheckpointFormat
    CheckpointStrictness
    DumpLoadCheckpointAdapter
    GlobalRNGState
    JSONCheckpointAdapter
    StateDictCheckpointAdapter
