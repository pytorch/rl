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

TorchRL checkpoints target local filesystems. URI paths and coordinated
distributed rank checkpoints are rejected rather than importing an optional
remote-storage stack implicitly.

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
objects normally use ``state_dict`` and ``load_state_dict``. Their tensor state
is stored with :func:`tensordict.save` by default, while a JSON schema preserves
the state-dict structure without pickle. JSON-compatible configuration,
metrics, and metadata are also stored without pickle.

Set ``save_components={"policy", "optimizer", "trainer_state"}`` on a
:class:`Checkpoint` to keep large components such as replay buffers out of
scheduled Trainer saves. An explicit ``components=`` argument to
:meth:`Checkpoint.save` overrides this default selection.

State-dict payload formats
--------------------------

The inferred :class:`StateDictCheckpointAdapter` writes a TensorDict directory.
The same adapter can write a TensorDict ZIP archive or consolidated file, and
loads auto-detect all of these payloads. This component payload choice is
independent of the outer :class:`Checkpoint` directory or archive container.

.. code-block:: python

    from torchrl.checkpoint import Checkpoint, StateDictCheckpointAdapter

    checkpoint = Checkpoint().register(
        "policy",
        policy,
        adapter=StateDictCheckpointAdapter(payload_format="archive"),
    )

Use ``payload_format="consolidated"`` for consolidated TensorDict storage.
Pickle-based :func:`torch.save` remains available explicitly with
``payload_format="torch"``. TensorDict payloads reject unsupported Python
objects with an error that points to this opt-in rather than silently falling
back to pickle.

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

Reading a component without its object
--------------------------------------

:meth:`Checkpoint.read_component` returns the stored payload of one state-dict
or JSON component, so a checkpoint can be inspected before the objects it
belongs to exist. Tensors are copied out of the checkpoint. Components stored
through ``dump`` and ``load`` require a live object and are rejected.

Checkpoint rotation
-------------------

:class:`CheckpointRotation` retains the newest checkpoints and can preserve an
older checkpoint with the best recorded metric. Metrics are read from manifest
metadata.

.. code-block:: python

    from torchrl.checkpoint import Checkpoint, CheckpointRotation

    checkpoint = Checkpoint(policy=policy, optimizer=optimizer)
    rotation = CheckpointRotation(
        "run/checkpoints",
        keep_last=3,
        keep_best=("eval_reward", "max"),
    )
    rotation.save(
        checkpoint,
        step=100_000,
        metadata={"eval_reward": 42.5},
    )
    rotation.load_latest(checkpoint)

Trainer integration
-------------------

Pass a rotation policy with a unified checkpoint to retain scheduled Trainer
checkpoints. The Trainer uses ``collected_frames`` as the checkpoint step and
adds ``collected_frames`` and ``optim_steps`` to the manifest metadata.

The Trainer registers the process-global RNG state under ``rng`` and restores
it after every other component. ``Trainer.load_from_file`` accepts a checkpoint
path or a rotation directory, in which case the newest retained checkpoint is
restored. Scheduled saves in asynchronous collection mode pause the collector
while the checkpoint is written.

A :class:`~torchrl.record.loggers.WandbLogger` refuses to load state written by
a different W&B run; construct it for the saved run, for example through
``get_logger(..., state_dict=...)``, before loading the trainer.

.. code-block:: python

    trainer = SACTrainer(
        ...,
        checkpoint=Checkpoint(),
        checkpoint_rotation=CheckpointRotation(
            "run/checkpoints",
            keep_last=3,
            keep_best=("eval_reward", "max"),
        ),
        checkpoint_metadata=lambda trainer: {
            "eval_reward": evaluation_state["reward"]
        },
    )

The metadata callback runs immediately before each save. Metrics used by
``keep_best`` should describe the checkpoint being saved rather than an older
evaluation.

Stopping at a safe boundary
---------------------------

:class:`StopOnSignal` turns ``SIGINT`` and ``SIGTERM`` into a stop request that
a training loop checks between batches, so the current batch completes and a
final checkpoint is written before the process exits. A second signal raises
``KeyboardInterrupt`` for loops that cannot reach a boundary. Previous handlers
are restored when the context exits. ``Trainer.stop_on_signal`` wraps the same
helper and calls ``Trainer.request_stop``:

.. code-block:: python

    with trainer.stop_on_signal():
        trainer.train()

Standalone scripts use the helper directly:

.. code-block:: python

    with StopOnSignal() as stop:
        for batch in collector:
            ...
            if stop.requested:
                break
        rotation.save(checkpoint, step=step)

Resuming recipes
----------------

The ``sota-implementations/*_trainer`` recipes and the standalone ``sac``,
``td3`` and ``ddpg`` recipes save rotated checkpoints under ``checkpoints/`` in
their Hydra run directory and stop cleanly on ``SIGINT`` or ``SIGTERM``. A run
continues from its checkpoint directory with a single override; the saved
configuration is the base and further overrides apply on top of it:

.. code-block:: bash

    python sota-implementations/sac_trainer/train.py resume=outputs/<date>/<time>/checkpoints
    python sota-implementations/sac/sac.py resume=outputs/<date>/<time>/checkpoints collector.total_frames=2_000_000

Three helpers implement this flow. :func:`resolve_checkpoint_path` maps a
rotation directory to its newest checkpoint. :func:`resume_config` reads the
saved ``config`` component with :meth:`Checkpoint.read_component` and applies
the current command-line overrides on top of it. The saved ``logger`` component,
passed to ``get_logger(..., state_dict=...)``, reopens a W&B run with
``resume="must"`` or keeps a CSV or TensorBoard logger appending to the saved
directory. Trainer recipes get all of this from
:func:`~torchrl.trainers.algorithms.configs.instantiate_trainer`; :class:`RunCheckpointer` gives standalone scripts the same behavior: it
restores every component but ``config`` and ``rng``, then ``rng`` last, saves
every ``interval`` steps through a :class:`CheckpointRotation` under
:class:`StopOnSignal`, and keeps saving next to the resumed checkpoint.

Compatibility
-------------

The manifest records the checkpoint format version, adapter versions, component
files, and TorchRL, TensorDict, and PyTorch versions. Newer unsupported formats
and incompatible adapters fail clearly. A dependency-version mismatch does not
block restoration, but emits a warning and is reported by
:attr:`CheckpointLoadResult.comparison`. This lets long-running off-policy jobs
resume across environment changes while retaining an explicit compatibility
signal. Manifests created before dependency provenance was recorded continue to
load silently.

Partial restoration reports loaded, missing, incompatible, and unrequested
components through :class:`CheckpointLoadResult`.

Trainer's legacy ``CKPT_BACKEND`` path remains available during the migration
window. Passing ``checkpoint=Checkpoint(...)`` to a trainer opts into the
unified format. Existing torch, torchsnapshot, and memmap trainer checkpoints
remain readable.

The :func:`torchrl.render.save_render_checkpoint` helper also keeps its legacy
``torch.save`` payload by default during the compatibility window. Pass
``format="archive"`` or ``format="directory"`` to opt into the unified format;
the default changes in v0.15.

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
    CheckpointRotation
    CheckpointFormat
    CheckpointStrictness
    DumpLoadCheckpointAdapter
    GlobalRNGState
    JSONCheckpointAdapter
    RunCheckpointer
    StateDictCheckpointAdapter
    StateDictFormat
    StopOnSignal

.. autosummary::
    :toctree: generated/
    :template: rl_template_fun.rst

    resolve_checkpoint_path
    resume_config
