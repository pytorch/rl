.. currentmodule:: torchrl.checkpoint

Checkpointing
=============

TorchRL checkpoints use one manifest-driven format for standalone scripts,
trainers, and policy-only consumers. Components are registered independently,
so a checkpoint may contain only a policy or a complete training state.

To interrupt a run and continue it later, see :ref:`checkpoint_resume`. To
ship a trained policy for inference, see the :ref:`export tutorial <export_tuto>`.
Those are different jobs: a resume checkpoint keeps optimizer, collector,
replay-buffer, and target-network state; an export drops them on purpose.

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

.. _checkpoint_resume:

Resume training
---------------

A **checkpoint** is a snapshot you can interrupt and resume from. An
**export** is a deployable inference artifact. They are not interchangeable.

The :ref:`export tutorial <export_tuto>` isolates a trained policy for ONNX,
AOTInductor, or another runtime. That path is inference-only: it drops the
optimizer, replay buffer, collector, exploration schedule, and target
networks. Use :class:`Checkpoint` when the goal is to continue training.

Reconstruct the same objects first (same module classes, same
:class:`~tensordict.nn.TensorDictModule` wiring, same replay-buffer storage
type), register those live objects, then save or load. :class:`Checkpoint`
does not rebuild an architecture from the file. What it persists is each
component's ``state_dict`` or ``dump`` payload -- the same mapping
``policy.state_dict()`` would return, not a pickled container.

What to save
~~~~~~~~~~~~

Resuming an RL job is more than ``torch.save(model.state_dict())``. The
pieces below are independent, and omitting any of them is why a
weights-only dump cannot restart training.

**Training setup**

* **Training model.** The :class:`~tensordict.nn.TensorDictModule` (or
  :class:`~torch.nn.Module`) being optimized. Save that module. The
  adapter writes its ``state_dict``; you do not need a wrapper type that
  exists only for serialization.
* **Optimizer.** Adam / RMSprop moments, step counts, and param-group
  state. Without it, a resume is a fresh run that happens to start from
  pretrained weights.
* **Target networks.** Off-policy losses such as
  :class:`~torchrl.objectives.DQNLoss` keep a delayed copy under
  ``target_*_params`` when ``delay_value=True`` (or the equivalent flag).
  Those tensors live on the loss, not on the policy the collector runs.
  Register the loss -- or the target params themselves -- or a resume
  re-bootstraps the targets from the current online weights.

There is no reserved ``target=`` keyword on :class:`Checkpoint`. Any extra
name is a regular component, so ``loss_module=loss`` is the usual way to
keep targets, and
``checkpoint.register("target_params", loss.target_value_network_params)``
is available when a consumer needs the targets on their own.

**Collector**

* **Inference / exploration policy.** The module the collector actually
  executes, including exploration state such as the ``eps`` buffer of
  :class:`~torchrl.modules.EGreedyModule`. This is often a wrapper around
  the training model rather than a second network.
* **Environment and transform state.** Observation-norm statistics,
  frame-stack buffers, step counters. A collector ``state_dict`` stores
  the env/transform state of a :class:`~torchrl.envs.TransformedEnv`.
* **Collector frame count.** ``frames`` and ``iter``, so schedules
  (epsilon annealing, ``init_random_frames``, total-frame budgets)
  continue instead of restarting at zero.

**Replay buffer**

:meth:`~torchrl.data.ReplayBuffer.dump` / :meth:`~torchrl.data.ReplayBuffer.load`
persist the writer (next-write cursor), the sampler (including priorities
on a prioritized buffer), the buffer transforms, and the underlying
storage. A restored buffer that is missing the writer or sampler is a
pile of transitions you can no longer address correctly.

**RNG**

:class:`GlobalRNGState` captures process-global Python, NumPy, and Torch
RNGs so sampling and exploration stay reproducible across a resume.

Shared modules
~~~~~~~~~~~~~~

:meth:`~torchrl.objectives.LossModule.convert_to_functional` wraps the
module you passed in. For ``DQNLoss(value_network=policy)``:

* ``loss.value_network`` is that same :class:`~tensordict.nn.TensorDictModule`;
* ``loss.value_network_params`` holds its parameters;
* ``loss.target_value_network_params`` is a detached copy used as the
  target.

The online weights are therefore **not** an independent copy of
``policy``. Saving both ``policy`` and ``loss_module`` as if they were
two networks writes the same tensors twice. Loading those two payloads
into a pair of *separately constructed* modules desynchronizes the
collector from the loss -- the confusion that
`#3032 <https://github.com/pytorch/rl/issues/3032>`_ describes.

Rebuild the same graph first (pass the same ``policy`` into the loss),
then persist one owner of the online weights:

* ``loss_module=loss`` covers online and target parameters;
* ``policy=policy`` is enough when a policy-only consumer will load just
  that component, but then register the target params (or the whole
  loss) as well if you intend to resume training.

A :class:`~torchrl.trainers.Trainer` may register both names for
convenience. They still share the online weights. Do not construct a
second network and save it under another name thinking it is a
distinct snapshot.

Construct, save, and load
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    from torchrl.checkpoint import Checkpoint, GlobalRNGState

    checkpoint = Checkpoint(
        policy=policy,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        collector=collector,
        loss_module=loss,
        exploration=exploration,
        rng=GlobalRNGState(),
    )
    checkpoint.save("run/checkpoint")
    result = checkpoint.load(
        "run/checkpoint",
        components={
            "policy",
            "optimizer",
            "replay_buffer",
            "collector",
            "loss_module",
            "exploration",
            "rng",
        },
        map_location="cpu",
    )

``components`` on :meth:`Checkpoint.load` selects which already-registered
objects to restore. Unrequested payloads, including a large replay
buffer, stay on disk. ``map_location`` is forwarded to the state-dict
adapter, the same way it is for :func:`torch.load`.

To persist only the :class:`~tensordict.nn.TensorDictModule` -- still a
``state_dict``, not a pickle of the Python object -- register that
module alone:

.. code-block:: python

    policy_only = Checkpoint(policy=policy)
    policy_only.save("run/policy-only")
    # Later, after reconstructing the same TensorDictModule:
    Checkpoint(policy=new_policy).load(
        "run/policy-only",
        components={"policy"},
        map_location="cpu",
    )

That policy-only file is a checkpoint of the training weights. It is
not an export: it still needs TorchRL to rebuild the module, and it
cannot resume an optimizer, buffer, or target network.

Copy-paste DQN example
~~~~~~~~~~~~~~~~~~~~~~

The snippet below rebuilds the same object graph before loading. Gym
is required; skip it in doctest environments that do not have it.

.. code-block:: python

    from torch import nn
    from torch.optim import Adam
    from tensordict.nn import TensorDictModule, TensorDictSequential

    from torchrl.checkpoint import Checkpoint, GlobalRNGState
    from torchrl.collectors import Collector
    from torchrl.data import LazyTensorStorage, ReplayBuffer
    from torchrl.envs import GymEnv, StepCounter, TransformedEnv
    from torchrl.modules import EGreedyModule, QValueModule
    from torchrl.objectives import DQNLoss, SoftUpdate

    def make_agent():  # doctest: +SKIP
        env = TransformedEnv(GymEnv("CartPole-v1"), StepCounter())
        value_net = TensorDictModule(
            nn.Sequential(nn.Linear(4, 64), nn.ReLU(), nn.Linear(64, 2)),
            in_keys=["observation"],
            out_keys=["action_value"],
        )
        policy = TensorDictSequential(
            value_net, QValueModule(spec=env.action_spec)
        )
        exploration = EGreedyModule(
            env.action_spec, annealing_num_steps=10_000
        )
        policy_explore = TensorDictSequential(policy, exploration)
        collector = Collector(
            env,
            policy_explore,
            frames_per_batch=64,
            total_frames=10_000,
        )
        replay_buffer = ReplayBuffer(storage=LazyTensorStorage(20_000))
        # Same TensorDictModule as ``policy``: online weights are shared;
        # target weights live only on the loss.
        loss = DQNLoss(
            value_network=policy,
            action_space=env.action_spec,
            delay_value=True,
        )
        optimizer = Adam(loss.parameters())
        SoftUpdate(loss, eps=0.99)
        return policy, exploration, collector, replay_buffer, loss, optimizer

    policy, exploration, collector, replay_buffer, loss, optimizer = (
        make_agent()
    )  # doctest: +SKIP
    batch = next(iter(collector))  # doctest: +SKIP
    replay_buffer.extend(batch)  # doctest: +SKIP

    checkpoint = Checkpoint(  # doctest: +SKIP
        policy=policy,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        collector=collector,
        loss_module=loss,
        exploration=exploration,
        rng=GlobalRNGState(),
    )
    checkpoint.save("run/checkpoint")  # doctest: +SKIP

    # Interrupt. Rebuild the same graph, then restore into those objects.
    policy, exploration, collector, replay_buffer, loss, optimizer = (
        make_agent()
    )  # doctest: +SKIP
    restored = Checkpoint(  # doctest: +SKIP
        policy=policy,
        optimizer=optimizer,
        replay_buffer=replay_buffer,
        collector=collector,
        loss_module=loss,
        exploration=exploration,
        rng=GlobalRNGState(),
    )
    restored.load(  # doctest: +SKIP
        "run/checkpoint",
        components={
            "policy",
            "optimizer",
            "replay_buffer",
            "collector",
            "loss_module",
            "exploration",
            "rng",
        },
        map_location="cpu",
    )

Train as usual between construction and ``save``. A
:class:`~torchrl.trainers.Trainer` can register the same names for you;
see :ref:`Trainer integration <checkpoint-trainer>` below and the
:doc:`checkpointing tutorial <../tutorials/checkpointing>`.

Checkpoint versus export
~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 20 40 40

   * -
     - Checkpoint
     - Export
   * - Intent
     - Interrupt and resume, or inspect an intermediate run
     - Deploy a policy
   * - Typical contents
     - Policy, optimizer, loss/targets, collector, replay buffer, RNG
     - Isolated inference module
   * - Consumer
     - The same training script
     - ONNX, AOTInductor, edge runtime
   * - API
     - :class:`Checkpoint`
     - :ref:`export tutorial <export_tuto>`

A checkpoint can later be reduced to a policy-only file (register just
``policy`` and save). Turning that file into a standalone executable is
the export tutorial's job, not :meth:`Checkpoint.save`.

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

.. _checkpoint-trainer:

Trainer integration
-------------------

Pass a rotation policy with a unified checkpoint to retain scheduled Trainer
checkpoints. The Trainer uses ``collected_frames`` as the checkpoint step and
adds ``collected_frames`` and ``optim_steps`` to the manifest metadata.

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
    StateDictCheckpointAdapter
    StateDictFormat
