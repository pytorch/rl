.. currentmodule:: torchrl.trainers

Learners and Learner Groups
===========================

Learners own bounded optimization work while a trainer remains responsible for
collection, scheduling, logging, checkpoint cadence, and stopping. A learner
group is the synchronization boundary used by a controller; the controller does
not issue collective-bearing commands to individual ranks.

Target-network replicas
-----------------------

Each learner rank owns its own online-network and target-network tensors. During
initialization, the learner broadcasts the complete loss module, including both
sets of parameters and buffers. Gradient averaging then makes the optimizer
update identical on every rank. After that synchronized optimizer step, each
rank invokes its actor-local target updater exactly once. Deterministic updaters
such as :class:`~torchrl.objectives.HardUpdate` and
:class:`~torchrl.objectives.SoftUpdate` therefore keep target replicas aligned
without an additional collective.

The updater must be constructed with the same loss module passed to
:class:`Learner`. Its counter is part of learner checkpoint state; restore loads
the same counter and replicated loss state on every rank before broadcasting the
module and resuming commands. A custom updater used with data parallelism must
apply the same deterministic transition on every rank.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    OptimizationContext
    Learner
    LearnerGroup
    LocalLearnerGroup

Ray learner groups
------------------

Passing a learner group and replay owner to
:class:`~torchrl.trainers.Trainer` selects the central-controller execution
path. Loss modules, optimizers, optimization steppers, target updaters, replay
sampling, and priority updates remain learner-owned. The controller converts
replay write progress into update credit, issues consecutive bounded rounds,
publishes rank-zero policy weights to collectors, and remains the only process
that logs or decides when the run stops. Driver-owned optimization objects and
legacy learner-side hooks are rejected in this mode.

:class:`~torchrl.trainers.algorithms.DQNTrainer` can additionally compose its
controller-owned exploration state with each versioned learner policy before
publication. See ``examples/distributed/replay_buffers/ray_learner_dqn.py`` for
a complete two-rank Ray example.

Off-policy actor-critic learners
--------------------------------

The learner boundary is loss-agnostic: :class:`Learner` accepts any TorchRL
:class:`~torchrl.objectives.LossModule`, and named models determine which
weights a controller can publish. For SAC, DDPG, and TD3, construct the loss,
optimizer or optimization stepper, and target updater inside the learner
factory, then expose the actor as ``models={"policy": loss.actor_network}``.
The learner owns both actor and critic optimization state while collectors
receive only the actor weights.

:class:`~torchrl.trainers.algorithms.SACTrainer`,
:class:`~torchrl.trainers.algorithms.DDPGTrainer`, and
:class:`~torchrl.trainers.algorithms.TD3Trainer` accept the same
``learner_group`` controller mode as DQN. Their driver-side ``loss_module``,
optimizer, optimization stepper, and target updater must be omitted. TD3 keeps
exploration state on the controller and composes it with each published actor
version, matching its local collector policy structure. This execution path is
for replay-backed off-policy training; on-policy batch distribution is outside
its current contract.

Distributed checkpoints
-----------------------

In learner-group mode, ``save_trainer_file`` names a checkpoint root directory
rather than a single file. Each save pauses background Ray collection after all
in-flight replay writes, barriers learner ranks, and writes a new versioned
subdirectory. Controller and scheduler state, collector state, the Ray-owned
replay state, rank-zero replicated learner state, and each rank's RNG state are
stored separately. ``manifest.json`` is written last before the temporary
directory is atomically renamed, so an interrupted save is not a restore
candidate.

Calling :meth:`~torchrl.trainers.Trainer.load_from_file` with either the root or
a complete versioned subdirectory restores a new fixed-size learner generation.
The world size, global batch size, and learner factory identity must match.
Ranks synchronize restored parameters and barrier before collection resumes.
Recovery is explicit: a failed or timed-out optimizer command is never retried,
and callers should construct a new group and load the last complete checkpoint.
Policy snapshots currently travel through the controller before collector
publication; direct learner-to-collector transport remains a follow-up if
benchmarks show that round trip to be a bottleneck.

.. currentmodule:: torchrl.trainers.distributed

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    RayLearnerGroup
