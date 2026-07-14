.. currentmodule:: torchrl.trainers

Learners and Learner Groups
===========================

Learners own bounded optimization work while a trainer remains responsible for
collection, scheduling, logging, checkpoint cadence, and stopping. A learner
group is the synchronization boundary used by a controller; the controller does
not issue collective-bearing commands to individual ranks.

Control-plane records
---------------------

The four frozen dataclasses below are messages at distinct boundaries, not four
independent learner implementations:

* :class:`LearnerContext` is created once per rank at startup. It contains live,
  actor-local dependencies used by the learner factory and is never returned to
  the controller.
* :class:`LearnerStepRequest` is the small atomic command sent unchanged to all
  ranks. Its explicit round and global batch size let ranks reject stale or
  inconsistent work before optimizer state is mutated.
* :class:`LearnerStepResult` certifies what actually completed. A learner group
  compares these receipts across ranks before the controller advances its
  authoritative counters.
* :class:`LearnerWeights` couples a serialized policy snapshot with its model
  name and version. It is fetched only at publication cadence rather than being
  attached to every optimization result.

The records remain separate so each boundary carries only valid data: startup
handles stay out of step commands, requested work remains distinct from
completed work, and weights are serialized only when requested. Plain
dictionaries or keyword arguments would create fewer named classes but would
remove the typed, atomic boundary checked before and after every Ray command.
TensorDict remains the data-plane representation for samples, metrics, and
weights; the dataclasses carry low-volume control metadata and references.

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
    LearnerContext
    LearnerStepRequest
    LearnerStepResult
    LearnerWeights
    Learner
    LearnerGroup
    LocalLearnerGroup

Ray learner groups
------------------

.. currentmodule:: torchrl.trainers.distributed

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    RayLearnerGroup
