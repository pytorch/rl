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

A single generic message was deliberately avoided: startup handles are invalid
in step commands, requested work is not completed work, and carrying weights in
every result would introduce unnecessary serialization and collectives. Using
plain dictionaries or keyword arguments would create fewer named classes but
would remove the typed, atomic boundary checked before and after every Ray
command. TensorDict remains the data-plane representation for samples, metrics,
and weights; the dataclasses carry low-volume control metadata and references.

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

.. currentmodule:: torchrl.trainers.distributed

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    RayLearnerGroup
