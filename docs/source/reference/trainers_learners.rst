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

.. currentmodule:: torchrl.trainers.distributed

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    RayLearnerGroup
