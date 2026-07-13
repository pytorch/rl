.. currentmodule:: torchrl.trainers

Learners and Learner Groups
===========================

Learners own bounded optimization work while a trainer remains responsible for
collection, scheduling, logging, checkpoint cadence, and stopping. A learner
group is the synchronization boundary used by a controller; the controller does
not issue collective-bearing commands to individual ranks.

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
