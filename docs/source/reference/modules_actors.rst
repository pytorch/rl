.. currentmodule:: torchrl.modules

Actor Modules
=============

Actor modules represent policies in RL. They map observations to actions, either deterministically
or stochastically.

TensorDictModules and SafeModules
---------------------------------

.. currentmodule:: torchrl.modules.tensordict_module

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    Actor
    MultiStepActorWrapper
    SafeModule
    SafeSequential
    TanhModule

Probabilistic actors
--------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ProbabilisticActor
    SafeProbabilisticModule
    SafeProbabilisticTensorDictSequential

Q-Value actors
--------------

.. currentmodule:: torchrl.modules

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    QValueActor
    DistributionalQValueActor
    QValueModule
    DistributionalQValueModule
