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
    DiffusionActor
    MultiStepActorWrapper
    SafeModule
    SafeSequential
    TanhModule
    RandomPolicy

Probabilistic actors
--------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ProbabilisticActor
    DreamerV3DiscreteActor
    DreamerV3SeededPolicy
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

Flow policies
-------------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    FlowMatchingPolicy
    OneStepPolicy

These modules accept vector observations and clip actions to ``[low, high]``
(default ``[-1, 1]``). Bounds can be scalars or tensors broadcast over actions.
Networks receive concatenated inputs::

    from torchrl.modules import Actor, FlowMatchingPolicy, MLP, OneStepPolicy

    flow = FlowMatchingPolicy(MLP(6, 2, num_cells=[64, 64]), action_dim=2)
    student = OneStepPolicy(MLP(5, 2, num_cells=[64, 64]), action_dim=2)
    policy = Actor(student)

The example uses three observation and two action coordinates. Flow integration
clips only its final output. The student's ``clamp=False`` option exposes raw
outputs for distillation; environment actions should use the default clipping.
