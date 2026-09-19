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

Flow policies
-------------

.. currentmodule:: torchrl.modules

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    FlowMatchingPolicy
    OneStepPolicy

These modules operate on vector observations and normalized actions in
``[-1, 1]``. Their networks receive concatenated inputs, so standard
:class:`~torch.nn.Sequential` networks or :class:`MLP` modules can be used.
Explicit noise can couple teacher and student outputs during distillation::

    from torchrl.modules import Actor, FlowMatchingPolicy, MLP, OneStepPolicy

    flow = FlowMatchingPolicy(MLP(6, 2, num_cells=[64, 64]), action_dim=2)
    student = OneStepPolicy(MLP(5, 2, num_cells=[64, 64]), action_dim=2)
    policy = Actor(student, in_keys=["observation"], out_keys=["action"])

The example uses three observation and two action coordinates. Flow integration
clips only its final output. The student's ``clamp=False`` option exposes raw
outputs for distillation; environment actions should use the default clipping.
    DistributionalQValueModule
