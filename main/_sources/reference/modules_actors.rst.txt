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

These actors read ``observation`` and optional ``noise`` keys and write ``action``.
They sample Gaussian noise when it is absent and clip actions to ``[low, high]``
(default ``[-1, 1]``). Bounds broadcast over actions; keys can be customized,
including nested keys. Networks receive concatenated inputs::

    import torch
    from tensordict import TensorDict
    from torchrl.modules import FlowMatchingPolicy, MLP, OneStepPolicy

    flow = FlowMatchingPolicy(MLP(6, 2, num_cells=[64, 64]), action_dim=2)
    student = OneStepPolicy(MLP(5, 2, num_cells=[64, 64]), action_dim=2)
    td = TensorDict(
        observation=torch.randn(4, 3), noise=torch.randn(4, 2), batch_size=[4]
    )
    teacher_action = flow(td.clone())["action"]
    student_action = student(td)["action"]

The example uses three observation and two action coordinates. Flow integration
clips only its final output. Both policies can be passed directly to collectors
and ``env.rollout``. Their tensor-only models are available as ``policy.module``;
``student.module(observation, noise, clamp=False)`` exposes raw outputs for
distillation. Environment actions should use the default clipping.
