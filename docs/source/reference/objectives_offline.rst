.. currentmodule:: torchrl.objectives

Offline RL Methods
==================

Loss modules for offline reinforcement learning.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    CQLLoss
    DiscreteCQLLoss
    IQLLoss
    DiscreteIQLLoss
    TD3BCLoss
    FQLLoss

Flow Q-learning
---------------

:class:`FQLLoss` trains a :class:`~torchrl.modules.FlowMatchingPolicy`, a
:class:`~torchrl.modules.OneStepPolicy`, and an ensemble of TensorDict critics.
Use normalized actions in ``[-1, 1]`` for both replay data and policies.

``loss_flow`` fits the behavior velocity along straight paths from Gaussian
noise to dataset actions. ``loss_actor`` combines raw student-to-teacher
distillation with the negative mean Q value of clipped student actions.
``loss_qvalue`` fits the Bellman target using the current student and target
critics. Each objective updates only its own network parameters.

The default TD0 estimator bootstraps truncations and masks true terminations.
Set the discount with ``loss.make_value_estimator(gamma=0.99)`` and configure
the critic target update with ``SoftUpdate(loss, tau=0.005)``. No target actor
is used. Sum the three loss entries, backpropagate once, and step the optimizer.
The reference implementation updates critic targets from pre-optimizer weights;
call the updater before the optimizer step to match that timing.

Inference uses only the one-step policy, wrapped in :class:`~torchrl.modules.Actor`
for use with TorchRL collectors. Exploration remains stochastic at evaluation,
as in the reference implementation.
