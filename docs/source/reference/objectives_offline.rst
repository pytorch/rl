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
    TransitiveQLoss

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
is used. Sum the three loss entries, backpropagate once, step the optimizer,
then call the updater, following the standard TorchRL training sequence.

Observation inputs follow the policies' and critics' ``in_keys``, including
nested keys. Wrap either policy in a :class:`~tensordict.nn.TensorDictSequential`
to prepend an encoder that reads multiple observation keys. Each policy's
encoder receives gradients only from its own objective.

Inference passes the one-step policy directly to TorchRL collectors.
Exploration remains stochastic at evaluation, as in the reference implementation.

Transitive RL
-------------

:class:`TransitiveQLoss` implements Transitive RL (Park et al., ICLR 2026,
https://arxiv.org/abs/2510.22512) for offline goal-conditioned RL. The critic
:math:`Q(s, a, g)` lives in :math:`(0, 1]` and is read as
:math:`\gamma^{d(s, g)}`; instead of a one-step backup, it is regressed toward
the product of two target-critic values through a subgoal the dataset
trajectory visited between the state and the goal.

Batches come from a replay buffer with the
:class:`~torchrl.data.replay_buffers.SubgoalTriplet` sample unit, which expands
each anchor into its state, subgoal, goal and policy-goal records and reports
the step offset of each one. The flat ``[4 * B]`` sample is passed to the loss
as is. Sum ``loss_qvalue``, ``loss_actor`` and, with an oracle critic,
``loss_oracle_qvalue``, then update the target critic with
:class:`~torchrl.objectives.SoftUpdate`.

.. code-block:: python

    from torchrl.data import LazyTensorStorage, TensorDictReplayBuffer
    from torchrl.data.replay_buffers import SubgoalTriplet
    from torchrl.objectives import SoftUpdate, TransitiveQLoss

    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(dataset.numel()),
        batch_size=1024,
        sample_unit=SubgoalTriplet(goal_discount=0.99),
    )
    rb.extend(dataset)
    loss = TransitiveQLoss(actor, critics, discount=0.99, expectile=0.7)
    updater = SoftUpdate(loss, tau=0.005)
    losses = loss(rb.sample())

``sota-implementations/transitive_rl/`` trains it on OGBench datasets.
