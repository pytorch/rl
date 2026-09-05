.. currentmodule:: torchrl.objectives

Value-Based Methods
===================

Loss modules for value-based RL algorithms.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    DQNLoss
    DistributionalDQNLoss
    IQLLoss
    DiscreteIQLLoss
    CQLLoss
    DiscreteCQLLoss

Parallel Q-Network lambda returns
---------------------------------

:class:`DQNLoss` supports the lambda-return target used by `Parallel Q-Networks
(PQN) <https://arxiv.org/abs/2407.04811>`_ through its existing value-estimator
interface:

.. code-block:: python

    loss = DQNLoss(value_network, action_space=action_spec)
    loss.make_value_estimator(
        ValueEstimators.TDLambda,
        gamma=0.99,
        lmbda=0.95,
    )

The value network's greedy action selection writes ``chosen_action_value``, so
the lambda return bootstraps from :math:`\max_a Q(s_{t+1}, a)`. It does not use
the behavior action at the next step, and no ``("next", "action")`` entry is
required.
