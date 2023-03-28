.. currentmodule:: torchrl.objectives

torchrl.objectives package
==========================

TorchRL provides a series of losses to use in your training scripts.
The aim is to have losses that are easily reusable/swappable and that have
a simple signature.

The main characteristics of TorchRL losses are:

- they are stateful objects: they contain a copy of the trainable parameters
  such that ``loss_module.parameters()`` gives whatever is needed to train the
  algorithm.
- They follow the ``tensordict`` convention: the :meth:`torch.nn.Module.forward`
  method will receive a tensordict as input that contains all the necessary
  information to return a loss value.
- They output a :class:`tensordict.TensorDict` instance with the loss values
  written under a ``"loss_<smth>`` where ``smth`` is a string describing the
  loss. Additional keys in the tensordict may be useful metrics to log during
  training time.
  .. note::
    The reason we return independent losses is to let the user use a different
    optimizer for different sets of parameters for instance. Summing the losses
    can be simply done via ``sum(loss for key, loss in loss_vals.items() if key.startswith("loss_")``.

Training value functions
------------------------

TorchRL provides a range of **value estimators** such as TD(0), TD(1), TD(:math:`\lambda`)
and GAE.
In a nutshell, a value estimator is a function of data (mostly
rewards and done states) and a state value (ie. the value
returned by a function that is fit to estimate state-values).
To learn more about value estimators, check the introduction to RL from `Sutton
and Barto <https://web.stanford.edu/class/psych209/Readings/SuttonBartoIPRLBook2ndEd.pdf>`_,
in particular the chapters about value iteration and TD learning.
It gives a somewhat biased estimation of the discounted return following a state
or a state-action pair based on data and proxy maps. These estimators are
used in two contexts:

- To train the value network to learn the "true" state value (or state-action
  value) map, one needs a target value to fit it to. The better (less bias,
  less variance) the estimator, the better the value network will be, which in
  turn can speed up the policy training significantly. Typically, the value
  network loss will look like:

    >>> value = value_network(states)
    >>> target_value = value_estimator(rewards, done, value_network(next_state))
    >>> value_net_loss = (value - target_value).pow(2).mean()

- Computing an "advantage" signal for policy-optimization. The advantage is
  the delta between the value estimate (from the estimator, ie from "real" data)
  and the output of the value network (ie the proxy to this value). A positive
  advantage can be seen as a signal that the policy actually performed better
  than expected, thereby signaling that there is room for improvement if that
  trajectory is to be taken as example. Conversely, a negative advantage signifies
  that the policy underperformed compared to what was to be expected.

Thins are not always as easy as in the example above and the formula to compute
the value estimator or the advantage may be slightly more intricate than this.
To help users flexibly use one or another value estimator, we provide a simple
API to change it on-the-fly. Here is an example with DQN, but all modules will
follow a similar structure:

  >>> from torchrl.objectives import DQNLoss, ValueEstimators
  >>> loss_module = DQNLoss(actor)
  >>> kwargs = {"gamma": 0.9, "lmbda": 0.9}
  >>> loss_module.make_value_estimator(ValueEstimators.TDLambda, **kwargs)

The :class:`torchrl.objectives.ValueEstimators` class enumerates the value
estimators to choose from. This makes it easy for the users to rely on
auto-completion to make their choice.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    LossModule

DQN
---

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    DQNLoss
    DistributionalDQNLoss

DDPG
----

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    DDPGLoss

SAC
---

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    SACLoss
    DiscreteSACLoss

REDQ
----

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    REDQLoss

IQL
----

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    IQLLoss

TD3
----

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    TD3Loss

PPO
---

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    PPOLoss
    ClipPPOLoss
    KLPENPPOLoss

A2C
---

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    A2CLoss

Reinforce
---------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ReinforceLoss

Dreamer
-------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    DreamerActorLoss
    DreamerModelLoss
    DreamerValueLoss


Returns
-------
.. currentmodule:: torchrl.objectives.value

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ValueEstimatorBase
    TD0Estimator
    TD1Estimator
    TDLambdaEstimator
    GAE
    functional.td0_return_estimate
    functional.td0_advantage_estimate
    functional.td1_return_estimate
    functional.vec_td1_return_estimate
    functional.td1_advantage_estimate
    functional.vec_td1_advantage_estimate
    functional.td_lambda_return_estimate
    functional.vec_td_lambda_return_estimate
    functional.td_lambda_advantage_estimate
    functional.vec_td_lambda_advantage_estimate
    functional.generalized_advantage_estimate
    functional.vec_generalized_advantage_estimate


Utils
-----
.. currentmodule:: torchrl.objectives

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    distance_loss
    hold_out_net
    hold_out_params
    next_state_value
    SoftUpdate
    HardUpdate
    ValueFunctions
    default_value_kwargs
