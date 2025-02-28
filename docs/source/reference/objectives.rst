.. currentmodule:: torchrl.objectives

torchrl.objectives package
==========================

.. _ref_objectives:

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
  written under a ``"loss_<smth>"`` where ``smth`` is a string describing the
  loss. Additional keys in the tensordict may be useful metrics to log during
  training time.

.. note::
    The reason we return independent losses is to let the user use a different
    optimizer for different sets of parameters for instance. Summing the losses
    can be simply done via

    >>> loss_val = sum(loss for key, loss in loss_vals.items() if key.startswith("loss_"))

.. note::
    Initializing parameters in losses can be done via a query to :meth:`~torchrl.objectives.LossModule.get_stateful_net`
    which will return a stateful version of the network that can be initialized like any other module.
    If the modification is done in-place, it will be downstreamed to any other module that uses the same parameter
    set (within and outside of the loss): for instance, modifying the ``actor_network`` parameters from the loss
    will also modify the actor in the collector.
    If the parameters are modified out-of-place, :meth:`~torchrl.objectives.LossModule.from_stateful_net` can be
    used to reset the parameters in the loss to the new value.

torch.vmap and randomness
-------------------------

TorchRL loss modules have plenty of calls to :func:`~torch.vmap` to amortize the cost of calling multiple similar models
in a loop, and instead vectorize these operations. `vmap` needs to be told explicitly what to do when random numbers
need to be generated within the call. To do this, a randomness mode need to be set and must be one of `"error"` (default,
errors when dealing with pseudo-random functions), `"same"` (replicates the results across the batch) or `"different"`
(each element of the batch is treated separately).
Relying on the default will typically result in an error such as this one:

  >>> RuntimeError: vmap: called random operation while in randomness error mode.

Since the calls to `vmap` are buried down the loss modules, TorchRL
provides an interface to set that vmap mode from the outside through `loss.vmap_randomness = str_value`, see
:meth:`~torchrl.objectives.LossModule.vmap_randomness` for more information.

``LossModule.vmap_randomness`` defaults to `"error"` if no random module is detected, and to `"different"` in
other cases. By default, only a limited number of modules are listed as random, but the list can be extended
using the :func:`~torchrl.objectives.common.add_random_module` function.

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

The :class:`~torchrl.objectives.ValueEstimators` class enumerates the value
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

CrossQ
------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    CrossQLoss

IQL
---

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    IQLLoss
    DiscreteIQLLoss

CQL
---

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    CQLLoss
    DiscreteCQLLoss

GAIL
----

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    GAILLoss

DT
--

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    DTLoss
    OnlineDTLoss

TD3
---

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    TD3Loss

TD3+BC
------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    TD3BCLoss

PPO
---

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    PPOLoss
    ClipPPOLoss
    KLPENPPOLoss

Using PPO with multi-head action policies
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note:: The main tools to consider when building multi-head policies are: :class:`~tensordict.nn.CompositeDistribution`,
  :class:`~tensordict.nn.ProbabilisticTensorDictModule` and :class:`~tensordict.nn.ProbabilisticTensorDictSequential`.
  When dealing with these, it is recommended to call `tensordict.nn.set_composite_lp_aggregate(False).set()` at the
  beginning of the script to instruct :class:`~tensordict.nn.CompositeDistribution` that log-probabilities should not
  be aggregated but rather written as leaves in the tensordict.

In some cases, we have a single advantage value but more than one action undertaken. Each action has its own
log-probability, and shape. For instance, it can be that the action space is structured as follows:

    >>> action_td = TensorDict(
    ...     agents=TensorDict(
    ...         action0=Tensor(batch, n_agents, f0),
    ...         action1=Tensor(batch, n_agents, f1, f2),
    ...         batch_size=torch.Size((batch, n_agents))
    ...     ),
    ...     batch_size=torch.Size((batch,))
    ... )

where `f0`, `f1` and `f2` are some arbitrary integers.

Note that, in TorchRL, the root tensordict has the shape of the environment (if the environment is batch-locked, otherwise it
has the shape of the number of batched environments being run). If the tensordict is sampled from the buffer, it will
also have the shape of the replay buffer `batch_size`. The `n_agent` dimension, although common to each action, does not
in general appear in the root tensordict's batch-size (although it appears in the sub-tensordict containing the
agent-specific data according to the :ref:`MARL API <MARL-environment-API>`).

There is a legitimate reason why this is the case: the number of agent may condition some but not all the specs of the
environment. For example, some environments have a shared done state among all agents. A more complete tensordict
would in this case look like

    >>> action_td = TensorDict(
    ...     agents=TensorDict(
    ...         action0=Tensor(batch, n_agents, f0),
    ...         action1=Tensor(batch, n_agents, f1, f2),
    ...         observation=Tensor(batch, n_agents, f3),
    ...         batch_size=torch.Size((batch, n_agents))
    ...     ),
    ...     done=Tensor(batch, 1),
    ...     [...] # etc
    ...     batch_size=torch.Size((batch,))
    ... )

Notice that `done` states and `reward` are usually flanked by a rightmost singleton dimension. See this :ref:`part of the doc <reward_done_singleton>`
to learn more about this restriction.

The log-probability of our actions given their respective distributions may look like anything like

    >>> action_td = TensorDict(
    ...     agents=TensorDict(
    ...         action0_log_prob=Tensor(batch, n_agents),
    ...         action1_log_prob=Tensor(batch, n_agents, f1),
    ...         batch_size=torch.Size((batch, n_agents))
    ...     ),
    ...     batch_size=torch.Size((batch,))
    ... )

or

    >>> action_td = TensorDict(
    ...     agents=TensorDict(
    ...         action0_log_prob=Tensor(batch, n_agents),
    ...         action1_log_prob=Tensor(batch, n_agents),
    ...         batch_size=torch.Size((batch, n_agents))
    ...     ),
    ...     batch_size=torch.Size((batch,))
    ... )

ie, the number of dimensions of distributions log-probabilities generally varies from the sample's dimensionality to
anything inferior to that, e.g. if the distribution is multivariate -- :class:`~torch.distributions.Dirichlet` for
instance -- or an :class:`~torch.distributions.Independent` instance.
The dimension of the tensordict, on the contrary, still matches the env's / replay-buffer's batch-size.

During a call to the PPO loss, the loss module will schematically execute the following set of operations:

    >>> def ppo(tensordict):
    ...     prev_log_prob = tensordict.select(*log_prob_keys)
    ...     action = tensordict.select(*action_keys)
    ...     new_log_prob = dist.log_prob(action)
    ...     log_weight = new_log_prob - prev_log_prob
    ...     advantage = tensordict.get("advantage") # computed by GAE earlier
    ...     # attempt to map shape
    ...     log_weight.batch_size = advantage.batch_size[:-1]
    ...     log_weight = sum(log_weight.sum(dim="feature").values(True, True)) # get a single tensor of log_weights
    ...     return minimum(log_weight.exp() * advantage, log_weight.exp().clamp(1-eps, 1+eps) * advantage)

To appreciate what a PPO pipeline looks like with multihead policies, an example can be found in the library's
`example directory <https://github.com/pytorch/rl/blob/main/examples/agents/composite_ppo.py>`__.


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

Multi-agent objectives
----------------------

.. currentmodule:: torchrl.objectives.multiagent

These objectives are specific to multi-agent algorithms.

QMixer
~~~~~~

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    QMixerLoss


Returns
-------

.. _ref_returns:

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
    functional.reward2go


Utils
-----

.. currentmodule:: torchrl.objectives

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    HardUpdate
    SoftUpdate
    ValueEstimators
    default_value_kwargs
    distance_loss
    group_optimizers
    hold_out_net
    hold_out_params
    next_state_value
