.. currentmodule:: torchrl.modules

torchrl.modules package
=======================

TensorDict modules: Actors, exploration, value models and generative models
---------------------------------------------------------------------------

TorchRL offers a series of module wrappers aimed at making it easy to build
RL models from the ground up. These wrappers are exclusively based on
:class:`tensordict.nn.TensorDictModule` and :class:`tensordict.nn.TensorDictSequential`.
They can loosely be split in three categories:
policies (actors), including exploration strategies,
value model and simulation models (in model-based contexts).

The main features are:

- Integration of the specs in your model to ensure that the model output matches
  what your environment expects as input;
- Probabilistic modules that can automatically sample from a chosen distribution
  and/or return the distribution of interest;
- Custom containers for Q-Value learning, model-based agents and others.

SafeModules
~~~~~~~~~~~

TorchRL :class:`~torchrl.modules.tensordict_module.SafeModule` allows you to
check the you model output matches what is to be expected for the environment.
This should be used whenever your model is to be recycled across multiple
environments for instance, and when you want to make sure that the outputs
(e.g. the action) always satisfies the bounds imposed by the environment.
Here is an example of how to use that feature with the
:class:`~torchrl.modules.tensordict_module.Actor` class:

    >>> env = GymEnv("Pendulum-v1")
    >>> action_spec = env.action_spec
    >>> model = nn.LazyLinear(action_spec.shape[-1])
    >>> policy = Actor(model, in_keys=["observation"], spec=action_spec, safe=True)

The ``safe`` flag ensures that the output is always within the bounds of the
``action_spec`` domain: if the network output violates these bounds it will be
projected (in a L1-manner) into the desired domain.

.. currentmodule:: torchrl.modules.tensordict_module

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    Actor
    SafeModule
    SafeSequential

Exploration wrappers
~~~~~~~~~~~~~~~~~~~~

To efficiently explore the environment, TorchRL proposes a series of wrappers
that will override the action sampled by the policy by a noisier version.
Their behaviour is controlled by :func:`~torchrl.envs.utils.exploration_mode`:
if the exploration is set to ``"random"``, the exploration is active. In all
other cases, the action written in the tensordict is simply the network output.

.. currentmodule:: torchrl.modules.tensordict_module

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    AdditiveGaussianWrapper
    EGreedyWrapper
    OrnsteinUhlenbeckProcessWrapper

Probabilistic actors
~~~~~~~~~~~~~~~~~~~~

Some algorithms such as PPO require a probabilistic policy to be implemented.
In TorchRL, these policies take the form of a model, followed by a distribution
constructor.

 .. note::
   The choice of a probabilistic or regular actor class depends on the algorithm
   that is being implemented. On-policy algorithms usually require a probabilistic
   actor, off-policy usually have a deterministic actor with an extra exploration
   strategy. There are, however, many exceptions to this rule.

The model reads an input (typically some observation from the environment)
and outputs the parameters of a distribution, while the distribution constructor
reads these parameters and gets a random sample from the distribution and/or
provides a :class:`torch.distributions.Distribution` object.

    >>> from tensordict.nn import NormalParamExtractor, TensorDictSequential
    >>> from torch.distributions import Normal
    >>> env = GymEnv("Pendulum-v1")
    >>> action_spec = env.action_spec
    >>> model = nn.Sequential(nn.LazyLinear(action_spec.shape[-1] * 2), NormalParamExtractor())
    >>> # build the first module, which maps the observation on the mean and sd of the normal distribution
    >>> model = TensorDictModule(model, in_keys=["observation"], out_keys=["loc", "scale"])
    >>> # build the distribution constructor
    >>> prob_module = SafeProbabilisticModule(
    ...     in_keys=["loc", "scale"],
    ...     out_keys=["action"],
    ...     distribution_class=Normal,
    ...     return_log_prob=True,
    ...     spec=action_spec,
    ... )
    >>> policy = TensorDictSequential(model, prob_module)
    >>> # execute a rollout
    >>> env.rollout(3, policy)

To facilitate the construction of probabilistic policies, we provide a dedicated
:class:`~torchrl.modules.tensordict_module.ProbabilisticActor`:

    >>> policy = ProbabilisticActor(
    ...     model,
    ...     in_keys=["loc", "scale"],
    ...     out_keys=["action"],
    ...     distribution_class=Normal,
    ...     return_log_prob=True,
    ...     spec=action_spec,
    ... )

which alleviates the need to specify a constructor and putting it with the
module in a sequence.

Outputs of this policy will contain a ``"loc"`` and ``"scale"`` entries, an
``"action"`` sampled according to the normal distribution and the log-probability
of this action.

.. currentmodule:: torchrl.modules.tensordict_module

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ProbabilisticActor
    SafeProbabilisticModule
    SafeProbabilisticTensorDictSequential

Q-Value actors
~~~~~~~~~~~~~~

Q-Value actors are a special type of policy that does not directly predict an action
from an observation, but picks the action that maximised the value (or *quality*)
of a (s,a) -> v map. This map can be a table or a function.
For discrete action spaces with continuous (or near-continuous such as pixels)
states, it is customary to use a non-linear model such as a neural network for
the map.
The semantic of the Q-Value network is hopefully quite simple: we just need to
feed a tensor-to-tensor map that given a certain state (the input tensor),
outputs a list of action values to choose from. The wrapper will write the
resulting action in the input tensordict along with the list of action values.

    >>> import torch
    >>> from tensordict import TensorDict
    >>> from tensordict.nn.functional_modules import make_functional
    >>> from torch import nn
    >>> from torchrl.data import OneHotDiscreteTensorSpec
    >>> from torchrl.modules.tensordict_module.actors import QValueActor
    >>> td = TensorDict({'observation': torch.randn(5, 3)}, [5])
    >>> # we have 4 actions to choose from
    >>> action_spec = OneHotDiscreteTensorSpec(4)
    >>> # the model reads a state of dimension 3 and outputs 4 values, one for each action available
    >>> module = nn.Linear(3, 4)
    >>> qvalue_actor = QValueActor(module=module, spec=action_spec)
    >>> qvalue_actor(td)
    >>> print(td)
    TensorDict(
        fields={
            action: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.int64, is_shared=False),
            action_value: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False),
            chosen_action_value: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
            observation: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
        batch_size=torch.Size([5]),
        device=None,
        is_shared=False)

Distributional Q-learning is slightly different: in this case, the value network
does not output a scalar value for each state-action value.
Instead, the value space is divided in a an arbitrary number of "bins". The
value network outputs a probability that the state-action value belongs to one bin
or another.
Hence, for a state space of dimension M, an action space of dimension N and a number of bins B,
the value network encodes a :math:`\mathbb{R}^{M} \rightarrow \mathbb{R}^{N \times B}`
map. The following example shows how this works in TorchRL with the :class:`~torchrl.modules.tensordict_module.DistributionalQValueActor`
class:

        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torch import nn
        >>> from torchrl.data import OneHotDiscreteTensorSpec
        >>> from torchrl.modules import DistributionalQValueActor, MLP
        >>> td = TensorDict({'observation': torch.randn(5, 4)}, [5])
        >>> nbins = 3
        >>> # our model reads the observation and outputs a stack of 4 logits (one for each action) of size nbins=3
        >>> module = MLP(out_features=(nbins, 4), depth=2)
        >>> action_spec = OneHotDiscreteTensorSpec(4)
        >>> qvalue_actor = DistributionalQValueActor(module=module, spec=action_spec, support=torch.arange(nbins))
        >>> td = qvalue_actor(td)
        >>> print(td)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.int64, is_shared=False),
                action_value: Tensor(shape=torch.Size([5, 3, 4]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([5, 4]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([5]),
            device=None,
            is_shared=False)


.. currentmodule:: torchrl.modules.tensordict_module

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    QValueActor
    QValueModule
    DistributionalQValueActor
    DistributionalQValueModule

Value operators and joined models
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. currentmodule:: torchrl.modules.tensordict_module

TorchRL provides a series of value operators that wrap value networks to
soften the interface with the rest of the library.
The basic building block is :class:`torchrl.modules.tensordict_module.ValueOperator`:
given an input state (and possibly action), it will automatically write a ``"state_value"``
(or ``"state_action_value"``) in the tensordict, depending on what the input is.
As such, this class accounts for both value and quality networks.
Three classes are also proposed to group together a policy and a value network.
The :class:`~.ActorCriticOperator` is an joined actor-quality network with shared parameters:
it reads an observation, pass it through a
common backbone, writes a hidden state, feeds this hidden state to the policy,
then takes the hidden state and the action and provides the quality of the state-action
pair.
The :class:`~.ActorValueOperator` is a joined actor-value network with shared parameters:
it reads an observation, pass it through a
common backbone, writes a hidden state, feeds this hidden state to the policy
and value modules to output an action and a state value.
Finally, the :class:`~.ActorCriticWrapper` is a joined actor and value network
without shared parameters. It is mainly intended as a replacement for
:class:`~.ActorValueOperator` when a script needs to account for both options.

    >>> actor = make_actor()
    >>> value = make_value()
    >>> if shared_params:
    ...     common = make_common()
    ...     model = ActorValueOperator(common, actor, value)
    ... else:
    ...     model = ActorValueOperator(actor, value)
    >>> policy = model.get_policy_operator()  # will work in both cases

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ActorCriticOperator
    ActorCriticWrapper
    ActorValueOperator
    ValueOperator


Other modules
~~~~~~~~~~~~~

.. currentmodule:: torchrl.modules.tensordict_module

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    WorldModelWrapper

Hooks
-----
.. currentmodule:: torchrl.modules

The Q-value hooks are used by the :class:`~.QValueActor` and :class:`~.DistributionalQValueActor`
modules and those should be preferred in general as they are easier to create
and use.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    QValueHook
    DistributionalQValueHook

Models
------
.. currentmodule:: torchrl.modules

TorchRL provides a series of useful "regular" (ie non-tensordict) nn.Module
classes for RL usage.

Regular modules
~~~~~~~~~~~~~~~

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    MLP
    ConvNet
    LSTMNet

Algorithm-specific modules
~~~~~~~~~~~~~~~~~~~~~~~~~~

These networks implement sub-networks that have shown to be useful for specific
algorithms, such as DQN, DDPG or Dreamer.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    DuelingCnnDQNet
    DistributionalDQNnet
    DdpgCnnActor
    DdpgCnnQNet
    DdpgMlpActor
    DdpgMlpQNet
    DreamerActor
    ObsEncoder
    ObsDecoder
    RSSMPrior
    RSSMPosterior


Exploration
-----------
.. currentmodule:: torchrl.modules

Noisy linear layers are a popular way of exploring the environment without
altering the actions, but by integrating the stochasticity in the weight
configuration.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    NoisyLinear
    NoisyLazyLinear
    reset_noise


Planners
--------
.. currentmodule:: torchrl.modules

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    CEMPlanner
    MPCPlannerBase
    MPPIPlanner


Distributions
-------------
.. currentmodule:: torchrl.modules

Some distributions are typically used in RL scripts.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    Delta
    IndependentNormal
    NormalParamWrapper
    TanhNormal
    TruncatedNormal
    TanhDelta
    OneHotCategorical
    MaskedCategorical

Utils
-----

.. currentmodule:: torchrl.modules.utils

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    mappings
    inv_softplus
    biased_softplus

.. currentmodule:: torchrl.modules.models.utils

    SqueezeLayer
    Squeeze2dLayer
