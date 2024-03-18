# -*- coding: utf-8 -*-
"""
Getting started with model optimization
=======================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

.. _gs_optim:

.. note:: To run this tutorial in a notebook, add an installation cell
  at the beginning containing:

    .. code-block::

        !pip install tensordict
        !pip install torchrl

"""

###################################
# In TorchRL, we try to treat optimization as it is custom to do in PyTorch,
# using dedicated loss modules which are designed with the sole purpose of
# optimizing the model. This approach efficiently decouples the execution of
# the policy from its training and allows us to design training loops that are
# similar to what can be found in traditional supervised learning examples.
#
# The typical training loop therefore looks like this:
#
#   >>> for i in range(n_collections):
#   ...     data = get_next_batch(env, policy)
#   ...     for j in range(n_optim):
#   ...         loss = loss_fn(data)
#   ...         loss.backward()
#   ...         optim.step()
#
# In this concise tutorial, you will receive a brief overview of the loss modules. Due to the typically
# straightforward nature of the API for basic usage, this tutorial will be kept brief.
#
# RL objective functions
# ----------------------
#
# In RL, innovation typically involves the exploration of novel methods
# for optimizing a policy (i.e., new sota-implementations), rather than focusing
# on new architectures, as seen in other domains. Within TorchRL,
# these sota-implementations are encapsulated within loss modules. A loss
# module orchestrates the various components of your algorithm and
# yields a set of loss values that can be backpropagated
# through to train the corresponding components.
#
# In this tutorial, we will take a popular
# off-policy algorithm as an example,
# `DDPG <https://arxiv.org/abs/1509.02971>`_.
#
# To build a loss module, the only thing one needs is a set of networks
# defined as :class:`~tensordict.nn.TensorDictModule`s. Most of the time, one
# of these modules will be the policy. Other auxiliary networks such as
# Q-Value networks or critics of some kind may be needed as well. Let's see
# what this looks like in practice: DDPG requires a deterministic
# map from the observation space to the action space as well as a value
# network that predicts the value of a state-action pair. The DDPG loss will
# attempt to find the policy parameters that output actions that maximize the
# value for a given state.
#
# To build the loss, we need both the actor and value networks.
# If they are built according to DDPG's expectations, it is all
# we need to get a trainable loss module:

from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")

from torchrl.modules import Actor, MLP, ValueOperator
from torchrl.objectives import DDPGLoss

n_obs = env.observation_spec["observation"].shape[-1]
n_act = env.action_spec.shape[-1]
actor = Actor(MLP(in_features=n_obs, out_features=n_act, num_cells=[32, 32]))
value_net = ValueOperator(
    MLP(in_features=n_obs + n_act, out_features=1, num_cells=[32, 32]),
    in_keys=["observation", "action"],
)

ddpg_loss = DDPGLoss(actor_network=actor, value_network=value_net)

###################################
# And that is it! Our loss module can now be run with data coming from the
# environment (we omit exploration, storage and other features to focus on
# the loss functionality):
#

rollout = env.rollout(max_steps=100, policy=actor)
loss_vals = ddpg_loss(rollout)
print(loss_vals)

###################################
# LossModule's output
# -------------------
#
# As you can see, the value we received from the loss isn't a single scalar
# but a dictionary containing multiple losses.
#
# The reason is simple: because more than one network may be trained at a time,
# and since some users may wish to separate the optimization of each module
# in distinct steps, TorchRL's objectives will return dictionaries containing
# the various loss components.
#
# This format also allows us to pass metadata along with the loss values. In
# general, we make sure that only the loss values are differentiable such that
# you can simply sum over the values of the dictionary to obtain the total
# loss. If you want to make sure you're fully in control of what is happening,
# you can sum over only the entries which keys start with the ``"loss_"`` prefix:
#

total_loss = 0
for key, val in loss_vals.items():
    if key.startswith("loss_"):
        total_loss += val

###################################
# Training a LossModule
# ---------------------
#
# Given all this, training the modules is not so different from what would be
# done in any other training loop. Because it wraps the modules,
# the easiest way to get the list of trainable parameters is to query
# the :meth:`~torchrl.objectives.LossModule.parameters` method.
#
# We'll need an optimizer (or one optimizer
# per module if that is your choice).
#

from torch.optim import Adam

optim = Adam(ddpg_loss.parameters())
total_loss.backward()

###################################
# The following items will typically be
# found in your training loop:

optim.step()
optim.zero_grad()

###################################
# Further considerations: Target parameters
# -----------------------------------------
#
# Another important aspect to consider is the presence of target parameters
# in off-policy sota-implementations like DDPG. Target parameters typically represent
# a delayed or smoothed version of the parameters over time, and they play
# a crucial role in value estimation during policy training. Utilizing target
# parameters for policy training often proves to be significantly more
# efficient compared to using the current configuration of value network
# parameters. Generally, managing target parameters is handled by the loss
# module, relieving users of direct concern. However, it remains the user's
# responsibility to update these values as necessary based on specific
# requirements. TorchRL offers a couple of updaters, namely
# :class:`~torchrl.objectives.HardUpdate` and
# :class:`~torchrl.objectives.SoftUpdate`,
# which can be easily instantiated without requiring in-depth
# knowledge of the underlying mechanisms of the loss module.
#
from torchrl.objectives import SoftUpdate

updater = SoftUpdate(ddpg_loss, eps=0.99)

###################################
# In your training loop, you will need to update the target parameters at each
# optimization step or each collection step:

updater.step()

###################################
# This is all you need to know about loss modules to get started!
#
# To further explore the topic, have a look at:
#
# - The :ref:`loss module reference page <ref_objectives>`;
# - The :ref:`Coding a DDPG loss tutorial <coding_ddpg>`;
# - Losses in action in :ref:`PPO <coding_ppo>` or :ref:`DQN <coding_dqn>`.
#
