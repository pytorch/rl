# -*- coding: utf-8 -*-
"""
Getting started with model optimization
=======================================

**Author**: `Vincent Moens <https://github.com/vmoens>`_

.. _gs_optim:

"""

###################################
# In TorchRL, we try to tread optimization as it is custom to do in PyTorch,
# using dedicated loss modules which are designed with the sole purpose of
# optimizing the model. This approach efficiently decouples the execution of
# the policy from its training and allows us to design training loops that are
# similar to what can be found in traditional supervised learning examples.
#
# In this tutorial, you will be given a quick look at the loss modules.
# Because the API is usually quite straightforward for a naive usage, this
# tutorial will be very short.
#
# RL objective functions
# ----------------------
#
# In RL, innovation usually goes through new methods of optimizing a policy
# (i.e., new algorithms) rather than new architectures like it can be the case
# in other domains. In TorchRL, these algorithms are encapsulated in loss
# modules. A loss modules orchestrates the various components of your algorithm
# and returns a set of loss values that can be backpropagated through to
# train the corresponding components.
#
# In this tutorial, we will take a popular
# off-policy algorithm as an example,
# `DDPG <https://arxiv.org/abs/1509.02971>`_.
#
# To build a loss module, the only thing one needs is a set of networks
# defined as :class:`~tensordict.nn.TensorDictModules`. Most of the time, one
# of these modules will be the policy. Other auxiliary networks such as
# Q-Value networks or critics of some kind may be needed as well. Let's see
# what this looks like in practice: DDPG requires a deterministic
# map from the observation space to the action space as well as a value
# network that predicts the value of a state-action pair. The DDPG loss will
# attempt to find the policy parameters that output actions that maximize the
# value for a given state.

from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")

from torchrl.objectives import DDPGLoss
from torchrl.modules import Actor, MLP, ValueOperator

n_obs = env.observation_spec['observation'].shape[-1]
n_act = env.action_spec.shape[-1]
actor = Actor(MLP(in_features=n_obs, out_features=n_act, num_cells=[32, 32]))
value_net = ValueOperator(MLP(in_features=n_obs+n_act, out_features=1, num_cells=[32, 32]), in_keys=["observation", "action"])

loss = DDPGLoss(actor_network=actor, value_network=value_net)

###################################
# And that is it! Our loss module can now be run with data coming from the
# environment (we omit exploration, storage and other features to focus on
# the loss functionality):
#

rollout = env.rollout(max_steps=100, policy=actor)
loss_vals = loss(rollout)
print(loss_vals)

###################################
# As you can see, the value we received from the loss isn't a single scalar
# but a dictionary containing multiple losses.
#
# The reason is simple: because more than one network may be trained at a time,
# and since some users may wish to separate the optimization of each module
# in distinct steps, TorchRL's objectives will return dictionaries containing
# the various loss components.
#