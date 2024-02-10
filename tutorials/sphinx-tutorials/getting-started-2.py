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
# Given all this, training the modules is not so different than what would be
# done in any other training loop. We'll need an optimizer (or one optimizer
# per module if that is your choice). The following items will typically be
# found in your training loop:

from torch.optim import Adam

optim = Adam(ddpg_loss.parameters())
total_loss.backward()
optim.step()
optim.zero_grad()

###################################
# Further considerations: Target parameters
# -----------------------------------------
#
# Another important consideration is that off-policy algorithms such as DDPG
# typically have target parameters associated with them. Target parameters are
# usually a version of the parameters that lags in time (or a smoothed
# average of that) and they are used for value estimation when training the
# policy. Training the policy using target parameters is usually much more
# efficient than using the configuraton of the value network parameters at the
# same time. You usually don't need to care too much about target parameters
# as your loss module will create them for you, **but** it is your
# responsibility to update these values when needed depending on your needs.
# TorchRL provides a couple of updaters, namely
# :class:`~torchrl.objectives.HardUpdate` and
# :class:`~torchrl.objectives.SoftUpdate`. Instantiating them is very easy and
# doesn't require any knowledge about the inner machinery of the loss module.
#
from torchrl.objectives import SoftUpdate

updater = SoftUpdate(ddpg_loss, eps=0.99)

###################################
# In your training loop, you will need to update the taget parameters at each
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
