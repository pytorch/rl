# -*- coding: utf-8 -*-
"""
Getting started with TorchRL
============================

Environments, TED and transforms
--------------------------------

**Author**: `Vincent Moens <https://github.com/vmoens>`_

"""

################################
# The typical RL training loop consist of a model (a policy) that is trained to solve
# a task in an environment. In many cases, this environment consists of a simulator
# which takes actions as input and outputs an observation as well as some metadata.
#
# In this document, we'll learn about TorchRL's environment API: how to create an
# environment, how to interact with it, and what data format is used.
#
# Creating an environment
# ~~~~~~~~~~~~~~~~~~~~~~~
#
# Per se, TorchRL does not provide environments but wrappers for other libraries
# that encode the simulators. You can think of :mod:`~torchrl.envs` as a provider for
# a generic environment API as well as a common hub for simulation backends such
# as `gym <https://arxiv.org/abs/1606.01540>`_, `Brax <https://arxiv.org/abs/2106.13281>`_
# or `DeepMind Control Suite <https://arxiv.org/abs/1801.00690>`_.
#
# Creating your environment is usually as easy as the underlying backend API permits.
# Here's an example with gym:

from torchrl.envs import GymEnv

env = GymEnv("Pendulum-v1")

################################
#
# Running an environment
# ~~~~~~~~~~~~~~~~~~~~~~
#
# Environments have two important methods: :meth:`~torchrl.envs.EnvBase.reset`
# which initiates an episode and :meth:`~torchrl.envs.EnvBase.step` which enacts an
# action chosen by the actor.
# In TorchRL, environments methods read and write :class:`~tensordict.TensorDict` instances.
# In short, :class:`~tensordict.TensorDict` is a generic key-based data carrier
# for tensors. The advantage of using :class:`~tensordict.TensorDict` instead of
# plain tensors is that it allows us to work with simple and complex data structures
# interchangeably: since the our function signatures are very generic, it removes
# the difficulty of accounting for different data formats. In other words: after
# this short tutorial, you will be able to act on simple as well as very complex
# environments!
#
# Let's put the environment into action:

reset = env.reset()
print(reset)

################################
# Let's take a random action in the action space. First, sample the action:
reset_with_action = env.rand_action(reset)
print(reset_with_action)

################################
# This tensordict has the same structure as the one from :meth:`~torchrl.envs.EnvBase`
# with an additional ``"action"`` entry.
#
# Next, let's pass this action in the environment:

stepped_data = env.step(reset_with_action)
print(stepped_data)

################################
# This new tensordict is identical to the previous one except for the fact that it has
# a ``"next"`` entry containing the observation, reward and done state resulting from
# our action.
#
# This format is called TED, for :ref:`TorchRL Episode Data format <TED-format>`_ amd is
# ubiquituous in the library.
#
# The last bit of information you need to run a rollout in the environment is
# how to bring that ``"next"`` entry at the root to perform the next step.
# TorchRL provides a dedicated :func:`~torchrl.envs.utils.step_mdp` function
# that does just that, filtering out the information you won't need and delivering
# a data structure corresponding to your observation after a step in the Markov
# Decision Process, or MDP.

from torchrl.envs import step_mdp

data = step_mdp(stepped_data)
print(data)

################################
# Writing down those three steps can be a bit tedious and repetitive. Fortunately,
# TorchRL provides a nice :meth:`~torchrl.envs.EnvBase.rollout` function that
# allows you to run them in a closed loop at will.
#

rollout = env.rollout(max_steps=10)
print(rollout)

################################
# This data looks pretty much like the ``stepped_data`` above with the exception
# of its batch-size which now equates the number of steps we provided through
# the ``max_steps`` argument. The magic of tensordict doesn't end there: if you're
# interested in a single transition of this environment, you can index the tensordict
# like you would index a tensor:

transition = rollout[3]
print(transition)

################################
# As such, the rollout may seem rather useless (it just runs random actions if no
# policy is provided) but it is useful to check what is to be expected from an environment
# at a glance.
#
# If you need to be convinced of how generic TorchRL's API is, think about the
# fact that the rollout method works across **all** use cases, whether you're working
# with a single environment like this one, several copies on multiple processes,
# a multi-agent environment or a stateless version of it! Let's compare the output
# of the rollout method with the
#
# Transforming an environment
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
# Most of the time, you will want to change the output of the environment to
# make it more suited for your needs. For instance, you may want to keep track
# of how many steps have been run since the last reset, resize images or stack
# consecutive observations together. We'll just see a simple transform here, the
# :class:`~torchrl.envs.transforms.StepCounter` transform, but the full list can be
# accessed :ref:`here <transforms>`_. The transform is combined with the environment
# through a :class:`~torchrl.envs.TransformedEnv`:

from torchrl.envs import StepCounter, TransformedEnv

transformed_env = TransformedEnv(env, StepCounter(max_steps=10))
rollout = transformed_env.rollout(max_steps=100)
print(rollout)

################################
# As you can see, our environment now has one more entry, ``"step_count"`` that
# tracks the number of steps since the last reset. Since we passed the optional
# argument ``max_steps=10`` to the transform constructor, we also truncated the
# trajectory after 10 steps (not completing a full rollout of 100 steps like
# we asked with the ``rollout`` call). We can see that the trajectory was truncated
# by looking at the truncated entry:

print(rollout["next", "truncated"])

################################
#
# We've now explored the basic functionality of TorchRL's environment.
#
# Next steps
# ~~~~~~~~~~
#
# To explore further what TorchRL's environments can do, go and check:
# - The :meth:`~torchrl.envs.EnvBase.step_and_maybe_reset` method that packs together
#   :meth:`~torchrl.envs.EnvBase.step`, :func:`~torchrl.envs.step_mdp` and :meth:`~torchrl.envs.EnvBase.reset`.
# - The batched environments, in particular :class:`~torchrl.envs.ParallelEnv` which
#   allows you to run multiple copies of one same (or different!) environments on multiple processes.
# - Design your own environment with the :ref:`Pendulum tutorial <>`_ and learn
#   about specs and stateless environments.
# - See the more in-depth tutorial about environments :ref:`here <env_tuto>`
#
