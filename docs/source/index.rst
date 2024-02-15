.. torchrl documentation master file, created by
   sphinx-quickstart on Mon Mar  7 13:23:20 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

TorchRL
=======

.. figure:: /_static/img/logo.png
   :width: 600

TorchRL is an open-source Reinforcement Learning (RL) library for PyTorch.

You can install TorchRL directly from PyPI (see more about installation
instructions in the dedicated section below):

.. code-block::

  $ pip install torchrl

TorchRL provides pytorch and python-first, low and high level abstractions for RL that are intended to be efficient, modular, documented and properly tested.
The code is aimed at supporting research in RL. Most of it is written in python in a highly modular way, such that researchers can easily swap components, transform them or write new ones with little effort.

This repo attempts to align with the existing pytorch ecosystem libraries in that it has a "dataset pillar"
:doc:`(environments) <reference/envs>`,
:ref:`transforms <reference/envs:Transforms>`,
:doc:`models <reference/modules>`,
data utilities (e.g. collectors and containers), etc.
TorchRL aims at having as few dependencies as possible (python standard library, numpy and pytorch).
Common environment libraries (e.g. OpenAI gym) are only optional.

On the low-level end, torchrl comes with a set of highly re-usable functionals
for :doc:`cost functions <reference/objectives>`, :ref:`returns <reference/objectives:Returns>` and data processing.

TorchRL aims at a high modularity and good runtime performance.

To read more about TorchRL philosophy and capabilities beyond this API reference,
check the `TorchRL paper <https://arxiv.org/abs/2306.00577>`__.

Installation
============

TorchRL releases are synced with PyTorch, so make sure you always enjoy the latest
features of the library with the `most recent version of PyTorch <https://pytorch.org/get-started/locally/>`__ (although core features
are guaranteed to be backward compatible with pytorch>=1.13).
Nightly releases can be installed via

.. code-block::

  $ pip install tensordict-nightly
  $ pip install torchrl-nightly

or via a ``git clone`` if you're willing to contribute to the library:

.. code-block::

  $ cd path/to/root
  $ git clone https://github.com/pytorch/tensordict
  $ git clone https://github.com/pytorch/rl
  $ cd tensordict
  $ python setup.py develop
  $ cd ../rl
  $ python setup.py develop

Getting started
===============

A series of quick tutorials to get ramped up with the basic features of the
library. If you're in a hurry, you can start by
:ref:`the last item of the series <gs_first_training>`
and navigate to the previous ones whenever you want to learn more!

.. toctree::
   :maxdepth: 1

   tutorials/getting-started-0
   tutorials/getting-started-1
   tutorials/getting-started-2
   tutorials/getting-started-3
   tutorials/getting-started-4
   tutorials/getting-started-5

Tutorials
=========

Basics
------

.. toctree::
   :maxdepth: 1

   tutorials/coding_ppo
   tutorials/pendulum
   tutorials/torchrl_demo

Intermediate
------------

.. toctree::
   :maxdepth: 1

   tutorials/multiagent_ppo
   tutorials/torchrl_envs
   tutorials/pretrained_models
   tutorials/dqn_with_rnn
   tutorials/rb_tutorial

Advanced
--------

.. toctree::
   :maxdepth: 1

   tutorials/multi_task
   tutorials/coding_ddpg
   tutorials/coding_dqn

References
==========

.. toctree::
   :maxdepth: 3

   reference/index

Knowledge Base
==============

.. toctree::
   :maxdepth: 2

   reference/knowledge_base

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
