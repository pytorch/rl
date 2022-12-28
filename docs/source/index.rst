.. torchrl documentation master file, created by
   sphinx-quickstart on Mon Mar  7 13:23:20 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to the TorchRL Documentation!
=====================================

.. note::
   The TensorDict class has been moved out of TorchRL into a dedicated library. Take a look at `the documentation <./tensordict>`_ or find the source code `on GitHub <https://github.com/pytorch-labs/tensordict>`_.

TorchRL is an open-source Reinforcement Learning (RL) library for PyTorch.

It provides pytorch and python-first, low and high level abstractions for RL that are intended to be efficient, modular, documented and properly tested.
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

.. toctree::
   :maxdepth: 2
   :caption: Tutorials:

   tutorials/torchrl_demo
   tutorials/tensordict_tutorial
   tutorials/tensordict_module
   tutorials/torch_envs
   tutorials/multi_task
   tutorials/coding_ddpg
   tutorials/coding_dqn

.. toctree::
   :maxdepth: 2
   :caption: References:

   reference/index
   reference/knowledge_base

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
