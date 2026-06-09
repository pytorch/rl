.. currentmodule:: torchrl.modules

torchrl.modules package
=======================

.. _ref_modules:
.. _tdmodules:

TorchRL offers a comprehensive collection of RL-specific neural network modules built on top of
:class:`tensordict.nn.TensorDictModule`. These modules are designed to work seamlessly with
tensordict data structures, making it easy to build and compose RL models.

Key Features
------------

- **Spec-based construction**: Automatically configure output layers based on action specs
- **Probabilistic modules**: Built-in support for stochastic policies
- **Exploration strategies**: Modular exploration wrappers (ε-greedy, Ornstein-Uhlenbeck, etc.)
- **Value networks**: Q-value, distributional, and dueling architectures
- **Safe modules**: Automatic projection to satisfy action constraints
- **Model-based RL**: World model and dynamics modules

Quick Example
-------------

.. code-block:: python

    from torchrl.modules import ProbabilisticActor, TanhNormal
    from torchrl.envs import GymEnv
    from tensordict.nn import TensorDictModule
    import torch.nn as nn
    
    env = GymEnv("Pendulum-v1")
    
    # Create a probabilistic actor
    actor = ProbabilisticActor(
        module=TensorDictModule(
            nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, 2)),
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        spec=env.action_spec,
    )

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2

   modules_actors
   modules_exploration
   modules_critics
   modules_mcts
   modules_models
   modules_distributions
   modules_inference_server
   modules_utils
