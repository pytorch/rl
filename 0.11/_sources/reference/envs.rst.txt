.. currentmodule:: torchrl.envs

torchrl.envs package
====================

.. _ref_envs:

TorchRL offers a comprehensive API to handle environments of different backends, making it easy to swap
environments in an experiment with minimal effort. The library provides wrappers for popular RL frameworks
including Gym, DMControl, Brax, Jumanji, and many others.

The :class:`~torchrl.envs.EnvBase` class serves as the foundation, providing a unified interface that uses
:class:`tensordict.TensorDict` for data organization. This design allows the framework to be generic and
handle an arbitrary number of inputs and outputs, as well as nested or batched data structures.

Key Features
------------

- **Unified API**: Consistent interface across different environment backends
- **Vectorization**: Built-in support for parallel and batched environments
- **Transforms**: Powerful transform system for preprocessing observations and actions
- **Multi-agent**: Native support for multi-agent RL with no additional infrastructure
- **Flexible backends**: Easy integration with Gym, DMControl, Brax, and custom environments

Quick Example
-------------

.. code-block:: python

    from torchrl.envs import GymEnv, ParallelEnv, TransformedEnv
    from torchrl.envs.transforms import RewardSum, StepCounter
    
    # Create a single environment
    env = GymEnv("Pendulum-v1")
    
    # Add transforms
    env = TransformedEnv(env, RewardSum())
    
    # Create parallel environments
    def make_env():
        return TransformedEnv(
            GymEnv("Pendulum-v1"),
            StepCounter(max_steps=200)
        )
    
    parallel_env = ParallelEnv(4, make_env)
    
    # Run a rollout
    rollout = parallel_env.rollout(100)

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2

   envs_api
   envs_vectorized
   envs_transforms
   envs_multiagent
   envs_libraries
   envs_recorders
