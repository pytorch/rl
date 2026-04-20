.. currentmodule:: torchrl.objectives

torchrl.objectives package
==========================

.. _ref_objectives:

TorchRL provides a comprehensive collection of loss modules for reinforcement learning algorithms.
These losses are designed to be stateful, reusable, and follow the tensordict convention.

Key Features
------------

- **Stateful objects**: Contain trainable parameters accessible via ``loss_module.parameters()``
- **TensorDict convention**: Input and output use TensorDict format
- **Structured output**: Loss values returned with ``"loss_<name>"`` keys
- **Value estimators**: Support for TD(0), TD(λ), GAE, and more
- **Vmap support**: Efficient batched operations with customizable randomness modes

Quick Example
-------------

.. code-block:: python

    from torchrl.objectives import DDPGLoss
    from torchrl.modules import Actor, ValueOperator
    
    # Create loss module
    loss = DDPGLoss(
        actor_network=actor,
        value_network=value,
        gamma=0.99,
    )
    
    # Compute loss
    td = collector.rollout()
    loss_vals = loss(td)
    
    # Get total loss
    total_loss = sum(v for k, v in loss_vals.items() if k.startswith("loss_"))

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2

   objectives_common
   objectives_value
   objectives_policy
   objectives_actorcritic
   objectives_offline
   objectives_other
