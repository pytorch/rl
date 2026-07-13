.. currentmodule:: torchrl.trainers

torchrl.trainers package
========================

.. _ref_trainers:

The trainer package provides utilities to write reusable training scripts. The core idea is to use a
trainer that implements a nested loop, where the outer loop runs the data collection steps and the inner
loop the optimization steps.

Key Features
------------

- **Modular hook system**: Customize training at 18 different points in the loop
- **Checkpointing support**: Pass a :class:`torchrl.checkpoint.Checkpoint` for
  the unified manifest format. Legacy ``torch``, ``torchsnapshot``, and
  ``memmap`` backends remain readable during the migration window.
- **Algorithm trainers**: High-level trainers for PPO, A2C, REINFORCE, SAC,
  offline-to-online SAC, DQN, DDPG, IQL, CQL, and TD3 with Hydra configuration
- **Builder helpers**: Utilities for constructing collectors, losses, and replay buffers

Quick Example
-------------

.. code-block:: python

    from torchrl.trainers import Trainer
    from torchrl.trainers import UpdateWeights, LogScalar
    
    # Create trainer
    trainer = Trainer(
        collector=collector,
        total_frames=1000000,
        loss_module=loss,
        optimizer=optimizer,
    )
    
    # Register hooks
    UpdateWeights(collector, 10).register(trainer)
    LogScalar("reward").register(trainer)
    
    # Train
    trainer.train()

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2

   trainers_basics
   trainers_learners
   trainers_loggers
   trainers_hooks
