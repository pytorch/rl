.. currentmodule:: torchrl.trainers

torchrl.trainers package
========================

.. _ref_trainers:

The trainer package provides utilities to write re-usable training scripts. The core idea is to use a
trainer that implements a nested loop, where the outer loop runs the data collection steps and the inner
loop the optimization steps.

Key Features
------------

- **Modular hook system**: Customize training at 10 different points in the loop
- **Checkpointing support**: Save and restore training state with torch or torchsnapshot
- **Algorithm trainers**: High-level trainers for PPO, SAC with Hydra configuration
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
   trainers_loggers
   trainers_hooks
