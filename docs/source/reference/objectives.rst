.. currentmodule:: torchrl.objectives

torchrl.objectives package
==========================

.. _ref_objectives:

TorchRL provides a comprehensive collection of loss modules for reinforcement learning algorithms.
These losses are designed to be stateful, reusable, and follow the tensordict convention.

Key Features
------------

- **Stateful objects**: Expose trainable parameters via ``loss_module.parameters()``.
  The loss does **not** copy the module you pass in; see :ref:`ref_lossmodule_weight_sharing`.
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

.. _ref_lossmodule_weight_sharing:

Weight sharing
--------------

Passing a module into a :class:`~torchrl.objectives.LossModule` does **not**
copy it. The loss stores the same module and the same parameter tensors; an
optimizer step on ``loss.parameters()`` updates the original network in-place.

That is enough for inference without a collector: after ``optim.step()``,
call the original actor (or ``loss.actor_network``, which is the same object).
No extra copy-back is required.

.. code-block:: python

    import torch
    from torch import nn
    from torchrl.modules import Actor, ValueOperator
    from torchrl.objectives import DDPGLoss

    actor = Actor(nn.Linear(3, 1))
    value = ValueOperator(nn.Linear(4, 1), in_keys=["observation", "action"])
    loss = DDPGLoss(actor, value, delay_actor=False, delay_value=False)

    # Same module, same storage.
    assert actor is loss.actor_network
    p_actor = next(actor.parameters())
    assert p_actor.data_ptr() == next(loss.actor_network.parameters()).data_ptr()

    # An optimizer step on the loss updates the original actor.
    optim = torch.optim.SGD(loss.parameters(), lr=1.0)
    before = p_actor.detach().clone()
    (p_actor ** 2).sum().backward()
    optim.step()
    assert not torch.equal(p_actor.detach(), before)

Collectors are a separate question. Synchronization depends on whether the
training and inference policies share parameter storage:

- A directly passed policy can retain the same storage when no device transfer
  or other copy is needed. No extra sync is required in that case.
- A worker-created policy, different-device copy, or remote policy has distinct
  storage and requires a configured synchronization path.
- Passing ``policy_device`` or ``device`` only creates a copy when the requested
  device differs from the policy's current device.

Calling :meth:`~torchrl.collectors.Collector.update_policy_weights_` anyway is
conservative good practice. See :ref:`ref_collectors_weightsync`.

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2

   objectives_common
   objectives_value
   objectives_policy
   objectives_actorcritic
   objectives_offline
   objectives_multiagent
   objectives_other
   dreamer_v3
