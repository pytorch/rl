.. currentmodule:: torchrl.trainers

Learners
========

.. _ref_learners:

A :class:`~torchrl.trainers.Learner` is an execution boundary around the same
:class:`~torchrl.trainers.OptimizationStepper` contract used by
:class:`~torchrl.trainers.Trainer`:

* the learner decides **where** an update runs and how policy weights are
  materialized;
* the optimization stepper decides **how** the update runs, including loss
  reduction, backward, accumulation, clipping, mixed precision, and optimizer
  stepping.

This keeps one optimization API across ordinary and LLM trainers. Construct
the loss and stepper once, pass both to the learner, then call
``learner.update(batch)``:

.. code-block:: python

    loss_module = MyLoss(policy)
    stepper = MixedPrecisionOptimizationStepper(
        Adam(loss_module.parameters()),
        mixed_precision=True,
        gradient_accumulation_steps=4,
    )
    learner = LocalLearner(policy, loss_module, stepper)
    metrics = learner.update(batch)

:class:`~torchrl.trainers.LocalLearner` runs in the current process.
:class:`~torchrl.trainers.FSDP2Learner` accepts a model already wrapped with
:func:`torch.distributed._composable.fsdp.fully_shard`; it uses the same
stepper and only changes gradient synchronization, weight gathering, and
checkpoint translation.

Weight publication
------------------

:meth:`~torchrl.trainers.Learner.get_weights` returns a detached TensorDict
snapshot accepted by :class:`~torchrl.weight_update.WeightSyncScheme`.
Only the published model is included; loss-only parameters such as entropy
temperature remain training state and are covered by checkpoints.

Checkpointing
-------------

:meth:`~torchrl.trainers.Learner.checkpoint` composes model, loss-module, and
stepper state. The stepper owns optimizer and accumulation state, so learner
checkpoints obey the same rule as trainer checkpoints: saving partway through
an accumulation window is rejected because partial gradients are not stored.

``state_dict`` keeps its ordinary :class:`torch.nn.Module` meaning, allowing a
learner to be nested safely inside another module.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Learner
    LearnerCapabilities
    LocalLearner
    FSDP2Learner
