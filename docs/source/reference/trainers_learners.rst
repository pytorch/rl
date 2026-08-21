.. currentmodule:: torchrl.trainers

Learners
========

.. _ref_learners:

A :class:`~torchrl.trainers.Learner` owns a trainable model and exposes a
single, backend-agnostic entry point -- :meth:`~torchrl.trainers.Learner.update`
-- for taking one optimization step on a tensordict batch with a given
:class:`~torchrl.objectives.common.LossModule`. It plays the same role for
training that :class:`~torchrl.collectors.Collector` plays for data collection
and :class:`~torchrl.modules.llm.LLMWrapperBase` plays for generation/scoring:
a fixed contract with interchangeable backends, so algorithm code does not
need to know whether the update runs on one device, under sharded training, or
on a remote training process.

:class:`~torchrl.trainers.LocalLearner` is the single-process reference
implementation. :class:`~torchrl.trainers.FSDP2Learner` shards the same model
with :func:`torch.distributed._composable.fsdp.fully_shard` and reuses
:meth:`~torchrl.trainers.Learner.update` unchanged -- FSDP2's sharding is
transparent to the training step; only construction (the caller wraps the
model before handing it to the learner) and :meth:`~torchrl.trainers.Learner.get_weights`
(which gathers sharded parameters into plain tensors) differ. Either
learner's :meth:`~torchrl.trainers.Learner.get_weights` output is accepted
as-is by :class:`~torchrl.weight_update.WeightSyncScheme`, so a ``Learner``
composes with the existing weight-sync path without changes on either side.

Building the optimizer
----------------------

The **optimizer** decides what is trained, not the ``model`` argument. ``model``
is the weight-sync source and the gradient-sync handle; the parameters that get
clipped and stepped are the ones in ``optimizer.param_groups``. Many TorchRL
losses hold their trainable parameters on the loss module itself as
:class:`~tensordict.nn.TensorDictParams`, and the ones that expand their
networks (:class:`~torchrl.objectives.SACLoss`,
:class:`~torchrl.objectives.REDQLoss`, ...) hold *copies* of the modules you
passed in, so the optimizer must be built over the loss module:

.. code-block:: python

    loss_module = SACLoss(actor, qvalue)
    learner = LocalLearner(actor, Adam(loss_module.parameters()))  # correct
    learner = LocalLearner(actor, Adam(actor.parameters()))        # critics never train

:meth:`~torchrl.trainers.Learner.update` verifies this before its first
optimizer step and raises if any parameter received a gradient that no param
group covers. A ``Learner`` owns exactly one optimizer, so algorithms that
deliberately use several (per-network learning rates, a separate
entropy-temperature optimizer) need one ``Learner`` per optimizer.

Checkpointing
-------------

Use :meth:`~torchrl.trainers.Learner.checkpoint` and
:meth:`~torchrl.trainers.Learner.load_checkpoint`, which cover the model, the
optimizer and the gradient-accumulation counter. ``state_dict`` /
``load_state_dict`` keep their plain :class:`~torch.nn.Module` meaning (a bare
:class:`~torch.optim.Optimizer` is not an ``nn.Module``, so they cannot carry
its state) -- which is what lets a ``Learner`` be nested inside a larger module
without losing state.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Learner
    LearnerCapabilities
    LocalLearner
    FSDP2Learner
