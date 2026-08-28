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

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    Learner
    LearnerCapabilities
    LocalLearner
    FSDP2Learner
