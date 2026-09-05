.. currentmodule:: torchrl.objectives

Common Components
=================

Base classes and common utilities for all loss modules.

:class:`LossModule` does **not** copy the module you pass in. The same
parameters are used in-place, so an optimizer step on the loss updates the
original network. :meth:`~torchrl.collectors.Collector.update_policy_weights_`
is required only when a collector remapped the policy via ``policy_device`` or
``device``. See :ref:`ref_lossmodule_weight_sharing` and
:ref:`ref_collectors_weightsync`.

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    LossModule
    add_random_module

Masked reduction
----------------

Batches of padded sequences carry a per-position validity mask, and positions
marked invalid must not contribute to the loss. Every loss reduces through
:meth:`LossModule._reduce_loss`, which reads that mask from the input according
to :attr:`LossModule.loss_mask_key`:

- ``"auto"`` (the default) looks for each entry of
  :data:`~torchrl.objectives.common.AUTO_LOSS_MASK_KEYS` and ANDs the ones it
  finds, so a batch from :class:`~torchrl.data.SliceSampler` with
  ``pad_output=True`` is handled without any configuration. On data carrying
  none of those entries the reduction is unchanged.
- a :class:`~tensordict.NestedKey` restricts masking to that single entry.
- ``None`` disables masking.

.. code-block:: python

    loss = PPOLoss(actor, critic)
    loss.loss_mask_key = ("my_masks", "valid")  # use this entry only
    loss.loss_mask_key = None                   # reduce over every position

Masked positions are selected out rather than multiplied by zero, so a
non-finite value at a masked position affects neither the loss nor the
gradients.

.. autosummary::
    :toctree: generated/

    AUTO_LOSS_MASK_KEYS

.. _ref_returns:

Value Estimators
----------------

.. currentmodule:: torchrl.objectives.value

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ValueEstimatorBase
    TD0Estimator
    TD1Estimator
    TDLambdaEstimator
    GAE
    VTrace
    MultiAgentGAE

.. currentmodule:: torchrl.objectives

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    ValueEstimators

Optimization utilities
----------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template_noinherit.rst

    KLAdaptiveLR
