.. currentmodule:: torchrl.trainers

Training Hooks
==============

Hooks for customizing the training loop at various points.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    BatchSubSampler
    ClearCudaCache
    CountFramesLog
    LogScalar
    OptimizerHook
    LogValidationReward
    ReplayBufferTrainer
    RewardNormalizer
    SelectKeys
    UpdateWeights
    TargetNetUpdaterHook
    UTDRHook
