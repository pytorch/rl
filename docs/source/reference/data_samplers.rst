.. currentmodule:: torchrl.data.replay_buffers

Sampling Strategies
===================

Samplers control how data is retrieved from the replay buffer storage.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    PrioritizedSampler
    PrioritizedSliceSampler
    RandomSampler
    Sampler
    SamplerWithoutReplacement
    SliceSampler
    SliceSamplerWithoutReplacement

Writers
-------

Writers control how data is written to the storage.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    RoundRobinWriter
    TensorDictMaxValueWriter
    TensorDictRoundRobinWriter
    Writer
