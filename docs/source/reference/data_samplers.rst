.. currentmodule:: torchrl.data.replay_buffers

Sampling Strategies
===================

Samplers control how data is retrieved from the replay buffer storage.

.. seealso::

    The trajectory-aware samplers (:class:`SliceSampler` and its variants)
    recover episode boundaries from the stored data at sampling time. The
    conventions they rely on — trajectory ids, end flags, circular-storage
    wraparound and the write cursor — are documented in
    :ref:`Trajectory boundaries <ref_traj_boundaries>`.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    PrioritizedSampler
    PrioritizedSliceSampler
    GeometricTrajectoryWindowSampler
    PromptGroupSampler
    ConsumingSampler
    RandomSampler
    Sampler
    SamplerEnsemble
    SamplerWithoutReplacement
    SliceSampler
    SliceSamplerWithoutReplacement
    StalenessAwareSampler

Writers
-------

Writers control how data is written to the storage. Writers that reuse
storage slots can stamp each slot with a reuse counter so consumers holding
an index can detect that it was overwritten -- see
:ref:`Detecting overwritten slots <ref_buffers_generations>`.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    RoundRobinWriter
    TensorDictMaxValueWriter
    TensorDictRoundRobinWriter
    Writer
    WriterEnsemble
