.. currentmodule:: torchrl.collectors.llm

LLM Collectors
==============

Specialized collector classes for LLM use cases.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    LLMCollector
    RayLLMCollector
    vLLMUpdater
    vLLMUpdaterV2

Weight Synchronization Schemes
------------------------------

.. currentmodule:: torchrl.weight_update.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    VLLMWeightSyncScheme
    VLLMWeightSender
    VLLMWeightReceiver
    VLLMCollectiveTransport
    VLLMDoubleBufferSyncScheme
    VLLMDoubleBufferWeightSender
    VLLMDoubleBufferWeightReceiver
    VLLMDoubleBufferTransport
    get_model_metadata
