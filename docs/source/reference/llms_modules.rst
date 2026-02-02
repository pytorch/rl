:orphan:

.. currentmodule:: torchrl.modules.llm

LLM Wrappers
============

The LLM wrapper API provides unified interfaces for different LLM backends, ensuring consistent 
input/output formats across training and inference pipelines.

Wrappers
--------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    LLMWrapperBase
    TransformersWrapper
    vLLMWrapper
    SGLangWrapper
    RemoteTransformersWrapper
    AsyncVLLM
    AsyncSGLang

Data Structure Classes
----------------------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ChatHistory
    Text
    LogProbs
    Masks
    Tokens

Utilities
---------

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    make_async_vllm_engine
    stateless_init_process_group_async
    make_vllm_worker
    stateless_init_process_group
