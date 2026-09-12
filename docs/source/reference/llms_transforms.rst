:orphan:

.. currentmodule:: torchrl.envs.llm.transforms

LLM Transforms
==============

Transforms for LLM environments, including tools and utilities.

:attr:`Tokenizer.out_device` describes the destination of token tensors and
attention masks. It follows the current parent environment device; ``None``
leaves outputs on the tokenizer's chosen device. ``Tokenizer.device`` is a
deprecated alias and will be removed in TorchRL v0.17.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    AddThinkingPrompt
    BrowserTransform
    DataLoadingPrimer
    ExecuteToolsInOrder
    IncrementalTokenizer
    JSONCallParser
    KLComputation
    KLRewardTransform
    MCPToolTransform
    PolicyVersion
    PythonExecutorService
    PythonInterpreter
    RayDataLoadingPrimer
    RetrieveKL
    RetrieveLogProb
    SimpleToolTransform
    TemplateTransform
    Tokenizer
    ToolCall
    ToolRegistry
    ToolService
    XMLBlockParser
    as_nested_tensor
    as_padded_tensor
