.. currentmodule:: torchrl

LLM interface
=============

.. _ref_llms:

TorchRL offers a set of tools for LLM post-training, as well as some examples for training or setup.

Collectors
----------

TorchRL offers a specialized collector class (:class:`~torchrl.collectors.llm.LLMCollector`) that is tailored for LLM
use cases. We also provide dedicated updaters for some inference engines.

.. currentmodule:: torchrl.collectors.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    vLLMUpdater
    LLMCollector


Data structures
---------------

To handle text-based data structures (such as conversations etc.), we offer a few data structures dedicated to carrying
data for LLM post-training.

.. currentmodule:: torchrl.data.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    History
    LLMData

Environments
------------

When fine-tuning an LLM using TorchRL, the environment is a crucial component of the inference pipeline, alongside the
policy and collector. Environments manage operations that are not handled by the LLM itself, such as interacting with
tools, loading prompts from datasets, computing rewards (when necessary), and formatting data.

The design of environments in TorchRL allows for flexibility and modularity. By framing tasks as environments, users can
easily extend or modify existing environments using transforms. This approach enables the isolation of individual
components within specific :class:`~torchrl.envs.EnvBase` or :class:`~torchrl.envs.Transform` subclasses, making it
simpler to augment or alter the environment logic.

Available Environment Classes and Utilities
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

TorchRL provides various environment classes and utilities for working with LLMs, including:

- Various environment classes (:class:`~torchrl.envs.llm.ChatEnv`, :class:`~torchrl.envs.llm.DatasetChatEnv`,
  :class:`~torchrl.envs.llm.GSM8KEnv`, etc.)
- Utility functions (:class:`~torchrl.envs.make_gsm8k_env`, :class:`~torchrl.envs.make_mlgym`, etc.)
- Transforms and other supporting classes (:class:`~torchrl.envs.KLRewardTransform`,
  :class:`~torchrl.envs.TemplateTransform`, :class:`~torchrl.envs.Tokenizer`, etc.)

These components can be used to create customized environments tailored to specific use cases and requirements.

.. currentmodule:: torchrl.envs.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ChatEnv
    DatasetChatEnv
    GSM8KEnv
    make_gsm8k_env
    GSM8KPrepareQuestion
    GSM8KEnv
    IFEvalEnv
    IfEvalScorer
    IFEvalScoreData
    LLMEnv
    LLMHashingEnv
    make_mlgym
    MLGymWrapper
    GSM8KRewardParser
    IfEvalScorer
    as_nested_tensor
    as_padded_tensor
    DataLoadingPrimer
    KLRewardTransform
    TemplateTransform
    Tokenizer

Modules
-------

The :ref:`~torchrl.modules.llm` section provides a set of wrappers and utility functions for popular training and
inference backends. The main goal of these primitives is to:

- Unify the input / output data format across training and inference pipelines;
- Unify the input / output data format across backends (to be able to use different backends across losses and
  collectors, for instance)
- Give appropriate tooling to construct these objects in typical RL settings (resource allocation, async execution,
  weight update, etc.)

Wrappers
~~~~~~~~

.. currentmodule:: torchrl.modules.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    TransformersWrapper
    vLLMWrapper

Utils
~~~~~

.. currentmodule:: torchrl.modules.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    CategoricalSequential
    LLMOnDevice
    make_vllm_worker
    stateless_init_process_group
    vLLMWorker

Objectives
----------

LLM post training require some appropriate versions of the losses implemented in TorchRL.

GRPO
~~~~

.. currentmodule:: torchrl.objectives.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    GRPOLoss
    GRPOLossOutput
    MCAdvantage
