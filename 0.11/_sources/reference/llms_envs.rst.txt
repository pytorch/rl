:orphan:

.. currentmodule:: torchrl.envs.llm

LLM Environments
================

The environment layer orchestrates data loading, tool execution, reward computation, and formatting.

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ChatEnv
    DatasetChatEnv
    GSM8KEnv
    make_gsm8k_env
    GSM8KPrepareQuestion
    GSM8KRewardParser
    IFEvalEnv
    IfEvalScorer
    IFEvalScoreData
    LLMEnv
    LLMHashingEnv
    make_mlgym
    MLGymWrapper
