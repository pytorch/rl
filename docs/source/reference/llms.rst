.. currentmodule:: torchrl

LLM Interface
=============

.. _ref_llms:

TorchRL provides a comprehensive framework for LLM post-training and fine-tuning. The LLM API is built around five core concepts that work 
together to create a complete reinforcement learning pipeline for language models.

Key Components
--------------

1. **Data Structures**: History class for conversation management, structured output classes
2. **LLM Wrappers**: Unified interfaces for Transformers, vLLM, and AsyncVLLM  
3. **Environments**: ChatEnv, task-specific environments, and transforms
4. **Collectors**: LLMCollector and RayLLMCollector for data collection
5. **Objectives**: GRPOLoss, SFTLoss for training

Quick Example
-------------

.. code-block:: python

    from torchrl.modules.llm import vLLMWrapper, AsyncVLLM
    from torchrl.envs.llm import ChatEnv
    from torchrl.collectors.llm import LLMCollector
    
    # Create vLLM engine
    engine = AsyncVLLM.from_pretrained("Qwen/Qwen2.5-7B", num_replicas=2)
    policy = vLLMWrapper(engine, input_mode="history")
    
    # Create environment
    env = ChatEnv(tokenizer=tokenizer)
    
    # Create collector
    collector = LLMCollector(env, policy, dialog_turns_per_batch=256)

.. warning:: The LLM API is still under development and may change in the future. 
    Feedback, issues and PRs are welcome!

Documentation Sections
----------------------

.. toctree::
   :maxdepth: 2

   llms_data
   llms_modules
   llms_envs
   llms_transforms
   llms_collectors
   llms_objectives
