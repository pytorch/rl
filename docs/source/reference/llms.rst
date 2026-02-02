.. currentmodule:: torchrl

LLM Interface
=============

.. _ref_llms:

TorchRL provides a comprehensive framework for LLM post-training and fine-tuning. The LLM API is built around five core concepts that work 
together to create a complete reinforcement learning pipeline for language models.

Key Components
--------------

1. **Data Structures**: History class for conversation management, structured output classes
2. **LLM Wrappers**: Unified interfaces for Transformers, vLLM, SGLang, and async variants  
3. **Environments**: ChatEnv, task-specific environments, and transforms
4. **Collectors**: LLMCollector and RayLLMCollector for data collection
5. **Objectives**: GRPOLoss, SFTLoss for training

Quick Example
-------------

**Using vLLM backend:**

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

**Using SGLang backend:**

.. code-block:: python

    from torchrl.modules.llm import SGLangWrapper, AsyncSGLang
    from torchrl.envs.llm import ChatEnv
    from torchrl.collectors.llm import LLMCollector
    
    # Create SGLang engine (connects to server or launches managed server)
    engine = AsyncSGLang.from_pretrained("Qwen/Qwen2.5-7B", tp_size=2)
    # Or connect to existing server:
    # engine = AsyncSGLang.connect("http://localhost:30000")
    policy = SGLangWrapper(engine, tokenizer=tokenizer, input_mode="history")
    
    # Create environment
    env = ChatEnv(tokenizer=tokenizer)
    
    # Create collector
    collector = LLMCollector(env, policy, dialog_turns_per_batch=256)

.. warning:: The LLM API is still under development and may change in the future. 
    Feedback, issues and PRs are welcome!

Documentation Sections
----------------------

Policy Version Tracking
~~~~~~~~~~~~~~~~~~~~~~~

LLM Collectors also allow to track the version of the policy, which is useful for some use cases.
This is done by adding a :class:`~torchrl.envs.llm.transforms.PolicyVersion` transform to the environment, which is
then incremented by the collector after each weight update. To do this, one either provides the stateful version of the
transform, or a boolean to the collector constructor.

    >>> from torchrl.envs.llm.transforms import PolicyVersion
    >>> from torchrl.collectors.llm import LLMCollector
    >>> from torchrl.weight_update.llm import VLLMWeightSyncScheme, get_model_metadata
    >>> env = make_env() # place your code here
    >>> policy = make_policy() # place your code here
    >>> scheme = VLLMWeightSyncScheme(master_port=29500, gpus_per_replica=1, num_replicas=1)
    >>> collector = LLMCollector(env, policy=policy, weight_sync_schemes={"policy": scheme}, track_policy_version=True)
    >>> # Get the sender and register model
    >>> sender = collector._weight_senders["policy"]
    >>> sender.register_model(training_model)
    >>> # Initialize the collective group
    >>> metadata = get_model_metadata(training_model)
    >>> sender.init_all_workers_group(metadata, vllm_engine=policy.model)
    >>> # Update weights
    >>> sender.update_weights()
    >>> print(collector.policy_version_tracker.version)
    >>> # the policy version is written in the data
    >>> for data in collector:
    ...     print(data["policy_version"])

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
    SGLangWeightSyncScheme
    SGLangWeightSender
    SGLangCollectiveTransport
    get_sglang_model_metadata

Legacy Weight Updaters (Deprecated)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. deprecated:: 0.11
    The `vLLMUpdater` and `vLLMUpdaterV2` classes are deprecated in favor of the new weight synchronization schemes
    (:class:`~torchrl.weight_update.llm.VLLMWeightSyncScheme` and :class:`~torchrl.weight_update.llm.VLLMDoubleBufferSyncScheme`).
    These schemes provide better performance, more flexibility, and cleaner integration with collectors.
    The legacy updaters will be removed in a future release.

    The legacy weight updaters (`vLLMUpdater` and `vLLMUpdaterV2`) are still available but are no longer recommended.
    Please migrate to the new weight synchronization schemes shown above.

.. currentmodule:: torchrl.collectors.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    vLLMUpdater
    vLLMUpdaterV2
    LLMCollector
    RayLLMCollector

Environments
------------

The environment layer orchestrates data loading, tool execution, reward computation, and formatting. When fine-tuning an LLM using TorchRL, the environment is a 
crucial component of the inference pipeline, alongside the policy and collector.

ChatEnv
~~~~~~~

:class:`~torchrl.envs.llm.ChatEnv` serves as a blank canvas for LLM environments - it's a basic tool designed to be extended with transforms that add 
specific functionality. The base ChatEnv provides the fundamental structure for managing conversation state using the 
:class:`~torchrl.data.llm.History` format, but it's intentionally minimal to allow maximum flexibility.

Core Functionality
^^^^^^^^^^^^^^^^^^

ChatEnv operates in three main modes:
- **History mode**: Uses :class:`~torchrl.data.llm.History` objects for conversation management
- **Text mode**: Uses simple text strings for input/output
- **Tokens mode**: Uses tokenized data for input/output

The environment maintains conversation state by:
- **Reset**: Initializes a new conversation with an optional system prompt
- **Step**: Takes the LLM's response and updates the conversation history, preparing the next prompt

Transform-Based Architecture
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Transforms are the main way to extend ChatEnv with specific capabilities:

- **Reward computation**: :class:`~torchrl.envs.llm.transforms.KLRewardTransform` for KL divergence rewards
- **Tool execution**: :class:`~torchrl.envs.llm.transforms.PythonInterpreter` for Python code 
  execution, :class:`~torchrl.envs.llm.transforms.MCPToolTransform` for general tool calling.
- **Data loading**: :class:`~torchrl.envs.llm.transforms.DataLoadingPrimer` for loading prompts from datasets
- **Thinking prompts**: :class:`~torchrl.envs.llm.transforms.AddThinkingPrompt` for chain-of-thought reasoning
- **Policy tracking**: :class:`~torchrl.envs.llm.transforms.PolicyVersion` for version control
- **Step counting**: Built-in step tracking and reset management using :class:`~torchrl.envs.transforms.StepCounter`.

Integration with LLM Wrappers
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. _ref_env_llm_step:

ChatEnv is designed to work seamlessly with both :class:`~torchrl.modules.llm.TransformersWrapper` and :class:`~torchrl.modules.llm.vLLMWrapper`. 
The environment handles the conversation state management while the wrapper handles the actual LLM inference, creating a clean separation of concerns.

On each call to `step`, the environment:

- Takes the LLM's output, specifically the `full` field, which contains the entire conversation so far, including the new response (e.g., `history.full`, `text.full`, `tokens.full`).
- Sets this `full` field as the new `prompt` for the next LLM step (e.g., `td["next", "history"].prompt`, `td["next", "text"].prompt`, `td["next", "tokens"].prompt`).
- Optionally, applies transforms to insert new user messages, tool calls, or other modifications to the conversation before the next LLM step to refine the prompt.

This mechanism enables seamless multi-turn interactions and supports complex workflows such as tool use and reward shaping.

Token-First API
^^^^^^^^^^^^^^^

For maximum reliability in multi-turn conversations, TorchRL provides a **token-first API** that maintains 
pre-tokenized inputs throughout the conversation. This ensures KV cache prefix consistency and consistent 
log-probabilities across turns, which is more robust than repeatedly detokenizing and re-tokenizing.

**How it works:**

1. Use ``ChatEnv`` with ``with_tokenizer=True`` to automatically wrap the environment with an 
   :class:`~torchrl.envs.llm.transforms.IncrementalTokenizer` transform
2. Set ``prefer_tokens=True`` in the LLM wrapper to use pre-tokenized inputs when available

.. code-block:: python

    from torchrl.envs.llm import ChatEnv
    from torchrl.modules.llm import TransformersWrapper
    from transformers import AutoTokenizer, AutoModelForCausalLM

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")

    # Create environment with automatic tokenization
    env = ChatEnv(
        tokenizer=tokenizer,
        system_prompt="You are a helpful assistant.",
        batch_size=[1],
        with_tokenizer=True,  # Wraps with IncrementalTokenizer
    )

    # Create wrapper that uses pre-tokenized inputs
    wrapper = TransformersWrapper(
        model=model,
        tokenizer=tokenizer,
        input_mode="history",
        prefer_tokens=True,  # Use tokens.prompt when available
    )

    # The environment maintains tokens.prompt in sync with history.prompt
    td = env.reset()
    assert ("tokens", "prompt") in td.keys(True, True)

    # The wrapper uses these tokens directly, bypassing re-tokenization
    td_out = wrapper(td)

**Benefits:**

- **KV cache consistency**: The token prefix stays exactly the same across turns
- **Consistent log-probs**: No tokenization variations between forward passes
- **Efficiency**: Avoids redundant tokenization work

The :class:`~torchrl.envs.llm.transforms.IncrementalTokenizer` transform automatically tokenizes 
``history.prompt`` on each reset and step, storing the result in ``tokens.prompt``. The LLM wrappers 
(:class:`~torchrl.modules.llm.TransformersWrapper` and :class:`~torchrl.modules.llm.vLLMWrapper`) 
check for ``tokens.prompt`` when ``prefer_tokens=True`` and use it directly as input instead of 
re-tokenizing from history.

Task-Specific Environments
~~~~~~~~~~~~~~~~~~~~~~~~~~

We provide a few task-specific environments, such as :class:`~torchrl.envs.llm.GSM8KEnv` for the GSM8K dataset,
:class:`~torchrl.envs.llm.IFEvalEnv` for the IFEval dataset, and :class:`~torchrl.envs.llm.MLGymEnv` for MLGym integration.

These environments wrap a :class:`~torchrl.envs.llm.ChatEnv` and add a :class:`~torchrl.envs.llm.transforms.DataLoadingPrimer` transform
(plus an optional reward parsing transform) in a :class:`~torchrl.envs.TransformedEnv` class.



.. currentmodule:: torchrl.envs.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    ChatEnv
    DatasetChatEnv
    GSM8KEnv
    make_gsm8k_env
    GSM8KPrepareQuestion
    IFEvalEnv
    IfEvalScorer
    IFEvalScoreData
    LLMEnv
    LLMHashingEnv
    make_mlgym
    MLGymWrapper
    GSM8KRewardParser

Transforms
~~~~~~~~~~

Transforms are used to modify the data before it is passed to the LLM.
Tools are usually implemented as transforms, and appended to a base environment
such as :class:`~torchrl.envs.llm.ChatEnv`.

An example of a tool transform is the :class:`~torchrl.envs.llm.transforms.PythonInterpreter` transform, which is used
to execute Python code in the context of the LLM. The PythonInterpreter can optionally use a shared 
:class:`~torchrl.envs.llm.transforms.PythonExecutorService` for efficient resource usage across multiple environments.
See :doc:`services` for more details on the service registry system.

    >>> from torchrl.envs.llm.transforms import PythonInterpreter
    >>> from torchrl.envs.llm import ChatEnv
    >>> from tensordict import TensorDict, set_list_to_stack
    >>> from transformers import AutoTokenizer
    >>> from pprint import pprint
    >>> set_list_to_stack(True).set()
    >>> tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    >>> base_env = ChatEnv(
    ...     tokenizer=tokenizer,
    ...     system_prompt="You are an assistant that can execute Python code. Decorate your code with ```python``` tags.",
    ...     user_role="user",
    ...     system_role="system",
    ...     batch_size=[1],
    ... )
    >>> env = base_env.append_transform(PythonInterpreter())
    >>> env.set_seed(0)
    >>> # Pass the reset data - the prompt - to the environment
    >>> reset_data = env.reset(TensorDict(
    ...     text="Let's write a Python function that returns the square of a number.",
    ...     batch_size=[1])
    ... )
    >>> # Simulate an action - i.e., a response from the LLM (as if we were an LLM) 
    >>> action = """Here is a block of code to be executed in python:
    ... ```python
    ... def square(x):
    ...     return x * x
    ... print('testing the square function with input 2:', square(2))
    ... ```
    ... <|im_end|>
    ... """
    >>> step_data = reset_data.set("text_response", [action])
    >>> s, s_ = env.step_and_maybe_reset(reset_data)
    >>> # The history is a stack of chat messages.
    >>> #  The python interpreter transform has executed the code in the last message.
    >>> pprint(s_["history"].apply_chat_template(tokenizer=tokenizer))
    ['<|im_start|>system\n'
     'You are an assistant that can execute Python code. Decorate your code with '
     '```python``` tags.<|im_end|>\n'
     '<|im_start|>user\n'
     "Let's write a Python function that returns the square of a "
     'number.<|im_end|>\n'
     '<|im_start|>assistant\n'
     'Here is a block of code to be executed in python:\n'
     '```python\n'
     'def square(x):\n'
     '    return x * x\n'
     "print('testing the square function with input 2:', square(2))\n"
     '```<|im_end|>\n'
     '<|im_start|>user\n'
     '<tool_response>\n'
     'Code block 1 executed successfully:\n'
     'testing the square function with input 2: 4\n'
     '\n'
     '</tool_response><|im_end|>\n'
     '<|im_start|>assistant\n']

Similarly, environments that load data from a dataset are just special instances of the :class:`~torchrl.envs.llm.ChatEnv`
augmented with a :class:`~torchrl.envs.llm.transforms.DataLoadingPrimer` transforms (and some dedicated reward parsing
transforms).

Designing Reward Transforms
^^^^^^^^^^^^^^^^^^^^^^^^^^^

When designing reward transforms for LLM environments, several key considerations must be 
addressed to ensure proper integration with the training pipeline. 
The examples of :class:`~torchrl.envs.llm.GSM8KRewardParser` and 
:class:`~torchrl.envs.llm.IfEvalScorer` provide excellent templates for reward transform design.

**Reward Shape Requirements**

The reward tensor must have the same number of dimensions as the logits, which is typically 
two more dimensions than the environment batch size:

- **Sparse rewards**: Shape ``(*bsz, 1, 1)`` - single reward per sequence
- **Dense rewards**: Shape ``(*bsz, num_tokens, 1)`` - per-token rewards

This shape requirement ensures compatibility with the loss computation pipeline. 
For example, in the GSM8K reward parser:

.. code-block:: python

    # Rewards need to have shape broadcastable to [batch x tokens x 1]
    tds = tds.apply(lambda t: t.unsqueeze(-1).unsqueeze(-1))

**Done State Management**

It is crucial to properly manage the done state to prevent endless generation. Common strategies include:

1. **Completion-based termination**: Set done when the response is complete (e.g., ``History.complete=True``)
2. **Content-based termination**: Set done when specific content is detected (e.g., ``<answer>`` blocks)
3. **Step-based termination**: Use :class:`~torchrl.envs.transforms.StepCounter` for predetermined step limits

Example from IFEvalScorer:

.. code-block:: python

    if self.set_done_if_answer and bool(answer_blocks):
        next_tensordict.set("done", torch.ones(...))
        next_tensordict.set("terminated", torch.ones(...))

**Input Mode Handling**

Reward transforms must handle different input modes correctly:

- **History mode**: Extract text from ``("history", "full")`` or ``("history", "response")``
- **Text mode**: Use text directly from ``("text", "full")`` or ``("text", "response")``
- **Tokens mode**: Decode tokens from ``("tokens", "full")`` or ``("tokens", "response")``

The GSM8K reward parser demonstrates this pattern:

.. code-block:: python

    if input_mode == "history":
        responses = lazy_stack([r[..., -1] for r in responses.unbind(0)])
        if hasattr(responses, "content"):
            text_completion = responses.content
    elif input_mode == "text":
        text_completion = responses
    elif input_mode == "tokens":
        text_completion = self.tokenizer.decode(responses.flatten(0, 1).tolist())

**Specification Management**

Accurate specification of reward and observation specs is essential for proper environment initialization. Both GSM8K and IFEval provide good examples:

.. code-block:: python

    def transform_reward_spec(self, reward_spec: Composite) -> Composite:
        shape = reward_spec.shape + (1, 1)
        reward_spec.update(
            Composite(
                reward_answer=Unbounded(shape),
                reward_think=Unbounded(shape),
                reward_right=Unbounded(shape),
                reward_contained=Unbounded(shape),
                reward=Unbounded(shape),
                success=Unbounded(shape, dtype=torch.bool),
            )
        )
        return reward_spec

**Batch Processing Considerations**

For efficient processing, handle batched data appropriately:

1. **Flatten batch dimensions**: Use ``tensordict.view(-1)`` for processing
2. **Reshape results**: Restore original batch structure after processing
3. **Handle variable-length sequences**: Use proper padding and masking

**Reward Aggregation Strategies**

Consider different reward aggregation approaches:

1. **Simple aggregation**: Sum or average multiple reward components
2. **Weighted aggregation**: Apply different weights to different components
3. **Conditional rewards**: Base rewards on specific conditions or thresholds

The IFEvalScorer demonstrates a sophisticated aggregation strategy:

.. code-block:: python

    def default_reward_aggregator(self, score: IFEvalScoreData, ...):
        # Format score (max 1.0)
        format_score = (format_components * weights).sum(dim=-1, keepdim=True)
        
        # Structure score (max 1.0)
        structure_score = think_score + answer_score
        
        # Completion bonus (max 0.2)
        completion_bonus = float(complete) * 0.2
        
        return format_score + structure_score + completion_bonus

**Post-Processing in Replay Buffers**

Rewards can also be computed after the fact by appending transforms to the replay buffer. However, done state capture must remain in the environment transform since it needs to occur on-the-fly during data collection.

**Error Handling and Robustness**

Implement robust error handling for parsing failures:

.. code-block:: python

    try:
        cot, potential_answer = self.extract_tags(compl)
    except ET.ParseError:
        cot, potential_answer = ("", "")

**Performance Considerations**

1. **Avoid redundant computations**: Cache parsed results when possible
2. **Use efficient text processing**: Leverage regex or XML parsing as appropriate
3. **Minimize memory allocations**: Reuse tensors and avoid unnecessary copies

By following these design principles, reward transforms can be effectively integrated into the LLM training pipeline while maintaining performance and reliability.

.. currentmodule:: torchrl.envs.llm.transforms

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    AddThinkingPrompt
    BrowserTransform
    DataLoadingPrimer
    IncrementalTokenizer
    KLComputation
    KLRewardTransform
    MCPToolTransform
    PolicyVersion
    PythonExecutorService
    PythonInterpreter
    RayDataLoadingPrimer
    RetrieveKL
    RetrieveLogProb
    TemplateTransform
    Tokenizer
    as_nested_tensor
    as_padded_tensor

Objectives
----------

LLM post-training requires specialized loss functions that are adapted to the unique characteristics of language models.

GRPO, DAPO, CISPO
^^^^^^^^^^^^^^^^^

.. currentmodule:: torchrl.objectives.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    LLMLossOutput
    GRPOLoss
    GRPOLossOutput
    CISPOLoss
    CISPOLossOutput
    DAPO
    DAPOLossOutput
    MCAdvantage

SFT
^^^

.. currentmodule:: torchrl.objectives.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    SFTLoss
    SFTLossOutput

.. currentmodule:: torchrl.data.llm

.. autosummary::
    :toctree: generated/
    :template: rl_template.rst

    TopKRewardSelector
