"""
LLM Wrappers in TorchRL
=======================

This tutorial demonstrates how to use TorchRL's LLM wrappers for integrating Large Language Models
into reinforcement learning workflows. TorchRL provides two main wrappers:

- :class:`~torchrl.modules.llm.policies.vLLMWrapper` for vLLM models
- :class:`~torchrl.modules.llm.policies.TransformersWrapper` for Hugging Face Transformers models

Both wrappers provide a unified API with consistent input/output interfaces using TensorClass objects,
making them interchangeable in RL environments.

Key Features:
- Multiple input modes: history, text, or tokens
- Configurable outputs: text, tokens, masks, and log probabilities
- TensorClass-based structured outputs
- Seamless integration with TorchRL's TensorDict framework
"""

# %%
# Setup and Imports
# -----------------
# First, let's set up the environment and import the necessary modules.

import os
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set vLLM environment variables
os.environ["VLLM_USE_V1"] = "0"

import torch
from tensordict import TensorDict
from torchrl.data.llm import History
from torchrl.modules.llm.policies import ChatHistory, TransformersWrapper, vLLMWrapper

# %%
# Example 1: vLLM Wrapper with History Input
# ------------------------------------------
# The vLLM wrapper is optimized for high-performance inference and is ideal for production environments.

try:
    from transformers import AutoTokenizer
    from vllm import LLM

    print("Loading vLLM model...")
    model = LLM(model="Qwen/Qwen2.5-0.5B")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Create conversation history
    chats = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Canada?"},
        ],
    ]
    history = History.from_chats(chats)
    chat_history = ChatHistory(prompt=history)

    # Create vLLM wrapper with history input (recommended for RL environments)
    vllm_wrapper = vLLMWrapper(
        model,
        tokenizer=tokenizer,
        input_mode="history",
        generate=True,
        return_log_probs=True,
        pad_output=False,  # Use False to avoid stacking issues
    )

    print(f"vLLM wrapper input keys: {vllm_wrapper.in_keys}")
    print(f"vLLM wrapper output keys: {vllm_wrapper.out_keys}")

    # Process the data
    data_history = TensorDict(history=chat_history, batch_size=(2,))
    result = vllm_wrapper(data_history)

    print("vLLM Results:")
    print(f"Generated responses: {result['text'].response}")
    print(
        f"Response tokens shape: {result['tokens'].response.shape if result['tokens'].response is not None else 'None'}"
    )
    print(f"Log probabilities available: {result['log_probs'].response is not None}")

except ImportError:
    print("vLLM not available, skipping vLLM example")

# %%
# Example 2: Transformers Wrapper with History Input
# --------------------------------------------------
# The Transformers wrapper provides more flexibility and is great for research and development.

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("\nLoading Transformers model...")
    transformers_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B")
    transformers_tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")

    # Create Transformers wrapper with same interface
    transformers_wrapper = TransformersWrapper(
        transformers_model,
        tokenizer=transformers_tokenizer,
        input_mode="history",
        generate=True,
        return_log_probs=True,
        pad_output=True,  # Transformers typically use padded outputs
        generate_kwargs={"max_new_tokens": 50},
    )

    print(f"Transformers wrapper input keys: {transformers_wrapper.in_keys}")
    print(f"Transformers wrapper output keys: {transformers_wrapper.out_keys}")

    # Create data for the Transformers wrapper
    chats = [
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of Canada?"},
        ],
    ]
    history = History.from_chats(chats)
    chat_history = ChatHistory(prompt=history)
    data_history = TensorDict(history=chat_history, batch_size=(2,))

    # Process the data
    result_tf = transformers_wrapper(data_history)

    print("Transformers Results:")
    print(f"Generated responses: {result_tf['text'].response}")
    print(
        f"Response tokens shape: {result_tf['tokens'].response.shape if result_tf['tokens'].response is not None else 'None'}"
    )
    print(f"Log probabilities available: {result_tf['log_probs'].response is not None}")

except ImportError:
    print("Transformers not available, skipping Transformers example")

# %%
# Example 3: Text Input Mode
# --------------------------
# Both wrappers support direct text input for simpler use cases.

try:
    # Create text input data
    prompts = ["The capital of France is", "The capital of Canada is"]
    data_text = TensorDict(text=prompts, batch_size=(2,))

    # vLLM with text input
    vllm_text_wrapper = vLLMWrapper(
        model,
        tokenizer=tokenizer,
        input_mode="text",
        generate=True,
        pad_output=False,
    )

    result_vllm_text = vllm_text_wrapper(data_text)
    print("\nvLLM Text Input Results:")
    print(f"Generated text: {result_vllm_text['text'].response}")

    # Transformers with text input
    transformers_text_wrapper = TransformersWrapper(
        transformers_model,
        tokenizer=transformers_tokenizer,
        input_mode="text",
        generate=True,
        pad_output=True,
        generate_kwargs={"max_new_tokens": 20},
    )

    result_tf_text = transformers_text_wrapper(data_text)
    print("Transformers Text Input Results:")
    print(f"Generated text: {result_tf_text['text'].response}")

except NameError:
    print("Models not loaded, skipping text input example")

# %%
# Example 4: Log Probabilities Only Mode
# --------------------------------------
# Both wrappers can compute log probabilities without generating new tokens.

try:
    # vLLM log-probs only
    vllm_logprobs_wrapper = vLLMWrapper(
        model,
        tokenizer=tokenizer,
        input_mode="history",
        generate=False,  # Only compute log-probs
        return_log_probs=True,
        pad_output=False,
    )

    result_vllm_lp = vllm_logprobs_wrapper(data_history)
    print("\nvLLM Log Probabilities:")
    print(
        f"Prompt log-probs shape: {result_vllm_lp['log_probs'].prompt.shape if result_vllm_lp['log_probs'].prompt is not None else 'None'}"
    )

    # Transformers log-probs only
    transformers_logprobs_wrapper = TransformersWrapper(
        transformers_model,
        tokenizer=transformers_tokenizer,
        input_mode="history",
        generate=False,
        return_log_probs=True,
        pad_output=True,
    )

    result_tf_lp = transformers_logprobs_wrapper(data_history)
    print("Transformers Log Probabilities:")
    print(
        "Prompt log-probs shape: {result_tf_lp['log_probs'].prompt.shape if result_tf_lp['log_probs'].prompt is not None else 'None'}"
    )

except NameError:
    print("Models not loaded, skipping log-probs example")

# %%
# Example 5: TensorClass Structure Exploration
# ----------------------------------------------
# Let's explore the structured outputs provided by both wrappers.

try:
    # Get a result from vLLM wrapper
    result = vllm_wrapper(data_history)

    print("\nTensorClass Structure Analysis:")
    print("=" * 50)

    # Explore Text TensorClass
    print("\nText TensorClass:")
    print(f"  Fields: {list(result['text'].__class__.__annotations__.keys())}")
    print(f"  Prompt: {result['text'].prompt}")
    print(f"  Response: {result['text'].response}")
    print(f"  Full: {result['text'].full}")
    print(f"  Padded: {result['text'].padded}")

    # Explore Tokens TensorClass
    print("\nTokens TensorClass:")
    print(f"  Fields: {list(result['tokens'].__class__.__annotations__.keys())}")
    print(
        f"  Prompt tokens shape: {result['tokens'].prompt.shape if result['tokens'].prompt is not None else 'None'}"
    )
    print(
        f"  Response tokens shape: {result['tokens'].response.shape if result['tokens'].response is not None else 'None'}"
    )
    print(
        f"  Full tokens shape: {result['tokens'].full.shape if result['tokens'].full is not None else 'None'}"
    )

    # Explore LogProbs TensorClass
    print("\nLogProbs TensorClass:")
    print(f"  Fields: {list(result['log_probs'].__class__.__annotations__.keys())}")
    print(
        f"  Prompt log-probs shape: {result['log_probs'].prompt.shape if result['log_probs'].prompt is not None else 'None'}"
    )
    print(
        f"  Response log-probs shape: {result['log_probs'].response.shape if result['log_probs'].response is not None else 'None'}"
    )

    # Explore Masks TensorClass
    print("\nMasks TensorClass:")
    print(f"  Fields: {list(result['masks'].__class__.__annotations__.keys())}")
    print(
        f"  Attention mask shape: {result['masks'].all_attention_mask.shape if result['masks'].all_attention_mask is not None else 'None'}"
    )
    print(
        f"  Assistant mask shape: {result['masks'].all_assistant_mask.shape if result['masks'].all_assistant_mask is not None else 'None'}"
    )

except NameError:
    print("Models not loaded, skipping structure exploration")

# %%
# Example 6: Error Handling and Validation
# ----------------------------------------
# Both wrappers provide clear error messages for invalid inputs.

print("\nError Handling Examples:")
print("=" * 30)

# Example of missing required key
try:
    wrapper = vLLMWrapper(
        model,
        tokenizer=tokenizer,
        input_mode="tokens",
        input_key="tokens",
    )
    result = wrapper(TensorDict(batch_size=(2,)))  # Missing tokens key
except (ValueError, NameError) as e:
    print(f"Expected error for missing key: {e}")

# Example of invalid input mode
try:
    wrapper = vLLMWrapper(
        model,
        tokenizer=tokenizer,
        input_mode="invalid_mode",  # Invalid mode
    )
except (ValueError, NameError) as e:
    print(f"Expected error for invalid input mode: {e}")

# %%
# Example 7: RL Environment Integration
# --------------------------------------
# The wrappers are designed to work seamlessly with TorchRL environments.

print("\nRL Environment Integration:")
print("=" * 35)

# Simulate an RL environment step
try:
    # Create a simple environment state
    env_state = TensorDict(
        {
            "history": history,
            "action_mask": torch.ones(2, 1000),  # Example action mask
            "reward": torch.zeros(2),
            "done": torch.zeros(2, dtype=torch.bool),
        },
        batch_size=(2,),
    )

    # Use the wrapper as a policy
    action_output = vllm_wrapper(env_state)

    print("Environment integration successful!")
    print(f"Generated actions: {action_output['text'].response}")
    print(
        f"Action log probabilities: {action_output['log_probs'].response is not None}"
    )

except NameError:
    print("Models not loaded, skipping RL integration example")

# %%
# Conclusion
# ----------
# TorchRL's LLM wrappers provide a unified interface for integrating Large Language Models
# into reinforcement learning workflows. Key benefits include:
#
# 1. **Consistent API**: Both vLLM and Transformers wrappers share the same interface
# 2. **Flexible Input Modes**: Support for history, text, and token inputs
# 3. **Structured Outputs**: TensorClass-based outputs for easy data handling
# 4. **RL Integration**: Seamless integration with TorchRL's TensorDict framework
# 5. **Configurable Outputs**: Selective return of text, tokens, masks, and log probabilities
#
# The wrappers are designed to be interchangeable, allowing you to switch between
# different LLM backends without changing your RL code.

print("\n" + "=" * 60)
print("Tutorial completed successfully!")
print("=" * 60)
