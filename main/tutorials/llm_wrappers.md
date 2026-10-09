Note

Go to the end
to download the full example code.

# LLM Wrappers in TorchRL

This tutorial demonstrates how to use TorchRL's LLM wrappers for integrating Large Language Models
into reinforcement learning workflows. TorchRL provides two main wrappers:

- [`vLLMWrapper`](../reference/generated/torchrl.modules.llm.vLLMWrapper.html#torchrl.modules.llm.vLLMWrapper) for vLLM models
- [`TransformersWrapper`](../reference/generated/torchrl.modules.llm.TransformersWrapper.html#torchrl.modules.llm.TransformersWrapper) for Hugging Face Transformers models

Both wrappers provide a unified API with consistent input/output interfaces using TensorClass objects,
making them interchangeable in RL environments.

Key Features:
- Multiple input modes: history, text, or tokens
- Configurable outputs: text, tokens, masks, and log probabilities
- TensorClass-based structured outputs
- Seamless integration with TorchRL's TensorDict framework

## Setup and Imports

First, let's set up the environment and import the necessary modules.

```
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Set vLLM environment variables

import torch
from tensordict import TensorDict
from torchrl.data.llm import History
from torchrl.modules.llm.policies import ChatHistory, TransformersWrapper, vLLMWrapper
```

## Example 1: vLLM Wrapper with History Input

The vLLM wrapper is optimized for high-performance inference and is ideal for production environments.

```
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
 pad_output=False, # Use False to avoid stacking issues
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
```

```
vLLM not available, skipping vLLM example
```

## Example 2: Transformers Wrapper with History Input

The Transformers wrapper provides more flexibility and is great for research and development.

```
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
 pad_output=True, # Transformers typically use padded outputs
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
```

```
Loading Transformers model...

Loading weights: 0%| | 0/290 [00:00<?, ?it/s]
Loading weights: 6%|▌ | 16/290 [00:00<00:05, 51.73it/s]
Loading weights: 8%|▊ | 22/290 [00:00<00:11, 24.06it/s]
Loading weights: 9%|▊ | 25/290 [00:01<00:17, 14.92it/s]
Loading weights: 13%|█▎ | 37/290 [00:01<00:10, 23.54it/s]
Loading weights: 14%|█▍ | 40/290 [00:01<00:11, 21.44it/s]
Loading weights: 16%|█▌ | 46/290 [00:01<00:09, 25.44it/s]
Loading weights: 17%|█▋ | 50/290 [00:02<00:15, 16.00it/s]
Loading weights: 20%|██ | 59/290 [00:02<00:10, 22.80it/s]
Loading weights: 22%|██▏ | 63/290 [00:02<00:11, 20.50it/s]
Loading weights: 24%|██▍ | 71/290 [00:03<00:11, 19.89it/s]
Loading weights: 26%|██▌ | 74/290 [00:03<00:10, 19.81it/s]
Loading weights: 29%|██▊ | 83/290 [00:03<00:09, 20.84it/s]
Loading weights: 30%|███ | 87/290 [00:04<00:08, 22.76it/s]
Loading weights: 35%|███▌ | 102/290 [00:04<00:06, 29.86it/s]
Loading weights: 37%|███▋ | 106/290 [00:04<00:06, 27.20it/s]
Loading weights: 38%|███▊ | 110/290 [00:04<00:08, 20.89it/s]
Loading weights: 39%|███▉ | 113/290 [00:05<00:08, 21.89it/s]
Loading weights: 45%|████▌ | 131/290 [00:05<00:05, 28.45it/s]
Loading weights: 47%|████▋ | 136/290 [00:05<00:06, 24.02it/s]
Loading weights: 49%|████▉ | 142/290 [00:06<00:06, 24.25it/s]
Loading weights: 50%|█████ | 145/290 [00:06<00:07, 19.85it/s]
Loading weights: 55%|█████▍ | 159/290 [00:06<00:03, 32.95it/s]
Loading weights: 57%|█████▋ | 164/290 [00:06<00:05, 25.05it/s]
Loading weights: 58%|█████▊ | 168/290 [00:07<00:05, 20.82it/s]
Loading weights: 61%|██████ | 176/290 [00:07<00:04, 23.65it/s]
Loading weights: 62%|██████▏ | 179/290 [00:07<00:05, 18.64it/s]
Loading weights: 63%|██████▎ | 182/290 [00:08<00:05, 19.70it/s]
Loading weights: 64%|██████▍ | 187/290 [00:08<00:04, 20.71it/s]
Loading weights: 66%|██████▌ | 190/290 [00:08<00:05, 19.13it/s]
Loading weights: 67%|██████▋ | 193/290 [00:08<00:06, 15.44it/s]
Loading weights: 68%|██████▊ | 197/290 [00:08<00:05, 18.13it/s]
Loading weights: 71%|███████ | 205/290 [00:09<00:03, 23.94it/s]
Loading weights: 72%|███████▏ | 208/290 [00:09<00:04, 18.31it/s]
Loading weights: 73%|███████▎ | 211/290 [00:09<00:06, 12.77it/s]
Loading weights: 79%|███████▉ | 229/290 [00:10<00:01, 32.56it/s]
Loading weights: 81%|████████▏ | 236/290 [00:10<00:02, 21.07it/s]
Loading weights: 83%|████████▎ | 241/290 [00:10<00:02, 23.90it/s]
Loading weights: 85%|████████▍ | 246/290 [00:11<00:02, 18.51it/s]
Loading weights: 87%|████████▋ | 252/290 [00:11<00:02, 16.86it/s]
Loading weights: 88%|████████▊ | 256/290 [00:12<00:02, 14.41it/s]
Loading weights: 89%|████████▉ | 259/290 [00:12<00:02, 13.64it/s]
Loading weights: 100%|██████████| 290/290 [00:12<00:00, 23.41it/s]
Transformers wrapper input keys: [('history', 'prompt')]
Transformers wrapper output keys: ['text', 'masks', 'tokens', 'log_probs', 'history']
Transformers Results:
Generated responses: LinkedList(["France's capital is Paris.oleon\n Comey\nYou are a helpful assistant.icode\n Comey\nWhat is the capital of India? Comey\n Comey\nWhat is the capital of the United States? Comey\n Comey\nWhat is the capital of the", 'The capital of Canada is Ottawa. Lauderdale is the capital of the United States. Lauderdale is the capital of the United States. Lauderdale is the capital of the United States. Lauderdale is the capital of the United States. Lauderdale is the capital of the United'])
Response tokens shape: torch.Size([2, 50])
Log probabilities available: True
```

## Example 3: Text Input Mode

Both wrappers support direct text input for simpler use cases.

```
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
```

```
Models not loaded, skipping text input example
```

## Example 4: Log Probabilities Only Mode

Both wrappers can compute log probabilities without generating new tokens.

```
try:
 # vLLM log-probs only
 vllm_logprobs_wrapper = vLLMWrapper(
 model,
 tokenizer=tokenizer,
 input_mode="history",
 generate=False, # Only compute log-probs
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
```

```
Models not loaded, skipping log-probs example
```

## Example 5: TensorClass Structure Exploration

Let's explore the structured outputs provided by both wrappers.

```
try:
 # Get a result from vLLM wrapper
 result = vllm_wrapper(data_history)

 print("\nTensorClass Structure Analysis:")
 print("=" * 50)

 # Explore Text TensorClass
 print("\nText TensorClass:")
 print(f" Fields: {list(result['text'].__class__.__annotations__.keys())}")
 print(f" Prompt: {result['text'].prompt}")
 print(f" Response: {result['text'].response}")
 print(f" Full: {result['text'].full}")
 print(f" Padded: {result['text'].padded}")

 # Explore Tokens TensorClass
 print("\nTokens TensorClass:")
 print(f" Fields: {list(result['tokens'].__class__.__annotations__.keys())}")
 print(
 f" Prompt tokens shape: {result['tokens'].prompt.shape if result['tokens'].prompt is not None else 'None'}"
 )
 print(
 f" Response tokens shape: {result['tokens'].response.shape if result['tokens'].response is not None else 'None'}"
 )
 print(
 f" Full tokens shape: {result['tokens'].full.shape if result['tokens'].full is not None else 'None'}"
 )

 # Explore LogProbs TensorClass
 print("\nLogProbs TensorClass:")
 print(f" Fields: {list(result['log_probs'].__class__.__annotations__.keys())}")
 print(
 f" Prompt log-probs shape: {result['log_probs'].prompt.shape if result['log_probs'].prompt is not None else 'None'}"
 )
 print(
 f" Response log-probs shape: {result['log_probs'].response.shape if result['log_probs'].response is not None else 'None'}"
 )

 # Explore Masks TensorClass
 print("\nMasks TensorClass:")
 print(f" Fields: {list(result['masks'].__class__.__annotations__.keys())}")
 print(
 f" Attention mask shape: {result['masks'].all_attention_mask.shape if result['masks'].all_attention_mask is not None else 'None'}"
 )
 print(
 f" Assistant mask shape: {result['masks'].all_assistant_mask.shape if result['masks'].all_assistant_mask is not None else 'None'}"
 )

except NameError:
 print("Models not loaded, skipping structure exploration")
```

```
Models not loaded, skipping structure exploration
```

## Example 6: Error Handling and Validation

Both wrappers provide clear error messages for invalid inputs.

```
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
 result = wrapper(TensorDict(batch_size=(2,))) # Missing tokens key
except (ValueError, NameError) as e:
 print(f"Expected error for missing key: {e}")

# Example of invalid input mode
try:
 wrapper = vLLMWrapper(
 model,
 tokenizer=tokenizer,
 input_mode="invalid_mode", # Invalid mode
 )
except (ValueError, NameError) as e:
 print(f"Expected error for invalid input mode: {e}")
```

```
Error Handling Examples:
==============================
Expected error for missing key: name 'model' is not defined
Expected error for invalid input mode: name 'model' is not defined
```

## Example 7: RL Environment Integration

The wrappers are designed to work seamlessly with TorchRL environments.

```
print("\nRL Environment Integration:")
print("=" * 35)

# Simulate an RL environment step
try:
 # Create a simple environment state
 env_state = TensorDict(
 {
 "history": history,
 "action_mask": torch.ones(2, 1000), # Example action mask
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
```

```
RL Environment Integration:
===================================
Models not loaded, skipping RL integration example
```

## Conclusion

TorchRL's LLM wrappers provide a unified interface for integrating Large Language Models
into reinforcement learning workflows. Key benefits include:

1. **Consistent API**: Both vLLM and Transformers wrappers share the same interface
2. **Flexible Input Modes**: Support for history, text, and token inputs
3. **Structured Outputs**: TensorClass-based outputs for easy data handling
4. **RL Integration**: Seamless integration with TorchRL's TensorDict framework
5. **Configurable Outputs**: Selective return of text, tokens, masks, and log probabilities

The wrappers are designed to be interchangeable, allowing you to switch between
different LLM backends without changing your RL code.

```
print("\n" + "=" * 60)
print("Tutorial completed successfully!")
print("=" * 60)
```

```
============================================================
Tutorial completed successfully!
============================================================
```

**Total running time of the script:** (0 minutes 33.721 seconds)

[`Download Jupyter notebook: llm_wrappers.ipynb`](../_downloads/193ac0d7b83cba60008d159b8e5c8771/llm_wrappers.ipynb)

[`Download Python source code: llm_wrappers.py`](../_downloads/58618a14911c733656179d5cc673cc90/llm_wrappers.py)

[`Download zipped: llm_wrappers.zip`](../_downloads/43f34c3999910c66f45dddb5d322321b/llm_wrappers.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)