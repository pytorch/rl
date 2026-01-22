# TransformersWrapper and ChatEnv Integration

This document describes the data flow between `TransformersWrapper` and `ChatEnv` when using `input_mode="history"`, and the `ChatHistory` contract that enables proper integration.

## Overview

When using LLM environments with policy wrappers in TorchRL, the data flows through a well-defined contract based on the `ChatHistory` class. Understanding this contract is essential for debugging integration issues.

## The ChatHistory Contract

`ChatHistory` is a TensorClass with three optional attributes:

```python
class ChatHistory(TensorClass):
    prompt: History | None = None    # The input conversation history
    response: History | None = None  # The LLM's generated response
    full: History | None = None      # Complete history (prompt + response)
```

### Key Requirements

1. **Policy wrappers MUST set `full`**: Both `TransformersWrapper` and `vLLMWrapper` must populate the `full` attribute after generation
2. **`ChatEnv` reads `full`**: The environment's `_step_history` method uses `chat_history.full` to get the complete conversation
3. **`full` becomes next `prompt`**: The environment sets `new_history = ChatHistory(prompt=full)` for the next step

## Data Flow Diagram

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           LLMCollector Loop                              │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ChatEnv.reset()                                                          │
│   - Creates initial ChatHistory(prompt=initial_history)                  │
│   - Returns TensorDict with ("history",) key                            │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ TransformersWrapper / vLLMWrapper                                        │
│   Input: TensorDict with ("history", "prompt") containing History       │
│                                                                          │
│   Processing (_from_*_generate_history):                                │
│   1. Apply chat template to prompt history                              │
│   2. Generate response tokens                                           │
│   3. Extract response as History objects                                │
│   4. Set ChatHistory attributes:                                        │
│      - history_chat.prompt = input_history                              │
│      - history_chat.response = extracted_responses                      │
│      - history_chat.full = prompt.extend(response)  ← CRITICAL!         │
│                                                                          │
│   Output: TensorDict with ("history",) containing ChatHistory           │
│           with prompt, response, AND full set                           │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│ ChatEnv._step_history()                                                  │
│   Input: TensorDict with action containing ChatHistory                  │
│                                                                          │
│   Processing:                                                           │
│   1. chat_history = tensordict["history"]                               │
│   2. full = chat_history.full  ← Reads the full conversation           │
│   3. new_history = ChatHistory(prompt=full)  ← Full becomes new prompt │
│                                                                          │
│   Output: TensorDict with ("history",) for next step                    │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
                            (Loop continues)
```

## Common Issues

### Issue: `chat_history.full` is None

**Symptom**: `ChatEnv._step_history` fails or produces empty history because `full` was not set.

**Root Cause**: The policy wrapper did not set the `full` attribute on `ChatHistory`.

**Fix**: Ensure the wrapper sets `full` after setting `prompt` and `response`:

```python
# In _from_*_generate_history method:
with history_chat.view(-1) as history_chat_flat:
    history_chat_flat.prompt = input_history
    history_chat_flat.response = extracted_responses
    # CRITICAL: Must set full!
    history_chat_flat.full = history_chat_flat.prompt.extend(
        extracted_responses, inplace=False, dim=-1
    )
```

### Issue: History batch dimensions mismatch

**Symptom**: Warnings about batch dimensions or shape errors.

**Root Cause**: `History` objects inside `ChatHistory` should have one more batch dimension than the `ChatHistory` itself (to handle multi-turn conversations).

**Fix**: The `ChatHistory.__post_init__` method handles this automatically with warnings, but it's better to construct histories with correct dimensions from the start.

## Debugging Tips

1. **Check `full` is set**: After policy forward, verify `tensordict["history"].full is not None`

2. **Inspect batch dimensions**: 
   ```python
   chat_history = tensordict["history"]
   print(f"ChatHistory batch_dims: {chat_history.batch_dims}")
   print(f"prompt batch_dims: {chat_history.prompt.batch_dims}")
   print(f"full batch_dims: {chat_history.full.batch_dims}")
   ```

3. **Verify History content**:
   ```python
   full_history = chat_history.full
   print(f"Roles: {full_history.role}")
   print(f"Content: {full_history.content}")
   ```

## Related Files

- `torchrl/modules/llm/policies/transformers_wrapper.py` - TransformersWrapper implementation
- `torchrl/modules/llm/policies/vllm_wrapper.py` - vLLMWrapper implementation  
- `torchrl/modules/llm/policies/common.py` - ChatHistory class definition
- `torchrl/envs/llm/chat.py` - ChatEnv implementation
- `torchrl/data/llm/history.py` - History class for conversation data
