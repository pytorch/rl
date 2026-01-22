# LLM Test Issues Tracker

This document tracks known issues with the LLM tests that need to be addressed in follow-up PRs.

## Issues Fixed in This PR

### 1. vLLM API Changes (vLLM 0.14+)
- **Files**: `torchrl/modules/llm/policies/vllm_wrapper.py`, `torchrl/modules/llm/backends/vllm/vllm_async.py`
- **Issue**: vLLM 0.14+ changed APIs:
  - `TokensPrompt` moved to `vllm.inputs`
  - `CompilationConfig` now uses `level` instead of `enabled`
- **Status**: Fixed

### 2. Python 3.12 Typing Issues
- **Files**: `torchrl/modules/llm/policies/vllm_wrapper.py`
- **Issue**: `SamplingParams` import from `typing.Any` fails on Python 3.12
- **Status**: Fixed

### 3. Model Selection for Chat Tests
- **Files**: `test/llm/test_llm_collectors.py`
- **Issue**: Base models (gpt2, opt-125m) don't produce chat-formatted output. Need chat-trained models like Qwen.
- **Status**: Fixed - switched to `Qwen/Qwen2.5-0.5B`

### 4. Generation Parameters
- **Files**: `test/llm/test_llm_collectors.py`
- **Issue**: Without `ignore_eos=True` and `max_new_tokens`, models may generate zero tokens
- **Status**: Fixed

### 5. AsyncEnvPool Spec Propagation
- **Files**: `torchrl/envs/async_envs.py`
- **Issue**: AsyncEnvPool was not properly propagating batch_size from child environments
- **Status**: Fixed in main (merged)

### 6. vLLM logprobs type error
- **Issue**: `msgspec.ValidationError: Expected `int | null`, got `bool` - at `$[3].logprobs``
- **Root Cause**: TorchRL was passing boolean to vLLM's `logprobs` parameter, but vLLM expects `int | None`
- **Fix Applied**: Convert bool to int (1 if True, None if False) in vllm_wrapper.py
- **Status**: Fixed

### 7. Ray + vLLM v1 "bundles" KeyError
- **Issue**: `KeyError: 'bundles'` during Ray async engine initialization
- **Root Cause**: vLLM subprocess starts new Ray instance instead of connecting to parent's cluster
- **Reference**: https://github.com/vllm-project/vllm/issues/19123 (fixed in PR #21540)
- **Fix Applied**: 
  - Set `RAY_ADDRESS` to current Ray GCS address in `vllm_async.py` before creating AsyncLLMEngine
  - Removed `VLLM_USE_V1=0` from tutorials (we need v1!)
- **Status**: Fixed

### 8. vLLM best_of parameter removed
- **Issue**: `TypeError: Unexpected keyword argument 'best_of'`
- **Root Cause**: vLLM removed `best_of` parameter from SamplingParams
- **Fix Applied**: Skip `num_beams` parameter mapping in vllm_wrapper.py
- **Status**: Fixed

### 9. AsyncEnvPool + LLMCollector yield_completed_trajectories
- **Files**: `torchrl/envs/async_envs.py`, `torchrl/collectors/llm/base.py`
- **Test**: `test_llm_collector_completed_async`
- **Issue**: 
  - `AsyncEnvPool` now correctly reports `batch_size=[num_envs, *child_batch_size]` (e.g., `[4, 1]`)
  - `LLMCollector.yield_completed_trajectories=True` was checking for single batch dimension
  - `_sort_results` was returning nested lists instead of scalar indices
  - Trajectory results had extra batch dimension from child envs
- **Fix Applied**:
  - `async_envs._sort_results`: Unwrap single-element sequences to get scalar indices
  - `base.py._rollout_yield_trajs_async`: Flatten env_ids to handle nested lists
  - Use 1D dones tensor (only tracking by env_id, not child batch)
  - Apply `view(-1)` to flatten trajectory results
- **Status**: Fixed

### 10. Gated HuggingFace Models
- **Files**: `test/llm/test_data.py`
- **Test**: `test_history_assistant_mask_llama`
- **Issue**: Tests using Llama tokenizer fail because model is gated on HuggingFace
- **Fix Applied**: Created a mock Llama tokenizer fixture using GPT-2 as base with Llama 3 special tokens added
  - Test now runs without requiring access to gated models
  - Mock tokenizer includes `<|begin_of_text|>`, `<|header_start|>`, `<|header_end|>`, `<|eot|>` tokens
  - Properly tests the Llama chat template parsing API
- **Status**: Fixed

### 11. TransformersWrapper History Output in Collector
- **Files**: `torchrl/modules/llm/policies/transformers_wrapper.py`
- **Test**: `test_llm_collector_with_transformers`
- **Issue**: TransformersWrapper history output not populated correctly in collector context
  - The `("next", "history", "prompt")` key was empty when using TransformersWrapper
  - Root cause: `_from_transformers_generate_history` was not setting `ChatHistory.full`
  - `ChatEnv._step_history` reads `chat_history.full` to get the complete conversation
- **Fix Applied**: Added missing `history_chat_flat.full = history_chat_flat.prompt.extend(h_responses, inplace=False, dim=-1)`
- **Documentation**: Added `test/llm/TRANSFORMERS_CHATENV_INTEGRATION.md` explaining the data flow
- **Status**: Fixed

## Issues Requiring Follow-up PRs

(No remaining issues)

## Test Configuration Changes

### pytest-timeout (5 minutes per test)
- Added `pytest-timeout` dependency
- Default 300s (5 min) timeout applied
- Prevents indefinite hangs

### RAY_ADDRESS in vLLM async code
- Set RAY_ADDRESS to current Ray GCS address before creating AsyncLLMEngine
- Ensures vLLM subprocess connects to same Ray cluster as parent
- Fixes vLLM v1 "bundles" KeyError issue

### Removed --isolate
- Too slow - each test in subprocess adds huge overhead
- CI took 1+ hours instead of ~10 minutes

### Removed --exitfirst
- Allows all tests to run and collect all failures
- Better for identifying all issues in a single CI run

### Removed --error-for-skips
- Many LLM tests use pytest.skip for optional dependencies
- This flag was causing legitimate skips to be errors

### conftest.py Cleanup Fixtures
- Session-scoped Ray shutdown after all tests
- Session-scoped GPU memory cleanup
- Function-scoped `ray_session` fixture for tests needing guaranteed Ray cleanup

## Test Cleanup Best Practices

### Ray Shutdown
- Use `ray_session` fixture for guaranteed cleanup
- Always wrap Ray tests in try/finally with `ray.shutdown()`
- Session-scoped autouse fixture handles cleanup if test crashes

### vLLM GPU Memory
- Use `gc.collect()` and `torch.cuda.empty_cache()` after vLLM tests
- Consider explicit engine shutdown when possible
- Module-scoped fixtures reuse engines to minimize memory churn

### Subprocess/Resource Cleanup
- pytest-isolate runs each test in subprocess for clean isolation
- 5-minute timeout prevents indefinite hangs
- try/finally blocks ensure cleanup runs even on failure
