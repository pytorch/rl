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

## Issues Requiring Follow-up PRs

### 1. AsyncEnvPool + LLMCollector yield_completed_trajectories
- **Files**: `torchrl/envs/async_envs.py`, `torchrl/collectors/llm/base.py`
- **Test**: `test_llm_collector_completed_async`
- **Issue**: 
  - `AsyncEnvPool` now correctly reports `batch_size=[num_envs, *child_batch_size]` (e.g., `[4, 1]`)
  - `LLMCollector.yield_completed_trajectories=True` requires single batch dimension
  - Proper fix needs LLMCollector to handle multi-dim batch sizes
- **Workaround**: Test marked as xfail with `strict=False` (will pass if fixed)
- **Priority**: Medium

### 2. vLLM Ray Async Backend - Engine Initialization
- **Files**: `torchrl/modules/llm/backends/vllm/vllm_async.py`
- **Test**: `test_chat_env_integration_ifeval`
- **Issue**: 
  - `KeyError: 'bundles'` during Ray async engine initialization
  - Appears to be vLLM 0.14+ / Ray compatibility issue
  - Error in `vllm/v1/engine/utils.py:wait_for_engine_startup`
- **Workaround**: Test marked as xfail with `strict=False`
- **Priority**: High - affects async vLLM usage

### 3. TransformersWrapper History Output in Collector
- **Test**: `test_llm_collector_with_transformers`
- **Issue**: TransformersWrapper history output not populated correctly in collector context
- **Workaround**: Test marked as xfail
- **Priority**: Medium

### 4. Gated HuggingFace Models
- **Files**: `test/llm/test_data.py`
- **Issue**: Tests using Llama tokenizer fail because model is gated
- **Workaround**: Tests marked as xfail
- **Priority**: Low - expected behavior for gated models

## Test Configuration Changes

### pytest-timeout (5 minutes per test)
- Added `pytest-timeout` dependency
- Default 300s (5 min) timeout applied via conftest.py
- Prevents indefinite hangs

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
- 5-minute timeout prevents indefinite hangs
- try/finally blocks ensure cleanup runs even on failure
