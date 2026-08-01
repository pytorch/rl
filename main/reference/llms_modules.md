# LLM Wrappers

The LLM wrapper API provides unified interfaces for different LLM backends, ensuring consistent
input/output formats across training and inference pipelines.

## Wrappers

| [`LLMWrapperBase`](generated/torchrl.modules.llm.LLMWrapperBase.html#torchrl.modules.llm.LLMWrapperBase)(*args, **kwargs) | A LLM wrapper base class. |
| --- | --- |
| [`TransformersWrapper`](generated/torchrl.modules.llm.TransformersWrapper.html#torchrl.modules.llm.TransformersWrapper)(*args, **kwargs) | A wrapper class for Hugging Face Transformers models, providing a consistent interface for text generation and log probability computation. |
| [`vLLMWrapper`](generated/torchrl.modules.llm.vLLMWrapper.html#torchrl.modules.llm.vLLMWrapper)(*args, **kwargs) | A wrapper class for vLLM models, providing a consistent interface for text generation and log probability computation. |
| [`SGLangWrapper`](generated/torchrl.modules.llm.SGLangWrapper.html#torchrl.modules.llm.SGLangWrapper)(*args, **kwargs) | A wrapper class for SGLang models, providing a consistent interface for text generation. |
| [`RemoteTransformersWrapper`](generated/torchrl.modules.llm.RemoteTransformersWrapper.html#torchrl.modules.llm.RemoteTransformersWrapper)(model[, ...]) | A remote Ray actor wrapper for TransformersWrapper that provides a simplified interface. |
| [`AsyncVLLM`](generated/torchrl.modules.llm.AsyncVLLM.html#torchrl.modules.llm.AsyncVLLM)(engine_args[, num_replicas, ...]) | A service that manages multiple async vLLM engine actors for distributed inference. |
| [`AsyncSGLang`](generated/torchrl.modules.llm.AsyncSGLang.html#torchrl.modules.llm.AsyncSGLang)([server_url, model_path, ...]) | Server-based SGLang inference service for TorchRL. |

## Data Structure Classes

| [`ChatHistory`](generated/torchrl.modules.llm.ChatHistory.html#torchrl.modules.llm.ChatHistory)([prompt, response, full, ...]) | |
| --- | --- |
| [`Text`](generated/torchrl.modules.llm.Text.html#torchrl.modules.llm.Text)([prompt, response, full, device, names]) | |
| [`LogProbs`](generated/torchrl.modules.llm.LogProbs.html#torchrl.modules.llm.LogProbs)([prompt, response, full, padded, ...]) | |
| [`Masks`](generated/torchrl.modules.llm.Masks.html#torchrl.modules.llm.Masks)([all_attention_mask, ...]) | |
| [`Tokens`](generated/torchrl.modules.llm.Tokens.html#torchrl.modules.llm.Tokens)([prompt, response, full, padded, ...]) | |

## Utilities

| [`make_async_vllm_engine`](generated/torchrl.modules.llm.make_async_vllm_engine.html#torchrl.modules.llm.make_async_vllm_engine)(*, model_name[, ...]) | Create an async vLLM engine service. |
| --- | --- |
| [`stateless_init_process_group_async`](generated/torchrl.modules.llm.stateless_init_process_group_async.html#torchrl.modules.llm.stateless_init_process_group_async)(...) | Initializes a stateless process group for distributed communication (async version). |
| [`make_vllm_worker`](generated/torchrl.modules.llm.make_vllm_worker.html#torchrl.modules.llm.make_vllm_worker)(*, model_name[, devices, ...]) | Creates a vLLM inference engine with tensor parallelism support. |
| [`stateless_init_process_group`](generated/torchrl.modules.llm.stateless_init_process_group.html#torchrl.modules.llm.stateless_init_process_group)(master_address, ...) | Initializes a stateless process group for distributed communication. |