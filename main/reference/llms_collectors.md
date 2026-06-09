# LLM Collectors

Specialized collector classes for LLM use cases.

| [`LLMCollector`](generated/torchrl.collectors.llm.LLMCollector.html#torchrl.collectors.llm.LLMCollector)(env, *[, policy, ...]) | A simplified version of Collector for LLM inference. |
| --- | --- |
| [`RayLLMCollector`](generated/torchrl.collectors.llm.RayLLMCollector.html#torchrl.collectors.llm.RayLLMCollector)(env, *[, policy, ...]) | A lightweight Ray implementation of the LLM Collector that can be extended and sampled remotely. |
| [`vLLMUpdater`](generated/torchrl.collectors.llm.vLLMUpdater.html#torchrl.collectors.llm.vLLMUpdater)(*args[, v2]) | A class that sends weights to vLLM workers. |
| [`vLLMUpdaterV2`](generated/torchrl.collectors.llm.vLLMUpdaterV2.html#torchrl.collectors.llm.vLLMUpdaterV2)(vllm_engine) | Simplified vLLM weight updater using the RLvLLMEngine interface. |

## Weight Synchronization Schemes

| [`VLLMWeightSyncScheme`](generated/torchrl.weight_update.llm.VLLMWeightSyncScheme.html#torchrl.weight_update.llm.VLLMWeightSyncScheme)([master_address, ...]) | Weight synchronization scheme for vLLM engines. |
| --- | --- |
| [`VLLMWeightSender`](generated/torchrl.weight_update.llm.VLLMWeightSender.html#torchrl.weight_update.llm.VLLMWeightSender)(scheme) | Sends weights to vLLM workers using collective communication. |
| [`VLLMWeightReceiver`](generated/torchrl.weight_update.llm.VLLMWeightReceiver.html#torchrl.weight_update.llm.VLLMWeightReceiver)(scheme, vllm_engine) | Receives weights in a vLLM worker using collective communication. |
| [`VLLMCollectiveTransport`](generated/torchrl.weight_update.llm.VLLMCollectiveTransport.html#torchrl.weight_update.llm.VLLMCollectiveTransport)(master_address, ...) | Transport for vLLM using vLLM's native WeightTransferConfig API (vLLM 0.17+). |
| [`VLLMDoubleBufferSyncScheme`](generated/torchrl.weight_update.llm.VLLMDoubleBufferSyncScheme.html#torchrl.weight_update.llm.VLLMDoubleBufferSyncScheme)(remote_addr[, ...]) | Weight synchronization scheme for vLLM using double-buffered storage. |
| [`VLLMDoubleBufferWeightSender`](generated/torchrl.weight_update.llm.VLLMDoubleBufferWeightSender.html#torchrl.weight_update.llm.VLLMDoubleBufferWeightSender)(scheme) | Sends weights to vLLM workers using double-buffered storage. |
| [`VLLMDoubleBufferWeightReceiver`](generated/torchrl.weight_update.llm.VLLMDoubleBufferWeightReceiver.html#torchrl.weight_update.llm.VLLMDoubleBufferWeightReceiver)(scheme, ...) | Receives weights in a vLLM worker using double-buffered storage. |
| [`VLLMDoubleBufferTransport`](generated/torchrl.weight_update.llm.VLLMDoubleBufferTransport.html#torchrl.weight_update.llm.VLLMDoubleBufferTransport)(remote_addr[, ...]) | Transport for vLLM using double-buffered memory-mapped storage. |
| [`get_model_metadata`](generated/torchrl.weight_update.llm.get_model_metadata.html#torchrl.weight_update.llm.get_model_metadata)(model) | Extract model metadata from a model. |