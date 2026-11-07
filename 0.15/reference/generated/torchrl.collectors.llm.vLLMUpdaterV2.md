# vLLMUpdaterV2

*class*torchrl.collectors.llm.vLLMUpdaterV2(*vllm_engine: RLvLLMEngine*)[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm_v2.html#vLLMUpdaterV2)

Simplified vLLM weight updater using the RLvLLMEngine interface.

This updater works with any vLLM engine that implements the RLvLLMEngine
interface, automatically extracting configuration and handling weight updates
through the engine's own methods.

Parameters:

**vllm_engine** - A vLLM engine implementing the RLvLLMEngine interface.

Note

This class can be created through [`torchrl.collectors.llm.vLLMUpdater`](torchrl.collectors.llm.vLLMUpdater.html#torchrl.collectors.llm.vLLMUpdater) with v2=True.

all_worker_ids()[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm_v2.html#vLLMUpdaterV2.all_worker_ids)

Return list of worker IDs.

*property*collector*: Any | None*

The collector or container of the receiver.

Returns None if the container is out-of-scope or not set.

*property*collectors*: list[Any] | None*

The collectors or container of the receiver.

*classmethod*from_policy(*policy: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*) → [WeightUpdaterBase](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase) | None

Optional classmethod to create a weight updater instance from a policy.

This method can be implemented by subclasses to provide custom initialization logic
based on the policy. If implemented, this method will be called before falling back
to the default constructor when initializing a weight updater in a collector.

Parameters:

**policy** (*TensorDictModuleBase*) - The policy to create the weight updater from.

Returns:

An instance of the weight updater, or None if the policy

cannot be used to create an instance.

Return type:

[WeightUpdaterBase](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase) | None

*classmethod*get_model_metadata(*model*) → dict[str, tuple[[dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)]][[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm_v2.html#vLLMUpdaterV2.get_model_metadata)

Get model metadata from a model.

Parameters:

**model** - A model with state_dict() method (e.g., TransformersWrapper)

Returns:

Mapping of parameter names to (dtype, shape) tuples

Return type:

dict

get_tp_size() → int[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm_v2.html#vLLMUpdaterV2.get_tp_size)

Get the tensor parallel size.

increment_version()

Increment the policy version.

init(*model_metadata: dict[str, tuple[[dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)]] | None = None*) → None[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm_v2.html#vLLMUpdaterV2.init)

Initialize the weight updater.

Parameters:

**model_metadata** - Optional model metadata. If not provided, uses engine's metadata.

*property*post_hooks*: list[Callable[[], None]]*

The list of post-hooks registered to the weight updater.

push_weights(*weights: Iterator[tuple[str, [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]] | [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*)[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm_v2.html#vLLMUpdaterV2.push_weights)

Push weights to the vLLM engine.

Parameters:

**weights** - Either an iterator of (name, tensor) pairs or a TensorDictBase

push_weights_from_transformers(*transformers_model*)[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm_v2.html#vLLMUpdaterV2.push_weights_from_transformers)

Push weights from a transformers model.

Parameters:

**transformers_model** - A transformers PreTrainedModel or TorchRL wrapper

push_weights_from_transformers_optimized(*transformers_model*, *batch_size=50*)[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm_v2.html#vLLMUpdaterV2.push_weights_from_transformers_optimized)

Optimized version of push_weights_from_transformers with GPU pre-loading.

This method provides several optimizations:
1. Pre-loads all weights to GPU before transfer
2. Optionally batches weights for better memory management
3. Uses non-blocking transfers when possible

Parameters:

- **transformers_model** - A transformers PreTrainedModel or TorchRL wrapper
- **batch_size** - Number of weights to transfer in each batch (0 = no batching)

register_collector(*collector*)[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm_v2.html#vLLMUpdaterV2.register_collector)

Register a collector and set up policy version increment post-hook.

Parameters:

**collector** - The collector to register (BaseCollector)

register_post_hook(*hook: Callable[[], None]*)

Registers a post-hook to be called after weights are updated.

Parameters:

**hook** (*Callable**[**[**]**,**None**]*) - The post-hook to register.