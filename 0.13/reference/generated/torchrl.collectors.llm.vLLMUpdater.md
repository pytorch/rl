# vLLMUpdater

*class*torchrl.collectors.llm.vLLMUpdater(**args*, *v2=False*, ***kwargs*)[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm.html#vLLMUpdater)

A class that sends weights to vLLM workers.

This class handles synchronizing weights between a training policy and vLLM inference workers.
It supports both local vLLM instances and remote Ray actors.

Parameters:

- **master_address** (*str**,**optional*) - The master address for distributed training. Defaults to localhost.
- **master_port** (*int**,**optional*) - The master port for distributed training. If None, will auto-assign.
- **model_metadata** (*dict**[**str**,**tuple**[*[*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,*[*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*]**]**,**optional*) - Model metadata mapping
parameter names to their dtype and shape. If not provided, will be extracted from policy.
- **vllm_tp_size** (*int**,**optional*) - vLLM tensor parallel size. Defaults to 1.
- **v2** (*bool**,**optional*) - If True, returns a vLLMUpdaterV2 instance instead. This is an experimental
feature that provides better integration with AsyncVLLM engines. When using v2=True, you must
provide a vllm_engine parameter instead of the above parameters. Defaults to False.

init()[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm.html#vLLMUpdater.init)

Initialize the updater with model metadata and initialize the group.

_sync_weights_with_worker()[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm.html#vLLMUpdater._sync_weights_with_worker)

Synchronize weights with a vLLM worker.

_get_server_weights()[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm.html#vLLMUpdater._get_server_weights)

Not used - weights must be passed directly.

_maybe_map_weights()[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm.html#vLLMUpdater._maybe_map_weights)

No mapping needed.

all_worker_ids()[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm.html#vLLMUpdater.all_worker_ids)

Returns [0] since we only have one worker.

Note

This class assumes the policy is a transformers model that can be loaded by vLLM.
The policy must have a state_dict() method that returns the model weights.

Warning

The v2=True option is experimental and may have backward-compatibility breaking changes
in future releases. However, it is generally considered a better option for working with
AsyncVLLM engines and provides improved performance and reliability.

all_worker_ids() → list[int][[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm.html#vLLMUpdater.all_worker_ids)

Returns [0] since we only have one worker.

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

*classmethod*get_model_metadata(*model: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*) → dict[str, tuple[[dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)]][[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm.html#vLLMUpdater.get_model_metadata)

Get the model metadata from a model.

Parameters:

**model** (*TensorDictModuleBase*) - The model to get the metadata from.
Must be a TransformersWrapper or equivalent.

Returns:

The model metadata.

Return type:

dict[str, tuple[[torch.dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)]]

increment_version()

Increment the policy version.

init(*model_metadata: dict[str, tuple[[dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)]]*) → None[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm.html#vLLMUpdater.init)

Initialize the updater with model metadata and initialize the group.

Parameters:

**model_metadata** (*dict**[**str**,**tuple**[*[*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,*[*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*]**]*) - The model metadata mapping
parameter names to their dtype and shape.

*property*post_hooks*: list[Callable[[], None]]*

The list of post-hooks registered to the weight updater.

push_weights(*policy_or_weights: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | dict | None = None*, *worker_ids: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | int | list[int] | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)] | None = None*)

Updates the weights of the policy, or on specified / all remote workers.

Parameters:

- **policy_or_weights** - The source to get weights from. Can be:
- TensorDictModuleBase: A policy module whose weights will be extracted
- TensorDictBase: A TensorDict containing weights
- dict: A regular dict containing weights
- None: Will try to get weights from server using _get_server_weights()
- **worker_ids** - An optional list of workers to update.

Returns: nothing.

register_collector(*collector: [BaseCollector](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)*)[[source]](../../_modules/torchrl/collectors/llm/weight_update/vllm.html#vLLMUpdater.register_collector)

Register a collector in the updater.

Once registered, the updater will not accept another collector.

Parameters:

**collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)) - The collector to register.

register_post_hook(*hook: Callable[[], None]*)

Registers a post-hook to be called after weights are updated.

Parameters:

**hook** (*Callable**[**[**]**,**None**]*) - The post-hook to register.