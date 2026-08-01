# RemoteTransformersWrapper

*class*torchrl.modules.llm.RemoteTransformersWrapper(*model*, *max_concurrency: int = 16*, *validate_model: bool = True*, *actor_name: str | None = None*, *num_gpus: int = 1*, *num_cpus: int = 1*, ***kwargs*)[[source]](../../_modules/torchrl/modules/llm/policies/transformers_wrapper.html#RemoteTransformersWrapper)

A remote Ray actor wrapper for TransformersWrapper that provides a simplified interface.

This class wraps a TransformersWrapper instance as a Ray actor, allowing remote execution
while providing a clean interface that doesn't require explicit remote() and get() calls.

Parameters:

- **model** (*str*) - The Hugging Face Transformers model to wrap.
Must be a string (model name or path) that will be passed to transformers.AutoModelForCausalLM.from_pretrained.
Transformers models are not serializable, so only model names/paths are supported.
- **max_concurrency** (*int**,**optional*) - Maximum number of concurrent calls to the remote actor. Defaults to 16.
- **validate_model** (*bool**,**optional*) - Whether to validate the model. Defaults to True.
- **num_gpus** (*int**,**optional*) - Number of GPUs to use. Defaults to 0.
- **num_cpus** (*int**,**optional*) - Number of CPUs to use. Defaults to 0.
- ****kwargs** - All other arguments are passed directly to TransformersWrapper.

Example

```
>>> import ray
>>> from torchrl.modules.llm.policies import RemoteTransformersWrapper
>>>
>>> # Initialize Ray if not already done
>>> if not ray.is_initialized():
... ray.init()
>>>
>>> # Create remote wrapper
>>> remote_wrapper = RemoteTransformersWrapper(
... model="gpt2",
... input_mode="history",
... generate=True,
... generate_kwargs={"max_new_tokens": 50}
... )
>>>
>>> # Use like a regular wrapper (no remote/get calls needed)
>>> result = remote_wrapper(tensordict_input)
>>> print(result["text"].response)
```

*property*batching

Whether batching is enabled.

cleanup_batching()[[source]](../../_modules/torchrl/modules/llm/policies/transformers_wrapper.html#RemoteTransformersWrapper.cleanup_batching)

Clean up batching resources.

*property*collector

The collector associated with the module.

*property*device

The device used for computation.

*property*dist_params_keys

The keys for distribution parameters.

*property*dist_sample_keys

The keys for distribution samples.

*property*generate

Whether text generation is enabled.

get_batching_state()[[source]](../../_modules/torchrl/modules/llm/policies/transformers_wrapper.html#RemoteTransformersWrapper.get_batching_state)

Get the current batching state.

get_dist(*tensordict*, ***kwargs*)[[source]](../../_modules/torchrl/modules/llm/policies/transformers_wrapper.html#RemoteTransformersWrapper.get_dist)

Get distribution from logits/log-probs with optional masking.

get_dist_with_prompt_mask(*tensordict*, ***kwargs*)[[source]](../../_modules/torchrl/modules/llm/policies/transformers_wrapper.html#RemoteTransformersWrapper.get_dist_with_prompt_mask)

Get distribution masked to only include response tokens (exclude prompt).

get_new_version(***kwargs*)[[source]](../../_modules/torchrl/modules/llm/policies/transformers_wrapper.html#RemoteTransformersWrapper.get_new_version)

Get a new version of the wrapper with altered parameters.

*property*in_keys

The input keys.

*property*inplace

Whether in-place operations are used.

*property*layout

The layout used for output tensors.

log_prob(*data*, ***kwargs*)[[source]](../../_modules/torchrl/modules/llm/policies/transformers_wrapper.html#RemoteTransformersWrapper.log_prob)

Compute log probabilities.

*property*log_prob_keys

The keys for log probabilities.

*property*log_probs_key

The key for log probabilities output.

*property*masks_key

The key for masks output.

*property*num_samples

The number of samples to generate.

*property*out_keys

The output keys.

*property*pad_output

Whether output sequences are padded.

*property*text_key

The key for text output.

*property*tokens_key

The key for tokens output.