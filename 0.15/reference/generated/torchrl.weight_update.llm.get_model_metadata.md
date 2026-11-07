# get_model_metadata

*class*torchrl.weight_update.llm.get_model_metadata(*model*)[[source]](../../_modules/torchrl/weight_update/llm/vllm_nccl.html#get_model_metadata)

Extract model metadata from a model.

Parameters:

**model** - A model with state_dict() or a model wrapper.

Returns:

Dict mapping parameter names to (dtype, shape) tuples.

Note

This function must extract keys in the same format as WeightStrategy.extract_weights()
to ensure consistency between metadata and actual weight keys during broadcasting.