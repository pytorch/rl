# as_nested_tensor

*class*torchrl.envs.llm.transforms.as_nested_tensor(*list_of_tensordicts: list[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*)[[source]](../../_modules/torchrl/envs/llm/transforms/dataloading.html#as_nested_tensor)

Stacks a list of tensordicts into a single tensordict with nested tensors.

Parameters:

**list_of_tensordicts** (*list**[**TensorDictBase**]*) - A list of tensordicts to stack.

Returns:

A tensordict with nested tensors.

Return type:

TensorDictBase