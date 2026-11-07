# as_padded_tensor

*class*torchrl.envs.llm.transforms.as_padded_tensor(*list_of_tensordicts: list[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, *dim=0*, *stack_dim: int = 0*)[[source]](../../_modules/torchrl/envs/llm/transforms/dataloading.html#as_padded_tensor)

Stacks a list of tensordicts into a single tensordict with padded tensors.

Parameters:

- **list_of_tensordicts** (*list**[**[**TensorDictBase**]**]*) - A list of tensordicts to stack.
- **dim** (*int**,**optional*) - The dimension along which to pad. Defaults to 0.
- **stack_dim** (*int**,**optional*) - The dimension along which to stack. Defaults to 0.

Returns:

A tensordict with padded tensors.

Return type:

TensorDictBase