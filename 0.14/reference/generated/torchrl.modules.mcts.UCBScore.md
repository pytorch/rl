# torchrl.modules.mcts.UCBScore

*class*torchrl.modules.mcts.UCBScore(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/mcts/scores.html#UCBScore)

Computes the UCB (Upper Confidence Bound) score, specifically UCB1, for MCTS.

UCB1 is a classic algorithm for the multi-armed bandit problem that balances
exploration and exploitation. In MCTS, it's used to select which action to
explore from a given node. The score encourages trying actions with high
empirical rewards and actions that have been visited less frequently.

The formula used is:
score = (win_count / visits) + c * sqrt(total_visits) / (1 + visits)

Parameters:

- **c** (*float*) - The exploration constant. A common value is sqrt(2).
- **win_count_key** (*NestedKey**,**optional*) - Key for the tensor in the input TensorDictBase
containing the sum of rewards (or win counts) for each action.
Defaults to "win_count".
- **visits_key** (*NestedKey**,**optional*) - Key for the tensor containing the visit
count for each action. Defaults to "visits".
- **total_visits_key** (*NestedKey**,**optional*) - Key for the tensor (or scalar)
representing the visit count of the parent node (N). This is used in the
exploration term. Defaults to "total_visits".
- **score_key** (*NestedKey**,**optional*) - Key where the calculated UCB scores
will be stored in the output TensorDictBase. Defaults to "score".

Input Keys:

- win_count_key (torch.Tensor): Tensor of shape (..., num_actions).
- visits_key (torch.Tensor): Tensor of shape (..., num_actions).
- total_visits_key (torch.Tensor): Scalar or tensor broadcastable to other inputs.

Output Keys:

- score_key (torch.Tensor): Tensor of the same shape as visits_key, containing
the calculated UCB scores.

Example

```python
from tensordict import TensorDict
from torchrl.modules.mcts.scores import UCBScore

# Create a UCBScore instance
ucb = UCBScore(c=1.414)

# Define a TensorDict with required keys
node = TensorDict(

> {
> 
> "win_count": torch.tensor([15.0, 25.0]),
> "visits": torch.tensor([10.0, 20.0]),
> "total_visits": torch.tensor(100.0),
> 
> 
> 
> 
> 
> 
> },
> batch_size=[],

)

# Compute the UCB scores
result = ucb(node)
print(result["score"]) # Output: Tensor with UCB scores
```

__init__(***, *c: float*, *win_count_key: NestedKey = 'win_count'*, *visits_key: NestedKey = 'visits'*, *total_visits_key: NestedKey = 'total_visits'*, *score_key: NestedKey = 'score'*)[[source]](../../_modules/torchrl/modules/mcts/scores.html#UCBScore.__init__)

Initialize internal Module state, shared by both nn.Module and ScriptModule.

Methods

| `__init__`(*, c[, win_count_key, visits_key, ...]) | Initialize internal Module state, shared by both nn.Module and ScriptModule. |
| --- | --- |
| `add_module`(name, module) | Add a child module to the current module. |
| `apply`(fn) | Apply `fn` recursively to every submodule (as returned by `.children()`) as well as self. |
| `bfloat16`() | Casts all floating point parameters and buffers to `bfloat16` datatype. |
| `buffers`([recurse]) | Return an iterator over module buffers. |
| `children`() | Return an iterator over immediate children modules. |
| `compile`(*args, **kwargs) | Compile this Module's forward using [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile). |
| `cpu`() | Move all model parameters and buffers to the CPU. |
| `cuda`([device]) | Move all model parameters and buffers to the GPU. |
| `double`() | Casts all floating point parameters and buffers to `double` datatype. |
| `eval`() | Set the module in evaluation mode. |
| `extra_repr`() | Return the extra representation of the module. |
| `float`() | Casts all floating point parameters and buffers to `float` datatype. |
| `forward`(node) | Define the computation performed at every call. |
| `get_buffer`(target) | Return the buffer given by `target` if it exists, otherwise throw an error. |
| `get_extra_state`() | Return any extra state to include in the module's state_dict. |
| `get_parameter`(target) | Return the parameter given by `target` if it exists, otherwise throw an error. |
| `get_submodule`(target) | Return the submodule given by `target` if it exists, otherwise throw an error. |
| `half`() | Casts all floating point parameters and buffers to `half` datatype. |
| `ipu`([device]) | Move all model parameters and buffers to the IPU. |
| `is_tdmodule_compatible`(module) | Checks if a module is compatible with TensorDictModule API. |
| `load_state_dict`(state_dict[, strict, assign]) | Copy parameters and buffers from `state_dict` into this module and its descendants. |
| `modules`([remove_duplicate]) | Return an iterator over all modules in the network. |
| `mtia`([device]) | Move all model parameters and buffers to the MTIA. |
| `named_buffers`([prefix, recurse, ...]) | Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself. |
| `named_children`() | Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself. |
| `named_modules`([memo, prefix, remove_duplicate]) | Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself. |
| `named_parameters`([prefix, recurse, ...]) | Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself. |
| `parameters`([recurse]) | Return an iterator over module parameters. |
| `register_backward_hook`(hook) | Register a backward hook on the module. |
| `register_buffer`(name, tensor[, persistent]) | Add a buffer to the module. |
| `register_forward_hook`(hook, *[, prepend, ...]) | Register a forward hook on the module. |
| `register_forward_pre_hook`(hook, *[, ...]) | Register a forward pre-hook on the module. |
| `register_full_backward_hook`(hook[, prepend]) | Register a backward hook on the module. |
| `register_full_backward_pre_hook`(hook[, prepend]) | Register a backward pre-hook on the module. |
| `register_load_state_dict_post_hook`(hook) | Register a post-hook to be run after module's `load_state_dict()` is called. |
| `register_load_state_dict_pre_hook`(hook) | Register a pre-hook to be run before module's `load_state_dict()` is called. |
| `register_module`(name, module) | Alias for `add_module()`. |
| `register_parameter`(name, param) | Add a parameter to the module. |
| `register_state_dict_post_hook`(hook) | Register a post-hook for the [`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) method. |
| `register_state_dict_pre_hook`(hook) | Register a pre-hook for the [`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) method. |
| `requires_grad_`([requires_grad]) | Change if autograd should record operations on parameters in this module. |
| `reset_out_keys`() | Resets the `out_keys` attribute to its orignal value. |
| `reset_parameters_recursive`([parameters]) | Recursively reset the parameters of the module and its children. |
| `select_out_keys`(*out_keys) | Selects the keys that will be found in the output tensordict. |
| `set_extra_state`(state) | Set extra state contained in the loaded state_dict. |
| `set_submodule`(target, module[, strict]) | Set the submodule given by `target` if it exists, otherwise throw an error. |
| `share_memory`() | See [`torch.Tensor.share_memory_()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.share_memory_.html#torch.Tensor.share_memory_). |
| `state_dict`(*args[, destination, prefix, ...]) | Return a dictionary containing references to the whole state of the module. |
| `to`(*args, **kwargs) | Move and/or cast the parameters and buffers. |
| `to_empty`(*, device[, recurse]) | Move the parameters and buffers to the specified device without copying storage. |
| `train`([mode]) | Set the module in training mode. |
| `type`(dst_type) | Casts all parameters and buffers to `dst_type`. |
| `xpu`([device]) | Move all model parameters and buffers to the XPU. |
| `zero_grad`([set_to_none]) | Reset gradients of all model parameters. |

Attributes

| `T_destination` | |
| --- | --- |
| `call_super_init` | |
| `dump_patches` | |
| `in_keys` | |
| `out_keys` | |
| `out_keys_source` | |
| `c` | |
| `training` | |