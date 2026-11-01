# torchrl.modules.mcts.UCB1TunedScore

*class*torchrl.modules.mcts.UCB1TunedScore(**args*, ***kwargs*)[[source]](../../_modules/torchrl/modules/mcts/scores.html#UCB1TunedScore)

Computes the UCB1-Tuned score for MCTS, using variance estimation.

UCB1-Tuned is an enhancement of the UCB1 algorithm that incorporates an estimate
of the variance of rewards for each action. This allows for a more refined
balance between exploration and exploitation, potentially leading to better
performance, especially when reward variances differ significantly across actions.

The score for an action i is calculated as:
score_i = avg_reward_i + sqrt(log(N) / N_i * min(0.25, V_i))

The variance estimate V_i for action i is calculated as:
V_i = (sum_squared_rewards_i / N_i) - avg_reward_i^2 + sqrt(exploration_constant * log(N) / N_i)

Where:
- avg_reward_i: Average reward obtained from action i.
- N_i: Number of times action i has been visited.
- N: Total number of times the parent node has been visited.
- sum_squared_rewards_i: Sum of the squares of rewards received from action i.
- exploration_constant: A constant used in the bias correction term of V_i.

> Auer et al. (2002) suggest a value of 2.0 for rewards in the range [0,1].

- The term min(0.25, V_i) implies that rewards are scaled to [0, 1], as 0.25 is
the maximum variance for a distribution in this range (e.g., Bernoulli(0.5)).

Reference: "Finite-time Analysis of the Multiarmed Bandit Problem"
(Auer, Cesa-Bianchi, Fischer, 2002).

Parameters:

- **exploration_constant** (*float**,**optional*) - The constant C used in the bias
correction term for the variance estimate V_i. Defaults to 2.0,
as suggested for rewards in [0,1].
- **win_count_key** (*NestedKey**,**optional*) - Key for the tensor in the input TensorDictBase
containing the sum of rewards for each action (Q_i * N_i). Defaults to "win_count".
- **visits_key** (*NestedKey**,**optional*) - Key for the tensor containing the visit
count for each action (N_i). Defaults to "visits".
- **total_visits_key** (*NestedKey**,**optional*) - Key for the tensor (or scalar)
representing the visit count of the parent node (N). Defaults to "total_visits".
- **sum_squared_rewards_key** (*NestedKey**,**optional*) - Key for the tensor containing
the sum of squared rewards received for each action. This is crucial for
calculating the empirical variance. Defaults to "sum_squared_rewards".
- **score_key** (*NestedKey**,**optional*) - Key where the calculated UCB1-Tuned scores
will be stored in the output TensorDictBase. Defaults to "score".

Input Keys:

- win_count_key (torch.Tensor): Sum of rewards for each action.
- visits_key (torch.Tensor): Visit counts for each action (N_i).
- total_visits_key (torch.Tensor): Parent node's visit count (N).
- sum_squared_rewards_key (torch.Tensor): Sum of squared rewards for each action.

Output Keys:

- score_key (torch.Tensor): Calculated UCB1-Tuned scores for each action.

Important Notes:

- **Unvisited Nodes**: Actions with zero visits (visits_key is 0) are assigned a
very large positive score to ensure they are selected for exploration.
- **Reward Range**: The min(0.25, V_i) term is theoretically most sound when
rewards are normalized to the range [0, 1].
- **Logarithm of N**: log(N) (log of parent visits) is calculated using torch.log(torch.clamp(N, min=1.0))
to prevent issues with N=0 or N between 0 and 1.

__init__(***, *win_count_key: NestedKey = 'win_count'*, *visits_key: NestedKey = 'visits'*, *total_visits_key: NestedKey = 'total_visits'*, *sum_squared_rewards_key: NestedKey = 'sum_squared_rewards'*, *score_key: NestedKey = 'score'*, *exploration_constant: float = 2.0*)[[source]](../../_modules/torchrl/modules/mcts/scores.html#UCB1TunedScore.__init__)

Initialize internal Module state, shared by both nn.Module and ScriptModule.

Methods

| `__init__`(*[, win_count_key, visits_key, ...]) | Initialize internal Module state, shared by both nn.Module and ScriptModule. |
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
| `training` | |