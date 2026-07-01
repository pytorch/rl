# WeightUpdaterBase

*class*torchrl.collectors.WeightUpdaterBase[[source]](../../_modules/torchrl/collectors/weight_update.html#WeightUpdaterBase)

A base class for updating remote policy weights on inference workers.

Deprecated since version WeightUpdaterBase: is deprecated and will be removed in a future version.
Please use WeightSyncScheme from torchrl.weight_update.weight_sync_schemes instead.

The weight updater is the central piece of the weight update scheme:

- In leaf collector nodes, it is responsible for sending the weights to the policy, which can be as simple as
updating a state-dict, or more complex if an inference server is being used.
- In server collector nodes, it is responsible for sending the weights to the leaf collectors.

In a collector, the updater is called within `update_policy_weights_()`.`

The main method of this class is the `_push_weights()` method, which updates the policy weights in the worker /
policy. This method is called by `push_weights()`, which also calls the post-hooks: only _push_weights should
be implemented by child classes.

To extend this class, implement the following abstract methods:

- _get_server_weights (optional): Define how to retrieve the weights from the server if they are not passed to

the updater directly. This method is only called if the weights (handle) is not passed directly.
- _sync_weights_with_worker: Define how to synchronize weights with a specific worker.

This method must be implemented by child classes.
- _maybe_map_weights: Optionally transform the server weights before distribution.

By default, this method returns the weights unchanged.
- all_worker_ids: Provide a list of all worker identifiers.

Returns None by default (no worker id).
- from_policy (optional classmethod): Define how to create an instance of the weight updater from a policy.

If implemented, this method will be called before falling back to the default constructor when initializing
a weight updater in a collector.

Variables:

**collector** - The collector (or any container) of the weight receiver. The collector is registered via
`register_collector()`.

push_weights()[[source]](../../_modules/torchrl/collectors/weight_update.html#WeightUpdaterBase.push_weights)

Updates the weights on specified or all remote workers.
The __call__ method is a proxy to push_weights.

register_collector()[[source]](../../_modules/torchrl/collectors/weight_update.html#WeightUpdaterBase.register_collector)

Registers the collector (or any container) in the receiver through a weakref.
This will be called automatically by the collector upon registration of the updater.

from_policy()[[source]](../../_modules/torchrl/collectors/weight_update.html#WeightUpdaterBase.from_policy)

Optional classmethod to create an instance from a policy.

Post-hooks:

- register_post_hook: Registers a post-hook to be called after the weights are updated.

The post-hook must be a callable that takes no arguments.
The post-hook will be called after the weights are updated.
The post-hook will be called in the same process as the weight updater.
The post-hook will be called in the same order as the post-hooks were registered.

See also

[`update_policy_weights_()`](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector.update_policy_weights_).

all_worker_ids() → list[int] | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)] | None[[source]](../../_modules/torchrl/collectors/weight_update.html#WeightUpdaterBase.all_worker_ids)

Gets list of all worker IDs.

Returns None by default. Subclasses should override to return actual worker IDs.

Returns:

List of worker IDs or None.

Return type:

list[int] | list[[torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)] | None

*property*collector*: Any | None*

The collector or container of the receiver.

Returns None if the container is out-of-scope or not set.

*property*collectors*: list[Any] | None*

The collectors or container of the receiver.

*classmethod*from_policy(*policy: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase)*) → WeightUpdaterBase | None[[source]](../../_modules/torchrl/collectors/weight_update.html#WeightUpdaterBase.from_policy)

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

WeightUpdaterBase | None

increment_version()[[source]](../../_modules/torchrl/collectors/weight_update.html#WeightUpdaterBase.increment_version)

Increment the policy version.

init(**args*, ***kwargs*)[[source]](../../_modules/torchrl/collectors/weight_update.html#WeightUpdaterBase.init)

Initialize the weight updater with custom arguments.

This method can be overridden by subclasses to handle custom initialization.
By default, this is a no-op.

Parameters:

- ***args** - Positional arguments for initialization
- ****kwargs** - Keyword arguments for initialization

*property*post_hooks*: list[Callable[[], None]]*

The list of post-hooks registered to the weight updater.

push_weights(*policy_or_weights: [TensorDictModuleBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) | [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | dict | None = None*, *worker_ids: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | int | list[int] | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)] | None = None*)[[source]](../../_modules/torchrl/collectors/weight_update.html#WeightUpdaterBase.push_weights)

Updates the weights of the policy, or on specified / all remote workers.

Parameters:

- **policy_or_weights** - The source to get weights from. Can be:
- TensorDictModuleBase: A policy module whose weights will be extracted
- TensorDictBase: A TensorDict containing weights
- dict: A regular dict containing weights
- None: Will try to get weights from server using _get_server_weights()
- **worker_ids** - An optional list of workers to update.

Returns: nothing.

register_collector(*collector*)[[source]](../../_modules/torchrl/collectors/weight_update.html#WeightUpdaterBase.register_collector)

Register a collector in the updater.

Once registered, the updater will not accept another collector.

Parameters:

**collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)) - The collector to register.

register_post_hook(*hook: Callable[[], None]*)[[source]](../../_modules/torchrl/collectors/weight_update.html#WeightUpdaterBase.register_post_hook)

Registers a post-hook to be called after weights are updated.

Parameters:

**hook** (*Callable**[**[**]**,**None**]*) - The post-hook to register.