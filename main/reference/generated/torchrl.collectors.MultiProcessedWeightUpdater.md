# MultiProcessedWeightUpdater

*class*torchrl.collectors.MultiProcessedWeightUpdater(***, *get_server_weights: Callable[[], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)] | None*, *policy_weights: dict[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device), [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*)[[source]](../../_modules/torchrl/collectors/weight_update.html#MultiProcessedWeightUpdater)

A remote weight updater for synchronizing policy weights across multiple processes or devices.

The MultiProcessedWeightUpdater class provides a mechanism for updating the weights
of a policy across multiple inference workers in a multiprocessed environment. It is designed
to handle the distribution of weights from a central server to various devices or processes
that are running the policy.
This class is typically used in multiprocessed data collectors where each process or device
requires an up-to-date copy of the policy weights.

Keyword Arguments:

- **get_server_weights** (*Callable**[**[**]**,**TensorDictBase**]**|**None*) - A callable that retrieves the
latest policy weights from the server or another centralized source.
- **policy_weights** (*Dict**[*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**TensorDictBase**]*) - A dictionary mapping each device or
process to its current policy weights, which will be updated.

Note

This class assumes that the server weights can be directly applied to the workers without
any additional processing. If your use case requires more complex weight mapping or synchronization
logic, consider extending WeightUpdaterBase with a custom implementation.

See also

[`WeightUpdaterBase`](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase) and
[`BaseCollector`](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector).

all_worker_ids() → list[int] | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)][[source]](../../_modules/torchrl/collectors/weight_update.html#MultiProcessedWeightUpdater.all_worker_ids)

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

increment_version()

Increment the policy version.

init(**args*, ***kwargs*)

Initialize the weight updater with custom arguments.

This method can be overridden by subclasses to handle custom initialization.
By default, this is a no-op.

Parameters:

- ***args** - Positional arguments for initialization
- ****kwargs** - Keyword arguments for initialization

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

register_collector(*collector*)

Register a collector in the updater.

Once registered, the updater will not accept another collector.

Parameters:

**collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)) - The collector to register.

register_post_hook(*hook: Callable[[], None]*)

Registers a post-hook to be called after weights are updated.

Parameters:

**hook** (*Callable**[**[**]**,**None**]*) - The post-hook to register.