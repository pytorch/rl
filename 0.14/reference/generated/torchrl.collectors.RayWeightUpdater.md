# RayWeightUpdater

*class*torchrl.collectors.RayWeightUpdater(*policy_weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *remote_collectors: list*, *max_interval: int = 0*)[[source]](../../_modules/torchrl/collectors/weight_update.html#RayWeightUpdater)

A remote weight updater for synchronizing policy weights across remote workers using Ray.

The RayWeightUpdater class provides a mechanism for updating the weights of a policy
across remote inference workers managed by Ray. It leverages Ray's distributed computing
capabilities to efficiently distribute policy weights to remote collectors.
This class is typically used in distributed data collectors where each remote worker requires
an up-to-date copy of the policy weights.

Parameters:

- **policy_weights** (*TensorDictBase*) - The current weights of the policy that need to be distributed
to remote workers.
- **remote_collectors** (*List*) - A list of remote collectors that will receive the updated policy weights.
- **max_interval** (*int**,**optional*) - The maximum number of batches between weight updates for each worker.
Defaults to 0, meaning weights are updated every batch.

all_worker_ids()[[source]](../../_modules/torchrl/collectors/weight_update.html#RayWeightUpdater.all_worker_ids)

Returns a list of all worker identifiers (indices of remote collectors).

_get_server_weights()[[source]](../../_modules/torchrl/collectors/weight_update.html#RayWeightUpdater._get_server_weights)

Retrieves the latest weights from the server and stores them in Ray's object store.

_maybe_map_weights()[[source]](../../_modules/torchrl/collectors/weight_update.html#RayWeightUpdater._maybe_map_weights)

Optionally maps server weights before distribution (no-op in this implementation).

_sync_weights_with_worker()[[source]](../../_modules/torchrl/collectors/weight_update.html#RayWeightUpdater._sync_weights_with_worker)

Synchronizes the server weights with a specific remote worker using Ray.

_skip_update()[[source]](../../_modules/torchrl/collectors/weight_update.html#RayWeightUpdater._skip_update)

Determines whether to skip the weight update for a specific worker based on the interval.

Note

This class assumes that the server weights can be directly applied to the remote workers without
any additional processing. If your use case requires more complex weight mapping or synchronization
logic, consider extending WeightUpdaterBase with a custom implementation.

See also

[`WeightUpdaterBase`](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase) and
[`RayCollector`](torchrl.collectors.distributed.RayCollector.html#torchrl.collectors.distributed.RayCollector).

all_worker_ids() → list[int] | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)][[source]](../../_modules/torchrl/collectors/weight_update.html#RayWeightUpdater.all_worker_ids)

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