# RPCWeightUpdater

*class*torchrl.collectors.distributed.RPCWeightUpdater(*collector_infos*, *collector_class*, *collector_rrefs*, *policy_weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *num_workers: int*)[[source]](../../_modules/torchrl/collectors/distributed/rpc.html#RPCWeightUpdater)

A remote weight updater for synchronizing policy weights across remote workers using RPC.

The RPCWeightUpdater class provides a mechanism for updating the weights of a policy
across remote inference workers using RPC. It is designed to work with the [`RPCCollector`](torchrl.collectors.distributed.RPCCollector.html#torchrl.collectors.distributed.RPCCollector)
to ensure that each worker receives the latest policy weights.
This class is typically used in distributed data collection scenarios where remote workers
are managed via RPC and need to be kept in sync with the central policy weights.

Parameters:

- **collector_infos** - Information about the collectors, used for RPC communication.
- **collector_class** - The class of the collectors being used.
- **collector_rrefs** - Remote references to the collectors.
- **policy_weights** (*TensorDictBase*) - The current weights of the policy that need to be distributed
to the workers.
- **num_workers** (*int*) - The number of remote workers that will receive the updated policy weights.

update_weights()

Updates the weights on specified or all remote workers using RPC.

all_worker_ids()[[source]](../../_modules/torchrl/collectors/distributed/rpc.html#RPCWeightUpdater.all_worker_ids)

Returns a list of all worker identifiers (not implemented in this class).

_sync_weights_with_worker()[[source]](../../_modules/torchrl/collectors/distributed/rpc.html#RPCWeightUpdater._sync_weights_with_worker)

Synchronizes the server weights with a specific worker (not implemented).

_get_server_weights()[[source]](../../_modules/torchrl/collectors/distributed/rpc.html#RPCWeightUpdater._get_server_weights)

Retrieves the latest weights from the server (not implemented).

_maybe_map_weights()[[source]](../../_modules/torchrl/collectors/distributed/rpc.html#RPCWeightUpdater._maybe_map_weights)

Optionally maps server weights before distribution (not implemented).

Note

This class assumes that the server weights can be directly applied to the remote workers
without any additional processing. If your use case requires more complex weight mapping or
synchronization logic, consider extending WeightUpdaterBase with a custom implementation.

See also

[`WeightUpdaterBase`](torchrl.collectors.WeightUpdaterBase.html#torchrl.collectors.WeightUpdaterBase) and
[`RPCCollector`](torchrl.collectors.distributed.RPCCollector.html#torchrl.collectors.distributed.RPCCollector).

all_worker_ids() → list[int] | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)][[source]](../../_modules/torchrl/collectors/distributed/rpc.html#RPCWeightUpdater.all_worker_ids)

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

push_weights(*weights: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *worker_ids: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | int | list[int] | list[[device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)] | None = None*, ***kwargs*)[[source]](../../_modules/torchrl/collectors/distributed/rpc.html#RPCWeightUpdater.push_weights)

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