# UpdateWeights

*class*torchrl.trainers.UpdateWeights(*collector: [BaseCollector](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)*, *update_weights_interval: int*, *policy_weights_getter: Callable[[Any], Any] | None = None*, *weight_update_map: dict[str, str] | None = None*, *trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer) | None = None*)[[source]](../../_modules/torchrl/trainers/trainers.html#UpdateWeights)

A collector weights update hook class.

This hook must be used whenever the collector policy weights sit on a
different device than the policy weights being trained by the Trainer.
In that case, those weights must be synced across devices at regular
intervals. If the devices match, this will result in a no-op.

Parameters:

- **collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)) - A data collector where the policy weights
must be synced.
- **update_weights_interval** (*int*) - Interval (in terms of number of batches
collected) where the sync must take place.
- **policy_weights_getter** (*Callable**,**optional*) - A callable that returns the policy
weights to sync. Used for backward compatibility. If both this and
weight_update_map are provided, weight_update_map takes precedence.
- **weight_update_map** (*dict**[**str**,**str**]**,**optional*) - A mapping from destination paths
(keys in collector's weight_sync_schemes) to source paths on the trainer.
Example: `{"policy": "loss_module.actor_network", "replay_buffer.transforms[0]": "loss_module.critic_network"}`.
- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*,**optional*) - The trainer instance, required when using
weight_update_map to resolve source paths.

Examples

```
>>> # Legacy usage with policy_weights_getter
>>> update_weights = UpdateWeights(
... trainer.collector, T,
... policy_weights_getter=lambda: TensorDict.from_module(policy)
... )
>>> trainer.register_op("post_steps", update_weights)
```

```
>>> # New usage with weight_update_map
>>> update_weights = UpdateWeights(
... trainer.collector, T,
... weight_update_map={
... "policy": "loss_module.actor_network",
... "replay_buffer.transforms[0]": "loss_module.critic_network"
... },
... trainer=trainer
... )
>>> trainer.register_op("post_steps", update_weights)
```

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'update_weights'*)[[source]](../../_modules/torchrl/trainers/trainers.html#UpdateWeights.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.