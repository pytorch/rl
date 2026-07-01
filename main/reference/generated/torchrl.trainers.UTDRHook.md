# UTDRHook

*class*torchrl.trainers.UTDRHook(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*)[[source]](../../_modules/torchrl/trainers/trainers.html#UTDRHook)

Hook for logging Update-to-Data (UTD) ratio during async collection.

The UTD ratio measures how many optimization steps are performed per
collected data sample, providing insight into training efficiency during
asynchronous data collection. This metric is particularly useful for
off-policy algorithms where data collection and training happen concurrently.

The UTD ratio is calculated as: (batch_size * update_count) / write_count
where:
- batch_size: Size of batches sampled from replay buffer
- update_count: Total number of optimization steps performed
- write_count: Total number of samples written to replay buffer

Parameters:

**trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - The trainer instance to monitor for UTD calculation.
Must have async_collection=True for meaningful results.

Note

This hook is only meaningful when async_collection is enabled, as it
relies on the replay buffer's write_count to track data collection progress.

load_state_dict(*state_dict: dict[str, Any]*) → None[[source]](../../_modules/torchrl/trainers/trainers.html#UTDRHook.load_state_dict)

Load state from dictionary.

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'utdr_hook'*)[[source]](../../_modules/torchrl/trainers/trainers.html#UTDRHook.register)

Register the UTD ratio hook with the trainer.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - The trainer to register with.
- **name** (*str*) - Name to use when registering the hook module.

state_dict() → dict[str, Any][[source]](../../_modules/torchrl/trainers/trainers.html#UTDRHook.state_dict)

Return state dictionary for checkpointing.