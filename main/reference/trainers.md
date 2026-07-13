# torchrl.trainers package

The trainer package provides utilities to write reusable training scripts. The core idea is to use a
trainer that implements a nested loop, where the outer loop runs the data collection steps and the inner
loop the optimization steps.

## Key Features

- **Modular hook system**: Customize training at 18 different points in the loop
- **Checkpointing support**: Pass a [`torchrl.checkpoint.Checkpoint`](generated/torchrl.checkpoint.Checkpoint.html#torchrl.checkpoint.Checkpoint) for
the unified manifest format. Legacy `torch`, `torchsnapshot`, and
`memmap` backends remain readable during the migration window.
- **Algorithm trainers**: High-level trainers for PPO, A2C, REINFORCE, SAC,
offline-to-online SAC, DQN, DDPG, IQL, CQL, and TD3 with Hydra configuration
- **Builder helpers**: Utilities for constructing collectors, losses, and replay buffers

## Quick Example

```
from torchrl.trainers import Trainer
from torchrl.trainers import UpdateWeights, LogScalar

# Create trainer
trainer = Trainer(
 collector=collector,
 total_frames=1000000,
 loss_module=loss,
 optimizer=optimizer,
)

# Register hooks
UpdateWeights(collector, 10).register(trainer)
LogScalar("reward").register(trainer)

# Train
trainer.train()
```

## Documentation Sections

- [Trainer Basics](trainers_basics.html)

- [Trainer and hooks](trainers_basics.html#trainer-and-hooks)
- [Algorithm-specific trainers](trainers_basics.html#algorithm-specific-trainers)
- [Builders](trainers_basics.html#builders)
- [Utils](trainers_basics.html#utils)
- [Loggers](trainers_loggers.html)

- [Logger](generated/torchrl.record.loggers.Logger.html)
- [ProcessLogger](generated/torchrl.record.loggers.ProcessLogger.html)
- [RayLogger](generated/torchrl.record.loggers.RayLogger.html)
- [CSVLogger](generated/torchrl.record.loggers.csv.CSVLogger.html)
- [MLFlowLogger](generated/torchrl.record.loggers.mlflow.MLFlowLogger.html)
- [TensorboardLogger](generated/torchrl.record.loggers.tensorboard.TensorboardLogger.html)
- [TrackioLogger](generated/torchrl.record.loggers.trackio.TrackioLogger.html)
- [WandbLogger](generated/torchrl.record.loggers.wandb.WandbLogger.html)
- [get_logger](generated/torchrl.record.loggers.get_logger.html)
- [generate_exp_name](generated/torchrl.record.loggers.generate_exp_name.html)
- [Recording utils](trainers_loggers.html#recording-utils)
- [Training Hooks](trainers_hooks.html)

- [BatchSubSampler](generated/torchrl.trainers.BatchSubSampler.html)
- [ClearCudaCache](generated/torchrl.trainers.ClearCudaCache.html)
- [CountFramesLog](generated/torchrl.trainers.CountFramesLog.html)
- [EarlyStopping](generated/torchrl.trainers.EarlyStopping.html)
- [LogScalar](generated/torchrl.trainers.LogScalar.html)
- [LRSchedulerHook](generated/torchrl.trainers.LRSchedulerHook.html)
- [OptimizerHook](generated/torchrl.trainers.OptimizerHook.html)
- [LogValidationReward](generated/torchrl.trainers.LogValidationReward.html)
- [ReplayBufferTrainer](generated/torchrl.trainers.ReplayBufferTrainer.html)
- [RewardNormalizer](generated/torchrl.trainers.RewardNormalizer.html)
- [SelectKeys](generated/torchrl.trainers.SelectKeys.html)
- [UpdateWeights](generated/torchrl.trainers.UpdateWeights.html)
- [TargetNetUpdaterHook](generated/torchrl.trainers.TargetNetUpdaterHook.html)
- [UTDRHook](generated/torchrl.trainers.UTDRHook.html)
- [ValueEstimatorHook](generated/torchrl.trainers.ValueEstimatorHook.html)