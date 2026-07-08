# Training Hooks

Hooks for customizing the training loop at various points.

| [`BatchSubSampler`](generated/torchrl.trainers.BatchSubSampler.html#torchrl.trainers.BatchSubSampler)(batch_size[, sub_traj_len, ...]) | Data subsampler for online RL sota-implementations. |
| --- | --- |
| [`ClearCudaCache`](generated/torchrl.trainers.ClearCudaCache.html#torchrl.trainers.ClearCudaCache)(interval) | Clears cuda cache at a given interval. |
| [`CountFramesLog`](generated/torchrl.trainers.CountFramesLog.html#torchrl.trainers.CountFramesLog)(*args, **kwargs) | A frame counter hook. |
| [`EarlyStopping`](generated/torchrl.trainers.EarlyStopping.html#torchrl.trainers.EarlyStopping)(*[, monitor, mode, min_delta, ...]) | Early stopping hook for [`Trainer`](generated/torchrl.trainers.Trainer.html#torchrl.trainers.Trainer). |
| [`LogScalar`](generated/torchrl.trainers.LogScalar.html#torchrl.trainers.LogScalar)([key, logname, log_pbar, ...]) | Generic scalar logger hook for any tensor values in the batch. |
| [`LRSchedulerHook`](generated/torchrl.trainers.LRSchedulerHook.html#torchrl.trainers.LRSchedulerHook)(scheduler[, interval]) | A hook that steps a learning-rate scheduler during training. |
| [`OptimizerHook`](generated/torchrl.trainers.OptimizerHook.html#torchrl.trainers.OptimizerHook)(optimizer[, loss_components]) | Add an optimizer for one or more loss components. |
| [`LogValidationReward`](generated/torchrl.trainers.LogValidationReward.html#torchrl.trainers.LogValidationReward)(*, record_interval, ...) | Recorder hook for [`Trainer`](generated/torchrl.trainers.Trainer.html#torchrl.trainers.Trainer). |
| [`ReplayBufferTrainer`](generated/torchrl.trainers.ReplayBufferTrainer.html#torchrl.trainers.ReplayBufferTrainer)(replay_buffer[, ...]) | Replay buffer hook provider. |
| [`RewardNormalizer`](generated/torchrl.trainers.RewardNormalizer.html#torchrl.trainers.RewardNormalizer)([decay, scale, eps, ...]) | Reward normalizer hook. |
| [`SelectKeys`](generated/torchrl.trainers.SelectKeys.html#torchrl.trainers.SelectKeys)(keys) | Selects keys in a TensorDict batch. |
| [`UpdateWeights`](generated/torchrl.trainers.UpdateWeights.html#torchrl.trainers.UpdateWeights)(collector, update_weights_interval) | A collector weights update hook class. |
| [`TargetNetUpdaterHook`](generated/torchrl.trainers.TargetNetUpdaterHook.html#torchrl.trainers.TargetNetUpdaterHook)(target_params_updater) | A hook for target parameters update. |
| [`UTDRHook`](generated/torchrl.trainers.UTDRHook.html#torchrl.trainers.UTDRHook)(trainer) | Hook for logging Update-to-Data (UTD) ratio during async collection. |
| [`ValueEstimatorHook`](generated/torchrl.trainers.ValueEstimatorHook.html#torchrl.trainers.ValueEstimatorHook)(value_estimator) | A hook that computes value estimates over a collected batch. |