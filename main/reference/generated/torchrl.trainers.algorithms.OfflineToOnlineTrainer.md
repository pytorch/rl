# OfflineToOnlineTrainer

*class*torchrl.trainers.algorithms.OfflineToOnlineTrainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/algorithms/offline_to_online.html#OfflineToOnlineTrainer)

A SAC trainer for the offline-pretrain -> online-finetune transition.

See also `OfflineToOnlineTrainerConfig`
for the Hydra configuration counterpart.

Builds on [`SACTrainer`](torchrl.trainers.algorithms.SACTrainer.html#torchrl.trainers.algorithms.SACTrainer), swapping the
plain replay buffer for an [`OfflineToOnlineReplayBuffer`](torchrl.data.OfflineToOnlineReplayBuffer.html#torchrl.data.OfflineToOnlineReplayBuffer).
Each collected batch is routed to the online buffer while optimization
samples a mixed batch whose offline fraction is linearly annealed to zero
over `anneal_frames` frames - warm-starting the policy on offline data
and smoothly handing it over to its own online experience. All other SAC
behaviour (target-net updates, weight sync, logging) is inherited.

Parameters:

- **collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)) - the data collector for online interactions.
- **total_frames** (*int*) - total number of frames to collect.
- **frame_skip** (*int*) - frames skipped between policy updates.
- **optim_steps_per_batch** (*int*) - optimization steps per collected batch.
- **loss_module** ([*LossModule*](torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)) - the SAC loss module.
- **replay_buffer** ([*OfflineToOnlineReplayBuffer*](torchrl.data.OfflineToOnlineReplayBuffer.html#torchrl.data.OfflineToOnlineReplayBuffer)) - the offline-to-online buffer.

Keyword Arguments:

- **anneal_frames** (*int**,**optional*) - frames over which `offline_fraction`
decays to 0. Defaults to `total_frames`; pass `<= 0` to keep the
fraction fixed.
- **batch_size** (*int**,**optional*) - replay-buffer sampling batch size.

See [`SACTrainer`](torchrl.trainers.algorithms.SACTrainer.html#torchrl.trainers.algorithms.SACTrainer) for the remaining
keyword arguments.

Note

Experimental/prototype feature; the API may change.

load_from_file(*file: str | Path*, ***kwargs*) → [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)

Loads a file and its state-dict in the trainer.

Keyword arguments are passed to the [`load()`](https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load) function.
They are ignored when `CKPT_BACKEND=memmap`.

Note

When `CKPT_BACKEND=torch`, `weights_only=True` is set by
default for safer deserialization. Pass `weights_only=False`
explicitly only if you have custom (non-stdlib) objects in your
state dict.

request_stop(*reason: str | None = None*) → None

Signal that training should stop at the next loop boundary.