# instantiate_trainer

torchrl.trainers.algorithms.configs.instantiate_trainer(*cfg: DictConfig*, ***, *overrides: Sequence[str] | None = None*) → [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)[[source]](../../_modules/torchrl/trainers/algorithms/configs/entrypoint.html#instantiate_trainer)

Instantiate `cfg.trainer`, resuming from `cfg.resume` when it is set.

Without `resume` this is `hydra.utils.instantiate()` on `cfg.trainer`
plus registration of the composed configuration on the trainer checkpoint.
With `resume` set to a checkpoint or a
[`CheckpointRotation`](torchrl.checkpoint.CheckpointRotation.html#torchrl.checkpoint.CheckpointRotation) directory, the saved
configuration becomes the base and the current command-line overrides apply
on top (config-group overrides cannot apply to a saved configuration and are
ignored with a warning); the saved logger run is reattached before the
logger is constructed (W&B resumes the saved id with `resume="must"`, CSV
and TensorBoard keep the saved directory); checkpoints keep accumulating in
the resumed rotation directory unless `checkpoint_rotation.directory` is
overridden; and [`load_from_file()`](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer.load_from_file) restores
the trainer state.

Parameters:

- **cfg** (*DictConfig*) - the composed Hydra configuration. It must hold a
`trainer` node and may hold `resume`.
- **overrides** ([*Sequence*](torchrl.data.Sequence.html#torchrl.data.Sequence)*[**str**]**,**optional*) - command-line overrides applied over
the saved configuration. Defaults to the task overrides of the
current Hydra run, or none outside a Hydra application.

Returns:

The instantiated trainer, restored from the checkpoint when resuming.

Examples

```
>>> @hydra.main(config_path="config", config_name="config", version_base="1.3") 
... def main(cfg):
... trainer = instantiate_trainer(cfg)
... with trainer.stop_on_signal():
... trainer.train()
```