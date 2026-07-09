# A2CTrainer

*class*torchrl.trainers.algorithms.A2CTrainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/algorithms/a2c.html#A2CTrainer)

A2C (Advantage Actor-Critic) trainer implementation.

See also `A2CTrainerConfig` for the
Hydra configuration counterpart.

Warning

This is an experimental/prototype feature. The API may change in future versions.
Please report any issues or feedback to help improve this implementation.

This trainer implements the A2C algorithm for training reinforcement learning agents.
It extends [`OnPolicyTrainer`](torchrl.trainers.algorithms.OnPolicyTrainer.html#torchrl.trainers.algorithms.OnPolicyTrainer) with A2C-specific
defaults; see that class for the full list of keyword arguments, covering
advantage estimation (GAE), replay-buffer wiring, collector weight
synchronization and logging. Entropy regularization is configured on the
loss module (see [`A2CLoss`](torchrl.objectives.A2CLoss.html#torchrl.objectives.A2CLoss)).

Unlike PPO, A2C performs a single optimization pass over each batch of collected
data. This trainer therefore defaults to 1 epoch per batch.

Examples

```
>>> # Basic usage with manual configuration
>>> from torchrl.trainers.algorithms.a2c import A2CTrainer
>>> from torchrl.trainers.algorithms.configs import A2CTrainerConfig
>>> from hydra.utils import instantiate
>>> config = A2CTrainerConfig(...) # Configure with required parameters
>>> trainer = instantiate(config)
>>> trainer.train()
```

Note

This trainer requires a configurable environment setup. See the
`configs` module for configuration options.

load_from_file(*file: str | Path*, ***kwargs*) → [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)

Loads a file and its state-dict in the trainer.

Keyword arguments are passed to the [`load()`](https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load) function.
They are ignored when `CKPT_BACKEND=memmap`.

Note

When `CKPT_BACKEND=torch`, `weights_only=True` is set by
default for safer deserialization. Pass `weights_only=False`
explicitly only if you have custom (non-stdlib) objects in your
state dict.

Note

When `CKPT_BACKEND=torch`, `mmap=True` is set by default so
the checkpoint is memory-mapped rather than materialized in RAM
at load time. Pass `mmap=False` if the checkpoint was saved
with the legacy (pre-zipfile) `torch.save` format or if
`file` is a file-like object rather than a path.

request_stop(*reason: str | None = None*) → None

Signal that training should stop at the next loop boundary.