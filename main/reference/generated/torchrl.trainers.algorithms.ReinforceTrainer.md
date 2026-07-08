# ReinforceTrainer

*class*torchrl.trainers.algorithms.ReinforceTrainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/algorithms/reinforce.html#ReinforceTrainer)

REINFORCE (policy gradient with baseline) trainer implementation.

See also `ReinforceTrainerConfig` for
the Hydra configuration counterpart.

Warning

This is an experimental/prototype feature. The API may change in future versions.
Please report any issues or feedback to help improve this implementation.

This trainer implements the REINFORCE algorithm for training reinforcement
learning agents, using a critic network as a baseline for advantage
estimation (GAE by default, matching
[`ReinforceLoss`](torchrl.objectives.ReinforceLoss.html#torchrl.objectives.ReinforceLoss)). It extends
[`OnPolicyTrainer`](torchrl.trainers.algorithms.OnPolicyTrainer.html#torchrl.trainers.algorithms.OnPolicyTrainer); see that class for the
full list of keyword arguments, covering advantage estimation,
replay-buffer wiring, collector weight synchronization and logging.

REINFORCE is a single-pass on-policy algorithm: each collected batch is
consumed once. This trainer therefore defaults to 1 epoch per batch.

Examples

```
>>> # Basic usage with manual configuration
>>> from torchrl.trainers.algorithms.reinforce import ReinforceTrainer
>>> from torchrl.trainers.algorithms.configs import ReinforceTrainerConfig
>>> from hydra.utils import instantiate
>>> config = ReinforceTrainerConfig(...) # Configure with required parameters
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

request_stop(*reason: str | None = None*) → None

Signal that training should stop at the next loop boundary.