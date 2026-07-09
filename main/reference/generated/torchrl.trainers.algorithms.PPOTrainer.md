# PPOTrainer

*class*torchrl.trainers.algorithms.PPOTrainer(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/algorithms/ppo.html#PPOTrainer)

PPO (Proximal Policy Optimization) trainer implementation.

See also `PPOTrainerConfig` for the
Hydra configuration counterpart.

Warning

This is an experimental/prototype feature. The API may change in future versions.
Please report any issues or feedback to help improve this implementation.

This trainer implements the PPO algorithm for training reinforcement learning agents.
It extends [`OnPolicyTrainer`](torchrl.trainers.algorithms.OnPolicyTrainer.html#torchrl.trainers.algorithms.OnPolicyTrainer) with PPO-specific
defaults; see that class for the full list of keyword arguments, covering
advantage estimation (GAE), replay-buffer wiring, collector weight
synchronization and logging.

PPO typically uses multiple epochs of optimization on the same batch of data.
This trainer defaults to 4 epochs, which is a common choice for PPO implementations.

Examples

```
>>> # Basic usage with manual configuration
>>> from torchrl.trainers.algorithms.ppo import PPOTrainer
>>> from torchrl.trainers.algorithms.configs import PPOTrainerConfig
>>> from hydra.utils import instantiate
>>> config = PPOTrainerConfig(...) # Configure with required parameters
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
state dict. On torch < 2.4 the default is `weights_only=False`
because the weights-only unpickler of those versions cannot
deserialize the `torch.device` instances contained in
TensorDict state-dicts.

Note

When `CKPT_BACKEND=torch`, `mmap=True` is set by default so
the checkpoint is memory-mapped rather than materialized in RAM
at load time. Pass `mmap=False` if the checkpoint was saved
with the legacy (pre-zipfile) `torch.save` format or if
`file` is a file-like object rather than a path. On Windows
the default is `mmap=False`: a mapped checkpoint would keep
the file locked, preventing it from being deleted or re-saved
while the loaded state is alive.

request_stop(*reason: str | None = None*) → None

Signal that training should stop at the next loop boundary.