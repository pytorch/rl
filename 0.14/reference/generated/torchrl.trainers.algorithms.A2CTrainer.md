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

compute_loss(*sub_batch: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *method: str | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | tuple[Any, ...]

Evaluate the configured loss through the active execution boundary.

load_from_file(*file: str | Path*, ***kwargs*) → [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)

Loads a file and its state-dict in the trainer.

Keyword arguments are passed to the [`load()`](https://docs.pytorch.org/docs/stable/generated/torch.load.html#torch.load) function for
legacy torch checkpoints and unified components explicitly saved with
the torch state-dict payload format. Unified checkpoints additionally
accept `strict` to control missing or incompatible components.
Arguments are ignored when `CKPT_BACKEND=memmap`.

Note

Unified state-dict components use TensorDict storage by default and
do not invoke the pickle loader. For explicit torch payloads and
`CKPT_BACKEND=torch` checkpoints, `weights_only=True` is the
default for safer deserialization. Pass `weights_only=False`
explicitly only if the state dict contains custom objects. On
torch < 2.4 the default is `weights_only=False` because the
weights-only unpickler of those versions cannot deserialize the
`torch.device` instances contained in TensorDict state-dicts.

Note

Explicit torch payloads and `CKPT_BACKEND=torch` checkpoints use
`mmap=True` by default. Pass `mmap=False` for legacy pre-zipfile
`torch.save` files or file-like objects. On Windows the default
is `mmap=False` because a mapped checkpoint keeps the file locked,
preventing deletion or re-save.

Note

Unified checkpoint tensors are mapped to CPU by default. Pass an
explicit `map_location` to select another device mapping.

Note

After restoring an independently registered policy component, the
trainer synchronizes the collector once so local policy copies and
remote workers observe the restored learner weights.

request_stop(*reason: str | None = None*) → None

Signal that training should stop at the next loop boundary.