# RewardNormalizer

*class*torchrl.trainers.RewardNormalizer(*decay: float = 0.999*, *scale: float = 1.0*, *eps: float | None = None*, *log_pbar: bool = False*, *reward_key=None*)[[source]](../../_modules/torchrl/trainers/trainers.html#RewardNormalizer)

Reward normalizer hook.

Parameters:

- **decay** (`float`, optional) - exponential moving average decay parameter.
Default is 0.999
- **scale** (`float`, optional) - the scale used to multiply the reward once
normalized. Defaults to 1.0.
- **eps** (`float`, optional) - the epsilon jitter used to prevent numerical
underflow. Defaults to `torch.finfo(DEFAULT_DTYPE).eps`
where `DEFAULT_DTYPE=torch.get_default_dtype()`.
- **reward_key** (*str**or**tuple**,**optional*) - the key where to find the reward
in the input batch. Defaults to `("next", "reward")`

Examples

```
>>> reward_normalizer = RewardNormalizer()
>>> trainer.register_op("batch_process", reward_normalizer.update_reward_stats)
>>> trainer.register_op("process_optim_batch", reward_normalizer.normalize_reward)
```

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'reward_normalizer'*)[[source]](../../_modules/torchrl/trainers/trainers.html#RewardNormalizer.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.