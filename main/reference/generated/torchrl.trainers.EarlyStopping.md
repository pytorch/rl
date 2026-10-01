# EarlyStopping

*class*torchrl.trainers.EarlyStopping(***, *monitor: NestedKey = 'r_evaluation'*, *mode: Literal['min', 'max'] = 'max'*, *min_delta: float = 0.0*, *patience: int = 100000*, *wait_for: int = 1000000*, *check_finite: bool = True*)[[source]](../../_modules/torchrl/trainers/trainers.html#EarlyStopping)

Early stopping hook for [`Trainer`](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer).

This hook monitors a scalar metric and stops training when that metric
does not improve according to a configured criterion.

By default, the hook monitors `"r_evaluation"`.

Parameters:

- **monitor** (*NestedKey**,**optional*) - Metric name to monitor.
Defaults to `"r_evaluation"`.
- **mode** (*Literal**[**"min"**,**"max"**]**,**optional*) - One of `"min"` or `"max"`.
In `"max"` mode, larger metric values are considered better.
Defaults to `"max"`.
- **min_delta** (*float**,**optional*) - Minimum absolute improvement required to
qualify as better. Defaults to `0.0`.
- **patience** (*int**,**optional*) - Maximum number of non-improving frames
allowed before stopping. Defaults to `100_000`.
- **wait_for** (*int**,**optional*) - Number of initial frames to ignore before
checking the stopping criterion. Defaults to `1_000_000`.
- **check_finite** (*bool**,**optional*) - If `True`, non-finite metric values
(NaN or inf) trigger early stopping. Defaults to `True`.

Examples

```
>>> LogScalar(("next", "reward"), "r_training").register(trainer)
>>> EarlyStopping(monitor="r_training", patience=10_000).register(trainer)
```

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'early_stopping'*) → None[[source]](../../_modules/torchrl/trainers/trainers.html#EarlyStopping.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.