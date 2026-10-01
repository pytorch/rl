# LogScalar

*class*torchrl.trainers.LogScalar(*key: NestedKey = ('next', 'reward')*, *logname: str | None = None*, *log_pbar: bool = False*, *include_std: bool = True*, *reduction: str = 'mean'*)[[source]](../../_modules/torchrl/trainers/trainers.html#LogScalar)

Generic scalar logger hook for any tensor values in the batch.

This hook can log any scalar values from the collected batch data, including
rewards, action norms, done states, and any other metrics. It automatically
handles masking and computes both mean and standard deviation.

Parameters:

- **key** (*NestedKey*) - the key where to find the value in the input batch.
Can be a string for simple keys or a tuple for nested keys.
Default is torchrl.trainers.trainers.REWARD_KEY (= ("next", "reward")).
- **logname** (*str**,**optional*) - name of the metric to be logged. If None, will use
the key as the log name. Default is None.
- **log_pbar** (*bool**,**optional*) - if `True`, the value will be logged on
the progression bar. Default is `False`.
- **include_std** (*bool**,**optional*) - if `True`, also log the standard deviation
of the values. Default is `True`.
- **reduction** (*str**,**optional*) - reduction method to apply. Can be "mean", "sum",
"min", "max". Default is "mean".

Examples

```
>>> # Log training rewards
>>> log_reward = LogScalar(("next", "reward"), "r_training", log_pbar=True)
>>> trainer.register_op("pre_steps_log", log_reward)
```

```
>>> # Log action norms
>>> log_action_norm = LogScalar("action", "action_norm", include_std=True)
>>> trainer.register_op("pre_steps_log", log_action_norm)
```

```
>>> # Log done states (as percentage)
>>> log_done = LogScalar(("next", "done"), "done_percentage", reduction="mean")
>>> trainer.register_op("pre_steps_log", log_done)
```

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str | None = None*)[[source]](../../_modules/torchrl/trainers/trainers.html#LogScalar.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.