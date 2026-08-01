# CountFramesLog

*class*torchrl.trainers.CountFramesLog(**args*, ***kwargs*)[[source]](../../_modules/torchrl/trainers/trainers.html#CountFramesLog)

A frame counter hook.

Parameters:

- **frame_skip** (*int*) - frame skip of the environment. This argument is
important to keep track of the total number of frames, not the
apparent one.
- **log_pbar** (*bool**,**optional*) - if `True`, the reward value will be logged on
the progression bar. Default is False.

Examples

```
>>> count_frames = CountFramesLog(frame_skip=frame_skip)
>>> trainer.register_op("pre_steps_log", count_frames)
```

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'count_frames_log'*)[[source]](../../_modules/torchrl/trainers/trainers.html#CountFramesLog.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.