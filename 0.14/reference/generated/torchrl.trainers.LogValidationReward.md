# LogValidationReward

*class*torchrl.trainers.LogValidationReward(***, *record_interval: int*, *record_frames: int*, *frame_skip: int = 1*, *policy_exploration: [TensorDictModule](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)*, *environment: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) = None*, *exploration_type: [InteractionType](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.InteractionType.html#tensordict.nn.InteractionType) = InteractionType.RANDOM*, *log_keys: list[str | tuple[str]] | None = None*, *out_keys: dict[str | tuple[str], str] | None = None*, *suffix: str | None = None*, *log_pbar: bool = False*, *recorder: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) = None*)[[source]](../../_modules/torchrl/trainers/trainers.html#LogValidationReward)

Recorder hook for [`Trainer`](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer).

Parameters:

- **record_interval** (*int*) - total number of optimization steps
between two calls to the recorder for testing.
- **record_frames** (*int*) - number of frames to be recorded during
testing.
- **frame_skip** (*int*) - frame_skip used in the environment. It is
important to let the trainer know the number of frames skipped at
each iteration, otherwise the frame count can be underestimated.
For logging, this parameter is important to normalize the reward.
Finally, to compare different runs with different frame_skip,
one must normalize the frame count and rewards. Defaults to `1`.
- **policy_exploration** (*ProbabilisticTDModule*) -

a policy
instance used for

1. updating the exploration noise schedule;
2. testing the policy on the recorder.

Given that this instance is supposed to both explore and render
the performance of the policy, it should be possible to turn off
the explorative behavior by calling the
set_exploration_type(ExplorationType.DETERMINISTIC) context manager.
- **environment** ([*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)) - An environment instance to be used
for testing.
- **exploration_type** (*ExplorationType**,**optional*) - exploration mode to use for the
policy. By default, no exploration is used and the value used is
`ExplorationType.DETERMINISTIC`. Set to `ExplorationType.RANDOM` to enable exploration
- **log_keys** (*sequence**of**str**or**tuples**or**str**,**optional*) - keys to read in the tensordict
for logging. Defaults to `[("next", "reward")]`.
- **out_keys** (*Dict**[**str**,**str**]**,**optional*) - a dictionary mapping the `log_keys`
to their name in the logs. Defaults to `{("next", "reward"): "r_evaluation"}`.
- **suffix** (*str**,**optional*) - suffix of the video to be recorded.
- **log_pbar** (*bool**,**optional*) - if `True`, the reward value will be logged on
the progression bar. Default is False.

register(*trainer: [Trainer](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)*, *name: str = 'recorder'*)[[source]](../../_modules/torchrl/trainers/trainers.html#LogValidationReward.register)

Registers the hook in the trainer at a default location.

Parameters:

- **trainer** ([*Trainer*](torchrl.trainers.Trainer.html#torchrl.trainers.Trainer)) - the trainer where the hook must be registered.
- **name** (*str*) - the name of the hook.

Note

To register the hook at another location than the default, use
`register_op()`.