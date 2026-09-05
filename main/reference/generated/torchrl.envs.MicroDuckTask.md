# MicroDuckTask

*class*torchrl.envs.MicroDuckTask(*commanded_x_velocity: float | ~collections.abc.Sequence[float] = (0.03*, *)*, *command_range: tuple[float*, *float] | None = None*, *warm_start_velocity: tuple[float*, *float] | None = None*, *warm_start_fraction: float = 0.0*, *joint_reset_noise_scale: float | None = None*, *action_scale: float = 0.35*, *gait_frequency_hz: float = 1.8913*, *gait_frequency_per_mps: float = 0.0*, *gait_phase_offset: float = -1.5237*, *gait_ramp_duration_s: float = 0.4*, *observe_lateral_velocity: bool = False*, *reward_scales: ~collections.abc.Mapping[str*, *float] = <factory>*, *compute_reward: bool = True*, *diagnostics: bool = False*)[[source]](../../_modules/torchrl/envs/custom/mujoco/microduck.html#MicroDuckTask)

Task parameters of [`MicroDuckEnv`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv).

Commands, reset distribution, actuation scale, gait clock, observation and
reward options live here, so a task is one object rather than a dozen
constructor arguments. Build one directly or start from a preset such as
[`MicroDuckEnv.tracking_task()`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv.tracking_task), [`MicroDuckEnv.standing_task()`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv.standing_task) or
[`MicroDuckEnv.speed_range_task()`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv.speed_range_task) and pass field overrides.

Parameters:

- **commanded_x_velocity** (*float**or*[*Sequence*](torchrl.data.Sequence.html#torchrl.data.Sequence)*[**float**]**,**optional*) - body-frame
longitudinal velocity command in m/s. Every reset draws one value
uniformly from the sequence for each env (a scalar is a fixed
command, and repeating a value weights the draw); the command
stays constant until the next reset. A `commanded_x_velocity`
entry of shape `(num_envs, 1)` or `(num_envs,)` in the reset
TensorDict overrides the draw; the key is in the env's
`state_spec` so `TransformedEnv` forwards
it. Defaults to `(0.03,)`. Ignored when `command_range` is
given.
- **command_range** (*tuple**[**float**,**float**]**,**optional*) - `(low, high)` interval
in m/s from which the command is drawn uniformly at every reset
instead, for training over a continuous speed range.
- **warm_start_velocity** (*tuple**[**float**,**float**]**,**optional*) - `(low, high)`
speed interval in m/s. At reset, a `warm_start_fraction` of the
environments start already moving at a speed drawn from it, along
their heading for a non-negative command and against it for a
negative one, so an untrained policy experiences locomotion states
in the commanded direction early.
- **warm_start_fraction** (*float**,**optional*) - fraction of resets that receive
the warm start. Defaults to `0.0`.
- **joint_reset_noise_scale** (*float**,**optional*) - uniform noise added to the
joint positions at reset, in radians. Defaults to the env's
`reset_noise_scale`. Larger values start episodes in diverse,
off-balance poses, including single-support ones, which a
from-scratch policy otherwise rarely visits.
- **action_scale** (*float**,**optional*) - position-target offset in radians for
a unit normalized action. Defaults to `0.35`.
- **gait_frequency_hz** (*float**,**optional*) - frequency of the gait clock
exposed in the observation, at zero command. Defaults to
`1.8913`.
- **gait_frequency_per_mps** (*float**,**optional*) - increase of the gait clock
frequency per m/s of commanded speed, so the cadence rewarded by
the single-support term follows the command. Defaults to `0.0`
(fixed clock).
- **gait_phase_offset** (*float**,**optional*) - phase of the gait clock at the
first step, in radians. Defaults to `-1.5237`.
- **gait_ramp_duration_s** (*float**,**optional*) - duration over which the gait
ramp feature grows from zero to one after a reset. Defaults to
`0.4`.
- **observe_lateral_velocity** (*bool**,**optional*) - if `True`, append the
body-frame lateral and vertical velocities to the observation,
which gives the lateral tracking term an input. Defaults to
`False`.
- **reward_scales** (*Mapping**[**str**,**float**]**,**optional*) - reward attribute names
of [`MicroDuckEnv`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv) such as `"TRACKING_WEIGHT"` or
`"TRACKING_STD"` mapped to values that override the class
defaults on the env instance.
- **compute_reward** (*bool**,**optional*) - if `False`, the env writes a zero
reward and leaves the reward to a transform, which can read the
observation or the `diagnostics` keys. Defaults to `True`.
- **diagnostics** (*bool**,**optional*) - if `True`, add each reward component
and pose diagnostics to the observation spec under
`diagnostic_*` keys. Off by default because it roughly doubles
the per-step task cost.

Examples

Start from a preset, override a field, and hand the task to the env:

```
>>> from dataclasses import replace
>>> from torchrl.envs import MicroDuckEnv, MicroDuckTask
>>> task = MicroDuckEnv.speed_range_task(0.1, 0.3, action_scale=1.0)
>>> task.command_range, task.gait_frequency_per_mps, task.action_scale
((0.1, 0.3), 5.0, 1.0)
>>> env = MicroDuckEnv(download=True, task=task, num_envs=4) 
>>> rollout = env.rollout(20) 
>>> rollout["commanded_x_velocity"][:, 0, 0] # one command per env, drawn in [0.1, 0.3] 
tensor([0.2731, 0.1207, 0.2942, 0.1685])
```

The same task with the reward turned off, for a transform to fill in:

```
>>> env = MicroDuckEnv(download=True, task=replace(task, compute_reward=False)) 
>>> env.rollout(5)["next", "reward"].sum() 
tensor(0.)
```

A task built field by field is equivalent to the presets:

```
>>> MicroDuckTask(commanded_x_velocity=0.2) == MicroDuckEnv.tracking_task(0.2)
True
```