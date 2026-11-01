# MenagerieTask

*class*torchrl.envs.MenagerieTask(*keyframe: str | None = None*, *site_names: Sequence[str] = ()*, *terminate_below_height: float | None = None*, *pose_weight: float = 0.0*, *pose_std: float = 0.5*, *control_cost_weight: float = 0.0*, *alive_bonus: float = 0.0*)[[source]](../../_modules/torchrl/envs/custom/mujoco/menagerie.html#MenagerieTask)

Task parameters of [`MenagerieEnv`](torchrl.envs.MenagerieEnv.html#torchrl.envs.MenagerieEnv).

Menagerie ships robots, not tasks, so the defaults describe the bare
simulator: reset around the model's `home` keyframe, observe the state,
never terminate before the horizon and pay no reward. Non-zero weights turn
the built-in reward terms on; leave them at zero to let a
`Transform` write `("next", "reward")` instead.
[`MenagerieEnv.hold_pose_task()`](torchrl.envs.MenagerieEnv.html#torchrl.envs.MenagerieEnv.hold_pose_task) is the preset that turns them on.

Parameters:

- **keyframe** (*str**,**optional*) - name of the MJCF keyframe whose `qpos` and
`qvel` the reset state is drawn around. `None` (default) uses
the `home` keyframe when the model defines one and the model's
`qpos0` at rest otherwise; a name the model does not define
raises `KeyError` at construction.
- **site_names** ([*Sequence*](torchrl.data.Sequence.html#torchrl.data.Sequence)*[**str**]**,**optional*) - MuJoCo sites whose world
positions are exposed as the `site_positions` observation,
shaped `(num_envs, len(site_names), 3)` in this order. Empty
(default) omits the entry.
- **terminate_below_height** (*float**,**optional*) - if set, the episode
terminates once the height (world `z`) of the floating base drops
below this value, in meters. The base is the first free joint in
model order, the robot's in Menagerie scenes that carry one; a
scene whose only free joint belongs to an object (a cube on a
table) would track that object instead. `None` (default) never
terminates on height.
- **pose_weight** (*float**,**optional*) - weight of the pose term,
`exp(-mean((q - q_key)^2) / pose_std^2)` over the hinge and
slide joints, where `q_key` is the reset keyframe. `0.0`
(default) turns the term off.
- **pose_std** (*float**,**optional*) - scale of the pose term, in the joints'
units. Defaults to `0.5`.
- **control_cost_weight** (*float**,**optional*) - weight of the control cost,
minus the mean squared action after mapping each actuator's
control range onto `[-1, 1]`. `0.0` (default) turns the term
off.
- **alive_bonus** (*float**,**optional*) - constant paid at every step that does
not terminate on height. Defaults to `0.0`.

Examples

```
>>> from dataclasses import replace
>>> from torchrl.envs import MenagerieEnv, MenagerieTask
>>> task = MenagerieTask(site_names=("imu",), terminate_below_height=0.15)
>>> task.pose_weight, task.site_names
(0.0, ('imu',))
>>> standing = replace(MenagerieEnv.hold_pose_task(), alive_bonus=0.5)
>>> standing.pose_weight, standing.alive_bonus
(1.0, 0.5)
>>> env = MenagerieEnv("unitree_go2", download=True, task=standing)
```