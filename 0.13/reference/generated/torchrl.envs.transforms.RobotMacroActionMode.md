# RobotMacroActionMode

*class*torchrl.envs.transforms.RobotMacroActionMode(*value*, *names=None*, ***, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[[source]](../../_modules/torchrl/envs/custom/mujoco/_ur_primitives.html#RobotMacroActionMode)

Readable modes for [`RobotMacroAction`](torchrl.envs.transforms.RobotMacroAction.html#torchrl.envs.transforms.RobotMacroAction).

`RobotMacroActionMode` mirrors the URScript primitive set and adds
`RESET`. The reset mode requires a parent environment exposing
`robot_home_qpos`.

Examples

```
>>> from torchrl.envs import RobotMacroActionMode
>>> RobotMacroActionMode.REACH_POSE.name
'REACH_POSE'
```