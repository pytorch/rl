# URScriptPrimitive

*class*torchrl.envs.transforms.URScriptPrimitive(*value*, *names=None*, ***, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[[source]](../../_modules/torchrl/envs/custom/mujoco/_ur_primitives.html#URScriptPrimitive)

Integer ids for URScript-style robot primitives.

The ids are specific to UR-style arm control with a binary gripper command;
they extend the generic [`MacroPrimitive`](torchrl.envs.transforms.MacroPrimitive.html#torchrl.envs.transforms.MacroPrimitive)
vocabulary (`WAIT`/`MOVE`) with joint, Cartesian and gripper moves.

Examples

```
>>> from torchrl.envs import URScriptPrimitive
>>> str(URScriptPrimitive.OPEN_GRIPPER)
'open_gripper'
```