# MacroPrimitive

*class*torchrl.envs.transforms.MacroPrimitive(*value*, *names=None*, ***, *module=None*, *qualname=None*, *type=None*, *start=1*, *boundary=None*)[[source]](../../_modules/torchrl/envs/transforms/_primitive.html#MacroPrimitive)

Generic primitive ids understood by [`MacroPrimitiveTransform`](torchrl.envs.transforms.MacroPrimitiveTransform.html#torchrl.envs.transforms.MacroPrimitiveTransform).

The base vocabulary is intentionally tiny and robot-agnostic: either hold
the current low-level action (`WAIT`) or interpolate toward a low-level
action target (`MOVE`). Domain-specific transforms can extend this enum in
their own modules (e.g. adding gripper or inverse-kinematics primitives).

Examples

```
>>> from torchrl.envs.transforms import MacroPrimitive
>>> int(MacroPrimitive.MOVE)
1
```