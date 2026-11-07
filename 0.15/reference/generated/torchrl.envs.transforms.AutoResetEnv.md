# AutoResetEnv

*class*torchrl.envs.transforms.AutoResetEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/transforms/_base.html#AutoResetEnv)

A subclass for auto-resetting envs.

insert_transform(*index: int*, *transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*) → None[[source]](../../_modules/torchrl/envs/transforms/_base.html#AutoResetEnv.insert_transform)

Inserts a transform to the env at the desired index.

[`Transform`](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) or callable are accepted.