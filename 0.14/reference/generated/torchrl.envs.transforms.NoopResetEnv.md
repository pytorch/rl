# NoopResetEnv

*class*torchrl.envs.transforms.NoopResetEnv(*noops: int = 30*, *random: bool = True*)[[source]](../../_modules/torchrl/envs/transforms/_env.html#NoopResetEnv)

Runs a series of random actions when an environment is reset.

Parameters:

- **env** ([*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)) - env on which the random actions have to be
performed. Can be the same env as the one provided to the
TransformedEnv class
- **noops** (*int**,**optional*) - upper-bound on the number of actions
performed after reset. Default is 30.
If noops is too high such that it results in the env being
done or truncated before the all the noops are applied,
in multiple trials, the transform raises a RuntimeError.
- **random** (*bool**,**optional*) - if False, the number of random ops will
always be equal to the noops value. If True, the number of
random actions will be randomly selected between 0 and noops.
Default is True.