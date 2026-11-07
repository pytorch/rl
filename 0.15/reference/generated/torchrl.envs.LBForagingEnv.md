# LBForagingEnv

torchrl.envs.LBForagingEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/libs/lbforaging.html#LBForagingEnv)

Level-Based Foraging environment wrapper, constructed from an environment name.

See `LBForagingWrapper` for a
description of the environment and the tensordict layout it produces.

Parameters:

**env_name** (*str*) - the name of a registered Level-Based Foraging
Gymnasium environment, e.g. `"Foraging-8x8-2p-3f-v3"` (an 8x8
grid, 2 players, 3 food items). See
`available_envs` for the full list.

Keyword Arguments:

- **categorical_actions** (*bool**,**optional*) - whether discrete actions
should be provided as categorical indices or one-hot encodings.
Defaults to `True`.
- **seed** (*int**,**optional*) - the seed to use to reset the environment on
the first call to [`reset()`](torchrl.envs.ModelBasedEnvBase.html#torchrl.envs.reset). Defaults to `None`.
- ****kwargs** - forwarded to `gymnasium.make`.

Examples

```
>>> from torchrl.envs.libs.lbforaging import LBForagingEnv
>>> env = LBForagingEnv("Foraging-8x8-2p-3f-v3", categorical_actions=False)
>>> rollout = env.rollout(3)
```