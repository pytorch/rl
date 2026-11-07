# check_env_specs

torchrl.envs.check_env_specs(*env: [torchrl.envs.EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)*, *return_contiguous: bool | None = None*, *check_dtype=True*, *seed: int | None = None*, *tensordict: TensorDictBase | None = None*, *break_when_any_done: bool | Literal['both'] | None = None*)[[source]](../../_modules/torchrl/envs/utils.html#check_env_specs)

Tests an environment specs against the results of short rollout.

This test function should be used as a sanity check for an env wrapped with
torchrl's EnvBase subclasses: any discrepancy between the expected data and
the data collected should raise an assertion error.

A broken environment spec will likely make it impossible to use parallel
environments.

Parameters:

- **env** ([*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)) - the env for which the specs have to be checked against data.
- **return_contiguous** (*bool**,**optional*) - if `True`, the random rollout will be called with
return_contiguous=True. This will fail in some cases (e.g. heterogeneous shapes
of inputs/outputs). Defaults to `None` (determined by the presence of dynamic specs).
- **check_dtype** (*bool**,**optional*) - if False, dtype checks will be skipped.
Defaults to True.
- **seed** (*int**,**optional*) - for reproducibility, a seed can be set.
The seed will be set in pytorch temporarily, then the RNG state will
be reverted to what it was before. For the env, we set the seed but since
setting the rng state back to what is was isn't a feature of most environment,
we leave it to the user to accomplish that.
Defaults to `None`.
- **tensordict** (*TensorDict**,**optional*) - an optional tensordict instance to use for reset.
- **break_when_any_done** (*bool**or**str**,**optional*) - value for `break_when_any_done` in [`rollout()`](torchrl.envs.EnvBase.html#id2).
If `"both"`, the test is run on both True and False.

Caution: this function resets the env seed. It should be used "offline" to
check that an env is adequately constructed, but it may affect the seeding
of an experiment and as such should be kept out of training scripts.