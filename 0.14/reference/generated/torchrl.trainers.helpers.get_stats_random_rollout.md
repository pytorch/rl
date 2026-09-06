# get_stats_random_rollout

torchrl.trainers.helpers.get_stats_random_rollout(*cfg: DictConfig*, *proof_environment: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) = None*, *key: str | None = None*)[[source]](../../_modules/torchrl/trainers/helpers/envs.html#get_stats_random_rollout)

Gathers stas (loc and scale) from an environment using random rollouts.

Parameters:

- **cfg** (*DictConfig*) - a config object with init_env_steps field, indicating
the total number of frames to be collected to compute the stats.
- **proof_environment** (*EnvBase instance**,**optional*) - if provided, this env will
be used to execute the rollouts. If not, it will be created using
the cfg object.
- **key** (*str**,**optional*) - if provided, the stats of this key will be gathered.
If not, it is expected that only one key exists in env.observation_spec.