# parallel_env_constructor

torchrl.trainers.helpers.parallel_env_constructor(*cfg: DictConfig*, ***kwargs*) → [ParallelEnv](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) | [EnvCreator](torchrl.envs.EnvCreator.html#torchrl.envs.EnvCreator)[[source]](../../_modules/torchrl/trainers/helpers/envs.html#parallel_env_constructor)

Returns a parallel environment from an argparse.Namespace built with the appropriate parser constructor.

Parameters:

- **cfg** (*DictConfig*) - config containing user-defined arguments
- **kwargs** - keyword arguments for the transformed_env_constructor method.