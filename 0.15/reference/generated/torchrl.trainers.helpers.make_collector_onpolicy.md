# make_collector_onpolicy

torchrl.trainers.helpers.make_collector_onpolicy(*make_env: Callable[[], [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)]*, *actor_model_explore: TensorDictModuleWrapper | ProbabilisticTensorDictSequential*, *cfg: DictConfig*, *make_env_kwargs: dict | None = None*) → [BaseCollector](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)[[source]](../../_modules/torchrl/trainers/helpers/collectors.html#make_collector_onpolicy)

Makes a collector in on-policy settings.

Parameters:

- **make_env** (*Callable*) - environment creator
- **actor_model_explore** ([*SafeModule*](torchrl.modules.tensordict_module.SafeModule.html#torchrl.modules.tensordict_module.SafeModule)) - Model instance used for evaluation and exploration update
- **cfg** (*DictConfig*) - config for creating collector object
- **make_env_kwargs** (*dict*) - kwargs for the env creator