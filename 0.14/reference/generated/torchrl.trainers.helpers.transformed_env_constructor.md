# transformed_env_constructor

torchrl.trainers.helpers.transformed_env_constructor(*cfg: DictConfig*, *video_tag: str = ''*, *logger: Logger | None = None*, *stats: dict | None = None*, *norm_obs_only: bool = False*, *use_env_creator: bool = False*, *custom_env_maker: Callable | None = None*, *custom_env: [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) | None = None*, *return_transformed_envs: bool = True*, *action_dim_gsde: int | None = None*, *state_dim_gsde: int | None = None*, *batch_dims: int | None = 0*, *obs_norm_state_dict: dict | None = None*) → Callable | [EnvCreator](torchrl.envs.EnvCreator.html#torchrl.envs.EnvCreator)[[source]](../../_modules/torchrl/trainers/helpers/envs.html#transformed_env_constructor)

Returns an environment creator from an argparse.Namespace built with the appropriate parser constructor.

Parameters:

- **cfg** (*DictConfig*) - a DictConfig containing the arguments of the script.
- **video_tag** (*str**,**optional*) - video tag to be passed to the Logger object
- **logger** (*Logger**,**optional*) - logger associated with the script
- **stats** (*dict**,**optional*) - a dictionary containing the `loc` and `scale` for the ObservationNorm transform
- **norm_obs_only** (*bool**,**optional*) - If True and VecNorm is used, the reward won't be normalized online.
Default is False.
- **use_env_creator** (*bool**,**optional*) - whether the EnvCreator class should be used. By using EnvCreator,
one can make sure that running statistics will be put in shared memory and accessible for all workers
when using a VecNorm transform. Default is True.
- **custom_env_maker** (*callable**,**optional*) - if your env maker is not part
of torchrl env wrappers, a custom callable
can be passed instead. In this case it will override the
constructor retrieved from args.
- **custom_env** ([*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)*,**optional*) - if an existing environment needs to be
transformed_in, it can be passed directly to this helper. custom_env_maker
and custom_env are exclusive features.
- **return_transformed_envs** (*bool**,**optional*) - if `True`, a transformed_in environment
is returned.
- **action_dim_gsde** (*int**,**Optional*) - if gSDE is used, this can present the action dim to initialize the noise.
Make sure this is indicated in environment executed in parallel.
- **state_dim_gsde** - if gSDE is used, this can present the state dim to initialize the noise.
Make sure this is indicated in environment executed in parallel.
- **batch_dims** (*int**,**optional*) - number of dimensions of a batch of data. If a single env is
used, it should be 0 (default). If multiple envs are being transformed in parallel,
it should be set to 1 (or the number of dims of the batch).
- **obs_norm_state_dict** (*dict**,**optional*) - the state_dict of the ObservationNorm transform to be loaded into the
environment