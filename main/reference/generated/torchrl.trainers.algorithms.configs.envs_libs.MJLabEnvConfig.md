# torchrl.trainers.algorithms.configs.envs_libs.MJLabEnvConfig

*class*torchrl.trainers.algorithms.configs.envs_libs.MJLabEnvConfig(*_partial_: bool = False*, *task_id: str = '???'*, *cfg: Any = None*, *play: bool = False*, *num_envs: int | None = None*, *from_pixels: bool = False*, *pixels_only: bool = False*, *pixels_key: str = 'pixels'*, *pixels_sensor: str | None = None*, *render_mode: str | None = None*, *native_autoreset: bool = False*, *device: str | None = None*, *batch_size: list[int] | None = None*, *allow_done_after_reset: bool = False*, *num_workers: int = 1*, *_target_: str = 'torchrl.envs.libs.mjlab.MJLabEnv'*)[[source]](../../_modules/torchrl/trainers/algorithms/configs/envs_libs.html#MJLabEnvConfig)

Configuration for MJLabEnv environment.

Hydra configuration for [`MJLabEnv`](torchrl.envs.MJLabEnv.html#torchrl.envs.MJLabEnv).