# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import deepcopy
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from typing import Callable, Optional, Union, Any, Sequence

from torchrl.data import NdUnboundedContinuousTensorSpec
from torchrl.envs import ParallelEnv
from torchrl.envs.common import EnvBase
from torchrl.envs.env_creator import env_creator, EnvCreator
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    CatTensors,
    DoubleToFloat,
    GrayScale,
    NoopResetEnv,
    ObservationNorm,
    Resize,
    RewardScaling,
    ToTensorImage,
    TransformedEnv,
    CenterCrop,
)
from torchrl.envs.transforms.transforms import (
    FlattenObservation,
    TensorDictPrimer,
)
from torchrl.record.recorder import VideoRecorder
from torchrl.trainers.loggers import Logger

__all__ = [
    "transformed_env_constructor",
    "parallel_env_constructor",
]

LIBS = {
    "gym": GymEnv,
    "dm_control": DMControlEnv,
}


def make_env_transforms(
    env,
    cfg,
    video_tag,
    logger,
    env_name,
    stats,
    norm_obs_only,
    env_library,
    action_dim_gsde,
    state_dim_gsde,
    batch_dims=0,
):
    env = TransformedEnv(env)

    from_pixels = cfg.from_pixels
    vecnorm = cfg.vecnorm
    norm_rewards = vecnorm and cfg.norm_rewards
    reward_scaling = cfg.reward_scaling
    reward_loc = cfg.reward_loc

    if len(video_tag):
        center_crop = cfg.center_crop
        if center_crop:
            center_crop = center_crop[0]
        env.append_transform(
            VideoRecorder(
                logger=logger,
                tag=f"{video_tag}_{env_name}_video",
                center_crop=center_crop,
            ),
        )

    if cfg.noops:
        env.append_transform(NoopResetEnv(cfg.noops))

    if from_pixels:
        if not cfg.catframes:
            raise RuntimeError(
                "this env builder currently only accepts positive catframes values"
                "when pixels are being used."
            )
        env.append_transform(ToTensorImage())
        if cfg.center_crop:
            env.append_transform(CenterCrop(*cfg.center_crop))
        env.append_transform(Resize(cfg.image_size, cfg.image_size))
        if cfg.grayscale:
            env.append_transform(GrayScale())
        env.append_transform(FlattenObservation(first_dim=batch_dims))
        env.append_transform(CatFrames(N=cfg.catframes, keys_in=["next_pixels"]))
        if stats is None:
            obs_stats = {"loc": 0.0, "scale": 1.0}
        else:
            obs_stats = stats
        obs_stats["standard_normal"] = True
        env.append_transform(ObservationNorm(**obs_stats, keys_in=["next_pixels"]))
    if norm_rewards:
        reward_scaling = 1.0
        reward_loc = 0.0
    if norm_obs_only:
        reward_scaling = 1.0
        reward_loc = 0.0
    if reward_scaling is not None:
        env.append_transform(RewardScaling(reward_loc, reward_scaling))

    double_to_float_list = []
    float_to_double_list = []
    if env_library is DMControlEnv:
        double_to_float_list += [
            "reward",
            "action",
        ]
        float_to_double_list += ["action"]  # DMControl requires double-precision
    env.append_transform(
        DoubleToFloat(keys_in=double_to_float_list, keys_inv_in=float_to_double_list)
    )

    default_dict = {
        # "prior_state": NdUnboundedContinuousTensorSpec(cfg.state_dim),
        "posterior_state": NdUnboundedContinuousTensorSpec(cfg.state_dim),
        "belief": NdUnboundedContinuousTensorSpec(cfg.rssm_hidden_dim),
        "action": deepcopy(env.action_spec),
    }
    env.append_transform(
        TensorDictPrimer(random=False, default_value=0, **default_dict)
    )
    env.append_transform(
        CatTensors(keys_in=["belief"], out_key="prev_belief", del_keys=False)
    )
    env.append_transform(
        CatTensors(keys_in=["action"], out_key="prev_action", del_keys=False)
    )
    env.append_transform(
        CatTensors(
            keys_in=["posterior_state"], out_key="prev_posterior_state", del_keys=False
        )
    )

    return env


def transformed_env_constructor(
    cfg: "DictConfig",  # noqa: F821
    video_tag: str = "",
    logger: Optional[Logger] = None,
    stats: Optional[dict] = None,
    norm_obs_only: bool = False,
    use_env_creator: bool = False,
    custom_env_maker: Optional[Callable] = None,
    custom_env: Optional[EnvBase] = None,
    return_transformed_envs: bool = True,
    action_dim_gsde: Optional[int] = None,
    state_dim_gsde: Optional[int] = None,
    batch_dims: Optional[int] = 0,
) -> Union[Callable, EnvCreator]:
    """
    Returns an environment creator from an argparse.Namespace built with the appropriate parser constructor.

    Args:
        cfg (DictConfig): a DictConfig containing the arguments of the script.
        video_tag (str, optional): video tag to be passed to the Logger object
        logger (Logger, optional): logger associated with the script
        stats (dict, optional): a dictionary containing the `loc` and `scale` for the `ObservationNorm` transform
        norm_obs_only (bool, optional): If `True` and `VecNorm` is used, the reward won't be normalized online.
            Default is `False`.
        use_env_creator (bool, optional): wheter the `EnvCreator` class should be used. By using `EnvCreator`,
            one can make sure that running statistics will be put in shared memory and accessible for all workers
            when using a `VecNorm` transform. Default is `True`.
        custom_env_maker (callable, optional): if your env maker is not part
            of torchrl env wrappers, a custom callable
            can be passed instead. In this case it will override the
            constructor retrieved from `args`.
        custom_env (EnvBase, optional): if an existing environment needs to be
            transformed_in, it can be passed directly to this helper. `custom_env_maker`
            and `custom_env` are exclusive features.
        return_transformed_envs (bool, optional): if True, a transformed_in environment
            is returned.
        action_dim_gsde (int, Optional): if gSDE is used, this can present the action dim to initialize the noise.
            Make sure this is indicated in environment executed in parallel.
        state_dim_gsde: if gSDE is used, this can present the state dim to initialize the noise.
            Make sure this is indicated in environment executed in parallel.
        batch_dims (int, optional): number of dimensions of a batch of data. If a single env is
            used, it should be 0 (default). If multiple envs are being transformed in parallel,
            it should be set to 1 (or the number of dims of the batch).
    """

    def make_transformed_env(**kwargs) -> TransformedEnv:
        env_name = cfg.env_name
        env_task = cfg.env_task
        env_library = LIBS[cfg.env_library]
        frame_skip = cfg.frame_skip
        from_pixels = cfg.from_pixels

        if custom_env is None and custom_env_maker is None:
            if isinstance(cfg.collector_devices, str):
                device = cfg.collector_devices
            elif isinstance(cfg.collector_devices, Sequence):
                device = cfg.collector_devices[0]
            else:
                raise ValueError(
                    "collector_devices must be either a string or a sequence of strings"
                )
            env_kwargs = {
                "env_name": env_name,
                "device": device,
                "frame_skip": frame_skip,
                "from_pixels": from_pixels or len(video_tag),
                "pixels_only": from_pixels,
            }
            if env_library is DMControlEnv:
                env_kwargs.update({"task_name": env_task})
            env_kwargs.update(kwargs)
            env = env_library(**env_kwargs)
        elif custom_env is None and custom_env_maker is not None:
            env = custom_env_maker(**kwargs)
        elif custom_env_maker is None and custom_env is not None:
            env = custom_env
        else:
            raise RuntimeError("cannot provive both custom_env and custom_env_maker")

        if not return_transformed_envs:
            return env

        return make_env_transforms(
            env,
            cfg,
            video_tag,
            logger,
            env_name,
            stats,
            norm_obs_only,
            env_library,
            action_dim_gsde,
            state_dim_gsde,
            batch_dims=batch_dims,
        )

    if use_env_creator:
        return env_creator(make_transformed_env)
    return make_transformed_env


def parallel_env_constructor(
    cfg: "DictConfig", **kwargs  # noqa: F821
) -> Union[ParallelEnv, EnvCreator]:
    """Returns a parallel environment from an argparse.Namespace built with the appropriate parser constructor.

    Args:
        cfg (DictConfig): config containing user-defined arguments
        kwargs: keyword arguments for the `transformed_env_constructor` method.
    """
    batch_transform = cfg.batch_transform
    if cfg.env_per_collector == 1:
        kwargs.update({"cfg": cfg, "use_env_creator": True})
        make_transformed_env = transformed_env_constructor(**kwargs)
        return make_transformed_env
    kwargs.update({"cfg": cfg, "use_env_creator": True})
    make_transformed_env = transformed_env_constructor(
        return_transformed_envs=not batch_transform, **kwargs
    )
    parallel_env = ParallelEnv(
        num_workers=cfg.env_per_collector,
        create_env_fn=make_transformed_env,
        create_env_kwargs=None,
        pin_memory=cfg.pin_memory,
    )
    if batch_transform:
        kwargs.update(
            {
                "cfg": cfg,
                "use_env_creator": False,
                "custom_env": parallel_env,
                "batch_dims": 1,
            }
        )
        env = transformed_env_constructor(**kwargs)()
        return env
    return parallel_env


@dataclass
class EnvConfig:
    env_library: str = "gym"
    # env_library used for the simulated environment. Default=gym
    env_name: str = "Humanoid-v2"
    # name of the environment to be created. Default=Humanoid-v2
    env_task: str = ""
    # task (if any) for the environment. Default=run
    from_pixels: bool = False
    # whether the environment output should be state vector(s) (default) or the pixels.
    frame_skip: int = 1
    # frame_skip for the environment. Note that this value does NOT impact the buffer size,
    # maximum steps per trajectory, frames per batch or any other factor in the algorithm,
    # e.g. if the total number of frames that has to be computed is 50e6 and the frame skip is 4
    # the actual number of frames retrieved will be 200e6. Default=1.
    reward_scaling: Optional[float] = None
    # scale of the reward.
    reward_loc: float = 0.0
    # location of the reward.
    init_env_steps: int = 1000
    # number of random steps to compute normalizing constants
    vecnorm: bool = False
    # Normalizes the environment observation and reward outputs with the running statistics obtained across processes.
    norm_rewards: bool = False
    # If True, rewards will be normalized on the fly. This may interfere with SAC update rule and should be used cautiously.
    norm_stats: bool = True
    # Deactivates the normalization based on random collection of data.
    noops: int = 0
    # number of random steps to do after reset. Default is 0
    catframes: int = 0
    # Number of frames to concatenate through time. Default is 0 (do not use CatFrames).
    center_crop: Any = dataclass_field(default_factory=lambda: [])
    # center crop size.
    grayscale: bool = True
    # Disables grayscale transform.
    max_frames_per_traj: int = 1000
    # Number of steps before a reset of the environment is called (if it has not been flagged as done before).
    batch_transform: bool = False
    # if True, the transforms will be applied to the parallel env, and not to each individual env.\
    image_size: int = 84
