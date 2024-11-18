# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from copy import copy
from dataclasses import dataclass, field as dataclass_field
from typing import Any, Callable, Optional, Sequence, Tuple, Union

import torch

from torchrl._utils import logger as torchrl_logger, VERBOSE
from torchrl.envs import ParallelEnv
from torchrl.envs.common import EnvBase
from torchrl.envs.env_creator import env_creator, EnvCreator
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import GymEnv
from torchrl.envs.transforms import (
    CatFrames,
    CatTensors,
    CenterCrop,
    Compose,
    DoubleToFloat,
    GrayScale,
    NoopResetEnv,
    ObservationNorm,
    Resize,
    RewardScaling,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
)
from torchrl.envs.transforms.transforms import (
    FlattenObservation,
    gSDENoise,
    InitTracker,
    StepCounter,
)
from torchrl.record.loggers import Logger
from torchrl.record.recorder import VideoRecorder

LIBS = {
    "gym": GymEnv,
    "dm_control": DMControlEnv,
}


def correct_for_frame_skip(cfg: "DictConfig") -> "DictConfig":  # noqa: F821
    """Correct the arguments for the input frame_skip, by dividing all the arguments that reflect a count of frames by the frame_skip.

    This is aimed at avoiding unknowingly over-sampling from the environment, i.e. targetting a total number of frames
    of 1M but actually collecting frame_skip * 1M frames.

    Args:
        cfg (DictConfig): DictConfig containing some frame-counting argument, including:
            "max_frames_per_traj", "total_frames", "frames_per_batch", "record_frames", "annealing_frames",
            "init_random_frames", "init_env_steps"

    Returns:
         the input DictConfig, modified in-place.

    """
    # Adapt all frame counts wrt frame_skip
    if cfg.frame_skip != 1:
        fields = [
            "max_frames_per_traj",
            "total_frames",
            "frames_per_batch",
            "record_frames",
            "annealing_frames",
            "init_random_frames",
            "init_env_steps",
            "noops",
        ]
        for field in fields:
            if hasattr(cfg, field):
                setattr(cfg, field, getattr(cfg, field) // cfg.frame_skip)
    return cfg


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
    obs_norm_state_dict=None,
):
    """Creates the typical transforms for and env."""
    env = TransformedEnv(env)

    from_pixels = cfg.from_pixels
    vecnorm = cfg.vecnorm
    norm_rewards = vecnorm and cfg.norm_rewards
    _norm_obs_only = norm_obs_only or not norm_rewards
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

    if from_pixels:
        if not cfg.catframes:
            raise RuntimeError(
                "this env builder currently only accepts positive catframes values "
                "when pixels are being used."
            )
        env.append_transform(ToTensorImage())
        if cfg.center_crop:
            env.append_transform(CenterCrop(*cfg.center_crop))
        env.append_transform(Resize(cfg.image_size, cfg.image_size))
        if cfg.grayscale:
            env.append_transform(GrayScale())
        env.append_transform(FlattenObservation(0, -3, allow_positive_dim=True))
        env.append_transform(CatFrames(N=cfg.catframes, in_keys=["pixels"], dim=-3))
        if stats is None and obs_norm_state_dict is None:
            obs_stats = {}
        elif stats is None:
            obs_stats = copy(obs_norm_state_dict)
        else:
            obs_stats = copy(stats)
        obs_stats["standard_normal"] = True
        obs_norm = ObservationNorm(**obs_stats, in_keys=["pixels"])
        env.append_transform(obs_norm)
    if norm_rewards:
        reward_scaling = 1.0
        reward_loc = 0.0
    if norm_obs_only:
        reward_scaling = 1.0
        reward_loc = 0.0
    if reward_scaling is not None:
        env.append_transform(RewardScaling(reward_loc, reward_scaling))

    if not from_pixels:
        selected_keys = [
            key
            for key in env.observation_spec.keys(True, True)
            if ("pixels" not in key) and (key not in env.state_spec.keys(True, True))
        ]

        # even if there is a single tensor, it'll be renamed in "observation_vector"
        out_key = "observation_vector"
        env.append_transform(CatTensors(in_keys=selected_keys, out_key=out_key))

        if not vecnorm:
            if stats is None and obs_norm_state_dict is None:
                _stats = {}
            elif stats is None:
                _stats = copy(obs_norm_state_dict)
            else:
                _stats = copy(stats)
            _stats.update({"standard_normal": True})
            obs_norm = ObservationNorm(
                **_stats,
                in_keys=[out_key],
            )
            env.append_transform(obs_norm)
        else:
            env.append_transform(
                VecNorm(
                    in_keys=[out_key, "reward"] if not _norm_obs_only else [out_key],
                    decay=0.9999,
                )
            )

        env.append_transform(DoubleToFloat())

        if hasattr(cfg, "catframes") and cfg.catframes:
            env.append_transform(CatFrames(N=cfg.catframes, in_keys=[out_key], dim=-1))

    else:
        env.append_transform(DoubleToFloat())

    if hasattr(cfg, "gSDE") and cfg.gSDE:
        env.append_transform(
            gSDENoise(action_dim=action_dim_gsde, state_dim=state_dim_gsde)
        )

    env.append_transform(StepCounter())
    env.append_transform(InitTracker())

    return env


def get_norm_state_dict(env):
    """Gets the normalization loc and scale from the env state_dict."""
    sd = env.state_dict()
    sd = {
        key: val
        for key, val in sd.items()
        if key.endswith("loc") or key.endswith("scale")
    }
    return sd


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
    obs_norm_state_dict: Optional[dict] = None,
) -> Union[Callable, EnvCreator]:
    """Returns an environment creator from an argparse.Namespace built with the appropriate parser constructor.

    Args:
        cfg (DictConfig): a DictConfig containing the arguments of the script.
        video_tag (str, optional): video tag to be passed to the Logger object
        logger (Logger, optional): logger associated with the script
        stats (dict, optional): a dictionary containing the :obj:`loc` and :obj:`scale` for the `ObservationNorm` transform
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
        return_transformed_envs (bool, optional): if ``True``, a transformed_in environment
            is returned.
        action_dim_gsde (int, Optional): if gSDE is used, this can present the action dim to initialize the noise.
            Make sure this is indicated in environment executed in parallel.
        state_dim_gsde: if gSDE is used, this can present the state dim to initialize the noise.
            Make sure this is indicated in environment executed in parallel.
        batch_dims (int, optional): number of dimensions of a batch of data. If a single env is
            used, it should be 0 (default). If multiple envs are being transformed in parallel,
            it should be set to 1 (or the number of dims of the batch).
        obs_norm_state_dict (dict, optional): the state_dict of the ObservationNorm transform to be loaded into the
            environment
    """

    def make_transformed_env(**kwargs) -> TransformedEnv:
        env_name = cfg.env_name
        env_task = cfg.env_task
        env_library = LIBS[cfg.env_library]
        frame_skip = cfg.frame_skip
        from_pixels = cfg.from_pixels
        categorical_action_encoding = cfg.categorical_action_encoding

        if custom_env is None and custom_env_maker is None:
            if isinstance(cfg.collector_device, str):
                device = cfg.collector_device
            elif isinstance(cfg.collector_device, Sequence):
                device = cfg.collector_device[0]
            else:
                raise ValueError(
                    "collector_device must be either a string or a sequence of strings"
                )
            env_kwargs = {
                "env_name": env_name,
                "device": device,
                "frame_skip": frame_skip,
                "from_pixels": from_pixels or len(video_tag),
                "pixels_only": from_pixels,
            }
            if env_library is GymEnv:
                env_kwargs.update(
                    {"categorical_action_encoding": categorical_action_encoding}
                )
            elif categorical_action_encoding:
                raise NotImplementedError(
                    "categorical_action_encoding=True is currently only compatible with GymEnvs."
                )
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

        if cfg.noops and custom_env is None:
            # this is a bit hacky: if custom_env is not None, it is probably a ParallelEnv
            # that already has its NoopResetEnv set for the contained envs.
            # There is a risk however that we're just skipping the NoopsReset instantiation
            env = TransformedEnv(env, NoopResetEnv(cfg.noops))
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
            obs_norm_state_dict=obs_norm_state_dict,
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
    if not batch_transform:
        raise NotImplementedError(
            "batch_transform must be set to True for the recorder to be synced "
            "with the collection envs."
        )
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


@torch.no_grad()
def get_stats_random_rollout(
    cfg: "DictConfig",  # noqa: F821
    proof_environment: EnvBase = None,
    key: Optional[str] = None,
):
    """Gathers stas (loc and scale) from an environment using random rollouts.

    Args:
        cfg (DictConfig): a config object with `init_env_steps` field, indicating
            the total number of frames to be collected to compute the stats.
        proof_environment (EnvBase instance, optional): if provided, this env will
            be used ot execute the rollouts. If not, it will be created using
            the cfg object.
        key (str, optional): if provided, the stats of this key will be gathered.
            If not, it is expected that only one key exists in `env.observation_spec`.

    """
    proof_env_is_none = proof_environment is None
    if proof_env_is_none:
        proof_environment = transformed_env_constructor(
            cfg=cfg, use_env_creator=False, stats={"loc": 0.0, "scale": 1.0}
        )()

    if VERBOSE:
        torchrl_logger.info("computing state stats")
    if not hasattr(cfg, "init_env_steps"):
        raise AttributeError("init_env_steps missing from arguments.")

    n = 0
    val_stats = []
    while n < cfg.init_env_steps:
        _td_stats = proof_environment.rollout(max_steps=cfg.init_env_steps)
        n += _td_stats.numel()
        val = _td_stats.get(key).cpu()
        val_stats.append(val)
        del _td_stats, val
    val_stats = torch.cat(val_stats, 0)

    if key is None:
        keys = list(proof_environment.observation_spec.keys(True, True))
        key = keys.pop()
        if len(keys):
            raise RuntimeError(
                f"More than one key exists in the observation_specs: {[key] + keys} were found, "
                "thus get_stats_random_rollout cannot infer which to compute the stats of."
            )

    if key == "pixels":
        m = val_stats.mean()
        s = val_stats.std()
    else:
        m = val_stats.mean(dim=0)
        s = val_stats.std(dim=0)
    m[s == 0] = 0.0
    s[s == 0] = 1.0

    if VERBOSE:
        torchrl_logger.info(
            f"stats computed for {val_stats.numel()} steps. Got: \n"
            f"loc = {m}, \n"
            f"scale = {s}"
        )
    if not torch.isfinite(m).all():
        raise RuntimeError("non-finite values found in mean")
    if not torch.isfinite(s).all():
        raise RuntimeError("non-finite values found in sd")
    stats = {"loc": m, "scale": s}
    if proof_env_is_none:
        proof_environment.close()
        if (
            proof_environment.device != torch.device("cpu")
            and torch.cuda.device_count() > 0
        ):
            torch.cuda.empty_cache()
        del proof_environment
    return stats


def initialize_observation_norm_transforms(
    proof_environment: EnvBase,
    num_iter: int = 1000,
    key: Union[str, Tuple[str, ...]] = None,
):
    """Calls :obj:`ObservationNorm.init_stats` on all uninitialized :obj:`ObservationNorm` instances of a :obj:`TransformedEnv`.

    If an :obj:`ObservationNorm` already has non-null :obj:`loc` or :obj:`scale`, a call to :obj:`initialize_observation_norm_transforms` will be a no-op.
    Similarly, if the transformed environment does not contain any :obj:`ObservationNorm`, a call to this function will have no effect.
    If no key is provided but the observations of the :obj:`EnvBase` contains more than one key, an exception will
    be raised.

    Args:
        proof_environment (EnvBase instance, optional): if provided, this env will
            be used ot execute the rollouts. If not, it will be created using
            the cfg object.
        num_iter (int): Number of iterations used for initializing the :obj:`ObservationNorms`
        key (str, optional): if provided, the stats of this key will be gathered.
            If not, it is expected that only one key exists in `env.observation_spec`.

    """
    if not isinstance(proof_environment.transform, Compose) and not isinstance(
        proof_environment.transform, ObservationNorm
    ):
        return

    if key is None:
        keys = list(proof_environment.base_env.observation_spec.keys(True, True))
        key = keys.pop()
        if len(keys):
            raise RuntimeError(
                f"More than one key exists in the observation_specs: {[key] + keys} were found, "
                "thus initialize_observation_norm_transforms cannot infer which to compute the stats of."
            )

    if isinstance(proof_environment.transform, Compose):
        for transform in proof_environment.transform:
            if isinstance(transform, ObservationNorm) and not transform.initialized:
                transform.init_stats(num_iter=num_iter, key=key)
    elif not proof_environment.transform.initialized:
        proof_environment.transform.init_stats(num_iter=num_iter, key=key)


def retrieve_observation_norms_state_dict(proof_environment: TransformedEnv):
    """Traverses the transforms of the environment and retrieves the :obj:`ObservationNorm` state dicts.

    Returns a list of tuple (idx, state_dict) for each :obj:`ObservationNorm` transform in proof_environment
    If the environment transforms do not contain any :obj:`ObservationNorm`, returns an empty list

    Args:
        proof_environment (EnvBase instance, optional): the :obj:``TransformedEnv` to retrieve the :obj:`ObservationNorm`
            state dict from
    """
    obs_norm_state_dicts = []

    if isinstance(proof_environment.transform, Compose):
        for idx, transform in enumerate(proof_environment.transform):
            if isinstance(transform, ObservationNorm):
                obs_norm_state_dicts.append((idx, transform.state_dict()))

    if isinstance(proof_environment.transform, ObservationNorm):
        obs_norm_state_dicts.append((0, proof_environment.transform.state_dict()))

    return obs_norm_state_dicts


@dataclass
class EnvConfig:
    """Environment config struct."""

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
    # if ``True``, the transforms will be applied to the parallel env, and not to each individual env.\
    image_size: int = 84
    # if True and environment has discrete action space, then it is encoded as categorical values rather than one-hot.
    categorical_action_encoding: bool = False
