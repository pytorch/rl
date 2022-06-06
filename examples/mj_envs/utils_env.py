# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from argparse import Namespace
from typing import Callable, Optional, Union

from torchrl.envs import DMControlEnv, GymEnv, RetroEnv, ParallelEnv
from torchrl.envs.common import _EnvClass
from torchrl.envs.env_creator import env_creator, EnvCreator
from torchrl.envs.transforms import (
    CatFrames,
    CatTensors,
    DoubleToFloat,
    FiniteTensorDictCheck,
    GrayScale,
    NoopResetEnv,
    ObservationNorm,
    Resize,
    RewardScaling,
    ToTensorImage,
    TransformedEnv,
    VecNorm,
    CenterCrop,
)
from torchrl.envs.transforms.transforms import gSDENoise, FlattenObservation
from torchrl.record.recorder import VideoRecorder
from torchrl.trainers.helpers.envs import LIBS


def make_env_transforms(
    env,
    args,
    video_tag,
    writer,
    env_name,
    stats_pixels,
    stats_states,
    norm_obs_only,
    env_library,
    action_dim_gsde,
    state_dim_gsde,
    batch_dims=0,
):
    env = TransformedEnv(env)

    from_pixels = args.from_pixels
    vecnorm = args.vecnorm
    norm_rewards = vecnorm and args.norm_rewards
    _norm_obs_only = norm_obs_only or not norm_rewards
    reward_scaling = args.reward_scaling
    reward_loc = args.reward_loc

    if len(video_tag):
        center_crop = args.center_crop
        if center_crop:
            center_crop = center_crop[0]
        env.append_transform(
            VideoRecorder(
                writer=writer,
                tag=f"{video_tag}_{env_name}_video",
                center_crop=center_crop,
            ),
        )

    if args.noops:
        env.append_transform(NoopResetEnv(env, args.noops))
    if from_pixels:
        if not args.catframes:
            raise RuntimeError(
                "this env builder currently only accepts positive catframes values"
                "when pixels are being used."
            )
        env.append_transform(ToTensorImage())
        if args.center_crop:
            env.append_transform(CenterCrop(*args.center_crop))
        env.append_transform(Resize(84, 84))
        if args.grayscale:
            env.append_transform(GrayScale())
        env.append_transform(FlattenObservation(first_dim=batch_dims))
        env.append_transform(CatFrames(N=args.catframes, keys=["next_pixels"]))
        if stats_pixels is None:
            obs_stats = {"loc": 0.0, "scale": 1.0}
        else:
            obs_stats = stats_pixels
        obs_stats["standard_normal"] = True
        env.append_transform(ObservationNorm(**obs_stats, keys=["next_pixels"]))
    if norm_rewards:
        reward_scaling = 1.0
        reward_loc = 0.0
    if norm_obs_only:
        reward_scaling = 1.0
        reward_loc = 0.0
    if reward_scaling is not None:
        env.append_transform(RewardScaling(reward_loc, reward_scaling))

    double_to_float_list = []
    if env_library is DMControlEnv:
        double_to_float_list += [
            "reward",
            "action",
        ]  # DMControl requires double-precision

    if not from_pixels or args.include_state:
        selected_keys = [
            key for key in env.observation_spec.keys() if "pixels" not in key
        ]

        # even if there is a single tensor, it'll be renamed in "next_observation_vector"
        out_key = "next_observation_vector"
        env.append_transform(CatTensors(keys=selected_keys, out_key=out_key))

        if hasattr(args, "catframes") and args.catframes:
            env.append_transform(
                CatFrames(N=args.catframes, keys=[out_key], cat_dim=-1)
            )

        if not vecnorm:
            if stats_states is None:
                obs_stats = {"loc": 0.0, "scale": 1.0}
            else:
                obs_stats = stats_states
            obs_stats["standard_normal"] = True
            env.append_transform(
                ObservationNorm(**obs_stats, keys=[out_key])
            )
        else:
            env.append_transform(
                VecNorm(
                    keys=[out_key, "reward"] if not _norm_obs_only else [out_key],
                    decay=0.9999,
                )
            )

        double_to_float_list.append(out_key)
        env.append_transform(DoubleToFloat(keys=double_to_float_list))


    else:
        env.append_transform(DoubleToFloat(keys=double_to_float_list))

    if hasattr(args, "gSDE") and args.gSDE:
        env.append_transform(
            gSDENoise(action_dim=action_dim_gsde, state_dim=state_dim_gsde)
        )

    env.append_transform(FiniteTensorDictCheck())
    return env


def transformed_env_constructor(
    args: Namespace,
    video_tag: str = "",
    writer: Optional["SummaryWriter"] = None,
    stats_pixels: Optional[dict] = None,
    stats_state: Optional[dict] = None,
    norm_obs_only: bool = False,
    use_env_creator: bool = True,
    custom_env_maker: Optional[Callable] = None,
    custom_env: Optional[_EnvClass] = None,
    return_transformed_envs: bool = True,
    action_dim_gsde: Optional[int] = None,
    state_dim_gsde: Optional[int] = None,
    batch_dims: Optional[int] = 0,
) -> Union[Callable, EnvCreator]:
    """
    Returns an environment creator from an argparse.Namespace built with the appropriate parser constructor.

    Args:
        args (argparse.Namespace): script arguments originating from the parser built with parser_env_args
        video_tag (str, optional): video tag to be passed to the SummaryWriter object
        writer (SummaryWriter, optional): tensorboard writer associated with the script
        stats_pixels (dict, optional): a dictionary containing the `loc` and `scale` for the `ObservationNorm` transform (pixels)
        stats_state (dict, optional): a dictionary containing the `loc` and `scale` for the `ObservationNorm` transform (state)
        norm_obs_only (bool, optional): If `True` and `VecNorm` is used, the reward won't be normalized online.
            Default is `False`.
        use_env_creator (bool, optional): wheter the `EnvCreator` class should be used. By using `EnvCreator`,
            one can make sure that running statistics will be put in shared memory and accessible for all workers
            when using a `VecNorm` transform. Default is `True`.
        custom_env_maker (callable, optional): if your env maker is not part
            of torchrl env wrappers, a custom callable
            can be passed instead. In this case it will override the
            constructor retrieved from `args`.
        custom_env (_EnvClass, optional): if an existing environment needs to be
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
        env_name = args.env_name
        env_task = args.env_task
        env_library = LIBS[args.env_library]
        frame_skip = args.frame_skip
        from_pixels = args.from_pixels

        if custom_env is None and custom_env_maker is None:
            env_kwargs = {
                "envname": env_name,
                "device": "cpu",
                "frame_skip": frame_skip,
                "from_pixels": from_pixels or len(video_tag),
                "pixels_only": from_pixels,
            }
            if env_library is DMControlEnv:
                env_kwargs.update({"taskname": env_task})
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
            args,
            video_tag,
            writer,
            env_name,
            stats_pixels,
            stats_state,
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
    args: Namespace, **kwargs
) -> Union[ParallelEnv, EnvCreator]:
    """Returns a parallel environment from an argparse.Namespace built with the appropriate parser constructor.

    Args:
        args (argparse.Namespace): script arguments originating from the parser built with parser_env_args
        kwargs: keyword arguments for the `transformed_env_constructor` method.
    """
    batch_transform = args.batch_transform
    if args.env_per_collector == 1:
        kwargs.update({"args": args, "use_env_creator": True})
        make_transformed_env = transformed_env_constructor(**kwargs)
        return make_transformed_env
    kwargs.update({"args": args, "use_env_creator": True})
    make_transformed_env = transformed_env_constructor(
        return_transformed_envs=not batch_transform, **kwargs
    )
    parallel_env = ParallelEnv(
        num_workers=args.env_per_collector,
        create_env_fn=make_transformed_env,
        create_env_kwargs=None,
        pin_memory=args.pin_memory,
    )
    if batch_transform:
        kwargs.update(
            {
                "args": args,
                "use_env_creator": False,
                "custom_env": parallel_env,
                "batch_dims": 1,
            }
        )
        env = transformed_env_constructor(**kwargs)()
        return env
    return parallel_env
