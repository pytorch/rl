# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import os
from pathlib import Path

import torch.nn

import torch.optim
from lamb import Lamb
from tensordict.nn import TensorDictModule

from torchrl.collectors import SyncDataCollector
from torchrl.data import (
    LazyMemmapStorage,
    RoundRobinWriter,
    TensorDictReplayBuffer,
    TensorStorage,
)
from torchrl.data.datasets.d4rl import D4RLExperienceReplay
from torchrl.data.replay_buffers import RandomSampler
from torchrl.envs import (
    CatFrames,
    Compose,
    DoubleToFloat,
    EnvCreator,
    ExcludeTransform,
    ObservationNorm,
    ParallelEnv,
    RandomCropTensorDict,
    RenameTransform,
    Reward2GoTransform,
    RewardScaling,
    RewardSum,
    TargetReturn,
    TensorDictPrimer,
    TransformedEnv,
    UnsqueezeTransform,
)
from torchrl.envs.libs.dm_control import DMControlEnv
from torchrl.envs.libs.gym import set_gym_backend
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import (
    DTActor,
    OnlineDTActor,
    ProbabilisticActor,
    TanhDelta,
    TanhNormal,
)

from torchrl.objectives import DTLoss, OnlineDTLoss
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.envs import LIBS

# ====================================================================
# Environment utils
# -----------------


def make_base_env(env_cfg, from_pixels=False, device=None):
    set_gym_backend(env_cfg.backend).set()

    env_library = LIBS[env_cfg.library]
    env_name = env_cfg.name
    frame_skip = env_cfg.frame_skip

    env_kwargs = {
        "env_name": env_name,
        "frame_skip": frame_skip,
        "from_pixels": from_pixels,
        "pixels_only": False,
    }
    if env_library is DMControlEnv:
        env_task = env_cfg.task
        env_kwargs.update({"task_name": env_task})
    env = env_library(**env_kwargs, device=device)
    return env


def make_transformed_env(base_env, env_cfg, obs_loc, obs_std, train=False):
    transformed_env = TransformedEnv(base_env)
    transformed_env.append_transform(
        RewardScaling(
            loc=0,
            scale=env_cfg.reward_scaling,
            in_keys=["reward"],
            standard_normal=False,
        )
    )
    if train:
        transformed_env.append_transform(
            TargetReturn(
                env_cfg.collect_target_return * env_cfg.reward_scaling,
                out_keys=["return_to_go"],
                mode=env_cfg.target_return_mode,
            )
        )
    else:
        transformed_env.append_transform(
            TargetReturn(
                env_cfg.eval_target_return * env_cfg.reward_scaling,
                out_keys=["return_to_go"],
                mode=env_cfg.target_return_mode,
            )
        )

    # copy action from the input tensordict to the output
    transformed_env.append_transform(TensorDictPrimer(base_env.full_action_spec))

    transformed_env.append_transform(DoubleToFloat())
    obsnorm = ObservationNorm(
        loc=obs_loc, scale=obs_std, in_keys="observation", standard_normal=True
    )
    transformed_env.append_transform(obsnorm)
    transformed_env.append_transform(
        UnsqueezeTransform(
            -2,
            in_keys=["observation", "action", "return_to_go"],
            out_keys=["observation_cat", "action_cat", "return_to_go_cat"],
        )
    )
    transformed_env.append_transform(
        CatFrames(
            in_keys=["observation_cat", "action_cat", "return_to_go_cat"],
            N=env_cfg.stacked_frames,
            dim=-2,
            padding="constant",
        )
    )

    if train:
        transformed_env.append_transform(RewardSum())

    return transformed_env


def make_parallel_env(
    env_cfg, obs_loc, obs_std, train=False, from_pixels=False, device=None
):
    if train:
        num_envs = env_cfg.num_train_envs
    else:
        num_envs = env_cfg.num_eval_envs

    def make_env():
        with set_gym_backend(env_cfg.backend):
            return make_base_env(env_cfg, from_pixels=from_pixels, device="cpu")

    env = make_transformed_env(
        ParallelEnv(
            num_envs, EnvCreator(make_env), serial_for_single=True, device=device
        ),
        env_cfg,
        obs_loc,
        obs_std,
        train,
    )
    env.start()
    return env


def make_env(env_cfg, obs_loc, obs_std, train=False, from_pixels=False, device=None):
    return make_parallel_env(
        env_cfg,
        obs_loc,
        obs_std,
        train=train,
        from_pixels=from_pixels,
        device=device,
    )


# ====================================================================
# Collector and replay buffer
# ---------------------------


def make_collector(cfg, policy):
    exclude_target_return = ExcludeTransform(
        "return_to_go",
        ("next", "return_to_go"),
        ("next", "action"),
        ("next", "observation"),
        "scale",
        "loc",
    )
    cat = CatFrames(
        in_keys=["action"], out_keys=["action_cat"], N=20, dim=-2, padding="constant"
    )
    transforms = Compose(
        exclude_target_return,
        cat,
    )
    collector_cfg = cfg.collector
    collector_class = SyncDataCollector
    collector = collector_class(
        make_env(cfg.env, train=True),
        policy,
        frames_per_batch=collector_cfg.frames_per_batch,
        total_frames=collector_cfg.total_frames,
        device=collector_cfg.devices,
        max_frames_per_traj=collector_cfg.max_frames_per_traj,
        postproc=transforms,
    )
    return collector


def make_offline_replay_buffer(rb_cfg, reward_scaling):
    r2g = Reward2GoTransform(
        gamma=1.0,
        in_keys=[("next", "reward"), "reward"],
        out_keys=[("next", "return_to_go"), "return_to_go"],
    )
    reward_scale = RewardScaling(
        loc=0,
        scale=reward_scaling,
        in_keys=[("next", "return_to_go"), "return_to_go"],
        standard_normal=False,
    )
    crop_seq = RandomCropTensorDict(sub_seq_len=rb_cfg.stacked_frames, sample_dim=-1)
    d2f = DoubleToFloat()
    rename = RenameTransform(
        in_keys=[
            "action",
            "observation",
            "return_to_go",
            ("next", "return_to_go"),
            ("next", "observation"),
        ],
        out_keys=[
            "action_cat",
            "observation_cat",
            "return_to_go_cat",
            ("next", "return_to_go_cat"),
            ("next", "observation_cat"),
        ],
    )
    exclude = ExcludeTransform(
        "terminal",
        "info",
        ("next", "timeout"),
        ("next", "terminal"),
        ("next", "observation"),
        ("next", "info"),
    )

    transforms = Compose(
        r2g,
        crop_seq,
        reward_scale,
        d2f,
        rename,
        exclude,
    )
    data = D4RLExperienceReplay(
        dataset_id=rb_cfg.dataset,
        split_trajs=True,
        batch_size=rb_cfg.batch_size,
        sampler=RandomSampler(),  # SamplerWithoutReplacement(drop_last=False),
        transform=None,
        use_truncated_as_done=True,
        direct_download=True,
        prefetch=4,
        writer=RoundRobinWriter(),
        root=Path(os.environ["HOME"]) / ".cache" / "torchrl" / "data" / "d4rl",
    )

    # since we're not extending the data, adding keys can only be done via
    # the creation of a new storage
    data_memmap = data[:]
    with data_memmap.unlock_():
        data_memmap = r2g.inv(data_memmap)
        data._storage = TensorStorage(data_memmap)

    loc = data[:]["observation"].flatten(0, -2).mean(axis=0).float()
    std = data[:]["observation"].flatten(0, -2).std(axis=0).float()

    obsnorm = ObservationNorm(
        loc=loc,
        scale=std,
        in_keys=["observation_cat", ("next", "observation_cat")],
        standard_normal=True,
    )
    for t in transforms:
        data.append_transform(t)
    data.append_transform(obsnorm)
    return data, loc, std


def make_online_replay_buffer(offline_buffer, rb_cfg, reward_scaling=0.001):
    r2g = Reward2GoTransform(gamma=1.0, out_keys=["return_to_go"])
    reward_scale = RewardScaling(
        loc=0,
        scale=reward_scaling,
        in_keys=["return_to_go"],
        out_keys=["return_to_go"],
        standard_normal=False,
    )
    catframes = CatFrames(
        in_keys=["return_to_go"],
        out_keys=["return_to_go_cat"],
        N=rb_cfg.stacked_frames,
        dim=-2,
        padding="constant",
        as_inverse=True,
    )
    transforms = Compose(
        r2g,
        reward_scale,
        catframes,
    )
    storage = LazyMemmapStorage(
        max_size=rb_cfg.capacity,
        scratch_dir=rb_cfg.scratch_dir,
        device=rb_cfg.device,
    )

    replay_buffer = TensorDictReplayBuffer(
        pin_memory=False,
        prefetch=rb_cfg.prefetch,
        storage=storage,
        batch_size=rb_cfg.batch_size,
    )
    # init buffer with offline data
    offline_data = offline_buffer[:100000]
    offline_data.del_("index")
    replay_buffer.extend(offline_data.clone().detach().to_tensordict())
    # add transforms after offline data extension to not trigger reward-to-go calculation
    replay_buffer.append_transform(transforms)

    return replay_buffer


# ====================================================================
# Model
# -----


def make_odt_model(cfg, device: torch.device | None = None) -> TensorDictModule:
    env_cfg = cfg.env
    proof_environment = make_transformed_env(
        make_base_env(env_cfg), env_cfg, obs_loc=0, obs_std=1
    )

    action_spec = proof_environment.action_spec_unbatched
    for key, value in proof_environment.observation_spec_unbatched.items():
        if key == "observation":
            state_dim = value.shape[-1]
    in_keys = [
        "observation_cat",
        "action_cat",
        "return_to_go_cat",
    ]

    actor_net = OnlineDTActor(
        state_dim=state_dim,
        action_dim=action_spec.shape[-1],
        transformer_config=cfg.transformer,
        device=device,
    )

    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys,
        out_keys=[
            "loc",
            "scale",
        ],
    )
    dist_class = TanhNormal
    dist_kwargs = {
        "low": -torch.ones((), device=device),
        "high": torch.ones((), device=device),
        "tanh_loc": False,
        "upscale": torch.full((), 5, device=device),
        # "safe_tanh": not cfg.compile.compile,
    }

    actor = ProbabilisticActor(
        spec=action_spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        cache_dist=False,
        return_log_prob=False,
    )

    # init the lazy layers
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = proof_environment.rollout(max_steps=100)
        td["action"] = td["next", "action"]
        actor(td.to(device))

    return actor


def make_dt_model(cfg, device: torch.device | None = None):
    env_cfg = cfg.env
    proof_environment = make_transformed_env(
        make_base_env(env_cfg), env_cfg, obs_loc=0, obs_std=1
    )

    action_spec = proof_environment.action_spec_unbatched
    in_keys = [
        "observation_cat",
        "action_cat",
        "return_to_go_cat",
    ]

    actor_net = DTActor(
        state_dim=proof_environment.observation_spec_unbatched["observation"].shape[-1],
        action_dim=action_spec.shape[-1],
        transformer_config=cfg.transformer,
        device=device,
    )

    actor_module = TensorDictModule(
        actor_net,
        in_keys=in_keys,
        out_keys=["param"],
    )
    dist_class = TanhDelta
    dist_kwargs = {
        "low": action_spec.space.low.to(device),
        "high": action_spec.space.high.to(device),
        "safe": not cfg.compile.compile,
    }

    actor = ProbabilisticActor(
        spec=action_spec.to(device),
        in_keys=["param"],
        out_keys=["action"],
        module=actor_module,
        distribution_class=dist_class,
        distribution_kwargs=dist_kwargs,
        cache_dist=False,
        return_log_prob=False,
    )

    # init the lazy layers
    with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
        td = proof_environment.fake_tensordict()
        td = td.expand((100, *td.shape))
        td["action"] = td["next", "action"]
        actor(td.to(device))

    return actor


# ====================================================================
# Online Decision Transformer Loss
# ---------


def make_odt_loss(loss_cfg, actor_network):
    loss = OnlineDTLoss(
        actor_network,
        alpha_init=loss_cfg.alpha_init,
        target_entropy=loss_cfg.target_entropy,
    )
    loss.set_keys(action_target="action_cat")
    return loss


def make_dt_loss(loss_cfg, actor_network, device: torch.device | None = None):
    loss = DTLoss(
        actor_network,
        loss_function=loss_cfg.loss_function,
        device=device,
    )
    loss.set_keys(action_target="action_cat")
    return loss


def make_odt_optimizer(optim_cfg, loss_module):
    if optim_cfg.optimizer == "lamb":
        dt_optimizer = Lamb(
            loss_module.actor_network_params.flatten_keys().values(),
            lr=torch.as_tensor(
                optim_cfg.lr, device=next(loss_module.parameters()).device
            ),
            weight_decay=optim_cfg.weight_decay,
            eps=1.0e-8,
        )
    elif optim_cfg.optimizer == "adam":
        dt_optimizer = torch.optim.Adam(
            loss_module.actor_network_params.flatten_keys().values(),
            lr=torch.as_tensor(
                optim_cfg.lr, device=next(loss_module.parameters()).device
            ),
            weight_decay=optim_cfg.weight_decay,
            eps=1.0e-8,
        )

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        dt_optimizer, lambda steps: min((steps + 1) / optim_cfg.warmup_steps, 1)
    )

    log_temp_optimizer = torch.optim.Adam(
        [loss_module.log_alpha],
        lr=torch.as_tensor(1e-4, device=next(loss_module.parameters()).device),
        betas=[0.9, 0.999],
    )

    return dt_optimizer, log_temp_optimizer, scheduler


def make_dt_optimizer(optim_cfg, loss_module, device):
    dt_optimizer = torch.optim.Adam(
        loss_module.actor_network_params.flatten_keys().values(),
        lr=torch.tensor(optim_cfg.lr, device=device),
        weight_decay=optim_cfg.weight_decay,
        eps=1.0e-8,
    )
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        dt_optimizer, lambda steps: min((steps + 1) / optim_cfg.warmup_steps, 1)
    )

    return dt_optimizer, scheduler


# ====================================================================
# Logging and recording
# ---------------------


def make_logger(cfg):
    if not cfg.logger.backend:
        return None
    exp_name = generate_exp_name(cfg.logger.model_name, cfg.logger.exp_name)
    logger = get_logger(
        cfg.logger.backend,
        logger_name=cfg.logger.model_name,
        experiment_name=exp_name,
        wandb_kwargs={
            "config": dict(cfg),
            "project": cfg.logger.project_name,
            "group": cfg.logger.group_name,
        },
    )
    return logger


# ====================================================================
# General utils
# ---------


def log_metrics(logger, metrics, step):
    for metric_name, metric_value in metrics.items():
        logger.log_scalar(metric_name, metric_value, step)


def dump_video(module):
    if isinstance(module, VideoRecorder):
        module.dump()
