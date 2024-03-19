# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid
from datetime import datetime

import hydra
import torch.cuda
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import OrnsteinUhlenbeckProcessWrapper
from torchrl.record import VideoRecorder
from torchrl.record.loggers import get_logger
from utils import (
    correct_for_frame_skip,
    get_norm_state_dict,
    initialize_observation_norm_transforms,
    make_collector_offpolicy,
    make_redq_loss,
    make_redq_model,
    make_replay_buffer,
    make_trainer,
    parallel_env_constructor,
    retrieve_observation_norms_state_dict,
    transformed_env_constructor,
)

DEFAULT_REWARD_SCALING = {
    "Hopper-v1": 5,
    "Walker2d-v1": 5,
    "HalfCheetah-v1": 5,
    "cheetah": 5,
    "Ant-v2": 5,
    "Humanoid-v2": 20,
    "humanoid": 100,
}


@hydra.main(version_base="1.1", config_path="", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821

    cfg = correct_for_frame_skip(cfg)

    if not isinstance(cfg.env.reward_scaling, float):
        cfg.env.reward_scaling = DEFAULT_REWARD_SCALING.get(cfg.env.name, 5.0)
        cfg.env.reward_loc = 0.0

    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    exp_name = "_".join(
        [
            "REDQ",
            cfg.logger.exp_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        ]
    )

    logger = get_logger(
        logger_type=cfg.logger.backend,
        logger_name="redq_logging",
        experiment_name=exp_name,
        wandb_kwargs={
            "mode": cfg.logger.mode,
            "config": dict(cfg),
            "project": cfg.logger.project_name,
            "group": cfg.logger.group_name,
        },
    )
    video_tag = exp_name if cfg.logger.record_video else ""

    key, init_env_steps, stats = None, None, None
    if not cfg.env.vecnorm and cfg.env.norm_stats:
        key = (
            ("next", "pixels")
            if cfg.env.from_pixels
            else ("next", "observation_vector")
        )
        init_env_steps = cfg.env.init_env_steps
        stats = {"loc": None, "scale": None}
    elif cfg.env.from_pixels:
        stats = {"loc": 0.5, "scale": 0.5}

    proof_env = transformed_env_constructor(
        cfg=cfg,
        use_env_creator=False,
        stats=stats,
    )()
    initialize_observation_norm_transforms(
        proof_environment=proof_env, num_iter=init_env_steps, key=key
    )
    _, obs_norm_state_dict = retrieve_observation_norms_state_dict(proof_env)[0]

    model = make_redq_model(
        proof_env,
        cfg=cfg,
        device=device,
    )
    loss_module, target_net_updater = make_redq_loss(model, cfg)

    actor_model_explore = model[0]
    if cfg.exploration.ou_exploration:
        if cfg.exploration.gSDE:
            raise RuntimeError("gSDE and ou_exploration are incompatible")
        actor_model_explore = OrnsteinUhlenbeckProcessWrapper(
            actor_model_explore,
            annealing_num_steps=cfg.exploration.annealing_frames,
            sigma=cfg.exploration.ou_sigma,
            theta=cfg.exploration.ou_theta,
        ).to(device)
    if device == torch.device("cpu"):
        # mostly for debugging
        actor_model_explore.share_memory()

    if cfg.exploration.gSDE:
        with torch.no_grad(), set_exploration_type(ExplorationType.RANDOM):
            # get dimensions to build the parallel env
            proof_td = actor_model_explore(proof_env.reset().to(device))
        action_dim_gsde, state_dim_gsde = proof_td.get("_eps_gSDE").shape[-2:]
        del proof_td
    else:
        action_dim_gsde, state_dim_gsde = None, None

    proof_env.close()
    create_env_fn = parallel_env_constructor(
        cfg=cfg,
        obs_norm_state_dict=obs_norm_state_dict,
        action_dim_gsde=action_dim_gsde,
        state_dim_gsde=state_dim_gsde,
    )

    collector = make_collector_offpolicy(
        make_env=create_env_fn,
        actor_model_explore=actor_model_explore,
        cfg=cfg,
    )

    replay_buffer = make_replay_buffer("cpu", cfg)

    recorder = transformed_env_constructor(
        cfg,
        video_tag=video_tag,
        norm_obs_only=True,
        obs_norm_state_dict=obs_norm_state_dict,
        logger=logger,
        use_env_creator=False,
    )()
    if isinstance(create_env_fn, ParallelEnv):
        raise NotImplementedError("This behaviour is deprecated")
    elif isinstance(create_env_fn, EnvCreator):
        recorder.transform[1:].load_state_dict(
            get_norm_state_dict(create_env_fn()), strict=False
        )
    elif isinstance(create_env_fn, TransformedEnv):
        recorder.transform = create_env_fn.transform.clone()
    else:
        raise NotImplementedError(f"Unsupported env type {type(create_env_fn)}")
    if logger is not None and video_tag:
        recorder.insert_transform(0, VideoRecorder(logger=logger, tag=video_tag))

    # reset reward scaling
    for t in recorder.transform:
        if isinstance(t, RewardScaling):
            t.scale.fill_(1.0)
            t.loc.fill_(0.0)

    trainer = make_trainer(
        collector,
        loss_module,
        recorder,
        target_net_updater,
        actor_model_explore,
        replay_buffer,
        logger,
        cfg,
    )

    trainer.train()
    if logger is not None:
        return (logger.log_dir, trainer._log_dict)


if __name__ == "__main__":
    main()
