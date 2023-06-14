# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import hydra
import torch.cuda
from hydra.core.config_store import ConfigStore
from torchrl.envs import EnvCreator, ParallelEnv
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.modules import EGreedyWrapper
from torchrl.record import VideoRecorder
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.collectors import (
    make_collector_offpolicy,
    OffPolicyCollectorConfig,
)
from torchrl.trainers.helpers.envs import (
    correct_for_frame_skip,
    EnvConfig,
    get_norm_state_dict,
    initialize_observation_norm_transforms,
    parallel_env_constructor,
    retrieve_observation_norms_state_dict,
    transformed_env_constructor,
)
from torchrl.trainers.helpers.logger import LoggerConfig
from torchrl.trainers.helpers.losses import LossConfig, make_dqn_loss
from torchrl.trainers.helpers.models import DiscreteModelConfig, make_dqn_actor
from torchrl.trainers.helpers.replay_buffer import make_replay_buffer, ReplayArgsConfig
from torchrl.trainers.helpers.trainers import make_trainer, TrainerConfig

config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        TrainerConfig,
        OffPolicyCollectorConfig,
        EnvConfig,
        LossConfig,
        DiscreteModelConfig,
        LoggerConfig,
        ReplayArgsConfig,
    )
    for config_field in dataclasses.fields(config_cls)
]
Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base="1.1", config_path=".", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821

    cfg = correct_for_frame_skip(cfg)

    if not isinstance(cfg.reward_scaling, float):
        cfg.reward_scaling = 1.0

    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    exp_name = generate_exp_name("DQN", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger, logger_name="dqn_logging", experiment_name=exp_name
    )
    video_tag = exp_name if cfg.record_video else ""

    key, init_env_steps, stats = None, None, None
    if not cfg.vecnorm and cfg.norm_stats:
        if not hasattr(cfg, "init_env_steps"):
            raise AttributeError("init_env_steps missing from arguments.")
        key = ("next", "pixels") if cfg.from_pixels else ("next", "observation_vector")
        init_env_steps = cfg.init_env_steps
        stats = {"loc": None, "scale": None}
    elif cfg.from_pixels:
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

    model = make_dqn_actor(
        proof_environment=proof_env,
        cfg=cfg,
        device=device,
    )

    loss_module, target_net_updater = make_dqn_loss(model, cfg)
    model_explore = EGreedyWrapper(model, annealing_num_steps=cfg.annealing_frames).to(
        device
    )

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
        actor_model_explore=model_explore,
        cfg=cfg,
    )

    replay_buffer = make_replay_buffer(device, cfg)

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
        _env = create_env_fn()
        _env.rollout(2)
        recorder.transform[1:].load_state_dict(get_norm_state_dict(_env), strict=False)
        del _env
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
        model,
        replay_buffer,
        logger,
        cfg,
    )

    final_seed = collector.set_seed(cfg.seed)
    print(f"init seed: {cfg.seed}, final seed: {final_seed}")

    trainer.train()
    return (logger.log_dir, trainer._log_dict)


if __name__ == "__main__":
    main()
