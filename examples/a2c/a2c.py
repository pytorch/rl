# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses

import hydra
import torch.cuda
from hydra.core.config_store import ConfigStore
from torchrl.envs.transforms import RewardScaling
from torchrl.envs.utils import set_exploration_mode
from torchrl.objectives.value import TD0Estimator
from torchrl.record.loggers import generate_exp_name, get_logger
from torchrl.trainers.helpers.collectors import (
    make_collector_onpolicy,
    OnPolicyCollectorConfig,
)
from torchrl.trainers.helpers.envs import (
    correct_for_frame_skip,
    EnvConfig,
    initialize_observation_norm_transforms,
    parallel_env_constructor,
    retrieve_observation_norms_state_dict,
    transformed_env_constructor,
)
from torchrl.trainers.helpers.logger import LoggerConfig
from torchrl.trainers.helpers.losses import A2CLossConfig, make_a2c_loss
from torchrl.trainers.helpers.models import A2CModelConfig, make_a2c_model
from torchrl.trainers.helpers.trainers import make_trainer, TrainerConfig

config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        TrainerConfig,
        OnPolicyCollectorConfig,
        EnvConfig,
        A2CLossConfig,
        A2CModelConfig,
        LoggerConfig,
    )
    for config_field in dataclasses.fields(config_cls)
]

Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_path="", config_name="config")
def main(cfg: "DictConfig"):  # noqa: F821

    cfg = correct_for_frame_skip(cfg)

    if not isinstance(cfg.reward_scaling, float):
        cfg.reward_scaling = 1.0

    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    exp_name = generate_exp_name("A2C", cfg.exp_name)
    logger = get_logger(
        logger_type=cfg.logger, logger_name="a2c_logging", experiment_name=exp_name
    )
    video_tag = exp_name if cfg.record_video else ""

    key, init_env_steps, stats = None, None, None
    if not cfg.vecnorm and cfg.norm_stats:
        if not hasattr(cfg, "init_env_steps"):
            raise AttributeError("init_env_steps missing from arguments.")
        key = "pixels" if cfg.from_pixels else "observation_vector"
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

    model = make_a2c_model(
        proof_env,
        cfg=cfg,
        device=device,
    )
    actor_model = model.get_policy_operator()

    loss_module = make_a2c_loss(model, cfg)
    if cfg.gSDE:
        with torch.no_grad(), set_exploration_mode("random"):
            # get dimensions to build the parallel env
            proof_td = model(proof_env.reset().to(device))
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

    collector = make_collector_onpolicy(
        make_env=create_env_fn,
        actor_model_explore=actor_model,
        cfg=cfg,
    )

    recorder = transformed_env_constructor(
        cfg,
        video_tag=video_tag,
        norm_obs_only=True,
        obs_norm_state_dict=obs_norm_state_dict,
        logger=logger,
        use_env_creator=False,
    )()

    # reset reward scaling
    for t in recorder.transform:
        if isinstance(t, RewardScaling):
            t.scale.fill_(1.0)
            t.loc.fill_(0.0)

    trainer = make_trainer(
        collector=collector,
        loss_module=loss_module,
        recorder=recorder,
        target_net_updater=None,
        policy_exploration=actor_model,
        replay_buffer=None,
        logger=logger,
        cfg=cfg,
    )

    critic_model = model.get_value_operator()
    advantage = TD0Estimator(
        gamma=cfg.gamma,
        value_network=critic_model,
        average_rewards=True,
        differentiable=True,
    )
    trainer.register_op(
        "process_optim_batch",
        torch.no_grad()(advantage),
    )

    final_seed = collector.set_seed(cfg.seed)
    print(f"init seed: {cfg.seed}, final seed: {final_seed}")

    trainer.train()
    return (logger.log_dir, trainer._log_dict)


if __name__ == "__main__":
    main()
