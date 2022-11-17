# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import os
import pathlib
import uuid
from datetime import datetime

import hydra
import torch.cuda
from hydra.core.config_store import ConfigStore
from torchrl.envs import ParallelEnv, EnvCreator
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.objectives.value import TDEstimate
from torchrl.record import VideoRecorder
from torchrl.trainers.helpers.collectors import (
    make_collector_onpolicy,
    OnPolicyCollectorConfig,
)
from torchrl.trainers.helpers.envs import (
    correct_for_frame_skip,
    get_stats_random_rollout,
    parallel_env_constructor,
    transformed_env_constructor,
    EnvConfig,
)
from torchrl.trainers.helpers.logger import LoggerConfig
from torchrl.trainers.helpers.losses import make_a2c_loss, A2CLossConfig
from torchrl.trainers.helpers.models import (
    make_a2c_model,
    A2CModelConfig,
)
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

    exp_name = "_".join(
        [
            "A2C",
            cfg.exp_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        ]
    )
    if cfg.logger == "tensorboard":
        from torchrl.trainers.loggers.tensorboard import TensorboardLogger

        logger = TensorboardLogger(log_dir="a2c_logging", exp_name=exp_name)
    elif cfg.logger == "csv":
        from torchrl.trainers.loggers.csv import CSVLogger

        logger = CSVLogger(log_dir="a2c_logging", exp_name=exp_name)
    elif cfg.logger == "wandb":
        from torchrl.trainers.loggers.wandb import WandbLogger

        logger = WandbLogger(log_dir="a2c_logging", exp_name=exp_name)
    elif cfg.logger == "mlflow":
        from torchrl.trainers.loggers.mlflow import MLFlowLogger

        logger = MLFlowLogger(
            tracking_uri=pathlib.Path(os.path.abspath("a2c_logging")).as_uri(),
            exp_name=exp_name,
        )
    video_tag = exp_name if cfg.record_video else ""

    stats = None
    if not cfg.vecnorm and cfg.norm_stats:
        proof_env = transformed_env_constructor(cfg=cfg, use_env_creator=False)()
        stats = get_stats_random_rollout(
            cfg,
            proof_env,
            key="pixels" if cfg.from_pixels else "observation_vector",
        )
        # make sure proof_env is closed
        proof_env.close()
    elif cfg.from_pixels:
        stats = {"loc": 0.5, "scale": 0.5}
    proof_env = transformed_env_constructor(
        cfg=cfg, use_env_creator=False, stats=stats
    )()

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
        stats=stats,
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
        stats=stats,
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

    if not cfg.advantage_in_loss:
        critic_model = model.get_value_operator()
        advantage = TDEstimate(
            cfg.gamma,
            value_network=critic_model,
            average_rewards=True,
            gradient_mode=False,
        )
        advantage = advantage.to(device)
        trainer.register_op(
            "process_optim_batch",
            advantage,
        )

    final_seed = collector.set_seed(cfg.seed)
    print(f"init seed: {cfg.seed}, final seed: {final_seed}")

    trainer.train()
    return (logger.log_dir, trainer._log_dict)


if __name__ == "__main__":
    main()
