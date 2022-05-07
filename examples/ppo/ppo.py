# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import uuid
from datetime import datetime

import hydra
import torch.cuda
from hydra.core.config_store import ConfigStore
from omegaconf import DictConfig
from torch.utils.tensorboard import SummaryWriter
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.trainers.helpers.collectors import (OnPolicyCollectorConfig,
                                                 make_collector_onpolicy)
from torchrl.trainers.helpers.envs import (EnvConfig, correct_for_frame_skip,
                                           get_stats_random_rollout,
                                           parallel_env_constructor,
                                           transformed_env_constructor)
from torchrl.trainers.helpers.losses import PPOLossConfig, make_ppo_loss
from torchrl.trainers.helpers.models import PPOModelConfig, make_ppo_model
from torchrl.trainers.helpers.recorder import RecorderConfig
from torchrl.trainers.helpers.trainers import TrainerConfig, make_trainer

config_fields = [(config_field.name, config_field.type, config_field) for config_cls in 
    (TrainerConfig, OnPolicyCollectorConfig, EnvConfig, RecorderConfig, PPOLossConfig, PPOModelConfig) 
    for config_field in dataclasses.fields(config_cls) 
]

Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

@hydra.main(config_path=None, config_name="config")
def main(cfg: DictConfig):

    # args = parser.parse_args()

    args = correct_for_frame_skip(cfg)

    if not isinstance(args.reward_scaling, float):
        args.reward_scaling = 1.0

    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    exp_name = "_".join(
        [
            "PPO",
            args.exp_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        ]
    )
    writer = SummaryWriter(f"ppo_logging/{exp_name}")
    video_tag = exp_name if args.record_video else ""

    proof_env = transformed_env_constructor(args=args, use_env_creator=False)()
    model = make_ppo_model(proof_env, args=args, device=device)
    actor_model = model.get_policy_operator()

    loss_module = make_ppo_loss(model, args)

    stats = None
    if not args.vecnorm:
        stats = get_stats_random_rollout(args, proof_env)
    # make sure proof_env is closed
    proof_env.close()

    create_env_fn = parallel_env_constructor(args=args, stats=stats)

    collector = make_collector_onpolicy(
        make_env=create_env_fn,
        actor_model_explore=actor_model,
        args=args,
    )

    recorder = transformed_env_constructor(
        args,
        video_tag=video_tag,
        norm_obs_only=True,
        stats=stats,
        writer=writer,
    )()

    # remove video recorder from recorder to have matching state_dict keys
    if args.record_video:
        recorder_rm = TransformedEnv(recorder.env, recorder.transform[1:])
    else:
        recorder_rm = recorder

    recorder_rm.load_state_dict(create_env_fn.state_dict()["worker0"])
    create_env_fn.close()

    # reset reward scaling
    for t in recorder.transform:
        if isinstance(t, RewardScaling):
            t.scale.fill_(1.0)

    trainer = make_trainer(
        collector, loss_module, recorder, None, actor_model, None, writer, args
    )
    if args.loss == "kl":
        trainer.register_op("pre_optim_steps", loss_module.reset)

    trainer.train()


if __name__ == "__main__":
    # make_config()
    main()
