# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import os
import uuid
from copy import deepcopy
from datetime import datetime

from torchrl.envs import ParallelEnv, EnvCreator, Compose, ObservationNorm, NoopResetEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.record import VideoRecorder
from torchrl.trainers.helpers.envs import LIBS
from utils import MJEnv

LIBS["mjenv"] = MJEnv

try:
    import configargparse as argparse

    _configargparse = True
except ImportError:
    import argparse

    _configargparse = False

import warnings

import hydra
import torch.cuda
from hydra.core.config_store import ConfigStore
from torchrl.envs import ParallelEnv, EnvCreator
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import OrnsteinUhlenbeckProcessWrapper
from torchrl.record import VideoRecorder
from torchrl.trainers.helpers.collectors import (
    make_collector_offpolicy,
    OffPolicyCollectorConfig,
)
from torchrl.trainers.helpers.envs import (
    correct_for_frame_skip,
    get_stats_random_rollout,
    parallel_env_constructor,
    transformed_env_constructor,
    EnvConfig,
)
from torchrl.trainers.helpers.losses import make_redq_loss, LossConfig
from torchrl.trainers.helpers.models import (
    make_redq_model,
    REDQModelConfig,
)
from torchrl.trainers.helpers.recorder import RecorderConfig
from torchrl.trainers.helpers.replay_buffer import (
    make_replay_buffer,
    ReplayArgsConfig,
)
from dataclasses import dataclass

from utils_env import transformed_env_constructor, parallel_env_constructor
from utils_redq import (
    make_redq_model_pixels,
    make_redq_model_pixels_shared,
    make_redq_model_state,
    make_redq_model_state_pixels,
    make_redq_model_state_pixels_shared,
)

warnings.filterwarnings(
    "ignore", message="Using the default interaction site of end-effector."
)
warnings.filterwarnings("ignore", message="Unused kwargs found.")
warnings.filterwarnings(
    "ignore", message="In future, it will be an error for 'np.bool_'"
)
warnings.filterwarnings("ignore", message="is deprecated and will be removed in Pillow")

from torchrl.trainers.helpers.trainers import make_trainer, TrainerConfig


@dataclass
class REDQConfig:
    shared_mapping: bool = False
    include_state: bool = False
    use_avg_pooling: bool = False


config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        TrainerConfig,
        OffPolicyCollectorConfig,
        EnvConfig,
        LossConfig,
        REDQModelConfig,
        RecorderConfig,
        ReplayArgsConfig,
        REDQConfig,
    )
    for config_field in dataclasses.fields(config_cls)
]

Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)

DEFAULT_REWARD_SCALING = {
    "Hopper-v1": 5,
    "Walker2d-v1": 5,
    "HalfCheetah-v1": 5,
    "cheetah": 5,
    "Ant-v2": 5,
    "Humanoid-v2": 20,
    "humanoid": 100,
}


@hydra.main(version_base=None, config_path=None, config_name="config")
def main(cfg: "DictConfig"):
    from torch.utils.tensorboard import SummaryWriter  # avoid loading on each process

    cfg_copy = deepcopy(cfg)
    cfg = correct_for_frame_skip(cfg)

    if not isinstance(cfg.reward_scaling, float):
        cfg.reward_scaling = DEFAULT_REWARD_SCALING.get(cfg.env_name, 5.0)

    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    exp_name = "-".join(
        [
            "REDQ",
            os.environ.get("SLURM_JOB_ID", ""),
            cfg.exp_name,
            # str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        ]
    )

    print("Gathering stats")
    stats_pixels = None
    stats_state = None
    if not cfg.vecnorm and cfg.norm_stats:
        print("Creating proof env without stats: ", end="\t")
        proof_env = transformed_env_constructor(
            cfg=cfg, use_env_creator=False, device="cuda:1"
        )()
        print(proof_env)
        if cfg.from_pixels:
            print("Pixel stats")
            stats_pixels = {"loc": 0.41300642490386963, "scale": 0.2709078788757324}
            # stats_pixels = get_stats_random_rollout(
            #     cfg,
            #     proof_env,
            #     key="next_pixels",
            # )
        if not cfg.from_pixels or cfg.include_state:
            print("State stats")
            stats_state = get_stats_random_rollout(
                cfg,
                proof_env,
                key="next_observation_vector",
            )
        # make sure proof_env is closed
        proof_env.close()
    elif cfg.from_pixels:
        stats_pixels = {"loc": 0.5, "scale": 0.5}
    print("Creating proof env with stats: ", end="\t")
    proof_env = transformed_env_constructor(
        cfg=cfg,
        use_env_creator=False,
        stats_pixels=stats_pixels,
        stats_state=stats_state,
        device="cuda:1",
    )()
    print(proof_env)

    print("Creating mode: ", end="\t")
    if cfg.from_pixels:
        if cfg.shared_mapping:
            if cfg.include_state:
                model = make_redq_model_state_pixels_shared(
                    proof_env,
                    cfg=cfg,
                    device=device,
                )
                actor_model_explore = model.get_policy_operator()
            else:
                model = make_redq_model_pixels_shared(
                    proof_env,
                    cfg=cfg,
                    device=device,
                )
                actor_model_explore = model.get_policy_operator()
        else:
            if cfg.include_state:
                model = make_redq_model_state_pixels(
                    proof_env,
                    cfg=cfg,
                    device=device,
                )
                actor_model_explore = model[0]
            else:
                model = make_redq_model_pixels(
                    proof_env,
                    cfg=cfg,
                    device=device,
                )
                actor_model_explore = model[0]
    else:
        model = make_redq_model_state(
            proof_env,
            cfg=cfg,
            device=device,
        )
        actor_model_explore = model[0]
    print(model)

    print("Creating loss: ", end="\t")
    loss_module, target_net_updater = make_redq_loss(model, cfg)
    del model

    print(loss_module, target_net_updater)
    if cfg.ou_exploration:
        if cfg.gSDE:
            raise RuntimeError("gSDE and ou_exploration are incompatible")
        actor_model_explore = OrnsteinUhlenbeckProcessWrapper(
            actor_model_explore, annealing_num_steps=cfg.annealing_frames
        ).to(device)
    if device == torch.device("cpu"):
        # mostly for debugging
        actor_model_explore.share_memory()

    if cfg.gSDE:
        with torch.no_grad(), set_exploration_mode("random"):
            # get dimensions to build the parallel env
            proof_td = actor_model_explore(proof_env.reset().to(device))
        action_dim_gsde, state_dim_gsde = proof_td.get("_eps_gSDE").shape[-2:]
        del proof_td
    else:
        action_dim_gsde, state_dim_gsde = None, None

    print("closing proof env")
    proof_env.close()
    del proof_env

    print("Creating parallel env")
    create_env_fn = parallel_env_constructor(
        cfg=cfg,
        stats_pixels=stats_pixels,
        stats_state=stats_state,
        action_dim_gsde=action_dim_gsde,
        state_dim_gsde=state_dim_gsde,
    )

    print("Creating collector")
    collector = make_collector_offpolicy(
        make_env=create_env_fn,
        actor_model_explore=actor_model_explore,
        cfg=cfg,
    )

    print("Creating replay buffer")
    replay_buffer = make_replay_buffer(device, cfg)

    print("Creating writer")
    writer = SummaryWriter(f"redq_logging/{exp_name}")
    video_tag = exp_name if cfg.record_video else ""

    print("Creating recorder")
    # recorder = None
    recorder = transformed_env_constructor(
        cfg,
        video_tag=video_tag,
        norm_obs_only=True,
        stats_state=stats_state,
        stats_pixels=stats_pixels,
        writer=writer,
        use_env_creator=False,
        device="cuda:1",
    )()

    # remove video recorder from recorder to have matching state_dict keys
    if cfg.record_video:
        recorder_rm = TransformedEnv(recorder.base_env)
        for transform in recorder.transform:
            if not isinstance(transform, VideoRecorder):
                recorder_rm.append_transform(transform)
    else:
        recorder_rm = recorder
    if isinstance(create_env_fn, ParallelEnv):
        recorder_rm.load_state_dict(create_env_fn.state_dict()["worker0"])
        create_env_fn.close()
    elif isinstance(create_env_fn, EnvCreator):
        recorder_rm.load_state_dict(create_env_fn().state_dict())
    else:
        recorder_rm.load_state_dict(create_env_fn.state_dict())

    # reset reward scaling
    for t in recorder.transform:
        if isinstance(t, RewardScaling):
            t.scale.fill_(1.0)
            t.loc.fill_(0.0)
    recorder.transform = Compose(
        *[t for t in recorder.transform if not isinstance(t, NoopResetEnv)]
    )

    # recorder = ParallelEnv(1, recorder)
    # recorder = TransformedEnv(
    #     recorder,
    #     Compose(
    #         ObservationNorm(loc=stats_pixels['loc'], scale=stats_pixels['scale'], standard_normal=False),
    #         VideoRecorder(tag=video_tag, writer=writer)
    #     ))

    print("Creating trainer")
    trainer = make_trainer(
        collector,
        loss_module,
        recorder,
        target_net_updater,
        actor_model_explore,
        replay_buffer,
        writer,
        cfg,
    )
    trainer.save_trainer_file = "/".join([writer.log_dir, "config.t"])
    torch.save(cfg_copy, "/".join([writer.log_dir, "saved.t"]))

    # def select_keys(batch):
    #     return batch.select(
    #         "reward",
    #         "done",
    #         "steps_to_next_obs",
    #         "pixels",
    #         "next_pixels",
    #         "observation_vector",
    #         "next_observation_vector",
    #         "action",
    #         "solved",
    #     )
    #
    # trainer.register_op("batch_process", select_keys)
    # trainer.register_op(
    #     "pre_steps_log",
    #     lambda batch: {"solved": batch["solved"].sum() / batch["solved"].numel()},
    # )

    final_seed = collector.set_seed(cfg.seed)
    print(f"init seed: {cfg.seed}, final seed: {final_seed}")

    trainer.train()
    return (writer.log_dir, trainer._log_dict, trainer.state_dict())


if __name__ == "__main__":
    main()
