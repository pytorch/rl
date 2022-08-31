import dataclasses
import uuid
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import hydra
import torch
import torch.cuda
import tqdm
from hydra.core.config_store import ConfigStore

# float16
from torch.cuda.amp import autocast, GradScaler
from torch.nn.utils import clip_grad_norm_
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
from torchrl.trainers.helpers.logger import LoggerConfig
from torchrl.trainers.helpers.losses import (
    make_sac_loss,
    LossConfig,
    make_mbpo_model_loss,
)
from torchrl.trainers.helpers.models import (
    SACModelConfig,
    make_sac_model,
)
from torchrl.trainers.helpers.models import (
    make_mbpo_model,
    MBPOConfig,
)
from torchrl.trainers.helpers.replay_buffer import (
    make_replay_buffer,
    make_model_replay_buffer,
    ReplayArgsConfig,
)
from torchrl.trainers.trainers import Recorder


@dataclass
class TrainingConfig:
    optim_steps_per_batch: int = 1000
    # Number of optimization steps in between two collection of data. See frames_per_batch below.
    # LR scheduler.
    grad_clip: float = 1000
    # batch size of the TensorDict retrieved from the replay buffer. Default=256.
    normalize_rewards_online: bool = False
    # Computes the running statistics of the rewards and normalizes them before they are passed to the loss module.
    normalize_rewards_online_scale: float = 1.0
    # Final scale of the normalized rewards.
    normalize_rewards_online_decay: float = 0.9999
    # Decay of the reward moving averaging


config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        OffPolicyCollectorConfig,
        EnvConfig,
        SACModelConfig,
        LoggerConfig,
        ReplayArgsConfig,
        LossConfig,
        MBPOConfig,
        TrainingConfig,
    )
    for config_field in dataclasses.fields(config_cls)
]
Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: "DictConfig"):

    cfg = correct_for_frame_skip(cfg)

    if not isinstance(cfg.reward_scaling, float):
        cfg.reward_scaling = 1.0

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    # elif torch.backends.mps.is_available():
    #     device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")
    exp_name = "_".join(
        [
            "MBPO",
            cfg.exp_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        ]
    )

    if cfg.logger == "tensorboard":
        from torchrl.trainers.loggers.tensorboard import TensorboardLogger

        logger = TensorboardLogger(log_dir="mbpo_logging", exp_name=exp_name)
    elif cfg.logger == "csv":
        from torchrl.trainers.loggers.csv import CSVLogger

        logger = CSVLogger(log_dir="mbpo_logging", exp_name=exp_name)
    elif cfg.logger == "wandb":
        from torchrl.trainers.loggers.wandb import WandbLogger

        logger = WandbLogger(
            f"mbpo/{exp_name}",
            project="torchrl",
            group=f"MBPO_{cfg.env_name}_with_elites",
        )
    video_tag = "MBPO_policy_test" if cfg.record_video else ""

    stats = None
    if not cfg.vecnorm and cfg.norm_stats:
        proof_env = transformed_env_constructor(cfg=cfg, use_env_creator=False)()
        stats = get_stats_random_rollout(
            cfg,
            proof_env,
            key="next_pixels" if cfg.from_pixels else "next_observation_vector",
        )
        # make sure proof_env is closed
        proof_env.close()
    elif cfg.from_pixels:
        stats = {"loc": 0.5, "scale": 0.5}
    proof_env = transformed_env_constructor(
        cfg=cfg, use_env_creator=False, stats=stats
    )()

    # MBPO models

    single_world_model = make_mbpo_model(
        proof_env,
        cfg,
        device=device,
        observation_key="observation_vector",
        action_key="action",
    )
    sac_model = make_sac_model(
        proof_env,
        cfg=cfg,
        device=device,
    )

    # Losses
    world_model_loss, model_based_env = make_mbpo_model_loss(
        single_world_model,
        proof_env,
        cfg,
        observation_key="observation_vector",
        device=device,
    )
    sac_loss, target_net_updater = make_sac_loss(sac_model, cfg)
    # optimizers
    world_model_opt = torch.optim.Adam(
        world_model_loss.parameters(), lr=cfg.world_model_lr
    )
    sac_opt = torch.optim.Adam(sac_loss.parameters(), lr=cfg.sac_lr)

    # Recorder
    if cfg.record_video:
        recorder = VideoRecorder(
            cfg=cfg,
            env=proof_env,
            device=device,
            logger=logger,
            tag=video_tag,
        )
    else:
        recorder = None

    # Actor and value network
    policy = sac_model[0]
    model_explore = sac_model[0]

    if cfg.ou_exploration:
        model_explore = OrnsteinUhlenbeckProcessWrapper(
            model_explore,
            annealing_num_steps=cfg.annealing_frames,
            sigma=cfg.ou_sigma,
            theta=cfg.ou_theta,
        ).to(device)

    action_dim_gsde, state_dim_gsde = None, None
    proof_env.close()
    create_env_fn = parallel_env_constructor(
        cfg=cfg,
        stats=stats,
        action_dim_gsde=action_dim_gsde,
        state_dim_gsde=state_dim_gsde,
    )

    collector = make_collector_offpolicy(
        make_env=create_env_fn,
        actor_model_explore=model_explore,
        cfg=cfg,
        # make_env_kwargs=[
        #     {"device": device}
        #     for device in cfg.collector_devices
        # ],
    )

    real_replay_buffer = make_replay_buffer(device, cfg)
    fake_replay_buffer = make_model_replay_buffer(device, cfg)

    recorder = transformed_env_constructor(
        cfg,
        video_tag=video_tag,
        norm_obs_only=True,
        stats=stats,
        logger=logger,
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

    record = Recorder(
        record_frames=cfg.record_frames,
        frame_skip=cfg.frame_skip,
        policy_exploration=policy,
        recorder=recorder,
        record_interval=cfg.record_interval,
        log_keys=cfg.recorder_log_keys,
    )

    final_seed = collector.set_seed(cfg.seed)
    print(f"init seed: {cfg.seed}, final seed: {final_seed}")
    # Training loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)
    r0 = None
    path = Path("./log")
    path.mkdir(exist_ok=True)

    scaler1 = GradScaler()
    scaler2 = GradScaler()
    torch.cuda.empty_cache()
    for i, tensordict in enumerate(collector):

        # update weights of the inference policy
        collector.update_policy_weights_()

        if r0 is None:
            r0 = tensordict["reward"].mean().item()
        pbar.update(tensordict.numel())
        current_frames = tensordict.numel()
        collected_frames += current_frames
        tensordict = tensordict.view(-1)
        original_keys = tensordict.keys()
        real_replay_buffer.extend(tensordict.cpu())

        if collected_frames >= cfg.init_random_frames:
            # Train model on current replay buffer
            for j in range(cfg.optim_steps_per_batch):
                # Train Model
                # Sample data from model and buffer it
                if j % cfg.train_model_every_k_optim_step == 0:
                    for _ in range(len(real_replay_buffer) // cfg.model_batch_size):
                        model_sampled_tensordict = real_replay_buffer.sample(
                            cfg.model_batch_size
                        ).to(device)

                        with autocast(dtype=torch.float16):
                            model_loss_td = world_model_loss(model_sampled_tensordict)
                        scaler1.scale(model_loss_td["loss_world_model"]).backward()
                        scaler1.unscale_(world_model_opt)
                        clip_grad_norm_(world_model_loss.parameters(), cfg.grad_clip)
                        scaler1.step(world_model_opt)
                        world_model_opt.zero_grad()
                        scaler1.update()
                    with torch.no_grad(), set_exploration_mode("random"):
                        for _ in range(cfg.train_model_every_k_optim_step):
                            model_sampled_tensordict = real_replay_buffer.sample(
                                cfg.num_model_rollouts
                            ).to(device)
                            fake_traj_tensordict = model_based_env.rollout(
                                max_steps=cfg.imagination_horizon,
                                policy=policy,
                                auto_reset=False,
                                tensordict=model_sampled_tensordict,
                            )
                            fake_traj_tensordict = fake_traj_tensordict.select(*original_keys)
                            fake_replay_buffer.extend(fake_traj_tensordict.view(-1).cpu())
                if len(fake_replay_buffer) > cfg.init_random_frames:
                    for _ in range(cfg.num_sac_training_steps_per_optim_step):

                        num_real_samples = int(cfg.sac_batch_size * cfg.real_data_ratio)

                        num_fake_samples = cfg.sac_batch_size - num_real_samples

                        # agent_sampled_tensordict = fake_replay_buffer.sample(cfg.sac_batch_size)

                        fake_sampled_tensordict = fake_replay_buffer.sample(
                            num_fake_samples
                        )

                        real_sampled_tensordict = real_replay_buffer.sample(
                            num_real_samples
                        )

                        agent_sampled_tensordict = torch.cat(
                            [
                                fake_sampled_tensordict,
                                real_sampled_tensordict,
                            ],
                            dim=0,
                        ).to(device)

                        # Train agent
                        with autocast(dtype=torch.float16):
                            sac_loss_td = sac_loss(agent_sampled_tensordict)
                            sac_loss_sum = (
                                sac_loss_td["loss_actor"]
                                + sac_loss_td["loss_qvalue"]
                                + sac_loss_td["loss_value"]
                                + sac_loss_td["loss_alpha"]
                            )
                        scaler2.scale(sac_loss_sum).backward()
                        scaler2.unscale_(sac_opt)
                        clip_grad_norm_(sac_loss.parameters(), cfg.grad_clip)
                        scaler2.step(sac_opt)
                        sac_opt.zero_grad()
                        scaler2.update()

                        target_net_updater.step()

                with torch.no_grad(), set_exploration_mode("mode"):
                    td_record = record(None)
                    if td_record is not None and logger is not None:
                        for key, value in td_record.items():
                            if key in ["r_evaluation", "total_r_evaluation"]:
                                logger.log_scalar(
                                    key,
                                    value.detach().cpu().numpy(),
                                    step=collected_frames,
                                )


if __name__ == "__main__":
    main()
