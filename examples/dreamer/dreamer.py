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
from torchrl.modules.tensordict_module.exploration import AdditiveGaussianWrapper
from torchrl.objectives.costs.dreamer import (
    DreamerActorLoss,
    DreamerModelLoss,
    DreamerValueLoss,
)
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
from torchrl.trainers.helpers.models import (
    make_dreamer,
    DreamerConfig,
)
from torchrl.trainers.helpers.replay_buffer import (
    make_replay_buffer,
    ReplayArgsConfig,
)
from torchrl.trainers.trainers import Recorder, RewardNormalizer


@dataclass
class TrainingConfig:
    optim_steps_per_batch: int = 500
    # Number of optimization steps in between two collection of data. See frames_per_batch below.
    # LR scheduler.
    batch_size: int = 256

    batch_length: int = 50
    # batch size of the TensorDict retrieved from the replay buffer. Default=256.
    log_interval: int = 10000
    # logging interval, in terms of optimization steps. Default=10000.


config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        OffPolicyCollectorConfig,
        EnvConfig,
        LoggerConfig,
        ReplayArgsConfig,
        DreamerConfig,
        TrainingConfig,
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


def grad_norm(optimizer: torch.optim.Optimizer):
    sum_of_sq = 0.0
    for pg in optimizer.param_groups:
        for p in pg["params"]:
            sum_of_sq += p.grad.pow(2).sum()
    return sum_of_sq.sqrt().item()


def make_recorder_env(cfg, video_tag, stats, logger, create_env_fn):
    recorder = transformed_env_constructor(
        cfg,
        video_tag=video_tag,
        norm_obs_only=True,
        stats=stats,
        logger=logger,
        use_env_creator=False,
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
    return recorder


@torch.inference_mode()
def call_record(
    logger,
    record,
    collected_frames,
    sampled_tensordict,
    stats,
    model_based_env,
    actor_model,
    cfg,
):
    td_record = record(None)
    if td_record is not None and logger is not None:
        for key, value in td_record.items():
            if key in ["r_evaluation", "total_r_evaluation"]:
                logger.log_scalar(
                    key,
                    value.cpu().item(),
                    step=collected_frames,
                )
    # Compute observation reco
    if cfg.record_video and record._count % cfg.record_interval == 0:
        world_model_td = sampled_tensordict

        true_pixels = recover_pixels(world_model_td["pixels"], stats)

        reco_pixels = recover_pixels(world_model_td["reco_pixels"], stats)
        with autocast(dtype=torch.float16):
            world_model_td = world_model_td.select("posterior_states", "next_belief")
            world_model_td.batch_size = [
                world_model_td.shape[0],
                world_model_td.get("next_belief").shape[1],
            ]
            world_model_td.rename_key("posterior_states", "prior_state")
            world_model_td.rename_key("next_belief", "belief")
            world_model_td = model_based_env.rollout(
                max_steps=true_pixels.shape[1],
                policy=actor_model,
                auto_reset=False,
                tensordict=world_model_td[:, 0],
            )
        imagine_pxls = recover_pixels(
            model_based_env.decode_obs(world_model_td)["reco_pixels"],
            stats,
        )

        stacked_pixels = torch.cat([true_pixels, reco_pixels, imagine_pxls], dim=-1)
        if logger is not None:
            logger.log_video(
                "pixels_rec_and_imag",
                stacked_pixels.detach().cpu().numpy(),
            )


@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: "DictConfig"):

    cfg = correct_for_frame_skip(cfg)

    if not isinstance(cfg.reward_scaling, float):
        cfg.reward_scaling = 1.0

    if torch.cuda.is_available() and not cfg.model_device != "":
        device = torch.device("cuda:0")
    elif cfg.model_device:
        device = torch.device(cfg.model_device)
    else:
        device = torch.device("cpu")
    print(f"Using device {device}")
    exp_name = "_".join(
        [
            "Dreamer",
            cfg.exp_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        ]
    )

    if cfg.logger == "wandb":
        from torchrl.trainers.loggers.wandb import WandbLogger

        logger = WandbLogger(
            f"dreamer/{exp_name}",
            project="torchrl",
            group=f"Dreamer_{cfg.env_name}",
        )
    elif cfg.logger == "csv":
        from torchrl.trainers.loggers.csv import CSVLogger

        logger = CSVLogger(
            f"{exp_name}",
            log_dir="dreamer",
        )
    elif cfg.logger == "tensorboard":
        from torchrl.trainers.loggers.tensorboard import TensorboardLogger

        logger = TensorboardLogger(
            f"{exp_name}",
            log_dir="dreamer",
        )
    else:
        raise NotImplementedError(cfg.logger)

    video_tag = f"Dreamer_{cfg.env_name}_policy_test" if cfg.record_video else ""

    stats = None
    if not cfg.vecnorm and cfg.norm_stats:
        stats = get_stats_random_rollout(
            cfg,
            key="next_pixels" if cfg.from_pixels else "next_observation_vector",
        )
    elif cfg.from_pixels:
        stats = {"loc": 0.5, "scale": 0.5}
    world_model, model_based_env, actor_model, value_model, policy = make_dreamer(
        stats=stats,
        cfg=cfg,
        device=device,
        use_decoder_in_env=True,
        action_key="action",
        value_key="predicted_value",
    )

    # reward normalization
    if cfg.normalize_rewards_online:
        # if used the running statistics of the rewards are computed and the
        # rewards used for training will be normalized based on these.
        reward_normalizer = RewardNormalizer(
            scale=cfg.normalize_rewards_online_scale,
            decay=cfg.normalize_rewards_online_decay,
        )
    else:
        reward_normalizer = None

    # Losses
    world_model_loss = DreamerModelLoss(world_model, cfg).to(device)
    actor_loss = DreamerActorLoss(actor_model, value_model, model_based_env, cfg).to(
        device
    )
    value_loss = DreamerValueLoss(value_model).to(device)

    # optimizers
    world_model_opt = torch.optim.Adam(world_model.parameters(), lr=cfg.world_model_lr)
    actor_opt = torch.optim.Adam(actor_model.parameters(), lr=cfg.actor_value_lr)
    value_opt = torch.optim.Adam(value_model.parameters(), lr=cfg.actor_value_lr)

    # Actor and value network
    model_explore = AdditiveGaussianWrapper(policy, sigma_init=0.3, sigma_end=0.3).to(
        device
    )

    action_dim_gsde, state_dim_gsde = None, None
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
    print("collector:", collector)

    replay_buffer = make_replay_buffer(device, cfg)

    record = Recorder(
        record_frames=cfg.record_frames,
        frame_skip=cfg.frame_skip,
        policy_exploration=policy,
        recorder=make_recorder_env(cfg, video_tag, stats, logger, create_env_fn),
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
    scaler3 = GradScaler()
    for i, tensordict in enumerate(collector):
        torch.cuda.empty_cache()

        # update weights of the inference policy
        collector.update_policy_weights_()
        if reward_normalizer is not None:
            reward_normalizer.update_reward_stats(tensordict)
        if r0 is None:
            r0 = tensordict["reward"].mean().item()
        pbar.update(tensordict.numel())
        current_frames = tensordict.numel()
        collected_frames += current_frames
        # tensordict = tensordict.reshape(-1, cfg.batch_length)
        replay_buffer.extend(tensordict.cpu())
        logger.log_scalar(
            "r_training",
            tensordict["reward"].mean().detach().cpu().item(),
            step=collected_frames,
        )

        if (i % cfg.record_interval) == 0:
            do_log = True
        else:
            do_log = False

        if collected_frames >= cfg.init_random_frames:
            for j in range(cfg.optim_steps_per_batch):
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(
                    device, non_blocking=True
                )
                if reward_normalizer is not None:
                    sampled_tensordict = reward_normalizer.normalize_reward(
                        sampled_tensordict
                    )

                with autocast(dtype=torch.float16):
                    model_loss_td, sampled_tensordict = world_model_loss(
                        sampled_tensordict
                    )
                    if (
                        cfg.record_video
                        and (record._count + 1) % cfg.record_interval == 0
                    ):
                        sampled_tensordict_save = (
                            sampled_tensordict.select(
                                "pixels",
                                "reco_pixels",
                                "posterior_states",
                                "next_belief",
                            )[:4]
                            .detach()
                            .to_tensordict()
                        )
                    else:
                        sampled_tensordict_save = None

                scaler1.scale(model_loss_td["loss_world_model"]).backward()
                scaler1.unscale_(world_model_opt)
                clip_grad_norm_(world_model.parameters(), cfg.grad_clip)
                scaler1.step(world_model_opt)
                if j == cfg.optim_steps_per_batch - 1 and do_log:
                    logger.log_scalar(
                        "loss_world_model",
                        model_loss_td["loss_world_model"].detach().cpu().item(),
                        step=collected_frames,
                    )
                    logger.log_scalar(
                        "grad_world_model",
                        grad_norm(world_model_opt),
                        step=collected_frames,
                    )
                world_model_opt.zero_grad()
                scaler1.update()

                with autocast(dtype=torch.float16):
                    actor_loss_td, sampled_tensordict = actor_loss(sampled_tensordict)
                scaler2.scale(actor_loss_td["loss_actor"]).backward()
                scaler2.unscale_(actor_opt)
                clip_grad_norm_(actor_model.parameters(), cfg.grad_clip)
                scaler2.step(actor_opt)
                if j == cfg.optim_steps_per_batch - 1 and do_log:
                    logger.log_scalar(
                        "loss_actor",
                        actor_loss_td["loss_actor"].detach().cpu().item(),
                        step=collected_frames,
                    )
                    logger.log_scalar(
                        "grad_actor",
                        grad_norm(actor_opt),
                        step=collected_frames,
                    )
                actor_opt.zero_grad()
                scaler2.update()

                with autocast(dtype=torch.float16):
                    value_loss_td, sampled_tensordict = value_loss(sampled_tensordict)
                scaler3.scale(value_loss_td["loss_value"]).backward()
                scaler3.unscale_(value_opt)
                clip_grad_norm_(value_model.parameters(), cfg.grad_clip)
                scaler3.step(value_opt)
                if j == cfg.optim_steps_per_batch - 1 and do_log:
                    logger.log_scalar(
                        "loss_value",
                        value_loss_td["loss_value"].detach().cpu().item(),
                        step=collected_frames,
                    )
                    logger.log_scalar(
                        "grad_value",
                        grad_norm(value_opt),
                        step=collected_frames,
                    )
                value_opt.zero_grad()
                scaler3.update()
                if j == cfg.optim_steps_per_batch - 1:
                    do_log = False

            call_record(
                logger,
                record,
                collected_frames,
                sampled_tensordict_save,
                stats,
                model_based_env,
                actor_model,
                cfg,
            )


def recover_pixels(pixels, stats):
    return (
        (255 * (pixels * stats["scale"] + stats["loc"]))
        .clamp(min=0, max=255)
        .to(torch.uint8)
    )


if __name__ == "__main__":
    main()
