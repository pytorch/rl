import dataclasses
import uuid
from dataclasses import dataclass
from datetime import datetime
import hydra
import torch
import torch.cuda
import tqdm
from hydra.core.config_store import ConfigStore
from torch.nn.utils import clip_grad_norm_
from torchrl.envs import ParallelEnv, EnvCreator
from torchrl.trainers.trainers import Recorder
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.modules import TensorDictModule
from torchrl.modules.tensordict_module.sequence import TensorDictSequence
from torchrl.objectives.costs.dreamer import DreamerActorLoss, DreamerModelLoss, DreamerValueLoss
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
from torchrl.trainers.helpers.models import (
    make_dreamer,
    DreamerConfig,
)
from torchrl.trainers.helpers.recorder import RecorderConfig
from torchrl.trainers.helpers.replay_buffer import (
    make_replay_buffer,
    ReplayArgsConfig,
)

from pathlib import Path

### bf16
from  torch.cuda.amp import autocast, GradScaler
@dataclass
class DreamerConfig:
    state_dim: int = 20
    rssm_hidden_dim: int = 200
    grad_clip: int = 100
    world_model_lr: float = 6e-4
    actor_value_lr: float = 8e-5
    imagination_horizon: int = 15


@dataclass
class TrainingConfig:
    optim_steps_per_batch: int = 500
    # Number of optimization steps in between two collection of data. See frames_per_batch below.
    # LR scheduler.
    batch_size: int = 256
    # batch size of the TensorDict retrieved from the replay buffer. Default=256.
    log_interval: int = 10000
    # logging interval, in terms of optimization steps. Default=10000.


config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        OffPolicyCollectorConfig,
        EnvConfig,
        RecorderConfig,
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

@hydra.main(version_base=None, config_path=".", config_name="config")
def main(cfg: "DictConfig"):

    from torchrl.trainers.loggers.wandb import WandbLogger

    cfg = correct_for_frame_skip(cfg)

    dtype = torch.float16

    if not isinstance(cfg.reward_scaling, float):
        cfg.reward_scaling = 1.0

    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda:0")
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
    logger = WandbLogger(
        f"dreamer/{exp_name}", project="torchrl", group=f"Dreamer_{cfg.env_name}"
    )
    video_tag = exp_name if cfg.record_video else ""

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
    world_model, model_based_env, actor_model, value_model, policy = make_dreamer(
        proof_environment=proof_env, cfg=cfg, device=device, use_decoder_in_env=True,action_key="action",
        value_key="predicted_value"
    )

    #### Losses
    world_model_loss = DreamerModelLoss(world_model, cfg).to(device)
    actor_loss = DreamerActorLoss(
        actor_model, value_model, model_based_env, cfg
    ).to(device)
    value_loss = DreamerValueLoss(value_model).to(device)

    ### optimizers
    world_model_opt = torch.optim.Adam(world_model.parameters(), lr=cfg.world_model_lr)
    actor_opt = torch.optim.Adam(actor_model.parameters(), lr=cfg.actor_value_lr)
    value_opt = torch.optim.Adam(value_model.parameters(), lr=cfg.actor_value_lr)

    #### Recorder
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

    #### Actor and value network
    model_explore =policy
    
    # model_explore = OrnsteinUhlenbeckProcessWrapper(
    #     policy,
    #     annealing_num_steps=cfg.annealing_frames,
    #     sigma=cfg.ou_sigma,
    #     theta=cfg.ou_theta,
    # ).to(device)

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
        #     {"device": device} if device >= 0 else {}
        #     for device in args.env_rendering_devices
        # ],
    )

    replay_buffer = make_replay_buffer(device, cfg)

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
    ## Training loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)
    r0 = None
    path = Path('./log')
    path.mkdir(exist_ok=True)

    scaler= GradScaler()
    for i, tensordict in enumerate(collector):

        # update weights of the inference policy
        collector.update_policy_weights_()

        if r0 is None:
            r0 = tensordict["reward"].mean().item()
        pbar.update(tensordict.numel())

        current_frames = tensordict.numel()
        collected_frames += current_frames
        replay_buffer.extend(tensordict.cpu())

        # optimization steps
        # with torch.profiler.profile(
        #     activities=[
        #     torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA], 
        #     schedule=torch.profiler.schedule(skip_first=125, wait=1, warmup=1, active=1, repeat=2),
        #     on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/dreamer'),
        #     record_shapes=True,
        #     profile_memory=True,
        #     with_stack=True,
        #     with_modules=True
        #     ) as prof:
        if collected_frames >= cfg.init_env_steps:
            for j in range(cfg.optim_steps_per_batch):
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample(cfg.batch_size).to(device)
                with autocast():
                    model_loss_td, sampled_tensordict = world_model_loss(sampled_tensordict)
                    actor_loss_td, sampled_tensordict = actor_loss(
                        sampled_tensordict
                    )
                    value_loss_td, sampled_tensordict = value_loss(
                        sampled_tensordict
                    )
                
                
                scaler.scale(model_loss_td["loss_world_model"]).backward()
                scaler.scale(actor_loss_td["loss_actor"]).backward()
                scaler.scale(value_loss_td["loss_value"]).backward()

                scaler.unscale_(world_model_opt)
                clip_grad_norm_(world_model.parameters(), cfg.grad_clip)
                scaler.unscale_(actor_opt)
                clip_grad_norm_(actor_model.parameters(), cfg.grad_clip)
                scaler.unscale_(value_opt)
                clip_grad_norm_(value_model.parameters(), cfg.grad_clip)

                scaler.step(world_model_opt)
                world_model_opt.zero_grad()   

                scaler.step(actor_opt)
                actor_opt.zero_grad()
                
                scaler.step(value_opt)
                value_opt.zero_grad()
                
                scaler.update()

                with torch.no_grad():
                    td_record = record(None)
                    if td_record is not None:
                        for key, value in td_record.items():
                            if key in ['r_evaluation', 'total_r_evaluation']:
                                logger.log_scalar(key, value.detach().cpu().numpy(), step=collected_frames)
                    # Compute observation reco
                    if record._count % cfg.record_interval == 0 and cfg.record_video:
                        reco_pxls = (model_based_env.decode_obs(
                            sampled_tensordict[:5]
                        ).detach()["reco_pixels"] - stats["loc"])/stats["scale"]
                        logger.log_video("reco_observation", reco_pxls.cpu().numpy())
    
        

if __name__ == "__main__":
    main()
