import dataclasses
from dataclasses import dataclass
import uuid
from datetime import datetime

import hydra
import torch.cuda
from hydra.core.config_store import ConfigStore
from torchrl.envs import ParallelEnv, EnvCreator
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.envs.model_based import ModelBasedEnv
from torchrl.modules.tensordict_module.actors import (
    ActorCriticWrapper,
    WorldModelWrapper,
)
from torchrl.modules.tensordict_module.sequence import TensorDictSequence
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
from torchrl.modules import TensorDictModule
from torchrl.modules.models import MLP, TanhActor
from torchrl.modules.tensordict_module.world_models import DreamerWorldModeler
from torchrl.objectives.costs.dreamer import DreamerBehaviourLoss, DreamerModelLoss
from torchrl.trainers.helpers.recorder import RecorderConfig
from torchrl.trainers.helpers.replay_buffer import (
    make_replay_buffer,
    ReplayArgsConfig,
)
import tqdm
import torch.nn as nn
import torch
from torch.nn.utils import clip_grad_norm_

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
        TrainingConfig

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

    if not isinstance(cfg.reward_scaling, float):
        cfg.reward_scaling = 1.0

    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

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

    #### World Model and reward model
    world_modeler = DreamerWorldModeler(rssm_hidden=cfg.rssm_hidden_dim, rnn_hidden_dim=cfg.rssm_hidden_dim, state_dim=cfg.state_dim)
    reward_model = TensorDictModule(
        MLP(out_features=1, depth=3, num_cells=300, activation_class=nn.ELU),
        in_keys=["posterior_state", "belief"],
        out_keys=["predicted_reward"],
    )
    world_model = WorldModelWrapper(world_modeler, reward_model)
    model_based_env = ModelBasedEnv(
        world_model=WorldModelWrapper(
            world_modeler.select_subsequence(
                in_keys=["prior_state", "belief", "action"],
                out_keys=["next_prior_state", "next_belief"],
            ),
            TensorDictModule(
                reward_model.module,
                in_keys=["next_prior_state", "next_belief"],
                out_keys=["predicted_reward"],
            ),
        ),
        device=device,
    )

    ### Actor and Value models
    actor_model = TensorDictModule(
        TanhActor(
            out_features=proof_env.action_spec.shape[0],
            depth=3,
            num_cells=300,
            activation_class=nn.ELU,
        ),
        in_keys=["prior_state", "belief"],
        out_keys=["action"],
    )
    value_model = TensorDictModule(
        MLP(out_features=1, depth=3, num_cells=400, activation_class=nn.ELU),
        in_keys=["prior_state", "belief"],
        out_keys=["predicted_value"],
    )
    actor_value_model = ActorCriticWrapper(actor_model, value_model)

    ### Policy to compute inference from observations
    policy = TensorDictSequence(
        world_modeler.select_subsequence(
            out_keys=["posterior_state", "belief"],
        ),
        TensorDictModule(
            actor_model.module,
            in_keys=["posterior_state", "belief"],
            out_keys=["action"],
        ),
    )
    #### Losses
    behaviour_loss = DreamerBehaviourLoss()
    world_model_loss = DreamerModelLoss()

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
            video_tag=video_tag,
        )
    else:
        recorder = None

    #### Actor and value network
    model_explore = policy

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

    # trainer = make_trainer(
    #     collector,
    #     loss_module,
    #     recorder,
    #     target_net_updater,
    #     model,
    #     replay_buffer,
    #     logger,
    #     cfg,
    # )

    final_seed = collector.set_seed(cfg.seed)
    print(f"init seed: {cfg.seed}, final seed: {final_seed}")
    ## Training loop
    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.total_frames//cfg.frame_skip)
    r0 = None
    for i, tensordict in enumerate(collector):

        # update weights of the inference policy
        collector.update_policy_weights_()
        
        if r0 is None:
            r0 = tensordict["reward"].mean().item()
        pbar.update(tensordict.numel())
        
        # extend the replay buffer with the new data
        if "mask" in tensordict.keys():
            # if multi-step, a mask is present to help filter padded values
            current_frames = tensordict["mask"].sum()
            tensordict = tensordict[tensordict.get("mask").squeeze(-1)]
        else:
            tensordict = tensordict.view(-1)
            current_frames = tensordict.numel()
        collected_frames += current_frames
        replay_buffer.extend(tensordict.cpu())

        # optimization steps
        if collected_frames >= cfg.init_env_steps:
            for j in range(cfg.optim_steps_per_batch):
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample(cfg.batch_size)
                sampled_tensordict.batch_size = [sampled_tensordict.shape[0]]
                sampled_tensordict["initial_state"] = torch.zeros((sampled_tensordict.batch_size[0],1,cfg.state_dim))
                sampled_tensordict["initial_belief"] = torch.zeros((sampled_tensordict.batch_size[0],1,cfg.rssm_hidden_dim))
                world_model.train()
                sampled_tensordict = world_model(sampled_tensordict)
                # compute model loss
                model_loss_td = world_model_loss(sampled_tensordict)
                world_model_opt.zero_grad()
                model_loss_td["loss"].backward()
                clip_grad_norm_(world_model.parameters(), cfg.grad_clip)
                world_model_opt.step()

                flattened_td = sampled_tensordict.select("prior_state", "belief").view(-1, sampled_tensordict.shape[-1]).detach()
                with torch.no_grad:
                    flattened_td = model_based_env.rollout(max_steps=cfg.imagination_horizon, policy=actor_model, auto_reset=False, tensordict=flattened_td)
                flattened_td = actor_value_model(flattened_td)
                # compute actor loss
                actor_value_loss_td = behaviour_loss(flattened_td)
                actor_opt.zero_grad()
                actor_value_loss_td["loss_actor"].backward()
                clip_grad_norm_(actor_model.parameters(), cfg.grad_clip)
                actor_opt.step()

                # Optimize value function
                value_opt.zero_grad()
                actor_value_loss_td["loss_value"].backward()
                clip_grad_norm_(value_model.parameters(), cfg.grad_clip)
                value_opt.step()
            td_record = recorder(None)

if __name__ == "__main__":
    main()


