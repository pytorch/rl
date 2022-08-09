import dataclasses
import uuid
from datetime import datetime

import hydra
import torch.cuda
from hydra.core.config_store import ConfigStore
from torchrl.envs import ParallelEnv, EnvCreator
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.envs.model_based import ModelBasedEnv
from torchrl.modules import EGreedyWrapper
from torchrl.modules.tensordict_module.actors import (
    ActorCriticWrapper,
    WorldModelWrapper,
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
from torchrl.modules import TensorDictModule
from torchrl.modules.models import MLP, TanhActor
from torchrl.modules.tensordict_module.world_models import DreamerWorldModeler
from torchrl.objectives.costs.dreamer import DreamerBehaviourLoss, DreamerModelLoss
from torchrl.trainers.helpers.recorder import RecorderConfig
from torchrl.trainers.helpers.replay_buffer import (
    make_replay_buffer,
    ReplayArgsConfig,
)

import torch.nn as nn

DEFAULT_REWARD_SCALING = {
    "Hopper-v1": 5,
    "Walker2d-v1": 5,
    "HalfCheetah-v1": 5,
    "cheetah": 5,
    "Ant-v2": 5,
    "Humanoid-v2": 20,
    "humanoid": 100,
}

config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        OffPolicyCollectorConfig,
        EnvConfig,
        RecorderConfig,
        ReplayArgsConfig,
    )
    for config_field in dataclasses.fields(config_cls)
]
Config = dataclasses.make_dataclass(cls_name="Config", fields=config_fields)
cs = ConfigStore.instance()
cs.store(name="config", node=Config)


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

    #### Model Based model and loss
    world_modeler = DreamerWorldModeler()
    reward_model = TensorDictModule(
        MLP(out_features=1, depth=3, num_cells=300, activation_class=nn.ELU),
        in_keys=["posterior_state", "belief"],
        out_keys=["predicted_reward"],
    )
    world_model = WorldModelWrapper(world_modeler, reward_model)
    model_based_env = ModelBasedEnv(
        world_model=WorldModelWrapper(
            world_modeler.select_subsequence(
                in_keys=["initial_state", "initial_rnn_hidden", "action"],
                out_keys=["prior_states", "prior_rnn_hiddens"],
            ),
            TensorDictModule(
                reward_model.module,
                in_keys=["prior_states", "belief"],
                out_keys=["predicted_reward"],
            ),
        ),
        device=device,
    )

    world_model_loss = DreamerModelLoss()
    world_model_opt = torch.optim.Adam(world_model.parameters(), lr=cfg.lr)
    ### Actor and Value models
    actor_model = TensorDictModule(
        TanhActor(
            out_features=proof_env.action_spec.shape[0],
            depth=3,
            num_cells=300,
            activation_class=nn.ELU,
        ),
        in_keys=["posterior_state", "belief"],
        out_keys=["action"],
    )
    value_model = TensorDictModule(
        MLP(out_features=1, depth=3, num_cells=400, activation_class=nn.ELU),
        in_keys=["posterior_state", "belief"],
        out_keys=["predicted_value"],
    )
    actor_value_model = ActorCriticWrapper(actor_model, value_model)
    behaviour_loss = DreamerBehaviourLoss()
    #### Off Policy Collector
    collector = make_collector_offpolicy(
        cfg=cfg,
        env=model_based_env,
        device=device,
        logger=logger,
        video_tag=video_tag,
    )

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

    #### Replay Buffer
    replay_buffer = make_replay_buffer(
        cfg=cfg, env=proof_env, device=device, logger=logger
    )

    #### Training Loop
    collector.train(
        replay_buffer=replay_buffer,
        recorder=recorder,
        num_episodes=cfg.num_episodes,
        num_steps=cfg.num_steps,
        num_steps_per_episode=cfg.num_steps_per_episode,
        num_steps_per_eval=cfg.num_steps_per_eval,
        num_steps_per_video=cfg.num_steps_per_video,
        num_steps_per_log=cfg.num_steps_per_log,
        num_steps_per_save=cfg.num_steps_per_save,
    )

    #### Actor and value network
    model_explore = EGreedyWrapper(model, annealing_num_steps=cfg.annealing_frames).to(
        device
    )

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
    ## Training loop
    trainer.train()
    return (logger.log_dir, trainer._log_dict)


if __name__ == "__main__":
    main()
