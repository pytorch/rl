# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import dataclasses
import uuid
from datetime import datetime

import hydra
import torch.cuda
import tqdm
from hydra.core.config_store import ConfigStore
from torch import optim
from torchrl.envs import ParallelEnv, EnvCreator
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import OrnsteinUhlenbeckProcessWrapper
from torchrl.record import VideoRecorder
from torchrl.trainers import Recorder
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
from torchrl.trainers.helpers.losses import make_ddpg_loss, LossConfig
from torchrl.trainers.helpers.models import (
    make_ddpg_actor,
    DDPGModelConfig,
)
from torchrl.trainers.helpers.recorder import RecorderConfig
from torchrl.trainers.helpers.replay_buffer import (
    make_replay_buffer,
    ReplayArgsConfig,
)
from torchrl.trainers.helpers.trainers import TrainerConfig

config_fields = [
    (config_field.name, config_field.type, config_field)
    for config_cls in (
        TrainerConfig,
        OffPolicyCollectorConfig,
        EnvConfig,
        LossConfig,
        DDPGModelConfig,
        RecorderConfig,
        ReplayArgsConfig,
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
    from torch.utils.tensorboard import SummaryWriter

    cfg = correct_for_frame_skip(cfg)

    if not isinstance(cfg.reward_scaling, float):
        cfg.reward_scaling = DEFAULT_REWARD_SCALING.get(cfg.env_name, 5.0)

    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    exp_name = "_".join(
        [
            "DDPG",
            cfg.exp_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        ]
    )
    writer = SummaryWriter(f"ddpg_logging/{exp_name}")
    video_tag = exp_name if cfg.record_video else ""

    stats = None
    if not cfg.vecnorm and cfg.norm_stats:
        proof_env = transformed_env_constructor(cfg=cfg, use_env_creator=False)()
        stats = get_stats_random_rollout(
            cfg, proof_env, key="next_pixels" if cfg.from_pixels else None
        )
        # make sure proof_env is closed
        proof_env.close()
    elif cfg.from_pixels:
        stats = {"loc": 0.5, "scale": 0.5}
    proof_env = transformed_env_constructor(
        cfg=cfg, use_env_creator=False, stats=stats
    )()

    model = make_ddpg_actor(
        proof_env,
        cfg=cfg,
        device=device,
    )
    loss_module, target_net_updater = make_ddpg_loss(model, cfg)

    actor_model_explore = model[0]
    if cfg.ou_exploration:
        if cfg.gSDE:
            raise RuntimeError("gSDE and ou_exploration are incompatible")
        actor_model_explore = OrnsteinUhlenbeckProcessWrapper(
            actor_model_explore,
            annealing_num_steps=cfg.annealing_frames,
            sigma=cfg.ou_sigma,
            theta=cfg.ou_theta,
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

    proof_env.close()
    create_env_fn = parallel_env_constructor(
        cfg=cfg,
        stats=stats,
        action_dim_gsde=action_dim_gsde,
        state_dim_gsde=state_dim_gsde,
    )

    collector = make_collector_offpolicy(
        make_env=create_env_fn,
        actor_model_explore=actor_model_explore,
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
        writer=writer,
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

    recorder_obj = Recorder(
        record_frames=cfg.record_frames,
        frame_skip=cfg.frame_skip,
        policy_exploration=actor_model_explore,
        recorder=recorder,
        record_interval=cfg.record_interval,
        log_keys=cfg.recorder_log_keys,
    )

    optimizer = optim.Adam(
        loss_module.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    collected_frames = 0
    pbar = tqdm.tqdm(total=cfg.total_frames)
    for i, tensordict in enumerate(collector):
        pbar.update(tensordict.numel())
        # extend the replay buffer with the new data
        if "mask" in tensordict.keys():
            # if multi-step, a mask is present to help filter padded values
            collected_frames += tensordict["mask"].sum()
            tensordict = tensordict[tensordict.get("mask").squeeze(-1)]
        else:
            collected_frames += tensordict.numel()
        replay_buffer.extend(tensordict.cpu())

        if collected_frames >= cfg.init_random_frames:
            for j in range(cfg.optim_steps_per_batch):
                # sample from replay buffer
                sampled_tensordict = replay_buffer.sample(cfg.batch_size)

                # compute loss
                loss_td = loss_module(sampled_tensordict)

                # update the target net
                target_net_updater.step()

                # update priority
                if cfg.prb:
                    replay_buffer.update_priority(sampled_tensordict)

                # step
                optimizer.zero_grad()
                loss = sum(
                    [
                        loss_td.get(key)
                        for key in loss_td.keys()
                        if key.startswith("loss_")
                    ]
                )
                loss.backward()
                optimizer.step()

                if j == cfg.optim_steps_per_batch - 1:
                    for key in loss_td.keys():
                        writer.add_scalar(
                            key, loss_td.get(key).mean(), global_step=collected_frames
                        )

        # some logging
        writer.add_scalar(
            "r_training", tensordict.get("reward").mean(), global_step=collected_frames
        )
        recorder_obj(None)

        # update weights of the inference policy
        collector.update_policy_weights_()


if __name__ == "__main__":
    main()
