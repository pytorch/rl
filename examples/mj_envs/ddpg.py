# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import uuid
from datetime import datetime

from torchrl.envs import ParallelEnv
from torchrl.trainers.helpers.envs import LIBS
from utils import MJEnv

LIBS["mjenv"] = MJEnv

try:
    import configargparse as argparse

    _configargparse = True
except ImportError:
    import argparse

    _configargparse = False
import torch.cuda
from torch.utils.tensorboard import SummaryWriter
from torchrl.envs.transforms import RewardScaling, TransformedEnv
from torchrl.modules import OrnsteinUhlenbeckProcessWrapper
from torchrl.trainers.helpers.collectors import (
    make_collector_offpolicy,
    parser_collector_args_offpolicy,
)
from torchrl.trainers.helpers.envs import (
    correct_for_frame_skip,
    get_stats_random_rollout,
    parallel_env_constructor,
    parser_env_args,
    transformed_env_constructor,
)
from torchrl.trainers.helpers.losses import make_ddpg_loss, parser_loss_args
from torchrl.trainers.helpers.models import (
    make_ddpg_actor,
    parser_model_args_continuous,
)
from torchrl.trainers.helpers.recorder import parser_recorder_args
from torchrl.trainers.helpers.replay_buffer import (
    make_replay_buffer,
    parser_replay_args,
)
from torchrl.trainers.helpers.trainers import make_trainer, parser_trainer_args


def make_args():
    parser = argparse.ArgumentParser()
    if _configargparse:
        parser.add_argument(
            "-c",
            "--config",
            required=True,
            is_config_file=True,
            help="config file path",
        )
    parser_trainer_args(parser)
    parser_collector_args_offpolicy(parser)
    parser_env_args(parser)
    parser_loss_args(parser, algorithm="DDPG")
    parser_model_args_continuous(parser, "DDPG")
    parser_recorder_args(parser)
    parser_replay_args(parser)
    parser.add_argument(
        "--env_rendering_device",
        "--env-rendering-device",
        type=int,
        nargs="+",
        default=[0],
    )
    return parser


parser = make_args()

DEFAULT_REWARD_SCALING = {
    "Hopper-v1": 5,
    "Walker2d-v1": 5,
    "HalfCheetah-v1": 5,
    "cheetah": 5,
    "Ant-v2": 5,
    "Humanoid-v2": 20,
    "humanoid": 100,
}


def main(args):
    args = correct_for_frame_skip(args)

    if not isinstance(args.reward_scaling, float):
        args.reward_scaling = DEFAULT_REWARD_SCALING.get(args.env_name, 5.0)

    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    exp_name = "_".join(
        [
            "DDPG",
            args.exp_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        ]
    )
    writer = SummaryWriter(f"ddpg_logging/{exp_name}")
    video_tag = exp_name if args.record_video else ""

    proof_env = transformed_env_constructor(args=args, use_env_creator=False)()
    model = make_ddpg_actor(
        proof_env,
        args,
        device=device,
    )
    loss_module, target_net_updater = make_ddpg_loss(model, args)
    actor_model_explore = model[0]
    if args.ou_exploration:
        actor_model_explore = OrnsteinUhlenbeckProcessWrapper(
            actor_model_explore, annealing_num_steps=args.annealing_frames
        ).to(device)
    if device == torch.device("cpu"):
        # mostly for debugging
        actor_model_explore.share_memory()

    stats = None
    if not args.vecnorm and args.norm_stats:
        stats = get_stats_random_rollout(args, proof_env)
    # make sure proof_env is closed
    proof_env.close()

    create_env_fn = parallel_env_constructor(args=args, stats=stats)

    collector = make_collector_offpolicy(
        make_env=create_env_fn,
        actor_model_explore=actor_model_explore,
        args=args,
        make_env_kwargs=[
            {"render_device": device, 'device': device} for device in args.env_rendering_device
        ],
    )

    replay_buffer = make_replay_buffer(device, args)

    recorder = transformed_env_constructor(
        args,
        video_tag=video_tag,
        norm_obs_only=True,
        stats=stats,
        writer=writer,
        use_env_creator=False,
    )()

    # remove video recorder from recorder to have matching state_dict keys
    if args.record_video:
        recorder_rm = TransformedEnv(recorder.env, recorder.transform[1:])
    else:
        recorder_rm = recorder
    if isinstance(create_env_fn, ParallelEnv):
        recorder_rm.load_state_dict(create_env_fn.state_dict()["worker0"])
    else:
        recorder_rm.load_state_dict(create_env_fn.state_dict())

    create_env_fn.close()
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
        actor_model_explore,
        replay_buffer,
        writer,
        args,
    )

    trainer.register_op("pre_steps_log", lambda batch: ("time", batch["time"].mean()))
    trainer.register_op(
        "pre_steps_log",
        lambda batch: ("solved", batch["solved"].sum() / batch["solved"].numel()),
    )
    trainer.register_op(
        "pre_steps_log", lambda batch: ("rwd_sparse", batch["rwd_sparse"].mean())
    )
    trainer.register_op(
        "pre_steps_log", lambda batch: ("rwd_sparse", batch["rwd_sparse"].mean())
    )

    trainer.train()
    return (writer.log_dir, trainer._log_dict, trainer.state_dict())


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
