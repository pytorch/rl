import uuid
from datetime import datetime

import configargparse
import torch.cuda
from torch.utils.tensorboard import SummaryWriter

from torchrl.agents.helpers.agents import parser_agent_args, make_agent
from torchrl.agents.helpers.collectors import (
    parser_collector_args_offline,
    make_collector_offline,
)
from torchrl.agents.helpers.envs import (
    parser_env_args,
    transformed_env_constructor,
    parallel_env_constructor,
    correct_for_frame_skip,
    get_stats_random_rollout,
)
from torchrl.agents.helpers.losses import parser_loss_args_offline, make_ddpg_loss
from torchrl.agents.helpers.models import parser_model_args_continuous, make_ddpg_actor
from torchrl.agents.helpers.recorder import parser_recorder_args
from torchrl.agents.helpers.replay_buffer import parser_replay_args, make_replay_buffer
from torchrl.data.transforms import TransformedEnv, RewardScaling
from torchrl.modules import OrnsteinUhlenbeckProcessWrapper


def make_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", required=True, is_config_file=True, help="config file path"
    )
    parser_agent_args(parser)
    parser_collector_args_offline(parser)
    parser_env_args(parser)
    parser_loss_args_offline(parser)
    parser_model_args_continuous(parser, "DDPG")
    parser_recorder_args(parser)
    parser_replay_args(parser)
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

if __name__ == "__main__":
    args = parser.parse_args()

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
        args.from_pixels,
        noisy=args.noisy,
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
    if not args.vecnorm:
        stats = get_stats_random_rollout(args, proof_env)

    create_env_fn = parallel_env_constructor(args=args, stats=stats)

    collector = make_collector_offline(
        make_env=create_env_fn,
        actor_model_explore=actor_model_explore,
        args=args,
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
    recorder_rm.load_state_dict(create_env_fn.state_dict()["worker0"])
    # reset reward scaling
    for t in recorder.transform:
        if isinstance(t, RewardScaling):
            t.scale.fill_(1.0)

    agent = make_agent(
        collector,
        loss_module,
        recorder,
        target_net_updater,
        actor_model_explore,
        replay_buffer,
        writer,
        args,
    )

    agent.train()
