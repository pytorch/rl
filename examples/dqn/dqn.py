import uuid
from datetime import datetime

import configargparse
import torch.cuda
from torch.utils.tensorboard import SummaryWriter

from torchrl.agents.helpers.agents import parser_agent_args, make_agent
from torchrl.agents.helpers.collectors import parser_collector_args_offline, make_collector_offline
from torchrl.agents.helpers.envs import transformed_env_constructor, \
    parallel_env_constructor, \
    correct_for_frame_skip, get_stats_random_rollout, parser_env_args
from torchrl.agents.helpers.losses import parser_loss_args_offline, make_dqn_loss
from torchrl.agents.helpers.models import make_dqn_actor, parser_model_args_discrete
from torchrl.agents.helpers.recorder import parser_recorder_args
from torchrl.agents.helpers.replay_buffer import parser_replay_args, make_replay_buffer
from torchrl.data.transforms import TransformedEnv, RewardScaling
from torchrl.modules import EGreedyWrapper


def make_args():
    parser = configargparse.ArgumentParser()
    parser.add_argument('-c', '--config', required=True, is_config_file=True, help='config file path')
    parser_agent_args(parser)
    parser_collector_args_offline(parser)
    parser_env_args(parser)
    parser_loss_args_offline(parser)
    parser_model_args_discrete(parser)
    parser_recorder_args(parser)
    parser_replay_args(parser)
    return parser


parser = make_args()

if __name__ == "__main__":
    args = parser.parse_args()

    args = correct_for_frame_skip(args)

    if not isinstance(args.reward_scaling, float):
        args.reward_scaling = 1.0

    device = torch.device("cpu") if torch.cuda.device_count() == 0 else torch.device('cuda:0')

    exp_name = "_".join(["DQN", args.exp_name, str(uuid.uuid4())[:8], datetime.now().strftime("%y_%m_%d-%H_%M_%S")])
    writer = SummaryWriter(f"dqn_logging/{exp_name}")
    video_tag = exp_name if args.record_video else ""

    proof_env = transformed_env_constructor(args=args, use_env_creator=False)()
    model = make_dqn_actor(
        proof_environment=proof_env,
        device=device,
        args=args,
    )

    loss_module, target_net_updater = make_dqn_loss(model, args)
    model_explore = EGreedyWrapper(model, annealing_num_steps=args.annealing_frames).to(device)

    stats = None
    if not args.vecnorm:
        stats = get_stats_random_rollout(args, proof_env)

    create_env_fn = parallel_env_constructor(args=args, stats=stats)

    collector = make_collector_offline(
        make_env=create_env_fn,
        actor_model_explore=model_explore,
        args=args,
    )

    replay_buffer = make_replay_buffer(device, args)

    recorder = transformed_env_constructor(
        args,
        video_tag=video_tag,
        norm_obs_only=True,
        stats=stats,
        writer=writer)()

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
        model_explore,
        replay_buffer,
        writer,
        args)

    agent.train()
