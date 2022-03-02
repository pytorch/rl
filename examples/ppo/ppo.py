import uuid
from datetime import datetime

try:
    import configargparse as argparse
    _configargparse = True
except:
    import argparse
    _configargparse = False
import torch.cuda
from torch.utils.tensorboard import SummaryWriter

from torchrl.agents.helpers.agents import parser_agent_args, make_agent
from torchrl.agents.helpers.collectors import (
    parser_collector_args_online,
    make_collector_online,
)
from torchrl.agents.helpers.envs import (
    transformed_env_constructor,
    parallel_env_constructor,
    correct_for_frame_skip,
    get_stats_random_rollout,
    parser_env_args,
)
from torchrl.agents.helpers.losses import make_ppo_loss, parser_loss_args_ppo
from torchrl.agents.helpers.models import make_ppo_model, parser_model_args_continuous
from torchrl.agents.helpers.recorder import parser_recorder_args
from torchrl.data.transforms import TransformedEnv, RewardScaling


def make_args():
    parser = argparse.ArgumentParser()
    if _configargparse:
        parser.add_argument(
            "-c", "--config", required=True, is_config_file=True, help="config file path"
        )
    parser_agent_args(parser)
    parser_collector_args_online(parser)
    parser_env_args(parser)
    parser_loss_args_ppo(parser)
    parser_model_args_continuous(parser, "PPO")

    parser_recorder_args(parser)
    return parser


parser = make_args()

if __name__ == "__main__":
    args = parser.parse_args()

    args = correct_for_frame_skip(args)

    if not isinstance(args.reward_scaling, float):
        args.reward_scaling = 1.0

    device = (
        torch.device("cpu")
        if torch.cuda.device_count() == 0
        else torch.device("cuda:0")
    )

    exp_name = "_".join(
        [
            "PPO",
            args.exp_name,
            str(uuid.uuid4())[:8],
            datetime.now().strftime("%y_%m_%d-%H_%M_%S"),
        ]
    )
    writer = SummaryWriter(f"ppo_logging/{exp_name}")
    video_tag = exp_name if args.record_video else ""

    proof_env = transformed_env_constructor(args=args, use_env_creator=False)()
    model = make_ppo_model(proof_env, args, device)
    actor_model = model.get_policy_operator()

    loss_module = make_ppo_loss(model, args)

    stats = None
    if not args.vecnorm:
        stats = get_stats_random_rollout(args, proof_env)

    create_env_fn = parallel_env_constructor(args=args, stats=stats)

    collector = make_collector_online(
        make_env=create_env_fn,
        actor_model_explore=actor_model,
        args=args,
    )

    recorder = transformed_env_constructor(
        args, video_tag=video_tag, norm_obs_only=True, stats=stats, writer=writer
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
        collector, loss_module, recorder, None, actor_model, None, writer, args
    )

    agent.train()
