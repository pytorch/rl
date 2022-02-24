import argparse
from argparse import ArgumentParser, Namespace

__all__ = ["make_agent", "parser_agent_args"]

from typing import Optional, Union
from warnings import warn

from torch import optim
from torch.utils.tensorboard import SummaryWriter

from torchrl.agents.agents import Agent
from torchrl.agents.helpers.collectors import parser_collector_args_offline
from torchrl.collectors.collectors import _DataCollector
from torchrl.data import ReplayBuffer
from torchrl.envs.common import _EnvClass
from torchrl.modules import TDModuleWrapper, TDModule
from torchrl.objectives.costs.common import _LossModule
from torchrl.objectives.costs.utils import _TargetNetUpdate

OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamax": optim.Adamax,
}


def make_agent(
    collector: _DataCollector,
    loss_module: _LossModule,
    recorder: Optional[_EnvClass] = None,
    target_net_updater: Optional[_TargetNetUpdate] = None,
    policy_exploration: Optional[Union[TDModuleWrapper, TDModule]] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    writer: Optional[SummaryWriter] = None,
    args: Optional[Namespace] = None,
) -> Agent:
    """
    Creates an Agent instance given its constituents.

    Args:
        collector (_DataCollector): A data collector to be used to collect data.
        loss_module (_LossModule): A TorchRL loss module
        recorder (_EnvClass, optional): a recorder environment. If None, the agent will train the policy without
            testing it.
        target_net_updater (_TargetNetUpdate, optional): A target network update object.
        policy_exploration (TDModule or TDModuleWrapper, optional): a policy to be used for recording and exploration
            updates (should be synced with the learnt policy).
        replay_buffer (ReplayBuffer, optional): a replay buffer to be used to collect data.
        writer (SummaryWriter, optional): a tensorboard SummaryWriter to be used for logging.
        args (argparse.Namespace, optional): a Namespace containing the arguments of the script. If None, the default
            arguments are used.

    Returns:
        An agent built with the input objects. The optimizer is built by this helper function using the args provided.

    Examples:
        >>> import torch
        >>> import tempfile
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> from torchrl.agents import Agent, EnvCreator
        >>> from torchrl.collectors.collectors import SyncDataCollector
        >>> from torchrl.data import TensorDictReplayBuffer
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.modules import TDModuleWrapper, TDModule, ValueOperator, EGreedyWrapper
        >>> from torchrl.objectives.costs.common import _LossModule
        >>> from torchrl.objectives.costs.utils import _TargetNetUpdate
        >>> from torchrl.objectives import DDPGLoss
        >>> env_maker = EnvCreator(lambda: GymEnv("Pendulum-v0"))
        >>> env_proof = env_maker()
        >>> obs_spec = env_proof.observation_spec
        >>> action_spec = env_proof.action_spec
        >>> net = torch.nn.Linear(env_proof.observation_spec.shape[-1], action_spec.shape[-1])
        >>> net_value = torch.nn.Linear(env_proof.observation_spec.shape[-1], 1)  # for the purpose of testing
        >>> policy = TDModule(action_spec, net, in_keys=["observation"], out_keys=["action"])
        >>> value = ValueOperator(net_value, in_keys=["observation"], out_keys=["state_action_value"])
        >>> collector = SyncDataCollector(env_maker, policy, total_frames=100)
        >>> loss_module = DDPGLoss(policy, value, gamma=0.99)
        >>> recorder = env_proof
        >>> target_net_updater = None
        >>> policy_exploration = EGreedyWrapper(policy)
        >>> replay_buffer = TensorDictReplayBuffer(1000)
        >>> dir = tempfile.gettempdir()
        >>> writer = SummaryWriter(log_dir=dir)
        >>> agent = make_agent(collector, loss_module, recorder, target_net_updater, policy_exploration,
        ...    replay_buffer, writer)
        >>> print(agent)

    """
    if args is None:
        warn(
            "Getting default args for the agent. This should be only used for debugging."
        )
        parser = parser_agent_args(argparse.ArgumentParser())
        parser.add_argument("--frame_skip", default=1)
        parser.add_argument("--total_frames", default=1000)
        parser.add_argument("--record_frames", default=10)
        parser.add_argument("--record_interval", default=10)
        args = parser.parse_args([])

    optimizer = OPTIMIZERS[args.optimizer](
        loss_module.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    optim_scheduler = None

    print(
        f"collector = {collector}; \n"
        f"loss_module = {loss_module}; \n"
        f"recorder = {recorder}; \n"
        f"target_net_updater = {target_net_updater}; \n"
        f"policy_exploration = {policy_exploration}; \n"
        f"replay_buffer = {replay_buffer}; \n"
        f"writer = {writer}; \n"
        f"args = {args}; \n"
    )

    if writer is not None:
        # log hyperparams
        txt = "\n\t".join([f"{k}: {val}" for k, val in sorted(vars(args).items())])
        writer.add_text("hparams", txt)

    return Agent(
        collector=collector,
        total_frames=args.total_frames * args.frame_skip,
        loss_module=loss_module,
        optimizer=optimizer,
        recorder=recorder,
        optim_scheduler=optim_scheduler,
        target_net_updater=target_net_updater,
        policy_exploration=policy_exploration,
        replay_buffer=replay_buffer,
        writer=writer,
        update_weights_interval=1,
        frame_skip=args.frame_skip,
        optim_steps_per_batch=args.optim_steps_per_collection,
        batch_size=args.batch_size,
        clip_grad_norm=args.clip_grad_norm,
        clip_norm=args.clip_norm,
        record_interval=args.record_interval,
        record_frames=args.record_frames,
        normalize_rewards_online=args.normalize_rewards_online,
        sub_traj_len=args.sub_traj_len,
    )


def parser_agent_args(parser: ArgumentParser) -> ArgumentParser:
    parser.add_argument(
        "--optim_steps_per_collection",
        type=int,
        default=500,
        help="Number of optimization steps in between two collection of data. See frames_per_batch "
        "below. "
        "Default=500",
    )
    parser.add_argument(
        "--optimizer", type=str, default="adam", help="Optimizer to be used."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=256,
        help="batch size of the TensorDict retrieved from the replay buffer. Default=64.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=10000,
        help="logging interval, in terms of optimization steps. Default=1000.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=3e-4,
        help="Learning rate used for the optimizer. Default=2e-4.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=2e-5,
        help="Weight-decay to be used with the optimizer. Default=0.0.",
    )
    parser.add_argument(
        "--clip_norm",
        type=float,
        default=1.0,
        help="value at which the total gradient norm should be clipped. Default=1.0",
    )
    parser.add_argument(
        "--clip_grad_norm",
        action="store_true",
        help="if called, the gradient will be clipped based on its L2 norm. Otherwise, single gradient "
        "values will be clipped to the desired threshold.",
    )
    parser.add_argument(
        "--normalize_rewards_online",
        "--normalize-rewards-online",
        action="store_true",
        help="Computes the running statistics of the rewards and normalizes them before they are "
        "passed to the loss module.",
    )
    parser.add_argument(
        "--sub_traj_len",
        "--sub-traj-len",
        type=int,
        default=-1,
        help="length of the trajectories that sub-samples must have in online settings.",
    )
    return parser
