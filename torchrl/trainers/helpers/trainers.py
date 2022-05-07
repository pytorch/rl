# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from argparse import ArgumentParser, Namespace
from dataclasses import dataclass
from typing import List, Optional, Union
from warnings import warn

from torch import optim
from torchrl.collectors.collectors import _DataCollector
from torchrl.data import ReplayBuffer
from torchrl.envs.common import _EnvClass
from torchrl.modules import TDModule, TDModuleWrapper, reset_noise
from torchrl.objectives.costs.common import _LossModule
from torchrl.objectives.costs.utils import _TargetNetUpdate
from torchrl.trainers.trainers import (BatchSubSampler, CountFramesLog,
                                       LogReward, Recorder,
                                       ReplayBufferTrainer, RewardNormalizer,
                                       SelectKeys, Trainer, UpdateWeights,
                                       mask_batch)

OPTIMIZERS = {
    "adam": optim.Adam,
    "sgd": optim.SGD,
    "adamax": optim.Adamax,
}

__all__ = [
    "make_trainer",
    # "parser_trainer_args",
]


def make_trainer(
    collector: _DataCollector,
    loss_module: _LossModule,
    recorder: Optional[_EnvClass] = None,
    target_net_updater: Optional[_TargetNetUpdate] = None,
    policy_exploration: Optional[Union[TDModuleWrapper, TDModule]] = None,
    replay_buffer: Optional[ReplayBuffer] = None,
    writer: Optional["SummaryWriter"] = None,
    args: Optional[Namespace] = None,
) -> Trainer:
    """Creates a Trainer instance given its constituents.

    Args:
        collector (_DataCollector): A data collector to be used to collect data.
        loss_module (_LossModule): A TorchRL loss module
        recorder (_EnvClass, optional): a recorder environment. If None, the trainer will train the policy without
            testing it.
        target_net_updater (_TargetNetUpdate, optional): A target network update object.
        policy_exploration (TDModule or TDModuleWrapper, optional): a policy to be used for recording and exploration
            updates (should be synced with the learnt policy).
        replay_buffer (ReplayBuffer, optional): a replay buffer to be used to collect data.
        writer (SummaryWriter, optional): a tensorboard SummaryWriter to be used for logging.
        args (argparse.Namespace, optional): a Namespace containing the arguments of the script. If None, the default
            arguments are used.

    Returns:
        A trainer built with the input objects. The optimizer is built by this helper function using the args provided.

    Examples:
        >>> import torch
        >>> import tempfile
        >>> from torch.utils.tensorboard import SummaryWriter
        >>> from torchrl.trainers import Trainer
        >>> from torchrl.envs import EnvCreator
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
        >>> trainer = make_trainer(collector, loss_module, recorder, target_net_updater, policy_exploration,
        ...    replay_buffer, writer)
        >>> print(trainer)

    """
    if args is None:
        warn(
            "Getting default args for the trainer. "
            "This should be only used for debugging."
        )
        parser = parser_trainer_args(argparse.ArgumentParser())
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

    trainer = Trainer(
        collector=collector,
        frame_skip=args.frame_skip,
        total_frames=args.total_frames * args.frame_skip,
        loss_module=loss_module,
        optimizer=optimizer,
        writer=writer,
        optim_steps_per_batch=args.optim_steps_per_batch,
        clip_grad_norm=args.clip_grad_norm,
        clip_norm=args.clip_norm,
    )

    if hasattr(args, "noisy") and args.noisy:
        trainer.register_op("pre_optim_steps", lambda: loss_module.apply(reset_noise))

    if args.selected_keys:
        trainer.register_op("batch_process", SelectKeys(args.selected_keys))

    if replay_buffer is not None:
        # replay buffer is used 2 or 3 times: to register data, to sample
        # data and to update priorities
        rb_trainer = ReplayBufferTrainer(replay_buffer, args.batch_size)
        trainer.register_op("batch_process", rb_trainer.extend)
        trainer.register_op("process_optim_batch", rb_trainer.sample)
        trainer.register_op("post_loss", rb_trainer.update_priority)
    else:
        trainer.register_op("batch_process", mask_batch)
        trainer.register_op(
            "process_optim_batch",
            BatchSubSampler(batch_size=args.batch_size, sub_traj_len=args.sub_traj_len),
        )

    if optim_scheduler is not None:
        trainer.register_op("post_optim", optim_scheduler.step)

    if target_net_updater is not None:
        trainer.register_op("post_optim", target_net_updater.step)

    if args.normalize_rewards_online:
        # if used the running statistics of the rewards are computed and the
        # rewards used for training will be normalized based on these.
        reward_normalizer = RewardNormalizer()
        trainer.register_op("batch_process", reward_normalizer.update_reward_stats)
        trainer.register_op("process_optim_batch", reward_normalizer.normalize_reward)

    if policy_exploration is not None and hasattr(policy_exploration, "step"):
        trainer.register_op("post_steps", policy_exploration.step)

    if recorder is not None:
        trainer.register_op(
            "post_steps_log",
            Recorder(
                record_frames=args.record_frames,
                frame_skip=args.frame_skip,
                policy_exploration=policy_exploration,
                recorder=recorder,
                record_interval=args.record_interval,
            ),
        )
    trainer.register_op("post_steps", UpdateWeights(collector, 1))

    trainer.register_op("pre_steps_log", LogReward())
    trainer.register_op("pre_steps_log", CountFramesLog(frame_skip=args.frame_skip))

    return trainer

@dataclass
class TrainerConfig:
    
    optim_steps_per_batch: int = 500
    # Number of optimization steps in between two collection of data. See frames_per_batch below.

    optimizer: str = "adam"
    # Optimizer to be used.

    selected_keys: Optional[List] = None
    # a list of strings that indicate the data that should be kept from the data collector. Since storing and
    # retrieving information from the replay buffer does not come for free, limiting the amount of data
    # passed to it can improve the algorithm performance.
    # Default is None, i.e. all keys are kept.


    batch_size: int = 256
    # batch size of the TensorDict retrieved from the replay buffer.

    log_interval: int = 10000
    # logging interval, in terms of optimization steps.

    lr: float = 3e-4

    # Learning rate used for the optimizer.

    weight_decay: float = 2e-5

    # Weight-decay to be used with the optimizer.

    clip_norm: float = 1.0
    # value at which the total gradient norm should be clipped.

    clip_grad_norm: bool = False

    # if called, the gradient will be clipped based on its L2 norm. Otherwise, single gradient values will be clipped to the desired threshold.
    
    normalize_rewards_online: bool = False
    # Computes the running statistics of the rewards and normalizes them before they are passed to the loss module.

    sub_traj_len: int = -1
    # length of the trajectories that sub-samples must have in online settings.
