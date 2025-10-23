# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tensordict.nn import TensorDictModuleBase

from torchrl.collectors import DataCollectorBase
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.objectives.value.advantages import GAE
from torchrl.trainers.algorithms.configs.common import ConfigBase
from torchrl.trainers.algorithms.ppo import PPOTrainer
from torchrl.trainers.algorithms.sac import SACTrainer


@dataclass
class TrainerConfig(ConfigBase):
    """Base configuration class for trainers."""

    def __post_init__(self) -> None:
        """Post-initialization hook for trainer configurations."""


@dataclass
class SACTrainerConfig(TrainerConfig):
    """Configuration class for SAC (Soft Actor Critic) trainer.

    This class defines the configuration parameters for creating a SAC trainer,
    including both required and optional fields with sensible defaults.
    """

    collector: Any
    total_frames: int
    optim_steps_per_batch: int | None
    loss_module: Any
    optimizer: Any
    logger: Any
    save_trainer_file: Any
    replay_buffer: Any
    frame_skip: int = 1
    clip_grad_norm: bool = True
    clip_norm: float | None = None
    progress_bar: bool = True
    seed: int | None = None
    save_trainer_interval: int = 10000
    log_interval: int = 10000
    create_env_fn: Any = None
    actor_network: Any = None
    critic_network: Any = None
    target_net_updater: Any = None
    async_collection: bool = False
    log_timings: bool = False

    _target_: str = "torchrl.trainers.algorithms.configs.trainers._make_sac_trainer"

    def __post_init__(self) -> None:
        """Post-initialization hook for SAC trainer configuration."""
        super().__post_init__()


def _make_sac_trainer(*args, **kwargs) -> SACTrainer:
    from torchrl.trainers.trainers import Logger

    collector = kwargs.pop("collector")
    total_frames = kwargs.pop("total_frames")
    if total_frames is None:
        total_frames = collector.total_frames
    frame_skip = kwargs.pop("frame_skip", 1)
    optim_steps_per_batch = kwargs.pop("optim_steps_per_batch", 1)
    loss_module = kwargs.pop("loss_module")
    optimizer = kwargs.pop("optimizer")
    logger = kwargs.pop("logger")
    clip_grad_norm = kwargs.pop("clip_grad_norm", True)
    clip_norm = kwargs.pop("clip_norm")
    progress_bar = kwargs.pop("progress_bar", True)
    replay_buffer = kwargs.pop("replay_buffer")
    save_trainer_interval = kwargs.pop("save_trainer_interval", 10000)
    log_interval = kwargs.pop("log_interval", 10000)
    save_trainer_file = kwargs.pop("save_trainer_file")
    seed = kwargs.pop("seed")
    actor_network = kwargs.pop("actor_network")
    critic_network = kwargs.pop("critic_network")
    kwargs.pop("create_env_fn")
    target_net_updater = kwargs.pop("target_net_updater")
    async_collection = kwargs.pop("async_collection", False)
    log_timings = kwargs.pop("log_timings", False)

    # Instantiate networks first
    if actor_network is not None:
        actor_network = actor_network()
    if critic_network is not None:
        critic_network = critic_network()

    if not isinstance(collector, DataCollectorBase):
        # then it's a partial config
        if not async_collection:
            collector = collector()
        elif replay_buffer is not None:
            collector = collector(replay_buffer=replay_buffer)
    elif getattr(collector, "replay_buffer", None) is None:
        if async_collection and (
            collector.replay_buffer is None or replay_buffer is None
        ):
            raise ValueError(
                "replay_buffer must be provided when async_collection is True"
            )

    if not isinstance(loss_module, LossModule):
        # then it's a partial config
        loss_module = loss_module(
            actor_network=actor_network, critic_network=critic_network
        )
    if not isinstance(target_net_updater, TargetNetUpdater):
        # target_net_updater must be a partial taking the loss as input
        target_net_updater = target_net_updater(loss_module)
    if not isinstance(optimizer, torch.optim.Optimizer):
        # then it's a partial config
        optimizer = optimizer(params=loss_module.parameters())

    # Quick instance checks
    if not isinstance(collector, DataCollectorBase):
        raise ValueError(
            f"collector must be a DataCollectorBase, got {type(collector)}"
        )
    if not isinstance(loss_module, LossModule):
        raise ValueError(f"loss_module must be a LossModule, got {type(loss_module)}")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError(
            f"optimizer must be a torch.optim.Optimizer, got {type(optimizer)}"
        )
    if not isinstance(logger, Logger) and logger is not None:
        raise ValueError(f"logger must be a Logger, got {type(logger)}")

    return SACTrainer(
        collector=collector,
        total_frames=total_frames,
        frame_skip=frame_skip,
        optim_steps_per_batch=optim_steps_per_batch,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        clip_grad_norm=clip_grad_norm,
        clip_norm=clip_norm,
        progress_bar=progress_bar,
        seed=seed,
        save_trainer_interval=save_trainer_interval,
        log_interval=log_interval,
        save_trainer_file=save_trainer_file,
        replay_buffer=replay_buffer,
        target_net_updater=target_net_updater,
        async_collection=async_collection,
        log_timings=log_timings,
    )


@dataclass
class PPOTrainerConfig(TrainerConfig):
    """Configuration class for PPO (Proximal Policy Optimization) trainer.

    This class defines the configuration parameters for creating a PPO trainer,
    including both required and optional fields with sensible defaults.

    Args:
        collector: The data collector for gathering training data.
        total_frames: Total number of frames to train for.
        optim_steps_per_batch: Number of optimization steps per batch.
        loss_module: The loss module for computing policy and value losses.
        optimizer: The optimizer for training.
        logger: Logger for tracking training metrics.
        save_trainer_file: File path for saving trainer state.
        replay_buffer: Replay buffer for storing data.
        frame_skip: Frame skip value for the environment. Default: 1.
        clip_grad_norm: Whether to clip gradient norms. Default: True.
        clip_norm: Maximum gradient norm value.
        progress_bar: Whether to show a progress bar. Default: True.
        seed: Random seed for reproducibility.
        save_trainer_interval: Interval for saving trainer state. Default: 10000.
        log_interval: Interval for logging metrics. Default: 10000.
        create_env_fn: Environment creation function.
        actor_network: Actor network configuration.
        critic_network: Critic network configuration.
        num_epochs: Number of epochs per batch. Default: 4.
        async_collection: Whether to use async collection. Default: False.
        add_gae: Whether to add GAE computation. Default: True.
        gae: Custom GAE module configuration.
        weight_update_map: Mapping from collector destination paths to trainer source paths.
            Required if collector has weight_sync_schemes configured.
            Example: {"policy": "loss_module.actor_network",
                     "replay_buffer.transforms[0]": "loss_module.critic_network"}
        log_timings: Whether to automatically log timing information for all hooks.
            If True, timing metrics will be logged to the logger (e.g., wandb, tensorboard)
            with prefix "time/" (e.g., "time/hook/UpdateWeights"). Default: False.
    """

    collector: Any
    total_frames: int
    optim_steps_per_batch: int | None
    loss_module: Any
    optimizer: Any
    logger: Any
    save_trainer_file: Any
    replay_buffer: Any
    frame_skip: int = 1
    clip_grad_norm: bool = True
    clip_norm: float | None = None
    progress_bar: bool = True
    seed: int | None = None
    save_trainer_interval: int = 10000
    log_interval: int = 10000
    create_env_fn: Any = None
    actor_network: Any = None
    critic_network: Any = None
    num_epochs: int = 4
    async_collection: bool = False
    add_gae: bool = True
    gae: Any = None
    weight_update_map: dict[str, str] | None = None
    log_timings: bool = False

    _target_: str = "torchrl.trainers.algorithms.configs.trainers._make_ppo_trainer"

    def __post_init__(self) -> None:
        """Post-initialization hook for PPO trainer configuration."""
        super().__post_init__()


def _make_ppo_trainer(*args, **kwargs) -> PPOTrainer:
    from torchrl.trainers.trainers import Logger

    collector = kwargs.pop("collector")
    total_frames = kwargs.pop("total_frames")
    if total_frames is None:
        total_frames = collector.total_frames
    frame_skip = kwargs.pop("frame_skip", 1)
    optim_steps_per_batch = kwargs.pop("optim_steps_per_batch", 1)
    loss_module = kwargs.pop("loss_module")
    optimizer = kwargs.pop("optimizer")
    logger = kwargs.pop("logger")
    clip_grad_norm = kwargs.pop("clip_grad_norm", True)
    clip_norm = kwargs.pop("clip_norm")
    progress_bar = kwargs.pop("progress_bar", True)
    replay_buffer = kwargs.pop("replay_buffer")
    save_trainer_interval = kwargs.pop("save_trainer_interval", 10000)
    log_interval = kwargs.pop("log_interval", 10000)
    save_trainer_file = kwargs.pop("save_trainer_file")
    seed = kwargs.pop("seed")
    actor_network = kwargs.pop("actor_network")
    critic_network = kwargs.pop("critic_network")
    add_gae = kwargs.pop("add_gae", True)
    gae = kwargs.pop("gae")
    create_env_fn = kwargs.pop("create_env_fn")
    weight_update_map = kwargs.pop("weight_update_map", None)
    log_timings = kwargs.pop("log_timings", False)

    if create_env_fn is not None:
        # could be referenced somewhere else, no need to raise an error
        pass
    num_epochs = kwargs.pop("num_epochs", 4)
    async_collection = kwargs.pop("async_collection", False)

    # Instantiate networks first
    if actor_network is not None:
        actor_network = actor_network()
    if critic_network is not None:
        critic_network = critic_network()
    else:
        critic_network = loss_module.critic_network

    # Ensure GAE in replay buffer uses the same value network instance as loss module
    # This fixes the issue where Hydra instantiates separate instances of value_model
    if (
        replay_buffer is not None
        and hasattr(replay_buffer, "_transform")
        and len(replay_buffer._transform) > 1
        and hasattr(replay_buffer._transform[1], "module")
        and hasattr(replay_buffer._transform[1].module, "value_network")
    ):
        replay_buffer._transform[1].module.value_network = critic_network

    if not isinstance(collector, DataCollectorBase):
        # then it's a partial config
        if not async_collection:
            collector = collector()
        else:
            collector = collector(replay_buffer=replay_buffer)
    elif async_collection and getattr(collector, "replay_buffer", None) is None:
        raise RuntimeError(
            "replay_buffer must be provided when async_collection is True"
        )
    if not isinstance(loss_module, LossModule):
        # then it's a partial config
        loss_module = loss_module(
            actor_network=actor_network, critic_network=critic_network
        )
    if not isinstance(optimizer, torch.optim.Optimizer):
        # then it's a partial config
        optimizer = optimizer(params=loss_module.parameters())

    # Quick instance checks
    if not isinstance(collector, DataCollectorBase):
        raise ValueError(
            f"collector must be a DataCollectorBase, got {type(collector)}"
        )
    if not isinstance(loss_module, LossModule):
        raise ValueError(f"loss_module must be a LossModule, got {type(loss_module)}")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError(
            f"optimizer must be a torch.optim.Optimizer, got {type(optimizer)}"
        )
    if not isinstance(logger, Logger) and logger is not None:
        raise ValueError(f"logger must be a Logger, got {type(logger)}")
    # instantiate gae if it is a partial config
    if not isinstance(gae, (GAE, TensorDictModuleBase)) and gae is not None:
        gae = gae()

    return PPOTrainer(
        collector=collector,
        total_frames=total_frames,
        frame_skip=frame_skip,
        optim_steps_per_batch=optim_steps_per_batch,
        loss_module=loss_module,
        optimizer=optimizer,
        logger=logger,
        clip_grad_norm=clip_grad_norm,
        clip_norm=clip_norm,
        progress_bar=progress_bar,
        seed=seed,
        save_trainer_interval=save_trainer_interval,
        log_interval=log_interval,
        save_trainer_file=save_trainer_file,
        replay_buffer=replay_buffer,
        num_epochs=num_epochs,
        async_collection=async_collection,
        add_gae=add_gae,
        gae=gae,
        weight_update_map=weight_update_map,
        log_timings=log_timings,
    )
