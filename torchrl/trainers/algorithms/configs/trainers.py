# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from tensordict.nn import TensorDictModuleBase

from torchrl.collectors import BaseCollector
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.objectives.value.advantages import GAE
from torchrl.trainers import TrainerHookBase
from torchrl.trainers.algorithms.configs.common import _normalize_hydra_key, ConfigBase
from torchrl.trainers.algorithms.cql import CQLTrainer
from torchrl.trainers.algorithms.ddpg import DDPGTrainer
from torchrl.trainers.algorithms.dqn import DQNTrainer
from torchrl.trainers.algorithms.iql import IQLTrainer
from torchrl.trainers.algorithms.ppo import PPOTrainer
from torchrl.trainers.algorithms.sac import SACTrainer
from torchrl.trainers.algorithms.td3 import TD3Trainer


@dataclass
class TrainerConfig(ConfigBase):
    """Base configuration class for trainers."""

    def __post_init__(self) -> None:
        """Post-initialization hook for trainer configurations."""


def _register_trainer_hooks(trainer: Any, hooks: list[Any] | None) -> None:
    if hooks is None:
        return
    for index, hook in enumerate(hooks):
        if not isinstance(hook, TrainerHookBase):
            raise TypeError(
                "trainer hooks must be TrainerHookBase instances with a "
                f"register(trainer) method, got {type(hook)} at index {index}."
            )
        hook.register(trainer)


@dataclass
class SACTrainerConfig(TrainerConfig):
    """Hydra configuration for :class:`~torchrl.trainers.algorithms.SACTrainer`.

    Every kwarg accepted by ``SACTrainer.__init__`` is exposed as a field here.
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
    auto_log_optim_steps: bool = True
    batch_size: int | None = None
    enable_logging: bool = True
    log_rewards: bool = True
    log_actions: bool = True
    log_observations: bool = False
    done_key: Any = "done"
    terminated_key: Any = "terminated"
    reward_key: Any = "reward"
    episode_reward_key: Any = "reward_sum"
    action_key: Any = "action"
    observation_key: Any = "observation"
    hooks: list[Any] | None = None

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
    auto_log_optim_steps = kwargs.pop("auto_log_optim_steps", True)
    batch_size = kwargs.pop("batch_size", None)
    enable_logging = kwargs.pop("enable_logging", True)
    log_rewards = kwargs.pop("log_rewards", True)
    log_actions = kwargs.pop("log_actions", True)
    log_observations = kwargs.pop("log_observations", False)
    done_key = _normalize_hydra_key(kwargs.pop("done_key", "done"))
    terminated_key = _normalize_hydra_key(kwargs.pop("terminated_key", "terminated"))
    reward_key = _normalize_hydra_key(kwargs.pop("reward_key", "reward"))
    episode_reward_key = _normalize_hydra_key(
        kwargs.pop("episode_reward_key", "reward_sum")
    )
    action_key = _normalize_hydra_key(kwargs.pop("action_key", "action"))
    observation_key = _normalize_hydra_key(kwargs.pop("observation_key", "observation"))
    hooks = kwargs.pop("hooks", None)

    # Instantiate networks first
    if actor_network is not None and not isinstance(actor_network, torch.nn.Module):
        actor_network = actor_network()
    if critic_network is not None and not isinstance(critic_network, torch.nn.Module):
        critic_network = critic_network()

    if not isinstance(collector, BaseCollector):
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
    if not isinstance(collector, BaseCollector):
        raise ValueError(f"collector must be a BaseCollector, got {type(collector)}")
    if not isinstance(loss_module, LossModule):
        raise ValueError(f"loss_module must be a LossModule, got {type(loss_module)}")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError(
            f"optimizer must be a torch.optim.Optimizer, got {type(optimizer)}"
        )
    if not isinstance(logger, Logger) and logger is not None:
        raise ValueError(f"logger must be a Logger, got {type(logger)}")

    trainer = SACTrainer(
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
        batch_size=batch_size,
        enable_logging=enable_logging,
        log_rewards=log_rewards,
        log_actions=log_actions,
        log_observations=log_observations,
        target_net_updater=target_net_updater,
        async_collection=async_collection,
        log_timings=log_timings,
        auto_log_optim_steps=auto_log_optim_steps,
        done_key=done_key,
        terminated_key=terminated_key,
        reward_key=reward_key,
        episode_reward_key=episode_reward_key,
        action_key=action_key,
        observation_key=observation_key,
    )
    _register_trainer_hooks(trainer, hooks)
    return trainer


@dataclass
class PPOTrainerConfig(TrainerConfig):
    """Hydra configuration for :class:`~torchrl.trainers.algorithms.PPOTrainer`.

    Every kwarg accepted by ``PPOTrainer.__init__`` is exposed as a field here.

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
            Example: ``{"policy": "loss_module.actor_network", "replay_buffer.transforms[0]": "loss_module.critic_network"}``.
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
    auto_log_optim_steps: bool = True
    batch_size: int | None = None
    gamma: float = 0.99
    lmbda: float = 0.95
    enable_logging: bool = True
    log_rewards: bool = True
    log_actions: bool = True
    log_observations: bool = False
    done_key: Any = "done"
    terminated_key: Any = "terminated"
    reward_key: Any = "reward"
    episode_reward_key: Any = "reward"
    action_key: Any = "action"
    observation_key: Any = "observation"
    hooks: list[Any] | None = None

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
    auto_log_optim_steps = kwargs.pop("auto_log_optim_steps", True)
    batch_size = kwargs.pop("batch_size", None)
    gamma = kwargs.pop("gamma", 0.99)
    lmbda = kwargs.pop("lmbda", 0.95)
    enable_logging = kwargs.pop("enable_logging", True)
    log_rewards = kwargs.pop("log_rewards", True)
    log_actions = kwargs.pop("log_actions", True)
    log_observations = kwargs.pop("log_observations", False)
    done_key = _normalize_hydra_key(kwargs.pop("done_key", "done"))
    terminated_key = _normalize_hydra_key(kwargs.pop("terminated_key", "terminated"))
    reward_key = _normalize_hydra_key(kwargs.pop("reward_key", "reward"))
    episode_reward_key = _normalize_hydra_key(
        kwargs.pop("episode_reward_key", "reward")
    )
    action_key = _normalize_hydra_key(kwargs.pop("action_key", "action"))
    observation_key = _normalize_hydra_key(kwargs.pop("observation_key", "observation"))
    hooks = kwargs.pop("hooks", None)

    if create_env_fn is not None:
        # could be referenced somewhere else, no need to raise an error
        pass
    num_epochs = kwargs.pop("num_epochs", 4)
    async_collection = kwargs.pop("async_collection", False)

    # Instantiate networks first
    if actor_network is not None and not isinstance(actor_network, torch.nn.Module):
        actor_network = actor_network()
    if critic_network is not None and not isinstance(critic_network, torch.nn.Module):
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

    if not isinstance(collector, BaseCollector):
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
    if not isinstance(collector, BaseCollector):
        raise ValueError(f"collector must be a BaseCollector, got {type(collector)}")
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

    trainer = PPOTrainer(
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
        batch_size=batch_size,
        gamma=gamma,
        lmbda=lmbda,
        enable_logging=enable_logging,
        log_rewards=log_rewards,
        log_actions=log_actions,
        log_observations=log_observations,
        num_epochs=num_epochs,
        async_collection=async_collection,
        add_gae=add_gae,
        gae=gae,
        weight_update_map=weight_update_map,
        log_timings=log_timings,
        auto_log_optim_steps=auto_log_optim_steps,
        done_key=done_key,
        terminated_key=terminated_key,
        reward_key=reward_key,
        episode_reward_key=episode_reward_key,
        action_key=action_key,
        observation_key=observation_key,
    )
    _register_trainer_hooks(trainer, hooks)
    return trainer


@dataclass
class DQNTrainerConfig(TrainerConfig):
    """Hydra configuration for :class:`~torchrl.trainers.algorithms.DQNTrainer`.

    Every kwarg accepted by ``DQNTrainer.__init__`` is exposed as a field here.
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
    value_network: Any = None
    target_net_updater: Any = None
    eps_init: float = 1.0
    eps_end: float = 0.05
    annealing_num_steps: int = 250_000
    async_collection: bool = False
    log_timings: bool = False
    auto_log_optim_steps: bool = True
    enable_logging: bool = True
    log_rewards: bool = True
    log_observations: bool = False
    hooks: list[Any] | None = None

    _target_: str = "torchrl.trainers.algorithms.configs.trainers._make_dqn_trainer"

    def __post_init__(self) -> None:
        super().__post_init__()


def _make_dqn_trainer(*args, **kwargs) -> DQNTrainer:
    from tensordict.nn import TensorDictSequential

    from torchrl.modules import EGreedyModule
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
    value_network = kwargs.pop("value_network")
    kwargs.pop("create_env_fn", None)
    target_net_updater = kwargs.pop("target_net_updater")
    eps_init = kwargs.pop("eps_init", 1.0)
    eps_end = kwargs.pop("eps_end", 0.05)
    annealing_num_steps = kwargs.pop("annealing_num_steps", 250_000)
    async_collection = kwargs.pop("async_collection", False)
    log_timings = kwargs.pop("log_timings", False)
    auto_log_optim_steps = kwargs.pop("auto_log_optim_steps", True)
    enable_logging = kwargs.pop("enable_logging", True)
    log_rewards = kwargs.pop("log_rewards", True)
    log_observations = kwargs.pop("log_observations", False)
    hooks = kwargs.pop("hooks", None)

    if value_network is not None and not isinstance(value_network, torch.nn.Module):
        value_network = value_network()

    from torchrl.data import Composite, OneHot

    action_spec = value_network.spec.get("action", default=None)
    if action_spec is None:
        n_actions = value_network.module[0].out_features
        action_spec = OneHot(n=n_actions)
    spec = Composite(action=action_spec)

    greedy_module = EGreedyModule(
        annealing_num_steps=annealing_num_steps,
        eps_init=eps_init,
        eps_end=eps_end,
        spec=spec,
    )
    exploration_policy = TensorDictSequential(value_network, greedy_module)

    if not isinstance(collector, BaseCollector):
        collector_kwargs = {"policy": exploration_policy}
        if not async_collection:
            collector = collector(**collector_kwargs)
        elif replay_buffer is not None:
            collector = collector(replay_buffer=replay_buffer, **collector_kwargs)

    if not isinstance(loss_module, LossModule):
        loss_module = loss_module(value_network=value_network)
    if not isinstance(target_net_updater, TargetNetUpdater):
        target_net_updater = target_net_updater(loss_module)
    if not isinstance(optimizer, torch.optim.Optimizer):
        optimizer = optimizer(params=loss_module.parameters())

    if not isinstance(collector, BaseCollector):
        raise ValueError(f"collector must be a BaseCollector, got {type(collector)}")
    if not isinstance(loss_module, LossModule):
        raise ValueError(f"loss_module must be a LossModule, got {type(loss_module)}")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError(
            f"optimizer must be a torch.optim.Optimizer, got {type(optimizer)}"
        )
    if not isinstance(logger, Logger) and logger is not None:
        raise ValueError(f"logger must be a Logger, got {type(logger)}")

    trainer = DQNTrainer(
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
        enable_logging=enable_logging,
        log_rewards=log_rewards,
        log_observations=log_observations,
        target_net_updater=target_net_updater,
        greedy_module=greedy_module,
        async_collection=async_collection,
        log_timings=log_timings,
        auto_log_optim_steps=auto_log_optim_steps,
    )
    _register_trainer_hooks(trainer, hooks)
    return trainer


@dataclass
class DDPGTrainerConfig(TrainerConfig):
    """Hydra configuration for :class:`~torchrl.trainers.algorithms.DDPGTrainer`.

    Every kwarg accepted by ``DDPGTrainer.__init__`` is exposed as a field here.
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
    auto_log_optim_steps: bool = True
    enable_logging: bool = True
    log_rewards: bool = True
    log_actions: bool = True
    log_observations: bool = False
    done_key: Any = "done"
    terminated_key: Any = "terminated"
    reward_key: Any = "reward"
    episode_reward_key: Any = "reward_sum"
    action_key: Any = "action"
    observation_key: Any = "observation"
    hooks: list[Any] | None = None

    _target_: str = "torchrl.trainers.algorithms.configs.trainers._make_ddpg_trainer"

    def __post_init__(self) -> None:
        super().__post_init__()


def _make_ddpg_trainer(*args, **kwargs) -> DDPGTrainer:
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
    kwargs.pop("create_env_fn", None)
    target_net_updater = kwargs.pop("target_net_updater")
    async_collection = kwargs.pop("async_collection", False)
    log_timings = kwargs.pop("log_timings", False)
    auto_log_optim_steps = kwargs.pop("auto_log_optim_steps", True)
    enable_logging = kwargs.pop("enable_logging", True)
    log_rewards = kwargs.pop("log_rewards", True)
    log_actions = kwargs.pop("log_actions", True)
    log_observations = kwargs.pop("log_observations", False)
    done_key = _normalize_hydra_key(kwargs.pop("done_key", "done"))
    terminated_key = _normalize_hydra_key(kwargs.pop("terminated_key", "terminated"))
    reward_key = _normalize_hydra_key(kwargs.pop("reward_key", "reward"))
    episode_reward_key = _normalize_hydra_key(
        kwargs.pop("episode_reward_key", "reward_sum")
    )
    action_key = _normalize_hydra_key(kwargs.pop("action_key", "action"))
    observation_key = _normalize_hydra_key(kwargs.pop("observation_key", "observation"))
    hooks = kwargs.pop("hooks", None)

    if actor_network is not None and not isinstance(actor_network, torch.nn.Module):
        actor_network = actor_network()
    if critic_network is not None and not isinstance(critic_network, torch.nn.Module):
        critic_network = critic_network()
    if not isinstance(collector, BaseCollector):
        if not async_collection:
            collector = collector()
        elif replay_buffer is not None:
            collector = collector(replay_buffer=replay_buffer)

    if not isinstance(loss_module, LossModule):
        loss_module = loss_module(
            actor_network=actor_network, value_network=critic_network
        )
    if not isinstance(target_net_updater, TargetNetUpdater):
        target_net_updater = target_net_updater(loss_module)
    if not isinstance(optimizer, torch.optim.Optimizer):
        optimizer = optimizer(params=loss_module.parameters())

    if not isinstance(collector, BaseCollector):
        raise ValueError(f"collector must be a BaseCollector, got {type(collector)}")
    if not isinstance(loss_module, LossModule):
        raise ValueError(f"loss_module must be a LossModule, got {type(loss_module)}")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError(
            f"optimizer must be a torch.optim.Optimizer, got {type(optimizer)}"
        )
    if not isinstance(logger, Logger) and logger is not None:
        raise ValueError(f"logger must be a Logger, got {type(logger)}")

    trainer = DDPGTrainer(
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
        enable_logging=enable_logging,
        log_rewards=log_rewards,
        log_actions=log_actions,
        log_observations=log_observations,
        target_net_updater=target_net_updater,
        async_collection=async_collection,
        log_timings=log_timings,
        auto_log_optim_steps=auto_log_optim_steps,
        done_key=done_key,
        terminated_key=terminated_key,
        reward_key=reward_key,
        episode_reward_key=episode_reward_key,
        action_key=action_key,
        observation_key=observation_key,
    )
    _register_trainer_hooks(trainer, hooks)
    return trainer


@dataclass
class IQLTrainerConfig(TrainerConfig):
    """Hydra configuration for :class:`~torchrl.trainers.algorithms.IQLTrainer`.

    Every kwarg accepted by ``IQLTrainer.__init__`` is exposed as a field here.
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
    qvalue_network: Any = None
    value_network: Any = None
    target_net_updater: Any = None
    async_collection: bool = False
    log_timings: bool = False
    auto_log_optim_steps: bool = True
    enable_logging: bool = True
    log_rewards: bool = True
    log_actions: bool = True
    log_observations: bool = False
    hooks: list[Any] | None = None

    _target_: str = "torchrl.trainers.algorithms.configs.trainers._make_iql_trainer"

    def __post_init__(self) -> None:
        super().__post_init__()


def _make_iql_trainer(*args, **kwargs) -> IQLTrainer:
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
    qvalue_network = kwargs.pop("qvalue_network")
    value_network = kwargs.pop("value_network")
    kwargs.pop("create_env_fn", None)
    target_net_updater = kwargs.pop("target_net_updater")
    async_collection = kwargs.pop("async_collection", False)
    log_timings = kwargs.pop("log_timings", False)
    auto_log_optim_steps = kwargs.pop("auto_log_optim_steps", True)
    enable_logging = kwargs.pop("enable_logging", True)
    log_rewards = kwargs.pop("log_rewards", True)
    log_actions = kwargs.pop("log_actions", True)
    log_observations = kwargs.pop("log_observations", False)
    hooks = kwargs.pop("hooks", None)

    if actor_network is not None:
        actor_network = actor_network()
    if qvalue_network is not None:
        qvalue_network = qvalue_network()
    if value_network is not None:
        value_network = value_network()

    if not isinstance(collector, BaseCollector):
        if not async_collection:
            collector = collector()
        elif replay_buffer is not None:
            collector = collector(replay_buffer=replay_buffer)

    if not isinstance(loss_module, LossModule):
        loss_module = loss_module(
            actor_network=actor_network,
            qvalue_network=qvalue_network,
            value_network=value_network,
        )
    if not isinstance(target_net_updater, TargetNetUpdater):
        target_net_updater = target_net_updater(loss_module)
    if not isinstance(optimizer, torch.optim.Optimizer):
        optimizer = optimizer(params=loss_module.parameters())

    if not isinstance(collector, BaseCollector):
        raise ValueError(f"collector must be a BaseCollector, got {type(collector)}")
    if not isinstance(loss_module, LossModule):
        raise ValueError(f"loss_module must be a LossModule, got {type(loss_module)}")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError(
            f"optimizer must be a torch.optim.Optimizer, got {type(optimizer)}"
        )
    if not isinstance(logger, Logger) and logger is not None:
        raise ValueError(f"logger must be a Logger, got {type(logger)}")

    trainer = IQLTrainer(
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
        enable_logging=enable_logging,
        log_rewards=log_rewards,
        log_actions=log_actions,
        log_observations=log_observations,
        target_net_updater=target_net_updater,
        async_collection=async_collection,
        log_timings=log_timings,
        auto_log_optim_steps=auto_log_optim_steps,
    )
    _register_trainer_hooks(trainer, hooks)
    return trainer


@dataclass
class CQLTrainerConfig(TrainerConfig):
    """Hydra configuration for :class:`~torchrl.trainers.algorithms.CQLTrainer`.

    Every kwarg accepted by ``CQLTrainer.__init__`` is exposed as a field here.
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
    qvalue_network: Any = None
    target_net_updater: Any = None
    async_collection: bool = False
    log_timings: bool = False
    auto_log_optim_steps: bool = True
    enable_logging: bool = True
    log_rewards: bool = True
    log_actions: bool = True
    log_observations: bool = False
    hooks: list[Any] | None = None

    _target_: str = "torchrl.trainers.algorithms.configs.trainers._make_cql_trainer"

    def __post_init__(self) -> None:
        super().__post_init__()


def _make_cql_trainer(*args, **kwargs) -> CQLTrainer:
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
    qvalue_network = kwargs.pop("qvalue_network")
    kwargs.pop("create_env_fn", None)
    target_net_updater = kwargs.pop("target_net_updater")
    async_collection = kwargs.pop("async_collection", False)
    log_timings = kwargs.pop("log_timings", False)
    auto_log_optim_steps = kwargs.pop("auto_log_optim_steps", True)
    enable_logging = kwargs.pop("enable_logging", True)
    log_rewards = kwargs.pop("log_rewards", True)
    log_actions = kwargs.pop("log_actions", True)
    log_observations = kwargs.pop("log_observations", False)
    hooks = kwargs.pop("hooks", None)

    if actor_network is not None:
        actor_network = actor_network()
    if qvalue_network is not None:
        qvalue_network = qvalue_network()

    if not isinstance(collector, BaseCollector):
        if not async_collection:
            collector = collector()
        elif replay_buffer is not None:
            collector = collector(replay_buffer=replay_buffer)

    if not isinstance(loss_module, LossModule):
        loss_module = loss_module(
            actor_network=actor_network, qvalue_network=qvalue_network
        )
    if not isinstance(target_net_updater, TargetNetUpdater):
        target_net_updater = target_net_updater(loss_module)
    if not isinstance(optimizer, torch.optim.Optimizer):
        optimizer = optimizer(params=loss_module.parameters())

    if not isinstance(collector, BaseCollector):
        raise ValueError(f"collector must be a BaseCollector, got {type(collector)}")
    if not isinstance(loss_module, LossModule):
        raise ValueError(f"loss_module must be a LossModule, got {type(loss_module)}")
    if not isinstance(optimizer, torch.optim.Optimizer):
        raise ValueError(
            f"optimizer must be a torch.optim.Optimizer, got {type(optimizer)}"
        )
    if not isinstance(logger, Logger) and logger is not None:
        raise ValueError(f"logger must be a Logger, got {type(logger)}")

    trainer = CQLTrainer(
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
        enable_logging=enable_logging,
        log_rewards=log_rewards,
        log_actions=log_actions,
        log_observations=log_observations,
        target_net_updater=target_net_updater,
        async_collection=async_collection,
        log_timings=log_timings,
        auto_log_optim_steps=auto_log_optim_steps,
    )
    _register_trainer_hooks(trainer, hooks)
    return trainer


@dataclass
class TD3TrainerConfig(TrainerConfig):
    """Hydra configuration for :class:`~torchrl.trainers.algorithms.TD3Trainer`.

    Every kwarg accepted by ``TD3Trainer.__init__`` is exposed as a field here.
    """

    collector: Any
    total_frames: int
    loss_module: Any
    logger: Any
    replay_buffer: Any
    save_trainer_file: Any
    optim_steps_per_batch: int | None = 1
    optimizer: Any | None = None
    optimizer_actor: Any | None = None
    optimizer_critic: Any | None = None
    actor_network: Any = None
    qvalue_network: Any = None
    exploration_module: Any = None
    seed: int | None = None
    clip_grad_norm: bool = True
    clip_norm: float | None = None
    frame_skip: int = 1
    progress_bar: bool = True
    save_trainer_interval: int = 10000
    log_interval: int = 10000
    num_epochs: int = 1
    async_collection: bool = False
    log_timings: bool = False
    auto_log_optim_steps: bool = True
    enable_logging: bool = True
    log_rewards: bool = True
    log_actions: bool = True
    log_observations: bool = False
    create_env_fn: Any = None
    target_net_updater: Any = None
    policy_update_delay: int = 2
    value_estimator_gamma: float | None = None
    hooks: list[Any] | None = None
    _target_: str = "torchrl.trainers.algorithms.configs.trainers._make_td3_trainer"

    def __post_init__(self) -> None:
        super().__post_init__()


def _make_td3_trainer(*args, **kwargs):
    from tensordict.nn import TensorDictSequential

    from torchrl.objectives.utils import TargetNetUpdater
    from torchrl.trainers.algorithms.td3 import TD3OptimizationStepper
    from torchrl.trainers.trainers import Logger

    collector = kwargs.pop("collector")
    total_frames = kwargs.pop("total_frames")
    if total_frames is None:
        total_frames = collector.total_frames
    frame_skip = kwargs.pop("frame_skip", 1)
    optim_steps_per_batch = kwargs.pop("optim_steps_per_batch", 1)
    loss_module = kwargs.pop("loss_module")
    optimizer = kwargs.pop("optimizer", None)
    optimizer_actor = kwargs.pop("optimizer_actor", None)
    optimizer_critic = kwargs.pop("optimizer_critic", None)
    logger = kwargs.pop("logger")
    clip_grad_norm = kwargs.pop("clip_grad_norm", True)
    clip_norm = kwargs.pop("clip_norm")
    progress_bar = kwargs.pop("progress_bar", True)
    replay_buffer = kwargs.pop("replay_buffer")
    save_trainer_interval = kwargs.pop("save_trainer_interval", 10000)
    log_interval = kwargs.pop("log_interval", 10000)
    save_trainer_file = kwargs.pop("save_trainer_file")
    seed = kwargs.pop("seed")
    num_epochs = kwargs.pop("num_epochs", 1)
    async_collection = kwargs.pop("async_collection", False)
    log_timings = kwargs.pop("log_timings", False)
    auto_log_optim_steps = kwargs.pop("auto_log_optim_steps", True)
    enable_logging = kwargs.pop("enable_logging", True)
    log_rewards = kwargs.pop("log_rewards", True)
    log_actions = kwargs.pop("log_actions", True)
    log_observations = kwargs.pop("log_observations", False)
    actor_network = kwargs.pop("actor_network", None)
    qvalue_network = kwargs.pop("qvalue_network", None)
    exploration_module = kwargs.pop("exploration_module", None)
    kwargs.pop("create_env_fn")
    target_net_updater = kwargs.pop("target_net_updater", None)
    policy_update_delay = kwargs.pop("policy_update_delay", 2)
    value_estimator_gamma = kwargs.pop("value_estimator_gamma", None)
    hooks = kwargs.pop("hooks", None)

    if actor_network is not None and not isinstance(actor_network, torch.nn.Module):
        actor_network = actor_network()

    if qvalue_network is not None and not isinstance(qvalue_network, torch.nn.Module):
        qvalue_network = qvalue_network()

    if exploration_module is not None and not isinstance(
        exploration_module, torch.nn.Module
    ):
        exploration_module = exploration_module()

    exploration_policy = (
        actor_network
        if exploration_module is None
        else TensorDictSequential(actor_network, exploration_module)
    )

    if not isinstance(collector, BaseCollector):
        collector_kwargs = {"policy": exploration_policy}
        if not async_collection:
            collector = collector(**collector_kwargs)
        elif replay_buffer is not None:
            collector = collector(replay_buffer=replay_buffer, **collector_kwargs)

    env = collector.env
    action_spec = getattr(env, "action_spec_unbatched", None) or env.action_spec
    if hasattr(action_spec, "get"):
        nested_action_spec = action_spec.get("action", default=None)
        if nested_action_spec is not None:
            action_spec = nested_action_spec

    if not callable(loss_module):
        # TD3Loss currently requires real action bounds from the environment. Therefore, we
        # require it to be a partial for now.
        raise TypeError(
            "TD3Trainer currently expects loss_module to be a Hydra partial/callable. "
            "Provide a partial loss config (e.g. loss._partial_=true) and let the "
            "trainer inject actor_network, qvalue_network, and action_spec."
        )
    else:
        loss_module = loss_module(
            action_spec=action_spec,
            actor_network=actor_network,
            qvalue_network=qvalue_network,
        )

    if value_estimator_gamma is not None:
        loss_module.make_value_estimator(gamma=value_estimator_gamma)

    if not isinstance(target_net_updater, TargetNetUpdater):
        target_net_updater = target_net_updater(loss_module)

    if optimizer_actor is None and optimizer_critic is None:
        optimizer_actor = optimizer
        optimizer_critic = optimizer
    elif optimizer_actor is None or optimizer_critic is None:
        raise TypeError(
            "TD3Trainer requires both optimizer_actor and optimizer_critic when overriding optimizer."
        )

    actor_params = list(loss_module.actor_network_params.flatten_keys().values())
    critic_params = list(loss_module.qvalue_network_params.flatten_keys().values())

    if not isinstance(optimizer_actor, torch.optim.Optimizer):
        optimizer_actor = optimizer_actor(params=actor_params)
    if not isinstance(optimizer_critic, torch.optim.Optimizer):
        optimizer_critic = optimizer_critic(params=critic_params)

    optimization_stepper = TD3OptimizationStepper(
        optimizer_actor=optimizer_actor,
        optimizer_critic=optimizer_critic,
        policy_update_delay=policy_update_delay,
        zero_grad_set_to_none=True,
    )

    if not isinstance(collector, BaseCollector):
        raise TypeError(f"collector must be a BaseCollector, got {type(collector)}")
    if not isinstance(loss_module, LossModule):
        raise TypeError(f"loss_module must be a LossModule, got {type(loss_module)}")
    if optimizer_actor is None or optimizer_critic is None:
        raise TypeError("TD3Trainer requires optimizer configuration.")
    if not isinstance(optimizer_actor, torch.optim.Optimizer):
        raise TypeError(
            f"TD3Trainer requires optimizer_actor to be a torch.optim.Optimizer, got {type(optimizer)}"
        )
    if not isinstance(optimizer_critic, torch.optim.Optimizer):
        raise TypeError(
            f"TD3Trainer requires optimizer_critic to be a torch.optim.Optimizer, got {type(optimizer)}"
        )
    if not isinstance(logger, Logger) and logger is not None:
        raise TypeError(f"logger must be a Logger or None, got {type(logger)}")

    trainer = TD3Trainer(
        collector=collector,
        total_frames=total_frames,
        frame_skip=frame_skip,
        optim_steps_per_batch=optim_steps_per_batch,
        loss_module=loss_module,
        optimizer=None,
        optimization_stepper=optimization_stepper,
        logger=logger,
        clip_grad_norm=clip_grad_norm,
        clip_norm=clip_norm,
        progress_bar=progress_bar,
        seed=seed,
        save_trainer_interval=save_trainer_interval,
        log_interval=log_interval,
        save_trainer_file=save_trainer_file,
        num_epochs=num_epochs,
        replay_buffer=replay_buffer,
        enable_logging=enable_logging,
        log_rewards=log_rewards,
        log_actions=log_actions,
        log_observations=log_observations,
        async_collection=async_collection,
        log_timings=log_timings,
        auto_log_optim_steps=auto_log_optim_steps,
        target_net_updater=target_net_updater,
        exploration_module=exploration_module,
    )
    _register_trainer_hooks(trainer, hooks)
    return trainer
