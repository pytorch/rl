# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TYPE_CHECKING

import torch
from tensordict.nn import TensorDictModuleBase, TensorDictSequential

from torchrl.collectors import BaseCollector
from torchrl.data import Categorical, Composite, OneHot
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.objectives.value.advantages import GAE
from torchrl.trainers import TrainerHookBase
from torchrl.trainers.algorithms.a2c import A2CTrainer
from torchrl.trainers.algorithms.configs.common import _normalize_hydra_key, ConfigBase
from torchrl.trainers.algorithms.cql import CQLTrainer
from torchrl.trainers.algorithms.ddpg import DDPGTrainer
from torchrl.trainers.algorithms.dqn import DQNTrainer
from torchrl.trainers.algorithms.iql import IQLTrainer
from torchrl.trainers.algorithms.offline_to_online import OfflineToOnlineTrainer
from torchrl.trainers.algorithms.ppo import PPOTrainer
from torchrl.trainers.algorithms.reinforce import ReinforceTrainer
from torchrl.trainers.algorithms.sac import SACTrainer
from torchrl.trainers.algorithms.td3 import TD3Trainer

if TYPE_CHECKING:
    _LearnerBackend = Literal["local", "ray"]
else:
    # OmegaConf structured configs do not support Literal on all supported versions.
    _LearnerBackend = str


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
    learner_backend: _LearnerBackend = "local"
    learner_backend_options: dict[str, Any] | None = None
    learner_poll_interval: float = 0.05
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
    checkpoint: Any = None

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
    checkpoint = kwargs.pop("checkpoint", None)
    seed = kwargs.pop("seed")
    actor_network = kwargs.pop("actor_network")
    critic_network = kwargs.pop("critic_network")
    kwargs.pop("create_env_fn")
    target_net_updater = kwargs.pop("target_net_updater")
    async_collection = kwargs.pop("async_collection", False)
    log_timings = kwargs.pop("log_timings", False)
    auto_log_optim_steps = kwargs.pop("auto_log_optim_steps", True)
    batch_size = kwargs.pop("batch_size", None)
    learner_backend = kwargs.pop("learner_backend", "local")
    learner_backend_options = kwargs.pop("learner_backend_options", None)
    learner_poll_interval = kwargs.pop("learner_poll_interval", 0.05)
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
    if target_net_updater is None:
        raise ValueError("SACTrainerConfig requires target_net_updater.")
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
        checkpoint=checkpoint,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        learner_backend=learner_backend,
        learner_backend_options=learner_backend_options,
        learner_poll_interval=learner_poll_interval,
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
class OfflineToOnlineTrainerConfig(SACTrainerConfig):
    """Hydra configuration for :class:`~torchrl.trainers.algorithms.OfflineToOnlineTrainer`.

    Every kwarg accepted by ``OfflineToOnlineTrainer.__init__`` is exposed as a
    field here, with SAC network-construction helper fields inherited from
    :class:`SACTrainerConfig`.
    """

    anneal_frames: int | None = None

    _target_: str = (
        "torchrl.trainers.algorithms.configs.trainers."
        "_make_offline_to_online_trainer"
    )

    def __post_init__(self) -> None:
        """Post-initialization hook for offline-to-online trainer configuration."""
        super().__post_init__()


def _make_offline_to_online_trainer(*args, **kwargs) -> OfflineToOnlineTrainer:
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
    checkpoint = kwargs.pop("checkpoint", None)
    seed = kwargs.pop("seed")
    actor_network = kwargs.pop("actor_network")
    critic_network = kwargs.pop("critic_network")
    kwargs.pop("create_env_fn")
    target_net_updater = kwargs.pop("target_net_updater")
    async_collection = kwargs.pop("async_collection", False)
    if async_collection:
        raise ValueError("OfflineToOnlineTrainer does not support async_collection.")
    log_timings = kwargs.pop("log_timings", False)
    auto_log_optim_steps = kwargs.pop("auto_log_optim_steps", True)
    batch_size = kwargs.pop("batch_size", None)
    anneal_frames = kwargs.pop("anneal_frames", None)
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
        collector = collector()

    if not isinstance(loss_module, LossModule):
        # then it's a partial config
        loss_module = loss_module(
            actor_network=actor_network, critic_network=critic_network
        )
    if target_net_updater is not None and not isinstance(
        target_net_updater, TargetNetUpdater
    ):
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

    trainer = OfflineToOnlineTrainer(
        collector=collector,
        total_frames=total_frames,
        frame_skip=frame_skip,
        optim_steps_per_batch=optim_steps_per_batch,
        loss_module=loss_module,
        replay_buffer=replay_buffer,
        anneal_frames=anneal_frames,
        batch_size=batch_size,
        optimizer=optimizer,
        logger=logger,
        clip_grad_norm=clip_grad_norm,
        clip_norm=clip_norm,
        progress_bar=progress_bar,
        seed=seed,
        save_trainer_interval=save_trainer_interval,
        log_interval=log_interval,
        save_trainer_file=save_trainer_file,
        checkpoint=checkpoint,
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
class OnPolicyTrainerConfig(TrainerConfig):
    """Base Hydra configuration for on-policy trainers.

    Exposes every kwarg accepted by
    :class:`~torchrl.trainers.algorithms.OnPolicyTrainer` as a field. Algorithm
    configs (:class:`PPOTrainerConfig`, :class:`A2CTrainerConfig`,
    :class:`ReinforceTrainerConfig`) subclass it, overriding only the
    algorithm-specific defaults and the factory ``_target_``.

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
        num_epochs: Number of epochs per batch.
        async_collection: Whether to use async collection. Default: False.
        add_gae: Whether to add GAE computation. Default: True.
        gae: Custom GAE module configuration.
        lr_scheduler: Learning-rate scheduler (or a partial configuration taking
            the optimizer as input), stepped once per collected batch via
            :class:`~torchrl.trainers.LRSchedulerHook`.
        weight_update_map: Mapping from collector destination paths to trainer source paths.
            Required if collector has weight_sync_schemes configured.
            Example: ``{"policy": "loss_module.actor_network", "replay_buffer.transforms[0]": "loss_module.critic_network"}``.
        log_timings: Whether to automatically log timing information for all hooks.
            If True, timing metrics will be logged to the logger (e.g., wandb, tensorboard)
            with prefix "time/" (e.g., "time/hook/UpdateWeights"). Default: False.
        auto_log_optim_steps: Whether to log the number of optimization steps after
            each optimization loop. Default: True.
        batch_size: Unused by on-policy trainers; set the batch size on the replay
            buffer instead.
        gamma: Discount factor for the default GAE module. Default: 0.99.
        lmbda: Lambda parameter for the default GAE module. Default: 0.95.
        enable_logging: Whether to enable logging. Default: True.
        log_rewards: Whether to log rewards. Default: True.
        log_actions: Whether to log actions. Default: True.
        log_observations: Whether to log observations. Default: False.
        done_key: Done key used by GAE, losses, and logging. Default: "done".
        terminated_key: Terminated key used by GAE, losses, and logging. Default: "terminated".
        reward_key: Reward key used by GAE, losses, and logging. Default: "reward".
        episode_reward_key: Episode reward key used for cumulative reward logging. Default: "reward".
        action_key: Action key used by losses and logging. Default: "action".
        observation_key: Observation key used for logging. Default: "observation".
        hooks: List of :class:`~torchrl.trainers.TrainerHookBase` instances to
            register on the trainer after construction.
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
    num_epochs: int = 1
    async_collection: bool = False
    add_gae: bool = True
    gae: Any = None
    lr_scheduler: Any = None
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
    checkpoint: Any = None

    def __post_init__(self) -> None:
        """Post-initialization hook for on-policy trainer configurations."""
        super().__post_init__()


@dataclass
class PPOTrainerConfig(OnPolicyTrainerConfig):
    """Hydra configuration for :class:`~torchrl.trainers.algorithms.PPOTrainer`.

    Every kwarg accepted by ``PPOTrainer.__init__`` is exposed as a field here;
    see :class:`OnPolicyTrainerConfig` for the full field list. PPO defaults to
    4 optimization epochs per collected batch.
    """

    num_epochs: int = 4

    _target_: str = "torchrl.trainers.algorithms.configs.trainers._make_ppo_trainer"


@dataclass
class A2CTrainerConfig(OnPolicyTrainerConfig):
    """Hydra configuration for :class:`~torchrl.trainers.algorithms.A2CTrainer`.

    Every kwarg accepted by ``A2CTrainer.__init__`` is exposed as a field here;
    see :class:`OnPolicyTrainerConfig` for the full field list. A2C performs a
    single optimization pass over each collected batch (``num_epochs=1``).
    """

    _target_: str = "torchrl.trainers.algorithms.configs.trainers._make_a2c_trainer"


@dataclass
class ReinforceTrainerConfig(OnPolicyTrainerConfig):
    """Hydra configuration for :class:`~torchrl.trainers.algorithms.ReinforceTrainer`.

    Every kwarg accepted by ``ReinforceTrainer.__init__`` is exposed as a field
    here; see :class:`OnPolicyTrainerConfig` for the full field list. REINFORCE
    performs a single optimization pass over each collected batch
    (``num_epochs=1``).
    """

    _target_: str = (
        "torchrl.trainers.algorithms.configs.trainers._make_reinforce_trainer"
    )


def _make_onpolicy_trainer(trainer_cls, *args, **kwargs):
    from torchrl.trainers.trainers import Logger

    collector = kwargs.pop("collector")
    total_frames = kwargs.pop("total_frames")
    if total_frames is None:
        total_frames = collector.total_frames
    frame_skip = kwargs.pop("frame_skip", 1)
    optim_steps_per_batch = kwargs.pop("optim_steps_per_batch", 1)
    loss_module = kwargs.pop("loss_module")
    optimizer = kwargs.pop("optimizer")
    logger = kwargs.pop("logger", None)
    clip_grad_norm = kwargs.pop("clip_grad_norm", True)
    clip_norm = kwargs.pop("clip_norm", None)
    progress_bar = kwargs.pop("progress_bar", True)
    replay_buffer = kwargs.pop("replay_buffer", None)
    save_trainer_interval = kwargs.pop("save_trainer_interval", 10000)
    log_interval = kwargs.pop("log_interval", 10000)
    save_trainer_file = kwargs.pop("save_trainer_file", None)
    checkpoint = kwargs.pop("checkpoint", None)
    seed = kwargs.pop("seed", None)
    actor_network = kwargs.pop("actor_network", None)
    critic_network = kwargs.pop("critic_network", None)
    add_gae = kwargs.pop("add_gae", True)
    gae = kwargs.pop("gae", None)
    kwargs.pop("create_env_fn", None)
    lr_scheduler = kwargs.pop("lr_scheduler", None)
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
    num_epochs = kwargs.pop("num_epochs", None)
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
    if lr_scheduler is not None and not isinstance(
        lr_scheduler, torch.optim.lr_scheduler.LRScheduler
    ):
        # then it's a partial config taking the optimizer as input
        lr_scheduler = lr_scheduler(optimizer)

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

    trainer = trainer_cls(
        collector=collector,
        total_frames=total_frames,
        frame_skip=frame_skip,
        optim_steps_per_batch=optim_steps_per_batch,
        loss_module=loss_module,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        logger=logger,
        clip_grad_norm=clip_grad_norm,
        clip_norm=clip_norm,
        progress_bar=progress_bar,
        seed=seed,
        save_trainer_interval=save_trainer_interval,
        log_interval=log_interval,
        save_trainer_file=save_trainer_file,
        checkpoint=checkpoint,
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


def _make_ppo_trainer(*args, **kwargs) -> PPOTrainer:
    return _make_onpolicy_trainer(PPOTrainer, *args, **kwargs)


def _make_a2c_trainer(*args, **kwargs) -> A2CTrainer:
    return _make_onpolicy_trainer(A2CTrainer, *args, **kwargs)


def _make_reinforce_trainer(*args, **kwargs) -> ReinforceTrainer:
    return _make_onpolicy_trainer(ReinforceTrainer, *args, **kwargs)


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
    batch_size: int | None = None
    learner_backend: _LearnerBackend = "local"
    learner_backend_options: dict[str, Any] | None = None
    learner_poll_interval: float = 0.05
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
    greedy_module: Any = None
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
    checkpoint: Any = None
    mixing_strategy: str | None = None
    done_key: Any = "done"
    terminated_key: Any = "terminated"
    reward_key: Any = "reward"
    episode_reward_key: Any = "reward_sum"
    aggregated_reward_key: Any = None
    aggregated_episode_reward_key: Any = None
    action_key: Any = "action"
    observation_key: Any = "observation"

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
    batch_size = kwargs.pop("batch_size", None)
    learner_backend = kwargs.pop("learner_backend", "local")
    learner_backend_options = kwargs.pop("learner_backend_options", None)
    learner_poll_interval = kwargs.pop("learner_poll_interval", 0.05)
    save_trainer_interval = kwargs.pop("save_trainer_interval", 10000)
    log_interval = kwargs.pop("log_interval", 10000)
    save_trainer_file = kwargs.pop("save_trainer_file")
    checkpoint = kwargs.pop("checkpoint", None)
    seed = kwargs.pop("seed")
    value_network = kwargs.pop("value_network")
    kwargs.pop("create_env_fn", None)
    target_net_updater = kwargs.pop("target_net_updater")
    greedy_module = kwargs.pop("greedy_module", None)
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
    mixing_strategy = kwargs.pop("mixing_strategy", None)
    done_key = _normalize_hydra_key(kwargs.pop("done_key", "done"))
    terminated_key = _normalize_hydra_key(kwargs.pop("terminated_key", "terminated"))
    reward_key = _normalize_hydra_key(kwargs.pop("reward_key", "reward"))
    episode_reward_key = _normalize_hydra_key(
        kwargs.pop("episode_reward_key", "reward_sum")
    )
    aggregated_reward_key = _normalize_hydra_key(
        kwargs.pop("aggregated_reward_key", None)
    )
    aggregated_episode_reward_key = _normalize_hydra_key(
        kwargs.pop("aggregated_episode_reward_key", None)
    )
    action_key = _normalize_hydra_key(kwargs.pop("action_key", "action"))
    observation_key = _normalize_hydra_key(kwargs.pop("observation_key", "observation"))

    if value_network is not None and not isinstance(value_network, torch.nn.Module):
        value_network = value_network()

    action_spec = value_network.spec.get(action_key, default=None)
    if action_spec is None:
        net = value_network.module[0]
        n_actions = (
            net.n_agent_outputs if hasattr(net, "n_agent_outputs") else net.out_features
        )
        if getattr(value_network, "action_space", None) == "categorical":
            action_spec = Categorical(n=n_actions)
        else:
            action_spec = OneHot(n=n_actions)
    spec = Composite({action_key: action_spec})

    if greedy_module is None:
        greedy_module = EGreedyModule(
            annealing_num_steps=annealing_num_steps,
            eps_init=eps_init,
            eps_end=eps_end,
            spec=spec,
            action_key=action_key,
        )
    elif not isinstance(greedy_module, torch.nn.Module):
        greedy_module = greedy_module(spec=spec, action_key=action_key)
    exploration_policy = TensorDictSequential(value_network, greedy_module)

    if not isinstance(collector, BaseCollector):
        collector_kwargs = {"policy": exploration_policy}
        if not async_collection:
            collector = collector(**collector_kwargs)
        elif replay_buffer is not None:
            collector = collector(replay_buffer=replay_buffer, **collector_kwargs)

    if not isinstance(loss_module, LossModule):
        if mixing_strategy in (None, "iql"):
            loss_module = loss_module(value_network=value_network)
        elif mixing_strategy in ("qmix", "vdn"):
            loss_module = loss_module(local_value_network=value_network)
        else:
            raise ValueError(
                "mixing_strategy must be one of None, 'iql', 'qmix', or 'vdn', "
                f"got {mixing_strategy}."
            )

    if target_net_updater is None:
        raise ValueError("DQNTrainerConfig requires target_net_updater.")
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
        checkpoint=checkpoint,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        learner_backend=learner_backend,
        learner_backend_options=learner_backend_options,
        learner_poll_interval=learner_poll_interval,
        enable_logging=enable_logging,
        log_rewards=log_rewards,
        log_observations=log_observations,
        target_net_updater=target_net_updater,
        greedy_module=greedy_module,
        mixing_strategy=mixing_strategy,
        done_key=done_key,
        terminated_key=terminated_key,
        reward_key=reward_key,
        episode_reward_key=episode_reward_key,
        aggregated_reward_key=aggregated_reward_key,
        aggregated_episode_reward_key=aggregated_episode_reward_key,
        action_key=action_key,
        observation_key=observation_key,
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
    batch_size: int | None = None
    learner_backend: _LearnerBackend = "local"
    learner_backend_options: dict[str, Any] | None = None
    learner_poll_interval: float = 0.05
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
    exploration_module: Any = None
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
    checkpoint: Any = None

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
    batch_size = kwargs.pop("batch_size", None)
    learner_backend = kwargs.pop("learner_backend", "local")
    learner_backend_options = kwargs.pop("learner_backend_options", None)
    learner_poll_interval = kwargs.pop("learner_poll_interval", 0.05)
    save_trainer_interval = kwargs.pop("save_trainer_interval", 10000)
    log_interval = kwargs.pop("log_interval", 10000)
    save_trainer_file = kwargs.pop("save_trainer_file")
    checkpoint = kwargs.pop("checkpoint", None)
    seed = kwargs.pop("seed")
    actor_network = kwargs.pop("actor_network")
    critic_network = kwargs.pop("critic_network")
    exploration_module = kwargs.pop("exploration_module", None)
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
    if exploration_module is not None and not isinstance(
        exploration_module, torch.nn.Module
    ):
        exploration_module = exploration_module()
    if exploration_module is not None and actor_network is None:
        raise ValueError(
            "DDPGTrainerConfig requires actor_network when exploration_module is set."
        )
    exploration_policy = (
        TensorDictSequential(actor_network, exploration_module)
        if exploration_module is not None
        else actor_network
    )
    if not isinstance(collector, BaseCollector):
        collector_kwargs = (
            {"policy": exploration_policy} if exploration_policy is not None else {}
        )
        if not async_collection:
            collector = collector(**collector_kwargs)
        elif replay_buffer is not None:
            collector = collector(replay_buffer=replay_buffer, **collector_kwargs)

    if not isinstance(loss_module, LossModule):
        loss_module = loss_module(
            actor_network=actor_network, value_network=critic_network
        )
    if target_net_updater is None:
        raise ValueError("DDPGTrainerConfig requires target_net_updater.")
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
        checkpoint=checkpoint,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        learner_backend=learner_backend,
        learner_backend_options=learner_backend_options,
        learner_poll_interval=learner_poll_interval,
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
        exploration_module=exploration_module,
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
    checkpoint: Any = None

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
    checkpoint = kwargs.pop("checkpoint", None)
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
        checkpoint=checkpoint,
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
    checkpoint: Any = None

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
    checkpoint = kwargs.pop("checkpoint", None)
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
        checkpoint=checkpoint,
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
    optimization_stepper: Any | None = None
    batch_size: int | None = None
    learner_backend: _LearnerBackend = "local"
    learner_backend_options: dict[str, Any] | None = None
    learner_poll_interval: float = 0.05
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
    checkpoint: Any = None
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
    optimization_stepper = kwargs.pop("optimization_stepper", None)
    logger = kwargs.pop("logger")
    clip_grad_norm = kwargs.pop("clip_grad_norm", True)
    clip_norm = kwargs.pop("clip_norm")
    progress_bar = kwargs.pop("progress_bar", True)
    replay_buffer = kwargs.pop("replay_buffer")
    batch_size = kwargs.pop("batch_size", None)
    learner_backend = kwargs.pop("learner_backend", "local")
    learner_backend_options = kwargs.pop("learner_backend_options", None)
    learner_poll_interval = kwargs.pop("learner_poll_interval", 0.05)
    save_trainer_interval = kwargs.pop("save_trainer_interval", 10000)
    log_interval = kwargs.pop("log_interval", 10000)
    save_trainer_file = kwargs.pop("save_trainer_file")
    checkpoint = kwargs.pop("checkpoint", None)
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

    if target_net_updater is None:
        raise ValueError("TD3TrainerConfig requires target_net_updater.")
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

    if optimization_stepper is None:
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
        checkpoint=checkpoint,
        num_epochs=num_epochs,
        replay_buffer=replay_buffer,
        batch_size=batch_size,
        learner_backend=learner_backend,
        learner_backend_options=learner_backend_options,
        learner_poll_interval=learner_poll_interval,
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
