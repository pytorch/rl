# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pathlib

from typing import Callable

from tensordict import TensorDictBase
from torch import optim

from torchrl.collectors.collectors import DataCollectorBase

from torchrl.objectives.common import LossModule
from torchrl.record.loggers import Logger
from torchrl.trainers.algorithms.configs.data import (
    LazyTensorStorageConfig,
    ReplayBufferConfig,
)
from torchrl.trainers.algorithms.configs.envs import GymEnvConfig
from torchrl.trainers.algorithms.configs.modules import MLPConfig, TanhNormalModelConfig
from torchrl.trainers.algorithms.configs.objectives import PPOLossConfig
from torchrl.trainers.algorithms.configs.utils import AdamConfig
from torchrl.trainers.trainers import Trainer

try:
    pass

    _has_tqdm = True
except ImportError:
    _has_tqdm = False

try:
    pass

    _has_ts = True
except ImportError:
    _has_ts = False


class PPOTrainer(Trainer):
    def __init__(
        self,
        *,
        collector: DataCollectorBase,
        total_frames: int,
        frame_skip: int,
        optim_steps_per_batch: int,
        loss_module: LossModule | Callable[[TensorDictBase], TensorDictBase],
        optimizer: optim.Optimizer | None = None,
        logger: Logger | None = None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = None,
        progress_bar: bool = True,
        seed: int | None = None,
        save_trainer_interval: int = 10000,
        log_interval: int = 10000,
        save_trainer_file: str | pathlib.Path | None = None,
        replay_buffer=None,
    ) -> None:
        super().__init__(
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
        )
        self.replay_buffer = replay_buffer

    @classmethod
    def default_config(cls, **kwargs) -> PPOTrainerConfig:  # type: ignore # noqa: F821
        """Creates a default config for the PPO trainer.

        The task is the Pendulum-v1 environment in Gym, with a 2-layer MLP actor and critic.

        Args:
            **kwargs: Override default values. Supports nested overrides using double underscore notation
                     (e.g., "actor_network__network__num_cells": 256)

        Returns:
            PPOTrainerConfig with default values, overridden by kwargs

        Examples:
            # Basic usage with defaults
            config = PPOTrainer.default_config()
            
            # Override top-level parameters
            config = PPOTrainer.default_config(
                total_frames=2_000_000,
                clip_norm=0.5
            )
            
            # Override nested network parameters
            config = PPOTrainer.default_config(
                actor_network__network__num_cells=256,
                actor_network__network__depth=3,
                critic_network__module__num_cells=256
            )
            
            # Override environment parameters
            config = PPOTrainer.default_config(
                env_cfg__env_name="HalfCheetah-v4",
                env_cfg__backend="gymnasium"
            )
            
            # Override multiple parameters at once
            config = PPOTrainer.default_config(
                total_frames=2_000_000,
                actor_network__network__num_cells=256,
                env_cfg__env_name="Walker2d-v4",
                replay_buffer_cfg__batch_size=512
            )
        """
        from torchrl.trainers.algorithms.configs.collectors import (
            SyncDataCollectorConfig,
        )
        from torchrl.trainers.algorithms.configs.modules import TensorDictModuleConfig
        from torchrl.trainers.algorithms.configs.trainers import PPOTrainerConfig

        # 1. Unflatten the kwargs using TensorDict to understand what the user wants
        from tensordict import TensorDict
        kwargs_td = TensorDict(kwargs)
        unflattened_kwargs = kwargs_td.unflatten_keys("__").to_dict()
        
        # Convert any torch tensors back to Python scalars for config compatibility
        def convert_tensors_to_scalars(obj):
            if isinstance(obj, dict):
                return {k: convert_tensors_to_scalars(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_tensors_to_scalars(v) for v in obj]
            elif hasattr(obj, 'item') and hasattr(obj, 'dim'):  # torch tensor
                if obj.dim() == 0:  # scalar tensor
                    return obj.item()
                else:
                    return obj.tolist()  # convert multi-dimensional tensors to lists
            else:
                return obj
        
        unflattened_kwargs = convert_tensors_to_scalars(unflattened_kwargs)

        # 2. Create configs by passing the appropriate nested configs to each Config object
        # Environment config
        env_overrides = unflattened_kwargs.get("env_cfg", {})
        env_cfg = GymEnvConfig.default_config(**env_overrides)

        # Collector config
        collector_overrides = unflattened_kwargs.get("collector_cfg", {})
        collector_cfg = SyncDataCollectorConfig.default_config(**collector_overrides)

        # Loss config
        loss_overrides = unflattened_kwargs.get("loss_cfg", {})
        loss_cfg = PPOLossConfig.default_config(**loss_overrides)

        # Optimizer config
        optimizer_overrides = unflattened_kwargs.get("optimizer_cfg", {})
        optimizer_cfg = AdamConfig.default_config(**optimizer_overrides)

        # Replay buffer config
        replay_buffer_overrides = unflattened_kwargs.get("replay_buffer_cfg", {})
        replay_buffer_cfg = ReplayBufferConfig.default_config(**replay_buffer_overrides)

        # Actor network config with proper out_features for Pendulum-v1 (action_dim=1)
        actor_overrides = unflattened_kwargs.get("actor_network", {})
        # For Pendulum-v1, action_dim=1, but TanhNormal needs 2 outputs (loc and scale)
        if "network" not in actor_overrides:
            actor_overrides["network"] = {}
        if "out_features" not in actor_overrides["network"]:
            actor_overrides["network"]["out_features"] = int(2)  # 2 for loc and scale
        actor_network = TanhNormalModelConfig.default_config(**actor_overrides)
        
        # Critic network config with proper out_features for value function (always 1)
        critic_overrides = unflattened_kwargs.get("critic_network", {})
        # For value function, out_features should be 1
        if "module" not in critic_overrides:
            critic_overrides["module"] = {}
        if "out_features" not in critic_overrides["module"]:
            critic_overrides["module"]["out_features"] = int(1)  # 1 for value function
        critic_network = TensorDictModuleConfig.default_config(**critic_overrides)

        # 3. Build the final config dict with the resulting config objects
        config_dict = {
            "collector": collector_cfg,
            "total_frames": unflattened_kwargs.get("total_frames", 1_000_000),
            "frame_skip": unflattened_kwargs.get("frame_skip", 1),
            "optim_steps_per_batch": unflattened_kwargs.get("optim_steps_per_batch", 1),
            "loss_module": loss_cfg,
            "optimizer": optimizer_cfg,
            "logger": unflattened_kwargs.get("logger", None),
            "clip_grad_norm": unflattened_kwargs.get("clip_grad_norm", True),
            "clip_norm": unflattened_kwargs.get("clip_norm", 1.0),
            "progress_bar": unflattened_kwargs.get("progress_bar", True),
            "seed": unflattened_kwargs.get("seed", 1),
            "save_trainer_interval": unflattened_kwargs.get("save_trainer_interval", 10000),
            "log_interval": unflattened_kwargs.get("log_interval", 10000),
            "save_trainer_file": unflattened_kwargs.get("save_trainer_file", None),
            "replay_buffer": replay_buffer_cfg,
            "create_env_fn": env_cfg,
            "actor_network": actor_network,
            "critic_network": critic_network,
        }
        
        return PPOTrainerConfig(**config_dict)
