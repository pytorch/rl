# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


from torchrl.objectives.ppo import ClipPPOLoss, KLPENPPOLoss, PPOLoss
from torchrl.trainers.algorithms.configs.common import ConfigBase


@dataclass
class LossConfig(ConfigBase):
    """A class to configure a loss.

    Args:
        loss_type: The type of loss to use.
    """

    _partial_: bool = False


@dataclass
class PPOLossConfig(LossConfig):
    """A class to configure a PPO loss.

    Args:
        loss_type: The type of loss to use.
    """

    loss_type: str = "clip"

    actor_network: Any = None
    critic_network: Any = None
    entropy_bonus: bool = True
    samples_mc_entropy: int = 1
    entropy_coeff: Any = None
    log_explained_variance: bool = True
    critic_coeff: float | None = None
    loss_critic_type: str = "smooth_l1"
    normalize_advantage: bool = False
    normalize_advantage_exclude_dims: tuple[int, ...] = ()
    gamma: float | None = None
    separate_losses: bool = False
    advantage_key: str | None = None
    value_target_key: str | None = None
    value_key: str | None = None
    functional: bool = True
    actor: Any = None
    critic: Any = None
    reduction: str | None = None
    clip_value: float | None = None
    device: Any = None
    _target_: str = "torchrl.trainers.algorithms.configs.objectives._make_ppo_loss"
    _partial_: bool = False

    @classmethod
    def default_config(cls, **kwargs) -> "PPOLossConfig":
        """Creates a default PPO loss configuration.
        
        Args:
            **kwargs: Override default values. Supports nested overrides using double underscore notation
                     (e.g., "actor_network__network__num_cells": 256)
            
        Returns:
            PPOLossConfig with default values, overridden by kwargs
        """
        from torchrl.trainers.algorithms.configs.modules import TanhNormalModelConfig, TensorDictModuleConfig
        from tensordict import TensorDict

        # Unflatten the kwargs using TensorDict to understand what the user wants
        kwargs_td = TensorDict(kwargs)
        unflattened_kwargs = kwargs_td.unflatten_keys("__").to_dict()

        # Create configs with nested overrides applied
        actor_overrides = unflattened_kwargs.get("actor_network", {})
        critic_overrides = unflattened_kwargs.get("critic_network", {})
        
        actor_network = TanhNormalModelConfig.default_config(**actor_overrides)
        critic_network = TensorDictModuleConfig.default_config(**critic_overrides)

        defaults = {
            "loss_type": unflattened_kwargs.get("loss_type", "clip"),
            "actor_network": actor_network,
            "critic_network": critic_network,
            "entropy_bonus": unflattened_kwargs.get("entropy_bonus", True),
            "samples_mc_entropy": unflattened_kwargs.get("samples_mc_entropy", 1),
            "entropy_coeff": unflattened_kwargs.get("entropy_coeff", None),
            "log_explained_variance": unflattened_kwargs.get("log_explained_variance", True),
            "critic_coeff": unflattened_kwargs.get("critic_coeff", 0.25),
            "loss_critic_type": unflattened_kwargs.get("loss_critic_type", "smooth_l1"),
            "normalize_advantage": unflattened_kwargs.get("normalize_advantage", True),
            "normalize_advantage_exclude_dims": unflattened_kwargs.get("normalize_advantage_exclude_dims", ()),
            "gamma": unflattened_kwargs.get("gamma", None),
            "separate_losses": unflattened_kwargs.get("separate_losses", False),
            "advantage_key": unflattened_kwargs.get("advantage_key", None),
            "value_target_key": unflattened_kwargs.get("value_target_key", None),
            "value_key": unflattened_kwargs.get("value_key", None),
            "functional": unflattened_kwargs.get("functional", True),
            "actor": unflattened_kwargs.get("actor", None),
            "critic": unflattened_kwargs.get("critic", None),
            "reduction": unflattened_kwargs.get("reduction", None),
            "clip_value": unflattened_kwargs.get("clip_value", None),
            "device": unflattened_kwargs.get("device", None),
            "_partial_": True,
        }
        
        return cls(**defaults)


def _make_ppo_loss(*args, **kwargs) -> PPOLoss:
    loss_type = kwargs.pop("loss_type", "clip")
    if loss_type == "clip":
        return ClipPPOLoss(*args, **kwargs)
    elif loss_type == "kl":
        return KLPENPPOLoss(*args, **kwargs)
    elif loss_type == "ppo":
        return PPOLoss(*args, **kwargs)
    else:
        raise ValueError(f"Invalid loss type: {loss_type}")
