# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchrl.trainers.algorithms.on_policy import OnPolicyTrainer


class PPOTrainer(OnPolicyTrainer):
    """PPO (Proximal Policy Optimization) trainer implementation.

    See also :class:`~torchrl.trainers.algorithms.configs.PPOTrainerConfig` for the
    Hydra configuration counterpart.

    .. warning::
        This is an experimental/prototype feature. The API may change in future versions.
        Please report any issues or feedback to help improve this implementation.

    This trainer implements the PPO algorithm for training reinforcement learning agents.
    It extends :class:`~torchrl.trainers.algorithms.OnPolicyTrainer` with PPO-specific
    defaults; see that class for the full list of keyword arguments, covering
    advantage estimation (GAE), replay-buffer wiring, collector weight
    synchronization and logging.

    PPO typically uses multiple epochs of optimization on the same batch of data.
    This trainer defaults to 4 epochs, which is a common choice for PPO implementations.

    Examples:
        >>> # Basic usage with manual configuration
        >>> from torchrl.trainers.algorithms.ppo import PPOTrainer
        >>> from torchrl.trainers.algorithms.configs import PPOTrainerConfig
        >>> from hydra.utils import instantiate
        >>> config = PPOTrainerConfig(...)  # Configure with required parameters
        >>> trainer = instantiate(config)
        >>> trainer.train()

    .. note::
        This trainer requires a configurable environment setup. See the
        :class:`~torchrl.trainers.algorithms.configs` module for configuration options.
    """

    _algo_name = "PPO"
    _default_num_epochs = 4
