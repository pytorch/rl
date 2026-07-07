# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchrl.trainers.algorithms.on_policy import OnPolicyTrainer


class A2CTrainer(OnPolicyTrainer):
    """A2C (Advantage Actor-Critic) trainer implementation.

    See also :class:`~torchrl.trainers.algorithms.configs.A2CTrainerConfig` for the
    Hydra configuration counterpart.

    .. warning::
        This is an experimental/prototype feature. The API may change in future versions.
        Please report any issues or feedback to help improve this implementation.

    This trainer implements the A2C algorithm for training reinforcement learning agents.
    It extends :class:`~torchrl.trainers.algorithms.OnPolicyTrainer` with A2C-specific
    defaults; see that class for the full list of keyword arguments, covering
    advantage estimation (GAE), replay-buffer wiring, collector weight
    synchronization and logging. Entropy regularization is configured on the
    loss module (see :class:`~torchrl.objectives.A2CLoss`).

    Unlike PPO, A2C performs a single optimization pass over each batch of collected
    data. This trainer therefore defaults to 1 epoch per batch.

    Examples:
        >>> # Basic usage with manual configuration
        >>> from torchrl.trainers.algorithms.a2c import A2CTrainer
        >>> from torchrl.trainers.algorithms.configs import A2CTrainerConfig
        >>> from hydra.utils import instantiate
        >>> config = A2CTrainerConfig(...)  # Configure with required parameters
        >>> trainer = instantiate(config)
        >>> trainer.train()

    .. note::
        This trainer requires a configurable environment setup. See the
        :class:`~torchrl.trainers.algorithms.configs` module for configuration options.
    """

    _algo_name = "A2C"
    _default_num_epochs = 1
