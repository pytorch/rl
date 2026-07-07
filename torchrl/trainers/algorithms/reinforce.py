# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from torchrl.trainers.algorithms.on_policy import OnPolicyTrainer


class ReinforceTrainer(OnPolicyTrainer):
    """REINFORCE (policy gradient with baseline) trainer implementation.

    See also :class:`~torchrl.trainers.algorithms.configs.ReinforceTrainerConfig` for
    the Hydra configuration counterpart.

    .. warning::
        This is an experimental/prototype feature. The API may change in future versions.
        Please report any issues or feedback to help improve this implementation.

    This trainer implements the REINFORCE algorithm for training reinforcement
    learning agents, using a critic network as a baseline for advantage
    estimation (GAE by default, matching
    :class:`~torchrl.objectives.ReinforceLoss`). It extends
    :class:`~torchrl.trainers.algorithms.OnPolicyTrainer`; see that class for the
    full list of keyword arguments, covering advantage estimation,
    replay-buffer wiring, collector weight synchronization and logging.

    REINFORCE is a single-pass on-policy algorithm: each collected batch is
    consumed once. This trainer therefore defaults to 1 epoch per batch.

    Examples:
        >>> # Basic usage with manual configuration
        >>> from torchrl.trainers.algorithms.reinforce import ReinforceTrainer
        >>> from torchrl.trainers.algorithms.configs import ReinforceTrainerConfig
        >>> from hydra.utils import instantiate
        >>> config = ReinforceTrainerConfig(...)  # Configure with required parameters
        >>> trainer = instantiate(config)
        >>> trainer.train()

    .. note::
        This trainer requires a configurable environment setup. See the
        :class:`~torchrl.trainers.algorithms.configs` module for configuration options.
    """

    _algo_name = "REINFORCE"
    _default_num_epochs = 1
