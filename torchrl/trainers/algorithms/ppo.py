# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import torch
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey
from torchrl.collectors import Collector
from torchrl.data import (
    LazyTensorStorage,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
)
from torchrl.envs.common import EnvBase
from torchrl.modules import set_recurrent_mode
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE
from torchrl.trainers.algorithms.on_policy import OnPolicyTrainer
from torchrl.trainers.trainers import BatchSubSampler


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

    Use :meth:`from_env` to construct standard PPO components from an environment,
    actor and critic; its docstring includes a complete runnable example.

    .. note::
        This trainer requires a configurable environment setup. See the
        :class:`~torchrl.trainers.algorithms.configs` module for configuration options.
    """

    _algo_name = "PPO"
    _default_num_epochs = 4

    @classmethod
    def from_env(
        cls,
        env: EnvBase,
        *,
        actor: TensorDictModuleBase,
        critic: TensorDictModuleBase,
        total_frames: int,
        frames_per_batch: int = 1024,
        minibatch_size: int = 256,
        sub_traj_len: int | None = None,
        learning_rate: float = 3e-4,
        value_key: NestedKey = "state_value",
        loss_kwargs: Mapping[str, Any] | None = None,
        gae_kwargs: Mapping[str, Any] | None = None,
        collector_kwargs: Mapping[str, Any] | None = None,
        **trainer_kwargs: Any,
    ) -> PPOTrainer:
        """Build a PPO trainer from an environment, actor and critic.

        Constructs a :class:`~torchrl.collectors.Collector`,
        :class:`~torchrl.objectives.ClipPPOLoss`, Adam optimizer, GAE and
        minibatch sampling. Use the ordinary constructor to supply custom
        collectors, losses, optimizers or replay buffers.

        Args:
            env (EnvBase): Environment owned by the resulting collector. Training
                closes it. The collector installs missing policy primers and
                initialization tracking by default.
            actor (TensorDictModuleBase): Probabilistic actor returning actions
                and their log probabilities.
            critic (TensorDictModuleBase): Value network writing ``value_key``.
            total_frames (int): Total environment transitions to collect. For
                closed-loop action deployment these count high-level decisions.
            frames_per_batch (int, optional): Transitions collected per update,
                across all environments. Defaults to 1024.
            minibatch_size (int, optional): Transitions per optimization step.
                Clamped to the collected batch size. Defaults to 256.
            sub_traj_len (int, optional): Consecutive time steps per recurrent
                training window. When set, uses :class:`~torchrl.trainers.BatchSubSampler`
                and recurrent-mode GAE instead of flattening time into replay.
                The minibatch size must be a multiple of this length.
                Defaults to ``None`` (feedforward PPO).
            learning_rate (float, optional): Adam learning rate. Defaults to 3e-4.
            value_key (NestedKey, optional): Critic output key, also configured on
                the loss and GAE. Defaults to ``"state_value"``.
            loss_kwargs (Mapping, optional): Extra ``ClipPPOLoss`` arguments.
                Advantage normalization defaults to ``True``, or ``False`` when
                GAE already normalizes advantages (e.g. within each task).
            gae_kwargs (Mapping, optional): Extra GAE arguments. For per-task
                normalization, pass ``{"group_key": "task_id", "average_gae": True}``.
                Recurrent windows default to ``shifted=True, deactivate_vmap=True``.
            collector_kwargs (Mapping, optional): Extra ``Collector`` arguments,
                such as ``policy_device`` and ``storing_device``.
            **trainer_kwargs: Additional :class:`OnPolicyTrainer` options, such
                as ``num_epochs``, ``gamma``, ``lmbda``, logging and checkpointing.
                Action/reward keys default to the environment's unique keys.
                Done/terminated keys use the reward's namespace when available,
                otherwise the root namespace; explicit key overrides take precedence.
                ``frame_skip`` defaults to 1 and ``clip_norm`` to 1.0.

        Returns:
            PPOTrainer: Configured trainer; call :meth:`train` to start learning.

        Examples:
            >>> import torch
            >>> from tensordict.nn import TensorDictModule, NormalParamExtractor
            >>> from torchrl.modules import ProbabilisticActor, TanhNormal
            >>> from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv
            >>> env = ContinuousActionVecMockEnv()
            >>> obs_dim = env.observation_spec["observation"].shape[-1]
            >>> action_dim = env.action_spec.shape[-1]
            >>> actor = ProbabilisticActor(
            ...     TensorDictModule(
            ...         torch.nn.Sequential(torch.nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor()),
            ...         in_keys=["observation"], out_keys=["loc", "scale"],
            ...     ),
            ...     in_keys=["loc", "scale"], distribution_class=TanhNormal,
            ...     return_log_prob=True,
            ... )
            >>> critic = TensorDictModule(
            ...     torch.nn.Linear(obs_dim, 1), in_keys=["observation"], out_keys=["state_value"],
            ... )
            >>> trainer = PPOTrainer.from_env(
            ...     env, actor=actor, critic=critic, total_frames=32,
            ...     frames_per_batch=16, minibatch_size=8, progress_bar=False,
            ... )
            >>> trainer.train()

        """
        if min(total_frames, frames_per_batch, minibatch_size) < 1:
            raise ValueError("Frame and minibatch sizes must be positive.")
        num_envs = env.batch_size.numel()
        frames_per_env = math.ceil(frames_per_batch / num_envs)
        collected_batch_size = frames_per_env * num_envs
        minibatch_size = min(minibatch_size, collected_batch_size)
        if sub_traj_len is not None and (
            sub_traj_len < 1
            or sub_traj_len > frames_per_env
            or minibatch_size % sub_traj_len
        ):
            raise ValueError(
                "sub_traj_len must fit within the collected time dimension and "
                "divide minibatch_size."
            )

        if "action_key" not in trainer_kwargs:
            trainer_kwargs["action_key"] = env.action_key
        if "reward_key" not in trainer_kwargs:
            trainer_kwargs["reward_key"] = env.reward_key
        trainer_kwargs.setdefault("episode_reward_key", trainer_kwargs["reward_key"])
        reward_key = trainer_kwargs["reward_key"]
        reward_path = reward_key if isinstance(reward_key, tuple) else (reward_key,)
        for name in ("done", "terminated"):
            candidate = (*reward_path[:-1], name)
            trainer_kwargs.setdefault(
                f"{name}_key",
                candidate if candidate in env.full_done_spec.keys(True, True) else name,
            )
        trainer_kwargs.setdefault("frame_skip", 1)
        trainer_kwargs.setdefault("clip_norm", 1.0)

        gae_options = dict(gae_kwargs or {})
        gae_options.setdefault("gamma", trainer_kwargs.get("gamma", 0.99))
        gae_options.setdefault("lmbda", trainer_kwargs.get("lmbda", 0.95))
        if sub_traj_len is not None:
            gae_options.setdefault("shifted", True)
            gae_options.setdefault("deactivate_vmap", True)
        loss_options = dict(loss_kwargs or {})
        loss_options.setdefault(
            "normalize_advantage", not gae_options.get("average_gae", False)
        )
        loss = ClipPPOLoss(actor, critic, **loss_options)
        loss.set_keys(
            value=value_key,
            **{
                name: trainer_kwargs[f"{name}_key"]
                for name in ("action", "reward", "done", "terminated")
            },
        )
        gae = GAE(value_network=critic, **gae_options)
        gae.set_keys(
            value=value_key,
            **{
                name: trainer_kwargs[f"{name}_key"]
                for name in ("reward", "done", "terminated")
            },
        )
        if sub_traj_len is not None:
            gae = set_recurrent_mode(True)(gae)
        optimizer = torch.optim.Adam(loss.parameters(), lr=learning_rate)
        replay = None
        if sub_traj_len is None:
            replay = TensorDictReplayBuffer(
                storage=LazyTensorStorage(
                    collected_batch_size, device=next(loss.parameters()).device
                ),
                sampler=SamplerWithoutReplacement(),
                batch_size=minibatch_size,
            )
        collector_options = dict(collector_kwargs or {})
        collector_options.setdefault("auto_register_policy_transforms", True)
        collector = Collector(
            env,
            actor,
            total_frames=total_frames,
            frames_per_batch=frames_per_batch,
            **collector_options,
        )
        try:
            trainer = cls(
                collector=collector,
                total_frames=total_frames,
                loss_module=loss,
                optimizer=optimizer,
                gae=gae,
                replay_buffer=replay,
                optim_steps_per_batch=(
                    None
                    if replay is not None
                    else math.ceil(collected_batch_size / minibatch_size)
                ),
                **trainer_kwargs,
            )
            if sub_traj_len is not None:
                BatchSubSampler(minibatch_size, sub_traj_len=sub_traj_len).register(
                    trainer
                )
            return trainer
        except Exception:
            collector.shutdown()
            raise
