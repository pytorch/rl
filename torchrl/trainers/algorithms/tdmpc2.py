# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Sequence
from functools import partial
from numbers import Real
from typing import Literal

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from torchrl.data.replay_buffers.samplers import PrioritizedSampler, SliceSampler
from torchrl.modules import TdMpc2Planner
from torchrl.objectives import TdMpc2Loss
from torchrl.trainers.trainers import (
    _OPTIM_STEPS_UNSET,
    OptimizationStepper,
    ReplayBufferTrainer,
    Trainer,
    UpdateWeights,
)


def _validate_tdmpc2_planner(
    planner: TensorDictModuleBase,
    loss_module,
    collector=None,
) -> TdMpc2Planner:
    """Validate planner/loss/collector identity invariants."""
    if not isinstance(planner, TdMpc2Planner):
        raise TypeError(
            f"TD-MPC2 planner must be a TdMpc2Planner, got {type(planner).__name__}."
        )
    if (
        planner.world_model is not loss_module.world_model
        or planner.policy_prior is not loss_module.policy_prior
        or planner.q_ensemble is not loss_module.q_ensemble
    ):
        raise ValueError(
            "TD-MPC2 planner must reference the world_model, policy_prior, and "
            "q_ensemble owned by loss_module."
        )
    if planner.horizon != loss_module.horizon:
        raise ValueError(
            "TD-MPC2 planner horizon must match loss_module.horizon, got "
            f"{planner.horizon} and {loss_module.horizon}."
        )
    if not math.isclose(planner.discount, loss_module.discount.item(), rel_tol=1e-6):
        raise ValueError("TD-MPC2 planner discount must match loss_module.discount.")
    if collector is not None:
        collector_policy = getattr(collector, "policy", None)
        if collector_policy is not None and collector_policy is not planner:
            raise ValueError(
                "The collector policy must be the same TdMpc2Planner validated "
                "against loss_module."
            )
    return planner


class TdMpc2OptimizationStepper(OptimizationStepper):
    """Execute the two-phase TD-MPC2 learner update.

    Each step performs the model and Q-function update first, then evaluates
    the policy objective on the detached imagined latent sequence captured by
    that model update. The policy update uses the model/Q parameters after the
    first optimizer step. The target Q-functions are soft-updated last.

    Args:
        loss_module: TD-MPC2 loss providing model and actor objectives.
        optimizer_model: Optimizer for the world model and online Q-functions.
        optimizer_actor: Optimizer for the policy prior.
        target_tau: Target-Q Polyak averaging factor. Defaults to ``0.01``.
        zero_grad_set_to_none: Whether optimizer ``zero_grad`` calls set
            gradients to ``None``. Defaults to ``True``.

    .. seealso::
        :class:`~torchrl.trainers.algorithms.configs.TdMpc2OptimizationStepperConfig`,
        `TD-MPC2: Scalable, Robust World Models for Continuous Control
        <https://arxiv.org/abs/2310.16828>`_.
    """

    def __init__(
        self,
        loss_module: TdMpc2Loss,
        optimizer_model: torch.optim.Optimizer,
        optimizer_actor: torch.optim.Optimizer,
        *,
        target_tau: float = 0.01,
        zero_grad_set_to_none: bool = True,
    ) -> None:
        if not isinstance(target_tau, Real) or not math.isfinite(target_tau):
            raise ValueError(f"target_tau must be finite, got {target_tau!r}.")
        if not 0 <= target_tau <= 1:
            raise ValueError(f"target_tau must be in [0, 1], got {target_tau!r}.")

        self.loss_module = loss_module
        self.optimizer_model = optimizer_model
        self.optimizer_actor = optimizer_actor
        self.target_tau = float(target_tau)
        self.zero_grad_set_to_none = bool(zero_grad_set_to_none)
        self._update_count = 0
        self._validate_parameter_ownership()

    @property
    def update_count(self) -> int:
        """Number of completed TD-MPC2 updates."""
        return self._update_count

    @staticmethod
    def _optimizer_parameters(optimizer: torch.optim.Optimizer):
        for group in optimizer.param_groups:
            yield from group["params"]

    def register(self, trainer: Trainer, name: str = "optimization_stepper") -> None:
        """Register the stepper and validate exclusive trainer ownership."""
        if getattr(trainer, "learner_backend", "local") != "local":
            # TODO: Investigate whether this could be supported.
            raise NotImplementedError("Distributed TD-MPC2 updates are not supported.")
        if getattr(trainer, "optimizer", None) is not None:
            raise ValueError(
                "TdMpc2OptimizationStepper owns its optimizers; pass "
                "optimizer=None to Trainer."
            )
        if getattr(trainer, "target_net_updater", None) is not None:
            raise ValueError(
                "TdMpc2OptimizationStepper owns target-Q updates; pass "
                "target_net_updater=None to Trainer."
            )
        super().register(trainer, name=name)

    def _validate_parameter_ownership(self) -> None:
        model_optimizer_parameters = {
            id(parameter)
            for parameter in self._optimizer_parameters(self.optimizer_model)
        }
        actor_optimizer_parameters = {
            id(parameter)
            for parameter in self._optimizer_parameters(self.optimizer_actor)
        }
        if model_optimizer_parameters & actor_optimizer_parameters:
            raise ValueError(
                "optimizer_model and optimizer_actor must not share parameters."
            )

    @staticmethod
    def _gradient_norm(parameters: Sequence[torch.Tensor]) -> torch.Tensor:
        gradients = [
            parameter.grad.detach()
            for parameter in parameters
            if parameter.grad is not None
        ]
        if not gradients:
            raise RuntimeError("TD-MPC2 update produced no parameter gradients.")
        return (
            torch.stack([gradient.float().pow(2).sum() for gradient in gradients])
            .sum()
            .sqrt()
        )

    def _clip_gradients(
        self,
        optimizer: torch.optim.Optimizer,
        trainer: Trainer,
    ) -> torch.Tensor:
        parameters = list(self._optimizer_parameters(optimizer))
        norm = self._gradient_norm(parameters)
        clip_norm = trainer.clip_norm
        if trainer.clip_grad_norm and clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(parameters, clip_norm)
        elif clip_norm is not None:
            torch.nn.utils.clip_grad_value_(parameters, clip_norm)
        return norm

    def state_dict(self) -> dict:
        """Return optimizer state and completed-update count."""
        return {
            "update_count": self._update_count,
            "optimizer_model": self.optimizer_model.state_dict(),
            "optimizer_actor": self.optimizer_actor.state_dict(),
        }

    def load_state_dict(self, state_dict: dict) -> None:
        """Restore optimizer state and completed-update count."""
        self._update_count = int(state_dict.get("update_count", 0))
        self.optimizer_model.load_state_dict(state_dict["optimizer_model"])
        self.optimizer_actor.load_state_dict(state_dict["optimizer_actor"])

    def step(self, trainer: Trainer, sub_batch: TensorDictBase) -> TensorDictBase:
        """Run one model, actor, and target-Q update and return detached metrics."""
        if trainer.loss_module is not self.loss_module:
            raise ValueError("The trainer and stepper must share the same loss module.")
        if getattr(trainer, "process_group", None) is not None:
            # TODO: Investigate whether this can be supported.
            raise NotImplementedError("Distributed TD-MPC2 updates are not supported.")
        if getattr(trainer, "target_net_updater", None) is not None:
            raise ValueError(
                "TD-MPC2 target Q-functions are updated by the optimization "
                "stepper; do not also configure trainer.target_net_updater."
            )

        was_training = self.loss_module.training
        self.loss_module.train()
        self.optimizer_model.zero_grad(set_to_none=self.zero_grad_set_to_none)
        self.optimizer_actor.zero_grad(set_to_none=self.zero_grad_set_to_none)

        model_loss, model_metadata = self.loss_module.model_loss(sub_batch)
        model_loss.backward()
        model_grad_norm = self._clip_gradients(self.optimizer_model, trainer)
        self.optimizer_model.step()
        self.optimizer_model.zero_grad(set_to_none=self.zero_grad_set_to_none)

        actor_loss, actor_metadata = self.loss_module.actor_loss_from_latents(
            model_metadata["actor_latents"]
        )
        actor_loss.backward()
        actor_grad_norm = self._clip_gradients(self.optimizer_actor, trainer)
        self.optimizer_actor.step()
        self.optimizer_actor.zero_grad(set_to_none=self.zero_grad_set_to_none)

        self.loss_module.q_ensemble.soft_update_target(self.target_tau)
        self.loss_module.train(was_training)
        self._update_count += 1
        detached_model_loss = model_loss.detach()
        return TensorDict(
            {
                "loss_model": detached_model_loss,
                "loss_actor": actor_loss.detach(),
                "loss_consistency": (
                    self.loss_module.consistency_coef
                    * model_metadata["loss_consistency"]
                ).detach(),
                "loss_reward": (
                    self.loss_module.reward_coef * model_metadata["loss_reward"]
                ).detach(),
                "loss_value": (
                    self.loss_module.value_coef * model_metadata["loss_value"]
                ).detach(),
                "pi_entropy": actor_metadata["pi_entropy"].mean(),
                "pi_scaled_entropy": actor_metadata["pi_scaled_entropy"].mean(),
                "pi_scale": actor_metadata["pi_scale"],
                "model_grad_norm": model_grad_norm.to(
                    device=detached_model_loss.device,
                    dtype=detached_model_loss.dtype,
                ),
                "actor_grad_norm": actor_grad_norm.to(
                    device=detached_model_loss.device,
                    dtype=detached_model_loss.dtype,
                ),
            },
            batch_size=[],
        )


class TdMpc2Trainer(Trainer):
    """A trainer class for the TD-MPC2 algorithm.

    See also :class:`~torchrl.trainers.algorithms.configs.TdMpc2TrainerConfig` for the
    Hydra configuration counterpart.

    This trainer implements TD-MPC2, a scalable and robust model-based reinforcement
    learning algorithm for continuous control. TD-MPC2 learns a latent world model,
    including transition, reward, value, and policy components, and uses model
    predictive control with MPPI-style planning to select actions.

    The trainer handles:
    - Sequence-based replay buffer sampling for model learning
    - Joint optimization of the latent world model and policy prior
    - Model-predictive action planning through a TD-MPC2 planner
    - Synchronization of planner weights with the data collector
    - Logging and checkpointing of training metrics

    Args:
        collector (BaseCollector): The data collector used to gather environment interactions.
        total_frames (int): Total number of frames to collect during training.
        loss_module (TdMpc2Loss): The TD-MPC2 loss module.
        planner (TdMpc2Planner): The planner used for model-predictive action selection.
        optimizer_model (optim.Optimizer): Optimizer for the world model and value functions.
        optimizer_actor (optim.Optimizer): Optimizer for the policy prior.
        replay_buffer (ReplayBuffer): Replay buffer containing training trajectories.
        batch_size (int, optional): Number of trajectory sequences sampled per update.
        optim_steps_per_batch (int, optional): Number of optimization steps per collected batch.
        logger (Logger, optional): Logger for recording training metrics.
        progress_bar (bool, optional): Whether to show a progress bar during training.
        seed (int, optional): Random seed for reproducibility.
    """

    def __init__(
        self,
        *,
        collector,
        total_frames: int | None,
        loss_module: TdMpc2Loss,
        planner: TensorDictModuleBase | None = None,
        optimizer_model: torch.optim.Optimizer | None = None,
        optimizer_actor: torch.optim.Optimizer | None = None,
        optimization_stepper: TdMpc2OptimizationStepper | None = None,
        replay_buffer=None,
        batch_size: int | None = None,
        frame_skip: int = 1,
        optim_steps_per_batch: int = 1,
        seed_pretrain_steps: int = 0,
        logger=None,
        clip_grad_norm: bool = True,
        clip_norm: float | None = 20.0,
        progress_bar: bool = True,
        seed: int | None = None,
        save_trainer_interval: int = 10000,
        log_interval: int = 10000,
        save_trainer_file=None,
        checkpoint=None,
        checkpoint_rotation=None,
        checkpoint_metadata=None,
        num_epochs: int = 1,
        async_collection: bool = False,
        log_timings: bool = False,
        auto_log_optim_steps: bool = True,
        learner_backend: Literal["local", "ray"] = "local",
        learner_backend_options: dict | None = None,
        learner_poll_interval: float = 0.05,
    ) -> None:
        if learner_backend != "local":
            # TODO: Investigate whether this can be supported.
            raise NotImplementedError(
                "The TD-MPC2 trainer supports local optimization only."
            )
        if async_collection:
            # TODO: Investigate supporting this.
            raise NotImplementedError(
                "The TD-MPC2 trainer supports synchronous collection only."
            )
        if total_frames is None:
            try:
                total_frames = collector.total_frames
            except AttributeError as err:
                raise TypeError(
                    "total_frames must be provided when the collector does not "
                    "expose a total_frames attribute."
                ) from err
        if planner is None:
            raise ValueError("planner is required for TD-MPC2 training.")
        _validate_tdmpc2_planner(planner, loss_module, collector)
        if replay_buffer is None:
            raise ValueError("replay_buffer is required for TD-MPC2 training.")
        if isinstance(seed_pretrain_steps, bool) or not isinstance(
            seed_pretrain_steps, int
        ):
            raise TypeError("seed_pretrain_steps must be an integer.")
        if seed_pretrain_steps < 0:
            raise ValueError("seed_pretrain_steps must be non-negative.")
        if batch_size is not None and (
            isinstance(batch_size, bool) or not isinstance(batch_size, int)
        ):
            raise TypeError("batch_size must be an integer or None.")
        if batch_size is not None and batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if replay_buffer is not None:
            sampler = getattr(replay_buffer, "sampler", None)
            if isinstance(sampler, PrioritizedSampler):
                raise NotImplementedError(
                    "Prioritized replay is not supported by the TD-MPC2 trainer."
                )
            if not isinstance(sampler, SliceSampler):
                raise TypeError(
                    "replay_buffer must use a SliceSampler to preserve TD-MPC2 "
                    "sequence continuity."
                )
            if sampler.slice_len != loss_module.horizon:
                raise ValueError(
                    "The replay SliceSampler slice_len must equal "
                    f"loss_module.horizon={loss_module.horizon}."
                )
            if sampler.num_slices is not None:
                raise ValueError(
                    "The TD-MPC2 replay SliceSampler must configure slice_len, "
                    "not num_slices."
                )
            if sampler.output_layout != "batch_time":
                raise ValueError(
                    "The TD-MPC2 replay SliceSampler must use "
                    "output_layout='batch_time'."
                )
            if not sampler.strict_length or sampler.pad_output:
                raise ValueError(
                    "The TD-MPC2 replay SliceSampler must use strict_length=True "
                    "without padding."
                )
            if (
                sampler.traj_key is None
                and sampler.end_key is None
                and sampler.end_keys is None
            ):
                raise ValueError(
                    "The replay SliceSampler must define traj_key, end_key, or "
                    "end_keys to preserve episode continuity."
                )

        if optimization_stepper is None:
            if optimizer_model is None or optimizer_actor is None:
                raise TypeError(
                    "optimizer_model and optimizer_actor are required when "
                    "optimization_stepper is not provided."
                )
            optimization_stepper = TdMpc2OptimizationStepper(
                loss_module,
                optimizer_model,
                optimizer_actor,
            )
        elif optimizer_model is not None or optimizer_actor is not None:
            raise ValueError(
                "Pass optimizers directly or provide optimization_stepper, not both."
            )
        elif optimization_stepper.loss_module is not loss_module:
            raise ValueError(
                "optimization_stepper must reference the trainer's loss_module."
            )

        self.planner = planner
        super().__init__(
            collector=collector,
            total_frames=total_frames,
            frame_skip=frame_skip,
            optim_steps_per_batch=optim_steps_per_batch,
            loss_module=loss_module,
            optimizer=None,
            optimization_stepper=optimization_stepper,
            replay_buffer=replay_buffer,
            target_net_updater=None,
            batch_size=batch_size,
            learner_backend=learner_backend,
            learner_backend_options=learner_backend_options,
            learner_poll_interval=learner_poll_interval,
            logger=logger,
            clip_grad_norm=clip_grad_norm,
            clip_norm=clip_norm,
            progress_bar=progress_bar,
            seed=seed,
            save_trainer_interval=save_trainer_interval,
            log_interval=log_interval,
            save_trainer_file=save_trainer_file,
            checkpoint=checkpoint,
            checkpoint_rotation=checkpoint_rotation,
            checkpoint_metadata=checkpoint_metadata,
            num_epochs=num_epochs,
            async_collection=async_collection,
            log_timings=log_timings,
            auto_log_optim_steps=auto_log_optim_steps,
        )
        self.seed_pretrain_steps = seed_pretrain_steps
        self._tdmpc2_replay_buffer = None

        if replay_buffer is not None:
            replay_batch_size = (
                None if batch_size is None else batch_size * loss_module.horizon
            )
            replay_hook = ReplayBufferTrainer(
                replay_buffer,
                batch_size=replay_batch_size,
                # TD-MPC2 flattens collector batches explicitly below.  The
                # generic flattening fallback also rewrites the final
                # ``truncated`` entry, which is not valid for already padded
                # or mask-free collector batches.
                flatten_tensordicts=False,
                memmap=False,
                device=getattr(replay_buffer.storage, "device", "cpu"),
            )
            self.register_op("batch_process", self._extend_replay_batch)
            self.register_op("process_optim_batch", replay_hook.sample)
            self.register_module("replay_buffer", replay_hook)
            self._tdmpc2_replay_buffer = replay_hook

        if planner is not None:
            update_weights = UpdateWeights(
                self.collector,
                1,
                policy_weights_getter=partial(TensorDict.from_module, planner),
            )
            self.register_op("post_steps", update_weights)

    def _seed_pretraining_pending(self) -> bool:
        """Return whether the one-time seed-data optimization burst is pending."""
        return (
            self.seed_pretrain_steps > 0
            and self.collected_frames
            >= getattr(self.collector, "init_random_frames", 0)
            and self.optimization_stepper.update_count < self.seed_pretrain_steps
        )

    def optim_steps(
        self,
        batch: TensorDictBase,
        *,
        optim_steps_per_batch: int | None | object = _OPTIM_STEPS_UNSET,
        num_epochs: int | object = _OPTIM_STEPS_UNSET,
    ) -> None:
        """Run seed-data pretraining once, then use the normal cadence."""
        warmup_frames = getattr(self.collector, "init_random_frames", 0)
        if (
            self.seed_pretrain_steps > self.optimization_stepper.update_count
            and self.collected_frames < warmup_frames
        ):
            raise RuntimeError(
                "TD-MPC2 seed pretraining cannot start before "
                f"collector.init_random_frames ({warmup_frames}) is reached."
            )
        if (
            optim_steps_per_batch is not _OPTIM_STEPS_UNSET
            or num_epochs is not _OPTIM_STEPS_UNSET
        ):
            super().optim_steps(
                batch,
                optim_steps_per_batch=optim_steps_per_batch,
                num_epochs=num_epochs,
            )
            return
        if self._seed_pretraining_pending():
            completed_before = self.optimization_stepper.update_count
            remaining_steps = self.seed_pretrain_steps - completed_before
            super().optim_steps(
                batch,
                optim_steps_per_batch=remaining_steps,
                num_epochs=1,
            )
            completed_after = self.optimization_stepper.update_count
            if completed_after != self.seed_pretrain_steps:
                raise RuntimeError(
                    "TD-MPC2 seed pretraining stopped before completing the "
                    f"requested {self.seed_pretrain_steps} updates: completed "
                    f"{completed_after}."
                )
            return
        super().optim_steps(batch)

    def _checkpoint_policy(self) -> TensorDictModuleBase | None:
        """Return the configured execution policy for trainer checkpoints."""
        if self.planner is not None:
            return self.planner
        return super()._checkpoint_policy()

    def _extend_replay_batch(self, batch: TensorDictBase) -> TensorDictBase:
        """Flatten canonical collector transitions for flat replay storage."""
        replay_hook = self._tdmpc2_replay_buffer
        if replay_hook is None:
            raise RuntimeError("TD-MPC2 replay buffer hook is not initialized.")
        mask = batch.get(("collector", "mask"), default=None)
        if mask is not None:
            batch = batch[mask]
        else:
            batch = batch.reshape(-1)
        return replay_hook.extend(batch)
