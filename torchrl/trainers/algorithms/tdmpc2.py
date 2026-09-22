"""TD-MPC2 trainer optimization components."""

from __future__ import annotations

import math
from collections.abc import Sequence
from numbers import Real

import torch
from tensordict import TensorDict, TensorDictBase
from torchrl.objectives import TdMpc2Loss
from torchrl.trainers.trainers import OptimizationStepper, Trainer

__all__ = ["TdMpc2OptimizationStepper"]


class TdMpc2OptimizationStepper(OptimizationStepper):
    """Execute the two-phase TD-MPC2 learner update.

    The model optimizer updates the world model and online Q-functions first;
    the actor optimizer then updates the policy objective on the detached
    imagined latent sequence captured by that model update. The policy update
    uses the model/Q parameters after the first optimizer step. The target
    Q-functions are soft-updated last.

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
        """Run model, actor, and target-Q updates and return detached metrics."""
        if trainer.loss_module is not self.loss_module:
            raise ValueError("The trainer and stepper must share the same loss module.")
        if getattr(trainer, "process_group", None) is not None:
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
