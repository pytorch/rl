# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any, Literal, TYPE_CHECKING

import torch
from tensordict import TensorDictBase
from tensordict.nn import CudaGraphModule

from torchrl.checkpoint import GlobalRNGState
from torchrl.objectives.dreamer_v3 import DreamerV3Loss
from torchrl.objectives.utils import TargetNetUpdater
from torchrl.trainers.trainers import OptimizationStepper

if TYPE_CHECKING:
    from torchrl.trainers import Trainer


class DreamerV3Optimizer(torch.optim.Optimizer):
    """DreamerV3 adaptive gradient clipping, RMS scaling and momentum.

    Clips each parameter's gradient by its parameter norm, normalizes it by a
    bias-corrected moving RMS, and applies bias-corrected momentum. A linear
    learning-rate warm-up starts at zero on the first step when enabled.
    Moment estimates are accumulated in float32. Parameters with no gradient
    are skipped; an update with no parameter gradients raises ``RuntimeError``.

    Reference: Hafner et al., "Mastering Diverse Domains through World Models"
    (2023), https://arxiv.org/abs/2301.04104.

    See also :class:`~torchrl.trainers.algorithms.configs.DreamerV3OptimizerConfig`.

    Args:
        parameters (iterable of Tensor or dict): Parameters to optimize, or
            parameter-group dictionaries. Group options override the defaults
            below; each group maintains its own update counter.

    Keyword Args:
        lr (float, optional): Learning rate after warm-up. Default: ``4e-5``.
        agc (float, optional): Maximum gradient norm as a fraction of the
            clamped parameter norm. Zero disables clipping. Default: ``0.3``.
        parameter_norm_min (float, optional): Lower bound on parameter norms
            used for clipping. Default: ``1e-3``.
        beta1 (float, optional): Decay of normalized-gradient momentum.
            Default: ``0.9``.
        beta2 (float, optional): Decay of the squared-gradient average.
            Default: ``0.999``.
        eps (float, optional): Added to the RMS denominator. Default: ``1e-20``.
        warmup_steps (int, optional): Number of updates before the full learning
            rate is reached. Zero disables warm-up. Default: ``1000``.

    Examples:
        >>> import torch
        >>> from torchrl.trainers.algorithms import DreamerV3Optimizer
        >>> parameter = torch.nn.Parameter(torch.tensor([1.0, -1.0]))
        >>> optimizer = DreamerV3Optimizer([parameter], lr=0.01, warmup_steps=0)
        >>> parameter.square().sum().backward()
        >>> optimizer.step()
        >>> bool((parameter.abs() < 1).all())
        True
        >>> optimizer.zero_grad(set_to_none=False)
        >>> checkpoint = optimizer.state_dict()
        >>> optimizer.load_state_dict(checkpoint)
    """

    def __init__(
        self,
        parameters: Iterable[torch.Tensor] | Iterable[dict[str, Any]],
        *,
        lr: float = 4e-5,
        agc: float = 0.3,
        parameter_norm_min: float = 1e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-20,
        warmup_steps: int = 1000,
    ):
        super().__init__(
            parameters,
            {
                "lr": lr,
                "agc": agc,
                "parameter_norm_min": parameter_norm_min,
                "beta1": beta1,
                "beta2": beta2,
                "eps": eps,
                "warmup_steps": warmup_steps,
                "step": 0,
            },
        )

    @torch.no_grad()
    def step(
        self, closure: Callable[[], torch.Tensor] | None = None
    ) -> torch.Tensor | None:
        """Update parameters with gradients and return the optional closure loss.

        Args:
            closure (callable, optional): Re-evaluates the model, computes
                gradients, and returns its loss. Default: ``None``.

        Returns:
            The closure's loss, or ``None`` when no closure is supplied.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        if not any(
            parameter.grad is not None
            for group in self.param_groups
            for parameter in group["params"]
        ):
            raise RuntimeError("DreamerV3 optimizer received no parameter gradients.")

        for group in self.param_groups:
            group["step"] += 1
            step = group["step"]
            warmup_steps = group["warmup_steps"]
            schedule_step = step - 1
            warmup = min(1.0, schedule_step / warmup_steps) if warmup_steps else 1.0
            learning_rate = group["lr"] * warmup

            # Group by device and dtype for the multi-tensor kernels.
            buckets: dict[
                tuple[torch.device, torch.dtype], list[torch.nn.Parameter]
            ] = {}
            for parameter in group["params"]:
                if parameter.grad is not None:
                    buckets.setdefault((parameter.device, parameter.dtype), []).append(
                        parameter
                    )

            for parameters in buckets.values():
                gradients = [parameter.grad.float() for parameter in parameters]
                if group["agc"]:
                    gradient_norms = list(torch._foreach_norm(gradients))
                    parameter_norms = list(
                        torch._foreach_norm(
                            [parameter.detach().float() for parameter in parameters]
                        )
                    )
                    torch._foreach_clamp_min_(
                        parameter_norms, group["parameter_norm_min"]
                    )
                    maximum_norms = torch._foreach_mul(parameter_norms, group["agc"])
                    gradient_denominators = torch._foreach_maximum(
                        gradient_norms, maximum_norms
                    )
                    gradient_scales = torch._foreach_div(
                        maximum_norms, gradient_denominators
                    )
                    gradients = list(torch._foreach_mul(gradients, gradient_scales))

                rms = []
                momentum = []
                for parameter in parameters:
                    state = self.state[parameter]
                    if not state:
                        state["rms"] = torch.zeros_like(parameter, dtype=torch.float32)
                        state["momentum"] = torch.zeros_like(
                            parameter, dtype=torch.float32
                        )
                    rms.append(state["rms"])
                    momentum.append(state["momentum"])
                beta1 = group["beta1"]
                beta2 = group["beta2"]
                torch._foreach_mul_(rms, beta2)
                torch._foreach_addcmul_(rms, gradients, gradients, value=1 - beta2)
                rms_hat = torch._foreach_div(rms, 1 - beta2**step)
                rms_denominator = torch._foreach_sqrt(rms_hat)
                torch._foreach_add_(rms_denominator, group["eps"])
                normalized = torch._foreach_div(gradients, rms_denominator)
                torch._foreach_mul_(momentum, beta1)
                torch._foreach_add_(momentum, normalized, alpha=1 - beta1)
                momentum_hat = torch._foreach_div(momentum, 1 - beta1**step)
                if parameters[0].dtype != torch.float32:
                    momentum_hat = [
                        update.to(parameter.dtype)
                        for update, parameter in zip(momentum_hat, parameters)
                    ]
                torch._foreach_add_(parameters, momentum_hat, alpha=-learning_rate)
        return loss


class DreamerV3OptimizationStepper(OptimizationStepper):
    """Execute a complete DreamerV3 forward/backward and optimizer update.

    One optional compile scope owns all shared loss modules. CUDA graph capture
    covers forward/backward only; optimizer and target updates run afterwards.
    Call :meth:`warmup` with a representative replay sample before starting
    collection when compilation or capture is enabled. Warm-up preserves model
    buffers and global RNG state and never advances the optimizer or targets.
    Returned scalar metrics and posterior features written to the input's
    ``replay_context`` key retain their values after later captured updates.

    Args:
        loss_module (DreamerV3Loss): Complete learner objective.
        optimizer (torch.optim.Optimizer): Optimizer owning the shared learner
            parameters once each.
        target_updater (TargetNetUpdater, optional): Target update performed
            after each optimizer step. Default: ``None``.

    Keyword Args:
        compile_train_step (bool, optional): Compile the complete
            forward/backward pass. Requires PyTorch's
            ``torch._dynamo.config.inline_inbuilt_nn_modules`` support to be
            enabled for functional parameter contexts. Default: ``False``.
        compile_mode (str, optional): PyTorch compile mode. Default: ``"default"``.
        cudagraph (bool, optional): Capture forward/backward on CUDA.
            Default: ``False``.
        warmup_steps (int, optional): Representative forward/backward calls
            before training. Must be positive. Default: ``5``.
        mixed_precision (bool, optional): Use bfloat16 autocast for CUDA
            forward/backward. Default: ``False``.

    .. note::
        Shared modules must not also have an independently compiled execution
        scope. Pause collection and synchronize pending replay operations before
        warm-up or checkpointing. Distributed execution is outside this stepper's
        supported modes.

    Examples:
        Continue from the runnable :class:`~torchrl.objectives.DreamerV3Loss`
        example, which constructs ``loss_module``, ``target_updater`` and
        ``sample`` from public components:

        >>> from torchrl.trainers.algorithms import (
        ...     DreamerV3OptimizationStepper, DreamerV3Optimizer,
        ... )
        >>> optimizer = DreamerV3Optimizer(loss_module.parameters(), warmup_steps=0)
        >>> stepper = DreamerV3OptimizationStepper(
        ...     loss_module, optimizer, target_updater, warmup_steps=1,
        ... )
        >>> stepper.warmup(sample)
        >>> before = [parameter.detach().clone() for parameter in loss_module.parameters()]
        >>> metrics = stepper.step(None, sample)
        >>> assert any(
        ...     not torch.equal(parameter, previous)
        ...     for parameter, previous in zip(loss_module.parameters(), before)
        ... )
        >>> assert not sample["replay_context", "state"].requires_grad

    See also :class:`~torchrl.trainers.algorithms.configs.DreamerV3OptimizationStepperConfig`.
    """

    def __init__(
        self,
        loss_module: DreamerV3Loss,
        optimizer: torch.optim.Optimizer,
        target_updater: TargetNetUpdater | None = None,
        *,
        compile_train_step: bool = False,
        compile_mode: Literal[
            "default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"
        ] = "default",
        cudagraph: bool = False,
        warmup_steps: int = 5,
        mixed_precision: bool = False,
    ):
        if warmup_steps < 1:
            raise ValueError("warmup_steps must be positive.")
        if compile_train_step and not getattr(
            torch._dynamo.config, "inline_inbuilt_nn_modules", False
        ):
            raise RuntimeError(
                "Whole-step compilation requires a PyTorch runtime with "
                "torch._dynamo.config.inline_inbuilt_nn_modules enabled."
            )
        self.loss_module = loss_module
        self.optimizer = optimizer
        self.target_updater = target_updater
        self.compile_train_step = compile_train_step
        self.compile_mode = compile_mode
        self.cudagraph = cudagraph
        self.warmup_steps = warmup_steps
        self.mixed_precision = mixed_precision
        self._ready = not (compile_train_step or cudagraph)
        self._train_step = self._forward_backward

    def _prepare_sample(self, sample: TensorDictBase) -> TensorDictBase:
        sample = sample.select(*self.loss_module.in_keys, strict=False)
        for key in (
            self.loss_module.tensor_keys.is_init,
            ("next", self.loss_module.value_loss.tensor_keys.done),
            ("next", self.loss_module.value_loss.tensor_keys.terminated),
        ):
            value = sample.get(key, None)
            if value is not None:
                sample.set(key, value.reshape(*sample.batch_size, 1))
        return sample

    def _forward_backward(self, sample: TensorDictBase) -> TensorDictBase:
        reference = sample.get(self.loss_module.in_keys[0])
        with torch.autocast(
            device_type=reference.device.type,
            dtype=torch.bfloat16,
            enabled=self.mixed_precision and reference.device.type == "cuda",
        ):
            losses = self.loss_module(sample)
            total = sum(
                value
                for key, value in losses.items()
                if isinstance(key, str) and key.startswith("loss_")
            )
        self.optimizer.zero_grad(set_to_none=False)
        total.backward()
        return losses.detach()

    def warmup(self, sample: TensorDictBase) -> None:
        """Prepare execution using representative data without training updates.

        Args:
            sample (TensorDictBase): Sample with the shape, keys, dtype and
                device used for subsequent updates. Collection and replay
                operations must be quiescent for CUDA capture.
        """
        sample = self._prepare_sample(sample)
        reference = sample.get(self.loss_module.in_keys[0])
        if self.cudagraph and reference.device.type != "cuda":
            raise RuntimeError("CUDA graph learner updates require CUDA inputs.")
        self._ready = False
        train_step = self._forward_backward
        if self.compile_train_step:
            train_step = torch.compile(train_step, mode=self.compile_mode)
        if self.cudagraph:
            train_step = CudaGraphModule(
                train_step, warmup=self.warmup_steps, device=reference.device
            )
        rng = GlobalRNGState()
        rng_state = rng.state_dict()
        buffers = [
            (buffer, buffer.detach().clone()) for buffer in self.loss_module.buffers()
        ]
        try:
            for _ in range(self.warmup_steps):
                train_step(sample)
            self.optimizer.zero_grad(set_to_none=False)
        finally:
            with torch.no_grad():
                for buffer, saved in buffers:
                    buffer.copy_(saved)
            rng.load_state_dict(rng_state)
        self._train_step = train_step
        self._ready = True

    def step(
        self, trainer: Trainer | None, sub_batch: TensorDictBase
    ) -> TensorDictBase:
        """Update learner parameters and targets, returning detached metrics.

        Args:
            trainer (Trainer or None): Owning trainer, or ``None`` for a custom
                training loop. An owning trainer must use this stepper's loss.
            sub_batch (TensorDictBase): Real transition sequences with the
                schema supplied to :meth:`warmup` when capture is enabled.
                Detached posterior features are written under the loss's
                configured ``replay_context`` key for subsequent replay updates.

        Returns:
            Detached scalar metrics suitable for Trainer logging.
        """
        if trainer is not None:
            if trainer.loss_module is not self.loss_module:
                raise ValueError(
                    "The trainer and stepper must share the same loss module."
                )
            if getattr(trainer, "process_group", None) is not None:
                raise NotImplementedError(
                    "Distributed DreamerV3 updates are not supported."
                )
        if not self._ready:
            raise RuntimeError(
                "Call warmup(sample) before compiled or captured learner updates."
            )
        result = self._train_step(self._prepare_sample(sub_batch))
        if not any(
            parameter.grad is not None
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ):
            raise RuntimeError("The learner update produced no parameter gradients.")
        self.optimizer.step()
        if self.target_updater is not None:
            self.target_updater.step()
        # CudaGraphModule returns owned outputs for this callable, whose result
        # is distinct from its input. Keep that ownership through the batch view.
        sub_batch.set(
            self.loss_module.tensor_keys.replay_context,
            result.get(self.loss_module.tensor_keys.replay_context),
        )
        return result.select(
            *(key for key, value in result.items() if isinstance(value, torch.Tensor))
        )

    def state_dict(self) -> dict[str, Any]:
        """Return optimizer and target-update progress; checkpoint the loss separately."""
        state = {"optimizer": self.optimizer.state_dict()}
        if self.target_updater is not None:
            state["target_updater"] = self.target_updater.state_dict()
        return state

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Restore optimizer state without replacing captured gradient buffers."""
        self.optimizer.load_state_dict(state_dict["optimizer"])
        if self.target_updater is not None:
            self.target_updater.load_state_dict(state_dict["target_updater"])
