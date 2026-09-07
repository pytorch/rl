# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch


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
