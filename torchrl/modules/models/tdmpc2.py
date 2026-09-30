# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import math
from collections.abc import Sequence
from copy import deepcopy
from numbers import Real
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from tensordict import from_modules, NestedKey, TensorDictBase
from tensordict.nn import TensorDictModuleBase, TensorDictParams

from ..functional import symexp


class SimplicialNormalization(nn.Module):
    """Apply softmax independently to fixed-size feature simplices.

    Args:
        dim (int): Number of features in each simplex. The last input
            dimension must be divisible by ``dim``.
    """

    def __init__(self, dim: int):
        super().__init__()
        if isinstance(dim, bool) or not isinstance(dim, int) or dim <= 0:
            raise ValueError(f"dim must be a positive integer, got {dim!r}.")
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize the last dimension of ``x`` in groups of ``dim``."""
        if x.shape[-1] % self.dim:
            raise ValueError(
                f"The last input dimension must be divisible by dim={self.dim}, "
                f"got {x.shape[-1]}."
            )
        shape = x.shape
        x = x.reshape(*shape[:-1], -1, self.dim)
        x = F.softmax(x, dim=-1)
        return x.reshape(shape)


def log_std(
    x: torch.Tensor,
    low: torch.Tensor | float,
    dif: torch.Tensor | float,
) -> torch.Tensor:
    """Map unconstrained values to a bounded log-standard deviation.

    Args:
        x (torch.Tensor): Unconstrained values.
        low (torch.Tensor or float): Lower bound of the output interval.
        dif (torch.Tensor or float): Width of the output interval.

    Returns:
        A tensor with values in ``[low, low + dif]``.
    """
    return low + 0.5 * dif * (torch.tanh(x) + 1)


def gaussian_logprob(eps: torch.Tensor, log_std: torch.Tensor) -> torch.Tensor:
    """Compute the log probability of diagonal Gaussian samples.

    Args:
        eps (torch.Tensor): Gaussian noise samples.
        log_std (torch.Tensor): Log standard deviations for each action.

    Returns:
        The summed log probability with a trailing singleton event dimension.
    """
    return (-0.5 * eps.pow(2) - log_std - 0.9189385175704956).sum(-1, keepdim=True)


def _squash(
    mu: torch.Tensor,
    pi: torch.Tensor,
    log_pi: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Squash policy locations and samples while correcting log probability."""
    mu = torch.tanh(mu)
    pi = torch.tanh(pi)
    squashed_pi = torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
    log_pi = log_pi - squashed_pi.sum(-1, keepdim=True)
    return mu, pi, log_pi


class _TdMpc2PolicyPrior(nn.Module):
    """Sample actions from the TD-MPC2 policy prior."""

    def __init__(
        self,
        network: nn.Module,
        log_std_min: float,
        log_std_max: float,
        device: torch.device | str | None = None,
    ):
        super().__init__()
        self.network = network
        self.register_buffer("log_std_min", torch.tensor(log_std_min, device=device))
        self.register_buffer(
            "log_std_dif",
            torch.tensor(log_std_max, device=device) - self.log_std_min,
        )

    def forward(
        self, latent: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, raw_log_std = self.network(latent).chunk(2, dim=-1)
        bounded_log_std = log_std(raw_log_std, self.log_std_min, self.log_std_dif)
        eps = torch.randn_like(mean)
        log_prob = gaussian_logprob(eps, bounded_log_std)
        scaled_log_prob = log_prob * eps.shape[-1]

        action = mean + eps * bounded_log_std.exp()
        mean, action, log_prob = _squash(mean, action, log_prob)

        entropy = -log_prob
        scaled_entropy = -scaled_log_prob
        return (
            action,
            mean,
            bounded_log_std,
            entropy,
            scaled_entropy,
        )


def symlog_two_hot_decode(
    logits: torch.Tensor,
    vmin: float,
    vmax: float,
    num_bins: int,
) -> torch.Tensor:
    """Decode TD-MPC2 categorical logits into scalar values.

    Args:
        logits (torch.Tensor): Unnormalized categorical predictions with
            categories in the final dimension.
        vmin (float): Lower endpoint of the symlog-space categorical support.
        vmax (float): Upper endpoint of the symlog-space categorical support.
        num_bins (int): Number of categorical support points.

    Returns:
        Decoded scalar values with a trailing singleton event dimension.
    """
    if num_bins <= 1:
        raise ValueError(f"num_bins must be greater than 1, got {num_bins}.")
    if vmax <= vmin:
        raise ValueError("vmax must be greater than vmin.")
    support = torch.linspace(
        vmin,
        vmax,
        num_bins,
        dtype=logits.dtype,
        device=logits.device,
    )
    value = (torch.softmax(logits, dim=-1) * support).sum(-1, keepdim=True)
    return symexp(value)


def _normalize_tdmpc2_key(key: NestedKey | list[str]) -> NestedKey:
    if isinstance(key, list):
        return tuple(key)
    return key


class TdMpc2QEnsemble(TensorDictModuleBase):
    """Vectorized TD-MPC2 ensemble of distributional Q-functions.

    Each Q-function receives the concatenation of a latent state and an action
    and returns logits for a scalar categorical representation. The ensemble
    dimension is exposed immediately before the category dimension.

    :meth:`forward` returns the logits of every Q-function. :meth:`reduce`
    follows TD-MPC2's value-estimation rule by selecting two Q-functions at
    random, decoding their logits, and taking either their minimum or average.
    Online, detached, and target parameter sources are available for both
    operations.

    Args:
        q_networks: Sequence of identically shaped single-network Q-functions.
        num_bins: Number of categorical bins. Must be greater than 1.
        vmin: Minimum value of the symlog-space categorical support.
        vmax: Maximum value of the symlog-space categorical support.
        in_keys: Two TensorDict keys for the latent state and action.
            Defaults to ``["latent", "action"]``.
        out_keys: One TensorDict key for the ensemble logits. Defaults to
            ``["q_logits"]``.
        q_value_key: TensorDict key written by :meth:`reduce`. Defaults to
            ``"q_value"``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.modules import MLP
        >>> from torchrl.modules.models.tdmpc2 import TdMpc2QEnsemble
        >>> q_networks = [
        ...     MLP(in_features=6, out_features=5, depth=2, num_cells=8)
        ...     for _ in range(5)
        ... ]
        >>> q_ensemble = TdMpc2QEnsemble(
        ...     q_networks, num_bins=5, vmin=-10.0, vmax=10.0
        ... )
        >>> data = TensorDict(
        ...     {"latent": torch.randn(4, 4), "action": torch.randn(4, 2)},
        ...     batch_size=[4],
        ... )
        >>> data = q_ensemble(data)
        >>> data["q_logits"].shape
        torch.Size([4, 5, 5])
        >>> q_ensemble.reduce(data, reduction="min")["q_value"].shape
        torch.Size([4, 1])
    """

    def __init__(
        self,
        q_networks: Sequence[nn.Module],
        *,
        num_bins: int,
        vmin: float,
        vmax: float,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        q_value_key: NestedKey = "q_value",
    ) -> None:
        super().__init__()
        q_networks = tuple(q_networks)
        if len(q_networks) < 2:
            raise ValueError("TdMpc2QEnsemble requires at least two Q-networks.")
        if num_bins <= 1:
            raise ValueError("num_bins must be greater than 1.")
        if vmax <= vmin:
            raise ValueError("vmax must be greater than vmin.")

        self.in_keys = (
            ["latent", "action"]
            if in_keys is None
            else [_normalize_tdmpc2_key(key) for key in in_keys]
        )
        self.out_keys = (
            ["q_logits"]
            if out_keys is None
            else [_normalize_tdmpc2_key(key) for key in out_keys]
        )
        if len(self.in_keys) != 2:
            raise ValueError("TdMpc2QEnsemble requires exactly two input keys.")
        if len(self.out_keys) != 1:
            raise ValueError("TdMpc2QEnsemble requires exactly one output key.")
        if len(set(self.in_keys)) != 2:
            raise ValueError("TdMpc2QEnsemble input keys must be distinct.")
        if self.out_keys[0] in self.in_keys:
            raise ValueError("The output key must be distinct from the input keys.")
        if isinstance(q_value_key, list):
            q_value_key = _normalize_tdmpc2_key(q_value_key)
            if len(q_value_key) == 1:
                q_value_key = q_value_key[0]
        if q_value_key in self.in_keys or q_value_key in self.out_keys:
            raise ValueError(
                "q_value_key must be distinct from the input and output keys."
            )

        self.num_q = len(q_networks)
        self.num_bins = num_bins
        self.vmin = vmin
        self.vmax = vmax
        self.q_value_key = q_value_key

        self.q_params = from_modules(*q_networks, as_module=True)
        self.target_q_params = TensorDictParams(
            self.q_params.data.clone(), no_convert=True
        )

        with self.q_params[0].data.to("meta").to_module(q_networks[0]):
            q_template = deepcopy(q_networks[0])
            target_q_template = deepcopy(q_networks[0])
        self.__dict__["_q_template"] = q_template
        self.__dict__["_target_q_template"] = target_q_template
        self._q_template.train(self.training)
        self._target_q_template.train(False)

    @staticmethod
    def _call(
        params: TensorDictBase,
        module: nn.Module,
        inputs: torch.Tensor,
    ) -> torch.Tensor:
        with params.to_module(module, preserve_module_state=False):
            return module(inputs)

    def _source(
        self, source: Literal["online", "detached", "target"]
    ) -> tuple[TensorDictBase, nn.Module]:
        if source == "online":
            return self.q_params, self._q_template
        if source == "detached":
            return self.q_params.detach(), self._q_template
        if source == "target":
            return self.target_q_params, self._target_q_template
        raise ValueError(
            f"source must be one of 'online', 'detached', or 'target', got {source!r}."
        )

    def _q_logits(
        self,
        latent: torch.Tensor,
        action: torch.Tensor,
        source: Literal["online", "detached", "target"],
    ) -> torch.Tensor:
        params, module = self._source(source)
        inputs = torch.cat((latent, action), dim=-1)
        return torch.vmap(
            self._call,
            in_dims=(0, None, None),
            randomness="different",
        )(params, module, inputs)

    def _inputs(self, tensordict: TensorDictBase) -> tuple[torch.Tensor, torch.Tensor]:
        values = tuple(tensordict.get(key) for key in self.in_keys)
        if any(value is None for value in values):
            missing = [key for key, value in zip(self.in_keys, values) if value is None]
            raise KeyError(f"Missing TD-MPC2 Q ensemble input keys: {missing}")
        return values[0], values[1]

    def forward(
        self,
        tensordict: TensorDictBase,
        *,
        source: Literal["online", "detached", "target"] = "online",
    ) -> TensorDictBase:
        """Write the logits of all Q-functions to the input TensorDict.

        Args:
            tensordict: TensorDict containing the latent state and action.
            source: Parameter source used for the Q-functions. Defaults to
                ``"online"``.
        """
        latent, action = self._inputs(tensordict)
        logits = self._q_logits(latent, action, source)
        tensordict.set(self.out_keys_source[0], logits.movedim(0, -2))
        return tensordict

    def reduce(
        self,
        tensordict: TensorDictBase,
        *,
        reduction: Literal["min", "avg"],
        source: Literal["online", "detached", "target"] = "online",
    ) -> TensorDictBase:
        """Write a TD-MPC2 two-Q value reduction to the input TensorDict.

        Args:
            tensordict: TensorDict containing the latent state and action.
            reduction: Either ``"min"`` or ``"avg"`` for the two sampled
                Q-functions.
            source: Parameter source used for the Q-functions. Defaults to
                ``"online"``.
        """
        if reduction not in {"min", "avg"}:
            raise ValueError(
                f"reduction must be either 'min' or 'avg', got {reduction!r}."
            )
        latent, action = self._inputs(tensordict)
        logits = self._q_logits(latent, action, source)
        qidx = torch.randperm(self.num_q, device=logits.device)[:2]
        values = symlog_two_hot_decode(
            logits[qidx],
            self.vmin,
            self.vmax,
            self.num_bins,
        )
        if reduction == "min":
            q_value = values.min(dim=0).values
        else:
            q_value = values.sum(dim=0) / 2
        tensordict.set(self.q_value_key, q_value)
        return tensordict

    @torch.no_grad()
    def soft_update_target(self, tau: float) -> None:
        """Update target Q-function buffers by Polyak averaging.

        Args:
            tau: Interpolation factor in ``[0, 1]``. A value of ``0`` keeps
                the target unchanged and a value of ``1`` copies the online
                parameters.
        """
        if not isinstance(tau, Real) or not math.isfinite(tau) or not 0 <= tau <= 1:
            raise ValueError(f"tau must be a finite value in [0, 1], got {tau!r}.")
        for key, target in self.target_q_params.data.items(True, True):
            source = self.q_params.data.get(key)
            if torch.is_floating_point(target) or torch.is_complex(target):
                target.lerp_(source, tau)
            else:
                target.copy_(source)

    def train(self, mode: bool = True):
        """Set training mode while keeping the target Q-functions in eval mode."""
        super().train(mode)
        self._q_template.train(mode)
        self._target_q_template.train(False)
        return self
