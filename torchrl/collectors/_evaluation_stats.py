# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Bootstrap statistics for comparing evaluation results."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from tensordict import TensorDict


def _mean(values: torch.Tensor) -> torch.Tensor:
    return values.mean(-1)


def _median(values: torch.Tensor) -> torch.Tensor:
    return values.quantile(0.5, dim=-1)


def _interquartile_mean(values: torch.Tensor) -> torch.Tensor:
    num_samples = values.shape[-1]
    trimmed = int(0.25 * num_samples)
    ordered = values.sort(dim=-1).values
    return ordered[..., trimmed : num_samples - trimmed].mean(-1)


_STATISTICS: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
    "mean": _mean,
    "median": _median,
    "iqm": _interquartile_mean,
}


def bootstrap_estimate(
    values: torch.Tensor,
    dim: int = -1,
    *,
    statistic: Literal["mean", "median", "iqm"] = "iqm",
    confidence: float = 0.95,
    num_resamples: int = 1000,
    generator: torch.Generator | None = None,
) -> TensorDict:
    """Reduce ``values`` along ``dim`` to an estimate with a percentile bootstrap interval.

    Args:
        values (torch.Tensor): the samples, e.g. per-episode returns.
        dim (int, optional): the dimension holding the samples. Defaults to ``-1``.

    Keyword Args:
        statistic (str, optional): ``"mean"``, ``"median"`` or ``"iqm"``
            (the mean of the samples left after dropping the lowest and
            highest 25%). Defaults to ``"iqm"``.
        confidence (float, optional): the coverage of the interval, in
            ``(0, 1)``. Defaults to ``0.95``.
        num_resamples (int, optional): the number of bootstrap resamples,
            each drawn with replacement. Defaults to ``1000``.
        generator (torch.Generator, optional): the generator used to draw
            resamples, for reproducible intervals. Defaults to ``None``.

    Returns:
        a :class:`~tensordict.TensorDict` with entries ``"estimate"``,
        ``"ci_low"`` and ``"ci_high"`` whose batch size is the shape of
        ``values`` without ``dim``.

    Examples:
        >>> import torch
        >>> from torchrl.collectors import bootstrap_estimate
        >>> returns = torch.tensor([[0.0, 0.0, 0.0, 100.0], [5.0, 5.0, 5.0, 5.0]])
        >>> summary = bootstrap_estimate(returns, statistic="iqm")
        >>> summary["estimate"]
        tensor([0., 5.])
        >>> summary["ci_low"][1], summary["ci_high"][1]
        (tensor(5.), tensor(5.))
    """
    if statistic not in _STATISTICS:
        raise ValueError(
            f"Unknown statistic {statistic!r}. Choose among {sorted(_STATISTICS)}."
        )
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}.")
    reduce = _STATISTICS[statistic]
    samples = values.movedim(dim, -1).to(torch.get_default_dtype())
    num_samples = samples.shape[-1]
    indices = torch.randint(
        num_samples,
        (num_resamples, num_samples),
        generator=generator,
        device=generator.device if generator is not None else samples.device,
    ).to(samples.device)
    alpha = (1 - confidence) / 2
    bounds = reduce(samples[..., indices]).quantile(
        torch.tensor([alpha, 1 - alpha], dtype=samples.dtype, device=samples.device),
        dim=-1,
    )
    return TensorDict(
        estimate=reduce(samples),
        ci_low=bounds[0],
        ci_high=bounds[1],
        batch_size=samples.shape[:-1],
    )


def paired_comparison(
    first: torch.Tensor,
    second: torch.Tensor,
    dim: int = -1,
    *,
    statistic: Literal["mean", "median", "iqm"] = "mean",
    confidence: float = 0.95,
    num_resamples: int = 1000,
    generator: torch.Generator | None = None,
) -> TensorDict:
    """Compare two sets of paired samples, e.g. two policies evaluated on the same seeds.

    Sample ``i`` of ``first`` and ``second`` must come from the same
    conditions, as with :meth:`~torchrl.collectors.Evaluator.evaluate` called
    with the same ``seed`` and ``episode_sampling="per_env"``.

    Args:
        first (torch.Tensor): the samples of the first candidate.
        second (torch.Tensor): the samples of the second candidate, with the
            same shape as ``first``.
        dim (int, optional): the dimension holding the samples. Defaults to ``-1``.

    Keyword Args:
        statistic (str, optional): the statistic of the paired differences
            ``first - second``: ``"mean"``, ``"median"`` or ``"iqm"``.
            Defaults to ``"mean"``.
        confidence (float, optional): see :func:`bootstrap_estimate`.
        num_resamples (int, optional): see :func:`bootstrap_estimate`.
        generator (torch.Generator, optional): see :func:`bootstrap_estimate`.

    Returns:
        the output of :func:`bootstrap_estimate` on ``first - second``, plus
        ``"probability_of_improvement"``: the fraction of pairs where
        ``first`` is larger, ties counting as half.

    Examples:
        >>> import torch
        >>> from torchrl.collectors import paired_comparison
        >>> first = torch.tensor([3.0, 5.0, 4.0, 6.0])
        >>> result = paired_comparison(first, first - 1)
        >>> result["estimate"], result["probability_of_improvement"]
        (tensor(1.), tensor(1.))
    """
    if first.shape != second.shape:
        raise ValueError(
            f"first and second must have the same shape, got {first.shape} and {second.shape}."
        )
    result = bootstrap_estimate(
        first - second,
        dim,
        statistic=statistic,
        confidence=confidence,
        num_resamples=num_resamples,
        generator=generator,
    )
    wins = (first > second).to(result["estimate"].dtype)
    ties = (first == second).to(result["estimate"].dtype)
    result["probability_of_improvement"] = (wins + ties / 2).mean(dim)
    return result
