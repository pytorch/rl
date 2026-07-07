# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from collections.abc import Callable, Iterator

import torch
from tensordict import TensorDictBase

_DEFAULT_TRAJECTORY_KEYS = (("collector", "traj_ids"), "traj_ids", "episode")
_DONE_KEYS = (("next", "done"), "done")


class Trajectory:
    """A trajectory-first view over a contiguous slice of transitions.

    Wraps a :class:`~tensordict.TensorDictBase` holding the transitions of a
    single trajectory and exposes its entries as attributes. Attribute lookup
    resolves keys against the root tensordict first and falls back to the
    ``("next", ...)`` sub-tensordict, so ``traj.reward`` and ``traj.done``
    return the conventional post-step entries while ``traj.observation``,
    ``traj.state`` or ``traj.action`` return root entries.

    Args:
        data (TensorDictBase): a tensordict with a single batch dimension of
            length ``T`` holding the trajectory's transitions in order.

    Examples:
        >>> trajectory = Trajectory(data)
        >>> trajectory.observation.shape
        torch.Size([200, 3])
        >>> trajectory.reward.sum()
        tensor(-1234.5678)
        >>> trajectory.length
        200
    """

    def __init__(self, data: TensorDictBase) -> None:
        if data.batch_dims != 1:
            raise ValueError(
                f"Trajectory expects data with a single batch dimension, got batch_size={data.batch_size}."
            )
        self._data = data

    @property
    def data(self) -> TensorDictBase:
        """The underlying tensordict of transitions."""
        return self._data

    @property
    def length(self) -> int:
        """Number of transitions in the trajectory."""
        return int(self._data.batch_size[0])

    @property
    def total_reward(self) -> torch.Tensor:
        """Sum of rewards over the trajectory."""
        return self.reward.sum()

    def get(self, key, default=None) -> torch.Tensor | None:
        """Resolve ``key`` against the root tensordict, then ``("next", key)``."""
        value = self._data.get(key, None)
        if value is not None:
            return value
        if isinstance(key, str):
            value = self._data.get(("next", key), None)
            if value is not None:
                return value
        return default

    def keys(self, *args, **kwargs):
        """The keys of the underlying tensordict."""
        return self._data.keys(*args, **kwargs)

    def __getattr__(self, name: str):
        if name.startswith("_"):
            raise AttributeError(name)
        value = self.get(name)
        if value is None:
            raise AttributeError(
                f"Trajectory has no entry {name!r} in its data (looked up {name!r} and {('next', name)!r}). "
                f"Available keys: {sorted(map(str, self._data.keys(include_nested=True, leaves_only=True)))}"
            )
        return value

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self) -> int:
        return self.length

    def __repr__(self) -> str:
        return f"Trajectory(length={self.length}, keys={sorted(map(str, self._data.keys()))})"


class TrajectoryPredicate:
    """A boolean predicate over a :class:`Trajectory`.

    Predicates are produced by comparing trajectory expressions (see
    :data:`traj`) and compose with ``&`` (and), ``|`` (or) and ``~`` (not).

    Examples:
        >>> predicate = (traj.reward.sum() > 100) & (traj.length < 200)
        >>> predicate(trajectory)
        True
    """

    def __init__(
        self, fn: Callable[[Trajectory], bool], description: str = "predicate"
    ) -> None:
        self._fn = fn
        self._description = description

    def __call__(self, trajectory: Trajectory) -> bool:
        return bool(self._fn(trajectory))

    def __and__(self, other: TrajectoryPredicate) -> TrajectoryPredicate:
        return TrajectoryPredicate(
            lambda t: self(t) and other(t),
            f"({self._description} & {other._description})",
        )

    def __or__(self, other: TrajectoryPredicate) -> TrajectoryPredicate:
        return TrajectoryPredicate(
            lambda t: self(t) or other(t),
            f"({self._description} | {other._description})",
        )

    def __invert__(self) -> TrajectoryPredicate:
        return TrajectoryPredicate(lambda t: not self(t), f"~{self._description}")

    def __repr__(self) -> str:
        return f"TrajectoryPredicate({self._description})"


class _ElementwisePredicate:
    """An elementwise comparison awaiting reduction via ``.any()`` or ``.all()``."""

    def __init__(
        self, fn: Callable[[Trajectory], torch.Tensor], description: str
    ) -> None:
        self._fn = fn
        self._description = description

    def any(self) -> TrajectoryPredicate:
        """True if the comparison holds for any transition."""
        return TrajectoryPredicate(
            lambda t: bool(self._fn(t).any()), f"{self._description}.any()"
        )

    def all(self) -> TrajectoryPredicate:
        """True if the comparison holds for every transition."""
        return TrajectoryPredicate(
            lambda t: bool(self._fn(t).all()), f"{self._description}.all()"
        )

    def __call__(self, trajectory: Trajectory | None = None):
        raise TypeError(
            f"{self._description} compares per-transition values and is ambiguous as a "
            "trajectory filter. Call .any() or .all() to reduce it to a single boolean."
        )

    def __repr__(self) -> str:
        return f"_ElementwisePredicate({self._description})"


class _ScalarExpr:
    """A scalar-valued expression over a trajectory, comparable into a predicate."""

    def __init__(
        self, fn: Callable[[Trajectory], torch.Tensor | float | int], description: str
    ) -> None:
        self._fn = fn
        self._description = description

    def _compare(self, op, other, symbol: str) -> TrajectoryPredicate:
        return TrajectoryPredicate(
            lambda t: bool(op(self._fn(t), other)),
            f"({self._description} {symbol} {other!r})",
        )

    def __gt__(self, other) -> TrajectoryPredicate:
        return self._compare(lambda a, b: a > b, other, ">")

    def __ge__(self, other) -> TrajectoryPredicate:
        return self._compare(lambda a, b: a >= b, other, ">=")

    def __lt__(self, other) -> TrajectoryPredicate:
        return self._compare(lambda a, b: a < b, other, "<")

    def __le__(self, other) -> TrajectoryPredicate:
        return self._compare(lambda a, b: a <= b, other, "<=")

    def __eq__(self, other) -> TrajectoryPredicate:  # type: ignore[override]
        return self._compare(lambda a, b: a == b, other, "==")

    def __ne__(self, other) -> TrajectoryPredicate:  # type: ignore[override]
        return self._compare(lambda a, b: a != b, other, "!=")

    __hash__ = None

    def __repr__(self) -> str:
        return f"_ScalarExpr({self._description})"


class _FieldExpr:
    """A per-transition field of a trajectory, reducible or comparable."""

    def __init__(self, key, description: str | None = None) -> None:
        self._key = key
        self._description = description if description is not None else f"traj.{key}"

    def _get(self, trajectory: Trajectory) -> torch.Tensor:
        value = trajectory.get(self._key)
        if value is None:
            raise KeyError(
                f"Key {self._key!r} not found in trajectory (looked up root and ('next', ...))."
            )
        return value

    def _reduce(self, fn, name: str) -> _ScalarExpr:
        return _ScalarExpr(lambda t: fn(self._get(t)), f"{self._description}.{name}()")

    def sum(self) -> _ScalarExpr:
        """Sum of the field over the trajectory."""
        return self._reduce(torch.sum, "sum")

    def mean(self) -> _ScalarExpr:
        """Mean of the field over the trajectory."""
        return self._reduce(lambda x: x.float().mean(), "mean")

    def max(self) -> _ScalarExpr:
        """Maximum of the field over the trajectory."""
        return self._reduce(torch.max, "max")

    def min(self) -> _ScalarExpr:
        """Minimum of the field over the trajectory."""
        return self._reduce(torch.min, "min")

    def first(self) -> _ScalarExpr:
        """The field's value at the first transition."""
        return _ScalarExpr(lambda t: self._get(t)[0], f"{self._description}.first()")

    def last(self) -> _ScalarExpr:
        """The field's value at the last transition."""
        return _ScalarExpr(lambda t: self._get(t)[-1], f"{self._description}.last()")

    def _compare(self, op, other, symbol: str) -> _ElementwisePredicate:
        return _ElementwisePredicate(
            lambda t: op(self._get(t), other),
            f"({self._description} {symbol} {other!r})",
        )

    def __gt__(self, other) -> _ElementwisePredicate:
        return self._compare(lambda a, b: a > b, other, ">")

    def __ge__(self, other) -> _ElementwisePredicate:
        return self._compare(lambda a, b: a >= b, other, ">=")

    def __lt__(self, other) -> _ElementwisePredicate:
        return self._compare(lambda a, b: a < b, other, "<")

    def __le__(self, other) -> _ElementwisePredicate:
        return self._compare(lambda a, b: a <= b, other, "<=")

    def __eq__(self, other) -> _ElementwisePredicate:  # type: ignore[override]
        return self._compare(lambda a, b: a == b, other, "==")

    def __ne__(self, other) -> _ElementwisePredicate:  # type: ignore[override]
        return self._compare(lambda a, b: a != b, other, "!=")

    __hash__ = None

    def __repr__(self) -> str:
        return f"_FieldExpr({self._description})"


class _TrajectoryRef:
    """Entry point of the trajectory query language.

    Attribute access returns a field expression resolved against each
    trajectory (root keys first, then ``("next", ...)``); indexing with an
    explicit (possibly nested) key does the same for keys that are not valid
    attribute names. Field expressions support elementwise comparisons
    (reduce with ``.any()``/``.all()``) and reductions (``.sum()``,
    ``.mean()``, ``.max()``, ``.min()``, ``.first()``, ``.last()``) that
    compare into :class:`TrajectoryPredicate` objects composable with
    ``&``, ``|`` and ``~``.

    Examples:
        >>> from torchrl.data import traj
        >>> predicate = (traj.reward.sum() > 100) & (traj.length >= 50)
        >>> high_reward_trajs = replay_buffer.query(predicate)
        >>> spiky = (traj.reward > 10).any() | (traj.step_count.last() > 500)
        >>> masked = traj[("collector", "mask")].all()
    """

    @property
    def length(self) -> _ScalarExpr:
        """The number of transitions in the trajectory."""
        return _ScalarExpr(lambda t: t.length, "traj.length")

    @property
    def total_reward(self) -> _ScalarExpr:
        """The sum of rewards over the trajectory."""
        return _ScalarExpr(lambda t: t.total_reward, "traj.total_reward")

    def __getattr__(self, name: str) -> _FieldExpr:
        if name.startswith("_"):
            raise AttributeError(name)
        return _FieldExpr(name)

    def __getitem__(self, key) -> _FieldExpr:
        return _FieldExpr(key, description=f"traj[{key!r}]")

    def __repr__(self) -> str:
        return "traj"


traj = _TrajectoryRef()


def _trajectory_ids(data: TensorDictBase, trajectory_key=None) -> torch.Tensor | None:
    keys = (trajectory_key,) if trajectory_key is not None else _DEFAULT_TRAJECTORY_KEYS
    for key in keys:
        ids = data.get(key, None)
        if ids is not None:
            return ids.reshape(data.batch_size[0])
    if trajectory_key is not None:
        raise KeyError(f"trajectory_key {trajectory_key!r} not found in data.")
    return None


def _split_boundaries(data: TensorDictBase, trajectory_key=None) -> list[int]:
    numel = data.batch_size[0]
    ids = _trajectory_ids(data, trajectory_key)
    if ids is not None:
        changes = (ids[1:] != ids[:-1]).nonzero().flatten() + 1
        return [0, *changes.tolist(), numel]
    for done_key in _DONE_KEYS:
        done = data.get(done_key, None)
        if done is not None:
            done = done.reshape(numel)
            ends = done.nonzero().flatten() + 1
            boundaries = [0, *ends.tolist()]
            if boundaries[-1] != numel:
                boundaries.append(numel)
            return boundaries
    raise KeyError(
        "Cannot split data into trajectories: no trajectory id entry found "
        f"(looked up {_DEFAULT_TRAJECTORY_KEYS}) and no done entry found "
        f"(looked up {_DONE_KEYS}). Pass trajectory_key explicitly."
    )


def iter_trajectories(
    data: TensorDictBase, trajectory_key=None
) -> Iterator[Trajectory]:
    """Iterate over the trajectories stored in a flat batch of transitions.

    Consecutive transitions are grouped into trajectories using, in order of
    preference: an explicit ``trajectory_key``, the conventional
    ``("collector", "traj_ids")`` / ``"traj_ids"`` / ``"episode"`` entries, or
    the ``("next", "done")`` flags. Transitions belonging to the same
    trajectory are assumed to be stored contiguously and in order, as written
    by the standard round-robin writers.

    Args:
        data (TensorDictBase): a tensordict of transitions with a single
            batch dimension.
        trajectory_key (NestedKey, optional): entry holding per-transition
            trajectory ids. Defaults to None (auto-detection).

    Yields:
        :class:`Trajectory` views over ``data``.
    """
    if not isinstance(data, TensorDictBase):
        raise TypeError(
            f"iter_trajectories expects a TensorDictBase, got {type(data)}. "
            "Trajectory queries require tensordict-backed storages."
        )
    if data.batch_dims != 1:
        raise ValueError(
            "iter_trajectories expects data with a single batch dimension "
            f"(flat transitions), got batch_size={data.batch_size}."
        )
    boundaries = _split_boundaries(data, trajectory_key)
    for start, stop in zip(boundaries[:-1], boundaries[1:]):
        if stop > start:
            yield Trajectory(data[start:stop])


def filter_trajectories(
    data: TensorDictBase,
    predicate: Callable[[Trajectory], bool] | None = None,
    *,
    trajectory_key=None,
) -> list[Trajectory]:
    """Split ``data`` into trajectories and keep those matching ``predicate``.

    Args:
        data (TensorDictBase): a tensordict of transitions with a single
            batch dimension.
        predicate (Callable[[Trajectory], bool], optional): a
            :class:`TrajectoryPredicate` built from :data:`traj`, or any
            callable mapping a :class:`Trajectory` to a boolean. Defaults to
            None (keep all trajectories).

    Keyword Args:
        trajectory_key (NestedKey, optional): entry holding per-transition
            trajectory ids. Defaults to None (auto-detection).

    Returns:
        A list of matching :class:`Trajectory` views.

    Examples:
        >>> from torchrl.data import filter_trajectories, traj
        >>> good = filter_trajectories(data, traj.reward.sum() > 100)
    """
    if isinstance(predicate, _ElementwisePredicate):
        predicate()
    return [
        trajectory
        for trajectory in iter_trajectories(data, trajectory_key)
        if predicate is None or predicate(trajectory)
    ]
