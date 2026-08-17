# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import operator
import warnings
from collections.abc import Callable, Iterator, Sequence

import numpy as np
import torch
from tensordict import NestedKey, TensorClass, TensorDictBase
from tensordict.utils import unravel_key

from torchrl._utils import DEFAULT_DONE_KEYS
from torchrl.data.replay_buffers.utils import find_start_stop_traj

_DEFAULT_TRAJECTORY_KEYS = (("collector", "traj_ids"), "traj_ids", "episode")
# TED convention: a trajectory ends when any of done / terminated / truncated
# is set, so all present end signals within a group are OR-ed together.
_END_KEY_GROUPS = (
    tuple(("next", key) for key in DEFAULT_DONE_KEYS),
    tuple(DEFAULT_DONE_KEYS),
)


class Trajectory(TensorClass["nocast"]):
    """A trajectory-first view over a contiguous slice of transitions.

    A single-field tensorclass wrapping a :class:`~tensordict.TensorDictBase`
    that holds the transitions of one trajectory, exposing its entries as
    attributes. Attribute lookup resolves keys against the root tensordict
    first and falls back to the ``("next", ...)`` sub-tensordict, so
    ``traj.reward`` and ``traj.done`` return the conventional post-step
    entries while ``traj.observation``, ``traj.state`` or ``traj.action``
    return root entries.

    Being a tensorclass, slicing and indexing return :class:`Trajectory`
    instances, and trajectories of different lengths can be assembled into a
    single ragged batch with :func:`~tensordict.lazy_stack`.

    Attributes:
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
        >>> trajectory[:10]
        Trajectory(length=10, keys=['action', 'next', 'observation'])
        >>> from tensordict import lazy_stack
        >>> stacked = lazy_stack([trajectory, other_trajectory])
    """

    data: TensorDictBase

    def __post_init__(self) -> None:
        if not self.batch_size:
            data = self.data
            if data is not None and data.batch_dims:
                if data.batch_dims != 1:
                    raise ValueError(
                        "Trajectory expects data with a single batch dimension, "
                        f"got batch_size={data.batch_size}."
                    )
                self.batch_size = data.batch_size

    @property
    def length(self) -> int:
        """Number of transitions in the trajectory."""
        return int(self.batch_size[0])

    @property
    def total_reward(self) -> torch.Tensor:
        """Sum of rewards over the trajectory."""
        return self.reward.sum()

    def get(self, key: NestedKey, default=None) -> torch.Tensor | None:
        """Resolve ``key`` against the root tensordict, then ``("next", key)``."""
        data = self.data
        value = data.get(key, None)
        if value is not None:
            return value
        if isinstance(key, str):
            value = data.get(("next", key), None)
            if value is not None:
                return value
        return default

    def keys(self, *args, **kwargs):
        """The keys of the underlying tensordict."""
        return self.data.keys(*args, **kwargs)


_TENSORCLASS_GETATTR = Trajectory.__getattr__
_TENSORCLASS_GETITEM = Trajectory.__getitem__


def _trajectory_getattr(self, name: str):
    """Falls back to data-key resolution for names the tensorclass does not own.

    The tensorclass machinery installs its own ``__getattr__`` after class
    creation, so the fallback is chained here rather than defined in the
    class body.
    """
    try:
        return _TENSORCLASS_GETATTR(self, name)
    except AttributeError:
        if name.startswith("_"):
            raise
        value = self.get(name)
        if value is None:
            raise AttributeError(
                f"Trajectory has no entry {name!r} in its data (looked up {name!r} and {('next', name)!r}). "
                f"Available keys: {sorted(map(str, self.data.keys(include_nested=True, leaves_only=True)))}"
            ) from None
        return value


def _trajectory_getitem(self, index):
    """Routes key indexing to the data and batch indexing to the tensorclass."""
    if isinstance(index, str) or (isinstance(index, tuple) and unravel_key(index)):
        return self.data[index]
    return _TENSORCLASS_GETITEM(self, index)


def _trajectory_repr(self) -> str:
    if len(self.batch_size) == 1:
        shape = f"length={self.length}"
    else:
        shape = f"batch_size={tuple(self.batch_size)}"
    data = self.data
    keys = sorted(map(str, data.keys())) if data is not None else None
    return f"Trajectory({shape}, keys={keys})"


Trajectory.__getattr__ = _trajectory_getattr
Trajectory.__getitem__ = _trajectory_getitem
Trajectory.__repr__ = _trajectory_repr


def _union_required_keys(
    first: frozenset | None, second: frozenset | None
) -> frozenset | None:
    if first is None or second is None:
        return None
    return first | second


class TrajectoryPredicate:
    """A boolean predicate over a :class:`Trajectory`.

    Predicates are produced by comparing trajectory expressions (see
    :data:`traj`) and compose with ``&`` (and), ``|`` (or) and ``~`` (not).

    Examples:
        >>> predicate = (traj.reward.sum() > 100) & (traj.length < 200)
        >>> predicate(trajectory)
        True
        >>> predicate.required_keys()
        frozenset({'reward'})
    """

    def __init__(
        self,
        fn: Callable[[Trajectory], bool],
        description: str = "predicate",
        keys: frozenset | None = None,
    ) -> None:
        self._fn = fn
        self._description = description
        self._keys = keys

    def required_keys(self) -> frozenset | None:
        """The data keys the predicate reads, or ``None`` when unknown.

        Predicates built from :data:`traj` expressions report the exact set
        of keys they need, which lets :meth:`ReplayBuffer.query
        <torchrl.data.ReplayBuffer.query>` fetch only those entries from the
        storage. Predicates wrapping opaque callables return ``None``.
        """
        return self._keys

    def __call__(self, trajectory: Trajectory) -> bool:
        return bool(self._fn(trajectory))

    def __and__(self, other: TrajectoryPredicate) -> TrajectoryPredicate:
        return TrajectoryPredicate(
            lambda t: self(t) and other(t),
            f"({self._description} & {other._description})",
            keys=_union_required_keys(self._keys, other._keys),
        )

    def __or__(self, other: TrajectoryPredicate) -> TrajectoryPredicate:
        return TrajectoryPredicate(
            lambda t: self(t) or other(t),
            f"({self._description} | {other._description})",
            keys=_union_required_keys(self._keys, other._keys),
        )

    def __invert__(self) -> TrajectoryPredicate:
        return TrajectoryPredicate(
            lambda t: not self(t), f"~{self._description}", keys=self._keys
        )

    def __repr__(self) -> str:
        return f"TrajectoryPredicate({self._description})"


class _ElementwisePredicate:
    """An elementwise comparison awaiting reduction via ``.any()`` or ``.all()``."""

    def __init__(
        self,
        fn: Callable[[Trajectory], torch.Tensor],
        description: str,
        keys: frozenset | None = None,
    ) -> None:
        self._fn = fn
        self._description = description
        self._keys = keys

    def any(self) -> TrajectoryPredicate:
        """True if the comparison holds for any transition."""
        return TrajectoryPredicate(
            lambda t: bool(self._fn(t).any()),
            f"{self._description}.any()",
            keys=self._keys,
        )

    def all(self) -> TrajectoryPredicate:
        """True if the comparison holds for every transition."""
        return TrajectoryPredicate(
            lambda t: bool(self._fn(t).all()),
            f"{self._description}.all()",
            keys=self._keys,
        )

    def __call__(self, trajectory: Trajectory | None = None):
        raise TypeError(
            f"{self._description} compares per-transition values and is ambiguous as a "
            "trajectory filter. Call .any() or .all() to reduce it to a single boolean."
        )

    def __repr__(self) -> str:
        return f"_ElementwisePredicate({self._description})"


class _ComparableExpr:
    """Mixin providing the comparison dunders shared by trajectory expressions.

    Subclasses set ``_comparison_cls`` to the wrapper produced by a
    comparison and implement ``_value`` to evaluate the expression on a
    trajectory.
    """

    _comparison_cls: type

    def _value(self, trajectory: Trajectory):
        raise NotImplementedError

    def _compare(self, op, other, symbol: str):
        return self._comparison_cls(
            lambda t: op(self._value(t), other),
            f"({self._description} {symbol} {other!r})",
            keys=self._keys,
        )

    def __gt__(self, other):
        return self._compare(operator.gt, other, ">")

    def __ge__(self, other):
        return self._compare(operator.ge, other, ">=")

    def __lt__(self, other):
        return self._compare(operator.lt, other, "<")

    def __le__(self, other):
        return self._compare(operator.le, other, "<=")

    def __eq__(self, other):  # type: ignore[override]
        return self._compare(operator.eq, other, "==")

    def __ne__(self, other):  # type: ignore[override]
        return self._compare(operator.ne, other, "!=")

    __hash__ = None


class _ScalarExpr(_ComparableExpr):
    """A scalar-valued expression over a trajectory, comparable into a predicate."""

    _comparison_cls = TrajectoryPredicate

    def __init__(
        self,
        fn: Callable[[Trajectory], torch.Tensor | float | int],
        description: str,
        keys: frozenset | None = None,
    ) -> None:
        self._fn = fn
        self._description = description
        self._keys = keys

    def _value(self, trajectory: Trajectory):
        return self._fn(trajectory)

    def __repr__(self) -> str:
        return f"_ScalarExpr({self._description})"


class _FieldExpr(_ComparableExpr):
    """A per-transition field of a trajectory, reducible or comparable."""

    _comparison_cls = _ElementwisePredicate

    def __init__(self, key: NestedKey, description: str | None = None) -> None:
        self._key = key
        self._description = description if description is not None else f"traj.{key}"
        self._keys = frozenset({unravel_key(key)})

    def _get(self, trajectory: Trajectory) -> torch.Tensor:
        value = trajectory.get(self._key)
        if value is None:
            raise KeyError(
                f"Key {self._key!r} not found in trajectory (looked up root and ('next', ...))."
            )
        return value

    def _value(self, trajectory: Trajectory):
        return self._get(trajectory)

    def _reduce(self, fn, name: str) -> _ScalarExpr:
        return _ScalarExpr(
            lambda t: fn(self._get(t)),
            f"{self._description}.{name}()",
            keys=self._keys,
        )

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
        return _ScalarExpr(
            lambda t: self._get(t)[0], f"{self._description}.first()", keys=self._keys
        )

    def last(self) -> _ScalarExpr:
        """The field's value at the last transition."""
        return _ScalarExpr(
            lambda t: self._get(t)[-1], f"{self._description}.last()", keys=self._keys
        )

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
        return _ScalarExpr(lambda t: t.length, "traj.length", keys=frozenset())

    @property
    def total_reward(self) -> _ScalarExpr:
        """The sum of rewards over the trajectory."""
        return _ScalarExpr(
            lambda t: t.total_reward, "traj.total_reward", keys=frozenset({"reward"})
        )

    def __getattr__(self, name: str) -> _FieldExpr:
        if name.startswith("_"):
            raise AttributeError(name)
        return _FieldExpr(name)

    def __getitem__(self, key: NestedKey) -> _FieldExpr:
        return _FieldExpr(key, description=f"traj[{key!r}]")

    def __repr__(self) -> str:
        return "traj"


traj = _TrajectoryRef()


def _last_write_index(cursor) -> int | None:
    """Best-effort extraction of the last written flat index from a storage cursor."""
    if cursor is None:
        return None
    if isinstance(cursor, torch.Tensor):
        if cursor.numel() == 0:
            return None
        return int(cursor.reshape(-1)[-1])
    if isinstance(cursor, np.ndarray):
        if cursor.size == 0:
            return None
        return int(cursor.reshape(-1)[-1])
    if isinstance(cursor, range):
        return cursor[-1] if len(cursor) else None
    if isinstance(cursor, int):
        return cursor
    return None


def _boundary_signal(
    source: TensorDictBase, trajectory_key: NestedKey | None
) -> tuple[str, torch.Tensor]:
    """Fetches the entry that individuates trajectories from ``source``.

    Returns a ``(kind, signal)`` pair where ``kind`` is ``"trajectory"`` for
    per-element trajectory ids and ``"end"`` for the OR-ed end-of-episode
    flags, with the signal reshaped to the batch shape of ``source``. Warns
    when falling back to end flags, since two back-to-back trajectories
    without a positive end flag in between cannot be told apart.
    """
    batch_size = source.batch_size
    keys = (trajectory_key,) if trajectory_key is not None else _DEFAULT_TRAJECTORY_KEYS
    for key in keys:
        ids = source.get(key, None)
        if ids is not None:
            return "trajectory", ids.reshape(batch_size)
    if trajectory_key is not None:
        raise KeyError(f"trajectory_key {trajectory_key!r} not found in data.")
    for key_group in _END_KEY_GROUPS:
        end = None
        found = []
        for end_key in key_group:
            flag = source.get(end_key, None)
            if flag is not None:
                flag = flag.reshape(batch_size)
                end = flag if end is None else end | flag
                found.append(end_key)
        if end is not None:
            warnings.warn(
                "No trajectory id entry was found; splitting on the end-of-episode "
                f"flags {tuple(found)}. Any trajectory whose last transition does not "
                "carry a positive end flag will be silently merged with the "
                "following one. Store trajectory ids (e.g. ('collector', "
                "'traj_ids')) or pass trajectory_key explicitly for reliable "
                "splitting.",
                category=UserWarning,
            )
            return "end", end
    raise KeyError(
        "Cannot split data into trajectories: no trajectory id entry found "
        f"(looked up {_DEFAULT_TRAJECTORY_KEYS}) and no end-of-episode entry "
        f"found (looked up {_END_KEY_GROUPS}). Pass trajectory_key explicitly."
    )


def _trajectory_boundaries(
    source: TensorDictBase,
    trajectory_key: NestedKey | None,
    *,
    at_capacity: bool = False,
    cursor=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes ``(start_idx, stop_idx, lengths)`` for the trajectories in ``source``.

    Boundary recovery is delegated to
    :func:`~torchrl.data.find_start_stop_traj`, the same machinery
    :class:`~torchrl.data.replay_buffers.samplers.SliceSampler`
    uses, so samplers and queries always agree on trajectory boundaries.
    """
    kind, signal = _boundary_signal(source, trajectory_key)
    return find_start_stop_traj(
        trajectory=signal if kind == "trajectory" else None,
        end=signal if kind == "end" else None,
        at_capacity=at_capacity,
        cursor=cursor,
    )


def _chronological_order(
    start_idx: torch.Tensor,
    *,
    storage_length: int,
    batch_shape: torch.Size,
    at_capacity: bool,
    cursor,
) -> torch.Tensor:
    """Orders trajectory boundaries chronologically within each batch column.

    Once a round-robin storage has wrapped, storage order no longer matches
    write order: the oldest remaining element sits right after the write
    cursor. Trajectories are keyed by their time offset from that position
    (grouped by batch coordinates for multi-dimensional storages).
    """
    offset = 0
    if at_capacity:
        last = _last_write_index(cursor)
        if last is not None:
            offset = (last % storage_length + 1) % storage_length
    logical = (start_idx[:, 0].to(torch.long) - offset) % storage_length
    key = logical
    if start_idx.shape[1] > 1:
        flat_batch = torch.zeros_like(logical)
        for dim in range(1, start_idx.shape[1]):
            flat_batch = flat_batch * batch_shape[dim] + start_idx[:, dim].to(
                torch.long
            )
        key = flat_batch * storage_length + logical
    return torch.argsort(key)


def _extract_trajectory_data(
    source: TensorDictBase,
    start: torch.Tensor,
    stop: torch.Tensor,
    length: int,
    storage_length: int,
) -> TensorDictBase:
    """Reads one trajectory from a time-first view, following the wrap point.

    Contiguous trajectories are returned as zero-copy slices; only a
    trajectory written across the storage wrap point requires a gather.
    """
    time_start = int(start[0])
    time_stop = int(stop[0])
    batch_coords = tuple(int(coord) for coord in start[1:])
    if time_start <= time_stop:
        return source[(slice(time_start, time_stop + 1), *batch_coords)]
    index = torch.arange(time_start, time_start + length) % storage_length
    return source[(index, *batch_coords)]


def _as_key_tuple(key: NestedKey) -> tuple:
    key = unravel_key(key)
    return key if isinstance(key, tuple) else (key,)


def _keys_overlap(first: NestedKey, second: NestedKey) -> bool:
    first = _as_key_tuple(first)
    second = _as_key_tuple(second)
    depth = min(len(first), len(second))
    return first[:depth] == second[:depth]


def _expand_required_keys(keys: frozenset) -> set:
    """Adds the ``("next", key)`` candidates used by attribute resolution."""
    expanded = set()
    for key in keys:
        key = unravel_key(key)
        expanded.add(key)
        if isinstance(key, str):
            expanded.add(("next", key))
    return expanded


def _minimal_transform_plan(
    transforms: Sequence, required: set
) -> tuple[list, set | None]:
    """Selects the transform suffix and storage keys a predicate needs.

    Walks the transform chain backward, keeping a transform only if its
    ``out_keys`` can affect a required key and accumulating its ``in_keys``
    into the required set. A transform without ``out_keys`` metadata has
    unknown effects, so it and every transform before it are kept and all
    keys are fetched (returned as ``None``).
    """
    kept_reversed = []
    required = set(required)
    for index in range(len(transforms) - 1, -1, -1):
        transform = transforms[index]
        out_keys = getattr(transform, "out_keys", None) or []
        if not out_keys:
            kept = list(transforms[: index + 1]) + list(reversed(kept_reversed))
            return kept, None
        if any(
            _keys_overlap(out_key, req_key)
            for out_key in out_keys
            for req_key in required
        ):
            kept_reversed.append(transform)
            in_keys = getattr(transform, "in_keys", None) or []
            required.update(unravel_key(key) for key in in_keys)
    return list(reversed(kept_reversed)), required


def _apply_transforms(data: TensorDictBase, transforms: Sequence) -> TensorDictBase:
    if not transforms:
        return data
    with data.unlock_():
        for transform in transforms:
            data = transform(data)
    return data


def _query_source(
    source: TensorDictBase,
    *,
    transforms: Sequence = (),
    predicate: Callable[[Trajectory], bool] | None = None,
    trajectory_key: NestedKey | None = None,
    at_capacity: bool = False,
    cursor=None,
) -> list[Trajectory]:
    """Splits a time-first storage view into trajectories and filters them.

    This is the engine behind :meth:`ReplayBuffer.query
    <torchrl.data.ReplayBuffer.query>`. Predicates built from :data:`traj`
    expose :meth:`TrajectoryPredicate.required_keys`, which is used to fetch
    only the entries the predicate reads (and run only the transforms that
    can affect them) during evaluation. Matching trajectories are then
    extracted in full, with the complete transform chain applied per
    trajectory, mirroring what a sampler would return.
    """
    if isinstance(predicate, _ElementwisePredicate):
        raise TypeError(
            f"{predicate._description} compares per-transition values and is "
            "ambiguous as a trajectory filter. Call .any() or .all() to "
            "reduce it to a single boolean."
        )
    if not source.batch_dims:
        raise ValueError(
            "Trajectory queries expect a storage with at least one batch "
            f"dimension, got batch_size={source.batch_size}."
        )
    storage_length = source.batch_size[0]
    if storage_length == 0:
        return []
    start_idx, stop_idx, lengths = _trajectory_boundaries(
        source, trajectory_key, at_capacity=at_capacity, cursor=cursor
    )
    if not start_idx.numel():
        return []
    order = _chronological_order(
        start_idx,
        storage_length=storage_length,
        batch_shape=source.batch_size,
        at_capacity=at_capacity,
        cursor=cursor,
    )

    transforms = list(transforms)
    eval_source = None
    eval_transforms = transforms
    eval_is_full = True
    if predicate is not None:
        required = None
        required_keys_fn = getattr(predicate, "required_keys", None)
        if callable(required_keys_fn):
            required = required_keys_fn()
        if required is None:
            eval_source = source
        else:
            expanded = _expand_required_keys(required)
            eval_transforms, fetch_keys = _minimal_transform_plan(transforms, expanded)
            if fetch_keys is None:
                eval_source = source
            else:
                eval_source = source.select(*fetch_keys, strict=False)
                eval_is_full = False
        eval_is_full = eval_is_full and len(eval_transforms) == len(transforms)
        if eval_transforms and eval_source is source:
            eval_source = source.copy()
        eval_source = _apply_transforms(eval_source, eval_transforms)

    results = []
    for position in order.tolist():
        start = start_idx[position]
        stop = stop_idx[position]
        length = int(lengths[position])
        if predicate is None:
            matched = True
            eval_trajectory = None
        else:
            eval_data = _extract_trajectory_data(
                eval_source, start, stop, length, storage_length
            )
            eval_trajectory = Trajectory(eval_data)
            matched = bool(predicate(eval_trajectory))
        if not matched:
            continue
        if eval_trajectory is not None and eval_is_full:
            results.append(eval_trajectory)
            continue
        full_data = _extract_trajectory_data(
            source, start, stop, length, storage_length
        )
        full_data = _apply_transforms(full_data, transforms)
        results.append(Trajectory(full_data))
    return results


def iter_trajectories(
    data: TensorDictBase, trajectory_key: NestedKey | None = None
) -> Iterator[Trajectory]:
    """Iterate over the trajectories stored in a flat batch of transitions.

    Consecutive transitions are grouped into trajectories using, in order of
    preference: an explicit ``trajectory_key``, the conventional
    ``("collector", "traj_ids")`` / ``"traj_ids"`` / ``"episode"`` entries, or
    the union of the ``("next", "done")`` / ``("next", "terminated")`` /
    ``("next", "truncated")`` end flags. Transitions belonging to the same
    trajectory are assumed to be stored contiguously and in order, as written
    by the standard round-robin writers. Boundary recovery shares the
    machinery of
    :class:`~torchrl.data.replay_buffers.samplers.SliceSampler`.

    .. warning::
        When no trajectory id entry is available, splitting falls back to the
        end-of-episode flags and a ``UserWarning`` is emitted: a trajectory
        whose last transition does not carry a positive end flag cannot be
        distinguished from the following one and the two are silently merged.
        Store trajectory ids for reliable splitting.

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
    if data.batch_size[0] == 0:
        return
    start_idx, stop_idx, _ = _trajectory_boundaries(data, trajectory_key)
    for start, stop in zip(start_idx[:, 0].tolist(), stop_idx[:, 0].tolist()):
        yield Trajectory(data[start : stop + 1])


def filter_trajectories(
    data: TensorDictBase,
    predicate: Callable[[Trajectory], bool] | None = None,
    *,
    trajectory_key: NestedKey | None = None,
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
        raise TypeError(
            f"{predicate._description} compares per-transition values and is "
            "ambiguous as a trajectory filter. Call .any() or .all() to "
            "reduce it to a single boolean."
        )
    return [
        trajectory
        for trajectory in iter_trajectories(data, trajectory_key)
        if predicate is None or predicate(trajectory)
    ]
