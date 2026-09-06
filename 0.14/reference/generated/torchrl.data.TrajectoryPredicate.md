# TrajectoryPredicate

*class*torchrl.data.TrajectoryPredicate(*fn: Callable[[[Trajectory](torchrl.data.Trajectory.html#torchrl.data.Trajectory)], bool]*, *description: str = 'predicate'*, *keys: frozenset | None = None*)[[source]](../../_modules/torchrl/data/replay_buffers/query.html#TrajectoryPredicate)

A boolean predicate over a [`Trajectory`](torchrl.data.Trajectory.html#torchrl.data.Trajectory).

Predicates are produced by comparing trajectory expressions (see
`traj`) and compose with `&` (and), `|` (or) and `~` (not).

Examples

```
>>> predicate = (traj.reward.sum() > 100) & (traj.length < 200)
>>> predicate(trajectory)
True
>>> predicate.required_keys()
frozenset({'reward'})
```

required_keys() → frozenset | None[[source]](../../_modules/torchrl/data/replay_buffers/query.html#TrajectoryPredicate.required_keys)

The data keys the predicate reads, or `None` when unknown.

Predicates built from `traj` expressions report the exact set
of keys they need, which lets [`ReplayBuffer.query`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.query) fetch only those entries from the
storage. Predicates wrapping opaque callables return `None`.