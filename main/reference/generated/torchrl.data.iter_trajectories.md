# iter_trajectories

torchrl.data.iter_trajectories(*data: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *trajectory_key: NestedKey | None = None*) → Iterator[[Trajectory](torchrl.data.Trajectory.html#torchrl.data.Trajectory)][[source]](../../_modules/torchrl/data/replay_buffers/query.html#iter_trajectories)

Iterate over the trajectories stored in a flat batch of transitions.

Consecutive transitions are grouped into trajectories using, in order of
preference: an explicit `trajectory_key`, the conventional
`("collector", "traj_ids")` / `"traj_ids"` / `"episode"` entries, or
the union of the `("next", "done")` / `("next", "terminated")` /
`("next", "truncated")` end flags. Transitions belonging to the same
trajectory are assumed to be stored contiguously and in order, as written
by the standard round-robin writers. Boundary recovery shares the
machinery of
[`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler).

Warning

When no trajectory id entry is available, splitting falls back to the
end-of-episode flags and a `UserWarning` is emitted: a trajectory
whose last transition does not carry a positive end flag cannot be
distinguished from the following one and the two are silently merged.
Store trajectory ids for reliable splitting.

Parameters:

- **data** (*TensorDictBase*) - a tensordict of transitions with a single
batch dimension.
- **trajectory_key** (*NestedKey**,**optional*) - entry holding per-transition
trajectory ids. Defaults to None (auto-detection).

Yields:

[`Trajectory`](torchrl.data.Trajectory.html#torchrl.data.Trajectory) views over `data`.