# filter_trajectories

torchrl.data.filter_trajectories(*data: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *predicate: Callable[[[Trajectory](torchrl.data.Trajectory.html#torchrl.data.Trajectory)], bool] | None = None*, ***, *trajectory_key: NestedKey | None = None*) → list[[Trajectory](torchrl.data.Trajectory.html#torchrl.data.Trajectory)][[source]](../../_modules/torchrl/data/replay_buffers/query.html#filter_trajectories)

Split `data` into trajectories and keep those matching `predicate`.

Parameters:

- **data** (*TensorDictBase*) - a tensordict of transitions with a single
batch dimension.
- **predicate** (*Callable**[**[*[*Trajectory*](torchrl.data.Trajectory.html#torchrl.data.Trajectory)*]**,**bool**]**,**optional*) - a
[`TrajectoryPredicate`](torchrl.data.TrajectoryPredicate.html#torchrl.data.TrajectoryPredicate) built from `traj`, or any
callable mapping a [`Trajectory`](torchrl.data.Trajectory.html#torchrl.data.Trajectory) to a boolean. Defaults to
None (keep all trajectories).

Keyword Arguments:

**trajectory_key** (*NestedKey**,**optional*) - entry holding per-transition
trajectory ids. Defaults to None (auto-detection).

Returns:

A list of matching [`Trajectory`](torchrl.data.Trajectory.html#torchrl.data.Trajectory) views.

Examples

```
>>> from torchrl.data import filter_trajectories, traj
>>> good = filter_trajectories(data, traj.reward.sum() > 100)
```