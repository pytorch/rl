# Glossary

TorchRL borrows much of its vocabulary from `tensordict` and the broader
RL literature, but a handful of terms appear in error messages and source code
without a dedicated definition in the API reference. This page lists those
terms with the minimum context needed to find the relevant code.

_AcceptedKeys

A dataclass nested inside most [`LossModule`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)
subclasses that declares the tensordict keys the loss expects to read or
write. Each field is a `NestedKey` with a
default value. Override the defaults via
[`set_keys()`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule.set_keys) rather than mutating the
dataclass directly; `set_keys` also propagates the change to the
underlying value estimator.

BatchedEnv

A TorchRL environment that owns more than one environment instance under
a single [`EnvBase`](generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) interface. The common
implementations are [`SerialEnv`](generated/torchrl.envs.SerialEnv.html#torchrl.envs.SerialEnv) and
[`ParallelEnv`](generated/torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv), both subclasses of the internal
`BatchedEnvBase`. Their `batch_size` is the
leading shape of reset, step, and collector outputs.

carrier

The [`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) stored as `self._carrier` inside
[`rollout()`](generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector.rollout). It persists across
collector batches and holds the post-reset environment output that the
next policy call consumes. See [Collector Internals](collectors_internals.html#ref-collectors-internals) for the
full lifecycle.

Collector

The main collector construction API,
[`Collector`](generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector). By default it alternates policy
calls and environment steps in the training process. `backend` and
`num_collectors` select local process, Ray, RPC, or distributed
implementations without changing the entry point.

compact_obs

Collector setting that drops observation and state keys from the
`("next", ...)` sub-tensordict of every persisted step. Within a
contiguous same-trajectory sample, those values can be reconstructed from
the root keys of the following step. At trajectory boundaries or in
non-contiguous random samples, reconstruction must use the configured fill
value; see [`NextStateReconstructor`](generated/torchrl.envs.transforms.NextStateReconstructor.html#torchrl.envs.transforms.NextStateReconstructor) and the
`compact_obs` argument on [`Collector`](generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector).

CompositeCompositeSpec

A nested spec container, currently named [`Composite`](generated/torchrl.data.Composite.html#torchrl.data.Composite),
that maps tensordict keys to leaf [`TensorSpec`](generated/torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)
objects. Environment specs such as `observation_spec`, `action_spec`,
and `reward_spec` are usually composites. `CompositeSpec` is an older
name that may still appear in discussions and issue reports.

Env

Short for environment: an object implementing the
[`EnvBase`](generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) API, including `reset`, `step`, specs,
device handling, and a tensordict-based input/output contract. TorchRL env
wrappers usually subclass [`Transform`](generated/torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) or compose a
[`TransformedEnv`](generated/torchrl.envs.transforms.TransformedEnv.html#torchrl.envs.transforms.TransformedEnv) rather than following the Gym
wrapper API directly.

env batch size

The leading batch shape of an environment, exposed as
[`batch_size`](generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase.batch_size). A single unbatched env has an
empty batch size; a [`ParallelEnv`](generated/torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) with `N` workers
usually has batch size `[N]`. Collectors append a time dimension to this
shape when they stack rollout steps.

env_device

The collector device slot used for environment `reset` and `step`
operations. When it differs from `policy_device` or from the storage
layout, the collector inserts the casts and sync points described in
[Collector Internals](collectors_internals.html#ref-collectors-internals).

EnvCreator

A small callable wrapper, [`EnvCreator`](generated/torchrl.envs.EnvCreator.html#torchrl.envs.EnvCreator), used to build
environments lazily or in worker processes. It is useful when constructors
need to be serialized for `Collector(num_collectors=N)` or another
distributed `Collector` backend.

functional (loss)

A [`LossModule`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule) is *functional* when it stores
its actor / critic parameters as a stateless tensordict and invokes the
networks with [`to_module()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictParams.html#tensordict.TensorDictParams.to_module) at call time.
This is what makes soft / target update, `separate_losses=True`, and
per-parameter optimiser groups possible without deep-copying the
underlying `nn.Module`. Check `loss.functional` to see which mode a
given loss is in.

in_keysout_keys

The list of tensordict keys a module reads from (`in_keys`) and writes
to (`out_keys`). Both [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule) and most
TorchRL loss / value-estimator components expose these as constructor
arguments. Modifying them lets you wire a module into a tensordict layout
that differs from the defaults; see data_layout
for naming conventions.

is_init

A boolean key (default name: `"is_init"`) written by
[`InitTracker`](generated/torchrl.envs.transforms.InitTracker.html#torchrl.envs.transforms.InitTracker) immediately after every env reset.
Recurrent modules and advantage estimators read this key to know where
trajectories begin so they can zero out stale hidden state or reset the
bootstrap target.

no_cuda_sync

A collector flag that suppresses the explicit CUDA, MPS, or NPU
synchronizations inserted after cross-device transfers. Safe to set only
when transfers are already correctly ordered or when running pure CPU.
Defaults to `False`.

policy_device

The collector device slot where the policy network runs. When it differs
from `env_device`, the collector casts the carrier before policy and env
calls.

recurrent mode

The flag controlling whether an RNN-bearing module
([`LSTMModule`](generated/torchrl.modules.LSTMModule.html#torchrl.modules.LSTMModule),
[`GRUModule`](generated/torchrl.modules.GRUModule.html#torchrl.modules.GRUModule)) processes a single timestep per call
(*sequential*) or a full `(B, T, ...)` sequence in one call
(*recurrent*). Toggled via the
[`set_recurrent_mode`](generated/torchrl.modules.set_recurrent_mode.html#torchrl.modules.set_recurrent_mode) context manager. Collectors
run in sequential mode; losses run in recurrent mode so the module can
split and pad on trajectory boundaries inside a replayed batch.

set_keys

The public method on [`LossModule`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule) and value
estimators used to override the default tensordict keys a loss expects.
Example: `loss.set_keys(value=("agents", "state_value"),
action=("agents", "action"))`. Prefer this over reaching into
`loss.tensor_keys` directly because it
also wires changes into the loss's value estimator if one exists.

Specs

Tensor constraints that describe valid values, shapes, dtypes, and
devices. TorchRL uses [`TensorSpec`](generated/torchrl.data.TensorSpec.html#torchrl.data.TensorSpec) leaves, such as
[`Bounded`](generated/torchrl.data.Bounded.html#torchrl.data.Bounded) and [`Unbounded`](generated/torchrl.data.Unbounded.html#torchrl.data.Unbounded), and
[`Composite`](generated/torchrl.data.Composite.html#torchrl.data.Composite) containers to validate and generate env
inputs and outputs.

storing_device

The collector device slot where a rollout batch is materialised before it
is yielded or extended into a replay buffer. Direct `replay_buffer.add`
writes bypass this materialisation path.

TED

TorchRL Episode Data: the standard offline dataset layout described in
[Datasets](data_datasets.html#ted-format). It stores a transition with root keys for the current
step and a `("next", ...)` sub-tensordict for next-step values.
Conversion helpers such as [`TED2Flat`](generated/torchrl.data.TED2Flat.html#torchrl.data.TED2Flat) and
[`Flat2TED`](generated/torchrl.data.Flat2TED.html#torchrl.data.Flat2TED) serialize and restore this layout.

tensor_keys

The instance attribute on every [`LossModule`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule)
holding the current values of the keys declared in `_AcceptedKeys`.
Read-only by convention; use
[`set_keys()`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule.set_keys) to modify them.

TensorDictPrimer

A [`Transform`](generated/torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) that injects keys into the
environment's reset / step output that the policy needs but the env does
not natively produce, most commonly RNN hidden states. Without a primer,
the first call to a recurrent policy after reset would have no hidden
state to read. See [`TensorDictPrimer`](generated/torchrl.envs.transforms.TensorDictPrimer.html#torchrl.envs.transforms.TensorDictPrimer) and
[`torchrl.modules.LSTMModule.make_tensordict_primer()`](generated/torchrl.modules.LSTMModule.html#id0).

trajectory ID

An integer that uniquely identifies which trajectory each frame belongs
to. Written by [`Collector`](generated/torchrl.collectors.Collector.html#torchrl.collectors.Collector) as
`("collector", "traj_ids")` when `track_traj_ids=True`. Used by
[`SliceSampler`](generated/torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) to draw whole trajectories from a
buffer and by [`split_trajectories()`](generated/torchrl.collectors.utils.split_trajectories.html#torchrl.collectors.utils.split_trajectories) to
slice a flat batch into per-trajectory chunks. See
[Trajectory boundaries](data_layout.html#ref-traj-boundaries) for how these ids
and the done/truncated/terminated flags are consumed to recover
episode boundaries from a replay buffer.

Transform

TorchRL's tensordict-native environment transform abstraction,
[`Transform`](generated/torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform). A transform can modify input specs,
output specs, reset data, step data, or inverse action data, and is
usually installed through [`TransformedEnv`](generated/torchrl.envs.transforms.TransformedEnv.html#torchrl.envs.transforms.TransformedEnv). This is
distinct from a Gym wrapper, which operates on non-tensordict values.

## See also

- ref_data_layout -- naming conventions for keys in collected batches
- [Collector Internals](collectors_internals.html#ref-collectors-internals) -- where carrier / sync / device flags appear
in the rollout loop
- [Knowledge Base](knowledge_base.html) -- longer-form debugging notes