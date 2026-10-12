# MicroDuckTask

*class*torchrl.envs.MicroDuckTask(*command_low: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *command_high: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *warm_start_velocity: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *warm_start_fraction: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *joint_reset_noise_scale: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *gait_frequency_hz: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *gait_frequency_per_mps: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *reward_weights: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *params: [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)*, *weight: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*, *name: str*, ***, *batch_size*, *device=None*, *names=None*)[[source]](../../_modules/torchrl/envs/custom/mujoco/microduck.html#MicroDuckTask)

One locomotion task of [`MicroDuckEnv`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv), as data.

A task is a row of tensors and a name: the command box, the reset
distribution, the gait clock, the reward weights and parameters, and a
label. A library is a stack of tasks, and every env of a
[`MicroDuckEnv`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv) batch holds one row of the library for the duration
of an episode. Gathering rows with `library[task_id]` copies the name
entries like the tensors, which needs `tensordict>=0.14.1`. Build tasks
with the presets
([`MicroDuckEnv.tracking_task()`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv.tracking_task), [`MicroDuckEnv.standing_task()`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv.standing_task),
[`MicroDuckEnv.speed_range_task()`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv.speed_range_task), [`MicroDuckEnv.sidestep_task()`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv.sidestep_task),
[`MicroDuckEnv.turning_task()`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv.turning_task), [`MicroDuckEnv.jump_task()`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv.jump_task)), which
fill every field and accept
overrides, and stack them with [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) or by handing a
sequence to the env. Stacking is the structural validation: every task
must carry the full reward weight vector and every parameter key.

Parameters:

- **command_low** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - lower corner of the planar command box
`(vx, vy)` in m/s, shape `(2,)`. A fixed command is a box with
`command_low == command_high`.
- **command_high** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - upper corner of the command box, shape
`(2,)`. The command is drawn uniformly in the box at reset and
held for the episode.
- **warm_start_velocity** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - `(low, high)` speed interval in
m/s, shape `(2,)`. At reset, a `warm_start_fraction` of the
envs whose command is nonzero start already moving along the
commanded direction at a speed drawn from it.
- **warm_start_fraction** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - fraction of resets that receive
the warm start, scalar in `[0, 1]`.
- **joint_reset_noise_scale** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - uniform noise added to the
joint positions at reset, in radians, scalar.
- **gait_frequency_hz** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - frequency of the gait clock exposed
in the observation at zero command, scalar.
- **gait_frequency_per_mps** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - increase of the clock frequency
per m/s of commanded speed, scalar.
- **reward_weights** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - one weight per registered reward term,
in the order of [`MicroDuckEnv.REWARD_TERMS`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv.REWARD_TERMS), shape
`(num_terms,)`. A zero weight turns the term off; an all-zero
row is "no built-in reward", for a transform to fill in.
- **params** (*TensorDict*) - scalar parameters read by the reward terms, one
entry per key of [`MicroDuckEnv.REWARD_PARAMS`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv.REWARD_PARAMS) (for example
`tracking_std` or `pose_std`).
- **weight** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - relative weight of the task when the env draws
a task per env at reset, scalar and non-negative.
- **name** (*str*) - label of the task, for logging and evaluation; the presets
derive it from their arguments (`"tracking+0.20"`,
`"sidestep-0.15"`, `"jump"`) and accept `name=` to override.

Examples

Two tasks stacked into a library, one row per env picked at reset:

```
>>> import torch
>>> from torchrl.envs import MicroDuckEnv
>>> library = torch.stack(
... [MicroDuckEnv.tracking_task(0.2, weight=2.0), MicroDuckEnv.jump_task()]
... )
>>> library.shape, library.command_high[:, 0], library.weight
(torch.Size([2]), tensor([0.2000, 0.0000]), tensor([2., 1.]))
>>> list(library.name)
['tracking+0.20', 'jump']
>>> env = MicroDuckEnv(download=True, tasks=library, num_envs=4) 
>>> rollout = env.rollout(20) 
>>> rollout["task_id"][:, 0, 0], rollout["command"][:, 0] 
(tensor([0, 1, 0, 0]), tensor([[0.2, 0.0], [0.0, 0.0], [0.2, 0.0], [0.2, 0.0]]))
```

Overrides retune one row: reward weights by term name, term
parameters and reset fields by name.

```
>>> task = MicroDuckEnv.tracking_task(
... 0.2, reward_weights={"tracking": 4.0}, tracking_std=0.2, warm_start_fraction=0.5,
... warm_start_velocity=(0.1, 0.3),
... )
>>> task.reward_weights[list(MicroDuckEnv.REWARD_TERMS).index("tracking")]
tensor(4.)
>>> task.params["tracking_std"], task.warm_start_fraction
(tensor(0.2000), tensor(0.5000))
```

See also

[`MicroDuckEnv`](torchrl.envs.MicroDuckEnv.html#torchrl.envs.MicroDuckEnv) consumes an ordered library of these rows;
[`MicroDuckTaskSampler`](torchrl.envs.MicroDuckTaskSampler.html#torchrl.envs.MicroDuckTaskSampler) chooses a row at reset; and
[`MicroDuckSkills`](torchrl.modules.tensordict_module.zoo.MicroDuckSkills.html#torchrl.modules.tensordict_module.zoo.MicroDuckSkills)
preserves the library alongside a trained skill policy.

*property*device*: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*

Retrieves the device type of tensor class.

dumps(*prefix: str | None = None*, *copy_existing: bool = False*, ***, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *robust_key: bool | None = True*, *archive: bool | None = None*, *compression: str | int | None = None*) → Self

Saves the tensordict to disk, and returns a memory-mapped copy of it.

`save` (and its alias `dumps`) is `memmap()`: it takes the
same arguments. `self` is not changed. Read the result back with
`load()`. See [Saving TensorDict and tensorclass objects](https://docs.pytorch.org/tensordict/stable/saving.html#saving).

Parameters:

**prefix** (*str**or**Path*) - the directory where the tensordict is
written, one file per leaf. If `prefix` ends with `".tdz"`
or `archive=True`, a single archive file is written instead.

Keyword Arguments:

- **archive** (*bool**,**optional*) - if `True`, write a single archive file.
Defaults to `None`, which writes an archive if `prefix`
ends with `".tdz"`.
- **compression** (*str**or**int**,**optional*) - the compression of the
archive. See `memmap()`.

All the other arguments are those of `memmap()`.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), nested=TensorDict(b=torch.ones(3), batch_size=[3]), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.save(path)
... loaded = TensorDict.load(path)
... print((loaded == td).all())
True
```

extend(*tensordict: list[TensorDictBase] | TensorDictBase*) → None

Appends tensordicts to the lazy stack that the instance holds.

Deprecated since version 0.15: Build a new instance with `tensordict.lazy_stack()` or
[`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) instead.

*classmethod*fields() → tuple[dataclasses.Field, ...]

Returns the fields of the tensorclass.

Deprecated since version 0.15: Use `dataclasses.fields(cls)` instead. It returns the same tuple of
`dataclasses.Field` objects.

from_csv(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *separator: str | None = None*, *dtype: [torch.dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*, ***kwargs*) → Self

Creates a TensorDict from a CSV file.

Requires either pandas or pyarrow to be installed.

Parameters:

**path** (*str**or**Path*) - Path to the CSV file.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - If `True`, the batch size will
be computed automatically. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - If `auto_batch_size` is `True`,
defines how many dimensions the output tensordict should have.
Defaults to `None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device for tensor data.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size. Defaults to
`[num_rows]`.
- **separator** (*str**,**optional*) - If provided, column names are split on
this separator to create nested TensorDicts. Defaults to `None`.
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - If provided, all numeric columns
are cast to this dtype. Defaults to `None`.
- ****kwargs** - Additional keyword arguments forwarded to the CSV reader
(`pandas.read_csv` or `pyarrow.csv.read_csv`).

Returns:

A TensorDict representation of the CSV data.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict(obs=torch.arange(3.0), reward=torch.ones(3), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... path = os.path.join(tmpdir, "data.csv")
... data.to_csv(path)
... td = TensorDict.from_csv(path)
... td32 = TensorDict.from_csv(path, dtype=torch.float32)
>>> td.batch_size, td["obs"].dtype, td32["obs"].dtype
(torch.Size([3]), torch.float64, torch.float32)
```

from_json(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *separator: str | None = None*, *dtype: [torch.dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*, *lines: bool = False*, ***kwargs*) → Self

Creates a TensorDict from a JSON file.

Supports both standard JSON (array of records) and JSON Lines format.
For nested JSON objects, use `from_dict()` instead.

Requires pandas for best results. Falls back to stdlib `json`
for simple cases.

Parameters:

**path** (*str**or**Path*) - Path to the JSON file.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - If `True`, the batch size will
be computed automatically. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - If `auto_batch_size` is `True`,
defines how many dimensions the output tensordict should have.
Defaults to `None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device for tensor data.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size. Defaults to
`[num_rows]`.
- **separator** (*str**,**optional*) - If provided, column names are split on
this separator to create nested TensorDicts. Defaults to `None`.
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - If provided, all numeric columns
are cast to this dtype. Defaults to `None`.
- **lines** (*bool**,**optional*) - If `True`, reads the file as JSON Lines
(one JSON object per line). Defaults to `False`.
- ****kwargs** - Additional keyword arguments forwarded to the JSON
reader.

Returns:

A TensorDict representation of the JSON data.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict(obs=torch.arange(3.0), reward=torch.ones(3), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... path = os.path.join(tmpdir, "data.json")
... data.to_json(path)
... td = TensorDict.from_json(path)
... path_lines = os.path.join(tmpdir, "data.jsonl")
... data.to_json(path_lines, lines=True)
... td_lines = TensorDict.from_json(path_lines, lines=True)
>>> td.batch_size, td_lines.batch_size
(torch.Size([3]), torch.Size([3]))
```

from_module(*as_module: bool = False*, *lock: bool = False*, *use_state_dict: bool = False*, *filter_empty: bool = True*)

Puts the parameters and buffers of a module in a tensordict.

The entries are the parameters and buffers of the module themselves,
not copies. Submodules become nested tensordicts.

Note

This method does not lock the result by default (`lock=False`).
The function `tensordict.from_module()` locks it by default
(`lock=True`).

Parameters:

- **module** (*nn.Module*) - the module to get the parameters from.
- **as_module** (*bool**,**optional*) - if `True`, a `TensorDictParams`
instance is returned, which can be used to store parameters
within a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Defaults to `False`.
- **lock** (*bool**,**optional*) - if `True`, the resulting tensordict is
locked. Defaults to `False`.
- **use_state_dict** (*bool**,**optional*) - if `True`, the state-dict from the
module is used and unflattened into a TensorDict with
the tree structure of the model. Use it when the module has
state-dict hooks. Defaults to `False`.
- **filter_empty** (*bool**,**optional*) - if `True`, submodules without
parameters or buffers (`nn.ReLU`, for example) get no
entry. If `False`, they get an empty tensordict.
Defaults to `True`.

Returns:

a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), or a
`TensorDictParams` if `as_module=True`.

Examples

```
>>> from torch import nn
>>> from tensordict import TensorDict
>>> module = nn.Sequential(nn.Linear(3, 4), nn.ReLU())
>>> params = TensorDict.from_module(module)
>>> params["0", "weight"] is module[0].weight
True
>>> list(params.keys())
['0']
>>> params.is_locked
False
```

from_pandas(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *separator: str | None = None*, *dtype: [torch.dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*) → Self

Converts a pandas DataFrame to a TensorDict.

Numeric columns become tensors, string/object columns become
[`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData).

Parameters:

**dataframe** (*pd.DataFrame*) - The pandas DataFrame to convert.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - If `True`, the batch size will
be computed automatically. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - If `auto_batch_size` is `True`,
defines how many dimensions the output tensordict should have.
Defaults to `None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device for tensor data.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size. Defaults to
`[num_rows]`.
- **separator** (*str**,**optional*) - If provided, column names are split on
this separator to create nested TensorDicts. For example, with
`separator="."`, a column `"obs.x"` becomes
`td["obs", "x"]`. Defaults to `None`.
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - If provided, all numeric columns
are cast to this dtype. Defaults to `None`.

Returns:

A TensorDict representation of the DataFrame.

Examples

```
>>> from tensordict import TensorDict
>>> import pandas as pd
>>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
>>> td = TensorDict.from_pandas(df)
>>> print(td)
TensorDict(
 fields={
 a: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.int64, is_shared=False),
 b: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float64, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
```

from_parquet(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *separator: str | None = None*, *dtype: [torch.dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*, *columns: list[str] | None = None*, ***kwargs*) → Self

Creates a TensorDict from a Parquet file.

Requires either pyarrow or pandas to be installed. Prefers pyarrow
when available for better performance.

Parameters:

**path** (*str**or**Path*) - Path to the Parquet file.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - If `True`, the batch size will
be computed automatically. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - If `auto_batch_size` is `True`,
defines how many dimensions the output tensordict should have.
Defaults to `None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device for tensor data.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size. Defaults to
`[num_rows]`.
- **separator** (*str**,**optional*) - If provided, column names are split on
this separator to create nested TensorDicts. Defaults to `None`.
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - If provided, all numeric columns
are cast to this dtype. Defaults to `None`.
- **columns** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**str**,**optional*) - If provided, only read these
columns from the file. Defaults to `None` (all columns).
- ****kwargs** - Additional keyword arguments forwarded to the Parquet
reader.

Returns:

A TensorDict representation of the Parquet data.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict(obs=torch.arange(3.0), reward=torch.ones(3), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... path = os.path.join(tmpdir, "data.parquet")
... data.to_parquet(path)
... td = TensorDict.from_parquet(path)
... td_obs = TensorDict.from_parquet(path, columns=["obs"])
>>> sorted(td.keys()), sorted(td_obs.keys())
(['obs', 'reward'], ['obs'])
```

from_schema(***, *batch_size: [Sequence](torchrl.data.Sequence.html#torchrl.data.Sequence)[int] | [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *storage: str | None = None*, *device=None*, ***kwargs*) → TensorDictBase

Pre-allocate a zero-filled TensorDict from a schema.

Creates a `TensorDictBase` whose storage backend is selected
by `storage`. Each entry in `schema` maps a field name to an
`(element_shape, dtype)` pair; the full stored shape is
`[*batch_size, *element_shape]`.

Parameters:

**schema** - Mapping from field name to `(element_shape, dtype)`.
`element_shape` is the per-element shape (excluding
`batch_size`).

Keyword Arguments:

- **batch_size** - Overall batch dimensions prepended to every element
shape. Defaults to `()`.
- **storage** (*str**or**None*) -

Backend selector:

- `None` - plain `TensorDict` with regular tensors.
- `"memmap"` - memory-mapped tensors on disk.
Pass `prefix=<dir>` in *kwargs*.
- `"h5"` - HDF5 via `PersistentTensorDict`.
Pass `filename=<path>` in *kwargs*.
- `"zarr"` - zarr (requires `zarr>=3.0`) via
`PersistentTensorDict`. Pass `filename=<path or store>`
in *kwargs*.
- `"shared"` - CPU shared-memory tensors.
- `"redis"` / `"dragonfly"` - delegates to
`TensorDictStore.from_schema()`.
- **device** - Device for the resulting tensors (ignored by some
backends).
- ****kwargs** - Backend-specific arguments forwarded to the
underlying constructor (e.g. `prefix` for memmap,
`filename` for h5, `host`/`port` for redis).

Returns:

A new `TensorDictBase` subclass instance with
pre-allocated (zero-filled) keys.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict.from_schema(
... {"obs": ([84, 84, 3], torch.uint8),
... "reward": ([], torch.float32)},
... batch_size=[1000],
... )
>>> td["obs"].shape
torch.Size([1000, 84, 84, 3])
```

```
>>> import tempfile
>>> with tempfile.TemporaryDirectory() as d:
... td_mm = TensorDict.from_schema(
... {"obs": ([4], torch.float32)},
... batch_size=[8],
... storage="memmap",
... prefix=d,
... )
... assert td_mm.is_memmap()
```

*classmethod*from_tensordict(*tensordict: TensorDictBase*, *non_tensordict: dict | None = None*, *safe: bool | None = None*) → Self

Wraps a tensordict in a new instance of the tensorclass, without copying the leaves.

If `tensordict` is a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), the entries of the fields
annotated with a tensorclass (or `Optional` of one) that are tensordicts become
instances of that tensorclass, recursively. They are set in a shallow copy of
`tensordict`, which keeps its own entries. An entry with keys that the annotated
class does not declare stays a tensordict. Other backends are wrapped as they are:
setting a tensorclass in them would write to their storage (e.g. an H5 file).

Parameters:

- **tensordict** (*TensorDictBase*) - the tensordict that holds the tensor fields.
- **non_tensordict** (*dict**,**optional*) - the values of the non-tensor fields. The
fields that neither argument holds are set to `None`.
- **safe** (*bool**,**optional*) - Deprecated in 0.15, to be removed in TensorDict 0.17. Whether to
raise an error if `tensordict` is not a [`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase).
Without it, such a tensordict raises an error.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Pose(TensorClass):
... q: torch.Tensor
>>> class Obs(TensorClass):
... a: Pose
>>> obs = Obs(a=Pose(q=torch.zeros(5, 4), batch_size=[5]), batch_size=[5])
>>> type(Obs.from_tensordict(obs.to_tensordict()).a).__name__
'Pose'
```

get(*key: NestedKey*, **args*, ***kwargs*)

Gets the value stored with the input key.

Parameters:

- **key** (*str**,**tuple**of**str*) - key to be queried. If tuple of str it is
equivalent to chained calls of getattr.
- **default** - default value if the key is not found in the tensorclass.

Returns:

value stored with the input key

*classmethod*load(*prefix: str | Path*, **args*, ***kwargs*) → Self

Loads a tensordict that `save()` or `memmap()` wrote to disk.

`load` is `load_memmap()`: it takes the same arguments,
including `allow_pickle`, `device` and `mode`. The leaves of the
result are memory-mapped tensors that read the files. See
[Saving TensorDict and tensorclass objects](https://docs.pytorch.org/tensordict/stable/saving.html#saving) and the example of `save()`.

Parameters:

**prefix** (*str**or**Path*) - the directory, or the `".tdz"` archive
file, to read.

load_(*prefix: str | Path*, **args*, ***kwargs*)

Loads a tensordict from disk within the current tensordict.

This method is a proxy to `load_memmap_()`.

Deprecated since version 0.15: Use `load_memmap_()` instead.

*classmethod*load_memmap(*prefix: str | Path*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *non_blocking: bool = False*, ***, *out: TensorDictBase | None = None*, *robust_key: bool | None = True*, *subpath: NestedKey | None = None*, *mode: str | None = None*, *num_threads: int = 0*, *allow_pickle: bool | None = None*) → Self

Loads a memory-mapped tensordict from disk.

Parameters:

- **prefix** (*str**or**Path to folder*) - the path to the folder where the
saved tensordict should be fetched, or the path to a memmap
archive file written through
`save(..., archive=True)` / a `".tdz"` prefix (or packed
with `pack_memmap()`). Archives are
memory-mapped once and every leaf is exposed as a zero-copy
view into the mapping: only the pages of the leaves that are
actually accessed are read from disk. Unlike directory-backed
tensordicts, in-place writes to the leaves of an
archive-loaded tensordict do not propagate to the file by
default (see `mode`).
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**equivalent**,**optional*) - if provided, the
data will be asynchronously cast to that device.
Supports "meta" device, in which case the data isn't loaded
but a set of empty "meta" tensors are created. This is
useful to get a sense of the total model size and structure
without actually opening any file.
- **non_blocking** (*bool**,**optional*) - if `True`, synchronize won't be
called after loading tensors on device. Defaults to `False`.
- **out** (*TensorDictBase**,**optional*) - optional tensordict where the data
should be loaded. Its nested containers are reused, but its
leaves are rebound to the loaded tensors, so their storage is
not reused. Keys of `out` that are missing from the saved
data are removed. If `out` has a device, the data is loaded
on it, and a different `device` raises a `ValueError`.
To write the data into the preallocated storage of `out`,
use `out.update_(TensorDict.load_memmap(prefix, device=device))`
instead.
- **robust_key** (*bool**,**optional*) - if `True` (default), expects robust key encoding was used
when saving and decodes filenames accordingly. If `False`, uses legacy
behavior. If `None`, uses the default robust behavior.
- **subpath** (*NestedKey**or**str path**,**optional*) - the location of a
nested tensordict to load, as a nested key (e.g.
`("module", "0")`, with arbitrary nesting allowed as usual)
or as a `"/"`-separated string path (e.g. `"module/0"`).
Only that subtree is loaded. Works both for directories
(equivalent to appending the path to `prefix`) and for
archives.
- **mode** (*str**,**optional*) - how the files are memory-mapped. With
`"r"`, the mapping is copy-on-write: in-place writes to the
leaves stay in memory. They reach the files only if the
tensordict is saved there, and `memmap_()` without a
prefix does not write back to `prefix`. With `"r+"`, the
mapping is shared: in-place writes propagate to the files,
and a file that is not writable raises a
`PermissionError`. Defaults to `None`, which maps
archives as with `"r"`, and each file of a directory as with
`"r+"` if the process can write it, as with `"r"`
otherwise. Prefer `"r"` for data that is only read: on some
network file systems (e.g. Lustre), page faults on a shared
writable mapping take write locks, so readers on different
nodes block each other. On Linux, copy-on-write mappings count
against the memory commit limit, so a file larger than the
available RAM and swap can fail to map with `"r"`. With
archives, `"r+"` requires uncompressed, aligned tensor
payloads (i.e. archives written by tensordict without
`compression`) and is not available for nested-tensor
leaves. In-place writes do not update the per-entry CRC-32
stored by the zip format; `load_memmap()` ignores
checksums, but call
`refresh_archive_checksums()` before handing
a modified archive to tools that verify them (`unzip`,
`unpack_memmap()`, ...).
- **num_threads** (*int**,**optional*) - number of threads used to decompress
the leaves of a compressed archive (deflate entries are
inflated in parallel, which scales nearly linearly). Without
compression, loading is a metadata-only operation and this
argument has no effect. Defaults to `0` (sequential).
- **allow_pickle** (*bool**,**optional*) - whether pickled non-tensor fields
may be loaded. Pickle can execute arbitrary code, so pass
`True` only for data from a trusted source and `False`
for untrusted data. In TensorDict 0.14 and 0.15, omitting
this option loads pickle with a `FutureWarning`; the
default will change to `False` in 0.16. Saves without
a pickle sidecar do not require this option.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict.fromkeys(["a", "b", "c", ("nested", "e")], 0)
>>> _ = td.memmap("./saved_td")
>>> td_load = TensorDict.load_memmap("./saved_td")
>>> assert (td == td_load).all()
```

This method also allows loading nested tensordicts.

Examples

```
>>> nested = TensorDict.load_memmap("./saved_td/nested")
>>> assert nested["e"] == 0
```

A tensordict can also be loaded on "meta" device or, alternatively,
as a fake tensor.

Examples

```
>>> import tempfile
>>> td = TensorDict({"a": torch.zeros(()), "b": {"c": torch.zeros(())}})
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.save(path)
... td_load = TensorDict.load_memmap(path, device="meta")
... print("meta:", td_load)
... from torch._subclasses import FakeTensorMode
... with FakeTensorMode():
... td_load = TensorDict.load_memmap(path)
... print("fake:", td_load)
meta: TensorDict(
 fields={
 a: Tensor(shape=torch.Size([]), device=meta, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: Tensor(shape=torch.Size([]), device=meta, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=meta,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=meta,
 is_shared=False)
fake: TensorDict(
 fields={
 a: FakeTensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False),
 b: TensorDict(
 fields={
 c: FakeTensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)},
 batch_size=torch.Size([]),
 device=cpu,
 is_shared=False)
```

load_memmap_(*prefix: str | Path*, *robust_key: bool | None = True*, ***, *allow_pickle: bool | None = None*, *mode: str | None = None*)

Loads the content of a memory-mapped tensorclass within the tensorclass where `load_memmap_` is called.

See [`load_memmap_()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase.load_memmap_) for more info.

load_state_dict(*state_dict: dict[str, Any]*, *strict=True*, *assign=False*, *from_flatten=None*)

Loads a state_dict into the tensorclass.

Supports both the new format (logical keys with `_metadata`) and the
legacy format (`_tensordict`/`_non_tensordict` wrapper keys).

memmap(*prefix: str | None = None*, *copy_existing: bool = False*, ***, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *existsok: bool = True*, *robust_key: bool | None = True*, *archive: bool | None = None*, *compression: str | int | None = None*) → Self

Writes all tensors onto a corresponding memory-mapped Tensor in a new tensordict.

Parameters:

- **prefix** (*str*) - directory prefix where the memory-mapped tensors will
be stored. The directory tree structure will mimic the tensordict's.
If `prefix` ends with `".tdz"` (or `archive=True` is
passed), a single-file archive is written instead of a
directory: a standard zip file whose entries replicate the
memmap directory layout. See `archive` below.
- **copy_existing** (*bool*) - If False (default), an exception will be raised if an
entry in the tensordict is already a tensor stored on disk
with an associated file, but is not saved in the correct
location according to prefix.
If `True`, any existing Tensor will be copied to the new location.

Keyword Arguments:

- **num_threads** (*int**,**optional*) - the number of threads used to write the memmap
tensors. Defaults to 0.
- **return_early** (*bool**,**optional*) - if `True` and `num_threads>0`,
the method will return a future of the tensordict.
- **share_non_tensor** (*bool**,**optional*) - if `True`, the non-tensor data will be
shared between the processes and writing operation (such as inplace update
or set) on any of the workers within a single node will update the value
on all other workers. If the number of non_tensor leaves is high (e.g.,
sharing large stacks of non-tensor data) this may result in OOM or similar
errors. Defaults to `False`.
- **existsok** (*bool**,**optional*) - if `False`, an exception will be raised if a tensor already
exists in the same path. Defaults to `True`.
- **robust_key** (*bool**,**optional*) - if `True` (default), uses robust key encoding that safely
handles keys with path separators and special characters. If `False`,
uses legacy behavior (keys used as-is). If `None`, uses the default
robust behavior.
- **archive** (*bool**,**optional*) - if `True`, `prefix` designates a
single file rather than a directory and the tensordict is
written as a memmap archive: a zip file mirroring the memmap
directory tree, with tensor payloads stored uncompressed and
aligned so that `load_memmap()` can memory-map the file
and expose every leaf as a zero-copy view. If `None`
(default), archive mode is enabled when `prefix` ends with
`".tdz"`. The result of `load_memmap()` on an archive
behaves like the result of `from_consolidated()`: all
leaves are views into a single storage, and in-place writes do
not propagate to the file. Archives and memmap directories are
mutually convertible with `pack_memmap()` /
`unpack_memmap()` (or any zip tool). Note that
archives are written sequentially (single data pass) and
`num_threads` has no effect on them. An existing archive
is replaced by a new file rather than rewritten in place: a
symlink is followed, but other hard links to the old file
keep the old contents.
- **compression** (*str**or**int**,**optional*) - compression for archive
entries (`"stored"`, `"deflate"`, `"bzip2"`, `"lzma"`
or a `zipfile` constant). Defaults to `"stored"`
(uncompressed), which is what enables zero-copy loading.
Compressed archives load correctly but leaves are
decompressed in memory on access. Only valid in archive mode.

The TensorDict is then locked, meaning that any writing operations that
isn't in-place will throw an exception (eg, rename, set or remove an
entry).
Once the tensordict is unlocked, the memory-mapped attribute is turned to `False`,
because cross-process identity is not guaranteed anymore.

Returns:

A new tensordict with the tensors stored on disk if `return_early=False`,
otherwise a `TensorDictFuture` instance.

Note

Serialising in this fashion might be slow with deeply nested tensordicts, so
it is not recommended to call this method inside a training loop.

memmap_(*prefix: str | None = None*, *copy_existing: bool = False*, ***, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *existsok: bool = True*, *robust_key: bool | None = True*) → Self

Writes all tensors onto a corresponding memory-mapped Tensor, in-place.

Parameters:

- **prefix** (*str*) - directory prefix where the memory-mapped tensors will
be stored. The directory tree structure will mimic the tensordict's.
- **copy_existing** (*bool*) - If False (default), an exception will be raised if an
entry in the tensordict is already a tensor stored on disk
with an associated file, but is not saved in the correct
location according to prefix.
If `True`, any existing Tensor will be copied to the new location.

Keyword Arguments:

- **num_threads** (*int**,**optional*) - the number of threads used to write the memmap
tensors. Defaults to 0.
- **return_early** (*bool**,**optional*) - if `True` and `num_threads>0`,
the method will return a future of the tensordict. The resulting
tensordict can be queried using future.result().
- **share_non_tensor** (*bool**,**optional*) - if `True`, the non-tensor data will be
shared between the processes and writing operation (such as inplace update
or set) on any of the workers within a single node will update the value
on all other workers. If the number of non-tensor leaves is high (e.g.,
sharing large stacks of non-tensor data) this may result in OOM or similar
errors. Defaults to `False`.
- **existsok** (*bool**,**optional*) - if `False`, an exception will be raised if a tensor already
exists in the same path. Defaults to `True`.
- **robust_key** (*bool**,**optional*) - if `True` (default), uses robust key encoding that safely
handles keys with path separators and special characters. If `False`,
uses legacy behavior (keys used as-is). If `None`, uses the default
robust behavior.

The TensorDict is then locked, meaning that any writing operations that
isn't in-place will throw an exception (eg, rename, set or remove an
entry).
Once the tensordict is unlocked, the memory-mapped attribute is turned to `False`,
because cross-process identity is not guaranteed anymore.

Returns:

self if `return_early=False`, otherwise a `TensorDictFuture` instance.

Note

Serialising in this fashion might be slow with deeply nested tensordicts, so
it is not recommended to call this method inside a training loop.

memmap_like(*prefix: str | None = None*, *copy_existing: bool = False*, ***, *existsok: bool = True*, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *robust_key: bool | None = True*, *archive: bool | None = None*) → Self

Creates a memory-mapped tensordict with the same keys, shapes and dtypes, without copying the values.

The files are filled with zeros. Write the values into the result
afterwards, for example with [`update_()`](torchrl.data.PrioritizedReplayBuffer.html#torchrl.data.PrioritizedReplayBuffer.update_) or indexing.

Parameters:

- **prefix** (*str*) - directory prefix where the memory-mapped tensors will
be stored. The directory tree structure will mimic the tensordict's.
If `prefix` ends with `".tdz"` (or `archive=True` is
passed), a preallocated single-file archive is created
instead. See `archive` below.
- **copy_existing** (*bool*) - If False (default), an exception will be raised if an
entry in the tensordict is already a tensor stored on disk
with an associated file, but is not saved in the correct
location according to prefix.
If `True`, any existing Tensor will be copied to the new location.

Keyword Arguments:

- **num_threads** (*int**,**optional*) - the number of threads used to write the memmap
tensors. Defaults to 0.
- **return_early** (*bool**,**optional*) - if `True` and `num_threads>0`,
the method will return a future of the tensordict.
- **share_non_tensor** (*bool**,**optional*) - if `True`, the non-tensor data will be
shared between the processes and writing operation (such as inplace update
or set) on any of the workers within a single node will update the value
on all other workers. If the number of non-tensor leaves is high (e.g.,
sharing large stacks of non-tensor data) this may result in OOM or similar
errors. Defaults to `False`.
- **existsok** (*bool**,**optional*) - if `False`, an exception will be raised if a tensor already
exists in the same path. Defaults to `True`.
- **robust_key** (*bool**,**optional*) - if `True` (default), uses robust key encoding that safely
handles keys with path separators and special characters. If `False`,
uses legacy behavior (keys used as-is). If `None`, uses the default
robust behavior.
- **archive** (*bool**,**optional*) - if `True`, `prefix` designates a
single file and a preallocated, zero-filled memmap archive is
created and loaded back with
`load_memmap(prefix, mode="r+")`: the returned tensordict
writes through to the archive. If `None` (default), archive
mode is enabled when `prefix` ends with `".tdz"`.
In-place writes leave the zip per-entry checksums stale; call
`refresh_archive_checksums()` before handing
the archive to tools that verify them. Nested tensors are not
supported in this mode.

The TensorDict is then locked, meaning that any writing operations that
isn't in-place will throw an exception (eg, rename, set or remove an
entry).
Once the tensordict is unlocked, the memory-mapped attribute is turned to `False`,
because cross-process identity is not guaranteed anymore.

Returns:

A new `TensorDict` instance with data stored as memory-mapped tensors if `return_early=False`,
otherwise a `TensorDictFuture` instance.

Note

This is the recommended method to write a set of large buffers
on disk, as `memmap_()` will copy the information, which can
be slow for large content.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict({
... "a": torch.zeros((3, 64, 64), dtype=torch.uint8),
... "b": torch.zeros(1, dtype=torch.int64),
... }, batch_size=[]).expand(1_000) # expand does not allocate new memory
>>> import tempfile
>>> with tempfile.TemporaryDirectory() as path:
... buffer = td.memmap_like(path)
... print(buffer.is_memmap(), buffer["a"].shape)
True torch.Size([1000, 3, 64, 64])
```

memmap_refresh_(***, *allow_pickle: bool | None = None*)

Refreshes the content of the memory-mapped tensordict if it has a [`saved_path`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.saved_path).

This method will raise an exception if no path is associated with it.

Parameters:

**allow_pickle** (*bool**,**optional*) - whether pickled non-tensor fields
may be loaded. See `load_memmap()`.

save(*prefix: str | None = None*, *copy_existing: bool = False*, ***, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *robust_key: bool | None = True*, *archive: bool | None = None*, *compression: str | int | None = None*) → Self

Saves the tensordict to disk, and returns a memory-mapped copy of it.

`save` (and its alias `dumps`) is `memmap()`: it takes the
same arguments. `self` is not changed. Read the result back with
`load()`. See [Saving TensorDict and tensorclass objects](https://docs.pytorch.org/tensordict/stable/saving.html#saving).

Parameters:

**prefix** (*str**or**Path*) - the directory where the tensordict is
written, one file per leaf. If `prefix` ends with `".tdz"`
or `archive=True`, a single archive file is written instead.

Keyword Arguments:

- **archive** (*bool**,**optional*) - if `True`, write a single archive file.
Defaults to `None`, which writes an archive if `prefix`
ends with `".tdz"`.
- **compression** (*str**or**int**,**optional*) - the compression of the
archive. See `memmap()`.

All the other arguments are those of `memmap()`.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), nested=TensorDict(b=torch.ones(3), batch_size=[3]), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.save(path)
... loaded = TensorDict.load(path)
... print((loaded == td).all())
True
```

select(**keys*, *inplace: bool = False*, *strict: bool = True*, *as_tensordict: bool = False*)

TensorClass-specific select that supports `as_tensordict`.

set(*key: NestedKey*, *value: Any*, *inplace: bool = False*, *non_blocking: bool = False*)

Sets a new key-value pair.

Parameters:

- **key** (*str**,**tuple**of**str*) - name of the key to be set.
If tuple of str it is equivalent to chained calls of getattr
followed by a final setattr.
- **value** (*Any*) - value to be stored in the tensorclass
- **inplace** (*bool**,**optional*) - if `True`, set will tentatively try to
update the value in-place. If `False` or if the key isn't present,
the value will be simply written at its destination.

Returns:

self

state_dict(*destination=None*, *prefix=''*, *keep_vars=False*, *flatten=True*) → dict[str, Any]

Returns a state_dict with logical keys, matching TensorDictBase conventions.

Tensor fields appear as data keys. Non-tensor fields (strings, ints, etc.)
and the tensorclass type are stored in `_metadata`. This replaces the
legacy `_tensordict`/`_non_tensordict` wrapper format.

to_tensordict(***, *retain_none: bool | None = None*) → TensorDict

Convert the tensorclass into a regular TensorDict.

Makes a copy of all entries. Memmap and shared memory tensors are converted to
regular tensors.

Keyword Arguments:

**retain_none** (*bool**,**optional*) - if `True`, fields set to `None` are
written as `None` entries. Defaults to `None`, which drops
them.

Returns:

A new TensorDict object containing the same values as the tensorclass.

Examples

```
>>> from typing import Any
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... x: torch.Tensor
... y: Any = None
>>> foo = Foo(x=torch.zeros(3))
>>> sorted(foo.to_tensordict().keys())
['x']
>>> sorted(foo.to_tensordict(retain_none=True).keys())
['x', 'y']
```

unbind(*dim: int*)

Returns a tuple of indexed tensorclass instances unbound along the indicated dimension.

Resulting tensorclass instances will share the storage of the initial tensorclass instance.