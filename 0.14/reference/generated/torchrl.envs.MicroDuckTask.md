# MicroDuckTask

*class*torchrl.envs.MicroDuckTask(*command_low: 'torch.Tensor'*, *command_high: 'torch.Tensor'*, *warm_start_velocity: 'torch.Tensor'*, *warm_start_fraction: 'torch.Tensor'*, *joint_reset_noise_scale: 'torch.Tensor'*, *gait_frequency_hz: 'torch.Tensor'*, *gait_frequency_per_mps: 'torch.Tensor'*, *reward_weights: 'torch.Tensor'*, *params: 'TensorDict'*, *weight: 'torch.Tensor'*, *name: 'str'*, ***, *batch_size*, *device=None*, *names=None*)[[source]](../../_modules/torchrl/envs/custom/mujoco/microduck.html#MicroDuckTask)

*property*device*: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*

Retrieves the device type of tensor class.

dumps(*prefix: str | None = None*, *copy_existing: bool = False*, ***, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *robust_key: bool | None = True*, *archive: bool | None = None*, *compression: str | int | None = None*) → Any

Saves the tensordict to disk.

This function is a proxy to `memmap()`.

*classmethod*fields()

Return a tuple describing the fields of this dataclass.

Accepts a dataclass or an instance of one. Tuple elements are of
type Field.

from_csv(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *separator: str | None = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*, ***kwargs*) → Any

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
>>> td = TensorDict.from_csv("data.csv")
>>> td = TensorDict.from_csv("data.csv", separator=".", dtype=torch.float32)
```

from_json(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *separator: str | None = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*, *lines: bool = False*, ***kwargs*) → Any

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
>>> td = TensorDict.from_json("data.json")
>>> td = TensorDict.from_json("data.jsonl", lines=True)
```

from_pandas(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *separator: str | None = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*) → Any

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

from_parquet(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *separator: str | None = None*, *dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*, *columns: list[str] | None = None*, ***kwargs*) → Any

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
>>> td = TensorDict.from_parquet("data.parquet")
>>> td = TensorDict.from_parquet("data.parquet", columns=["obs", "reward"])
```

from_schema(***, *batch_size: Sequence[int] | [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *storage: str | None = None*, *device=None*, ***kwargs*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

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

*classmethod*from_tensordict(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *non_tensordict: dict | None = None*, *safe: bool = True*) → Any

Tensor class wrapper to instantiate a new tensor class object.

Parameters:

- **tensordict** (*TensorDictBase*) - Dictionary of tensor types
- **non_tensordict** (*dict*) - Dictionary with non-tensor and nested tensor class objects
- **safe** (*bool*) - Whether to raise an error if the tensordict is not a TensorDictBase instance

get(*key: NestedKey*, **args*, ***kwargs*)

Gets the value stored with the input key.

Parameters:

- **key** (*str**,**tuple**of**str*) - key to be queried. If tuple of str it is
equivalent to chained calls of getattr.
- **default** - default value if the key is not found in the tensorclass.

Returns:

value stored with the input key

*classmethod*load(*prefix: str | Path*, **args*, ***kwargs*) → Any

Loads a tensordict from disk.

This class method is a proxy to `load_memmap()`.

load_(*prefix: str | Path*, **args*, ***kwargs*)

Loads a tensordict from disk within the current tensordict.

This class method is a proxy to `load_memmap_()`.

*classmethod*load_memmap(*prefix: str | Path*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *non_blocking: bool = False*, ***, *out: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *robust_key: bool | None = True*, *subpath: NestedKey | None = None*, *mode: str = 'r'*, *num_threads: int = 0*, *allow_pickle: bool | None = None*) → Any

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
archive-loaded tensordict do not propagate to the file.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**equivalent**,**optional*) - if provided, the
data will be asynchronously cast to that device.
Supports "meta" device, in which case the data isn't loaded
but a set of empty "meta" tensors are created. This is
useful to get a sense of the total model size and structure
without actually opening any file.
- **non_blocking** (*bool**,**optional*) - if `True`, synchronize won't be
called after loading tensors on device. Defaults to `False`.
- **out** (*TensorDictBase**,**optional*) - optional tensordict where the data
should be written.
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
- **mode** (*str**,**optional*) - `"r"` (default) or `"r+"`. Only
relevant when loading an archive: with `"r"` the archive is
mapped copy-on-write and in-place writes to the leaves stay
in memory; with `"r+"` the mapping is shared and in-place
writes propagate to the file, like directory-backed
tensordicts. `"r+"` requires uncompressed, aligned tensor
payloads (i.e. archives written by tensordict without
`compression`) and is not available for nested-tensor
leaves. In-place writes do not update the per-entry CRC-32
stored by the zip format; `load_memmap()` ignores
checksums, but call
`refresh_archive_checksums()` before handing
a modified archive to tools that verify them (`unzip`,
`unpack_memmap()`, ...). Directory prefixes
are always write-through and ignore this argument.
- **num_threads** (*int**,**optional*) - number of threads used to decompress
the leaves of a compressed archive (deflate entries are
inflated in parallel, which scales nearly linearly). Without
compression, loading is a metadata-only operation and this
argument has no effect. Defaults to `0` (sequential).
- **allow_pickle** (*bool**,**optional*) - whether pickled non-tensor fields
may be loaded. Pickle can execute arbitrary code, so pass
`True` only for data from a trusted source and `False`
for untrusted data. During the 0.14 compatibility window,
omitting this option loads pickle with a `FutureWarning`;
the default will change to `False` in 0.15. Saves without
a pickle sidecar do not require this option.

Examples

```
>>> from tensordict import TensorDict
>>> td = TensorDict.fromkeys(["a", "b", "c", ("nested", "e")], 0)
>>> td.memmap("./saved_td")
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
... td.save(path)
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

load_state_dict(*state_dict: dict[str, Any]*, *strict=True*, *assign=False*, *from_flatten=None*)

Loads a state_dict into the tensorclass.

Supports both the new format (logical keys with `_metadata`) and the
legacy format (`_tensordict`/`_non_tensordict` wrapper keys).

memmap(*prefix: str | None = None*, *copy_existing: bool = False*, ***, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *existsok: bool = True*, *robust_key: bool | None = True*, *archive: bool | None = None*, *compression: str | int | None = None*) → Any

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
`num_threads` has no effect on them.
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

memmap_(*prefix: str | None = None*, *copy_existing: bool = False*, ***, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *existsok: bool = True*, *robust_key: bool | None = True*) → Any

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

memmap_like(*prefix: str | None = None*, *copy_existing: bool = False*, ***, *existsok: bool = True*, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *robust_key: bool | None = True*, *archive: bool | None = None*) → Any

Creates a contentless Memory-mapped tensordict with the same shapes as the original one.

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
>>> td = TensorDict({
... "a": torch.zeros((3, 64, 64), dtype=torch.uint8),
... "b": torch.zeros(1, dtype=torch.int64),
... }, batch_size=[]).expand(1_000_000) # expand does not allocate new memory
>>> buffer = td.memmap_like("/path/to/dataset")
```

memmap_refresh_(***, *allow_pickle: bool | None = None*)

Refreshes the content of the memory-mapped tensordict if it has a [`saved_path`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict.saved_path).

This method will raise an exception if no path is associated with it.

Parameters:

**allow_pickle** (*bool**,**optional*) - whether pickled non-tensor fields
may be loaded. See `load_memmap()`.

save(*prefix: str | None = None*, *copy_existing: bool = False*, ***, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *robust_key: bool | None = True*, *archive: bool | None = None*, *compression: str | int | None = None*) → Any

Saves the tensordict to disk.

This function is a proxy to `memmap()`.

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

to_tensordict(***, *retain_none: bool | None = None*) → [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)

Convert the tensorclass into a regular TensorDict.

Makes a copy of all entries. Memmap and shared memory tensors are converted to
regular tensors.

Parameters:

**retain_none** (*bool*) - if `True`, the `None` values will be written in the
tensordict. Otherwise they will be discrarded. Default: `True`.

Returns:

A new TensorDict object containing the same values as the tensorclass.

unbind(*dim: int*)

Returns a tuple of indexed tensorclass instances unbound along the indicated dimension.

Resulting tensorclass instances will share the storage of the initial tensorclass instance.