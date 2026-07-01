# VLAImages

*class*torchrl.data.vla.VLAImages(*image: 'torch.Tensor | None' = None*, *wrist_image: 'torch.Tensor | None' = None*, *extra: 'TensorDictBase | None' = None*, *padded: 'bool | None' = None*, ***, *batch_size*, *device=None*, *names=None*)[[source]](../../_modules/torchrl/data/vla/containers.html#VLAImages)

cat(*dim: int = 0*, ***, *out=None*)

Concatenates tensordicts into a single tensordict along the given dimension.

This call is equivalent to calling [`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) but is compatible with torch.compile.

*property*device*: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*

Retrieves the device type of tensor class.

*classmethod*fields()

Return a tuple describing the fields of this dataclass.

Accepts a dataclass or an instance of one. Tuple elements are of
type Field.

from_any(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*)

Recursively converts any object to a TensorDict.

Note

`from_any` is less restrictive than the regular TensorDict constructor. It can cast data structures like
dataclasses or tuples to a tensordict using custom heuristics. This approach may incur some extra overhead and
involves more opinionated choices in terms of mapping strategies.

Note

This method recursively converts the input object to a TensorDict. If the object is already a
TensorDict (or any similar tensor collection object), it will be returned as is.

Parameters:

**obj** - The object to be converted.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - if `True`, the batch size will be computed automatically.
Defaults to `False`.
- **batch_dims** (*int**,**optional*) - If auto_batch_size is `True`, defines how many dimensions the output tensordict
should have. Defaults to `None` (full batch-size at each level).
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which the TensorDict will be created.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict.
Exclusive with `auto_batch_size`.

Returns:

A TensorDict representation of the input object.

Supported objects:

- Dataclasses through `from_dataclass()` (dataclasses will be converted to TensorDict instances, not tensorclasses).
- Namedtuples through `from_namedtuple()`.
- Dictionaries through `from_dict()`.
- Tuples through `from_tuple()`.
- NumPy's structured arrays through `from_struct_array()`.
- HDF5 objects through `from_h5()`.

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

from_dataclass(***, *dest_cls: Type | None = None*, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *as_tensorclass: bool = False*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*)

Converts a dataclass into a TensorDict instance.

Parameters:

**dataclass** - The dataclass instance to be converted.

Keyword Arguments:

- **dest_cls** (*tensorclass**,**optional*) - A tensorclass type to be used to map the data. If not provided, a new
class is created. Without effect if `obj` is a type or as_tensorclass is False.
- **auto_batch_size** (*bool**,**optional*) - If `True`, automatically determines and applies batch size to the
resulting TensorDict. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - If `auto_batch_size` is `True`, defines how many dimensions the output
tensordict should have. Defaults to `None` (full batch-size at each level).
- **as_tensorclass** (*bool**,**optional*) - If `True`, delegates the conversion to the free function
`from_dataclass()` and returns a tensor-compatible class (`tensorclass()`)
or instance instead of a TensorDict. Defaults to `False`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which the TensorDict will be created.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict.
Defaults to `None`.

Returns:

A TensorDict instance derived from the provided dataclass, unless as_tensorclass is True, in which case a tensor-compatible class or instance is returned.

Raises:

**TypeError** - If the provided input is not a dataclass instance.

Warning

This method is distinct from the free function from_dataclass and serves a different purpose.
While the free function returns a tensor-compatible class or instance, this method returns a TensorDict instance.

Note

- This method creates a new TensorDict instance with keys corresponding to the fields of the input dataclass.
- Each key in the resulting TensorDict is initialized using the cls.from_any method.
- The auto_batch_size option allows for automatic batch size determination and application to the
resulting TensorDict.

from_h5(***, *mode: str = 'r'*, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*)

Creates a PersistentTensorDict from a h5 file.

Parameters:

**filename** (*str*) - The path to the h5 file.

Keyword Arguments:

- **mode** (*str**,**optional*) - Reading mode. Defaults to `"r"`.
- **auto_batch_size** (*bool**,**optional*) - If `True`, the batch size will be computed automatically.
Defaults to `False`.
- **batch_dims** (*int**,**optional*) - If auto_batch_size is `True`, defines how many dimensions the output
tensordict should have. Defaults to `None` (full batch-size at each level).
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict. Defaults to `None`.

Returns:

A PersistentTensorDict representation of the input h5 file.

Examples

```
>>> td = TensorDict.from_h5("path/to/file.h5")
>>> print(td)
PersistentTensorDict(
 fields={
 key1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 key2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
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

from_modules(***, *as_module: bool = False*, *lock: bool = True*, *use_state_dict: bool = False*, *lazy_stack: bool = False*, *expand_identical: bool = False*)

Retrieves the parameters of several modules for ensebmle learning/feature of expects applications through vmap.

Parameters:

**modules** (*sequence**of**nn.Module*) - the modules to get the parameters from.
If the modules differ in their structure, a lazy stack is needed
(see the `lazy_stack` argument below).

Keyword Arguments:

- **as_module** (*bool**,**optional*) - if `True`, a `TensorDictParams`
instance will be returned which can be used to store parameters
within a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Defaults to `False`.
- **lock** (*bool**,**optional*) - if `True`, the resulting tensordict will be locked.
Defaults to `True`.
- **use_state_dict** (*bool**,**optional*) -

if `True`, the state-dict from the
module will be used and unflattened into a TensorDict with
the tree structure of the model. Defaults to `False`.

Note

This is particularly useful when state-dict hooks have to be used.
- **lazy_stack** (*bool**,**optional*) -

whether parameters should be densly or
lazily stacked. Defaults to `False` (dense stack).

Note

`lazy_stack` and `as_module` are exclusive features.

Warning

There is a crucial difference between lazy and non-lazy outputs
in that non-lazy output will reinstantiate parameters with the
desired batch-size, while `lazy_stack` will just represent
the parameters as lazily stacked. This means that whilst the
original parameters can safely be passed to an optimizer
when `lazy_stack=True`, the new parameters need to be passed
when it is set to `True`.

Warning

Whilst it can be tempting to use a lazy stack to keep the
orignal parameter references, remember that lazy stack
perform a stack each time [`get()`](torchrl.data.Composite.html#torchrl.data.Composite.get) is called. This will
require memory (N times the size of the parameters, more if a
graph is built) and time to be computed.
It also means that the optimizer(s) will contain more
parameters, and operations like [`step()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.step.html#torch.optim.Optimizer.step)
or [`zero_grad()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad) will take longer
to be executed. In general, `lazy_stack` should be reserved
to very few use cases.
- **expand_identical** (*bool**,**optional*) - if `True` and the same parameter (same
identity) is being stacked to itself, an expanded version of this parameter
will be returned instead. This argument is ignored when `lazy_stack=True`.

Examples

```
>>> from torch import nn
>>> from tensordict import TensorDict
>>> torch.manual_seed(0)
>>> empty_module = nn.Linear(3, 4, device="meta")
>>> n_models = 2
>>> modules = [nn.Linear(3, 4) for _ in range(n_models)]
>>> params = TensorDict.from_modules(*modules)
>>> print(params)
TensorDict(
 fields={
 bias: Parameter(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 weight: Parameter(shape=torch.Size([2, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False)
>>> # example of batch execution
>>> def exec_module(params, x):
... with params.to_module(empty_module):
... return empty_module(x)
>>> x = torch.randn(3)
>>> y = torch.vmap(exec_module, (0, None))(params, x)
>>> assert y.shape == (n_models, 4)
>>> # since lazy_stack = False, backprop leaves the original params untouched
>>> y.sum().backward()
>>> assert params["weight"].grad.norm() > 0
>>> assert modules[0].weight.grad is None
```

With `lazy_stack=True`, things are slightly different:

```
>>> params = TensorDict.from_modules(*modules, lazy_stack=True)
>>> print(params)
LazyStackedTensorDict(
 fields={
 bias: Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 weight: Tensor(shape=torch.Size([2, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
 exclusive_fields={
 },
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False,
 stack_dim=0)
>>> # example of batch execution
>>> y = torch.vmap(exec_module, (0, None))(params, x)
>>> assert y.shape == (n_models, 4)
>>> y.sum().backward()
>>> assert modules[0].weight.grad is not None
```

from_namedtuple(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*)

Converts a namedtuple to a TensorDict recursively.

Parameters:

**named_tuple** - The namedtuple instance to be converted.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - if `True`, the batch size will be computed automatically.
Defaults to `False`.
- **batch_dims** (*int**,**optional*) - If `auto_batch_size` is `True`, defines how many dimensions the output
tensordict should have. Defaults to `None` (full batch-size at each level).
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which the TensorDict will be created.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict.
Defaults to `None`.

Returns:

A TensorDict representation of the input namedtuple.

Examples

```
>>> from tensordict import TensorDict
>>> import torch
>>> data = TensorDict({
... "a_tensor": torch.zeros((3)),
... "nested": {"a_tensor": torch.zeros((3)), "a_string": "zero!"}}, [3])
>>> nt = data.to_namedtuple()
>>> print(nt)
GenericDict(a_tensor=tensor([0., 0., 0.]), nested=GenericDict(a_tensor=tensor([0., 0., 0.]), a_string='zero!'))
>>> TensorDict.from_namedtuple(nt, auto_batch_size=True)
TensorDict(
 fields={
 a_tensor: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
 nested: TensorDict(
 fields={
 a_string: NonTensorData(data=zero!, batch_size=torch.Size([3]), device=None),
 a_tensor: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
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

from_pytree(***, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *auto_batch_size: bool = False*, *batch_dims: int | None = None*)

Converts a pytree to a TensorDict instance.

This method is designed to keep the pytree nested structure as much as possible.

Additional non-tensor keys are added to keep track of each level's identity, providing
a built-in pytree-to-tensordict bijective transform API.

Accepted classes currently include lists, tuples, named tuples and dict.

Note

For dictionaries, non-NestedKey keys are registered separately as [`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData)
instances.

Note

Tensor-castable types (such as int, float or np.ndarray) will be converted to torch.Tensor instances.
Note that this transformation is surjective: transforming back the tensordict to a pytree will not
recover the original types.

Examples

```
>>> # Create a pytree with tensor leaves, and one "weird"-looking dict key
>>> class WeirdLookingClass:
... pass
...
>>> weird_key = WeirdLookingClass()
>>> # Make a pytree with tuple, lists, dict and namedtuple
>>> pytree = (
... [torch.randint(10, (3,)), torch.zeros(2)],
... {
... "tensor": torch.randn(
... 2,
... ),
... "td": TensorDict({"one": 1}),
... weird_key: torch.randint(10, (2,)),
... "list": [1, 2, 3],
... },
... {"named_tuple": TensorDict({"two": torch.ones(1) * 2}).to_namedtuple()},
... )
>>> # Build a TensorDict from that pytree
>>> td = TensorDict.from_pytree(pytree)
>>> # Recover the pytree
>>> pytree_recon = td.to_pytree()
>>> # Check that the leaves match
>>> def check(v1, v2):
>>> assert (v1 == v2).all()
>>>
>>> torch.utils._pytree.tree_map(check, pytree, pytree_recon)
>>> assert weird_key in pytree_recon[1]
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

from_struct_array(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*) → Any

Converts a structured numpy array to a TensorDict.

The resulting TensorDict will share the same memory content as the numpy array (it is a zero-copy operation).
Changing values of the structured numpy array in-place will affect the content of the TensorDict.

Note

This method performs a zero-copy operation, meaning that the resulting TensorDict will share the same memory
content as the input numpy array. Therefore, changing values of the numpy array in-place will affect the content
of the TensorDict.

Parameters:

**struct_array** (*np.ndarray*) - The structured numpy array to be converted.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - If `True`, the batch size will be computed automatically. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - If `auto_batch_size` is `True`, defines how many dimensions the output
tensordict should have. Defaults to `None` (full batch-size at each level).
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) -

The device on which the TensorDict will be created.
Defaults to `None`.

Note

Changing the device (i.e., specifying any device other than `None` or `"cpu"`) will transfer the data,
resulting in a change to the memory location of the returned data.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict. Defaults to None.

Returns:

A TensorDict representation of the input structured numpy array.

Examples

```
>>> x = np.array(
... [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
... dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
... )
>>> td = TensorDict.from_struct_array(x)
>>> x_recon = td.to_struct_array()
>>> assert (x_recon == x).all()
>>> assert x_recon.shape == x.shape
>>> # Try modifying x age field and check effect on td
>>> x["age"] += 1
>>> assert (td["age"] == np.array([10, 4])).all()
```

*classmethod*from_tensordict(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*, *non_tensordict: dict | None = None*, *safe: bool = True*) → Any

Tensor class wrapper to instantiate a new tensor class object.

Parameters:

- **tensordict** (*TensorDictBase*) - Dictionary of tensor types
- **non_tensordict** (*dict*) - Dictionary with non-tensor and nested tensor class objects
- **safe** (*bool*) - Whether to raise an error if the tensordict is not a TensorDictBase instance

from_tuple(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*)

Converts a tuple to a TensorDict.

Parameters:

**obj** - The tuple instance to be converted.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - If `True`, the batch size will be computed automatically. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - If auto_batch_size is `True`, defines how many dimensions the output tensordict
should have. Defaults to `None` (full batch-size at each level).
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device on which the TensorDict will be created. Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*,**optional*) - The batch size of the TensorDict. Defaults to `None`.

Returns:

A TensorDict representation of the input tuple.

Examples

```
>>> my_tuple = (1, 2, 3)
>>> td = TensorDict.from_tuple(my_tuple)
>>> print(td)
TensorDict(
 fields={
 0: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 1: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False),
 2: Tensor(shape=torch.Size([]), device=cpu, dtype=torch.int64, is_shared=False)},
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
```

fromkeys(*value: Any = 0*)

Creates a tensordict from a list of keys and a single value.

Parameters:

- **keys** (*list**of**NestedKey*) - An iterable specifying the keys of the new dictionary.
- **value** (*compatible type**,**optional*) - The value for all keys. Defaults to `0`.

lazy_stack(*dim: int = 0*, ***, *out=None*, ***kwargs*)

Creates a lazy stack of tensordicts.

See `lazy_stack()` for details.

load(**args*, ***kwargs*) → Any

Loads a tensordict from disk.

This class method is a proxy to `load_memmap()`.

load_memmap(*device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *non_blocking: bool = False*, ***, *out: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *robust_key: bool | None = True*) → Any

Loads a memory-mapped tensordict from disk.

Parameters:

- **prefix** (*str**or**Path to folder*) - the path to the folder where the
saved tensordict should be fetched.
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

maybe_dense_stack(*dim: int = 0*, ***, *out=None*, ***kwargs*)

Attempts to make a dense stack of tensordicts, and falls back on lazy stack when required..

See `maybe_dense_stack()` for details.

stack(*dim: int = 0*, ***, *out=None*)

Stacks tensordicts into a single tensordict along the given dimension.

This call is equivalent to calling [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack) but is compatible with torch.compile.