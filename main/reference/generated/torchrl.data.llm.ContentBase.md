# ContentBase

*class*torchrl.data.llm.ContentBase(*type: Literal['text', 'image', 'audio', 'video', 'file', 'function_call']*, *text: str | None = None*, *url: str | None = None*, *data: str | None = None*, *mime_type: str | None = None*, *name: str | None = None*, *size: int | None = None*, *function_name: str | None = None*, *function_args: dict | None = None*, ***, *batch_size*, *device=None*, *names=None*)[[source]](../../_modules/torchrl/data/llm/history.html#ContentBase)

Base class for all message content types.

Variables:

- **type** (*str*) - The type of the content.
- **text** (*str**,**optional*) - The text content.
- **url** (*str**,**optional*) - The URL content.
- **data** (*str**,**optional*) - The data content.
- **mime_type** (*str**,**optional*) - The MIME type of the content.
- **name** (*str**,**optional*) - The name of the content.
- **size** (*int**,**optional*) - The size of the content.
- **function_name** (*str**,**optional*) - The name of the function.
- **function_args** (*dict**,**optional*) - The arguments of the function.

Examples

```
>>> from tensordict import lazy_stack
>>> content1 = ContentBase(type="text", text="Hello, world!")
>>> print(content1)
ContentBase(
 text=NonTensorData(data=Hello, world!, batch_size=torch.Size([]), device=None),
 type=NonTensorData(data=text, batch_size=torch.Size([]), device=None),
 url=None,
 data=None,
 mime_type=None,
 name=None,
 size=None,
 function_name=None,
 function_args=None,
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> content2 = ContentBase(type="image", url="https://example.com/image.jpg")
>>> print(content2)
ContentBase(
 type=NonTensorData(data=image, batch_size=torch.Size([]), device=None),
 url=NonTensorData(data=https://example.com/image.jpg, batch_size=torch.Size([]), device=None),
 text=None,
 data=None,
 mime_type=None,
 name=None,
 size=None,
 function_name=None,
 function_args=None,
 batch_size=torch.Size([]),
 device=None,
 is_shared=False)
>>> content = lazy_stack([content1, content2])
>>> print(content)
ContentBase(
 type=NonTensorStack(
 ['text', 'image'],
 batch_size=torch.Size([2]),
 device=None),
 url=None,
 data=None,
 mime_type=None,
 name=None,
 size=None,
 function_name=None,
 function_args=None,
 text=None,
 batch_size=torch.Size([2]),
 device=None,
 is_shared=False)
>>> # A content is typically used in a History object. Usually, its batch dimension is
>>> # one dimension greater than the History object.
>>> history = History(role="user", content=content)
```

cat(*dim: int = 0*, ***, *out=None*)

Concatenates tensordicts along an existing batch dimension `dim`.

It is equivalent to [`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat), and also works under
[`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile). The tensordicts must have the same keys, or a
`KeyError` is raised, and the same batch size except along `dim`.

Parameters:

- **input** (*sequence**of**TensorDictBase*) - the tensordicts to concatenate.
- **dim** (*int**,**optional*) - the batch dimension to concatenate along.
Defaults to `0`.

Keyword Arguments:

**out** (*TensorDictBase**,**optional*) - a tensordict to write the result
into. Defaults to `None`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> a = TensorDict(x=torch.zeros(3), batch_size=[3])
>>> b = TensorDict(x=torch.ones(3), batch_size=[3])
>>> TensorDict.cat([a, b]).batch_size
torch.Size([6])
```

*property*data*: Any*

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

*property*device*: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*

The device of the tensorclass, or `None` if its leaves may be on different devices.

It is read-only: use [`to()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.to.html#tensordict.TensorDictBase.to) to move the
tensorclass to another device.

dumps(*prefix: str | Path | None = None*, *copy_existing: bool = False*, ***, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *robust_key: bool | None = True*, *archive: bool | None = None*, *compression: str | int | None = None*, *atomic: bool = False*) → Self

Saves the tensordict to disk, and returns a memory-mapped copy of it.

`save` (and its alias `dumps`) calls `memmap()` with
`existsok=True`. `memmap()` documents the arguments in full.
The tensordict is not changed. Read the files back with `load()`.
See [Save and load tensordicts](https://docs.pytorch.org/tensordict/stable/saving.html#saving).

Parameters:

- **prefix** (*str**or**Path**,**optional*) - the directory where the tensordict
is written, one file per leaf. If `prefix` ends with
`".tdz"` or `archive=True`, a single archive file is
written instead. Defaults to `None`, which writes into the
directory of the tensordict if it is already memory-mapped
there, and otherwise into anonymous shared memory, which has no
file name and lasts as long as the process.
- **copy_existing** (*bool**,**optional*) - if `False`, a leaf that is
already a memory-mapped tensor with a file at another path
raises a `RuntimeError`. If `True`, the file is copied to
the new path. Defaults to `False`.

Keyword Arguments:

- **num_threads** (*int**,**optional*) - the number of threads that write the
files. Defaults to `0`, which writes them in the calling
thread.
- **return_early** (*bool**,**optional*) - if `True` and `num_threads > 1`,
returns a [`TensorDictFuture`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) while
the threads write the files. Defaults to `False`.
- **share_non_tensor** (*bool**,**optional*) - if `True`, non-tensor data is
shared between the processes of the node. Defaults to
`False`.
- **robust_key** (*bool**,**optional*) - if `True`, keys are percent-encoded
in file names, so that every file stays inside `prefix`.
Defaults to `True`.
- **archive** (*bool**,**optional*) - if `True`, writes a single archive
file. Defaults to `None`, which writes an archive if
`prefix` ends with `".tdz"`.
- **compression** (*str**or**int**,**optional*) - the compression of the
archive entries. Defaults to `None`, which stores them
uncompressed.
- **atomic** (*bool**,**optional*) - if `True`, the directory is written
next to `prefix` and then renamed to it, so that an error or
a crash during the write leaves `prefix` as it was. Not
supported on Windows for a directory. An archive is always
written this way.
`memmap()` gives the details. Defaults to `False`.

Returns:

A new memory-mapped tensordict, as `memmap()` returns.

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

Deprecated since version 0.15: Build a new instance with [`tensordict.lazy_stack()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.lazy_stack.html#tensordict.lazy_stack) or
[`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) instead.

*classmethod*fields() → tuple[dataclasses.Field, ...]

Returns the fields of the tensorclass.

Deprecated since version 0.15: Use `dataclasses.fields(cls)` instead. It returns the same tuple of
`dataclasses.Field` objects.

from_any(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*)

Converts a dict, tuple, dataclass, array or file to a tensordict, recursively.

`from_any` picks the conversion from the type of `obj`, and
applies it to the values inside `obj` too. It accepts more types
than the [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) constructor, at the cost of
some overhead and of opinionated choices about how each type maps to
entries:

- a dict or `collections.UserDict`: `from_dict()`;
- a named tuple: `from_namedtuple()`;
- a tuple: `from_tuple()`, with the keys `"0"`, `"1"`, ...
This includes the results of `torch.return_types` (`max`,
`sort`, `topk`, ...), which also emit a `FutureWarning`.
Starting with TensorDict 0.17, the default will change to their
field names (`"values"`, `"indices"`, ...), as
`from_namedtuple()` gives;
- a dataclass instance: `from_dataclass()` (the result is a
tensordict, not a tensorclass). A dataclass type whose fields all
have defaults gives a tensordict of the defaults too, with a
`FutureWarning`. Starting with TensorDict 0.17, the default
will change to returning the type unchanged, as for any other
class;
- a NumPy structured array: `from_struct_array()`;
- a pandas DataFrame: `from_pandas()`;
- an open `h5py.File`: a
[`PersistentTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) (see `from_h5()`).

A tensor or a tensor collection is returned as it is. A list becomes a
tensor if [`torch.tensor()`](https://docs.pytorch.org/docs/stable/generated/torch.tensor.html#torch.tensor) takes it, and a
[`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack) otherwise. Any other object is
returned unchanged.

Parameters:

**obj** - the object to convert.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`, except for a dict, where
`batch_size` wins, and for an `h5py.File`, where
`auto_batch_size` wins. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**str**or**int**,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`, except for a dict, where
`batch_size` wins and no batch size is computed, and for an
`h5py.File`, where `auto_batch_size` wins and
`batch_size` is ignored. Defaults to `None`.

Returns:

A tensordict (a [`PersistentTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) for an
HDF5 file), a tensor or a [`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack) for
a list, or `obj` itself for the other types.

Examples

```
>>> import dataclasses
>>> import torch
>>> from tensordict import TensorDict
>>> @dataclasses.dataclass
... class Data:
... obs: torch.Tensor
... extra: tuple
>>> data = Data(obs=torch.zeros(3, 4), extra=(torch.ones(3), {"x": torch.ones(3)}))
>>> td = TensorDict.from_any(data, auto_batch_size=True)
>>> td.batch_size
torch.Size([3])
>>> sorted(td.keys(include_nested=True, leaves_only=True), key=str)
[('extra', '0'), ('extra', '1', 'x'), 'obs']
>>> TensorDict.from_any([1, 2, 3])
tensor([1, 2, 3])
```

from_consolidated()

Reconstructs a tensordict from a file written by `consolidate(filename=...)`.

The result is consolidated: its leaves are views into one
copy-on-write memory map of the file, so in-place writes into them do
not reach the file. It is not locked. The batch sizes and names are as
they were when the file was written. Same as
[`from_consolidated()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.from_consolidated.html#tensordict.from_consolidated).

Parameters:

**filename** (*str**or**Path*) - the file to read.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), b={"c": torch.ones(3)}, batch_size=[3])
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.consolidate(filename=f"{path}/td.bin")
... loaded = TensorDict.from_consolidated(f"{path}/td.bin")
... print(loaded["b", "c"], loaded.is_consolidated())
tensor([1., 1., 1.]) True
```

from_csv(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *separator: str | None = None*, *dtype: [torch.dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*, ***kwargs*) → Self

Reads a CSV file into a tensordict, with one entry per column.

Requires pandas or pyarrow: the file is read with
`pandas.read_csv()` when pandas is installed, and with
`pyarrow.csv.read_csv` otherwise. The columns are converted as in
`from_pandas()`. `to_csv()` writes such a file. Through
pandas, a string column is one value for the whole batch (see the
note in `from_pandas()`); through pyarrow, it is a
[`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack) with one value per row.

Parameters:

**path** (*str**or**Path*) - the path of the CSV file.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**str**or**int**,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`,
which gives `[num_rows]`.
- **separator** (*str**,**optional*) - if given, splits the column names on it
to build nested entries: with `separator="."`, the column
`"obs.x"` becomes the entry `("obs", "x")`. Defaults to
`None`, which keeps the column names as keys.
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - if given, casts the numeric and
boolean columns to it. Defaults to `None`, which keeps the
dtype of each column.
- ****kwargs** - passed to the CSV reader.

Returns:

A tensordict of the class this method is called on
([`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) for `TensorDictBase`).

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

from_dataclass(***, *dest_cls: Type | None = None*, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *as_tensorclass: bool = False*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*)

Converts a dataclass instance to a tensordict, with one entry per field.

Each field value is converted with `from_any()`, so a nested
dataclass or dict becomes a nested tensordict and a number becomes a
tensor. With `as_tensorclass=True`, the conversion is that of the
function [`from_dataclass()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.from_dataclass.html#tensordict.from_dataclass) instead, which returns a
tensorclass.

Parameters:

**dataclass** - the dataclass instance to convert. With
`as_tensorclass=True`, it can also be a dataclass type.
Without it, a dataclass type whose fields all have defaults
gives a tensordict of the defaults, with a
`FutureWarning`. Starting with TensorDict 0.17, the
default will change to raising a `TypeError` for it.

Keyword Arguments:

- **dest_cls** (*type**,**optional*) - with `as_tensorclass=True`, the
tensorclass to convert to. Ignored otherwise. Defaults to
`None`, which creates a new tensorclass with the fields of
the dataclass.
- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **as_tensorclass** (*bool**,**optional*) - if `True`, returns a
tensorclass instance (or a tensorclass type, for a dataclass
type) through [`from_dataclass()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.from_dataclass.html#tensordict.from_dataclass). Defaults to
`False`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**str**or**int**,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`, an
empty batch size.

Returns:

A [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), or a tensorclass instance or
type if `as_tensorclass=True`.

Raises:

**TypeError** - if `dataclass` is not a dataclass and
 `as_tensorclass=False`, if it is a dataclass type with a
 field that cannot be read from the class (a field without a
 default, or a field of a tensorclass type) and
 `as_tensorclass=False`, or if both `auto_batch_size=True`
 and `batch_size` are given.

Examples

```
>>> import dataclasses
>>> import torch
>>> from tensordict import TensorDict
>>> @dataclasses.dataclass
... class Data:
... obs: torch.Tensor
... reward: torch.Tensor
>>> data = Data(obs=torch.zeros(3, 4), reward=torch.ones(3))
>>> td = TensorDict.from_dataclass(data, auto_batch_size=True)
>>> type(td).__name__, td.batch_size
('TensorDict', torch.Size([3]))
>>> tc = TensorDict.from_dataclass(data, as_tensorclass=True, batch_size=[3])
>>> type(tc).__name__, tc.batch_size
('Data_tc', torch.Size([3]))
```

*classmethod*from_dict(*input_dict*, ***, *auto_batch_size: bool | None = None*, *batch_size=None*, *device=None*, *batch_dims=None*)

Builds an instance of the tensorclass from a dictionary of its fields.

The fields that `input_dict` misses are set to `None`, also those that
declare a default or a default factory. Starting with TensorDict 0.17, the
default will change to filling these fields with their defaults, as the
constructor does. Until then, a call that misses such a field warns with a
`FutureWarning`.

Parameters:

**input_dict** (*dict*) - the values of the fields. Nested dictionaries become
tensordicts, or tensorclasses for the fields annotated with one.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size to
the largest leading shape that all the entries share. Defaults to
`None`, which behaves like `False`, unless `batch_dims` is
passed: then it behaves like `True`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Defaults to `None`, which gives an empty batch size unless
`auto_batch_size` is on.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**str**or**int**,**optional*) - the device of the
result. Leaves on another device are moved to it. Defaults to
`None`, which leaves each leaf where it is.
- **batch_dims** (*int**,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Exclusive with `batch_size`.
Defaults to `None`.

Returns:

A new instance of the class.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... x: torch.Tensor
... y: torch.Tensor
>>> foo = Foo.from_dict({"x": torch.zeros(3, 4), "y": torch.zeros(3)}, auto_batch_size=True)
>>> foo.batch_size
torch.Size([3])
```

from_h5(***, *mode: str = 'r'*, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *non_tensor_as_stack: bool = False*)

Opens an HDF5 file as a [`PersistentTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict).

The data stays in the file: entries are read when they are accessed.
Requires `h5py`. See [Save and load tensordicts](https://docs.pytorch.org/tensordict/stable/saving.html#saving) and
[`PersistentTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict).

Parameters:

**filename** (*str**or**path*) - the path to the HDF5 file.

Keyword Arguments:

- **mode** (*str**,**optional*) - how the file is opened, as in
`h5py.File`: `"r"` (read only), `"r+"` (read and
write), and so on. Defaults to `"r"`.
- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`, which
gives each nested tensordict the largest batch size that its
entries share, and every nested tensordict `batch_size` when
it is passed. A file written by `to_h5()` stores the
batch sizes and the dim names. Starting with TensorDict 0.17,
the default will change to restoring the stored ones, as
`from_zarr` does: with `batch_size`, the nested
tensordicts get their stored batch sizes, and with
`auto_batch_size=True`, the stored names are kept and the
batch sizes are inferred from the stored ones, which an empty
tensordict or a non-tensor entry keeps. `from_h5` warns when
this changes its result.
- **non_tensor_as_stack** (*bool**,**optional*) - if `True`, a non-tensor
dataset that tensordict did not write, such as a string column
with one string per row, reads as a
[`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack) of its rows when its
leading dimensions equal the batch size. Otherwise it reads as
one [`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData) that holds the whole
dataset. Defaults to `False`.

Returns:

A [`PersistentTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) backed by the file.

Raises:

**TypeError** - if both `auto_batch_size=True` and `batch_size`
 are given.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict(key1=torch.zeros(3), key2=torch.ones(3), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... path = os.path.join(tmpdir, "file.h5")
... _ = data.to_h5(path)
... td = TensorDict.from_h5(path)
... print(td)
PersistentTensorDict(batch_size=[3], fields={key1: f32[], key2: f32[]})
```

from_json(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *separator: str | None = None*, *dtype: [torch.dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*, *lines: bool = False*, ***kwargs*) → Self

Reads a JSON file of records into a tensordict, with one entry per field.

The file holds an array of records (`[{"a": 1, "b": 2.0}, ...]`),
or one record per line with `lines=True` (JSON Lines). Each field
becomes an entry with one value per record. The file is read with
`pandas.read_json()` when pandas is installed, and with the
`json` module otherwise. Through pandas, a string field is one
value for the whole batch (see the note in `from_pandas()`);
through the `json` module, it is a
[`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack) with one value per record. To
convert a single JSON object, load
it with `json.load()` and pass it to `from_dict()`.
[`to_json()`](torchrl.data.vla.RobotDatasetMetadata.html#torchrl.data.vla.RobotDatasetMetadata.to_json) writes such a file.

Parameters:

**path** (*str**or**Path*) - the path of the JSON file.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**str**or**int**,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`,
which gives `[num_rows]`.
- **separator** (*str**,**optional*) - if given, splits the column names on it
to build nested entries: with `separator="."`, the column
`"obs.x"` becomes the entry `("obs", "x")`. Defaults to
`None`, which keeps the column names as keys.
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - if given, casts the numeric and
boolean columns to it. Defaults to `None`, which keeps the
dtype of each column.
- **lines** (*bool**,**optional*) - if `True`, reads the file as JSON Lines,
one record per line. Defaults to `False`.
- ****kwargs** - passed to `pandas.read_json()`. Ignored without
pandas.

Returns:

A tensordict of the class this method is called on
([`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) for `TensorDictBase`).

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

from_list(***, *auto_batch_size: bool | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_dims: int | None = None*, *names: list[str] | None = None*, *lazy: bool | None = None*) → Self

Builds a tensordict by stacking a list of dicts or tensor collections.

Each dict is converted with `from_dict()` and the results are
stacked along a new leading dimension. A tensordict or tensorclass in
the list is stacked as it is: the keyword arguments do not apply to
it. When the result does not get the `batch_size` or `device`
asked for, a `FutureWarning` is emitted: starting with TensorDict
0.17, the default will change to applying them to the result, which
raises if the batch size does not fit, as for a list of dicts.

Parameters:

**input** (*list*) - the dicts or tensor collections to stack.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size
of each element to the largest leading shape that all its
entries share. Defaults to `None`, which behaves like
`False` unless `batch_dims` is given.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result, with the length of the list first. Each element
gets `batch_size[1:]`. Defaults to `None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**str**or**int**,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_dims** (*int**,**optional*) - the maximum number of batch dimensions
of the result, the list dimension included. Turns on
`auto_batch_size` when that is `None`. Exclusive with
`batch_size`. Defaults to `None`.
- **names** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**str**,**optional*) - the names of the batch dimensions
of the result, one per dimension, with the name of the list
dimension first. Defaults to `None`.
- **lazy** (*bool**,**optional*) - if `True`, the result is a lazy stack;
if `False`, a dense stack. Defaults to `None`, which builds
a dense stack when the elements have the same shape and a lazy
stack otherwise. See [Stacking and variable-size data](https://docs.pytorch.org/tensordict/stable/concepts/stacking.html#stacking).

Raises:

**ValueError** - if `batch_size[0]` is not the length of the list.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict.from_list(
... [{"a": torch.zeros(3)}, {"a": torch.zeros(3)}],
... batch_size=[2, 3],
... names=["x", "y"],
... )
>>> td.batch_size
torch.Size([2, 3])
>>> td.names
['x', 'y']
```

from_module(*as_module: bool = False*, *lock: bool = False*, *use_state_dict: bool = False*, *filter_empty: bool = True*)

Returns a tensordict that holds the parameters and buffers of a module.

The entries are the parameters and buffers of the module themselves,
not copies. Submodules become nested tensordicts.

Note

This method does not lock the result by default (`lock=False`).
The function [`tensordict.from_module()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.from_module.html#tensordict.from_module) locks it by default
(`lock=True`).

Parameters:

- **module** ([*torch.nn.Module*](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)) - the module to get the parameters from.
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

A [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict), or a
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

from_modules(***, *as_module: bool = False*, *lock: bool = True*, *use_state_dict: bool = False*, *lazy_stack: bool = False*, *expand_identical: bool = False*)

Stacks the parameters and buffers of several modules in one tensordict.

The result has one batch dimension, with one element per module, and
the structure that `from_module()` gives. Use it to run an
ensemble of modules that have the same structure with
[`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap) and `to_module()` (see the example).

The default, dense result holds new parameters and buffers: the
stacks of those of the modules, with the same `requires_grad`. Pass
these new parameters to the optimizer: gradients reach them, not the
parameters of the modules. With `lazy_stack=True`, the result is a
lazy stack of the parameters of the modules, which receive the
gradients.

Parameters:

***modules** (*nn.Module*) - the modules to get the parameters from. With
a dense stack, they must have the same parameters and buffers,
with the same shapes.

Keyword Arguments:

- **as_module** (*bool**,**optional*) - if `True`, returns a
`TensorDictParams`, which can be stored
as a submodule of a [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Exclusive with
`lazy_stack`. Defaults to `False`.
- **lock** (*bool**,**optional*) - if `True`, the result is locked (see
[Locking and temporary changes](https://docs.pytorch.org/tensordict/stable/concepts/keys.html#keys-locking)). Defaults to `True`.
- **use_state_dict** (*bool**,**optional*) - if `True`, reads the
`state_dict()` of each module, so that its state-dict hooks
run. Defaults to `False`.
- **lazy_stack** (*bool**,**optional*) -

if `True`, returns a
[`LazyStackedTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) of the parameters of
the modules; if `False`, stacks them into new parameters.
Exclusive with `as_module`, and not compatible with lazy
modules. Defaults to `False`.

Warning

A lazy stack stacks the parameters each time an entry is
read with `get()`, which costs memory (`N` times
the size of the parameters, more if a graph is built) and
time, and the optimizer holds `N` times more parameters.
Use `lazy_stack=True` only when you need to train the
parameters of the modules themselves.
- **expand_identical** (*bool**,**optional*) - if `True`, a parameter that is
the same object in every module is expanded instead of
stacked, so that the result shares its memory. Ignored when
`lazy_stack=True`. Defaults to `False`.

Raises:

**RuntimeError** - if `lazy_stack=True` and `as_module=True`, or if
 `lazy_stack=True` and a module has uninitialized (lazy)
 parameters.

Examples

```
>>> import torch
>>> from torch import nn
>>> from tensordict import TensorDict
>>> _ = torch.manual_seed(0)
>>> modules = [nn.Linear(3, 4) for _ in range(2)]
>>> params = TensorDict.from_modules(*modules)
>>> params.batch_size, params["weight"].shape
(torch.Size([2]), torch.Size([2, 4, 3]))
```

Run the two modules on one input with [`torch.vmap()`](https://docs.pytorch.org/docs/stable/generated/torch.vmap.html#torch.vmap), through a
module on the `"meta"` device that holds no data. The gradients
reach the stacked parameters, not those of the modules:

```
>>> empty_module = nn.Linear(3, 4, device="meta")
>>> def exec_module(params, x):
... with params.to_module(empty_module):
... return empty_module(x)
>>> x = torch.randn(3)
>>> y = torch.vmap(exec_module, (0, None))(params, x)
>>> y.shape
torch.Size([2, 4])
>>> y.sum().backward()
>>> params["weight"].grad is not None, modules[0].weight.grad is None
(True, True)
```

With `lazy_stack=True`, the gradients reach the parameters of
the modules:

```
>>> params = TensorDict.from_modules(*modules, lazy_stack=True)
>>> type(params).__name__
'LazyStackedTensorDict'
>>> y = torch.vmap(exec_module, (0, None))(params, x)
>>> y.sum().backward()
>>> modules[0].weight.grad is not None
True
```

from_namedtuple(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*)

Converts a named tuple to a tensordict whose keys are its field names.

Each field value is converted with `from_any()`, so a nested
named tuple or dict becomes a nested tensordict and a number becomes a
tensor. The `torch.return_types` results of `torch.max`,
`torch.sort` and the like are accepted too. `to_namedtuple()`
converts back.

Parameters:

**named_tuple** (*NamedTuple*) - the named tuple to convert.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**str**or**int**,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
It applies to the nested tensordicts too. Exclusive with
`auto_batch_size`. Defaults to `None`, an empty batch size.

Returns:

A [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict).

Raises:

**TypeError** - if both `auto_batch_size=True` and `batch_size`
 are given.

Examples

```
>>> from tensordict import TensorDict
>>> import torch
>>> data = TensorDict({
... "a_tensor": torch.zeros((3)),
... "nested": {"a_tensor": torch.zeros((3)), "a_string": "zero!"}}, [3])
>>> nt = data.to_namedtuple()
>>> nt
GenericDict(a_tensor=tensor([0., 0., 0.]), nested=GenericDict(a_tensor=tensor([0., 0., 0.]), a_string='zero!'))
>>> td = TensorDict.from_namedtuple(nt, auto_batch_size=True)
>>> td.batch_size, td["nested", "a_string"]
(torch.Size([3]), 'zero!')
>>> sorted(TensorDict.from_namedtuple(torch.tensor([2, 0, 1]).max(0)).keys())
['indices', 'values']
```

from_pandas(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *separator: str | None = None*, *dtype: [torch.dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*) → Self

Converts a pandas DataFrame to a tensordict, with one entry per column.

The batch size is `[num_rows]`. Numeric and boolean columns become
tensors, a string or object column becomes one
[`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData) that holds the whole column, a
categorical column becomes the tensor of its codes, and a datetime
column becomes an `int64` tensor of nanoseconds.
`to_pandas()` converts back.

Note

A string or object column is one value for the whole batch, so
`td[i]["name"]` gives the values of every row, not the value of
row `i`. To get one value per row, set the column again as a
list: `td["name"] = df["name"].tolist()` stores a
[`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack), whose `td[i]["name"]` is
the value of row `i`. It builds one object per row, about 20
microseconds each, so it is much slower than reading the column
for large frames.

Parameters:

**dataframe** (*pd.DataFrame*) - the DataFrame to convert.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**str**or**int**,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`,
which gives `[num_rows]`.
- **separator** (*str**,**optional*) - if given, splits the column names on it
to build nested entries: with `separator="."`, the column
`"obs.x"` becomes the entry `("obs", "x")`. Defaults to
`None`, which keeps the column names as keys.
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - if given, casts the numeric and
boolean columns to it. Defaults to `None`, which keeps the
dtype of each column.

Returns:

A tensordict of the class this method is called on
([`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) for `TensorDictBase`).

Examples

```
>>> from tensordict import TensorDict
>>> import pandas as pd
>>> df = pd.DataFrame({"a": [1, 2, 3], "b": [4.0, 5.0, 6.0]})
>>> td = TensorDict.from_pandas(df)
>>> print(td)
TensorDict(batch_size=[3], fields={a: i64[], b: f64[]})
```

from_parquet(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *separator: str | None = None*, *dtype: [torch.dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | None = None*, *columns: list[str] | None = None*, ***kwargs*) → Self

Reads a Parquet file into a tensordict, with one entry per column.

Requires pyarrow or pandas: the file is read with
`pyarrow.parquet.read_table` when pyarrow is installed, and with
`pandas.read_parquet()` otherwise. The columns are converted as in
`from_pandas()`. `to_parquet()` writes such a file. Through
pyarrow, a string column is a [`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack)
with one value per row; through pandas, it is one value for the whole
batch (see the note in `from_pandas()`).

Parameters:

**path** (*str**or**Path*) - the path of the Parquet file.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**str**or**int**,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`,
which gives `[num_rows]`.
- **separator** (*str**,**optional*) - if given, splits the column names on it
to build nested entries: with `separator="."`, the column
`"obs.x"` becomes the entry `("obs", "x")`. Defaults to
`None`, which keeps the column names as keys.
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - if given, casts the numeric and
boolean columns to it. Defaults to `None`, which keeps the
dtype of each column.
- **columns** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**str**,**optional*) - the columns to read. Defaults to
`None`, which reads all the columns.
- ****kwargs** - passed to the Parquet reader.

Returns:

A tensordict of the class this method is called on
([`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) for `TensorDictBase`).

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

from_pytree(***, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *auto_batch_size: bool = False*, *batch_dims: int | None = None*)

Converts a pytree to a tensordict that `to_pytree()` converts back.

Each list, tuple, named tuple and dict of the pytree becomes a nested
tensordict, and its type is stored in a `"_pytree_type"` entry so
that `to_pytree()` can rebuild the pytree. The elements of a list
or tuple get the keys `"0"`, `"1"`, ... The subclasses of these
types, such as `collections.OrderedDict`,
`collections.defaultdict` and the `torch.return_types`
results of `torch.max` and the like, are accepted too. A
`defaultdict` is rebuilt without its `default_factory`.

A tensor or a tensor collection is returned as it is. Python ints and
floats and NumPy arrays become tensors, so `to_pytree()` returns
them as tensors. A dict key that is not a `NestedKey` (such as an
int or an object) is stored in a [`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData)
entry next to its value. Other types raise a `NotImplementedError`.

Parameters:

**pytree** - the pytree to convert: a list, tuple, named tuple or dict,
nested to any depth, whose leaves are tensors, tensor
collections, Python numbers or NumPy arrays.

Keyword Arguments:

- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Defaults to `None`, which gives an empty batch size unless
`auto_batch_size=True`.
- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Defaults to `False`.
- **batch_dims** (*int**,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.

Note

A module `state_dict()` round-trips through `from_pytree`, and its
`memmap()` output loads with `load_memmap(path, allow_pickle=False)`.
The state dicts of an optimizer and of a learning-rate scheduler hold
`None`, `str` values and Python numbers, which `from_pytree` rejects
(`None` and `str`) or returns as tensors (numbers): save them with
`torch.save(sd, path)` and load them with `torch.load(path, weights_only=True)`,
which round-trips them exactly without unpickling arbitrary objects.

Note

The list fields of a named tuple, and some of its dict and tuple
fields, are converted with `from_any()` and do not
round-trip through `to_pytree()`:

- a list of numbers comes back as a tensor;
- an empty tuple, an empty dict or named tuple, and a dict that
holds a str or a list come back as a `GenericDict` named
tuple;
- a dict that holds a tuple, a tuple subclass (such as a
`torch.return_types` result), and a tuple that holds anything
other than tensors, Python numbers, dicts and named tuples make
`to_pytree()` raise a `ValueError`.

The other dict, named tuple and tuple fields come back with their
own type.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> pytree = {"obs": torch.zeros(3, 4), "pair": (torch.zeros(3), torch.ones(3))}
>>> td = TensorDict.from_pytree(pytree, auto_batch_size=True)
>>> td.batch_size
torch.Size([3])
>>> sorted(td.keys()), sorted(td["pair"].keys())
(['_pytree_type', 'obs', 'pair'], ['0', '1', '_pytree_type'])
>>> type(td.to_pytree()["pair"]).__name__
'tuple'
```

A pytree with lists, dicts, a named tuple and a dict key that is
not a string round-trips:

```
>>> class Key:
... pass
>>> key = Key()
>>> pytree = (
... [torch.arange(3), torch.zeros(2)],
... {"tensor": torch.ones(2), key: torch.arange(2), "list": [1, 2, 3]},
... {"named_tuple": TensorDict(two=torch.ones(1)).to_namedtuple()},
... )
>>> pytree_recon = TensorDict.from_pytree(pytree).to_pytree()
>>> def check(v1, v2):
... assert (v1 == v2).all()
>>> _ = torch.utils._pytree.tree_map(check, pytree, pytree_recon)
>>> key in pytree_recon[1]
True
```

from_schema(***, *batch_size: [Sequence](torchrl.data.Sequence.html#torchrl.data.Sequence)[int] | [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *storage: str | None = None*, *device=None*, ***kwargs*) → TensorDictBase

Allocates a zero-filled tensordict from a schema of shapes and dtypes.

Each entry of `schema` maps a key to an `(element_shape, dtype)`
pair, and the entry gets the shape `[*batch_size, *element_shape]`.
`storage` selects where the zeros are stored, so that a buffer on
disk or on a server can be created without building it in memory
first.

Parameters:

**schema** (*dict*) - maps each key (a `str`) to a pair
`(element_shape, dtype)`, where `element_shape` (a list of
int or a `torch.Size`) is the shape of the entry after the
batch dimensions.

Keyword Arguments:

- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result, prepended to every element shape. Defaults to
`None`, an empty batch size.
- **storage** (*str**,**optional*) -

where the entries are stored:

- `None`: a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) of regular
tensors.
- `"memmap"`: memory-mapped tensors, in the directory given
by `prefix=<dir>` (see `memmap_like()`).
- `"h5"`: a [`PersistentTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) in the
HDF5 file given by `filename=<path>`.
- `"zarr"`: a [`PersistentTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) in
the zarr store given by `filename=<path or store>`
(requires `zarr>=3.1.1`).
- `"shared"`: tensors in shared memory (see
[`share_memory_()`](torchrl.objectives.llm.MCAdvantage.html#torchrl.objectives.llm.MCAdvantage.share_memory_)).
- `"redis"` or `"dragonfly"`: a
[`TensorDictStore`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore) on a server,
through [`TensorDictStore.from_schema`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore.from_schema).

Defaults to `None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**str**or**int**,**optional*) - the device of the
result. The `"memmap"` storage ignores it. Defaults to
`None`.
- ****kwargs** - passed to the backend: `prefix` and the other
arguments of `memmap_like()` for `"memmap"`,
`filename` for `"h5"` and `"zarr"`, `host`, `port`
and the like for `"redis"` and `"dragonfly"`.

Returns:

A new tensordict of the type that `storage` selects, filled with
zeros.

Raises:

**ValueError** - if `storage` is not one of the values above.

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
... print(td_mm.is_memmap())
True
```

from_struct_array(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*) → Self

Converts a NumPy structured array to a tensordict, with one entry per field.

The batch size is the shape of the array, and each entry holds a
field. The conversion is zero-copy: the tensors share memory with the
array, so an in-place change to the array changes the tensordict, and
the other way around. Two cases copy the data: a `device` other
than the CPU, and a field whose item size does not divide the size of
a record, such as the `"i8"` field of a packed array of `"i8"` and
`"f4"` fields (12 bytes per record), which torch cannot view. String
fields become non-tensor data. `to_struct_array()` converts
back.

Parameters:

**struct_array** (*np.ndarray*) - the structured array to convert.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share,
which can add the dimensions of array fields. Exclusive with
`batch_size`. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**str**or**int**,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`,
which uses the shape of the array.

Returns:

A tensordict of the class this method is called on
([`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) for `TensorDictBase`).

Raises:

**TypeError** - if both `auto_batch_size=True` and `batch_size`
 are given.

Examples

```
>>> import numpy as np
>>> from tensordict import TensorDict
>>> x = np.array(
... [("Rex", 9, 81.0), ("Fido", 3, 27.0)],
... dtype=[("name", "U10"), ("age", "i4"), ("weight", "f4")],
... )
>>> td = TensorDict.from_struct_array(x)
>>> td.batch_size, td["age"]
(torch.Size([2]), tensor([9, 3], dtype=torch.int32))
>>> x["age"] += 1 # the tensordict shares memory with the array
>>> td["age"]
tensor([10, 4], dtype=torch.int32)
>>> bool((td.to_struct_array() == x).all())
True
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
fields that neither argument holds are set to `None`. Defaults
to `None`.
- **safe** (*bool**,**optional*) - deprecated in 0.15, will be removed in TensorDict 0.17.
If `False`, skips the check that `tensordict` is a
[`TensorDictBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase). By default, the check runs, and
another type raises a `RuntimeError`.

Returns:

A new instance of the class, whose fields are the entries of
`tensordict`.

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

from_tuple(***, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*)

Converts a tuple to a tensordict with the keys `"0"`, `"1"`, ...

Each element is converted with `from_any()`, so a nested tuple
or dict becomes a nested tensordict and a number becomes a tensor.

Parameters:

**obj** (*tuple*) - the tuple to convert.

Keyword Arguments:

- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**str**or**int**,**optional*) - the device of the
result. Leaves on another device are moved to it. `None`
leaves each leaf where it is, and the result has no device.
Defaults to `None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
It applies to the nested tensordicts too. Exclusive with
`auto_batch_size`. Defaults to `None`, an empty batch size.

Returns:

A [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict).

Raises:

**TypeError** - if both `auto_batch_size=True` and `batch_size`
 are given.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict.from_tuple((torch.zeros(3, 4), 1), auto_batch_size=True)
>>> td["0"].shape, td["1"], td.batch_size
(torch.Size([3, 4]), tensor(1), torch.Size([]))
>>> td = TensorDict.from_tuple((torch.zeros(3, 4), torch.ones(3)), auto_batch_size=True)
>>> td.batch_size
torch.Size([3])
```

from_zarr(***, *mode: str = 'r'*, *auto_batch_size: bool = False*, *batch_dims: int | None = None*, *batch_size: [torch.Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size) | None = None*, *non_tensor_as_stack: bool = False*, *allow_pickle: bool | None = None*)

Opens a zarr store as a [`PersistentTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict).

The data stays in the store: entries are read when they are accessed.
Requires `zarr>=3.1.1`. See [Save and load tensordicts](https://docs.pytorch.org/tensordict/stable/saving.html#saving) and
[`PersistentTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict).

Parameters:

**filename** (*str**,**path**or**zarr store*) - the path to the zarr store (a
directory), or a `zarr.abc.store.Store` instance (e.g. a
`zarr.storage.ZipStore`).

Keyword Arguments:

- **mode** (*str**,**optional*) - how the store is opened: `"r"` (read
only), `"r+"` (read and write), and so on, as in
`zarr.open_group()`. Defaults to `"r"`.
- **auto_batch_size** (*bool**,**optional*) - if `True`, sets the batch size
to the largest leading shape that all the entries share.
Exclusive with `batch_size`. Defaults to `False`.
- **batch_dims** (*int**,**optional*) - with `auto_batch_size`, the maximum
number of batch dimensions. Ignored without
`auto_batch_size`, with a `FutureWarning` when it would
change the result. Starting with TensorDict 0.17, the default
will change to turning on `auto_batch_size` when
`batch_dims` is given, as `from_dict` does. Defaults to
`None`.
- **batch_size** ([*torch.Size*](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*or*[*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**int**,**optional*) - the batch size of
the result: the leading dimensions that all its entries share.
Exclusive with `auto_batch_size`. Defaults to `None`, which
reads the batch size and the names of the batch dimensions that `to_zarr()` stored, or, for
a store written by another tool, gives each nested tensordict
the largest batch size that its entries share.
- **non_tensor_as_stack** (*bool**,**optional*) - if `True`, a non-tensor
array that tensordict did not write, such as a string column
with one string per row, reads as a
[`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack) of its rows when its
leading dimensions equal the batch size. Otherwise it reads as
one [`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData) that holds the whole
array. Defaults to `False`.
- **allow_pickle** (*bool**,**optional*) - whether to decode pickled
non-tensor values. `False` rejects them before decoding.
`True` permits them: use it only for data you trust.
`None` permits them with a `FutureWarning` when read.
Starting with TensorDict 0.17, the default will change to
`allow_pickle=False`. Tensor and JSON entries are unaffected.
Defaults to `None`.

Returns:

A [`PersistentTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict) backed by the store.

Raises:

**TypeError** - if both `auto_batch_size=True` and `batch_size`
 are given.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> data = TensorDict(key1=torch.zeros(3), key2=torch.ones(3), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as tmpdir:
... path = os.path.join(tmpdir, "store.zarr")
... _ = data.to_zarr(path)
... td = TensorDict.from_zarr(path)
... print(td)
PersistentTensorDict(batch_size=[3], fields={key1: f32[], key2: f32[]})
```

fromkeys(*value: Any = 0*)

Creates a tensordict with the same value for every key, like `dict.fromkeys()`.

The result is a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) with an empty batch
size. A nested key creates nested tensordicts. A Python number becomes
a tensor for each key, and a tensor is stored as it is: every key then
holds the same tensor object.

Parameters:

- **keys** (*list**of**NestedKey*) - the keys of the result.
- **value** (*Any**,**optional*) - the value of every key: a tensor, a number,
or any value that the [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)
constructor takes. Defaults to `0`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict.fromkeys(["a", ("b", "c")], torch.zeros(3))
>>> td["b", "c"]
tensor([0., 0., 0.])
>>> td["a"] is td["b", "c"]
True
```

get(*key: NestedKey*, **args*, ***kwargs*)

Returns the value of a field, or `default` if the key is missing.

`tc.get("x")` is the same as `tc.x`: a non-tensor field returns the
Python object, not a [`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData). The keyword
arguments of [`get()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) (`as_list`,
`as_padded_tensor`, ...) are passed to the tensordict that holds the
fields.

A nested key that runs through a tensor field, such as
`tc.get(("x", "y"))` for a tensor `x`, returns `None` and warns.
Starting with TensorDict 0.17, the default will change to raising the
`ValueError` that [`get()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.get.html#tensordict.TensorDictBase.get) raises. A
default passed explicitly, as in `tc.get(("x", "y"), None)`, is
returned without a warning.

Parameters:

- **key** (*NestedKey*) - the key to read. A tuple reads an entry of a nested
field: `tc.get(("a", "b"))` is `tc.a.get("b")`.
- **default** (*optional*) - the value returned if the key is missing.
Defaults to `None`.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... x: torch.Tensor
... name: str
>>> foo = Foo(x=torch.zeros(3), name="a", batch_size=[3])
>>> foo.get("name")
'a'
>>> foo.get("z", "missing")
'missing'
```

lazy_stack(*dim: int = 0*, ***, *out=None*, ***kwargs*)

Stacks tensordicts into a lazy stack.

A lazy stack keeps the tensordicts as they are and stacks their values
only when you read them. The tensordicts can have different keys and
different feature shapes. See [Stacking and variable-size data](https://docs.pytorch.org/tensordict/stable/concepts/stacking.html#stacking).

Parameters:

- **input** (*sequence**of**TensorDictBase*) - the tensordicts to stack.
- **dim** (*int**,**optional*) - the new batch dimension. Defaults to `0`.

Keyword Arguments:

- **out** (*TensorDictBase**,**optional*) - a lazy stack to write the result
into. Defaults to `None`.
- ****kwargs** - passed to
[`LazyStackedTensorDict.lazy_stack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict.lazy_stack), for example
`stack_dim_name` or `strict_shape`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> a = TensorDict(x=torch.zeros(3), batch_size=[3])
>>> b = TensorDict(y=torch.ones(3), batch_size=[3])
>>> stack = TensorDict.lazy_stack([a, b])
>>> type(stack).__name__, stack.batch_size
('LazyStackedTensorDict', torch.Size([2, 3]))
```

load(**args*, ***kwargs*) → Self

Loads a tensordict that `save()` or `memmap()` wrote to disk.

`load` calls `load_memmap()`, which documents the other
arguments, such as `device`, `mode`, `subpath` and
`allow_pickle`. The leaves of the result read the files lazily. See
[Save and load tensordicts](https://docs.pytorch.org/tensordict/stable/saving.html#saving).

Parameters:

- **prefix** (*str**or**Path*) - the directory, or the archive file, to read.
- ***args** - passed to `load_memmap()`.
- ****kwargs** - passed to `load_memmap()`.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.arange(3), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.save(path)
... loaded = TensorDict.load(path)
... print(loaded["a"], loaded.batch_size)
MemoryMappedTensor([0, 1, 2]) torch.Size([3])
```

load_(*prefix: str | Path*, **args*, ***kwargs*)

Loads a saved tensordict into the tensordict.

Calls `load_memmap_()` with the same arguments.

Deprecated since version 0.15: Use `load_memmap_()` instead. It will be removed in
TensorDict 0.17.

Parameters:

- **prefix** (*str**or**Path*) - the directory, or the archive file, to read.
- ***args** - passed to `load_memmap_()`.
- ****kwargs** - passed to `load_memmap_()`.

load_memmap(*device: [torch.device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, *non_blocking: bool = False*, ***, *out: TensorDictBase | None = None*, *robust_key: bool | None = True*, *subpath: NestedKey | None = None*, *mode: str | None = None*, *num_threads: int = 0*, *allow_pickle: bool | None = None*) → Self

Loads a tensordict that `memmap()` or `save()` wrote, from a directory or an archive.

The leaves of the result read the files lazily: from a directory, each
leaf is a [`MemoryMappedTensor`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor) of its file, and from
an archive, each leaf is a view into one memory map of the archive.
Only the pages of the leaves that you use are read from disk. The
result has the class that was saved (a tensorclass is loaded as that
tensorclass), and it is not locked. `load()` is this method. See
[Save and load tensordicts](https://docs.pytorch.org/tensordict/stable/saving.html#saving).

Parameters:

- **prefix** (*str**or**Path*) - the directory to read, or an archive file
written by `save(..., archive=True)`, by a `".tdz"` prefix,
or by [`pack_memmap()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.pack_memmap.html#tensordict.pack_memmap).
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**str**or**int**,**optional*) - the device of the
result. The leaves are copied to it, which reads the files,
unless it is the CPU.
With `"meta"`, no file is read: the leaves are meta tensors,
which show the structure and size of the data. Defaults to
`None`, which keeps the leaves on the CPU, memory-mapped.
- **non_blocking** (*bool**,**optional*) - if `True`, does not synchronize
the device after the copies to `device`. Defaults to
`False`.

Keyword Arguments:

- **out** (*TensorDictBase**,**optional*) - a tensordict to load the data into.
Its nested tensordicts are reused, but its leaves are replaced
by the loaded tensors, so their storage is not reused. Its keys
that are not in the saved data are removed. If `out` has a
device, the data is loaded on it, and another `device` raises
a `ValueError`. To write the data into the storage of
`out`, use
`out.update_(TensorDict.load_memmap(prefix, device=device))`.
Defaults to `None`.
- **robust_key** (*bool**,**optional*) - if `True`, reads the
percent-encoded file names that `memmap()` writes by
default, and falls back to the legacy names of keys that are
safe file names. If `False`, reads the legacy layout only.
`None` is the same as `True`. Defaults to `True`.
- **subpath** (*NestedKey**or**str**,**optional*) - the nested tensordict to
load, as a nested key (`("module", "0")`) or as a
`"/"`-separated path (`"module/0"`). Only that subtree is
read, from a directory or an archive. A nested key reaches the
entries of a saved tensorclass as `obj.get(key)` does.
Defaults to `None`, which loads the whole tensordict.
- **mode** (*str**,**optional*) - how the files are mapped. With `"r"`, the
mapping is copy-on-write: in-place writes into the leaves stay
in memory, and the files do not change. With `"r+"`, in-place
writes reach the files, and a file that the process cannot
write raises a `PermissionError`. Defaults to `None`, which
maps an archive as with `"r"`, and each file of a directory
as with `"r+"` if the process can write it, and as with
`"r"` otherwise. See the note below.
- **num_threads** (*int**,**optional*) - the number of threads that decompress
the leaves of a compressed archive. It has no effect on an
uncompressed archive or a directory. Defaults to `0`, which
decompresses them one after the other.
- **allow_pickle** (*bool**,**optional*) - whether non-tensor data saved with
`pickle` may be loaded. Pickle can run arbitrary code: pass
`True` only for data you trust, and `False` for other data,
which raises a `RuntimeError` on pickled data. Defaults to
`None`, which loads pickled data with a `FutureWarning`.
Starting with TensorDict 0.16, the default will change to
`False`. Data saved without pickle loads with any value.

Returns:

The loaded tensordict, or `out` if it is given.

Raises:

**ValueError** - if `prefix` is a file that is not an archive, if
 `subpath` names no saved tensordict, or if `mode` is not
 `None`, `"r"` or `"r+"`.

Note

Load with `mode="r"` data that you only read: on some network
file systems (for example Lustre), page faults on a shared
writable mapping take write locks, so readers on different nodes
block each other. On Linux, a copy-on-write mapping counts against
the memory commit limit, so a file larger than RAM plus swap can
fail to map with `"r"`. With `"r"`, `memmap_()` without a
prefix does not write back to `prefix`.

Note

An archive opened with `mode="r+"` must have uncompressed,
aligned payloads (as written by tensordict without
`compression`) and no nested tensor. In-place writes do not
update the CRC-32 checksums of its entries: `load_memmap` does
not check them, but call
[`refresh_archive_checksums()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.refresh_archive_checksums.html#tensordict.refresh_archive_checksums) before you give the
archive to a tool that does, such as `unzip`.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(a=torch.zeros(3), nested=TensorDict(b=torch.ones(3)))
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.memmap(path)
... loaded = TensorDict.load_memmap(path)
... print((loaded == td).all(), type(loaded["a"]).__name__)
... # load one nested tensordict
... print(TensorDict.load_memmap(path, subpath="nested")["b"])
... # read the structure without reading the data
... print(TensorDict.load_memmap(path, device="meta")["a"].device)
True MemoryMappedTensor
MemoryMappedTensor([1., 1., 1.])
meta
```

With a `FakeTensorMode` active, the leaves are fake tensors and
no file is read:

```
>>> from torch._subclasses import FakeTensorMode
>>> with tempfile.TemporaryDirectory() as path:
... _ = td.memmap(path)
... with FakeTensorMode():
... print(type(TensorDict.load_memmap(path)["a"]).__name__)
FakeTensor
```

load_memmap_(*prefix: str | Path*, *robust_key: bool | None = True*, ***, *allow_pickle: bool | None = None*, *mode: str | None = None*)

Loads a memory-mapped tensorclass into this instance, and returns `self`.

Works as [`load_memmap_()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.load_memmap_.html#tensordict.TensorDictBase.load_memmap_), and also loads
the non-tensor fields.

Parameters:

- **prefix** (*str**or**Path*) - the directory, or the `".tdz"` archive file,
to read.
- **robust_key** (*bool**,**optional*) - the key encoding used when the data was
saved. See [`memmap()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.memmap.html#tensordict.TensorDictBase.memmap). Defaults to
`True`.

Keyword Arguments:

- **allow_pickle** (*bool**,**optional*) - whether pickled non-tensor fields may
be loaded. See [`load_memmap()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap).
Defaults to `None`, which loads pickled fields with a
`FutureWarning`.
- **mode** (*str**,**optional*) - the mode of the memory maps. See
[`load_memmap()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.load_memmap.html#tensordict.TensorDictBase.load_memmap). Defaults to
`None`.

load_state_dict(*state_dict: dict[str, Any]*, *strict=True*, *assign=False*, *from_flatten=None*)

Loads a state dict made by `state_dict()` into the tensorclass, and returns `self`.

Works as [`load_state_dict()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.load_state_dict.html#tensordict.TensorDictBase.load_state_dict), which
describes the arguments, and also loads the non-tensor fields. It also
reads the state dicts of older versions, which hold `"_tensordict"` and
`"_non_tensordict"` entries.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... x: torch.Tensor
... name: str
>>> sd = Foo(x=torch.zeros(3), name="a", batch_size=[3]).state_dict()
>>> foo = Foo(x=torch.ones(3), name="b", batch_size=[3])
>>> _ = foo.load_state_dict(sd)
>>> foo.x, foo.name
(tensor([0., 0., 0.]), 'a')
```

maybe_dense_stack(*dim: int = 0*, ***, *out=None*, ***kwargs*)

Stacks tensordicts densely if possible, or into a lazy stack otherwise.

The result is a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) if the tensordicts
have the same keys and shapes, and a
[`LazyStackedTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict) otherwise. See
[Stacking and variable-size data](https://docs.pytorch.org/tensordict/stable/concepts/stacking.html#stacking).

Parameters:

- **input** (*sequence**of**TensorDictBase*) - the tensordicts to stack.
- **dim** (*int**,**optional*) - the new batch dimension. Defaults to `0`.

Keyword Arguments:

- **out** (*TensorDictBase**,**optional*) - a tensordict to write the result
into. Defaults to `None`.
- ****kwargs** - passed to
[`LazyStackedTensorDict.maybe_dense_stack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.LazyStackedTensorDict.html#tensordict.LazyStackedTensorDict.maybe_dense_stack), for
example `strict`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> a = TensorDict(x=torch.zeros(3), batch_size=[3])
>>> b = TensorDict(x=torch.ones(3), batch_size=[3])
>>> c = TensorDict(y=torch.ones(3), batch_size=[3])
>>> type(TensorDict.maybe_dense_stack([a, b])).__name__
'TensorDict'
>>> type(TensorDict.maybe_dense_stack([a, c])).__name__
'LazyStackedTensorDict'
```

memmap(*prefix: str | Path | None = None*, *copy_existing: bool = False*, ***, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *existsok: bool = True*, *robust_key: bool | None = True*, *archive: bool | None = None*, *compression: str | int | None = None*, *atomic: bool = False*) → Self

Writes every leaf to a memory-mapped file and returns a new tensordict of them.

The tensordict is not changed: the result holds new
[`MemoryMappedTensor`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor) leaves. Each leaf is written to
`<prefix>/<key>.memmap`, each nested tensordict to a subdirectory,
and each level has a `meta.json` file. Read the files back with
`load_memmap()`. `save()` and `dumps()` call this
method, and `memmap_()` writes in place. See [Save and load tensordicts](https://docs.pytorch.org/tensordict/stable/saving.html#saving).

The result is locked (see [Locking and temporary changes](https://docs.pytorch.org/tensordict/stable/concepts/keys.html#keys-locking)): you can write into its
leaves in place, but you cannot add, remove or replace an entry. Once
it is unlocked, `is_memmap()` returns `False`.

Parameters:

- **prefix** (*str**or**Path**,**optional*) - the directory to write to. If it
ends with `".tdz"`, a single archive file is written instead
(see `archive`). Defaults to `None`, which writes into the
directory of the tensordict if it is already memory-mapped
there, and otherwise into anonymous shared memory, which has no
file name and lasts as long as the process.
- **copy_existing** (*bool**,**optional*) - if `False`, a leaf that is
already a memory-mapped tensor with a file at another path
raises a `RuntimeError`, so that data on disk is not
duplicated by accident. If `True`, the file is copied to the
new path. Defaults to `False`.

Keyword Arguments:

- **num_threads** (*int**,**optional*) - the number of threads that write the
files. `0` and `1` write them one after the other in the
calling thread. Threads help with a few large leaves and slow
down many small ones. Defaults to `0`.
- **return_early** (*bool**,**optional*) - if `True` and `num_threads > 1`,
returns a [`TensorDictFuture`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) at once,
while the threads write the files: `future.result()` waits
for them and returns the tensordict. Not supported for an
archive. Defaults to `False`.
- **share_non_tensor** (*bool**,**optional*) - if `True`, non-tensor data is
shared between the processes of the node, so that an in-place
write of a non-tensor entry in one process is seen by the
others. Many non-tensor leaves (for example a large stack of
non-tensor data) can use up the memory. Defaults to `False`.
- **existsok** (*bool**,**optional*) - if `False`, a file that already exists
at the path of a leaf (or at the archive path) raises a
`RuntimeError`. If `True`, it is overwritten. Defaults to
`True`.
- **robust_key** (*bool**,**optional*) - if `True`, keys are percent-encoded
in file and directory names (`"a/b"` is written to
`a%2Fb.memmap`), so that every file stays inside `prefix`.
If `False`, keys are used as file names unchanged, as in the
legacy layout: do not use it with keys you do not trust.
`None` is the same as `True`. Defaults to `True`.
- **archive** (*bool**,**optional*) - if `True`, `prefix` is a file, and
the tensordict is written to it as an archive: a zip file
whose entries are the files of the directory layout, with
uncompressed and aligned tensor payloads. Defaults to `None`,
which writes an archive if `prefix` ends with `".tdz"`.
- **compression** (*str**or**int**,**optional*) - the compression of the archive
entries: `"stored"`, `"deflate"`, `"bzip2"`, `"lzma"`,
or a `zipfile` constant. A compressed leaf is
decompressed in memory when it is loaded, instead of being
memory-mapped. Only valid for an archive. Defaults to `None`,
which stores the entries uncompressed.
- **atomic** (*bool**,**optional*) - if `True`, the files are written to a
new directory next to `prefix`, which is then renamed to
`prefix`, so that an error or a crash during the write
leaves `prefix` as it was. The new directory is removed if
the write fails (but not after a crash). An existing directory
at `prefix` is replaced as a whole once the new one is
complete, so files that the tensordict does not write are
removed: it must be empty or hold a saved tensordict (a
`meta.json` file). Readers see the old or the new directory,
except for the short time between two renames, during which
`prefix` is missing. The write needs about twice the disk
space of the tensordict while it runs, and a leaf that is
already memory-mapped to a file is copied, which needs
`copy_existing=True`. The files are not synced to the disk,
so this guards against a killed process, not against a power
failure. Not supported with `return_early=True`, nor for a
directory on Windows, where a directory that holds open files
cannot be renamed. An archive is always written this way.
Defaults to `False`.

Returns:

A new memory-mapped tensordict, or a
[`TensorDictFuture`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) of it if
`return_early=True` and `num_threads > 1`.

Raises:

- **RuntimeError** - if a leaf already has a file elsewhere and
 `copy_existing=False`, if a file exists and
 `existsok=False`, or with `atomic=True`, if `prefix` is
 a non-empty directory and `existsok=False`, or if it holds
 no saved tensordict.
- **NotADirectoryError** - if `atomic=True` and `prefix` is a file.
- **ValueError** - if `compression` is given without an archive, or if
 an archive or `atomic=True` is requested without `prefix`.
- **NotImplementedError** - if `return_early=True` for an archive or
 with `atomic=True`, or `atomic=True` for a directory on
 Windows.

Note

For an archive, the result is `load_memmap(prefix)`: its leaves
are views into one copy-on-write mapping of the file, so in-place
writes do not reach the file, and it is not locked. Archives are
written in one pass, so `num_threads` does not speed them up. An
existing archive is replaced by a new file: a symlink is followed,
but other hard links keep the old content.
[`pack_memmap()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.pack_memmap.html#tensordict.pack_memmap) and
[`unpack_memmap()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.unpack_memmap.html#tensordict.unpack_memmap) convert between archives and
directories.

Examples

```
>>> import os
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... obs=torch.zeros(3, 4),
... next=TensorDict(reward=torch.ones(3, 1), batch_size=[3]),
... batch_size=[3],
... )
>>> with tempfile.TemporaryDirectory() as path:
... mm = td.memmap(path)
... print(type(mm["obs"]).__name__, mm.is_locked)
... print(sorted(os.listdir(path)))
MemoryMappedTensor True
['meta.json', 'next', 'obs.memmap']
```

memmap_(*prefix: str | Path | None = None*, *copy_existing: bool = False*, ***, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *existsok: bool = True*, *robust_key: bool | None = True*, *atomic: bool = False*) → Self

Writes every leaf to a memory-mapped file in place, and returns `self`.

Writes the same files as `memmap()`, then replaces each leaf of
the tensordict with its [`MemoryMappedTensor`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.MemoryMappedTensor.html#tensordict.MemoryMappedTensor), so the
`data_ptr()` of every leaf changes. The device of the tensordict
becomes `"cpu"`. See [Save and load tensordicts](https://docs.pytorch.org/tensordict/stable/saving.html#saving).

The tensordict is then locked (see [Locking and temporary changes](https://docs.pytorch.org/tensordict/stable/concepts/keys.html#keys-locking)): you can write
into its leaves in place, but you cannot add, remove or replace an
entry. Once it is unlocked, `is_memmap()` returns `False`.

Parameters:

- **prefix** (*str**or**Path**,**optional*) - the directory to write to.
Defaults to `None`, which writes into the directory of the
tensordict if it is already memory-mapped there, and otherwise
into anonymous shared memory, which has no file name and lasts
as long as the process.
- **copy_existing** (*bool**,**optional*) - if `False`, a leaf that is
already a memory-mapped tensor with a file at another path
raises a `RuntimeError`, so that data on disk is not
duplicated by accident. If `True`, the file is copied to the
new path. Defaults to `False`.

Keyword Arguments:

- **num_threads** (*int**,**optional*) - the number of threads that write the
files. `0` and `1` write them one after the other in the
calling thread. Threads help with a few large leaves and slow
down many small ones. Defaults to `0`.
- **return_early** (*bool**,**optional*) - if `True` and `num_threads > 1`,
returns a [`TensorDictFuture`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) at once,
while the threads write the files: `future.result()` waits
for them and returns the tensordict. Defaults to `False`.
- **share_non_tensor** (*bool**,**optional*) - if `True`, non-tensor data is
shared between the processes of the node, so that an in-place
write of a non-tensor entry in one process is seen by the
others. Many non-tensor leaves (for example a large stack of
non-tensor data) can use up the memory. Defaults to `False`.
- **existsok** (*bool**,**optional*) - if `False`, a file that already exists
at the path of a leaf raises a `RuntimeError`. If `True`,
it is overwritten. Defaults to `True`.
- **robust_key** (*bool**,**optional*) - if `True`, keys are percent-encoded
in file and directory names (`"a/b"` is written to
`a%2Fb.memmap`), so that every file stays inside `prefix`.
If `False`, keys are used as file names unchanged, as in the
legacy layout: do not use it with keys you do not trust.
`None` is the same as `True`. Defaults to `True`.
- **atomic** (*bool**,**optional*) - if `True`, the files are written to a
new directory next to `prefix`, which is then renamed to
`prefix`, so that an error or a crash during the write
leaves `prefix` as it was. The new directory is removed if
the write fails (but not after a crash). An existing directory
at `prefix` is replaced as a whole once the new one is
complete, so files that the tensordict does not write are
removed: it must be empty or hold a saved tensordict (a
`meta.json` file). Readers see the old or the new directory,
except for the short time between two renames, during which
`prefix` is missing. The write needs about twice the disk
space of the tensordict while it runs, and a leaf that is
already memory-mapped to a file is copied, which needs
`copy_existing=True`. The files are not synced to the disk,
so this guards against a killed process, not against a power
failure. If the write fails, the leaves already converted map
deleted files, so a retry needs `copy_existing=True`. Not
supported with `return_early=True`, nor on Windows, where a
directory that holds open files cannot be renamed. Defaults
to `False`.

Returns:

`self`, or a [`TensorDictFuture`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) of it if
`return_early=True` and `num_threads > 1`.

Raises:

- **RuntimeError** - if a leaf already has a file elsewhere and
 `copy_existing=False`, if a file exists and
 `existsok=False`, if the tensordict is in shared memory
 (see [`share_memory_()`](torchrl.objectives.llm.MCAdvantage.html#torchrl.objectives.llm.MCAdvantage.share_memory_)), or if it is a
 [`PersistentTensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.PersistentTensorDict.html#tensordict.PersistentTensorDict), a
 [`TensorDictStore`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.store.TensorDictStore.html#tensordict.store.TensorDictStore), a
 [`LazyStackedTensorDictStore`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.store.LazyStackedTensorDictStore.html#tensordict.store.LazyStackedTensorDictStore) or one of
 their elements. Use `memmap()` to make a memory-mapped
 copy of those. With `atomic=True`, also if `prefix` is a
 non-empty directory and `existsok=False`, or if it holds no
 saved tensordict.
- **NotADirectoryError** - if `atomic=True` and `prefix` is a file.
- **ValueError** - if `atomic=True` and there is no directory to write
 to.
- **NotImplementedError** - if `return_early=True` with
 `atomic=True`, or `atomic=True` on Windows.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(obs=torch.zeros(3, 4), batch_size=[3])
>>> with tempfile.TemporaryDirectory() as path:
... out = td.memmap_(path)
... print(out is td, td.is_memmap(), td.is_locked)
... print(type(td["obs"]).__name__)
True True True
MemoryMappedTensor
```

memmap_like(*prefix: str | Path | None = None*, *copy_existing: bool = False*, ***, *existsok: bool = True*, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *robust_key: bool | None = True*, *archive: bool | None = None*) → Self

Creates a memory-mapped tensordict with the same keys, shapes and dtypes, without copying the values.

Use it to preallocate a large buffer on disk, then fill it with
`update_()` or by indexing: `memmap()` would write every
value first. New files are filled with zeros. A file that already
exists at the path of a leaf (with `existsok=True`) is mapped as it
is and keeps its content, which lets you reopen a buffer, or its
start with a smaller leaf. If the file is smaller than the leaf, it
is grown with zeros and a `FutureWarning` is emitted. Starting with
TensorDict 0.17, the default will change to raising a
`RuntimeError` in that case. A file written with another dtype or
shape is not detected otherwise, and its bytes are read as the new
dtype. An archive (`".tdz"`) is always
rebuilt with zeros. The tensordict is not changed. See
[Save and load tensordicts](https://docs.pytorch.org/tensordict/stable/saving.html#saving).

The result is locked (see [Locking and temporary changes](https://docs.pytorch.org/tensordict/stable/concepts/keys.html#keys-locking)): you can write into its
leaves in place, but you cannot add, remove or replace an entry. Once
it is unlocked, `is_memmap()` returns `False`.

Parameters:

- **prefix** (*str**or**Path**,**optional*) - the directory to write to. If it
ends with `".tdz"`, a single archive file is written instead
(see `archive`). Defaults to `None`, which allocates the
leaves in anonymous shared memory, which has no file name and
lasts as long as the process.
- **copy_existing** (*bool**,**optional*) - accepted for symmetry with
`memmap()`. It has no effect, since no value is copied.
Defaults to `False`.

Keyword Arguments:

- **existsok** (*bool**,**optional*) - if `False`, a file that already exists
at the path of a leaf (or at the archive path) raises a
`RuntimeError`. If `True`, it is reused. Defaults to
`True`.
- **num_threads** (*int**,**optional*) - the number of threads that create the
files. `0` and `1` create them one after the other in the
calling thread. Defaults to `0`.
- **return_early** (*bool**,**optional*) - if `True` and `num_threads > 1`,
returns a [`TensorDictFuture`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) at once,
while the threads create the files: `future.result()` waits
for them and returns the tensordict. Not supported for an
archive. Defaults to `False`.
- **share_non_tensor** (*bool**,**optional*) - if `True`, non-tensor data is
shared between the processes of the node, so that an in-place
write of a non-tensor entry in one process is seen by the
others. Many non-tensor leaves (for example a large stack of
non-tensor data) can use up the memory. Defaults to `False`.
- **robust_key** (*bool**,**optional*) - if `True`, keys are percent-encoded
in file and directory names (`"a/b"` is written to
`a%2Fb.memmap`), so that every file stays inside `prefix`.
If `False`, keys are used as file names unchanged, as in the
legacy layout: do not use it with keys you do not trust.
`None` is the same as `True`. Defaults to `True`.
- **archive** (*bool**,**optional*) - if `True`, `prefix` is a file, and a
zero-filled archive is created there and loaded back with
`load_memmap(prefix, mode="r+")`, so that in-place writes into
the result reach the archive. Nested tensors are not supported.
Defaults to `None`, which creates an archive if `prefix`
ends with `".tdz"`.

Returns:

A new memory-mapped tensordict, or a
[`TensorDictFuture`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) of it if
`return_early=True` and `num_threads > 1`.

Note

For an archive, the result is not locked, and in-place writes do
not update the CRC-32 checksums of the zip entries. Call
[`refresh_archive_checksums()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.refresh_archive_checksums.html#tensordict.refresh_archive_checksums) before you give the
archive to a tool that checks them, such as `unzip`.

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

Reloads the tensordict from its `saved_path`, and returns `self`.

Use it when another process has changed the files: added or removed
entries, or written non-tensor data. It calls `load_memmap_()`
on `saved_path`, then memory-maps the tensordict there again.

Keyword Arguments:

**allow_pickle** (*bool**,**optional*) - whether non-tensor data saved with
`pickle` may be loaded. See `load_memmap()`.
Defaults to `None`.

Raises:

**RuntimeError** - if the tensordict is not memory-mapped in a
 directory, for example a tensordict returned by
 `load_memmap()` (for which `is_memmap()` returns
 `False`) or by `memmap()` without a prefix.

Examples

```
>>> import tempfile
>>> import torch
>>> from tensordict import TensorDict
>>> with tempfile.TemporaryDirectory() as path:
... td = TensorDict(a=torch.zeros(3)).memmap(path)
... _ = TensorDict(a=torch.ones(3), b=torch.ones(3)).memmap(path)
... _ = td.memmap_refresh_()
... print(sorted(td.keys()), td["a"])
['a', 'b'] MemoryMappedTensor([1., 1., 1.])
```

pack(*dim: int = -1*, ***, *layout: PackLayout | None = None*) → tuple[[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), PackLayout]

Packs all the leaves of a tensordict into one tensor of features.

Each leaf is flattened after the batch dimensions, and the leaves are
concatenated in the order of `keys(True, True, sort=True)`, so the
columns depend on the set of keys only. Leaves can have different
ranks. The result has shape `(*batch_size, F)` for `dim=-1`, where
`F` is the sum of the numbers of features of the leaves, and it is
contiguous for any `dim`. [`PackLayout.unpack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.PackLayout.html#tensordict.PackLayout.unpack) inverts it.

Packing along the last dim copies the data once, with one
[`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat), unless the leaves are channels-last:
[`torch.cat()`](https://docs.pytorch.org/docs/stable/generated/torch.cat.html#torch.cat) keeps their strides, and making the result
contiguous takes a second copy. Along another dim the result can take
a second copy too, since moving the feature dim makes a 4-d or 5-d
result channels-last.

All the leaves must share one dtype and one device. Non-tensor data
cannot be packed: drop it with `filter_non_tensor_data()` first.
A lazy stack packs like its dense equivalent if all its members have
the same keys and feature shapes.

Parameters:

**dim** (*int**,**optional*) - the position of the packed dimension in the
result, in `[-(batch_dims + 1), batch_dims]` as for
[`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack). Defaults to `-1`.

Keyword Arguments:

**layout** (*PackLayout**,**optional*) - a layout to pack into. The columns
then follow its key order, and a missing, extra or reshaped
leaf raises. `dim` must match the layout's dim. Defaults to
`None`, which builds a new layout in sorted key order.

Returns:

A tuple of the packed tensor and its
[`PackLayout`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.PackLayout.html#tensordict.PackLayout).

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> td = TensorDict(
... p=torch.zeros(5),
... v=torch.ones(5, 3),
... nested={"stress": torch.ones(5, 3, 3)},
... batch_size=[5],
... )
>>> flat, layout = td.pack()
>>> flat.shape
torch.Size([5, 13])
>>> layout.keys
(('nested', 'stress'), 'p', 'v')
>>> out = layout.unpack(flat.index_select(0, torch.tensor([0, 2])))
>>> out.batch_size
torch.Size([2])
>>> flat2, _ = td.pack(layout=layout)
>>> assert (flat2 == flat).all()
```

rename_keys(*mapping*, ***, *strict: bool = True*)

Returns a tensorclass with several keys renamed at once.

Works as [`rename_keys()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.rename_keys.html#tensordict.TensorDictBase.rename_keys), which describes
`mapping` and `strict`, with two differences: each destination must be
a field of the class, or the call raises an `AttributeError`, and a
field that the renaming empties is kept.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... x: torch.Tensor
... y: torch.Tensor
>>> foo = Foo(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
>>> foo.rename_keys({"x": "y", "y": "x"}).x
tensor([1., 1., 1.])
```

save(*prefix: str | Path | None = None*, *copy_existing: bool = False*, ***, *num_threads: int = 0*, *return_early: bool = False*, *share_non_tensor: bool = False*, *robust_key: bool | None = True*, *archive: bool | None = None*, *compression: str | int | None = None*, *atomic: bool = False*) → Self

Saves the tensordict to disk, and returns a memory-mapped copy of it.

`save` (and its alias `dumps`) calls `memmap()` with
`existsok=True`. `memmap()` documents the arguments in full.
The tensordict is not changed. Read the files back with `load()`.
See [Save and load tensordicts](https://docs.pytorch.org/tensordict/stable/saving.html#saving).

Parameters:

- **prefix** (*str**or**Path**,**optional*) - the directory where the tensordict
is written, one file per leaf. If `prefix` ends with
`".tdz"` or `archive=True`, a single archive file is
written instead. Defaults to `None`, which writes into the
directory of the tensordict if it is already memory-mapped
there, and otherwise into anonymous shared memory, which has no
file name and lasts as long as the process.
- **copy_existing** (*bool**,**optional*) - if `False`, a leaf that is
already a memory-mapped tensor with a file at another path
raises a `RuntimeError`. If `True`, the file is copied to
the new path. Defaults to `False`.

Keyword Arguments:

- **num_threads** (*int**,**optional*) - the number of threads that write the
files. Defaults to `0`, which writes them in the calling
thread.
- **return_early** (*bool**,**optional*) - if `True` and `num_threads > 1`,
returns a [`TensorDictFuture`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.utils.TensorDictFuture.html#tensordict.utils.TensorDictFuture) while
the threads write the files. Defaults to `False`.
- **share_non_tensor** (*bool**,**optional*) - if `True`, non-tensor data is
shared between the processes of the node. Defaults to
`False`.
- **robust_key** (*bool**,**optional*) - if `True`, keys are percent-encoded
in file names, so that every file stays inside `prefix`.
Defaults to `True`.
- **archive** (*bool**,**optional*) - if `True`, writes a single archive
file. Defaults to `None`, which writes an archive if
`prefix` ends with `".tdz"`.
- **compression** (*str**or**int**,**optional*) - the compression of the
archive entries. Defaults to `None`, which stores them
uncompressed.
- **atomic** (*bool**,**optional*) - if `True`, the directory is written
next to `prefix` and then renamed to it, so that an error or
a crash during the write leaves `prefix` as it was. Not
supported on Windows for a directory. An archive is always
written this way.
`memmap()` gives the details. Defaults to `False`.

Returns:

A new memory-mapped tensordict, as `memmap()` returns.

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

Returns a tensorclass that holds only the fields of `keys`, with the other fields set to `None`.

Works as [`select()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.select.html#tensordict.TensorDictBase.select), which describes
`keys`: the leaves are not copied.

Keyword Arguments:

- **inplace** (*bool**,**optional*) - if `True`, sets the other fields of the
tensorclass to `None` and returns `self`. Defaults to
`False`.
- **strict** (*bool**,**optional*) - if `True`, a missing key raises a
`KeyError`. If `False`, missing keys are ignored. Defaults to
`True`.
- **as_tensordict** (*bool**,**optional*) - if `True`, returns the selected
entries in a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) instead of a
tensorclass. Defaults to `False`.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... x: torch.Tensor
... y: torch.Tensor
>>> foo = Foo(x=torch.zeros(3), y=torch.ones(3), batch_size=[3])
>>> print(foo.select("x").y)
None
>>> type(foo.select("x", as_tensordict=True)).__name__
'TensorDict'
```

set(*key: NestedKey*, *value: Any*, *inplace: bool = False*, *non_blocking: bool = False*)

Sets a field, and returns the tensorclass.

`tc.set("x", value)` is the same as `tc.x = value`: the value is cast
according to the class flags (see [`TensorClass`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorClass.html#tensordict.TensorClass)).
Otherwise, it works as [`set()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.set.html#tensordict.TensorDictBase.set).

Parameters:

- **key** (*NestedKey*) - the field to set. A tuple sets an entry of a nested
field: `tc.set(("a", "b"), value)` is `tc.a.set("b", value)`.
- **value** (*Any*) - the value of the field.
- **inplace** (*bool**,**optional*) - if `True` and the field holds a tensor,
the value is copied into it, which keeps its storage. Defaults to
`False`. See [In-place operations](https://docs.pytorch.org/tensordict/stable/concepts/operations.html#inplace-conventions).
- **non_blocking** (*bool**,**optional*) - if `True`, copies between devices
can run asynchronously with respect to the host. Defaults to
`False`.

Returns:

`self`.

Raises:

- **AttributeError** - if `key` is not a field of the class.
- **RuntimeError** - if the tensorclass is locked, also with
 `inplace=True`.
- **ValueError** - if `value` is the tensorclass itself.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... x: torch.Tensor
>>> foo = Foo(x=torch.zeros(3), batch_size=[3])
>>> foo.set("x", torch.ones(3)).x
tensor([1., 1., 1.])
>>> foo.set("z", torch.ones(3))
Traceback (most recent call last):
 ...
AttributeError: Cannot set the attribute 'z': Foo has no such field. Its fields are ['x'].
```

*property*size*: Any*

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

stack(*dim: int = 0*, ***, *out=None*)

Stacks tensordicts into one dense tensordict along a new dimension `dim`.

It is equivalent to [`torch.stack()`](https://docs.pytorch.org/docs/stable/generated/torch.stack.html#torch.stack), and also works under
[`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile). The tensordicts must have the same batch size
and the same keys, or a `RuntimeError` is raised. To stack
tensordicts with different keys or shapes, use `lazy_stack()`.
See [Stacking and variable-size data](https://docs.pytorch.org/tensordict/stable/concepts/stacking.html#stacking).

Parameters:

- **input** (*sequence**of**TensorDictBase*) - the tensordicts to stack.
- **dim** (*int**,**optional*) - the new batch dimension. Defaults to `0`.

Keyword Arguments:

**out** (*TensorDictBase**,**optional*) - a tensordict to write the result
into. Defaults to `None`.

Examples

```
>>> import torch
>>> from tensordict import TensorDict
>>> a = TensorDict(x=torch.zeros(3), batch_size=[3])
>>> b = TensorDict(x=torch.ones(3), batch_size=[3])
>>> TensorDict.stack([a, b]).batch_size
torch.Size([2, 3])
```

state_dict(*destination=None*, *prefix=''*, *keep_vars=False*, *flatten=True*) → dict[str, Any]

Returns the state dict of the tensorclass.

Works as [`state_dict()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.state_dict.html#tensordict.TensorDictBase.state_dict), which describes
the arguments. The tensor fields are the entries of the state dict. The
non-tensor fields and the name of the class go to the `_metadata`
attribute, under the `"_non_tensor"` and `"_type"` keys of their
level.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... x: torch.Tensor
... name: str
>>> foo = Foo(x=torch.zeros(3), name="a", batch_size=[3])
>>> sd = foo.state_dict()
>>> list(sd.keys())
['x']
>>> sd._metadata[""]["_non_tensor"]
{'name': 'a'}
```

to_tensordict(***, *retain_none: bool | None = None*) → TensorDict

Returns a [`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) with a copy of the fields of the tensorclass.

The tensors are copied, and memory-mapped or shared tensors become regular
tensors. Nested tensorclasses become tensordicts, and non-tensor fields
become [`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData) entries.

Keyword Arguments:

**retain_none** (*bool**,**optional*) - if `True`, fields set to `None` are
written as `None` entries. Defaults to `None`, which drops
them.

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

*property*type*: Any*

partial(func, *args, **keywords) - new function with partial application
of the given arguments and keywords.

unbind(*dim: int*)

Returns a tuple of tensorclasses, one per index along a batch dimension.

Works as [`unbind()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.unbind.html#tensordict.TensorDictBase.unbind): the results share
the storage of the tensorclass.

Parameters:

**dim** (*int*) - the batch dimension to remove.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... x: torch.Tensor
>>> foo = Foo(x=torch.arange(6).reshape(2, 3), batch_size=[2, 3])
>>> [item.x for item in foo.unbind(1)]
[tensor([0, 3]), tensor([1, 4]), tensor([2, 5])]
```

update_(*input_dict_or_td: dict[str, CompatibleType] | T | None = None*, *clone: bool = False*, *inplace: bool = False*, ***, *keys_to_update: [Sequence](torchrl.data.Sequence.html#torchrl.data.Sequence)[NestedKey] | None = None*, *non_blocking: bool = False*, ***kwargs*)

Copies the entries of a dictionary, a tensordict or a tensorclass into the existing entries.

Returns `self`. Works as [`update_()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.update_.html#tensordict.TensorDictBase.update_), which describes the
arguments: the existing tensors keep their storage. The fields can also
be passed as keyword arguments, `tc.update_(x=value)`. `inplace` has
no effect.

A tensorclass or dict source ignores `clone`, `non_blocking` and
`keys_to_update`: the entries that `keys_to_update` leaves out are
written too, and the call warns. Starting with TensorDict 0.17, the default
will change to writing only the entries in `keys_to_update`, as for a
TensorDict source.

Examples

```
>>> import torch
>>> from tensordict import TensorClass
>>> class Foo(TensorClass):
... x: torch.Tensor
>>> foo = Foo(x=torch.zeros(3), batch_size=[3])
>>> x = foo.x
>>> _ = foo.update_({"x": torch.ones(3)})
>>> x
tensor([1., 1., 1.])
```

update_at_(*input_dict_or_td: dict[str, CompatibleType] | T*, *idx: IndexType*, *clone: bool = False*, ***, *keys_to_update: [Sequence](torchrl.data.Sequence.html#torchrl.data.Sequence)[NestedKey] | None = None*, *non_blocking: bool = False*)

Writes the entries of a dictionary, a tensordict or a tensorclass in place at an index, and returns `self`.

Works as [`update_at_()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.update_at_.html#tensordict.TensorDictBase.update_at_), which describes
the arguments, with one difference. A tensorclass or dict source that
[`update()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update) accepts, such as one with the
batch size of the tensorclass, goes to
[`update()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.update.html#tensordict.TensorDictBase.update) instead: its entries replace
those of the tensorclass whatever the index, and the call warns. Starting
with TensorDict 0.17, the default will change to writing such a source in
place at the index, as [`update_at_()`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.update_at_.html#tensordict.TensorDictBase.update_at_)
does.