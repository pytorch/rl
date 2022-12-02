# -*- coding: utf-8 -*-
"""
TensorDict
============================
``TensorDict`` is a new tensor structure introduced in TorchRL.
"""
##############################################################################
# With RL, you need to be able to deal with multiple tensors such as actions,
# observations and reward. ``TensorDict`` makes it more convenient to deal
# with multiple tensors at the same time for operations such as casting to
# device, reshaping, stacking etc.
#
# Furthermore, different RL algorithms can deal with different input and
# outputs. The ``TensorDict`` class makes it possible to abstract away the
# differences between these algorithms.
#
# TensorDict combines the convenience of using ``dicts`` to organize your
# data with the power of pytorch tensors.
#
# Improving the modularity of codes
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# Let's suppose we have 2 datasets: Dataset A which has images and labels and
# Dataset B which has images, segmentation maps and labels.
#
# Suppose we want to train a common algorithm over these two datasets (i.e. an
# algorithm that would ignore the mask or infer it when needed).
#
# In classical pytorch we would need to do the following:
#
# **Method A**
#     .. code-block:: python
#
#       >>> for i in range(optim_steps):
#       ...     images, labels = get_data_A()
#       ...     loss = loss_module(images, labels)
#       ...     loss.backward()
#       ...     optim.step()
#       ...     optim.zero_grad()

###############################################################################
# **Method B**
#     .. code-block:: python
#
#       >>> for i in range(optim_steps):
#       ...     images, labels = get_data_B()
#       ...     loss = loss_module(images, labels)
#       ...     loss.backward()
#       ...     optim.step()
#       ...     optim.zero_grad()

###############################################################################
# We can see that this limits the reusability of code. A lot of code has to be
# rewriten because of the modality difference between the 2 datasets.
# The idea of TensorDict is to do the following:

###############################################################################
# **General Method**
#     .. code-block:: python
#
#       >>> for i in range(optim_steps):
#       ...     images, labels = get_data()
#       ...     loss = loss_module(images, labels)
#       ...     loss.backward()
#       ...     optim.step()
#       ...     optim.zero_grad()

###############################################################################
# We can now reuse the same training loop across datasets and losses.
#
# Can't I do this with a python dict?
# --------------------------------------
#
# One could argue that you could achieve the same results with a dataset
# that outputs a pytorch dict.
#     .. code-block:: python
#
#       >>> class DictDataset(Dataset):
#       ...     ...
#       ...     def __getitem__(self, idx)
#       ...         ...
#       ...         return {"images": image, "masks": mask}

###############################################################################
# However to achieve this you would need to write a complicated collate
# function that make sure that every modality is aggregated properly.

# sphinx_gallery_start_ignore
import warnings

warnings.filterwarnings("ignore")
# sphinx_gallery_end_ignore


def collate_dict_fn(dict_list):
    final_dict = {}
    for key in dict_list[0].keys():
        final_dict[key] = []
        for single_dict in dict_list:
            final_dict[key].append(single_dict[key])
        final_dict[key] = torch.stack(final_dict[key], dim=0)
    return final_dict


###############################################################################
# With TensorDicts this is now much simpler:
#
# **dataloader = Dataloader(DictDataset(), collate_fn = collate_dict_fn)**
#     .. code-block:: python
#
#       >>> class DictDataset(Dataset):
#       ...   ...
#       ...   def __getitem__(self, idx)
#       ...       ...
#       ...       return TensorDict({"images": image, "masks": mask})

###############################################################################
# Here, the collate function is as simple as:
#
# **collate_tensordict_fn = lambda tds : torch.stack(tds, dim=0)**
#
# **dataloader = Dataloader(DictDataset(), collate_fn = collate_tensordict_fn)**
#
# This is even more useful when considering nested structures
# (Which ``TensorDict`` supports).
#
# TensorDict inherits multiple properties from ``torch.Tensor`` and ``dict``
# that we will detail furtherdown.
#
# TensorDict structure
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^


import torch

###############################################################################

from tensordict.tensordict import (
    _PermutedTensorDict,
    _UnsqueezedTensorDict,
    _ViewedTensorDict,
    TensorDict,
)

###############################################################################
# TensorDict is a Datastructure indexed by either keys or numerical indices.
# The values can either be tensors, memory-mapped tensors or ``TensorDict``. The
# values need to share the same memory location (device or shared memory).
# They can however have different dtypes.
#
# Another essential property of TensorDict is the ``batch_size`` (or ``shape``)
# which is defined as the n-first dimensions of the tensors. It must be common
# across values, and it must be set explicitly when instantiating a ``TensorDict``.

a = torch.zeros(3, 4)
b = torch.zeros(3, 4, 5)

# works
tensordict = TensorDict({"a": a, "b": b}, batch_size=[3, 4])
tensordict = TensorDict({"a": a, "b": b}, batch_size=[3])
tensordict = TensorDict({"a": a, "b": b}, batch_size=[])

# does not work
try:
    tensordict = TensorDict({"a": a, "b": b}, batch_size=[3, 4, 5])
except RuntimeError:
    print("caramba!")

###############################################################################
# Nested ``TensorDict`` have therefore the following property: the parent
# ``TensorDict`` needs to have a batch_size included in the childs
# ``TensorDict`` batch size.

a = torch.zeros(3, 4)
b = TensorDict(
    {
        "c": torch.zeros(3, 4, 5, dtype=torch.int32),
        "d": torch.zeros(3, 4, 5, 6, dtype=torch.float32),
    },
    batch_size=[3, 4, 5],
)
tensordict = TensorDict({"a": a, "b": b}, batch_size=[3, 4])
print(tensordict)

###############################################################################
# ``TensorDict`` does not support algebraic operations by design.
#
# TensorDict Dictionary Features
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# ``TensorDict`` shares a lot of features with python dictionaries.

a = torch.zeros(3, 4, 5)
b = torch.zeros(3, 4)
tensordict = TensorDict({"a": a, "b": b}, batch_size=[3, 4])
print(tensordict)

###############################################################################
# ``get(key)``
# ------------------------------
# If we want to access a certain key, we can index the tensordict
# or alternatively use the ``get`` method:

print("get and __getitem__ match:", tensordict["a"] is tensordict.get("a") is a)
print(tensordict["a"].shape)

###############################################################################
# The ``get`` method also supports default values:

out = tensordict.get("foo", torch.ones(3))
print(out)

###############################################################################
# ``set(key, value)``
# ------------------------------
# The ``set()`` method can be used to set new values.
# Regular indexing also does the job:

c = torch.zeros((3, 4, 2, 2))
tensordict.set("c", c)
print(f"td[\"c\"] is c: {c is tensordict['c']}")

d = torch.zeros((3, 4, 2, 2))
tensordict["d"] = d
print(f"td[\"d\"] is d: {d is tensordict['d']}")

###############################################################################
# ``keys()``
# ------------------------------
# We can access the keys of a tensordict:

tensordict["c"] = torch.zeros(tensordict.shape)
tensordict.set("d", torch.ones(tensordict.shape))
assert (tensordict["c"] == 0).all()
assert (tensordict["d"] == 1).all()

###############################################################################
# ``values()``
# ------------------------------
# The values of a ``TensorDict`` can be retrieved with the ``values()`` function.
# Note that, unlike python ``dicts``, the ``values()`` method returns a
# generator and not a list.

for value in tensordict.values():
    print(value.shape)

###############################################################################
# ``update(tensordict_or_dict)``
# ------------------------------
# The ``update`` method can be used to update a TensorDict with another one
# (or with a dict):

tensordict.update({"a": torch.ones((3, 4, 5)), "d": 2 * torch.ones((3, 4, 2))})
# Also works with tensordict.update(TensorDict({"a":torch.ones((3, 4, 5)),
# "c":torch.ones((3, 4, 2))}, batch_size=[3,4]))
print(f"a is now equal to 1: {(tensordict['a'] == 1).all()}")
print(f"d is now equal to 2: {(tensordict['d'] == 2).all()}")

###############################################################################
# ``del``
# ------------------------------
# TensorDict also support keys deletion with the ``del`` operator:

print("before")
for k in tensordict.keys():
    print(k)

###############################################################################

del tensordict["c"]
print("after")
for k in tensordict.keys():
    print(k)

###############################################################################
# TensorDict tensor features
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# On many regards, TensorDict is a Tensor-like class: a great deal of tensor
# operation also work on tensordicts, making it easy to cast them across
# multiple tensors.
#
# Batch size
# ------------------------------
# ``TensorDict`` has a batch size which is shared across all tensors. The batch
# size can be [], unidimensional or multidimensional according to your needs,
# but it must be shared across tensors. Indeed, you cannot have items that don't
# share the batch size inside the same TensorDict:

tensordict = TensorDict(
    {"a": torch.zeros(3, 4, 5), "b": torch.zeros(3, 4)}, batch_size=[3, 4]
)
print(f"Our TensorDict is of size {tensordict.shape}")

###############################################################################
# The batch size can be changed if needed:

# we cannot add tensors that violate the batch size:
try:
    tensordict.update({"c": torch.zeros(4, 3, 1)})
except RuntimeError as err:
    print(f"Caramba! We got this error: {err}")

###############################################################################
# but it must comply with the tensor shapes:

tensordict.batch_size = [3]
assert tensordict.batch_size == torch.Size([3])
tensordict.batch_size = [3, 4]

###############################################################################

try:
    tensordict.batch_size = [4, 4]
except RuntimeError as err:
    print(f"Caramba! We got this error: {err}")

###############################################################################
# We can also fill the values of a TensorDict sequentially

tensordict = TensorDict({}, [10])
for i in range(10):
    tensordict[i] = TensorDict({"a": torch.randn(3, 4)}, [])
print(tensordict)

###############################################################################
# If all values are not filled, they get the default value of zero.

tensordict = TensorDict({}, [10])
for i in range(2):
    tensordict[i] = TensorDict({"a": torch.randn(3, 4)}, [])
assert (tensordict[9]["a"] == torch.zeros((3, 4))).all()
tensordict = TensorDict(
    {"a": torch.zeros(3, 4, 5), "b": torch.zeros(3, 4)}, batch_size=[3, 4]
)

###############################################################################
# Devices
# ------------------------------
# TensorDict can be sent to the desired devices like a pytorch tensor with
# ``td.cuda()`` or ``td.to(device)`` with ``device`` the desired device.
#
# Memory sharing via physical memory usage
# ------------------------------
# When on cpu, one can use either ``tensordict.memmap_()`` or
# ``tensordict.share_memory_()`` to send a ``tensordict`` to represent it as
# a memory-mapped collection of tensors or put it in shared memory resp.
#
# Tensor operations
# ------------------------------
# We can perform tensor operations among the batch dimensions:
#
# **Cloning**
#
# TensorDict supports cloning. Cloning returns the same TensorDict class
# than the original item.

tensordict_clone = tensordict.clone()
print(
    f"Content is identical ({(tensordict['a'] == tensordict_clone['a']).all()}) but duplicated ({tensordict['a'] is not tensordict_clone['a']})"
)

###############################################################################
# **Slicing and Indexing**
#
# Slicing and indexing is supported along the batch dimensions.

print(tensordict[0])

###############################################################################

print(tensordict[1:])

###############################################################################

print(tensordict[:, 2:])

###############################################################################
# **Setting Values with Indexing**
#
# In general, ``tensodict[tuple_index] = new_tensordict`` will work as long as
# the batch sizes match.
#
# If one wants to build a tensordict that keeps track of the original tensordict,
# the ``get_sub_tensordict`` method can be used: in that case, a
# ``SubTensorDict`` instance will be returned. This class will store a pointer
# to the original tensordict as well as the desired index such that tensor
# modifications can be achieved easily.

tensordict = TensorDict(
    {"a": torch.zeros(3, 4, 5), "b": torch.zeros(3, 4)}, batch_size=[3, 4]
)
# a SubTensorDict keeps track of the original one: it does not create a copy in memory of the original data
subtd = tensordict.get_sub_tensordict((slice(None), torch.tensor([1, 3])))
tensordict.fill_("a", -1)
assert (subtd["a"] == -1).all(), subtd["a"]  # the "a" key-value pair has changed

###############################################################################
# We can set values easily just by indexing the tensordict:

td2 = TensorDict({"a": torch.zeros(2, 4, 5), "b": torch.zeros(2, 4)}, batch_size=[2, 4])
tensordict[:-1] = td2
print(tensordict["a"], tensordict["b"])

###############################################################################
# **Masking**
#
# We mask ``TensorDict`` as we mask tensors.

mask = torch.BoolTensor([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]])
tensordict[mask]

###############################################################################
# **Stacking**
#
# ``TensorDict`` supports stacking. By default, stacking is done in a lazy
# fashion, returning a ``LazyStackedTensorDict`` item.

# Stack
clonned_tensordict = tensordict.clone()
staked_tensordict = torch.stack([tensordict, clonned_tensordict], dim=0)
print(staked_tensordict)

# indexing a lazy stack returns the original tensordicts
if staked_tensordict[0] is tensordict and staked_tensordict[1] is clonned_tensordict:
    print("every tensordict is awesome!")

###############################################################################
# If we want to have a contiguous tensordict, we can call ``.to_tensordict()``
# or ``.contiguous()``. It is recommended to perform this operation before
# accessing the values of the stacked tensordict for efficiency purposes.

assert isinstance(staked_tensordict.contiguous(), TensorDict)
assert isinstance(staked_tensordict.to_tensordict(), TensorDict)

###############################################################################
# **Unbind**
#
# TensorDict can be unbound along a dim over the tensordict batch size.

list_tensordict = tensordict.unbind(0)
assert type(list_tensordict) == tuple
assert len(list_tensordict) == 3
assert (torch.stack(list_tensordict, dim=0).contiguous() == tensordict).all()

###############################################################################
# **Cat**
#
# TensorDict supports cat to concatenate among a dim. The dim must be lower
# than the ``batch_dims`` (i.e. the length of the batch_size).

list_tensordict = tensordict.unbind(0)
assert torch.cat(list_tensordict, dim=0).shape[0] == 12

###############################################################################
# **View**
#
# Support for the view operation returning a ``_ViewedTensorDict``.
# Use ``to_tensordict`` to comeback to retrieve TensorDict.

assert type(tensordict.view(-1)) == _ViewedTensorDict
assert tensordict.view(-1).shape[0] == 12

###############################################################################
# **Permute**
#
# We can permute the dims of ``TensorDict``. Permute is a Lazy operation that
# returns _PermutedTensorDict. Use ``to_tensordict`` to convert to ``TensorDict``.

assert type(tensordict.permute(1, 0)) == _PermutedTensorDict
assert tensordict.permute(1, 0).batch_size == torch.Size([4, 3])

###############################################################################
# **Reshape**
#
# Reshape allows reshaping the ``TensorDict`` batch size.

assert tensordict.reshape(-1).batch_size == torch.Size([12])

###############################################################################
# **Squeeze and Unsqueeze**
#
# Tensordict also supports squeeze and unsqueeze. Unsqueeze is a lazy operation
# that returns _UnsqueezedTensorDict. Use ``to_tensordict`` to retrieve a
# tensordict after unsqueeze. Calling ``unsqueeze(dim).squeeze(dim)`` returns
# the original tensordict.

unsqueezed_tensordict = tensordict.unsqueeze(0)
assert type(unsqueezed_tensordict) == _UnsqueezedTensorDict
assert unsqueezed_tensordict.batch_size == torch.Size([1, 3, 4])

assert type(unsqueezed_tensordict.squeeze(0)) == TensorDict
assert unsqueezed_tensordict.squeeze(0) is tensordict

###############################################################################
# Have fun with TensorDict!
