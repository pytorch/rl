# Hash

*class*torchrl.envs.transforms.Hash(*in_keys: Sequence[NestedKey]*, *out_keys: Sequence[NestedKey]*, *in_keys_inv: Sequence[NestedKey] = None*, *out_keys_inv: Sequence[NestedKey] = None*, ***, *hash_fn: Callable = None*, *seed: Any | None = None*, *use_raw_nontensor: bool = False*, *repertoire: tuple[tuple[int], Any] = None*)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#Hash)

Adds a hash value to a tensordict.

Parameters:

- **in_keys** (*sequence**of**NestedKey*) - the keys of the values to hash.
- **out_keys** (*sequence**of**NestedKey*) - the keys of the resulting hashes.
- **in_keys_inv** (*sequence**of**NestedKey**,**optional*) - the keys of the values to hash during inv call.
- **out_keys_inv** (*sequence**of**NestedKey**,**optional*) - the keys of the resulting hashes during inv call.

Keyword Arguments:

- **hash_fn** (*Callable**,**optional*) - the hash function to use. The function
signature must be
`(input: Any, seed: Any | None) -> torch.Tensor`.
`seed` is only used if this transform is initialized with the
`seed` argument. Default is `Hash.reproducible_hash`.
- **seed** (*optional*) - seed to use for the hash function, if it requires one.
- **use_raw_nontensor** (*bool**,**optional*) - if `False`, data is extracted from
[`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData)/[`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack) inputs before `fn` is called
on them. If `True`, the raw [`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData)/[`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack)
inputs are given directly to `fn`, which must support those
inputs. Default is `False`.
- **repertoire** (*Dict**[**Tuple**[**int**]**,**Any**]**,**optional*) - If given, this dict stores
the inverse mappings from hashes to inputs. This repertoire isn't
copied, so it can be modified in the same workspace after the
transform instantiation and these modifications will be reflected in
the map. Missing hashes will be mapped to `None`. Default: `None`

Examples

```
>>> from torchrl.envs import GymEnv, UnaryTransform, Hash
>>> env = GymEnv("Pendulum-v1")
>>> # Add a string output
>>> env = env.append_transform(
... UnaryTransform(
... in_keys=["observation"],
... out_keys=["observation_str"],
... fn=lambda tensor: str(tensor.numpy().tobytes())))
>>> # process the string output
>>> env = env.append_transform(
... Hash(
... in_keys=["observation_str"],
... out_keys=["observation_hash"],)
... )
>>> env.observation_spec
Composite(
 observation: BoundedContinuous(
 shape=torch.Size([3]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous),
 observation_str: NonTensor(
 shape=torch.Size([]),
 space=None,
 device=cpu,
 dtype=None,
 domain=None),
 observation_hash: UnboundedDiscrete(
 shape=torch.Size([32]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.uint8, contiguous=True),
 high=Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.uint8, contiguous=True)),
 device=cpu,
 dtype=torch.uint8,
 domain=discrete),
 device=None,
 shape=torch.Size([]))
>>> env.rollout(3)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_hash: Tensor(shape=torch.Size([3, 32]), device=cpu, dtype=torch.uint8, is_shared=False),
 observation_str: NonTensorStack(
 ["b'g\\x08\\x8b\\xbexav\\xbf\\x00\\xee(>'", "b'\\x...,
 batch_size=torch.Size([3]),
 device=None),
 reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False),
 observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 observation_hash: Tensor(shape=torch.Size([3, 32]), device=cpu, dtype=torch.uint8, is_shared=False),
 observation_str: NonTensorStack(
 ["b'\\xb5\\x17\\x8f\\xbe\\x88\\xccu\\xbf\\xc0Vr?'"...,
 batch_size=torch.Size([3]),
 device=None),
 terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3]),
 device=None,
 is_shared=False)
>>> env.check_env_specs()
[torchrl][INFO] check_env_specs succeeded!
```

get_input_from_hash(*hash_tensor*) → Any[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#Hash.get_input_from_hash)

Look up the input that was given for a particular hash output.

This feature is only available if, during initialization, either the
`repertoire` argument was given or both the `in_keys_inv` and
`out_keys_inv` arguments were given.

Parameters:

**hash_tensor** (*Tensor*) - The hash output.

Returns:

The input that the hash was generated from.

Return type:

Any

*classmethod*reproducible_hash(*string*, *seed=None*)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#Hash.reproducible_hash)

Creates a reproducible 256-bit hash from a string using a seed.

Parameters:

- **string** (*str**or**None*) - The input string. If `None`, null string `""` is used.
- **seed** (*str**,**optional*) - The seed value. Default is `None`.

Returns:

Shape `(32,)` with dtype `torch.uint8`.

Return type:

Tensor

state_dict(**args*, *destination=None*, *prefix=''*, *keep_vars=False*)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#Hash.state_dict)

Return a dictionary containing references to the whole state of the module.

Both parameters and persistent buffers (e.g. running averages) are
included. Keys are corresponding parameter and buffer names.
Parameters and buffers set to `None` are not included.

Note

The returned object is a shallow copy. It contains references
to the module's parameters and buffers.

Warning

Currently `state_dict()` also accepts positional arguments for
`destination`, `prefix` and `keep_vars` in order. However,
this is being deprecated and keyword arguments will be enforced in
future releases.

Warning

Please avoid the use of argument `destination` as it is not
designed for end-users.

Parameters:

- **destination** (*dict**,**optional*) - If provided, the state of module will
be updated into the dict and the same object is returned.
Otherwise, an `OrderedDict` will be created and returned.
Default: `None`.
- **prefix** (*str**,**optional*) - a prefix added to parameter and buffer
names to compose the keys in state_dict. Default: `''`.
- **keep_vars** (*bool**,**optional*) - by default the [`Tensor`](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) s
returned in the state dict are detached from autograd. If it's
set to `True`, detaching will not be performed.
Default: `False`.

Returns:

a dictionary containing a whole state of the module

Return type:

dict

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> module.state_dict().keys()
['bias', 'weight']
```