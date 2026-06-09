# LLMHashingEnv

*class*torchrl.envs.llm.LLMHashingEnv(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/llm/envs.html#LLMHashingEnv)

A text generation environment that uses a hashing module to identify unique observations.

The primary goal of this environment is to identify token chains using a hashing function.
This allows the data to be stored in a `MCTSForest` using nothing but hashes as node
identifiers, or easily prune repeated token chains in a data structure.

Parameters:

**vocab_size** (*int*) - The size of the vocabulary. Can be omitted if the tokenizer is passed.

Keyword Arguments:

- **hashing_module** (*Callable**[**[*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*]**,*[*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)*]**,**optional*) - A hashing function that takes a tensor as input and returns a hashed tensor.
Defaults to `SipHash` if not provided.
- **observation_key** (*NestedKey**,**optional*) - The key for the observation in the TensorDict.
Defaults to "observation".
- **text_output** (*bool**,**optional*) - Whether to include the text output in the observation.
Defaults to True.
- **tokenizer** (*transformers.Tokenizer**|**None**,**optional*) - A tokenizer function that converts text to tensors.
Only used when text_output is True.
Must implement the following methods: decode and batch_decode.
Defaults to `None`.
- **text_key** (*NestedKey**|**None**,**optional*) - The key for the text output in the TensorDict.
Defaults to "text".

Examples

```
>>> from tensordict import TensorDict
>>> from torchrl.envs import LLMHashingEnv
>>> from transformers import GPT2Tokenizer
>>> tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
>>> x = tokenizer(["Check out TorchRL!"])["input_ids"]
>>> env = LLMHashingEnv(tokenizer=tokenizer)
>>> td = TensorDict(observation=x, batch_size=[1])
>>> td = env.reset(td)
>>> print(td)
TensorDict(
 fields={
 done: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 hash: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 observation: Tensor(shape=torch.Size([1, 5]), device=cpu, dtype=torch.int64, is_shared=False),
 terminated: Tensor(shape=torch.Size([1, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 text: NonTensorStack(
 ['Check out TorchRL!'],
 batch_size=torch.Size([1]),
 device=None)},
 batch_size=torch.Size([1]),
 device=None,
 is_shared=False)
```

*property*action_key*: NestedKey*

The action key of an environment.

By default, this will be "action".

If there is more than one action key in the environment, this function will raise an exception.

*property*action_keys*: list[NestedKey]*

The action keys of an environment.

By default, there will only be one key named "action".

Keys are sorted by depth in the data tree.

*property*action_spec*: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*

The `action` spec.

The `action_spec` is always stored as a composite spec.

If the action spec is provided as a simple spec, this will be returned.

```
>>> env.action_spec = Unbounded(1)
>>> env.action_spec
UnboundedContinuous(
 shape=torch.Size([1]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous)
```

If the action spec is provided as a composite spec and contains only one leaf,
this function will return just the leaf.

```
>>> env.action_spec = Composite({"nested": {"action": Unbounded(1)}})
>>> env.action_spec
UnboundedContinuous(
 shape=torch.Size([1]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous)
```

If the action spec is provided as a composite spec and has more than one leaf,
this function will return the whole spec.

```
>>> env.action_spec = Composite({"nested": {"action": Unbounded(1), "another_action": Categorical(1)}})
>>> env.action_spec
Composite(
 nested: Composite(
 action: UnboundedContinuous(
 shape=torch.Size([1]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous),
 another_action: Categorical(
 shape=torch.Size([]),
 space=DiscreteBox(n=1),
 device=cpu,
 dtype=torch.int64,
 domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))
```

To retrieve the full spec passed, use:

```
>>> env.input_spec["full_action_spec"]
```

This property is mutable.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> env = GymEnv("Pendulum-v1")
>>> env.action_spec
BoundedContinuous(
 shape=torch.Size([1]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous)
```

*property*action_spec_unbatched*: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*

Returns the action spec of the env as if it had no batch dimensions.

add_module(*name: str*, *module: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | None*) → None

Add a child module to the current module.

The module can be accessed as an attribute using the given name.

Parameters:

- **name** (*str*) - name of the child module. The child module can be
accessed from this module using the given name
- **module** (*Module*) - child module to be added to the module.

add_truncated_keys() → [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)

Adds truncated keys to the environment.

all_actions(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Generates all possible actions from the action spec.

This only works in environments with fully discrete actions.

Parameters:

**tensordict** (*TensorDictBase**,**optional*) - If given, `reset()`
is called with this tensordict.

Returns:

a tensordict object with the "action" entry updated with a batch of
all possible actions. The actions are stacked together in the
leading dimension.

any_done(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → bool

Checks if the tensordict is in a "done" state (or if an element of the batch is).

Writes the result under the "_reset" entry.

Returns: a bool indicating whether there is an element in the tensordict that is marked

as done.

Note

The tensordict passed should be a "next" tensordict or equivalent - i.e., it should not
contain a "next" value.

append_transform(*transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) | Callable[[TensorDictBase], TensorDictBase]*) → torchrl.envs.TransformedEnv

Returns a transformed environment where the callable/transform passed is applied.

Parameters:

**transform** ([*Transform*](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)*or**Callable**[**[**TensorDictBase**]**,**TensorDictBase**]*) - the transform to apply
to the environment.

Examples

```
>>> from torchrl.envs import GymEnv
>>> import torch
>>> env = GymEnv("CartPole-v1")
>>> loc = 0.5
>>> scale = 1.0
>>> transform = lambda data: data.set("observation", (data.get("observation") - loc)/scale)
>>> env = env.append_transform(transform=transform)
>>> print(env)
TransformedEnv(
 env=GymEnv(env=CartPole-v1, batch_size=torch.Size([]), device=cpu),
 transform=_CallableTransform(keys=[]))
```

apply(*fn: Callable[[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)], None]*) → Self

Apply `fn` recursively to every submodule (as returned by `.children()`) as well as self.

Typical use includes initializing the parameters of a model
(see also [torch.nn.init](https://docs.pytorch.org/docs/stable/nn.init.html#nn-init-doc)).

Parameters:

**fn** (`Module` -> None) - function to be applied to each submodule

Returns:

self

Return type:

Module

Example:

```
>>> @torch.no_grad()
>>> def init_weights(m):
>>> print(m)
>>> if type(m) is nn.Linear:
>>> m.weight.fill_(1.0)
>>> print(m.weight)
>>> net = nn.Sequential(nn.Linear(2, 2), nn.Linear(2, 2))
>>> net.apply(init_weights)
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[1., 1.],
 [1., 1.]], requires_grad=True)
Linear(in_features=2, out_features=2, bias=True)
Parameter containing:
tensor([[1., 1.],
 [1., 1.]], requires_grad=True)
Sequential(
 (0): Linear(in_features=2, out_features=2, bias=True)
 (1): Linear(in_features=2, out_features=2, bias=True)
)
```

auto_specs_(*policy: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)]*, ***, *tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *action_key: NestedKey | list[NestedKey] = 'action'*, *done_key: NestedKey | list[NestedKey] | None = None*, *observation_key: NestedKey | list[NestedKey] = 'observation'*, *reward_key: NestedKey | list[NestedKey] = 'reward'*)

Automatically sets the specifications (specs) of the environment based on a random rollout using a given policy.

This method performs a rollout using the provided policy to infer the input and output specifications of the environment.
It updates the environment's specs for actions, observations, rewards, and done signals based on the data collected
during the rollout.

Parameters:

**policy** (*Callable**[**[**TensorDictBase**]**,**TensorDictBase**]*) - A callable policy that takes a TensorDictBase as input and returns a TensorDictBase as output.
This policy is used to perform the rollout and determine the specs.

Keyword Arguments:

- **tensordict** (*TensorDictBase**,**optional*) - An optional TensorDictBase instance to be used as the initial state for the rollout.
If not provided, the environment's reset method will be called to obtain the initial state.
- **action_key** (*NestedKey**or**List**[**NestedKey**]**,**optional*) - The key(s) used to identify actions in the TensorDictBase. Defaults to "action".
- **done_key** (*NestedKey**or**List**[**NestedKey**]**,**optional*) - The key(s) used to identify done signals in the TensorDictBase. Defaults to `None`, which will
attempt to use ["done", "terminated", "truncated"] as potential keys.
- **observation_key** (*NestedKey**or**List**[**NestedKey**]**,**optional*) - The key(s) used to identify observations in the TensorDictBase. Defaults to "observation".
- **reward_key** (*NestedKey**or**List**[**NestedKey**]**,**optional*) - The key(s) used to identify rewards in the TensorDictBase. Defaults to "reward".

Returns:

The environment instance with updated specs.

Return type:

[EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)

Raises:

**RuntimeError** - If there are keys in the output specs that are not accounted for in the provided keys.

*property*batch_dims*: int*

Number of batch dimensions of the env.

*property*batch_locked*: bool*

Whether the environment can be used with a batch size different from the one it was initialized with or not.

If True, the env needs to be used with a tensordict having the same batch size as the env.
batch_locked is an immutable property.

*property*batch_size*: [Size](https://docs.pytorch.org/docs/stable/size.html#torch.Size)*

Number of envs batched in this environment instance organised in a torch.Size() object.

Environment may be similar or different but it is assumed that they have little if
not no interactions between them (e.g., multi-task or batched execution
in parallel).

bfloat16() → Self

Casts all floating point parameters and buffers to `bfloat16` datatype.

Note

This method modifies the module in-place.

Returns:

self

Return type:

Module

buffers(*recurse: bool = True*) → Iterator[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]

Return an iterator over module buffers.

Parameters:

**recurse** (*bool*) - if True, then yields buffers of this module
and all submodules. Otherwise, yields only buffers that
are direct members of this module.

Yields:

*torch.Tensor* - module buffer

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> for buf in model.buffers():
>>> print(type(buf), buf.size())
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```

cardinality(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*) → int

The cardinality of the action space.

By default, this is just a wrapper around `env.action_space.cardinality`.

This class is useful when the action spec is variable:

- The number of actions can be undefined, e.g., `Categorical(n=-1)`;
- The action cardinality may depend on the action mask;
- The shape can be dynamic, as in `Unbound(shape=(-1))`.

In these cases, the `cardinality()` should be overwritten,

Parameters:

**tensordict** (*TensorDictBase**,**optional*) - a tensordict containing the data required to compute the cardinality.

check_env_specs(**args*, ***kwargs*)

Tests an environment specs against the results of short rollout.

This test function should be used as a sanity check for an env wrapped with
torchrl's EnvBase subclasses: any discrepancy between the expected data and
the data collected should raise an assertion error.

A broken environment spec will likely make it impossible to use parallel
environments.

Parameters:

- **env** ([*EnvBase*](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)) - the env for which the specs have to be checked against data.
- **return_contiguous** (*bool**,**optional*) - if `True`, the random rollout will be called with
return_contiguous=True. This will fail in some cases (e.g. heterogeneous shapes
of inputs/outputs). Defaults to `None` (determined by the presence of dynamic specs).
- **check_dtype** (*bool**,**optional*) - if False, dtype checks will be skipped.
Defaults to True.
- **seed** (*int**,**optional*) - for reproducibility, a seed can be set.
The seed will be set in pytorch temporarily, then the RNG state will
be reverted to what it was before. For the env, we set the seed but since
setting the rng state back to what is was isn't a feature of most environment,
we leave it to the user to accomplish that.
Defaults to `None`.
- **tensordict** (*TensorDict**,**optional*) - an optional tensordict instance to use for reset.
- **break_when_any_done** (*bool**or**str**,**optional*) - value for `break_when_any_done` in [`rollout()`](torchrl.envs.EnvBase.html#id2).
If `"both"`, the test is run on both True and False.

Caution: this function resets the env seed. It should be used "offline" to
check that an env is adequately constructed, but it may affect the seeding
of an experiment and as such should be kept out of training scripts.

children() → Iterator[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)]

Return an iterator over immediate children modules.

Yields:

*Module* - a child module

*property*collector*: [BaseCollector](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector) | None*

Returns the collector associated with the container, if it exists.

compile(***, *warmup: int | None = None*, ***kwargs*)

Compile `step_and_maybe_reset()` and return `self`.

Parameters:

- **warmup** (*int**,**optional*) - if provided, the first `warmup` calls to
`step_and_maybe_reset()` will run eagerly so the input
[`TensorDict`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict) layout (keys, names, nesting)
stabilizes before tracing. Compilation kicks in on call
`warmup`. This avoids the recompile that otherwise happens
because the first post-`reset()` tensordict and the
steady-state post-`step_mdp` tensordict have different
layouts. Defaults to `None` (compile immediately).
- ****kwargs** - forwarded to [`torch.compile()`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html#torch.compile). Common ones include
`backend`, `mode`, `fullgraph`, and `dynamic`.

The same behavior is reachable directly from the constructor of every
`EnvBase` subclass (and of
`TransformedEnv`) via the `compile=` kwarg:

```
env = GymEnv(
 "HalfCheetah-v4",
 compile={"warmup": 4, "fullgraph": True, "mode": "reduce-overhead"},
)

env = TransformedEnv(
 GymEnv("HalfCheetah-v4"),
 Compose(...),
 compile={"warmup": 4, "fullgraph": True},
)
```

See `eager()` to undo it.

configure_parallel(***, *use_buffers: bool | None = None*, *shared_memory: bool | None = None*, *memmap: bool | None = None*, *mp_start_method: str | None = None*, *num_threads: int | None = None*, *num_sub_threads: int | None = None*, *non_blocking: bool | None = None*, *daemon: bool | None = None*) → [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)

Configure parallel execution parameters.

This method allows configuring parameters for parallel environment
execution before the environment is started. It is only effective
on `BatchedEnvBase` and its subclasses.

Parameters:

- **use_buffers** (*bool**,**optional*) - whether communication between workers should
occur via circular preallocated memory buffers.
- **shared_memory** (*bool**,**optional*) - whether the returned tensordict will be
placed in shared memory.
- **memmap** (*bool**,**optional*) - whether the returned tensordict will be placed
in memory map.
- **mp_start_method** (*str**,**optional*) - the multiprocessing start method.
- **num_threads** (*int**,**optional*) - number of threads for this process.
- **num_sub_threads** (*int**,**optional*) - number of threads of the subprocesses.
- **non_blocking** (*bool**,**optional*) - if `True`, device moves will be done using
the `non_blocking=True` option.
- **daemon** (*bool**,**optional*) - whether the processes should be daemonized.

Returns:

Returns self for method chaining.

Return type:

self

Raises:

- **NotImplementedError** - If called on an environment that does not support
 parallel configuration.
- **RuntimeError** - If called after the environment has already started.

Example

```
>>> env = DMControlEnv("cheetah", "run", num_envs=4)
>>> env.configure_parallel(use_buffers=True, num_threads=2)
>>> env.reset() # Environment starts here, configure_parallel no longer effective
```

cpu() → Self

Move all model parameters and buffers to the CPU.

Note

This method modifies the module in-place.

Returns:

self

Return type:

Module

cuda(*device: int | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*) → Self

Move all model parameters and buffers to the GPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on GPU while being optimized.

Note

This method modifies the module in-place.

Parameters:

**device** (*int**,**optional*) - if specified, all parameters will be
copied to that device

Returns:

self

Return type:

Module

*property*done_key

The done key of an environment.

By default, this will be "done".

If there is more than one done key in the environment, this function will raise an exception.

*property*done_keys*: list[NestedKey]*

The done keys of an environment.

By default, there will only be one key named "done".

Keys are sorted by depth in the data tree.

*property*done_keys_groups

A list of done keys, grouped as the reset keys.

This is a list of lists. The outer list has the length of reset keys, the
inner lists contain the done keys (eg, done and truncated) that can
be read to determine a reset when it is absent.

*property*done_spec*: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*

The `done` spec.

The `done_spec` is always stored as a composite spec.

If the done spec is provided as a simple spec, this will be returned.

```
>>> env.done_spec = Categorical(2, dtype=torch.bool)
>>> env.done_spec
Categorical(
 shape=torch.Size([]),
 space=DiscreteBox(n=2),
 device=cpu,
 dtype=torch.bool,
 domain=discrete)
```

If the done spec is provided as a composite spec and contains only one leaf,
this function will return just the leaf.

```
>>> env.done_spec = Composite({"nested": {"done": Categorical(2, dtype=torch.bool)}})
>>> env.done_spec
Categorical(
 shape=torch.Size([]),
 space=DiscreteBox(n=2),
 device=cpu,
 dtype=torch.bool,
 domain=discrete)
```

If the done spec is provided as a composite spec and has more than one leaf,
this function will return the whole spec.

```
>>> env.done_spec = Composite({"nested": {"done": Categorical(2, dtype=torch.bool), "another_done": Categorical(2, dtype=torch.bool)}})
>>> env.done_spec
Composite(
 nested: Composite(
 done: Categorical(
 shape=torch.Size([]),
 space=DiscreteBox(n=2),
 device=cpu,
 dtype=torch.bool,
 domain=discrete),
 another_done: Categorical(
 shape=torch.Size([]),
 space=DiscreteBox(n=2),
 device=cpu,
 dtype=torch.bool,
 domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))
```

To always retrieve the full spec passed, use:

```
>>> env.output_spec["full_done_spec"]
```

This property is mutable.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> env = GymEnv("Pendulum-v1")
>>> env.done_spec
Categorical(
 shape=torch.Size([1]),
 space=DiscreteBox(n=2),
 device=cpu,
 dtype=torch.bool,
 domain=discrete)
```

*property*done_spec_unbatched*: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*

Returns the done spec of the env as if it had no batch dimensions.

double() → Self

Casts all floating point parameters and buffers to `double` datatype.

Note

This method modifies the module in-place.

Returns:

self

Return type:

Module

eager()

Restore eager `step_and_maybe_reset()` execution and return `self`.

empty_cache()

Erases all the cached values.

For regular envs, the key lists (reward, done etc) are cached, but in some cases
they may change during the execution of the code (eg, when adding a transform).

eval() → Self

Set the module in evaluation mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e. whether they are affected, e.g. `Dropout`, `BatchNorm`,
etc.

This is equivalent with [`self.train(False)`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.train).

See [Locally disabling gradient computation](https://docs.pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc) for a comparison between
.eval() and several similar mechanisms that may be confused with it.

Returns:

self

Return type:

Module

extra_repr() → str

Return the extra representation of the module.

To print customized extra information, you should re-implement
this method in your own modules. Both single-line and multi-line
strings are acceptable.

fake_tensordict() → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Returns a fake tensordict with key-value pairs that match in shape, device and dtype what can be expected during an environment rollout.

float() → Self

Casts all floating point parameters and buffers to `float` datatype.

Note

This method modifies the module in-place.

Returns:

self

Return type:

Module

forward(**args*, ***kwargs*)

Define the computation performed at every call.

Should be overridden by all subclasses.

Note

Although the recipe for forward pass needs to be defined within
this function, one should call the `Module` instance afterwards
instead of this since the former takes care of running the
registered hooks while the latter silently ignores them.

*property*full_action_spec*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

The full action spec.

`full_action_spec` is a Composite` instance
that contains all the action entries.

Examples

```
>>> from torchrl.envs import BraxEnv
>>> for envname in BraxEnv.available_envs:
... break
>>> env = BraxEnv(envname)
>>> env.full_action_spec
Composite(
 action: BoundedContinuous(
 shape=torch.Size([8]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([8]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous), device=cpu, shape=torch.Size([]))
```

*property*full_action_spec_unbatched*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

Returns the action spec of the env as if it had no batch dimensions.

*property*full_done_spec*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

The full done spec.

`full_done_spec` is a Composite` instance
that contains all the done entries.
It can be used to generate fake data with a structure that mimics the
one obtained at runtime.

Examples

```
>>> import gymnasium
>>> from torchrl.envs import GymWrapper
>>> env = GymWrapper(gymnasium.make("Pendulum-v1"))
>>> env.full_done_spec
Composite(
 done: Categorical(
 shape=torch.Size([1]),
 space=DiscreteBox(n=2),
 device=cpu,
 dtype=torch.bool,
 domain=discrete),
 truncated: Categorical(
 shape=torch.Size([1]),
 space=DiscreteBox(n=2),
 device=cpu,
 dtype=torch.bool,
 domain=discrete), device=cpu, shape=torch.Size([]))
```

*property*full_done_spec_unbatched*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

Returns the done spec of the env as if it had no batch dimensions.

*property*full_observation_spec_unbatched*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

Returns the observation spec of the env as if it had no batch dimensions.

*property*full_reward_spec*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

The full reward spec.

`full_reward_spec` is a Composite` instance
that contains all the reward entries.

Examples

```
>>> import gymnasium
>>> from torchrl.envs import GymWrapper, TransformedEnv, RenameTransform
>>> base_env = GymWrapper(gymnasium.make("Pendulum-v1"))
>>> env = TransformedEnv(base_env, RenameTransform("reward", ("nested", "reward")))
>>> env.full_reward_spec
Composite(
 nested: Composite(
 reward: UnboundedContinuous(
 shape=torch.Size([1]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous), device=None, shape=torch.Size([])), device=cpu, shape=torch.Size([]))
```

*property*full_reward_spec_unbatched*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

Returns the reward spec of the env as if it had no batch dimensions.

*property*full_state_spec*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

The full state spec.

`full_state_spec` is a Composite` instance
that contains all the state entries (ie, the input data that is not action).

Examples

```
>>> from torchrl.envs import BraxEnv
>>> for envname in BraxEnv.available_envs:
... break
>>> env = BraxEnv(envname)
>>> env.full_state_spec
Composite(
 state: Composite(
 pipeline_state: Composite(
 q: UnboundedContinuous(
 shape=torch.Size([15]),
 space=None,
 device=cpu,
 dtype=torch.float32,
 domain=continuous),
 [...], device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))
```

*property*full_state_spec_unbatched*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

Returns the state spec of the env as if it had no batch dimensions.

get_buffer(*target: str*) → [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Return the buffer given by `target` if it exists, otherwise throw an error.

See the docstring for `get_submodule` for a more detailed
explanation of this method's functionality as well as how to
correctly specify `target`.

Parameters:

**target** - The fully-qualified string name of the buffer
to look for. (See `get_submodule` for how to specify a
fully-qualified string.)

Returns:

The buffer referenced by `target`

Return type:

[torch.Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)

Raises:

**AttributeError** - If the target string references an invalid
 path or resolves to something that is not a
 buffer

get_extra_state() → Any

Return any extra state to include in the module's state_dict.

Implement this and a corresponding `set_extra_state()` for your module
if you need to store extra state. This function is called when building the
module's state_dict().

Note that extra state should be picklable to ensure working serialization
of the state_dict. We only provide backwards compatibility guarantees
for serializing Tensors; other objects may break backwards compatibility if
their serialized pickled form changes.

Returns:

Any extra state to store in the module's state_dict

Return type:

object

get_parameter(*target: str*) → [Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)

Return the parameter given by `target` if it exists, otherwise throw an error.

See the docstring for `get_submodule` for a more detailed
explanation of this method's functionality as well as how to
correctly specify `target`.

Parameters:

**target** - The fully-qualified string name of the Parameter
to look for. (See `get_submodule` for how to specify a
fully-qualified string.)

Returns:

The Parameter referenced by `target`

Return type:

torch.nn.Parameter

Raises:

**AttributeError** - If the target string references an invalid
 path or resolves to something that is not an
 `nn.Parameter`

get_submodule(*target: str*) → [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Return the submodule given by `target` if it exists, otherwise throw an error.

For example, let's say you have an `nn.Module` `A` that
looks like this:

```
A(
 (net_b): Module(
 (net_c): Module(
 (conv): Conv2d(16, 33, kernel_size=(3, 3), stride=(2, 2))
 )
 (linear): Linear(in_features=100, out_features=200, bias=True)
 )
)
```

(The diagram shows an `nn.Module` `A`. `A` which has a nested
submodule `net_b`, which itself has two submodules `net_c`
and `linear`. `net_c` then has a submodule `conv`.)

To check whether or not we have the `linear` submodule, we
would call `get_submodule("net_b.linear")`. To check whether
we have the `conv` submodule, we would call
`get_submodule("net_b.net_c.conv")`.

The runtime of `get_submodule` is bounded by the degree
of module nesting in `target`. A query against
`named_modules` achieves the same result, but it is O(N) in
the number of transitive modules. So, for a simple check to see
if some submodule exists, `get_submodule` should always be
used.

Parameters:

**target** - The fully-qualified string name of the submodule
to look for. (See above example for how to specify a
fully-qualified string.)

Returns:

The submodule referenced by `target`

Return type:

[torch.nn.Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)

Raises:

**AttributeError** - If at any point along the path resulting from
 the target string the (sub)path resolves to a non-existent
 attribute name or an object that is not an instance of `nn.Module`.

half() → Self

Casts all floating point parameters and buffers to `half` datatype.

Note

This method modifies the module in-place.

Returns:

self

Return type:

Module

*property*input_spec*: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*

Input spec.

The composite spec containing all specs for data input to the environments.

It contains:

- "full_action_spec": the spec of the input actions
- "full_state_spec": the spec of all other environment inputs

This attribute is locked and should be read-only.
Instead, to set the specs contained in it, use the respective properties.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> env = GymEnv("Pendulum-v1")
>>> env.input_spec
Composite(
 full_state_spec: None,
 full_action_spec: Composite(
 action: BoundedContinuous(
 shape=torch.Size([1]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([1]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))
```

*property*input_spec_unbatched*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

Returns the input spec of the env as if it had no batch dimensions.

ipu(*device: int | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*) → Self

Move all model parameters and buffers to the IPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on IPU while being optimized.

Note

This method modifies the module in-place.

Parameters:

**device** (*int**,**optional*) - if specified, all parameters will be
copied to that device

Returns:

self

Return type:

Module

*property*is_spec_locked

Gets whether the environment's specs are locked.

This property can be modified directly.

Returns:

True if the specs are locked, False otherwise.

Return type:

bool

See also

[Locking environment specs](../envs_api.html#environment-lock).

load_state_dict(*state_dict: Mapping[str, Any]*, *strict: bool = True*, *assign: bool = False*)

Copy parameters and buffers from `state_dict` into this module and its descendants.

If `strict` is `True`, then
the keys of `state_dict` must exactly match the keys returned
by this module's [`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) function.

Warning

If `assign` is `True` the optimizer must be created after
the call to `load_state_dict` unless
[`get_swap_module_params_on_conversion()`](https://docs.pytorch.org/docs/stable/future_mod.html#torch.__future__.get_swap_module_params_on_conversion) is `True`.

Parameters:

- **state_dict** (*dict*) - a dict containing parameters and
persistent buffers.
- **strict** (*bool**,**optional*) - whether to strictly enforce that the keys
in `state_dict` match the keys returned by this module's
[`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) function. Default: `True`
- **assign** (*bool**,**optional*) - When set to `False`, the properties of the tensors
in the current module are preserved whereas setting it to `True` preserves
properties of the Tensors in the state dict. The only
exception is the `requires_grad` field of `Parameter`
for which the value from the module is preserved. Default: `False`

Returns:

- `missing_keys` is a list of str containing any keys that are expected

by this module but missing from the provided `state_dict`.
- `unexpected_keys` is a list of str containing the keys that are not

expected by this module but present in the provided `state_dict`.

Return type:

`NamedTuple` with `missing_keys` and `unexpected_keys` fields

Note

If a parameter or buffer is registered as `None` and its corresponding key
exists in `state_dict`, `load_state_dict()` will raise a
`RuntimeError`.

*classmethod*make_parallel(*create_env_fn*, ***, *num_envs: int = 1*, *create_env_kwargs: dict | Sequence[dict] | None = None*, *pin_memory: bool = False*, *share_individual_td: bool | None = None*, *shared_memory: bool = True*, *memmap: bool = False*, *policy_proof: Callable | None = None*, *device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*, *allow_step_when_done: bool = False*, *num_threads: int | None = None*, *num_sub_threads: int = 1*, *serial_for_single: bool = False*, *non_blocking: bool = False*, *mp_start_method: str | None = None*, *use_buffers: bool | None = None*, *consolidate: bool = True*, *daemon: bool = False*, ***parallel_kwargs*) → [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)

Factory method to create a ParallelEnv from an environment creator.

This method provides a convenient way to create parallel environments
with the same signature as [`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv).

Parameters:

- **create_env_fn** (*callable*) - A callable that creates an environment instance.
- **num_envs** (*int**,**optional*) - Number of parallel environments. Defaults to 1.
- **create_env_kwargs** (*dict**or**list**of**dicts**,**optional*) - kwargs to be used
with the environments being created.
- **pin_memory** (*bool**,**optional*) - Whether to pin memory. Defaults to False.
- **share_individual_td** (*bool**,**optional*) - if `True`, a different tensordict
is created for every process/worker and a lazy stack is returned.
- **shared_memory** (*bool**,**optional*) - whether the returned tensordict will be
placed in shared memory. Defaults to True.
- **memmap** (*bool**,**optional*) - whether the returned tensordict will be placed
in memory map. Defaults to False.
- **policy_proof** (*callable**,**optional*) - if provided, it'll be used to get
the list of tensors to return through step() and reset() methods.
- **device** (*str**,**int**,*[*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*,**optional*) - The device of the batched
environment.
- **allow_step_when_done** (*bool**,**optional*) - Allow stepping when done.
Defaults to False.
- **num_threads** (*int**,**optional*) - number of threads for this process.
- **num_sub_threads** (*int**,**optional*) - number of threads of the subprocesses.
Defaults to 1.
- **serial_for_single** (*bool**,**optional*) - if `True`, creating a parallel
environment with a single worker will return a SerialEnv instead.
Defaults to False.
- **non_blocking** (*bool**,**optional*) - if `True`, device moves will be done
using the `non_blocking=True` option. Defaults to False.
- **mp_start_method** (*str**,**optional*) - the multiprocessing start method.
- **use_buffers** (*bool**,**optional*) - whether communication between workers
should occur via circular preallocated memory buffers.
- **consolidate** (*bool**,**optional*) - Whether to consolidate tensordicts.
Defaults to True.
- **daemon** (*bool**,**optional*) - whether the processes should be daemonized.
Defaults to False.
- ****parallel_kwargs** - Additional keyword arguments passed to ParallelEnv.

Returns:

A ParallelEnv (or SerialEnv if serial_for_single=True and num_envs=1).

Return type:

[EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)

make_tensordict(*input: str | list[str]*) → [TensorDict](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)[[source]](../../_modules/torchrl/envs/llm/envs.html#LLMHashingEnv.make_tensordict)

Converts a string or list of strings in a TensorDict with appropriate shape and device.

maybe_reset(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Checks the done keys of the input tensordict and, if needed, resets the environment where it is done.

Parameters:

**tensordict** (*TensorDictBase*) - a tensordict coming from the output of `step_mdp()`.

Returns:

A tensordict that is identical to the input where the environment was
not reset and contains the new reset data where the environment was reset.

modules(*remove_duplicate: bool = True*) → Iterator[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)]

Return an iterator over all modules in the network.

Parameters:

**remove_duplicate** - whether to remove the duplicated module instances in the result
or not.

Yields:

*Module* - a module in the network

Note

Duplicate modules are returned only once by default. In the following
example, `l` will be returned only once.

Example:

```
>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.modules()):
... print(idx, '->', m)

0 -> Sequential(
 (0): Linear(in_features=2, out_features=2, bias=True)
 (1): Linear(in_features=2, out_features=2, bias=True)
)
1 -> Linear(in_features=2, out_features=2, bias=True)
```

mtia(*device: int | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*) → Self

Move all model parameters and buffers to the MTIA.

This also makes associated parameters and buffers different objects. So
it should be called before constructing the optimizer if the module will
live on MTIA while being optimized.

Note

This method modifies the module in-place.

Parameters:

**device** (*int**,**optional*) - if specified, all parameters will be
copied to that device

Returns:

self

Return type:

Module

named_buffers(*prefix: str = ''*, *recurse: bool = True*, *remove_duplicate: bool = True*) → Iterator[tuple[str, [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)]]

Return an iterator over module buffers, yielding both the name of the buffer as well as the buffer itself.

Parameters:

- **prefix** (*str*) - prefix to prepend to all buffer names.
- **recurse** (*bool**,**optional*) - if True, then yields buffers of this module
and all submodules. Otherwise, yields only buffers that
are direct members of this module. Defaults to True.
- **remove_duplicate** (*bool**,**optional*) - whether to remove the duplicated buffers in the result. Defaults to True.

Yields:

*(str, torch.Tensor)* - Tuple containing the name and buffer

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, buf in self.named_buffers():
>>> if name in ['running_var']:
>>> print(buf.size())
```

named_children() → Iterator[tuple[str, [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)]]

Return an iterator over immediate children modules, yielding both the name of the module as well as the module itself.

Yields:

*(str, Module)* - Tuple containing a name and child module

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, module in model.named_children():
>>> if name in ['conv4', 'conv5']:
>>> print(module)
```

named_modules(*memo: set[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)] | None = None*, *prefix: str = ''*, *remove_duplicate: bool = True*)

Return an iterator over all modules in the network, yielding both the name of the module as well as the module itself.

Parameters:

- **memo** - a memo to store the set of modules already added to the result
- **prefix** - a prefix that will be added to the name of the module
- **remove_duplicate** - whether to remove the duplicated module instances in the result
or not

Yields:

*(str, Module)* - Tuple of name and module

Note

Duplicate modules are returned only once. In the following
example, `l` will be returned only once.

Example:

```
>>> l = nn.Linear(2, 2)
>>> net = nn.Sequential(l, l)
>>> for idx, m in enumerate(net.named_modules()):
... print(idx, '->', m)

0 -> ('', Sequential(
 (0): Linear(in_features=2, out_features=2, bias=True)
 (1): Linear(in_features=2, out_features=2, bias=True)
))
1 -> ('0', Linear(in_features=2, out_features=2, bias=True))
```

named_parameters(*prefix: str = ''*, *recurse: bool = True*, *remove_duplicate: bool = True*) → Iterator[tuple[str, [Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)]]

Return an iterator over module parameters, yielding both the name of the parameter as well as the parameter itself.

Parameters:

- **prefix** (*str*) - prefix to prepend to all parameter names.
- **recurse** (*bool*) - if True, then yields parameters of this module
and all submodules. Otherwise, yields only parameters that
are direct members of this module.
- **remove_duplicate** (*bool**,**optional*) - whether to remove the duplicated
parameters in the result. Defaults to True.

Yields:

*(str, Parameter)* - Tuple containing the name and parameter

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> for name, param in self.named_parameters():
>>> if name in ['bias']:
>>> print(param.size())
```

*property*observation_keys*: list[NestedKey]*

The observation keys of an environment.

By default, there will only be one key named "observation".

Keys are sorted by depth in the data tree.

*property*observation_spec*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

Observation spec.

Must be a [`torchrl.data.Composite`](torchrl.data.Composite.html#torchrl.data.Composite) instance.
The keys listed in the spec are directly accessible after reset and step.

In TorchRL, even though they are not properly speaking "observations"
all info, states, results of transforms etc. outputs from the environment are stored in the
`observation_spec`.

Therefore, `"observation_spec"` should be thought as
a generic data container for environment outputs that are not done or reward data.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> env = GymEnv("Pendulum-v1")
>>> env.observation_spec
Composite(
 observation: BoundedContinuous(
 shape=torch.Size([3]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous), device=cpu, shape=torch.Size([]))
```

*property*observation_spec_unbatched*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

Returns the observation spec of the env as if it had no batch dimensions.

*property*output_spec*: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*

Output spec.

The composite spec containing all specs for data output from the environments.

It contains:

- "full_reward_spec": the spec of reward
- "full_done_spec": the spec of done
- "full_observation_spec": the spec of all other environment outputs

This attribute is locked and should be read-only.
Instead, to set the specs contained in it, use the respective properties.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> env = GymEnv("Pendulum-v1")
>>> env.output_spec
Composite(
 full_reward_spec: Composite(
 reward: UnboundedContinuous(
 shape=torch.Size([1]),
 space=None,
 device=cpu,
 dtype=torch.float32,
 domain=continuous), device=cpu, shape=torch.Size([])),
 full_observation_spec: Composite(
 observation: BoundedContinuous(
 shape=torch.Size([3]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous), device=cpu, shape=torch.Size([])),
 full_done_spec: Composite(
 done: Categorical(
 shape=torch.Size([1]),
 space=DiscreteBox(n=2),
 device=cpu,
 dtype=torch.bool,
 domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))
```

*property*output_spec_unbatched*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

Returns the output spec of the env as if it had no batch dimensions.

parameters(*recurse: bool = True*) → Iterator[[Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter)]

Return an iterator over module parameters.

This is typically passed to an optimizer.

Parameters:

**recurse** (*bool*) - if True, then yields parameters of this module
and all submodules. Otherwise, yields only parameters that
are direct members of this module.

Yields:

*Parameter* - module parameter

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> for param in model.parameters():
>>> print(type(param), param.size())
<class 'torch.Tensor'> (20L,)
<class 'torch.Tensor'> (20L, 1L, 5L, 5L)
```

rand_action(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*)

Performs a random action given the action_spec attribute.

Parameters:

**tensordict** (*TensorDictBase**,**optional*) - tensordict where the resulting action should be written.

Returns:

a tensordict object with the "action" entry updated with a random
sample from the action-spec.

rand_step(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Performs a random step in the environment given the action_spec attribute.

Parameters:

**tensordict** (*TensorDictBase**,**optional*) - tensordict where the resulting info should be written.

Returns:

a tensordict object with the new observation after a random step in the environment. The action will
be stored with the "action" key.

register_backward_hook(*hook: Callable[[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)], tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None]*) → RemovableHandle

Register a backward hook on the module.

This function is deprecated in favor of [`register_full_backward_hook()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook) and
the behavior of this function will change in future versions.

Returns:

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type:

`torch.utils.hooks.RemovableHandle`

register_buffer(*name: str*, *tensor: [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None*, *persistent: bool = True*) → None

Add a buffer to the module.

This is typically used to register a buffer that should not be
considered a model parameter. For example, BatchNorm's `running_mean`
is not a parameter, but is part of the module's state. Buffers, by
default, are persistent and will be saved alongside parameters. This
behavior can be changed by setting `persistent` to `False`. The
only difference between a persistent buffer and a non-persistent buffer
is that the latter will not be a part of this module's
`state_dict`.

Buffers can be accessed as attributes using given names.

Parameters:

- **name** (*str*) - name of the buffer. The buffer can be accessed
from this module using the given name
- **tensor** (*Tensor**or**None*) - buffer to be registered. If `None`, then operations
that run on buffers, such as `cuda`, are ignored. If `None`,
the buffer is **not** included in the module's `state_dict`.
- **persistent** (*bool*) - whether the buffer is part of this module's
`state_dict`.

Example:

```
>>> # xdoctest: +SKIP("undefined vars")
>>> self.register_buffer('running_mean', torch.zeros(num_features))
```

register_collector(*collector: [BaseCollector](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)*)

Registers a collector with the environment.

Parameters:

**collector** ([*BaseCollector*](torchrl.collectors.BaseCollector.html#torchrl.collectors.BaseCollector)) - The collector to register.

register_forward_hook(*hook: Callable[[T, tuple[Any, ...], Any], Any | None] | Callable[[T, tuple[Any, ...], dict[str, Any], Any], Any | None]*, ***, *prepend: bool = False*, *with_kwargs: bool = False*, *always_call: bool = False*) → RemovableHandle

Register a forward hook on the module.

The hook will be called every time after `forward()` has computed an output.

If `with_kwargs` is `False` or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the `forward`. The hook can modify the
output. It can modify the input inplace but it will not have effect on
forward since this is called after `forward()` is called. The hook
should have the following signature:

```
hook(module, args, output) -> None or modified output
```

If `with_kwargs` is `True`, the forward hook will be passed the
`kwargs` given to the forward function and be expected to return the
output possibly modified. The hook should have the following signature:

```
hook(module, args, kwargs, output) -> None or modified output
```

Parameters:

- **hook** (*Callable*) - The user defined hook to be registered.
- **prepend** (*bool*) - If `True`, the provided `hook` will be fired
before all existing `forward` hooks on this
[`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Otherwise, the provided
`hook` will be fired after all existing `forward` hooks on
this [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Note that global
`forward` hooks registered with
`register_module_forward_hook()` will fire before all hooks
registered by this method.
Default: `False`
- **with_kwargs** (*bool*) - If `True`, the `hook` will be passed the
kwargs given to the forward function.
Default: `False`
- **always_call** (*bool*) - If `True` the `hook` will be run regardless of
whether an exception is raised while calling the Module.
Default: `False`

Returns:

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type:

`torch.utils.hooks.RemovableHandle`

register_forward_pre_hook(*hook: Callable[[T, tuple[Any, ...]], Any | None] | Callable[[T, tuple[Any, ...], dict[str, Any]], tuple[Any, dict[str, Any]] | None]*, ***, *prepend: bool = False*, *with_kwargs: bool = False*) → RemovableHandle

Register a forward pre-hook on the module.

The hook will be called every time before `forward()` is invoked.

If `with_kwargs` is false or not specified, the input contains only
the positional arguments given to the module. Keyword arguments won't be
passed to the hooks and only to the `forward`. The hook can modify the
input. User can either return a tuple or a single modified value in the
hook. We will wrap the value into a tuple if a single value is returned
(unless that value is already a tuple). The hook should have the
following signature:

```
hook(module, args) -> None or modified input
```

If `with_kwargs` is true, the forward pre-hook will be passed the
kwargs given to the forward function. And if the hook modifies the
input, both the args and kwargs should be returned. The hook should have
the following signature:

```
hook(module, args, kwargs) -> None or a tuple of modified input and kwargs
```

Parameters:

- **hook** (*Callable*) - The user defined hook to be registered.
- **prepend** (*bool*) - If true, the provided `hook` will be fired before
all existing `forward_pre` hooks on this
[`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Otherwise, the provided
`hook` will be fired after all existing `forward_pre` hooks
on this [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Note that global
`forward_pre` hooks registered with
`register_module_forward_pre_hook()` will fire before all
hooks registered by this method.
Default: `False`
- **with_kwargs** (*bool*) - If true, the `hook` will be passed the kwargs
given to the forward function.
Default: `False`

Returns:

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type:

`torch.utils.hooks.RemovableHandle`

register_full_backward_hook(*hook: Callable[[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)], tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None]*, *prepend: bool = False*) → RemovableHandle

Register a backward hook on the module.

The hook will be called every time the gradients with respect to a module are computed, and its firing rules are as follows:

> 1. Ordinarily, the hook fires when the gradients are computed with respect to the module inputs.
> 2. If none of the module inputs require gradients, the hook will fire when the gradients are computed
> with respect to module outputs.
> 3. If none of the module outputs require gradients, then the hooks will not fire.

The hook should have the following signature:

```
hook(module, grad_input, grad_output) -> tuple(Tensor) or None
```

The `grad_input` and `grad_output` are tuples that contain the gradients
with respect to the inputs and outputs respectively. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the input that will be used in place of `grad_input` in
subsequent computations. `grad_input` will only correspond to the inputs given
as positional arguments and all kwarg arguments are ignored. Entries
in `grad_input` and `grad_output` will be `None` for all non-Tensor
arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

Warning

Modifying inputs or outputs inplace is not allowed when using backward hooks and
will raise an error.

Parameters:

- **hook** (*Callable*) - The user-defined hook to be registered.
- **prepend** (*bool*) - If true, the provided `hook` will be fired before
all existing `backward` hooks on this
[`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Otherwise, the provided
`hook` will be fired after all existing `backward` hooks on
this [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Note that global
`backward` hooks registered with
`register_module_full_backward_hook()` will fire before
all hooks registered by this method.

Returns:

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type:

`torch.utils.hooks.RemovableHandle`

register_full_backward_pre_hook(*hook: Callable[[[Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module), tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)], tuple[[Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor), ...] | [Tensor](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor) | None]*, *prepend: bool = False*) → RemovableHandle

Register a backward pre-hook on the module.

The hook will be called every time the gradients for the module are computed.
The hook should have the following signature:

```
hook(module, grad_output) -> tuple[Tensor, ...], Tensor or None
```

The `grad_output` is a tuple. The hook should
not modify its arguments, but it can optionally return a new gradient with
respect to the output that will be used in place of `grad_output` in
subsequent computations. Entries in `grad_output` will be `None` for
all non-Tensor arguments.

For technical reasons, when this hook is applied to a Module, its forward function will
receive a view of each Tensor passed to the Module. Similarly the caller will receive a view
of each Tensor returned by the Module's forward function.

Warning

Modifying inputs inplace is not allowed when using backward hooks and
will raise an error.

Parameters:

- **hook** (*Callable*) - The user-defined hook to be registered.
- **prepend** (*bool*) - If true, the provided `hook` will be fired before
all existing `backward_pre` hooks on this
[`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Otherwise, the provided
`hook` will be fired after all existing `backward_pre` hooks
on this [`torch.nn.Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module). Note that global
`backward_pre` hooks registered with
`register_module_full_backward_pre_hook()` will fire before
all hooks registered by this method.

Returns:

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type:

`torch.utils.hooks.RemovableHandle`

*classmethod*register_gym(*id: str*, ***, *entry_point: Callable | None = None*, *transform: [Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform) | None = None*, *info_keys: list[NestedKey] | None = None*, *backend: str | None = None*, *to_numpy: bool = False*, *reward_threshold: float | None = None*, *nondeterministic: bool = False*, *max_episode_steps: int | None = None*, *order_enforce: bool = True*, *autoreset: bool | None = None*, *disable_env_checker: bool = False*, *apply_api_compatibility: bool = False*, ***kwargs*)

Registers an environment in gym(nasium).

This method is designed with the following scopes in mind:

- Incorporate a TorchRL-first environment in a framework that uses Gym;
- Incorporate another environment (eg, DeepMind Control, Brax, Jumanji, ...)
in a framework that uses Gym.

Parameters:

**id** (*str*) - the name of the environment. Should follow the
[gym naming convention](https://www.gymlibrary.dev/content/environment_creation/#registering-envs).

Keyword Arguments:

- **entry_point** (*callable**,**optional*) -

the entry point to build the environment.
If none is passed, the parent class will be used as entry point.
Typically, this is used to register an environment that does not
necessarily inherit from the base being used:

```
>>> from torchrl.envs import DMControlEnv
>>> DMControlEnv.register_gym("DMC-cheetah-v0", env_name="cheetah", task="run")
>>> # equivalently
>>> EnvBase.register_gym("DMC-cheetah-v0", entry_point=DMControlEnv, env_name="cheetah", task="run")
```
- **transform** (*torchrl.envs.Transform*) - a transform (or list of transforms
within a `torchrl.envs.Compose` instance) to be used with the env.
This arg can be passed during a call to `make()` (see
example below).
- **info_keys** (*List**[**NestedKey**]**,**optional*) -

if provided, these keys will
be used to build the info dictionary and will be excluded from
the observation keys.
This arg can be passed during a call to `make()` (see
example below).

Warning

It may be the case that using `info_keys` makes a spec empty
because the content has been moved to the info dictionary.
Gym does not like empty `Dict` in the specs, so this empty
content should be removed with [`RemoveEmptySpecs`](torchrl.envs.transforms.RemoveEmptySpecs.html#torchrl.envs.transforms.RemoveEmptySpecs).
- **backend** (*str**,**optional*) - the backend. Can be either "gym" or "gymnasium"
or any other backend compatible with `set_gym_backend`.
- **to_numpy** (*bool**,**optional*) - if `True`, the result of calls to step and
reset will be mapped to numpy arrays. Defaults to `False`
(results are tensors).
This arg can be passed during a call to `make()` (see
example below).
- **reward_threshold** (`float`, optional) - [Gym kwarg] The reward threshold
considered to have learnt an environment.
- **nondeterministic** (*bool**,**optional*) - [Gym kwarg If the environment is nondeterministic
(even with knowledge of the initial seed and all actions). Defaults to
`False`.
- **max_episode_steps** (*int**,**optional*) - [Gym kwarg] The maximum number
of episodes steps before truncation. Used by the Time Limit wrapper.
- **order_enforce** (*bool**,**optional*) - [Gym >= 0.14] Whether the order
enforcer wrapper should be applied to ensure users run functions
in the correct order.
Defaults to `True`.
- **autoreset** (*bool**,**optional*) - [Gym >= 0.14 and <1.0.0] Whether the autoreset wrapper
should be added such that reset does not need to be called.
Defaults to `False`.
- **disable_env_checker** - [Gym >= 0.14] Whether the environment
checker should be disabled for the environment. Defaults to `False`.
- **apply_api_compatibility** - [Gym >= 0.26 and <1.0.0] If to apply the StepAPICompatibility wrapper.
Defaults to `False`.
- ****kwargs** - arbitrary keyword arguments which are passed to the environment constructor.

Note

TorchRL's environment do not have the concept of an `"info"` dictionary,
as `TensorDict` offers all the storage requirements deemed necessary
in most training settings. Still, you can use the `info_keys` argument to
have a fine grained control over what is deemed to be considered
as an observation and what should be seen as info.

Examples

```
>>> # Register the "cheetah" env from DMControl with the "run" task
>>> from torchrl.envs import DMControlEnv
>>> import torch
>>> DMControlEnv.register_gym("DMC-cheetah-v0", to_numpy=False, backend="gym", env_name="cheetah", task_name="run")
>>> import gym
>>> envgym = gym.make("DMC-cheetah-v0")
>>> envgym.seed(0)
>>> torch.manual_seed(0)
>>> envgym.reset()
({'position': tensor([-0.0855, 0.0215, -0.0881, -0.0412, -0.1101, 0.0080, 0.0254, 0.0424],
 dtype=torch.float64), 'velocity': tensor([ 1.9609e-02, -1.9776e-04, -1.6347e-03, 3.3842e-02, 2.5338e-02,
 3.3064e-02, 1.0381e-04, 7.6656e-05, 1.0204e-02],
 dtype=torch.float64)}, {})
>>> envgym.step(envgym.action_space.sample())
({'position': tensor([-0.0833, 0.0275, -0.0612, -0.0770, -0.1256, 0.0082, 0.0186, 0.0476],
 dtype=torch.float64), 'velocity': tensor([ 0.2221, 0.2256, 0.5930, 2.6937, -3.5865, -1.5479, 0.0187, -0.6825,
 0.5224], dtype=torch.float64)}, tensor([0.0018], dtype=torch.float64), tensor([False]), tensor([False]), {})
>>> # same environment with observation stacked
>>> from torchrl.envs import CatTensors
>>> envgym = gym.make("DMC-cheetah-v0", transform=CatTensors(in_keys=["position", "velocity"], out_key="observation"))
>>> envgym.reset()
({'observation': tensor([-0.1005, 0.0335, -0.0268, 0.0133, -0.0627, 0.0074, -0.0488, -0.0353,
 -0.0075, -0.0069, 0.0098, -0.0058, 0.0033, -0.0157, -0.0004, -0.0381,
 -0.0452], dtype=torch.float64)}, {})
>>> # same environment with numpy observations
>>> envgym = gym.make("DMC-cheetah-v0", transform=CatTensors(in_keys=["position", "velocity"], out_key="observation"), to_numpy=True)
>>> envgym.reset()
({'observation': array([-0.11355747, 0.04257728, 0.00408397, 0.04155852, -0.0389733 ,
 -0.01409826, -0.0978704 , -0.08808327, 0.03970837, 0.00535434,
 -0.02353762, 0.05116226, 0.02788907, 0.06848346, 0.05154399,
 0.0371798 , 0.05128025])}, {})
>>> # If gymnasium is installed, we can register the environment there too.
>>> DMControlEnv.register_gym("DMC-cheetah-v0", to_numpy=False, backend="gymnasium", env_name="cheetah", task_name="run")
>>> import gymnasium
>>> envgym = gymnasium.make("DMC-cheetah-v0")
>>> envgym.seed(0)
>>> torch.manual_seed(0)
>>> envgym.reset()
({'position': tensor([-0.0855, 0.0215, -0.0881, -0.0412, -0.1101, 0.0080, 0.0254, 0.0424],
 dtype=torch.float64), 'velocity': tensor([ 1.9609e-02, -1.9776e-04, -1.6347e-03, 3.3842e-02, 2.5338e-02,
 3.3064e-02, 1.0381e-04, 7.6656e-05, 1.0204e-02],
 dtype=torch.float64)}, {})
```

Note

This feature also works for stateless environments (eg, [`BraxEnv`](torchrl.envs.BraxEnv.html#torchrl.envs.BraxEnv)).

```
>>> import gymnasium
>>> import torch
>>> from tensordict import TensorDict
>>> from torchrl.envs import BraxEnv, SelectTransform
>>>
>>> # get action for dydactic purposes
>>> env = BraxEnv("ant", batch_size=[2])
>>> env.set_seed(0)
>>> torch.manual_seed(0)
>>> td = env.rollout(10)
>>>
>>> actions = td.get("action")
>>>
>>> # register env
>>> env.register_gym("Brax-Ant-v0", env_name="ant", batch_size=[2], info_keys=["state"])
>>> gym_env = gymnasium.make("Brax-Ant-v0")
>>> gym_env.seed(0)
>>> torch.manual_seed(0)
>>>
>>> gym_env.reset()
>>> obs = []
>>> for i in range(10):
... obs, reward, terminated, truncated, info = gym_env.step(td[..., i].get("action"))
```

register_load_state_dict_post_hook(*hook*)

Register a post-hook to be run after module's `load_state_dict()` is called.

It should have the following signature::

hook(module, incompatible_keys) -> None

The `module` argument is the current module that this hook is registered
on, and the `incompatible_keys` argument is a `NamedTuple` consisting
of attributes `missing_keys` and `unexpected_keys`. `missing_keys`
is a `list` of `str` containing the missing keys and
`unexpected_keys` is a `list` of `str` containing the unexpected keys.

The given incompatible_keys can be modified inplace if needed.

Note that the checks performed when calling `load_state_dict()` with
`strict=True` are affected by modifications the hook makes to
`missing_keys` or `unexpected_keys`, as expected. Additions to either
set of keys will result in an error being thrown when `strict=True`, and
clearing out both missing and unexpected keys will avoid an error.

Returns:

a handle that can be used to remove the added hook by calling
`handle.remove()`

Return type:

`torch.utils.hooks.RemovableHandle`

register_load_state_dict_pre_hook(*hook*)

Register a pre-hook to be run before module's `load_state_dict()` is called.

It should have the following signature::

hook(module, state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs) -> None # noqa: B950

Parameters:

**hook** (*Callable*) - Callable hook that will be invoked before
loading the state dict.

register_module(*name: str*, *module: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) | None*) → None

Alias for `add_module()`.

register_parameter(*name: str*, *param: [Parameter](https://docs.pytorch.org/docs/stable/generated/torch.nn.parameter.Parameter.html#torch.nn.parameter.Parameter) | None*) → None

Add a parameter to the module.

The parameter can be accessed as an attribute using given name.

Parameters:

- **name** (*str*) - name of the parameter. The parameter can be accessed
from this module using the given name
- **param** (*Parameter**or**None*) - parameter to be added to the module. If
`None`, then operations that run on parameters, such as `cuda`,
are ignored. If `None`, the parameter is **not** included in the
module's `state_dict`.

register_state_dict_post_hook(*hook*)

Register a post-hook for the [`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) method.

It should have the following signature::

hook(module, state_dict, prefix, local_metadata) -> None

The registered hooks can modify the `state_dict` inplace.

register_state_dict_pre_hook(*hook*)

Register a pre-hook for the [`state_dict()`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module.state_dict) method.

It should have the following signature::

hook(module, prefix, keep_vars) -> None

The registered hooks can be used to perform pre-processing before the `state_dict`
call is made.

requires_grad_(*requires_grad: bool = True*) → Self

Change if autograd should record operations on parameters in this module.

This method sets the parameters' `requires_grad` attributes
in-place.

This method is helpful for freezing part of the module for finetuning
or training parts of a model individually (e.g., GAN training).

See [Locally disabling gradient computation](https://docs.pytorch.org/docs/stable/notes/autograd.html#locally-disable-grad-doc) for a comparison between
.requires_grad_() and several similar mechanisms that may be confused with it.

Parameters:

**requires_grad** (*bool*) - whether autograd should record operations on
parameters in this module. Default: `True`.

Returns:

self

Return type:

Module

reset(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, ***, *set_state: bool | None = None*, ***kwargs*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Resets the environment.

As for step and _step, only the private method `_reset` should be overwritten by EnvBase subclasses.

Parameters:

**tensordict** (*TensorDictBase**,**optional*) - tensordict to be used to contain the resulting new observation.
In some cases, this input can also be used to pass argument to the reset function.

Keyword Arguments:

- **set_state** (*bool**,**optional*) - if `True`, the environment is reset
*deterministically* to the state contained in `tensordict`
(for stateless envs such as [`PendulumEnv`](torchrl.envs.PendulumEnv.html#torchrl.envs.PendulumEnv),
the relevant state entries - e.g. `"th"`/`"thdot"` - are
honored; for stateful envs that support it, the underlying
set-state API is used). Passing `set_state=True` to an env that
cannot honor a provided state raises `NotImplementedError`.
If `False`, any state present in `tensordict` is ignored and a
fresh (typically random) initial state is generated. The default
(`None`) preserves the historical behavior of honoring state
found in `tensordict`, but emits a `FutureWarning`: from
**v0.15** an unspecified `set_state` will be treated as
`False`. This is a keyword argument, deliberately *not* a
tensordict key, so it never stacks/pads across a rollout.
- **kwargs** (*optional*) - other arguments to be passed to the native
reset function.

Returns:

a tensordict (or the input tensordict, if any), modified in place with the resulting observations.

Note

reset should not be overwritten by [`EnvBase`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase) subclasses. The method to
modify is `_reset()`.

*property*reset_keys*: list[NestedKey]*

Returns a list of reset keys.

Reset keys are keys that indicate partial reset, in batched, multitask or multiagent
settings. They are structured as `(*prefix, "_reset")` where `prefix` is
a (possibly empty) tuple of strings pointing to a tensordict location
where a done state can be found.

Keys are sorted by depth in the data tree.

*property*reward_key

The reward key of an environment.

By default, this will be "reward".

If there is more than one reward key in the environment, this function will raise an exception.

*property*reward_keys*: list[NestedKey]*

The reward keys of an environment.

By default, there will only be one key named "reward".

Keys are sorted by depth in the data tree.

*property*reward_spec*: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*

The `reward` spec.

The `reward_spec` is always stored as a composite spec.

If the reward spec is provided as a simple spec, this will be returned.

```
>>> env.reward_spec = Unbounded(1)
>>> env.reward_spec
UnboundedContinuous(
 shape=torch.Size([1]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous)
```

If the reward spec is provided as a composite spec and contains only one leaf,
this function will return just the leaf.

```
>>> env.reward_spec = Composite({"nested": {"reward": Unbounded(1)}})
>>> env.reward_spec
UnboundedContinuous(
 shape=torch.Size([1]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous)
```

If the reward spec is provided as a composite spec and has more than one leaf,
this function will return the whole spec.

```
>>> env.reward_spec = Composite({"nested": {"reward": Unbounded(1), "another_reward": Categorical(1)}})
>>> env.reward_spec
Composite(
 nested: Composite(
 reward: UnboundedContinuous(
 shape=torch.Size([1]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous),
 another_reward: Categorical(
 shape=torch.Size([]),
 space=DiscreteBox(n=1),
 device=cpu,
 dtype=torch.int64,
 domain=discrete), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))
```

To retrieve the full spec passed, use:

```
>>> env.output_spec["full_reward_spec"]
```

This property is mutable.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> env = GymEnv("Pendulum-v1")
>>> env.reward_spec
UnboundedContinuous(
 shape=torch.Size([1]),
 space=None,
 device=cpu,
 dtype=torch.float32,
 domain=continuous)
```

*property*reward_spec_unbatched*: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*

Returns the reward spec of the env as if it had no batch dimensions.

rollout(*max_steps: int*, *policy: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)], [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)] | None = None*, *callback: Callable[[[TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase), ...], Any] | None = None*, ***, *actions: Iterable[Any] | None = None*, *auto_reset: bool = True*, *auto_cast_to_device: bool = False*, *break_when_any_done: bool | None = None*, *break_when_all_done: bool | None = None*, *return_contiguous: bool | None = False*, *tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) | None = None*, *set_truncated: bool = False*, *out=None*, *trust_policy: bool = False*, *storing_device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | None = None*, *set_state: bool | None = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Executes a rollout in the environment.

The function will return as soon as any of the contained environments
reaches any of the done states.

Parameters:

- **max_steps** (*int*) - maximum number of steps to be executed. The actual number of steps can be smaller if
the environment reaches a done state before max_steps have been executed.
- **policy** (*callable**,**optional*) - callable to be called to compute the desired action.
If no policy is provided, actions will be called using `env.rand_step()`.
The policy can be any callable that reads either a tensordict or
the entire sequence of observation entries __sorted as__ the `env.observation_spec.keys()`.
Defaults to None.
- **callback** (*Callable**[**[**TensorDict**]**,**Any**]**,**optional*) - function to be called at each iteration with the given
TensorDict. Defaults to `None`. The output of `callback` will not be collected, it is the user
responsibility to save any result within the callback call if data needs to be carried over beyond
the call to `rollout`.

Keyword Arguments:

- **actions** (*iterable**,**optional*) - an iterable of pre-computed actions to
drive the rollout instead of a `policy`. Each item is written
under the environment's (top-level) action key before stepping,
making open-loop replay a one-liner
(`env.rollout(max_steps, actions=[...])`). Mutually exclusive
with `policy`. When the iterable is sized, `max_steps` is
capped to its length. To stop early on a goal condition, combine
with [`TerminateTransform`](torchrl.envs.transforms.TerminateTransform.html#torchrl.envs.transforms.TerminateTransform) and
`break_when_any_done=True`. Defaults to `None`.
- **auto_reset** (*bool**,**optional*) - if `True`, the contained environments will be reset before starting the
rollout. If `False`, then the rollout will continue from a previous state, which requires the
`tensordict` argument to be passed with the previous rollout. Default is `True`.
- **auto_cast_to_device** (*bool**,**optional*) - if `True`, the device of the tensordict is automatically cast to the
policy device before the policy is used. Default is `False`.
- **break_when_any_done** (*bool*) -

if `True`, break when any of the contained environments reaches any of the
done states. If `False`, then the done environments are reset automatically. Default is `True`.

See also

The [Partial resets](../envs_vectorized.html#ref-partial-resets) of the documentation gives more
information about partial resets.
- **break_when_all_done** (*bool**,**optional*) -

if `True`, break if all of the contained environments reach any
of the done states. If `False`, break if at least one environment reaches any of the done states.
Default is `False`.

See also

The [Partial steps](../envs_vectorized.html#ref-partial-steps) of the documentation gives more
information about partial resets.
- **return_contiguous** (*bool*) - if False, a LazyStackedTensorDict will be returned. Default is True if
the env does not have dynamic specs, otherwise False.
- **tensordict** (*TensorDict**,**optional*) - if `auto_reset` is False, an initial
tensordict must be provided. Rollout will check if this tensordict has done flags and reset the
environment in those dimensions (if needed).
This normally should not occur if `tensordict` is the output of a reset, but can occur
if `tensordict` is the last step of a previous rollout.
A `tensordict` can also be provided when `auto_reset=True` if metadata need to be passed
to the `reset` method, such as a batch-size or a device for stateless environments.
- **set_truncated** (*bool**,**optional*) - if `True`, `"truncated"` and `"done"` keys will be set to
`True` after completion of the rollout. If no `"truncated"` is found within the
`done_spec`, an exception is raised.
Truncated keys can be set through `env.add_truncated_keys`.
Defaults to `False`.
- **trust_policy** (*bool**,**optional*) - if `True`, a non-TensorDictModule policy will be trusted to be
assumed to be compatible with the collector. This defaults to `True` for CudaGraphModules
and `False` otherwise.
- **storing_device** (*Device**,**optional*) - if provided, the tensordict will be stored on this device.
Defaults to `None`.
- **set_state** (*bool**,**optional*) - forwarded to the initial
[`reset()`](torchrl.envs.EnvBase.html#id1) (only when `auto_reset=True`).
Pass `set_state=True` to start the rollout *deterministically*
from the state contained in `tensordict`. See
[`reset()`](torchrl.envs.EnvBase.html#id1) for details. Defaults to
`None`.

Returns:

TensorDict object containing the resulting trajectory.

The data returned will be marked with a "time" dimension name for the last
dimension of the tensordict (at the `env.ndim` index).

`rollout` is quite handy to display what the data structure of the
environment looks like.

Examples

```
>>> # Using rollout without a policy
>>> from torchrl.envs.libs.gym import GymEnv
>>> from torchrl.envs.transforms import TransformedEnv, StepCounter
>>> env = TransformedEnv(GymEnv("Pendulum-v1"), StepCounter(max_steps=20))
>>> rollout = env.rollout(max_steps=1000)
>>> print(rollout)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([20, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 step_count: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 truncated: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([20]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([20, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 step_count: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 truncated: Tensor(shape=torch.Size([20, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([20]),
 device=cpu,
 is_shared=False)
>>> print(rollout.names)
['time']
>>> # with envs that contain more dimensions
>>> from torchrl.envs import SerialEnv
>>> env = SerialEnv(3, lambda: TransformedEnv(GymEnv("Pendulum-v1"), StepCounter(max_steps=20)))
>>> rollout = env.rollout(max_steps=1000)
>>> print(rollout)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 20, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 step_count: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3, 20]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([3, 20, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 step_count: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 20, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3, 20]),
 device=cpu,
 is_shared=False)
>>> print(rollout.names)
[None, 'time']
```

Using a policy (a regular [`Module`](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module) or a [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule))
is also easy:

Examples

```
>>> from torch import nn
>>> env = GymEnv("CartPole-v1", categorical_action_encoding=True)
>>> class ArgMaxModule(nn.Module):
... def forward(self, values):
... return values.argmax(-1)
>>> n_obs = env.observation_spec["observation"].shape[-1]
>>> n_act = env.action_spec.n
>>> # A deterministic policy
>>> policy = nn.Sequential(
... nn.Linear(n_obs, n_act),
... ArgMaxModule())
>>> env.rollout(max_steps=10, policy=policy)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False),
 done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10]),
 device=cpu,
 is_shared=False)
>>> # Under the hood, rollout will wrap the policy in a TensorDictModule
>>> # To speed things up we can do that ourselves
>>> from tensordict.nn import TensorDictModule
>>> policy = TensorDictModule(policy, in_keys=list(env.observation_spec.keys()), out_keys=["action"])
>>> env.rollout(max_steps=10, policy=policy)
TensorDict(
 fields={
 action: Tensor(shape=torch.Size([10]), device=cpu, dtype=torch.int64, is_shared=False),
 done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: TensorDict(
 fields={
 done: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([10, 4]), device=cpu, dtype=torch.float32, is_shared=False),
 terminated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 truncated: Tensor(shape=torch.Size([10, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([10]),
 device=cpu,
 is_shared=False)
```

In some instances, contiguous tensordict cannot be obtained because
they cannot be stacked. This can happen when the data returned at
each step may have a different shape, or when different environments
are executed together. In that case, `return_contiguous=False`
will cause the returned tensordict to be a lazy stack of tensordicts:

Examples of non-contiguous rollout:

```
>>> rollout = env.rollout(4, return_contiguous=False)
>>> print(rollout)
LazyStackedTensorDict(
 fields={
 action: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 done: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 next: LazyStackedTensorDict(
 fields={
 done: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 reward: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.float32, is_shared=False),
 step_count: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3, 4]),
 device=cpu,
 is_shared=False),
 observation: Tensor(shape=torch.Size([3, 4, 3]), device=cpu, dtype=torch.float32, is_shared=False),
 step_count: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.int64, is_shared=False),
 truncated: Tensor(shape=torch.Size([3, 4, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
 batch_size=torch.Size([3, 4]),
 device=cpu,
 is_shared=False)
 >>> print(rollout.names)
 [None, 'time']
```

Rollouts can be used in a loop to emulate data collection.
To do so, you need to pass as input the last tensordict coming from the previous rollout after calling
`step_mdp()` on it.

Examples of data collection rollouts:

```
>>> from torchrl.envs import GymEnv, step_mdp
>>> env = GymEnv("CartPole-v1")
>>> epochs = 10
>>> input_td = env.reset()
>>> for i in range(epochs):
... rollout_td = env.rollout(
... max_steps=100,
... break_when_any_done=False,
... auto_reset=False,
... tensordict=input_td,
... )
... input_td = step_mdp(
... rollout_td[..., -1],
... )
```

set_extra_state(*state: Any*) → None

Set extra state contained in the loaded state_dict.

This function is called from `load_state_dict()` to handle any extra state
found within the state_dict. Implement this function and a corresponding
`get_extra_state()` for your module if you need to store extra state within its
state_dict.

Parameters:

**state** (*dict*) - Extra state from the state_dict

set_seed(*seed: int | None = None*, *static_seed: bool = False*) → int | None

Sets the seed of the environment and returns the next seed to be used (which is the input seed if a single environment is present).

Parameters:

- **seed** (*int*) - seed to be set. The seed is set only locally in the environment. To handle the global seed,
see [`manual_seed()`](https://docs.pytorch.org/docs/stable/generated/torch.manual_seed.html#torch.manual_seed).
- **static_seed** (*bool**,**optional*) - if `True`, the seed is not incremented.
Defaults to False

Returns:

i.e. the seed that should be
used for another environment if created concomitantly to this environment.

Return type:

integer representing the "next seed"

set_spec_lock_(*mode: bool = True*) → [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)

Locks or unlocks the environment's specs.

Parameters:

**mode** (*bool*) - Whether to lock (True) or unlock (False) the specs. Defaults to True.

Returns:

The environment instance itself.

Return type:

[EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)

See also

[Locking environment specs](../envs_api.html#environment-lock).

set_submodule(*target: str*, *module: [Module](https://docs.pytorch.org/docs/stable/generated/torch.nn.Module.html#torch.nn.Module)*, *strict: bool = False*) → None

Set the submodule given by `target` if it exists, otherwise throw an error.

Note

If `strict` is set to `False` (default), the method will replace an existing submodule
or create a new submodule if the parent module exists. If `strict` is set to `True`,
the method will only attempt to replace an existing submodule and throw an error if
the submodule does not exist.

For example, let's say you have an `nn.Module` `A` that
looks like this:

```
A(
 (net_b): Module(
 (net_c): Module(
 (conv): Conv2d(3, 3, 3)
 )
 (linear): Linear(3, 3)
 )
)
```

(The diagram shows an `nn.Module` `A`. `A` has a nested
submodule `net_b`, which itself has two submodules `net_c`
and `linear`. `net_c` then has a submodule `conv`.)

To override the `Conv2d` with a new submodule `Linear`, you
could call `set_submodule("net_b.net_c.conv", nn.Linear(1, 1))`
where `strict` could be `True` or `False`

To add a new submodule `Conv2d` to the existing `net_b` module,
you would call `set_submodule("net_b.conv", nn.Conv2d(1, 1, 1))`.

In the above if you set `strict=True` and call
`set_submodule("net_b.conv", nn.Conv2d(1, 1, 1), strict=True)`, an AttributeError
will be raised because `net_b` does not have a submodule named `conv`.

Parameters:

- **target** - The fully-qualified string name of the submodule
to look for. (See above example for how to specify a
fully-qualified string.)
- **module** - The module to set the submodule to.
- **strict** - If `False`, the method will replace an existing submodule
or create a new submodule if the parent module exists. If `True`,
the method will only attempt to replace an existing submodule and throw an error
if the submodule doesn't already exist.

Raises:

- **ValueError** - If the `target` string is empty or if `module` is not an instance of `nn.Module`.
- **AttributeError** - If at any point along the path resulting from
 the `target` string the (sub)path resolves to a non-existent
 attribute name or an object that is not an instance of `nn.Module`.

*property*shape

Equivalent to `batch_size`.

share_memory() → Self

See [`torch.Tensor.share_memory_()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.share_memory_.html#torch.Tensor.share_memory_).

*property*specs*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

Returns a Composite container where all the environment are present.

This feature allows one to create an environment, retrieve all of the specs in a single data container and then
erase the environment from the workspace.

state_dict(**args*, *destination=None*, *prefix=''*, *keep_vars=False*)

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

*property*state_keys*: list[NestedKey]*

The state keys of an environment.

By default, there will only be one key named "state".

Keys are sorted by depth in the data tree.

*property*state_spec*: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*

State spec.

Must be a [`torchrl.data.Composite`](torchrl.data.Composite.html#torchrl.data.Composite) instance.
The keys listed here should be provided as input alongside actions to the environment.

In TorchRL, even though they are not properly speaking "state"
all inputs to the environment that are not actions are stored in the
`state_spec`.

Therefore, `"state_spec"` should be thought as
a generic data container for environment inputs that are not action data.

Examples

```
>>> from torchrl.envs import BraxEnv
>>> for envname in BraxEnv.available_envs:
... break
>>> env = BraxEnv(envname)
>>> env.state_spec
Composite(
 state: Composite(
 pipeline_state: Composite(
 q: UnboundedContinuous(
 shape=torch.Size([15]),
 space=None,
 device=cpu,
 dtype=torch.float32,
 domain=continuous),
 [...], device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))
```

*property*state_spec_unbatched*: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*

Returns the state spec of the env as if it had no batch dimensions.

step(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Makes a step in the environment.

Step accepts a single argument, tensordict, which usually carries an 'action' key which indicates the action
to be taken.
Step will call an out-place private method, _step, which is the method to be re-written by EnvBase subclasses.

Parameters:

**tensordict** (*TensorDictBase*) - Tensordict containing the action to be taken.
If the input tensordict contains a `"next"` entry, the values contained in it
will prevail over the newly computed values. This gives a mechanism
to override the underlying computations.

Returns:

the input tensordict, modified in place with the resulting observations, done state and reward
(+ others if needed).

step_mdp(*next_tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)

Advances the environment state by one step using the provided next_tensordict.

This method updates the environment's state by transitioning from the current
state to the next, as defined by the next_tensordict. The resulting tensordict
includes updated observations and any other relevant state information, with
keys managed according to the environment's specifications.

Internally, this method utilizes a precomputed `_StepMDP` instance to efficiently
handle the transition of state, observation, action, reward, and done keys. The
`_StepMDP` class optimizes the process by precomputing the keys to include and
exclude, reducing runtime overhead during repeated calls. The `_StepMDP` instance
is created with exclude_action=False, meaning that action keys are retained in
the root tensordict.

Parameters:

**next_tensordict** (*TensorDictBase*) - A tensordict containing the state of the
environment at the next time step. This tensordict should include keys
for observations, actions, rewards, and done flags, as defined by the
environment's specifications.

Returns:

A new tensordict representing the environment state after
advancing by one step.

Return type:

TensorDictBase

Note

The method ensures that the environment's key specifications are validated
against the provided next_tensordict, issuing warnings if discrepancies
are found.

Note

This method is designed to work efficiently with environments that have
consistent key specifications, leveraging the _StepMDP class to minimize
overhead.

Example

```
>>> from torchrl.envs import GymEnv
>>> env = GymEnv("Pendulum-1")
>>> data = env.reset()
>>> for i in range(10):
... # compute action
... env.rand_action(data)
... # Perform action
... next_data = env.step(reset_data)
... data = env.step_mdp(next_data)
```

to(*device: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int*) → [EnvBase](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)

Move and/or cast the parameters and buffers.

This can be called as

to(*device=None*, *dtype=None*, *non_blocking=False*)

to(*dtype*, *non_blocking=False*)

to(*tensor*, *non_blocking=False*)

to(*memory_format=torch.channels_last*)

Its signature is similar to [`torch.Tensor.to()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to), but only accepts
floating point or complex `dtype`s. In addition, this method will
only cast the floating point or complex parameters and buffers to `dtype`
(if given). The integral parameters and buffers will be moved
`device`, if that is given, but with dtypes unchanged. When
`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

Note

This method modifies the module in-place.

Parameters:

- **device** ([`torch.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) - the desired device of the parameters
and buffers in this module
- **dtype** ([`torch.dtype`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - the desired floating point or complex dtype of
the parameters and buffers in this module
- **tensor** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Tensor whose dtype and device are the desired
dtype and device for all parameters and buffers in this module
- **memory_format** ([`torch.memory_format`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format)) - the desired memory
format for 4D parameters and buffers in this module (keyword
only argument)

Returns:

self

Return type:

Module

Examples:

```
>>> # xdoctest: +IGNORE_WANT("non-deterministic")
>>> linear = nn.Linear(2, 2)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
 [-0.5113, -0.2325]])
>>> linear.to(torch.double)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
 [-0.5113, -0.2325]], dtype=torch.float64)
>>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
>>> gpu1 = torch.device("cuda:1")
>>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
 [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
>>> cpu = torch.device("cpu")
>>> linear.to(cpu)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
 [-0.5112, -0.2324]], dtype=torch.float16)

>>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
>>> linear.weight
Parameter containing:
tensor([[ 0.3741+0.j, 0.2382+0.j],
 [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
>>> linear(torch.ones(3, 2, dtype=torch.cdouble))
tensor([[0.6122+0.j, 0.1150+0.j],
 [0.6122+0.j, 0.1150+0.j],
 [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
```

to_empty(***, *device: str | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | int | None*, *recurse: bool = True*) → Self

Move the parameters and buffers to the specified device without copying storage.

Parameters:

- **device** ([`torch.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) - The desired device of the parameters
and buffers in this module.
- **recurse** (*bool*) - Whether parameters and buffers of submodules should
be recursively moved to the specified device.

Returns:

self

Return type:

Module

train(*mode: bool = True*) → Self

Set the module in training mode.

This has an effect only on certain modules. See the documentation of
particular modules for details of their behaviors in training/evaluation
mode, i.e., whether they are affected, e.g. `Dropout`, `BatchNorm`,
etc.

Parameters:

**mode** (*bool*) - whether to set training mode (`True`) or evaluation
mode (`False`). Default: `True`.

Returns:

self

Return type:

Module

type(*dst_type: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | str*) → Self

Casts all parameters and buffers to `dst_type`.

Note

This method modifies the module in-place.

Parameters:

**dst_type** (*type**or**string*) - the desired type

Returns:

self

Return type:

Module

xpu(*device: int | [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*) → Self

Move all model parameters and buffers to the XPU.

This also makes associated parameters and buffers different objects. So
it should be called before constructing optimizer if the module will
live on XPU while being optimized.

Note

This method modifies the module in-place.

Parameters:

**device** (*int**,**optional*) - if specified, all parameters will be
copied to that device

Returns:

self

Return type:

Module

zero_grad(*set_to_none: bool = True*) → None

Reset gradients of all model parameters.

See similar function under [`torch.optim.Optimizer`](https://docs.pytorch.org/docs/stable/optim.html#torch.optim.Optimizer) for more context.

Parameters:

**set_to_none** (*bool*) - instead of setting to zero, set the grads to None.
See [`torch.optim.Optimizer.zero_grad()`](https://docs.pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html#torch.optim.Optimizer.zero_grad) for details.