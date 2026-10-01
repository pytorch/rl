# CatFrames

*class*torchrl.envs.transforms.CatFrames(*N: int*, *dim: int*, *in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *padding='same'*, *padding_value=0*, *as_inverse=False*, *reset_key: NestedKey | None = None*, *done_key: NestedKey | None = None*, *future: bool = False*, *mask_key: NestedKey | None = None*)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#CatFrames)

Concatenates successive observation frames into a single tensor.

This transform is useful for creating a sense of movement or velocity in the observed features.
It can also be used with models that require access to past observations such as transformers and the like.
It was first proposed in "Playing Atari with Deep Reinforcement Learning" ([https://arxiv.org/pdf/1312.5602.pdf](https://arxiv.org/pdf/1312.5602.pdf)).

When used within a transformed environment,
`CatFrames` is a stateful class, and it can be reset to its native state by
calling the `reset` method. This method accepts tensordicts with a
`"_reset"` entry that indicates which buffer to reset.

Parameters:

- **N** (*int*) - number of observation to concatenate.
- **dim** (*int*) - dimension along which concatenate the
observations. Should be negative, to ensure that it is compatible
with environments of different batch_size.
- **in_keys** (*sequence**of**NestedKey**,**optional*) - keys pointing to the frames that have
to be concatenated. Defaults to ["pixels"].
- **out_keys** (*sequence**of**NestedKey**,**optional*) - keys pointing to where the output
has to be written. Defaults to the value of in_keys.
- **padding** (*str**,**optional*) - the padding method. One of `"same"` or `"constant"`.
Defaults to `"same"`, ie. the first value is used for padding.
- **padding_value** (`float`, optional) - the value to use for padding if `padding="constant"`.
Defaults to 0.
- **as_inverse** (*bool**,**optional*) - if `True`, the transform is applied as an inverse transform. Defaults to `False`.
- **reset_key** (*NestedKey**,**optional*) - the reset key to be used as partial
reset indicator. Must be unique. If not provided, defaults to the
only reset key of the parent environment (if it has only one)
and raises an exception otherwise.
- **done_key** (*NestedKey**,**optional*) - the done key to be used as partial
done indicator. Must be unique. If not provided, defaults to `"done"`.
- **future** (*bool**,**optional*) -

if `True`, each step's window gathers the
`N` *upcoming* frames `[t, t + 1, ..., t + N - 1]` instead of
the `N` most recent ones `[t - N + 1, ..., t]`. With
`padding="same"` the slots that run past the end of the
trajectory repeat the last in-trajectory frame. Forward-looking
windows require the full trajectory, so this mode is only
available offline (replay buffer / data pipelines): attaching the
transform to an environment raises a `RuntimeError` on the step
path. Defaults to `False`.

New in version 0.14.
- **mask_key** (*NestedKey**,**optional*) -

if provided, the offline (forward /
unfolding) path also writes a boolean mask of shape
`[*batch, time, N]` flagging, for each window, the slots that
were fabricated by padding (`True` = padded slot, either out of
the trajectory or out of the sampled window). This is the
convention of the `action_is_pad` entry of chunked-action
datasets. The mask is not available on the online (env step)
path. Defaults to `None` (no mask is written).

New in version 0.14.

Examples

```
>>> from torchrl.envs.libs.gym import GymEnv
>>> env = TransformedEnv(GymEnv('Pendulum-v1'),
... Compose(
... UnsqueezeTransform(-1, in_keys=["observation"]),
... CatFrames(N=4, dim=-1, in_keys=["observation"]),
... )
... )
>>> print(env.rollout(3))
```

The `CatFrames` transform can also be used offline to reproduce the
effect of the online frame concatenation at a different scale (or for the
purpose of limiting the memory consumption). The following example
gives the complete picture, together with the usage of a [`torchrl.data.ReplayBuffer`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer):

Examples

```
>>> from torchrl.modules import RandomPolicy >>> >>> >>> from torchrl.envs import UnsqueezeTransform, CatFrames
>>> from torchrl.collectors import Collector
>>> # Create a transformed environment with CatFrames: notice the usage of UnsqueezeTransform to create an extra dimension
>>> env = TransformedEnv(
... GymEnv("CartPole-v1", from_pixels=True),
... Compose(
... ToTensorImage(in_keys=["pixels"], out_keys=["pixels_trsf"]),
... Resize(in_keys=["pixels_trsf"], w=64, h=64),
... GrayScale(in_keys=["pixels_trsf"]),
... UnsqueezeTransform(-4, in_keys=["pixels_trsf"]),
... CatFrames(dim=-4, N=4, in_keys=["pixels_trsf"]),
... )
... )
>>> # we design a collector
>>> collector = Collector(
... env,
... RandomPolicy(env.action_spec),
... frames_per_batch=10,
... total_frames=1000,
... )
>>> for data in collector:
... print(data)
... break
>>> # now let's create a transform for the replay buffer. We don't need to unsqueeze the data here.
>>> # however, we need to point to both the pixel entry at the root and at the next levels:
>>> t = Compose(
... ToTensorImage(in_keys=["pixels", ("next", "pixels")], out_keys=["pixels_trsf", ("next", "pixels_trsf")]),
... Resize(in_keys=["pixels_trsf", ("next", "pixels_trsf")], w=64, h=64),
... GrayScale(in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
... CatFrames(dim=-4, N=4, in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
... )
>>> from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
>>> rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(1000), transform=t, batch_size=16)
>>> data_exclude = data.exclude("pixels_trsf", ("next", "pixels_trsf"))
>>> rb.add(data_exclude)
>>> s = rb.sample(1) # the buffer has only one element
>>> # let's check that our sample is the same as the batch collected during inference
>>> assert (data.exclude("collector")==s.squeeze(0).exclude("index", "collector")).all()
```

Note

`CatFrames` currently only supports `"done"`
signal at the root. Nested `done`,
such as those found in MARL settings, are currently not supported.
If this feature is needed, please raise an issue on TorchRL repo.

Note

Storing stacks of frames in the replay buffer can significantly increase memory consumption (by N times).
To mitigate this, you can store trajectories directly in the replay buffer and apply `CatFrames` at sampling time.
This approach involves sampling slices of the stored trajectories and then applying the frame stacking transform.
For convenience, `CatFrames` provides a `make_rb_transform_and_sampler()` method that creates:

- A modified version of the transform suitable for use in replay buffers
- A corresponding `SliceSampler` to use with the buffer

See also

The offline (contiguous trajectory slice) windowing performed
by this transform is also available as a pure functional,
[`torchrl.envs.transforms.functional.cat_frames()`](torchrl.envs.transforms.functional.cat_frames.html#torchrl.envs.transforms.functional.cat_frames), which operates
directly on a plain tensor.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#CatFrames.forward)

Reads the input tensordict, and for the selected keys, applies the transform.

By default, this method:

- calls directly `_apply_transform()`.
- does not call `_step()` or `_call()`.

This method is not called within env.step at any point. However, is is called within
[`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample).

Note

`forward` also works with regular keyword arguments using [`dispatch`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch) to cast the args
names to the keys.

Examples

```
>>> class TransformThatMeasuresBytes(Transform):
... '''Measures the number of bytes in the tensordict, and writes it under `"bytes"`.'''
... def __init__(self):
... super().__init__(in_keys=[], out_keys=["bytes"])
...
... def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
... bytes_in_td = tensordict.bytes()
... tensordict["bytes"] = bytes
... return tensordict
>>> t = TransformThatMeasuresBytes()
>>> env = env.append_transform(t) # works within envs
>>> t(TensorDict(a=0)) # Works offline too.
```

make_rb_transform_and_sampler(*batch_size: int*, ***sampler_kwargs*) → tuple[[Transform](torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform), [torchrl.data.replay_buffers.SliceSampler](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler)][[source]](../../_modules/torchrl/envs/transforms/_observation.html#CatFrames.make_rb_transform_and_sampler)

Creates a transform and sampler to be used with a replay buffer when storing frame-stacked data.

This method helps reduce redundancy in stored data by avoiding the need to
store the entire stack of frames in the buffer. Instead, it creates a
transform that stacks frames on-the-fly during sampling, and a sampler that
ensures the correct sequence length is maintained.

Parameters:

- **batch_size** (*int*) - The batch size to use for the sampler.
- ****sampler_kwargs** - Additional keyword arguments to pass to the
[`SliceSampler`](torchrl.data.replay_buffers.SliceSampler.html#torchrl.data.replay_buffers.SliceSampler) constructor.

Returns:

- transform (Transform): A transform that stacks frames on-the-fly during sampling.
- sampler (SliceSampler): A sampler that ensures the correct sequence length is maintained.

Return type:

A tuple containing

Example

```
>>> env = TransformedEnv(...)
>>> catframes = CatFrames(N=4, ...)
>>> transform, sampler = catframes.make_rb_transform_and_sampler(batch_size=32)
>>> rb = ReplayBuffer(..., sampler=sampler, transform=transform)
```

Note

When working with images, it's recommended to use distinct `in_keys` and `out_keys` in the preceding
`ToTensorImage` transform. This ensures that the tensors stored in the buffer are separate
from their processed counterparts, which we don't want to store.
For non-image data, consider inserting a `RenameTransform` before `CatFrames` to create
a copy of the data that will be stored in the buffer.

Note

When adding the transform to the replay buffer, one should pay attention to also pass the transforms
that precede CatFrames, such as `ToTensorImage` or `UnsqueezeTransform`
in such a way that the `CatFrames` transforms sees data formatted as it was during data
collection.

Note

For a more complete example, refer to torchrl's github repo examples folder:
[pytorch/rl](https://github.com/pytorch/rl/tree/main/examples/replay-buffers/catframes-in-buffer.py)

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#CatFrames.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform