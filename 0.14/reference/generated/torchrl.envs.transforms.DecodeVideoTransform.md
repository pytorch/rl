# DecodeVideoTransform

*class*torchrl.envs.transforms.DecodeVideoTransform(***, *in_keys: Sequence[NestedKey]*, *out_keys: Sequence[NestedKey] | None = None*, *device: Any = None*, *dtype: Any = None*)[[source]](../../_modules/torchrl/envs/transforms/_video.html#DecodeVideoTransform)

Decodes [`VideoClipRef`](torchrl.data.VideoClipRef.html#torchrl.data.VideoClipRef) leaves to dense frame tensors.

This is a forward / sample-path transform: it reads the lazy video references
found at `in_keys` and writes the decoded `uint8` frames at `out_keys`. It
is meant to be appended to a [`ReplayBuffer`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer) so that
indexing the buffer stays cheap (no materialized frames) while `rb.sample()`
returns decoded frames aligned to the sampled steps. It is a read-side codec, so
no inverse is defined.

Decoding is delegated to `VideoClipRef.decode()`, which groups the sampled
references by source file and uses ranged reads for contiguous indices. This is
what makes it compose with `SliceSampler`: a contiguous
window of sampled steps maps to consecutive frame indices and decodes as a
single ranged read per source.

Keyword Arguments:

- **in_keys** (*sequence**of**NestedKey*) - the keys holding the
[`VideoClipRef`](torchrl.data.VideoClipRef.html#torchrl.data.VideoClipRef) leaves to decode.
- **out_keys** (*sequence**of**NestedKey**,**optional*) - destination keys for the decoded
frames. Defaults to `in_keys` (in-place replacement).
- **device** ([*torch.device*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)*or**str**,**optional*) - device for the decoded frames. A
CUDA device enables GPU (NVDEC) decoding. Defaults to `None` (uses the
reference's `out_device`, else CPU).
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - dtype for the decoded frames. Defaults to
`None` (uses the reference's `out_dtype`, else `uint8`).

Note

This transform requires torchcodec. The lightweight
[`VideoClipRef`](torchrl.data.VideoClipRef.html#torchrl.data.VideoClipRef) leaves stored in the buffer are
picklable and hold no open decoder; decoders are opened lazily and cached
per worker process.

Examples

```
>>> import tempfile, os, torch
>>> from torchcodec.encoders import VideoEncoder
>>> from tensordict import TensorDict
>>> from torchrl.data import (
... LazyTensorStorage, ReplayBuffer, SliceSampler, VideoClipRef)
>>> from torchrl.envs.transforms import DecodeVideoTransform
>>> frames = torch.arange(20, dtype=torch.uint8).reshape(20, 1, 1, 1)
>>> frames = frames.expand(20, 3, 8, 8).contiguous()
>>> path = os.path.join(tempfile.mkdtemp(), "clip.mp4")
>>> VideoEncoder(frames=frames, frame_rate=10).to_file(path)
>>> ref = VideoClipRef.from_file(path) # 20 frames, lazy
>>> data = TensorDict(
... {"frame": ref, "episode": torch.zeros(20, dtype=torch.long)},
... batch_size=[20],
... )
>>> rb = ReplayBuffer(
... storage=LazyTensorStorage(20),
... sampler=SliceSampler(slice_len=4, traj_key="episode"),
... batch_size=8,
... transform=DecodeVideoTransform(in_keys=["frame"], out_keys=["pixels"]),
... )
>>> _ = rb.extend(data)
>>> sample = rb.sample()
>>> sample["pixels"].shape # decoded on sample
torch.Size([8, 3, 8, 8])
```

See also

[`VideoClipRef`](torchrl.data.VideoClipRef.html#torchrl.data.VideoClipRef).