# NextObservationDelta

*class*torchrl.envs.transforms.NextObservationDelta(*in_keys: Sequence[NestedKey] | None = None*, ***, *delta_dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) = torch.float16*, *restore_dtype: [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype) | Literal['root'] = 'root'*, *drop_delta: bool = True*, *excluded_dtypes: tuple[[dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype), ...] = (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64, torch.bool)*)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#NextObservationDelta)

Stores `("next", obs)` as a low-precision delta in a sibling key.

A single transform handles both sides of the compression:

- **Env side** (`_step` + `_post_step_mdp_hooks`): for each
in-key `k`, write `(next_obs - obs).to(delta_dtype)` under
the sibling key `("next", "delta", k)`, then drop the full
`("next", k)` from the post-step tensordict that the collector
stacks. The full slot survives only long enough for
`step_mdp()` to promote it to root, so the
policy sees a full-precision observation on the next step.
- **RB side** (`forward`): on
[`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample), reconstruct
`("next", k) = data[k] + data[("next", "delta", k)]` and
(optionally) drop the delta key. Unlike
[`NextStateReconstructor`](torchrl.envs.transforms.NextStateReconstructor.html#torchrl.envs.transforms.NextStateReconstructor), the
delta encodes the actual transition, so trajectory-boundary
transitions reconstruct exactly within the round-trip precision
of `delta_dtype` rather than falling back to `NaN`.

Use the **same instance** (or two instances with matching `in_keys`)
on the env and on the replay buffer; the env-side and RB-side methods
are dispatched automatically.

Parameters:

**in_keys** (*sequence**of**NestedKey**,**optional*) - observation keys to
compress. Defaults to `None`, in which case the transform
lazily walks `parent.observation_spec` and picks every
floating-point leaf whose dtype is not in `excluded_dtypes`.
When the transform is used on a replay buffer (no env parent),
`in_keys` must be passed explicitly.

Keyword Arguments:

- **delta_dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - dtype in which the delta is
stored. Must be a floating dtype. Defaults to `torch.float16`.
- **restore_dtype** (torch.dtype or `"root"`, optional) - dtype of the
reconstructed `("next", k)` on the RB side. `"root"`
(default) matches the dtype of the corresponding root key in
the sampled batch.
- **drop_delta** (*bool**,**optional*) - if `True` (default), the
`("next", "delta", k)` entry is removed from the sampled
tensordict after RB-side reconstruction so downstream consumers
see the same key layout as an uncompressed pipeline.
- **excluded_dtypes** (*tuple**of*[*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - dtypes to skip
when auto-inferring `in_keys`. Defaults to the integer +
bool family.

Warning

The compression is **lossy**: round-tripping through `delta_dtype`
loses precision, particularly for unnormalized observations whose
magnitudes exceed the dtype range or fall below its smallest
representable step.

Warning

The transform must live **outside** any batched env
(`TransformedEnv(ParallelEnv(N, factory), NextObservationDelta())`).
Building a [`SerialEnv`](torchrl.envs.SerialEnv.html#torchrl.envs.SerialEnv) /
[`ParallelEnv`](torchrl.envs.ParallelEnv.html#torchrl.envs.ParallelEnv) whose worker contains a
`NextObservationDelta` raises at construction time.

Example

```
>>> import torch
>>> from torchrl.envs import GymEnv, TransformedEnv
>>> from torchrl.envs.transforms import NextObservationDelta
>>> env = TransformedEnv(GymEnv("Pendulum-v1"), NextObservationDelta())
>>> td_root = env.reset()
>>> _ = td_root.set("action", env.action_spec.rand())
>>> td, td_ = env.step_and_maybe_reset(td_root)
>>> td["next", "delta", "observation"].dtype
torch.float16
>>> ("next", "observation") in td.keys(True, True)
False
>>> td_["observation"].dtype
torch.float32
```

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#NextObservationDelta.forward)

Reconstruct `("next", k)` from the stored delta at sample time.

Invoked by [`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample) when this
transform is appended to a replay buffer. Reads `data[k]` (root
observation at step `i`) and `data[("next", "delta", k)]` (the
casted delta produced on the env side), writes
`data[("next", k)] = (data[k] + delta).to(restore_dtype)`, and
(when `drop_delta=True`, the default) removes the delta key.
Keys for which either side is missing are silently skipped.

transform_fake_tensordict(*fake_tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#NextObservationDelta.transform_fake_tensordict)

Adjust the env's `fake_tensordict` after it is built from specs.

[`fake_tensordict()`](torchrl.envs.EnvBase.html#torchrl.envs.EnvBase.fake_tensordict) constructs a zero-filled
tensordict from the env's specs, which is used by data collectors to
pre-allocate the rollout storage. The TorchRL spec system shares the
observation spec between the root and `("next", ...)` leaves, so
transforms that want the runtime `("next", k)` dtype to differ from
the root `k` dtype need a way to fix up the fake tensordict here.

The default is a no-op. Override only when the runtime tensordict your
transform produces does not match what the spec-derived fake
tensordict would imply.