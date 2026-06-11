# Vision-Language-Action (VLA)

Vision-Language-Action (VLA) models map one or more camera images,
optional proprioceptive state, and a natural-language instruction to robot
actions - usually emitted as a short *action chunk* of future steps. TorchRL
treats a VLA as an ordinary TensorDict-first policy: a [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)
fed by composable transforms, trained by a [`LossModule`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule),
and rolled out by the standard collectors. This page documents the data
schema, transforms, policies and objectives that make robot VLA workflows
TensorDict-native.

Note

The VLA stack never hard-depends on the robot-learning ecosystem. Packages
such as `transformers`, `lerobot` or simulator backends are optional and
imported lazily; `import torchrl` stays lightweight.

## Canonical TensorDict schema

VLA components agree on a single `NestedKey` layout so
that datasets, transforms, policies and losses interoperate without lossy
conversion. The layout mirrors [`OpenXExperienceReplay`](generated/torchrl.data.datasets.OpenXExperienceReplay.html#torchrl.data.datasets.OpenXExperienceReplay)
and the LeRobot dataset format:

```
TensorDict(
 observation: TensorDict(
 image: {<camera>: uint8/float [*B, T, C, H, W]}, # or a single tensor
 state: float [*B, T, state_dim], # proprioception
 language_instruction: NonTensorData | Text, # raw or tokenized
 ),
 action: float [*B, T, action_dim], # raw, per-step
 action_chunk: float [*B, T, chunk, action_dim], # built for training
 action_is_pad: bool [*B, T, chunk], # chunk validity mask
 action_tokens: long [*B, T, chunk, action_dim], # tokenized actions
 next: TensorDict(...), # TED layout
)
```

The default keys are exported from `torchrl.data.vla` (`IMAGE_KEY`,
`STATE_KEY`, `INSTRUCTION_KEY`, `ACTION_KEY`, `ACTION_CHUNK_KEY`,
`ACTION_IS_PAD_KEY`, `ACTION_TOKENS_KEY`). Every component also lets you
override its keys, so these are merely the shared defaults.

## Data and metadata

| [`RobotDatasetMetadata`](generated/torchrl.data.vla.RobotDatasetMetadata.html#torchrl.data.vla.RobotDatasetMetadata)(dataset_id[, ...]) | |
| --- | --- |
| [`validate_vla_tensordict`](generated/torchrl.data.vla.validate_vla_tensordict.html#torchrl.data.vla.validate_vla_tensordict)(tensordict, *[, ...]) | Validate that a tensordict follows the canonical VLA schema. |

Robot VLA trajectories can be loaded into the canonical schema from
[`OpenXExperienceReplay`](generated/torchrl.data.datasets.OpenXExperienceReplay.html#torchrl.data.datasets.OpenXExperienceReplay) (Open X-Embodiment) and
[`LeRobotExperienceReplay`](generated/torchrl.data.datasets.LeRobotExperienceReplay.html#torchrl.data.datasets.LeRobotExperienceReplay) (the LeRobot format),
both of which expose trajectory-aware slice sampling.

## Transforms

VLA-specific transforms are standard [`Transform`](generated/torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)
subclasses, so they compose with [`Compose`](generated/torchrl.envs.transforms.Compose.html#torchrl.envs.transforms.Compose),
replay buffers and transformed environments. They are documented in full on the
[transforms reference page](envs_transforms.html#transforms).

- [`ActionChunkTransform`](generated/torchrl.envs.transforms.ActionChunkTransform.html#torchrl.envs.transforms.ActionChunkTransform) - build fixed-length
action chunks (`[*B, T, H, action_dim]`) and a padding mask from a sampled
trajectory window, the standard training target for chunked VLA policies.
- [`ActionScaling`](generated/torchrl.envs.transforms.ActionScaling.html#torchrl.envs.transforms.ActionScaling) - affine action
normalization; built with the
[`from_metadata()`](generated/torchrl.envs.transforms.ActionScaling.html#torchrl.envs.transforms.ActionScaling.from_metadata) /
[`from_stats()`](generated/torchrl.envs.transforms.ActionScaling.html#torchrl.envs.transforms.ActionScaling.from_stats) constructors it
normalizes expert actions on the replay-buffer sample path (pass
`in_keys_inv=[]` for a buffer that raw data is written to through
`extend`, which applies the inverse) and denormalizes a policy's predicted
actions on the env action-input path.
- [`ActionTokenizerTransform`](generated/torchrl.envs.transforms.ActionTokenizerTransform.html#torchrl.envs.transforms.ActionTokenizerTransform) - encode
continuous actions into discrete tokens (wrapping an action tokenizer) for
autoregressive token VLAs.

## Action representations

Action tokenizers map continuous actions to discrete token ids and back, so
that autoregressive (RT-2 / OpenVLA-style) VLA policies can emit actions through
a language-model head.

| [`ActionTokenizerBase`](generated/torchrl.data.vla.ActionTokenizerBase.html#torchrl.data.vla.ActionTokenizerBase)(*args, **kwargs) | Base class for action tokenizers. |
| --- | --- |
| [`UniformActionTokenizer`](generated/torchrl.data.vla.UniformActionTokenizer.html#torchrl.data.vla.UniformActionTokenizer)(num_bins, *, low, high) | Per-dimension uniform-bin action tokenizer (RT-2 / OpenVLA style). |