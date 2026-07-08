# Vision-Language-Action (VLA)

Vision-Language-Action (VLA) models map one or more camera images,
optional proprioceptive state, and a natural-language instruction to robot
actions - usually emitted as a short *action chunk* of future steps. TorchRL
treats a VLA as an ordinary TensorDict-first policy: a [`TensorDictModule`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModule.html#tensordict.nn.TensorDictModule)
fed by composable transforms, trained by a [`LossModule`](generated/torchrl.objectives.LossModule.html#torchrl.objectives.LossModule),
and rolled out by the standard collectors. This page documents the data
schema, transforms, policies and objectives that make robot VLA workflows
TensorDict-native. See the [VLA tutorial](../tutorials/vla.html#vla-tuto) for an end-to-end
example (data, chunking, behavior cloning, chunked inference and RL fine-tuning).

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
 ),
 language_instruction: NonTensorData | Text, # raw or tokenized (per-traj)
 action: float [*B, T, action_dim], # raw, per-step
 vla_action: VLAAction(
 chunk: float [*B, T, chunk, action_dim], # built for training
 tokens: long [*B, T, chunk, action_dim], # tokenized actions
 log_probs: float [*B, T] or [*B, T, chunk, action_dim],
 logits: float [*B, T, chunk, action_dim, vocab],
 mask: bool [*B, T, chunk, action_dim, vocab],
 ),
 action_is_pad: bool [*B, T, chunk], # chunk validity mask
 next: TensorDict(...), # TED layout
)
```

Like [`OpenXExperienceReplay`](generated/torchrl.data.datasets.OpenXExperienceReplay.html#torchrl.data.datasets.OpenXExperienceReplay), the image and state
live under `observation` while the (per-trajectory) language instruction and
the action live at the tensordict root.

The default keys are exported from `torchrl.data.vla` (`OBSERVATION_KEY`,
`IMAGE_KEY`, `STATE_KEY`, `INSTRUCTION_KEY`, `ACTION_KEY`,
`ACTION_CHUNK_KEY`, `ACTION_IS_PAD_KEY`, `ACTION_TOKENS_KEY`). Every
component also lets you override its keys, so these are merely the shared
defaults.

## Data and metadata

| [`RobotDatasetMetadata`](generated/torchrl.data.vla.RobotDatasetMetadata.html#torchrl.data.vla.RobotDatasetMetadata)(dataset_id[, ...]) | |
| --- | --- |
| [`VLAAction`](generated/torchrl.data.vla.VLAAction.html#torchrl.data.vla.VLAAction)([chunk, tokens, raw_tokens, ...]) | |
| [`VLAImages`](generated/torchrl.data.vla.VLAImages.html#torchrl.data.vla.VLAImages)([image, wrist_image, extra, ...]) | |
| [`VLAObservation`](generated/torchrl.data.vla.VLAObservation.html#torchrl.data.vla.VLAObservation)([images, state, instruction, ...]) | |
| [`validate_vla_tensordict`](generated/torchrl.data.vla.validate_vla_tensordict.html#torchrl.data.vla.validate_vla_tensordict)(tensordict, *[, ...]) | Validate that a tensordict follows the canonical VLA schema. |

Robot VLA trajectories can be loaded into the canonical schema from
[`OpenXExperienceReplay`](generated/torchrl.data.datasets.OpenXExperienceReplay.html#torchrl.data.datasets.OpenXExperienceReplay) (Open X-Embodiment) and
[`LeRobotExperienceReplay`](generated/torchrl.data.datasets.LeRobotExperienceReplay.html#torchrl.data.datasets.LeRobotExperienceReplay) (the LeRobot format),
both of which expose trajectory-aware slice sampling.

## Transforms

The VLA data path is built from general [`Transform`](generated/torchrl.envs.transforms.Transform.html#torchrl.envs.transforms.Transform)
subclasses - none of them are VLA-specific (they apply to any action-based
pipeline) and they live alongside the other transforms, documented in full on the
[transforms reference page](envs_transforms.html#transforms). Here is how they combine for VLA:

- [`ActionChunkTransform`](generated/torchrl.envs.transforms.ActionChunkTransform.html#torchrl.envs.transforms.ActionChunkTransform) - build fixed-length
action chunks (`[*B, T, H, action_dim]`) and a padding mask from a sampled
trajectory window, the standard training target for chunked VLA policies.
Internally a thin recipe over a forward-looking
[`CatFrames`](generated/torchrl.envs.transforms.CatFrames.html#torchrl.envs.transforms.CatFrames); when the window carries its
done state, chunks stop (pad and mask) at the trajectory boundaries.
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
- [`SuccessReward`](generated/torchrl.envs.transforms.SuccessReward.html#torchrl.envs.transforms.SuccessReward) - a sparse 0/1 success
reward for RL fine-tuning.

## Action representations

Action tokenizers map continuous actions to discrete token ids and back, so
that autoregressive (RT-2 / OpenVLA-style) VLA policies can emit actions through
a language-model head.

| [`ActionTokenizerBase`](generated/torchrl.data.vla.ActionTokenizerBase.html#torchrl.data.vla.ActionTokenizerBase)(*args, **kwargs) | Base class for action tokenizers. |
| --- | --- |
| [`UniformActionTokenizer`](generated/torchrl.data.vla.UniformActionTokenizer.html#torchrl.data.vla.UniformActionTokenizer)(num_bins, *, low, high) | Per-dimension uniform-bin action tokenizer (RT-2 / OpenVLA style). |
| [`VocabTailActionTokenizer`](generated/torchrl.data.vla.VocabTailActionTokenizer.html#torchrl.data.vla.VocabTailActionTokenizer)([num_bins, ...]) | OpenVLA-style vocab-tail action tokenizer. |

## Image preprocessing

The reusable [`OpenVLAImagePreprocessor`](generated/torchrl.data.vla.OpenVLAImagePreprocessor.html#torchrl.data.vla.OpenVLAImagePreprocessor) implements the
OpenVLA-style image preprocessing order used by OpenVLA-OFT policies: square
resize, JPEG quality-95 round trip, optional 0.9-area center crop and resize
back. Its default backend keeps images as tensors and uses `torchvision` JPEG
codecs, while `backend="pil"` remains available as a reference path.

| [`OpenVLAImagePreprocessor`](generated/torchrl.data.vla.OpenVLAImagePreprocessor.html#torchrl.data.vla.OpenVLAImagePreprocessor)(*[, size, ...]) | OpenVLA-style image resize, JPEG round-trip and optional center crop. |
| --- | --- |

## Policy and environment contract

TorchRL keeps the environment boundary intentionally simple: base environments
do **not** need to know about a model-specific VLA wrapper or its structured
TensorClasses. The default env/policy contract is still the usual flat
TensorDict contract:

- on `reset` / `step` the environment writes observations under the
canonical keys, usually `("observation", "image")`, optional
`("observation", "state")` and `"language_instruction"`;
- on `step` the environment consumes one continuous action under
`"action"` with shape `[*B, action_dim]`;
- datasets and replay buffers may additionally carry training targets such as
`("vla_action", "chunk")`, `("vla_action", "tokens")` and
`"action_is_pad"`.

The VLA TensorClasses are structured policy/data containers, not a required env
API:

- [`VLAImages`](generated/torchrl.data.vla.VLAImages.html#torchrl.data.vla.VLAImages) groups primary, wrist and extra camera
tensors.
- [`VLAObservation`](generated/torchrl.data.vla.VLAObservation.html#torchrl.data.vla.VLAObservation) lets a data pipeline store a
structured or already-preprocessed observation under `"vla_observation"`
when a wrapper is built with `input_mode="preprocessed"`.
- [`VLAAction`](generated/torchrl.data.vla.VLAAction.html#torchrl.data.vla.VLAAction) stores policy outputs under
`"vla_action"`. Its fields are also regular nested TensorDict keys such as
`("vla_action", "chunk")` and can be discovered by collectors, transforms
and losses.

In other words, a wrapper may read either canonical env keys
(`input_mode="canonical"`) or a preprocessed
[`VLAObservation`](generated/torchrl.data.vla.VLAObservation.html#torchrl.data.vla.VLAObservation)
(`input_mode="preprocessed"`), and it exposes structured output keys by
default:

Here `H` is `chunk_size` and `*B` is the TensorDict batch shape.

| `output_mode` | Default keys | dtype | shape | TensorClass field |
| --- | --- | --- | --- | --- |
| `"chunk"` | `("vla_action", "chunk")` | floating point | `[*B, H, action_dim]` | `"vla_action".chunk` |
| `"tokens"` | `("vla_action", "tokens")`, optional `("vla_action", "log_probs")` / `("vla_action", "logits")` / `("vla_action", "mask")` | `long` for tokens, floating point for log-probs/logits, `bool` for mask | tokens: `[*B, H, action_dim]`; log-probs: `[*B]` in `log_probs_mode="sequence"` or `[*B, H, action_dim]` in `log_probs_mode="token"`; logits/mask: `[*B, H, action_dim, vocab_size]` | `"vla_action".tokens`, `.log_probs`, `.logits`, `.mask` |
| `"both"` | `("vla_action", "chunk")` and `("vla_action", "tokens")` plus optional token fields | floating point for chunk; token-field dtypes as above | chunk: `[*B, H, action_dim]`; token-field shapes as above | Both `"vla_action".chunk` and `"vla_action".tokens` |

Chunked policies predict `[*B, H, action_dim]` while a standard env consumes
`[*B, action_dim]`. The bridge is explicit. Use
[`MultiStepActorWrapper`](generated/torchrl.modules.tensordict_module.MultiStepActorWrapper.html#torchrl.modules.tensordict_module.MultiStepActorWrapper) when the env
should keep its one-step MDP: tell the wrapper which policy key contains the
chunk, let it cache the chunk and write the one-step env key (`"action"` by
default), and re-query the expensive VLA only when the cache expires or an
`"is_init"` reset flag is observed. The wrapper auto-discovers the default
VLA key `("vla_action", "chunk")` from the policy's `out_keys` and keeps
that key as the cache rather than copying it to a second root key. Use
[`MultiAction`](generated/torchrl.envs.transforms.MultiAction.html#torchrl.envs.transforms.MultiAction) only when you want the env-side
transform to execute a whole chunk per policy call and accept the resulting
re-timed MDP; in that case use [`from_vla()`](generated/torchrl.envs.transforms.MultiAction.html#torchrl.envs.transforms.MultiAction.from_vla)
or pass `chunk_key=("vla_action", "chunk")` to make the env consume the
policy chunk key directly.

### Choosing a chunk executor

Use [`MultiStepActorWrapper`](generated/torchrl.modules.tensordict_module.MultiStepActorWrapper.html#torchrl.modules.tensordict_module.MultiStepActorWrapper) when you
want the environment clock to stay unchanged:

- one outer rollout step is one base-environment step;
- the collector records one reward/done/next-observation per base step;
- the VLA is called only on re-plan steps and skipped while the cached chunk is
being served;
- `replan_interval=None` is open-loop over the whole chunk,
`replan_interval=1` is closed-loop, and intermediate values are
receding-horizon execution.

Use [`MultiAction`](generated/torchrl.envs.transforms.MultiAction.html#torchrl.envs.transforms.MultiAction) when you intentionally want a
macro-action environment:

- one outer rollout step executes the whole chunk inside the env transform;
- rewards and observations can be stacked over the chunk or collapsed to the
last base step;
- the MDP is re-timed, which can be convenient for evaluation or scripted
macro-actions but changes the meaning of the rollout time dimension for RL.

For online VLA RL, prefer [`MultiStepActorWrapper`](generated/torchrl.modules.tensordict_module.MultiStepActorWrapper.html#torchrl.modules.tensordict_module.MultiStepActorWrapper)
unless you explicitly want macro-steps.

## Inference loop sketch

The following pseudocode shows the default one-step env contract. The VLA
returns chunks under `("vla_action", "chunk")`, and
[`MultiStepActorWrapper`](generated/torchrl.modules.tensordict_module.MultiStepActorWrapper.html#torchrl.modules.tensordict_module.MultiStepActorWrapper) serves one
cached action under the env-facing `"action"` key per environment step.

```
from torchrl.envs import InitTracker, TransformedEnv
from torchrl.modules import MultiStepActorWrapper
from torchrl.modules.vla import TinyVLA

H = 8
base_env = make_robot_env()
# base_env writes observation.image, observation.state and language_instruction
env = TransformedEnv(base_env, InitTracker()) # writes is_init after reset

policy = TinyVLA(
 action_dim=env.action_spec.shape[-1],
 chunk_size=H,
)

actor = MultiStepActorWrapper(
 policy,
 n_steps=H,
 replan_interval=None, # open-loop: consume the full chunk before re-querying
)

td = env.reset()
for _ in range(max_steps):
 td = actor(td)
 # td now contains:
 # - ("vla_action", "chunk"): the cached H-step policy chunk
 # - action: the single env-facing action served from the cache
 td = env.step(td)
 td = td["next"]
 if td["done"].any():
 td = env.reset(td)
```

For token-output policies, request decoded chunks with `output_mode="both"`
and pass an `action_tokenizer` to the wrapper. The env still consumes
`"action"`; token fields remain available for logging or RL fine-tuning.
If the base environment action key is not `"action"`, pass
`action_keys=[env_action_key]`.

## Training loop sketch

For offline chunked behavior cloning, the replay buffer stores canonical
observations and raw actions. Transforms build the training target
`("vla_action", "chunk")` and its padding mask; the VLA wrapper reads
observations and predicts a chunk with the same nested key that
[`BCLoss`](generated/torchrl.objectives.BCLoss.html#torchrl.objectives.BCLoss) uses.

```
from torchrl.envs.transforms import ActionChunkTransform, Compose
from torchrl.modules.vla import TinyVLA
from torchrl.objectives import BCLoss

H = 8
replay_buffer = make_vla_replay_buffer(
 transform=Compose(
 # Optional: ActionScaling.from_metadata(...),
 ActionChunkTransform(chunk_size=H),
 )
)

policy = TinyVLA(action_dim=action_dim, chunk_size=H)
loss_module = BCLoss(policy, loss_function="l1")
loss_module.set_keys(action=("vla_action", "chunk"), pad_mask="action_is_pad")

optimizer = make_optimizer(policy.parameters())
for _ in range(num_updates):
 batch = replay_buffer.sample(batch_size)
 loss_td = loss_module(batch)
 loss = loss_td["loss_bc"]

 optimizer.zero_grad()
 loss.backward()
 optimizer.step()
```

Token RL fine-tuning uses the same policy contract. The rollout stores sampled
`("vla_action", "tokens")` and behavior-policy
`("vla_action", "log_probs")`. During the update, `get_dist` /
`log_prob` recompute the current policy probabilities from the same
observations, and [`ClipPPOLoss`](generated/torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss) consumes the nested
token keys.

```
from torchrl.modules.vla import TinyVLA
from torchrl.objectives import ClipPPOLoss

policy = TinyVLA(
 action_dim=action_dim,
 chunk_size=H,
 action_head="tokens",
 vocab_size=tokenizer.vocab_size,
 action_tokenizer=tokenizer,
 output_mode="both", # tokens for the loss, decoded chunks for the env
 return_log_probs=True,
)
ppo_loss = ClipPPOLoss(policy, critic_network=None, entropy_bonus=False)
ppo_loss.set_keys(
 action=("vla_action", "tokens"),
 sample_log_prob=("vla_action", "log_probs"),
 advantage="advantage",
)

for _ in range(num_updates):
 rollout = collector.next() # contains sampled tokens and old log_probs
 rollout["advantage"] = compute_group_relative_advantage(rollout)
 # ClipPPOLoss calls the policy to recompute current token log-probs
 # against the stored actions, while log_probs remains the behavior
 # policy value collected during rollout.
 loss_td = ppo_loss(rollout)
 loss = loss_td["loss_objective"] + loss_td.get("loss_entropy", 0)

 optimizer.zero_grad()
 loss.backward()
 optimizer.step()
```

If you write a custom token objective instead of using
[`ClipPPOLoss`](generated/torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss), use [`log_prob()`](generated/torchrl.modules.vla.VLAWrapperBase.html#torchrl.modules.vla.VLAWrapperBase.log_prob)
with a separate output key so the behavior-policy log-probabilities remain
available for ratios:

```
batch = policy.log_prob(batch, log_probs_key="new_log_probs")
ratio = (batch["new_log_probs"] - batch["vla_action", "log_probs"]).exp()
```

## Policies

A VLA policy is an ordinary [`TensorDictModuleBase`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.TensorDictModuleBase.html#tensordict.nn.TensorDictModuleBase) that
maps images, optional proprioceptive state and a language instruction to an
action chunk (continuous), action tokens (discrete), or both.
[`VLAWrapperBase`](generated/torchrl.modules.vla.VLAWrapperBase.html#torchrl.modules.vla.VLAWrapperBase) fixes that contract with explicit
`input_mode` / `output_mode` settings, `tensordict_out` and
`logits_only` forward paths, plus `get_dist` and `log_prob` methods for
loss-time recomputation. The default output keys are the nested
`VLAAction` fields, and can still be overridden through `set_keys` when a
legacy flat layout is required.
[`TinyVLA`](generated/torchrl.modules.vla.TinyVLA.html#torchrl.modules.vla.TinyVLA) is a small reference policy for tests and
tutorials.

| [`VLAWrapperBase`](generated/torchrl.modules.vla.VLAWrapperBase.html#torchrl.modules.vla.VLAWrapperBase)(*args, **kwargs) | Base class for TensorDict-native Vision-Language-Action policies. |
| --- | --- |
| [`TinyVLA`](generated/torchrl.modules.vla.TinyVLA.html#torchrl.modules.vla.TinyVLA)(*args, **kwargs) | A tiny, dependency-free reference VLA policy for CI and tutorials. |
| [`LeRobotPolicyWrapper`](generated/torchrl.modules.vla.LeRobotPolicyWrapper.html#torchrl.modules.vla.LeRobotPolicyWrapper)(*args, **kwargs) | Expose an external (LeRobot-style) policy as a TorchRL VLA policy. |

At inference a chunk policy predicts `H` actions while the environment consumes
one per step - and chunking only pays off if the (expensive) policy is *not*
queried at every step.
[`MultiStepActorWrapper`](generated/torchrl.modules.tensordict_module.MultiStepActorWrapper.html#torchrl.modules.tensordict_module.MultiStepActorWrapper) provides
this: it caches the predicted actions, emits one per step and skips the
wrapped actor while the cache lasts - open-loop by default, receding horizon
with `replan_interval`, re-planning on env resets via `is_init`.
[`MultiAction`](generated/torchrl.envs.transforms.MultiAction.html#torchrl.envs.transforms.MultiAction) is the env-side alternative
(one base step per chunk action, a single policy call per chunk, at the price
of a re-timed MDP). [`ToyVLAEnv`](generated/torchrl.envs.ToyVLAEnv.html#torchrl.envs.ToyVLAEnv) - a tiny synthetic env
speaking the canonical schema, whose state echoes the executed action - lets
you smoke-test this machinery without any simulator dependency.

## Objectives

VLA fine-tuning needs no dedicated loss classes; the standard objectives
apply directly:

- *Chunked behavior cloning* is [`BCLoss`](generated/torchrl.objectives.BCLoss.html#torchrl.objectives.BCLoss) with the
action chunk as the `action` and the chunk-padding mask excluded via its
`pad_mask` key:

```
loss = BCLoss(policy, loss_function="l1")
loss.set_keys(action=("vla_action", "chunk"), pad_mask="action_is_pad")
```
- *Token RL fine-tuning* (GRPO-style, following SimpleVLA-RL / RL4VLA) is
[`ClipPPOLoss`](generated/torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss) over the action tokens: advantages
are precomputed (group-relative), so no critic is needed, and the token
head's sequence-level log-probabilities match the `sample_log_prob`
contract. The advantage carries the trailing singleton value-dim the PPO
losses expect (`[batch, 1]`, not `[batch]`):

```
loss = ClipPPOLoss(policy, critic_network=None, entropy_bonus=False)
loss.set_keys(
 action=("vla_action", "tokens"),
 sample_log_prob=("vla_action", "log_probs"),
 advantage="advantage",
)
```