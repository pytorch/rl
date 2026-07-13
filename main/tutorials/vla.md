Note

Go to the end
to download the full example code.

# Vision-Language-Action (VLA) policies with TorchRL

**Author**: [Vincent Moens](https://github.com/vmoens)

Note

To run this tutorial in a notebook, add an installation cell
at the beginning containing:

> ```
> !pip install tensordict
> !pip install torchrl
> ```

```

```

## What you will learn

Vision-Language-Action (VLA) models map camera images, proprioceptive state
and a natural-language instruction to robot actions - usually emitted as a
short *action chunk* of future steps. TorchRL treats a VLA as an ordinary
TensorDict-first policy, so the same replay buffers, transforms, losses and
collectors you already know apply.

In this tutorial we will:

- get a bird's-eye view of what VLAs are and how they are trained;
- meet the canonical VLA TensorDict schema;
- build action chunks and normalize actions with VLA transforms;
- train a small reference policy by chunked behavior cloning;
- execute the chunk policy one step at a time with a receding-horizon executor;
- run one step of RL fine-tuning of a token policy with a GRPO objective.

Everything below runs on CPU with synthetic data and a tiny model.

## VLAs in a nutshell

Structurally, a VLA is a vision-language model (VLM) with an *action head*.
The backbone - a pretrained multimodal transformer such as
[PaliGemma](https://huggingface.co/google/paligemma-3b-pt-224) (used by
[pi0](https://github.com/Physical-Intelligence/openpi)),
[SmolVLM](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct) (used by
[SmolVLA](https://huggingface.co/lerobot/smolvla_base)) or the
[OpenVLA](https://github.com/openvla/openvla) Llama/SigLIP stack -
encodes the camera images and the instruction; a comparatively small head
turns that representation into robot actions. Two head families dominate:

- **token heads** ([RT-2](https://robotics-transformer2.github.io/),
[OpenVLA](https://huggingface.co/openvla/openvla-7b)): actions are
discretized into tokens and emitted through the language-model head,
autoregressively - which is what makes LLM-style RL objectives applicable
to robotics;
- **continuous chunk heads** ([ACT](https://github.com/tonyzhaozh/act),
[OpenVLA-OFT](https://github.com/moojink/openvla-oft),
[pi0](https://huggingface.co/lerobot/pi0_base),
[SmolVLA](https://huggingface.co/lerobot/smolvla_base)): a regression,
diffusion or flow-matching head predicts a short horizon of `H` future
actions (an *action chunk*) in one forward pass.

Chunking is the signature trick of the field: one (expensive) inference
yields several control steps, and the executor decides how many of them to
apply before re-planning.

## The training lifecycle

Training a VLA is best understood as a pipeline of stages, each with its own
data, objective and - importantly - its own *practitioner profile*. Most
users only ever run the last two stages.

1. **VLM pre-training** (inherited). The multimodal backbone is trained on
web-scale image-text data. Nobody does this *for* robotics: you inherit
it by picking a pretrained VLM.
2. **VLA pre-training** (sometimes called *mid-training*). The VLM + action
head is trained by large-scale behavior cloning on cross-embodiment
teleoperation corpora - Open X-Embodiment (~1M episodes), DROID, the
LeRobot community datasets. This is what turns a VLM into a *generalist*
VLA, and it is compute-heavy (hundreds of GPU-days). In practice you
consume its output as a released checkpoint: OpenVLA, pi0, SmolVLA.
3. **Supervised post-training** (what most people mean by "training a
VLA"). The generalist checkpoint is fine-tuned by behavior cloning on a
small, task- and robot-specific dataset: typically 50-500 teleoperated
episodes recorded in the LeRobot format. Needs: a checkpoint, your
demonstrations, and dataset action statistics for normalization. This is
a single-GPU affair.
4. **RL post-training** (optional, increasingly standard). Behavior cloning
is capped by demonstration quality and compounds errors over long
horizons. RL fine-tuning against a sparse task-success reward
(SimpleVLA-RL, RL4VLA - GRPO / PPO-style objectives over action tokens)
pushes success rates beyond the demos. Needs everything above *plus* a
task you can roll out: a simulator, or a real robot with a success
detector.
5. **Evaluation / deployment**. Closed-loop rollout with chunked execution
(receding horizon), measuring success rate over episodes.

## What do you need to bring?

Depending on where you enter the pipeline:

- *Evaluate a released VLA*: a checkpoint
([`LeRobotPolicyWrapper`](../reference/generated/torchrl.modules.vla.LeRobotPolicyWrapper.html#torchrl.modules.vla.LeRobotPolicyWrapper)) and a task -
[`MultiStepActorWrapper`](../reference/generated/torchrl.modules.tensordict_module.MultiStepActorWrapper.html#torchrl.modules.tensordict_module.MultiStepActorWrapper) executes
it chunk by chunk (one inference per chunk, re-planning on resets) and
[`SuccessReward`](../reference/generated/torchrl.envs.transforms.SuccessReward.html#torchrl.envs.transforms.SuccessReward) scores it.
- *Fine-tune on your task* (stage 3, the common case): a checkpoint, your
demos ([`LeRobotExperienceReplay`](../reference/generated/torchrl.data.datasets.LeRobotExperienceReplay.html#torchrl.data.datasets.LeRobotExperienceReplay)), chunk
targets ([`ActionChunkTransform`](../reference/generated/torchrl.envs.transforms.ActionChunkTransform.html#torchrl.envs.transforms.ActionChunkTransform)),
normalization from the dataset statistics
([`ActionScaling.from_metadata`](../reference/generated/torchrl.envs.transforms.ActionScaling.html#torchrl.envs.transforms.ActionScaling.from_metadata))
and [`BCLoss`](../reference/generated/torchrl.objectives.BCLoss.html#torchrl.objectives.BCLoss) with its `pad_mask` key.
- *RL fine-tune* (stage 4): all of the above, a rollout-able task with a
success signal, and [`ClipPPOLoss`](../reference/generated/torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss) over the
action tokens for token policies.
- *Study the mechanics or research from scratch*: no checkpoint, no robot -
a tiny reference policy ([`TinyVLA`](../reference/generated/torchrl.modules.vla.TinyVLA.html#torchrl.modules.vla.TinyVLA)) and
synthetic data, which is exactly what this tutorial does. Every component
below is the same one you would use at full scale; only the model and the
data shrink.

The rest of the tutorial walks the pipeline in that order: data schema,
transforms, behavior cloning, chunked execution, RL fine-tuning.

```
import torch
from tensordict import NonTensorStack, TensorDict

torch.manual_seed(0)
```

```
<torch._C.Generator object at 0x7efb80b8c390>
```

## The canonical VLA schema

VLA components agree on a single key layout: camera image(s) and proprioceptive
state live under `observation`, while the per-trajectory language instruction
and the action live at the tensordict root (mirroring
[`OpenXExperienceReplay`](../reference/generated/torchrl.data.datasets.OpenXExperienceReplay.html#torchrl.data.datasets.OpenXExperienceReplay)). A single observation
therefore looks like this:

```
batch, n_cam_c, hw, state_dim, action_dim = 8, 3, 16, 6, 4

def make_observation(batch=batch):
 return TensorDict(
 {
 "observation": {
 "image": torch.randint(
 0, 255, (batch, n_cam_c, hw, hw), dtype=torch.uint8
 ),
 "state": torch.randn(batch, state_dim),
 },
 "language_instruction": NonTensorStack(
 *[f"pick up object {i}" for i in range(batch)]
 ),
 },
 batch_size=[batch],
 )

obs = make_observation()
obs
```

```
TensorDict(
 fields={
 language_instruction: NonTensorStack(
 ['pick up object 0', 'pick up object 1', 'pick up ...,
 batch_size=torch.Size([8]),
 device=None),
 observation: TensorDict(
 fields={
 image: Tensor(shape=torch.Size([8, 3, 16, 16]), device=cpu, dtype=torch.uint8, is_shared=False),
 state: Tensor(shape=torch.Size([8, 6]), device=cpu, dtype=torch.float32, is_shared=False)},
 batch_size=torch.Size([8]),
 device=None,
 is_shared=False)},
 batch_size=torch.Size([8]),
 device=None,
 is_shared=False)
```

## Action chunking and normalization

[`ActionChunkTransform`](../reference/generated/torchrl.envs.transforms.ActionChunkTransform.html#torchrl.envs.transforms.ActionChunkTransform) turns a per-step action
tensor `[*B, T, action_dim]` into the chunked training target
`("vla_action", "chunk")` `[*B, T, H, action_dim]` (plus an
`action_is_pad` mask): for every step `t` it gathers the next `H`
actions, stopping (padding and masking) at the trajectory boundaries when
the sampled window carries its done state. This is the training target of
modern chunked VLA policies (ACT, OpenVLA-OFT, pi0).

Chunks mean different things on the two sides of the pipeline, and keeping
the two pictures apart avoids a classic confusion:

```
Training (behavior cloning) | Inference (chunked execution)
-----------------------------------+----------------------------------
dataset actions: a0 a1 a2 a3 ... | o_t --> VLA --> chunk [b0 b1 b2]
 | | (one chunk per query)
 | sample a trajectory slice | |
 v | v
ActionChunkTransform | MultiStepActorWrapper (policy
 | | wrapper: 1 policy call per chunk,
 v | pops 1 action per env step)
[[a0, a1, a2], <- target at t=0 | or MultiAction (re-timed env:
 [a1, a2, a3], <- target at t=1 | one base step per chunk action)
 [a2, a3, a3], <- target at t=2 | |
 ...] + action_is_pad mask | v
 | | step: b0 -> b1 -> b2 -> re-query
 v | --> [c0 c1 c2] -> c0 -> ...
BCLoss(policy(o_t), row t) |
 | executed trace (open loop):
overlapping rows, one per dataset | b0 b1 b2 | c0 c1 c2 | ...
step: the policy may be queried | non-overlapping tiles of time
at any t |
```

The *training table* (left) is stride-1 and overlapping - one supervised
example per dataset step, because at deployment the policy can be queried at
any phase. The *executed trace* (right) tiles time without overlap when run
open-loop: each committed chunk is consumed before the next one. Both
[`MultiStepActorWrapper`](../reference/generated/torchrl.modules.tensordict_module.MultiStepActorWrapper.html#torchrl.modules.tensordict_module.MultiStepActorWrapper) (used
later in this tutorial; one policy call per chunk, with receding-horizon
re-planning via `replan_interval` and reset handling via `is_init`) and
[`MultiAction`](../reference/generated/torchrl.envs.transforms.MultiAction.html#torchrl.envs.transforms.MultiAction) (which re-times the env
instead, stepping it once per chunk action) realize that open-loop trace.

```
from torchrl.envs.transforms import ActionChunkTransform, ActionScaling

T, H = 6, 4
window = TensorDict({"action": torch.randn(2, T, action_dim)}, batch_size=[2, T])
chunked = ActionChunkTransform(chunk_size=H)(window)
chunked["vla_action", "chunk"].shape # [2, T, H, action_dim]
```

```
torch.Size([2, 6, 4, 4])
```

[`ActionScaling`](../reference/generated/torchrl.envs.transforms.ActionScaling.html#torchrl.envs.transforms.ActionScaling) handles action normalization.
With explicit statistics (`loc`/`scale`, or the
[`from_stats()`](../reference/generated/torchrl.envs.transforms.ActionScaling.html#torchrl.envs.transforms.ActionScaling.from_stats) /
[`from_metadata()`](../reference/generated/torchrl.envs.transforms.ActionScaling.html#torchrl.envs.transforms.ActionScaling.from_metadata) constructors)
the transform normalizes expert actions on the replay-buffer sample path
(build it with `in_keys_inv=[]` for a buffer raw data is written to
through `extend`, which applies the inverse) and denormalizes the policy's
predicted actions when attached to an environment (or explicitly via
[`denormalize()`](../reference/generated/torchrl.envs.transforms.ActionScaling.html#torchrl.envs.transforms.ActionScaling.denormalize)).

```
normalize = ActionScaling(loc=torch.zeros(action_dim), scale=torch.ones(action_dim) * 2)
normalized = normalize(TensorDict({"action": torch.full((4, action_dim), 2.0)}, [4]))
normalized["action"] # all ones
```

```
tensor([[1., 1., 1., 1.],
 [1., 1., 1., 1.],
 [1., 1., 1., 1.],
 [1., 1., 1., 1.]])
```

## A reference policy

[`VLAWrapperBase`](../reference/generated/torchrl.modules.vla.VLAWrapperBase.html#torchrl.modules.vla.VLAWrapperBase) is the thin base class for VLA
policies; [`TinyVLA`](../reference/generated/torchrl.modules.vla.TinyVLA.html#torchrl.modules.vla.TinyVLA) is a small reference policy
(convolutional image encoder + state MLP + hashed instruction embedding +
action head) for tutorials and tests. With a continuous head it predicts an
action chunk.

```
from torchrl.modules.vla import TinyVLA

policy = TinyVLA(action_dim=action_dim, chunk_size=H, hidden_dim=64)
policy(make_observation())["vla_action", "chunk"].shape # [batch, H, action_dim]
```

```
torch.Size([8, 4, 4])
```

## Behavior cloning

Chunked behavior cloning is plain [`BCLoss`](../reference/generated/torchrl.objectives.BCLoss.html#torchrl.objectives.BCLoss): the
action chunk is the `action` (an elementwise loss does not care about the
extra horizon dim) and the `pad_mask` key excludes padded chunk steps from
the loss. Here we overfit a tiny synthetic dataset to confirm the policy
learns.

```
from torchrl.objectives import BCLoss

data = make_observation()
# a synthetic "expert": a fixed linear map from the state to an action chunk
expert = (
 data["observation", "state"] @ torch.randn(state_dim, H * action_dim)
).reshape(batch, H, action_dim)
data["vla_action", "chunk"] = expert
data["action_is_pad"] = torch.zeros(batch, H, dtype=torch.bool)

bc_loss = BCLoss(policy, loss_function="l1")
bc_loss.set_keys(action=("vla_action", "chunk"), pad_mask="action_is_pad")
initial = bc_loss(data)["loss_bc"].item()
optimizer = torch.optim.Adam(bc_loss.parameters(), lr=1e-2)
for _ in range(100):
 optimizer.zero_grad()
 bc_loss(data)["loss_bc"].backward()
 optimizer.step()
```

The behavior-cloning loss drops sharply as the policy fits the expert chunks:

```
(initial, bc_loss(data)["loss_bc"].item())
```

```
(1.6689702272415161, 0.07800992578268051)
```

## Chunked inference

At inference, a chunk policy predicts `H` actions but the environment
consumes one action per step - and a VLA forward pass is expensive, so the
whole point of chunking is to *not* run the policy at every step.
[`MultiStepActorWrapper`](../reference/generated/torchrl.modules.tensordict_module.MultiStepActorWrapper.html#torchrl.modules.tensordict_module.MultiStepActorWrapper) does
exactly that: it caches the predicted actions and emits one per step,
skipping the wrapped actor entirely while the cache lasts; with
`replan_interval` it re-queries before the cache runs out (receding
horizon), and an env reset (tracked through `is_init`) re-plans the
affected envs. ([`MultiAction`](../reference/generated/torchrl.envs.transforms.MultiAction.html#torchrl.envs.transforms.MultiAction) is the
env-side alternative: it steps the base env once per chunk action, also
with a single policy call per chunk, at the price of re-timing the MDP.)

To see it in action we need an environment. Real evaluations would use a
simulator (e.g. `gym-pusht`) or a robot; TorchRL ships
[`ToyVLAEnv`](../reference/generated/torchrl.envs.ToyVLAEnv.html#torchrl.envs.ToyVLAEnv), a tiny synthetic env that speaks the
canonical VLA schema and whose state echoes the executed action - ideal for
watching execution machinery work (its ~40-line source is also a template
for wrapping your own robot in [`EnvBase`](../reference/generated/torchrl.envs.EnvBase.html#torchrl.envs.EnvBase)). Its specs
tell us exactly what it consumes and produces: camera image (`uint8`) and
proprioceptive state under `observation`, the instruction at the root.

```
from torchrl.envs import ToyVLAEnv

base_env = ToyVLAEnv(
 action_dim=action_dim,
 state_dim=state_dim,
 image_shape=(n_cam_c, hw, hw),
 batch_size=[2],
)
base_env.observation_spec
```

```
Composite(
 observation: Composite(
 image: UnboundedDiscrete(
 shape=torch.Size([2, 3, 16, 16]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([2, 3, 16, 16]), device=cpu, dtype=torch.uint8, contiguous=True),
 high=Tensor(shape=torch.Size([2, 3, 16, 16]), device=cpu, dtype=torch.uint8, contiguous=True)),
 device=cpu,
 dtype=torch.uint8,
 domain=discrete),
 state: UnboundedContinuous(
 shape=torch.Size([2, 6]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([2, 6]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([2, 6]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous),
 device=None,
 shape=torch.Size([2]),
 data_cls=None),
 language_instruction: NonTensor(
 shape=torch.Size([2]),
 space=None,
 device=None,
 dtype=None,
 domain=None,
 example_data=push the T-shaped block onto the target),
 device=None,
 shape=torch.Size([2]),
 data_cls=None)
```

The action interface is a single continuous action per step (the wrapper
lives on the policy side, so the env specs are untouched):

```
base_env.action_spec
```

```
BoundedContinuous(
 shape=torch.Size([2, 4]),
 space=ContinuousBox(
 low=Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, contiguous=True),
 high=Tensor(shape=torch.Size([2, 4]), device=cpu, dtype=torch.float32, contiguous=True)),
 device=cpu,
 dtype=torch.float32,
 domain=continuous)
```

The wrapper auto-discovers the policy's `("vla_action", "chunk")` output
and serves the base env's `"action"` key from it. `InitTracker` provides
the `is_init` flag the wrapper uses to re-plan on resets.

```
from torchrl.envs.transforms import InitTracker, TransformedEnv
from torchrl.modules import MultiStepActorWrapper

actor = MultiStepActorWrapper(policy, n_steps=H, replan_interval=2)
env = TransformedEnv(base_env, InitTracker())
```

A plain [`rollout()`](../reference/generated/torchrl.envs.EnvBase.html#id2) runs the interaction loop. The
executed per-step actions are recorded under `action`. The wrapper counter
shows the receding-horizon cadence: with `replan_interval=2` it serves two
actions from the cache, then re-plans.

```
eval_rollout = env.rollout(6, actor)
eval_rollout["action"].shape # [2, 6, action_dim]: the executed actions
```

```
torch.Size([2, 6, 4])
```

```
eval_rollout["counter"][0, :, 0] # tensor([1, 2, 1, 2, 1, 2])
```

```
tensor([1, 2, 1, 2, 1, 2], dtype=torch.int32)
```

The env's state echo confirms that the executed cadence matches what the
wrapper served from its cache:

```
executed = eval_rollout["next", "observation", "state"][..., :action_dim]
torch.allclose(executed, eval_rollout["action"]) # True: echo of what env got
```

```
True
```

## RL fine-tuning

VLAs are increasingly post-trained with RL. GRPO-style fine-tuning of a
*token* VLA (action tokens emitted through a language-model head) is plain
[`ClipPPOLoss`](../reference/generated/torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss): group-relative advantages are
precomputed (so no critic, `critic_network=None`), and the token head's
sequence-level `log_probs` match the loss's `sample_log_prob` contract
out of the box - only the keys need remapping.

We first roll out the token policy to obtain action tokens and their
behavior-policy log-probabilities, attach a (here synthetic) advantage, then
take one optimization step.

```
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.objectives import ClipPPOLoss

token_policy = TinyVLA(
 action_dim=action_dim,
 chunk_size=H,
 action_head="tokens",
 vocab_size=64,
)
# The token head follows the ambient exploration context: collectors roll out
# under ``ExplorationType.RANDOM`` (so actions are sampled), while greedy
# evaluation uses ``set_exploration_type(ExplorationType.DETERMINISTIC)``. We
# sample here to mimic a behavior-policy rollout -- no policy mutation needed.
with set_exploration_type(ExplorationType.RANDOM):
 rollout = token_policy(make_observation()) # writes nested tokens + log-probs
# one advantage per sample, with the trailing singleton value-dim the PPO
# losses expect (a flat [batch] advantage would silently broadcast wrong)
rollout["advantage"] = torch.randn(batch, 1)
rollout["vla_action", "log_probs"] = rollout["vla_action", "log_probs"].detach()

grpo_loss = ClipPPOLoss(
 token_policy, critic_network=None, entropy_bonus=False, clip_epsilon=0.2
)
grpo_loss.set_keys(
 action=("vla_action", "tokens"),
 sample_log_prob=("vla_action", "log_probs"),
 advantage="advantage",
)
grpo_optimizer = torch.optim.Adam(grpo_loss.parameters(), lr=1e-3)
grpo_optimizer.zero_grad()
grpo_loss(rollout)["loss_objective"].backward()
grpo_optimizer.step()
```

## Conclusion

We loaded VLA-shaped data into the canonical schema, built action chunks and
normalized actions, trained a reference policy by chunked behavior cloning
([`BCLoss`](../reference/generated/torchrl.objectives.BCLoss.html#torchrl.objectives.BCLoss) with a pad mask), executed it with a
receding-horizon actor wrapper that skips the policy between re-plans, and
ran one step of token GRPO fine-tuning
([`ClipPPOLoss`](../reference/generated/torchrl.objectives.ClipPPOLoss.html#torchrl.objectives.ClipPPOLoss), no critic) - all with the
standard TorchRL primitives; the only VLA-specific pieces are the data
schema, the policies and the transforms. To scale up,
swap [`TinyVLA`](../reference/generated/torchrl.modules.vla.TinyVLA.html#torchrl.modules.vla.TinyVLA) for a wrapped open checkpoint
([`LeRobotPolicyWrapper`](../reference/generated/torchrl.modules.vla.LeRobotPolicyWrapper.html#torchrl.modules.vla.LeRobotPolicyWrapper)) and stream real data with
[`LeRobotExperienceReplay`](../reference/generated/torchrl.data.datasets.LeRobotExperienceReplay.html#torchrl.data.datasets.LeRobotExperienceReplay) or
[`OpenXExperienceReplay`](../reference/generated/torchrl.data.datasets.OpenXExperienceReplay.html#torchrl.data.datasets.OpenXExperienceReplay).

## Further reading

- OpenVLA-OFT (chunked continuous fine-tuning): [https://arxiv.org/abs/2502.19645](https://arxiv.org/abs/2502.19645)
- pi0 (flow-matching VLA): [https://arxiv.org/abs/2410.24164](https://arxiv.org/abs/2410.24164)
- FAST (action tokenization): [https://arxiv.org/abs/2501.09747](https://arxiv.org/abs/2501.09747)
- SimpleVLA-RL (GRPO fine-tuning): [https://arxiv.org/abs/2509.09674](https://arxiv.org/abs/2509.09674)
- The [VLA reference documentation](../reference/vla.html#ref-vla).

**Total running time of the script:** (0 minutes 0.388 seconds)

[`Download Jupyter notebook: vla.ipynb`](../_downloads/4f2ab836eee1cdeaf65dd37f30e821be/vla.ipynb)

[`Download Python source code: vla.py`](../_downloads/7ac3fe12741ad5659eaba37b19d94844/vla.py)

[`Download zipped: vla.zip`](../_downloads/1b3f10566d2ea1d6fd6d0c14bf546984/vla.zip)

[Gallery generated by Sphinx-Gallery](https://sphinx-gallery.github.io)