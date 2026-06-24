# VLA GRPO: reinforcement learning for action-token VLA policies

This example shows how to take a supervised Vision-Language-Action (VLA)
policy and improve it with reinforcement learning from environment success.
The short version is:

1. start from a policy that can read an image, a language instruction, and a
   robot state;
2. let it sample several attempts from the same initial state;
3. keep the attempts that give a useful success/failure contrast; and
4. update the action-token probabilities so successful attempts become more
   likely.

That is what **VLA GRPO** means in this recipe: GRPO-style grouped policy
optimization applied to a VLA policy whose actions are represented as discrete
language-model tokens.

## Why do RL on top of supervised learning?

Supervised fine-tuning (SFT) is the usual first step for VLA models. It teaches
an image-and-language model to imitate actions from demonstrations: given this
camera frame and this instruction, predict the demonstrated robot action.
That is necessary, but it is not always enough.

SFT optimizes next-action imitation, not task success. A model can match the
training action distribution and still fail after a few small errors compound in
closed-loop control. It also treats all demonstrated tokens as targets, even
when several different actions would work, and it cannot directly prefer the
action sequence that completes the task over one that almost completes it.

This example adds a second stage: run the policy in an environment, measure
whether the task actually succeeds, and fine-tune the policy from those returns.
There is no learned critic here. Instead, each initial state is repeated several
times under the same policy. If some attempts succeed and others fail, the
successful attempts get positive relative advantage and the failed attempts get
negative relative advantage. If all attempts are identical failures or identical
successes, the group carries no ranking signal and can be dropped.

In the toy example, the result is a small policy that learns to choose action
tokens that land near a hidden target. In the LIBERO configuration, the same
machinery is used to fine-tune an OpenVLA-style robot policy on manipulation
benchmarks.

## The pieces in this directory

The directory contains two versions of the same idea.

### Toy scale: learn the algorithm without a robot simulator

The default config, `config/vla_grpo_toy.yaml`, trains `TinyVLA` on
`ToyVLAEnv`. This is the quickest way to understand and test the stack.

`ToyVLAEnv` is a tiny TorchRL environment that speaks the same TensorDict schema
as a real VLA robot environment:

- `("observation", "image")` is a camera-like image tensor;
- `("observation", "state")` is a proprioceptive state vector;
- `"language_instruction"` is a text instruction;
- `"action"` is a continuous action vector in `[-1, 1]`.

In tracking mode, every episode samples a target action. The target is exposed
in the state so the task is learnable. The policy succeeds when its executed
action stays within a tolerance of that target for a few consecutive steps. The
random camera image is only there to exercise the image path; rendered eval
videos draw the executed action and target as markers so you can see the task.

`TinyVLA` is a deliberately small VLA policy. It is not meant to be a strong
robot policy. It is a dependency-light stand-in that lets CI and local runs test
the same data flow that the large model uses:

- a small convolutional encoder reads the image;
- an MLP reads the state;
- a hashed embedding represents the instruction string without requiring a
  tokenizer;
- the fused features feed an action-token head.

The important point is that `TinyVLA` predicts **tokens**, not continuous
actions directly. Those tokens are decoded back into continuous actions by the
same TorchRL tokenizer transform used for the larger VLA policy.

Run the toy recipe with:

```bash
python sota-implementations/vla_grpo/vla-grpo.py
```

The greedy evaluation success rate should climb from zero (a randomly
initialized argmax policy never solves the task) to roughly 0.4-0.6 within
about 200 iterations on CPU. The sampled training success rate starts near the
random baseline and climbs alongside it.

### LIBERO scale: run the same recipe on robot manipulation

`config/vla_grpo_libero.yaml` swaps the toy environment for
`torchrl.envs.LiberoEnv` and swaps `TinyVLA` for an OpenVLA-OFT token policy.
LIBERO provides simulated robot manipulation tasks: a scene is reset to an
initial state, the policy receives camera observations and an instruction such
as a pick-and-place command, and success is measured by the simulator task
checker.

The high-level loop is unchanged from the toy setting. The policy samples
action-token chunks, those chunks are decoded to continuous robot actions, the
environment reports success or failure, and GRPO updates the token probabilities
from repeated attempts at the same initial state.

## How the environment and model are brought together

One outer TorchRL environment step is one policy decision. The policy does not
emit a single action; it emits a short **action chunk**. For a chunk length `H`
and action dimension `A`, the token policy emits:

```text
vla_action.tokens [B, H, A]
```

Each entry is a discrete token. The transforms then turn that into environment
interaction:

```text
policy forward
  -> ("vla_action", "tokens") [B, H, A]
  -> ActionTokenizerTransform.inverse   tokens -> continuous action_chunk
  -> MultiAction(stack_rewards=False)   execute H base-env actions
  -> SuccessReward                      convert task success into decision reward
  -> StepCounter(max_outer_steps)       truncate in policy decisions
```

`MCAdvantage` sits on the replay-buffer path. It receives completed
trajectories, groups trajectories that started from the same initial state, and
computes a Monte-Carlo advantage from the group-normalized return. With
`trajectory_return="sum"`, the trajectory-level success return is broadcast to
every chunk decision in that trajectory, which is the shape PPO needs.

The loss is `ClipPPOLoss` configured like a GRPO policy update: no critic, no
entropy bonus, no KL-to-reference term, and asymmetric DAPO Clip-Higher bounds.
For token policies the ratio can be computed per token, which matches the
SimpleVLA-RL semantics used by the OpenVLA configuration.

## What is OpenVLA-OFT (token)?

OpenVLA is a large vision-language-action model: it uses a vision encoder and a
language model backbone to map an image plus an instruction to robot actions.
OpenVLA-OFT is the OpenVLA-style model family used by the SimpleVLA-RL recipe.
In this README, **OpenVLA-OFT (token)** means the specific SimpleVLA-RL variant
where robot actions are represented as language-model tokens.

That detail matters because there are two common ways to put an action head on a
VLA model:

- a continuous head predicts real-valued robot actions directly;
- a token head discretizes each action dimension into bins and predicts those
  bins as tokens.

This recipe uses the token head. A single forward pass emits the whole action
chunk, sampled from a 256-way categorical over the action-token window at the
tail of the LLaMA-2 vocabulary. The tokenizer decodes those tokens back to
continuous robot actions before the environment sees them.

`openvla.py` wraps this policy as a TorchRL `VLAWrapperBase`. The wrapper owns
the model-side details that would otherwise obscure the RL loop: prompt
construction, image preprocessing, action-token decoding, temperature handling,
and log-probability recomputation for PPO. It writes the canonical structured
VLA action container, so sampled tokens and their behavior log-probabilities
live at `("vla_action", "tokens")` and `("vla_action", "log_probs")`. The
vendored modeling code under `openvla_oft/` comes from
[SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL) (MIT).

Important compatibility note: the official continuous-head OpenVLA-OFT
checkpoints are not interchangeable with this token-head variant. Use the
SimpleVLA-RL SFT checkpoints, for example `Haozhan72/*`:

```python
from openvla import OpenVLAOFTWrapper

policy = OpenVLAOFTWrapper.from_pretrained(
    "Haozhan72/Openvla-oft-SFT-libero10-traj1",
    temperature=1.6,
    device="cuda",
)
tokenizer = policy.action_tokenizer  # decode tokens -> env actions
```

Before spending RL compute on a checkpoint, validate the loading path by
evaluating the SFT checkpoint greedily on its LIBERO suite through
`torchrl.envs.LiberoEnv` (`init_state_mode="cycle"`, 50 trials/task) and compare
against the SimpleVLA-RL paper's SFT numbers. For example, LIBERO-Spatial
one-trajectory SFT is about 63.6%, with a few points of expected variance.

The OpenVLA wrapper tests use a tiny random-weight model with the same token
layout, so they do not download a checkpoint:

```bash
pytest sota-implementations/vla_grpo/test_openvla.py
```

## What gets logged

With a logger configured (`logger.backend=wandb`, the default), each iteration
logs reward curves (`train/reward_mean`, `train/reward_max`), success rate, and
throughput split into collection and optimization:

- `throughput/inference_env_steps_per_s`
- `throughput/inference_decisions_per_s`
- `throughput/train_decisions_per_s`
- `throughput/optim_steps_per_s`

The collector can also be switched between synchronous and async execution
paths for throughput experiments:

```bash
# fully synchronous baseline
python sota-implementations/vla_grpo/vla-grpo.py \
  collector.async_env=false collector.async_policy=false

# asynchronous env slots, but no policy auto-batching
python sota-implementations/vla_grpo/vla-grpo.py \
  collector.async_env=true collector.async_policy=false env.num_envs=8

# asynchronous env slots plus auto-batched policy inference
python sota-implementations/vla_grpo/vla-grpo.py \
  collector.async_env=true collector.async_policy=true env.num_envs=8 \
  collector.server_max_batch_size=8 collector.server_timeout=0.01
```

`collector.async_env=true` uses `AsyncBatchedCollector` so faster environment
slots do not wait at a global step barrier. `collector.async_policy=true` routes
policy calls through an inference server; with multiple async env slots this
enables auto-batching and logs `policy_server/*` counters such as average batch
size, request rate, and queue/forward latency. The `false/true` combination is
available as a policy-server plumbing ablation, but policy auto-batching is most
meaningful when several env slots submit requests concurrently.

Eval rollouts can also be rendered to video (`logger.record_video=true`, on by
default). A dedicated single-environment recorder is built with
`from_pixels=True`: `ToyVLAEnv` renders the tracking scene, while `LiberoEnv`
exposes its camera. `torchrl.record.VideoRecorder` writes
`logger.video_episodes` greedy episodes to `eval/video` on every eval. wandb
video encoding needs `moviepy` from the `dev` dependency group. Disable videos
with `logger.record_video=false`.

Checkpointing is shared by the toy and LIBERO configs. `checkpoint_latest.pt`
is written to the hydra run directory every `checkpoint.save_iter` iterations;
resume with:

```bash
python sota-implementations/vla_grpo/vla-grpo.py \
  checkpoint.resume=/path/to/checkpoint_latest.pt
```

## LIBERO configuration details

The full LIBERO config follows the SimpleVLA-RL hyper-parameter shape:

- groups of `n=8` rollouts per initial state;
- 64 initial states per iteration, for 512 trajectories before dynamic
  filtering;
- 512 base environment steps, or 64 chunk decisions, per episode;
- rollout temperature 1.6 and greedy evaluation;
- dynamic sampling bounds `(0.1, 0.9)` to drop groups that are all failure or
  all success;
- learning rate `5e-6` with constant-with-warmup scheduling;
- gradient accumulation and gradient clipping at 1.0;
- asymmetric clip `(0.2, 0.28)` applied to per-token importance ratios by
  default (`loss.ratio_level: token`).

A sequence-level ratio remains available as a config switch for ablations, but
for a 56-token action chunk it saturates the clip range much more easily than
per-token ratios.

LIBERO simulation runs in parallel worker processes (`env.num_envs`, one MuJoCo
instance each), and policy inference batches across workers. Group accounting is
the main thing to keep in mind: GRPO needs repeated attempts from the same
initial state under the same policy. Each worker owns a disjoint `group_id`
block so advantages never mix across unrelated groups.

Because groups are repeated serially within each worker, `env.num_envs` should
not exceed `collector.groups_per_iter`; otherwise many same-policy collection
polls are needed before each worker can finish all `group_size` rollouts for a
group and the replay buffer receives advantaged decisions. For best throughput,
set `env.num_envs` to a divisor of `collector.groups_per_iter`, often the same
value.

When `env.parallel_group_repeats=true`, `env.num_envs / collector.group_size`
logical workers each run one repeated-initial-state group in parallel. In this
mode, prefer setting `collector.groups_per_iter` to that logical worker count
so one target group wave is aligned. If `collector.candidate_group_size` is
larger than `collector.group_size`, each worker in a logical group repeats the
same initial state serially enough times to produce up to the requested
candidate count. For example, 8 parallel workers x 2 serial repeats gives at
most 16 candidates. Groups can be written earlier if the candidates already
contain a useful selected subset.

The replay-buffer writer polls the collector at one outer step per worker, so
complete trajectories are handed to the replay buffer shortly after they finish
instead of waiting for a full max-length rollout from every worker.
`MCAdvantage` runs as the replay-buffer transform and keeps incomplete groups
queued across same-policy polls until all siblings arrive.
`max_collect_batches_per_iter` sets the safety cap in target group waves, and
`collector.min_replay_decisions` can require a minimum number of useful replay
decisions before the PPO update.

Candidate selection is delegated to `MCAdvantageSelector` (`first`, `uniform`,
or `balanced`), so the replay-buffer transform owns the sample-selection policy
while the collector only supplies same-policy completed trajectories. At the
policy-update boundary the replay buffer, incomplete advantage queues, and
in-flight collector trajectories are cleared before the next policy is rolled
out. LIBERO workers stamp parallel-repeat group ids from the cycled
initial-state id so fast and slow sibling workers can still complete a
same-initial-state GRPO group under the same policy even when their episode
lengths differ.

Run the LIBERO recipe with:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO
python sota-implementations/vla_grpo/vla-grpo.py --config-name vla_grpo_libero
# or: sbatch sota-implementations/vla_grpo/vla-grpo.sbatch
```

Requirements beyond the toy scale: LIBERO (see the `torchrl.envs.LiberoEnv`
docs for install notes), `transformers`, `timm`, `Pillow`, and `peft` when
`policy.lora_rank` is set.

## Hardware notes

- The default configuration trains a LoRA adapter (`policy.lora_rank: 32`) on a
  single GPU while the simulation workers occupy CPU cores. Rollout wall-clock
  dominates, so scale `env.num_envs` with the available cores first, while
  keeping it within the GRPO grouping constraint above.
- Set `collector.policy_device` to a different CUDA device to keep rollout
  inference on a separate policy replica. The training loop copies only the
  trainable state dict after optimizer updates, so this split is intended for
  LoRA/adapters rather than full-parameter fine-tuning.
- Headless LIBERO rendering uses MuJoCo/robosuite EGL by default
  (`env.render_backend: egl`). `env.render_gpu_ids` controls the EGL-visible
  render device ids assigned to workers, round-robin. The default `[0]` works
  on a single-GPU allocation; on a multi-GPU node, override it, for example
  `env.render_gpu_ids=[0,1,2,3]`, to spread render workers across GPUs. These
  ids are the devices visible to EGL inside the process/container and may not
  match global CUDA ordinals.
- Set `logger.eval_process=true` to move greedy eval into a dedicated process.
  Use `logger.eval_device` for its policy device and `env.eval_render_gpu_ids`
  for its EGL render workers; when the latter is left null, eval reuses
  `env.render_gpu_ids`.
- Minimal CUDA containers often lack the NVIDIA EGL/GLVND userspace stack.
  Before debugging TorchRL, verify that `libEGL_nvidia`, `libnvidia-eglcore`,
  `libGLX_nvidia`, and `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` are
  visible in the runtime.
- Full-parameter fine-tuning of the 7B model requires sharded training (FSDP)
  and a multi-GPU inference/training split with explicit weight
  synchronization. That topology should be sized on the target hardware:
  profile the sync split before committing to async overlap or colocation. It
  is the next step for this script, not covered by this configuration.
