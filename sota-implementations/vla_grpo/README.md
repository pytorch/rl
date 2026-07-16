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

The directory contains two versions of the same idea, plus
`compare_simplevla_rollouts.py`, a standalone script that rolls the same
checkpoint through both the TorchRL stack and the SimpleVLA-RL/VeRL reference
rollout and writes a JSON parity report.

### Toy scale: learn the algorithm without a robot simulator

The opt-in `config/vla_grpo_toy.yaml` config trains `TinyVLA` on
`ToyVLAEnv`. This is the quickest way to understand and test the stack; the
script default is the at-scale LIBERO recipe described below.

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
python sota-implementations/vla_grpo/vla-grpo.py --config-name vla_grpo_toy
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
  -> StepCounter(max_outer_steps)       truncate in policy decisions
```

LIBERO emits `reward=float(success)` from the base environment step. When a
chunk terminates early, `MultiAction(stack_rewards=False)` keeps the reward from
the last action that actually executed, so the outer transition preserves the
terminal success reward instead of reporting a zero from a skipped trailing
action slot.

`MCAdvantage` sits on the replay-buffer path. It consumes `("next", "reward")`,
receives completed trajectories, groups trajectories that started from the same
initial state, and computes a Monte-Carlo advantage from the group-normalized
return. With `trajectory_return="sum"`, the trajectory-level success return is
broadcast to every chunk decision in that trajectory, which is the shape PPO
needs.

The loss is `ClipPPOLoss` configured like a GRPO policy update: no critic, zero
entropy coefficient, no KL-to-reference term, and asymmetric DAPO Clip-Higher
bounds. Entropy is still reported as a diagnostic. For token policies the ratio
can be computed per token, which matches the SimpleVLA-RL semantics used by the
OpenVLA configuration.

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
checkpoints are not interchangeable with this token-head variant for GRPO
training. Use the SimpleVLA-RL SFT checkpoints, for example `Haozhan72/*`:

```python
from openvla import OpenVLAOFTWrapper

policy = OpenVLAOFTWrapper.from_pretrained(
    "Haozhan72/Openvla-oft-SFT-libero-spatial-trajall",
    temperature=1.0,
    device="cuda",
)
tokenizer = policy.action_tokenizer  # decode tokens -> env actions
```

The wrapper also has an explicit `policy.mode=l1` path for the official
continuous OpenVLA-OFT reference checkpoints. That mode loads the released
`action_head--150000_checkpoint.pt` and
`proprio_projector--150000_checkpoint.pt` components, uses two images
(`agentview` plus wrist) and the 8-D OpenVLA proprio vector, and writes a
continuous `("vla_action", "chunk")`. It is meant to validate the environment,
image preprocessing, proprio normalization, and evaluator/collector path
against the supervised reference policy:

```text
policy.mode=l1
policy.checkpoint=moojink/openvla-7b-oft-finetuned-libero-spatial
policy.use_wrist_image=true
policy.use_proprio=true
policy.num_images_in_input=2
policy.lora_rank=0
```

The PPO update still expects token log-probabilities, so this mode is for
reference/evaluation probes until a continuous-action GRPO loss is added.

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
logs the training success rate (`train/success_rate`), trajectory-return
aggregates (`collector/trajectory_return_sum`,
`collector/trajectory_return_max`), frozen and periodic greedy evaluation
success (`eval/success_rate`), PPO diagnostics (`train/loss_objective`,
`train/kl_approx`, `train/mean_ratio`, `train/clip_fraction`, `train/ESS`,
`train/entropy`), group filtering (`buffer/kept_groups`,
`buffer/skipped_groups`, `buffer/rescued_groups`,
`buffer/queued_trajectories`), and throughput split into collection and
optimization:

- `throughput/inference_env_steps_per_s`
- `throughput/inference_decisions_per_s`
- `throughput/train_decisions_per_s`
- `throughput/optim_steps_per_s`

LIBERO uses ``logger.service_backend=process`` so evaluator workers receive a
lightweight logger client. Scalar evaluation and video recording are split:
the scalar evaluator runs the full `logger.eval_episodes` sweep without pixels,
while `logger.record_video=true` launches a separate bounded visual evaluator.
By default it records `logger.video_episodes=1` trajectory with
`logger.video_num_envs=1`; set `logger.video_episodes` to an integer count or a
fraction in `(0, 1)` to record a larger sample without making every scalar eval
rollout video-capable.

The collector path is fixed: a TorchRL `MultiCollector` launches rollout
workers, each worker owns a sync `ParallelEnv(envs_per_collector)`, and all
workers plus the evaluator share one process policy server. The rollout
builder receives a policy factory rather than the learner model; only the
inference-server child invokes that factory. The main process owns the distinct
trainable learner policy required by the PPO loss and optimizer.

Each nested `ParallelEnv` is built from one homogeneous environment factory and
per-worker `create_env_kwargs`. Worker-specific task ids, init-state groups,
language instructions, seeds, and render-device assignments still come from the
real worker index, but metadata collection only needs the homogeneous schema
instead of constructing every LIBERO worker once in the parent and again in the
subprocesses.
The LIBERO recipe also enables `env.metadata_from_workers=true`, so the
metadata schema comes from the real subprocess environments and no parent-side
temporary MuJoCo/EGL environment is needed for those rollout and eval pools.

```bash
python sota-implementations/vla_grpo/vla-grpo.py
```

Rollout clients request random sampling; eval and video clients request
deterministic decoding from the same server, so rollout and eval are synced by
one explicit TensorDict weight update after each optimizer step. The replay
buffer is passed to the collector and receives complete trajectories as they
finish; training waits for the configured number of complete, non-degenerate
GRPO groups. An optional decision-count floor can be added, but it is not used
as a proxy for complete groups.

The environment-side OpenVLA action tokenizer is constructed directly from
`dataset_statistics.json`. It therefore does not require access to either the
learner model or the inference-server model.

The at-scale LIBERO config is set up for a single eight-GPU node: GPU 0 holds
the learner, GPU 1 holds the shared inference server, GPUs 2-6 render rollout
workers, and GPU 7 renders evaluation/video. It uses `candidate_group_size=32`,
`candidate_selection_min_size=32`, and balanced selection to collect more
rollouts than PPO consumes, while dropping all-failure and all-success groups
from optimization because they have zero group-relative advantage.

With `logger.record_video=true`, the bounded video evaluator is rendered through
the configured evaluator backend with `from_pixels=True`: `ToyVLAEnv` renders
the tracking scene, while `LiberoEnv` exposes its camera.
`torchrl.record.VideoRecorder` writes the sampled visual rollout to
`eval/video` on every scheduled eval. wandb video encoding needs `moviepy` from
the `dev` dependency group.

Checkpointing is shared by the toy and LIBERO configs. `checkpoint_latest`
is written as a TensorDict directory in the hydra run directory every
`checkpoint.save_iter` iterations. It includes the optimizer iteration and
completed-trajectory count so a resumed paper-budget run stops correctly;
resume with:

```bash
python sota-implementations/vla_grpo/vla-grpo.py \
  checkpoint.resume=/path/to/checkpoint_latest
```

## LIBERO configuration details

### Canonical execution recipe

[`recipe.toml`](recipe.toml) describes the complete LIBERO execution
environment independently of any particular scheduler. It declares the
TorchRL extras, compatible LIBERO source revision and Python packages,
headless EGL requirement, environment variables, resource shape, optional
demonstration data, persistent cache categories, entrypoint, and
[`smoke.py`](smoke.py) readiness check.

A compatible executor must not report the recipe as ready until the smoke
check has verified CUDA, the TensorFlow-free OpenVLA image preprocessor, and
an actual headless `LiberoEnv.reset()` call. The LIBERO demonstration dataset
is intentionally optional because the online GRPO configuration uses the
simulator tasks and assets but does not consume demonstrations.

The default LIBERO config follows the SimpleVLA-RL hyper-parameter shape:

- groups of `n=8` rollouts per initial state;
- 64 complete useful initial-state groups per update
  (`collector.groups_per_iter`), for 512 selected trajectories;
- 700 optimizer iterations; with 512 selected trajectories per update, this
  matches the paper-scale selected-trajectory budget while allowing discarded
  oversampled candidates not to shorten training;
- 512 base environment steps, or 64 chunk decisions, per episode;
- rollout temperature 1.0 and greedy evaluation;
- 500 suite-wide greedy evaluation rollouts over cycled initial states,
  including an asynchronous frozen-policy baseline launched before the first
  update so collection and optimization are not blocked by the full eval sweep;
- dynamic sampling bounds `(0.1, 0.9)` to drop groups that are all failure or
  all success;
- learning rate `5e-6` with constant-with-warmup scheduling;
- optimizer batches of 128 trajectories, with model forwards limited to 16
  decisions and gradient clipping at 1.0;
- asymmetric clip `(0.2, 0.28)` applied to per-token importance ratios by
  default (`loss.ratio_level: token`).

The optimizer reduction is trajectory-aware: token losses are averaged into
one loss per decision, decisions are averaged within each trajectory, and
trajectories are averaged within the optimizer batch. Thus short successful
trajectories and long failed trajectories have equal trajectory weight, and
`loss.mini_batch_size` changes memory use without changing this objective.

A sequence-level ratio remains available as a config switch for ablations, but
for a 56-token action chunk it saturates the clip range much more easily than
per-token ratios.

LIBERO simulation runs through `collector.num_collectors` MultiCollector
workers. Each worker hosts a synchronous
`ParallelEnv(collector.envs_per_collector)`. Policy inference runs on the
shared process server and each worker owns a disjoint `group_id` block so
advantages never mix across unrelated groups.

Without `env.parallel_group_repeats`, groups are repeated serially within each
worker. The total rollout worker count controls collection concurrency, while
`collector.groups_per_iter` independently controls optimizer-update cadence.

When `env.parallel_group_repeats=true`, the shared replay buffer centralizes
`MCAdvantage` write state, so same-initial-state groups may straddle
subcollectors. The logical worker count is the total rollout worker count
divided by `collector.group_size`, but `collector.groups_per_iter` need not be a
multiple of it: the setting is only the minimum number of complete useful
groups that triggers an optimizer update. If `collector.candidate_group_size`
is larger than `collector.group_size`, each worker in a logical group repeats
the same initial state serially enough times to produce up to the requested
candidate count. For example, 8 parallel workers x 2 serial repeats gives at
most 16 candidates. Groups can be written earlier if the candidates already
contain a useful selected subset.

The training script evaluates the frozen SFT policy at step 0, starts the
collector once, waits for the target number of complete kept groups, pauses
collection, runs the PPO update, pushes the TensorDict policy weights to the
shared policy server, and then lets the collector resume. Incomplete
`MCAdvantage` groups and in-flight environment trajectories are preserved
across that boundary. Each inference response carries its actual
`policy_version`; PPO uses the corresponding stored action log-probability,
while the trainer reports decision staleness and policy-version spans within
trajectories and GRPO groups.

`collector.min_replay_decisions` can require a minimum number of useful replay
decisions before the PPO update. Set `TORCHRL_MC_ADVANTAGE_LOCAL_QUEUES=1` to
keep grouping state in each replay writer instead of a multiprocessing manager.
At every policy boundary the trainer reads and resets worker-local accounting
counters while collection is paused, without clearing queues or resetting
in-flight trajectories, before the policy version advances.

Candidate selection is delegated to `MCAdvantageSelector` (`first`, `uniform`,
or `balanced`), so the replay-buffer transform owns the sample-selection policy
while the collector supplies completed trajectories with per-decision behavior
policy metadata. The consuming replay buffer removes sampled decisions after
`buffer.consume_after_n_samples` samples, and the policy-boundary pause keeps
rollout and optimization phases explicit. LIBERO workers stamp
parallel-repeat group ids from the cycled initial-state id so fast and slow
sibling workers can still complete a same-initial-state GRPO group even when
their episode lengths cross a policy update.

Run the LIBERO recipe with:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO
python sota-implementations/vla_grpo/vla-grpo.py --config-name vla_grpo_libero
# or: sbatch sota-implementations/vla_grpo/vla-grpo.sbatch
```

Requirements beyond the toy scale: LIBERO (see the `torchrl.envs.LiberoEnv`
docs for install notes), `transformers`, `timm`, `Pillow`, and `peft` when
`policy.lora_rank` is set. The VLA extra pins the OpenVLA-compatible
`transformers`, `tokenizers`, `timm`, and `peft` versions because their newer
APIs are not backward-compatible with the vendored OpenVLA-OFT model.

Install the Python requirements with the TorchRL VLA extra:

```bash
pip install -e '.[vla]'
```

The default `policy.image_backend=torch_reference` uses torchvision's JPEG
codec and matches SimpleVLA's JPEG-before-resize order, antialiased Lanczos3
resize, and fractional center-crop interpolation semantics without requiring
TensorFlow. Set `policy.image_backend=torchvision` for the faster bicubic path,
or `policy.image_backend=tensorflow` when the TensorFlow codec itself is
required for exact reference comparisons. Normalized vocabulary-tail action
tokens are detokenized through the NumPy float64 CPU path before the gripper
transform is applied once in the environment. Use
`env.train_init_state_mode=fixed env.train_init_state_id=<id>` for a fixed
LIBERO initial state. `collector.policy_micro_batch_size` only slices actual
model calls inside the inference-server policy; it does not change PPO
minibatching. `compare_simplevla_rollouts.py` automates the parity check: it
evaluates the same `(task_id, init-state id)` trajectories through the TorchRL
stack and the SimpleVLA-RL/VeRL reference rollout side by side and reports
per-trajectory token/action/success agreement.

## Hardware notes

- The default H100 configuration trains a LoRA adapter
  (`policy.lora_rank: 32`) on `policy.device: cuda:0` and serves rollout plus
  evaluator inference from `collector.policy_device: cuda:1`. Five
  collectors each run `ParallelEnv(64)` for 320 rollout envs total. They
  asynchronously fill the 64-group/512-selected-trajectory update batch. On
  single-GPU runs, override `policy.device=null`,
  `collector.policy_device=null`, `collector.num_collectors=1`, and
  `collector.envs_per_collector` to the number of local envs.
- Rollout wall-clock dominates, so scale `collector.num_collectors` and
  `collector.envs_per_collector` with the available CPU cores while keeping
  the GRPO grouping constraint above. The H100 default uses
  `collector.num_collectors=5`, `collector.envs_per_collector=64`,
  `collector.groups_per_iter=64`, and parallel group repeats enabled. The
  training loop pushes TensorDict policy weights to the shared policy server
  after optimizer updates.
- Headless LIBERO rendering uses MuJoCo/robosuite EGL by default
  (`env.render_backend: egl`). `env.render_gpu_ids` controls the EGL-visible
  render device ids assigned to rollout workers, round-robin. The H100 default
  spreads rollout rendering over `[2,3,4,5,6]` and reserves
  `env.eval_render_gpu_ids=[7]` for eval/video rendering. These ids are the
  devices visible to EGL inside the process/container and may not match global
  CUDA ordinals.
- Use `logger.eval_backend` for the TorchRL evaluator backend. The evaluator
  shares the same policy server as rollout and uses `env.eval_render_gpu_ids`
  for EGL rendering; when the latter is left null, eval reuses
  `env.render_gpu_ids`. The LIBERO default uses `process` to isolate simulator
  work. ``Evaluator`` dispatches ``env.transform.dump(step=...)`` through the
  collector RPC path, so ``VideoRecorder`` works for both thread and process
  evaluation without a separate recorder environment.
- Minimal CUDA containers often lack the NVIDIA EGL/GLVND userspace stack.
  Before debugging TorchRL, verify that `libEGL_nvidia`, `libnvidia-eglcore`,
  `libGLX_nvidia`, and `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` are
  visible in the runtime.
- On an H200 container with the 595 driver, the userspace libraries can be
  extracted without installing Debian packages into the image:

  ```bash
  mkdir -p /opt/nvidia-595-deb/download /opt/nvidia-595-deb/extract
  cd /opt/nvidia-595-deb/download
  apt-get download \
    libnvidia-gl-595 libegl1 libglvnd0 libopengl0 libgl1 libgles2 libglx0
  for deb in ./*.deb; do
    dpkg-deb -x "$deb" /opt/nvidia-595-deb/extract
  done

  export LIBDIR=/opt/nvidia-595-deb/extract/usr/lib/x86_64-linux-gnu
  export LD_LIBRARY_PATH="$LIBDIR:${LD_LIBRARY_PATH:-}"
  export LD_PRELOAD="$LIBDIR/libOpenGL.so.0${LD_PRELOAD:+:$LD_PRELOAD}"
  export __EGL_VENDOR_LIBRARY_FILENAMES=/opt/nvidia-595-deb/extract/usr/share/glvnd/egl_vendor.d/10_nvidia.json
  export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO

  python - <<'PY'
  from OpenGL import EGL, GL
  import mujoco
  from libero.libero import benchmark

  assert EGL is not None and GL.glGetError is not None
  assert mujoco is not None and benchmark is not None
  PY
  ```

  The validated parity runtime pins `mujoco==3.2.3`, `robosuite==1.4.1`,
  `transformers==4.40.1`, and `peft==0.11.1`.
- Full-parameter fine-tuning of the 7B model requires sharded training (FSDP)
  and a multi-GPU inference/training split with explicit weight
  synchronization. That topology should be sized on the target hardware:
  profile the sync split before committing to async overlap or colocation. It
  is the next step for this script, not covered by this configuration.
