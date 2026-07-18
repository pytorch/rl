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
checkpoints are not interchangeable with this token-head variant for GRPO
training. Use the SimpleVLA-RL SFT checkpoints, for example `Haozhan72/*`:

```python
from openvla import OpenVLAOFTWrapper

policy = OpenVLAOFTWrapper.from_pretrained(
    "Haozhan72/Openvla-oft-SFT-libero10-traj1",
    temperature=1.6,
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
`collector/trajectory_return_max`), and throughput split into collection and
optimization:

- `throughput/inference_env_steps_per_s`
- `throughput/inference_decisions_per_s`
- `throughput/train_decisions_per_s`
- `throughput/optim_steps_per_s`

LIBERO uses ``logger.service_backend=process`` so evaluator workers receive a
lightweight logger client. Evaluation video is one synchronized grid recorded
from the same ``env.eval_num_envs`` parallel rollouts used for reward and
success metrics; it does not launch an extra rollout or select a single task.

The collector path is fixed: TorchRL's `Collector` uses its process backend to
launch rollout workers, each worker owns a sync
`ParallelEnv(envs_per_collector)`, and all workers plus the evaluator share one
process policy server.

```bash
python sota-implementations/vla_grpo/vla-grpo.py --config-name vla_grpo_libero
```

The LIBERO recipe enables worker-originated environment metadata. This keeps
MuJoCo/EGL construction inside the long-lived rollout workers and avoids a
temporary parent-side environment wave. The nested startup path can be measured
without loading the VLA policy:

```bash
python sota-implementations/vla_grpo/bench_libero_startup.py \
  --mode legacy-parent --inner-start-method spawn \
  --output-dir /root/artifacts/libero-startup/legacy-spawn
python sota-implementations/vla_grpo/bench_libero_startup.py \
  --mode worker-metadata --inner-start-method spawn \
  --output-dir /root/artifacts/libero-startup/worker-spawn
python sota-implementations/vla_grpo/bench_libero_startup.py \
  --mode worker-metadata --inner-start-method forkserver \
  --output-dir /root/artifacts/libero-startup/worker-forkserver
```

Outer subcollector processes always use ``spawn``. The script permits
``forkserver`` or diagnostic ``fork`` only in worker-metadata mode, after the
subcollector parent has been kept free of EGL contexts. Each output directory
contains the exact command, per-construction marker files, and a JSON summary
covering nested environment readiness, first-step latency, peak process count,
and shutdown cleanup.

Rollout clients request random sampling; eval and video clients request
deterministic decoding from the same server, so rollout and eval are synced by
one explicit TensorDict weight update after each optimizer step. The replay
buffer is passed to the collector and receives complete trajectories as they
finish; training waits until the consuming replay buffer has enough sampleable
decisions.

With the thread evaluator backend, eval rollouts are rendered to video whenever
a logger is configured. A dedicated single-environment evaluator is built with
`from_pixels=True`: `ToyVLAEnv` renders the tracking scene, while `LiberoEnv`
exposes its camera. `torchrl.record.VideoRecorder` writes the evaluator rollout
to `eval/video` on every eval. wandb video encoding needs `moviepy` from the
`dev` dependency group. Process-backend evaluator video dumping still needs a
TorchRL-side remote `VideoRecorder.dump` path.

Checkpointing is shared by the toy and LIBERO configs. `checkpoint_latest`
is written as a TensorDict directory in the hydra run directory every
`checkpoint.save_iter` iterations;
resume with:

```bash
python sota-implementations/vla_grpo/vla-grpo.py \
  checkpoint.resume=/path/to/checkpoint_latest
```

## LIBERO configuration details

The full LIBERO config follows the SimpleVLA-RL hyper-parameter shape:

- groups of `n=8` rollouts per initial state;
- 40 initial states per iteration (`collector.groups_per_iter`), for 320
  trajectories before dynamic filtering -- one aligned group wave across the
  320 rollout envs; the paper uses 64 initial states (512 trajectories) per
  iteration, which is a known deviation of the shipped config;
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

LIBERO simulation runs through `collector.num_collectors` process-backed
collector workers. Each worker hosts a synchronous
`ParallelEnv(collector.envs_per_collector)`. Policy inference runs on the
shared process server and each worker owns a disjoint `group_id` block so
advantages never mix across unrelated groups.

Without `env.parallel_group_repeats`, groups are repeated serially within each
worker, so the total rollout worker count should not exceed
`collector.groups_per_iter`. For that serial mode, set
`collector.num_collectors * collector.envs_per_collector` to a divisor of
`collector.groups_per_iter`, often the same value.

When `env.parallel_group_repeats=true`, the shared replay buffer centralizes
`MCAdvantage` write state, so same-initial-state groups may straddle
subcollectors. The logical worker count is the total rollout worker count
divided by `collector.group_size`. In this mode, prefer setting
`collector.groups_per_iter` to that logical worker count so one target group
wave is aligned. If `collector.candidate_group_size` is larger than
`collector.group_size`, each worker in a logical group repeats the same initial
state serially enough times to produce up to the requested candidate count. For
example, 8 parallel workers x 2 serial repeats gives at most 16 candidates.
Groups can be written earlier if the candidates already contain a useful
selected subset.

The training script starts the collector once, waits until the consuming replay
buffer has enough sampleable decisions, pauses collection, runs the PPO update,
clears incomplete same-policy advantage queues and partial trajectories, pushes
the TensorDict policy weights to the shared policy server, and then lets the
collector resume.
`MCAdvantage` runs as the replay-buffer transform and keeps incomplete groups
queued only within a single policy window. `collector.min_replay_decisions` can
require a minimum number of useful replay decisions before the PPO update.
Set `TORCHRL_MC_ADVANTAGE_LOCAL_QUEUES=1` to keep grouping state in each replay
writer instead of a multiprocessing manager. At every policy boundary the
trainer reads those worker-local counters while collection is paused, clears
their queues, and resets in-flight collector trajectories before the policy
version advances.

Candidate selection is delegated to `MCAdvantageSelector` (`first`, `uniform`,
or `balanced`), so the replay-buffer transform owns the sample-selection policy
while the collector only supplies same-policy completed trajectories. The
consuming replay buffer removes sampled decisions after
`buffer.consume_after_n_samples` samples, and the policy-boundary pause keeps
rollout and optimization phases explicit. LIBERO workers stamp
parallel-repeat group ids from the cycled initial-state id so fast and slow
sibling workers can still complete a same-initial-state GRPO group under the
same policy even when their episode lengths differ.

Run the LIBERO recipe with:

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO
python sota-implementations/vla_grpo/vla-grpo.py --config-name vla_grpo_libero
# or: sbatch sota-implementations/vla_grpo/vla-grpo.sbatch
```

Requirements beyond the toy scale: LIBERO (see the `torchrl.envs.LiberoEnv`
docs for install notes), `transformers`, `timm`, `Pillow`, and `peft` when
`policy.lora_rank` is set.

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
  evaluator inference from `collector.policy_device: cuda:1`. Four
  collectors each run `ParallelEnv(80)` for 320 rollout envs total. On
  single-GPU runs, override `policy.device=null`,
  `collector.policy_device=null`, `collector.num_collectors=1`, and
  `collector.envs_per_collector` to the number of local envs.
- Rollout wall-clock dominates, so scale `collector.num_collectors` and
  `collector.envs_per_collector` with the available CPU cores while keeping
  the GRPO grouping constraint above. The H100 default uses
  `collector.num_collectors=4`, `collector.envs_per_collector=80`,
  `collector.groups_per_iter=40`, and parallel group repeats enabled. The
  training loop pushes TensorDict policy weights to the shared policy server
  after optimizer updates.
- Headless LIBERO rendering uses MuJoCo/robosuite EGL by default
  (`env.render_backend: egl`). `env.render_gpu_ids` controls the EGL-visible
  render device ids assigned to rollout workers, round-robin. The H100 default
  spreads rollout rendering over `[2,3,4,5]` and reserves
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
