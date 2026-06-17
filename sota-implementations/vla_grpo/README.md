# VLA GRPO: RL fine-tuning of token-head VLA policies

GRPO-style reinforcement fine-tuning of a Vision-Language-Action policy with
a discrete action-token head, following the SimpleVLA-RL recipe
([arXiv:2509.09674](https://arxiv.org/abs/2509.09674)):

- the policy emits a whole action chunk (one token per action dimension and
  chunk step) in a single forward;
- training rollouts are grouped: each initial state is replayed `group_size`
  times and the advantage is the group-normalized binary success return,
  broadcast to every chunk decision of the trajectory
  (`MCAdvantage(trajectory_return="sum")`);
- groups whose rollouts all failed or all succeeded carry no learning signal
  and are dropped (dynamic sampling, `keep_return_bounds`);
- the objective is `ClipPPOLoss` with no critic, no entropy bonus, no
  KL-to-reference, and an asymmetric (DAPO Clip-Higher) clip threshold.

## The chunk-decision data path

One outer step of the transformed environment is one policy decision:

```text
policy -> action_tokens [B, H, A]           (one forward per chunk)
  ActionTokenizerTransform (inverse)        decode tokens -> continuous chunk
  MultiAction(stack_rewards=False)          unbind: H base-env steps
  SuccessReward                             decision reward = success flag
  StepCounter(max_outer_steps)              truncation in decisions
```

Per-iteration accounting: `groups_per_iter` initial states x
`candidate_group_size` candidate rollouts, each contributing up to
`max_outer_steps` decisions. When `candidate_group_size` is unset it defaults
to `group_size`; when it is larger, `MCAdvantage` treats it as the maximum
number of candidates to try. It starts selecting once `group_size` trajectories
are available, writes the group as soon as a useful subset exists, and keeps
queueing candidates up to `candidate_group_size` only if no useful subset has
appeared yet. The dynamic sampling filter makes the effective optimization
batch variable.

## Toy scale (this version)

`vla-grpo.py` with `config/vla_grpo_toy.yaml` trains `TinyVLA` on the
`ToyVLAEnv` tracking task (a per-episode target action is exposed in the
state; success requires staying within tolerance of it for a few consecutive
steps). The task is sized so a random policy succeeds occasionally (so the
group advantage has signal from the start) while an oracle solves it exactly.
Single process, single device, runs on CPU or one small GPU:

```bash
python sota-implementations/vla_grpo/vla-grpo.py
```

The greedy evaluation success rate should climb from zero (a randomly
initialized argmax policy never solves the task) to roughly 0.4-0.6 within
~200 iterations (about two minutes on CPU); the sampled training success
rate starts at the random baseline (~0.25) and climbs alongside it.

Checkpointing: `checkpoint_latest.pt` is written to the hydra run directory
every `checkpoint.save_iter` iterations; resume with
`checkpoint.resume=/path/to/checkpoint_latest.pt`.

## Logging: metrics and eval videos

With a logger configured (`logger.backend=wandb`, the default), each iteration
logs the reward curves `train/reward_mean` and `train/reward_max` (per-episode
return, averaged and best-of-batch), the success rate, and throughput split
into inference (`throughput/inference_env_steps_per_s`,
`throughput/inference_decisions_per_s`, measured over collection) and training
(`throughput/train_decisions_per_s`, `throughput/optim_steps_per_s`, measured
over the PPO update).

Eval rollouts are also rendered to video (`logger.record_video=true`, on by
default; a no-op without a logger). A dedicated single-environment recorder is
built with `from_pixels=True` -- `ToyVLAEnv` renders the tracking scene (the
executed action and the target as markers), `LiberoEnv` exposes its camera --
and a `torchrl.record.VideoRecorder` writes `logger.video_episodes` greedy
episodes to `eval/video` on every eval. wandb video encoding needs `moviepy`
(in the `dev` dependency group). Disable with `logger.record_video=false`.

## The OpenVLA-OFT (token) policy

`openvla.py` wraps the SimpleVLA-RL token variant of OpenVLA-OFT (parallel
decoding and action chunking, with the continuous L1 head reverted to
discrete action tokens) as a `VLAWrapperBase`: one forward emits the whole
56-token chunk, sampled from the 256-way categorical over the action-token
window at the tail of the LLaMA-2 vocabulary. The modeling code is vendored
verbatim from [SimpleVLA-RL](https://github.com/PRIME-RL/SimpleVLA-RL) (MIT)
under `openvla_oft/` -- the official OpenVLA-OFT checkpoints are
*incompatible* with this variant; use the SimpleVLA-RL SFT checkpoints
(HF: `Haozhan72/*`):

```python
from openvla import OpenVLAOFTWrapper

policy = OpenVLAOFTWrapper.from_pretrained(
    "Haozhan72/Openvla-oft-SFT-libero10-traj1",
    temperature=1.6,
    device="cuda",
)
tokenizer = policy.action_tokenizer()  # decode tokens -> env actions
```

The wrapper owns all model-side preprocessing (prompt construction, image
transforms) and applies the temperature identically when sampling and when
recomputing log-probabilities at loss time, so the PPO importance ratio is
exactly 1 with identical weights -- `test_openvla.py` pins this contract on
a tiny random-weight model of the same token layout (no checkpoint
download; requires `transformers`, `timm`, `Pillow`):

```bash
pytest sota-implementations/vla_grpo/test_openvla.py
```

Before spending RL compute on a checkpoint, validate the loading path by
evaluating the SFT checkpoint greedily on its LIBERO suite through
`torchrl.envs.LiberoEnv` (`init_state_mode="cycle"`, 50 trials/task) and
comparing to the SimpleVLA-RL paper's SFT numbers (e.g. LIBERO-Spatial
one-trajectory SFT: ~63.6%, +-3pts).

## LIBERO scale

`config/vla_grpo_libero.yaml` carries the full SimpleVLA-RL hyper-parameter
set: groups of n=8 rollouts over 64 initial states per iteration (512
trajectories), 512 env steps / 64 chunk decisions per episode, asymmetric
clip (0.2, 0.28) applied to *per-token* importance ratios
(`loss.ratio_level: token`, the paper's semantics -- a single summed ratio
over the 56-token chunk saturates these bounds almost immediately; the
sequence-level variant remains available as a config switch for ablations),
rollout temperature 1.6 with greedy evaluation, dynamic sampling bounds
(0.1, 0.9), LR 5e-6 with constant-with-warmup scheduling, gradient
accumulation, gradient clip 1.0, evaluation every 4 iterations on cycled
initial states (one cycle slot per counted trial). LIBERO simulation runs in parallel worker processes
(`env.num_envs`, one MuJoCo instance each, batched policy forwards across
workers); each worker owns a disjoint `group_id` block so group advantages
never mix across workers.
Because groups are repeated serially within each worker, `env.num_envs` should
not exceed `collector.groups_per_iter`: otherwise many same-policy collection
polls are needed before each worker can finish all `group_size` rollouts for a
group and the replay buffer receives advantaged decisions. For best throughput,
set `env.num_envs` to a divisor of `collector.groups_per_iter` (often the same
value), so every worker contributes complete groups regularly.

When `env.parallel_group_repeats=true`, `env.num_envs / collector.group_size`
logical workers each run one repeated-initial-state group in parallel. In this
mode, prefer setting `collector.groups_per_iter` to that logical worker count
so one target group wave is aligned. If `collector.candidate_group_size` is
larger than `collector.group_size`, each worker in a logical group repeats the
same initial state serially enough times to produce up to the requested
candidate count (for example, 8 parallel workers x 2 serial repeats = at most
16 candidates). Groups can be written earlier if the candidates already contain
a useful selected subset.
The replay-buffer writer polls the
collector at one outer step per worker, so complete trajectories are handed to
the replay buffer shortly after they finish instead of waiting for a full
max-length rollout from every worker. `MCAdvantage` runs as the replay-buffer
transform and keeps incomplete groups queued across same-policy polls until all
siblings arrive. `max_collect_batches_per_iter` sets the safety cap in target
group waves, and `collector.min_replay_decisions` can require a minimum number
of useful replay decisions before the PPO update.
Candidate selection is delegated to `MCAdvantageSelector` (`first`, `uniform`,
or `balanced`), so the replay-buffer transform owns the sample-selection policy
while the collector only supplies same-policy completed trajectories.
At the policy-update boundary the replay buffer, incomplete advantage queues,
and in-flight collector trajectories are cleared before the next policy is
rolled out. LIBERO workers stamp parallel-repeat group ids from the cycled
initial-state id so fast and slow sibling workers can still complete a
same-initial-state GRPO group under the same policy even when their episode
lengths differ.

```bash
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl ROBOT_PLATFORM=LIBERO
python sota-implementations/vla_grpo/vla-grpo.py --config-name vla_grpo_libero
# or: sbatch sota-implementations/vla_grpo/vla-grpo.sbatch
```

Requirements beyond the toy scale: LIBERO (see the
`torchrl.envs.LiberoEnv` docs for install notes), `transformers`, `timm`,
`Pillow`, and `peft` when `policy.lora_rank` is set.

Hardware notes:

- The default configuration trains a LoRA adapter (`policy.lora_rank: 32`,
  the RL4VLA-validated de-risk path) on a single GPU while the simulation
  workers occupy CPU cores. Rollout wall-clock dominates: scale
  `env.num_envs` with the available cores first, but keep it within the GRPO
  grouping constraint above.
- Set `collector.policy_device` to a different CUDA device to keep rollout
  inference on a separate policy replica. The training loop copies only the
  trainable state dict after optimizer updates, so this split is intended for
  LoRA/adapters rather than full-parameter fine-tuning.
- Headless LIBERO rendering uses MuJoCo/robosuite EGL by default
  (`env.render_backend: egl`). `env.render_gpu_ids` controls the EGL-visible
  render device ids assigned to workers, round-robin. The default `[0]` works
  on a single-GPU allocation; on a multi-GPU node, override it, for example
  `env.render_gpu_ids=[0,1,2,3]`, to spread render workers across GPUs.
  These ids are the devices visible to EGL inside the process/container and
  may not match global CUDA ordinals.
- Set `logger.eval_process=true` to move greedy eval into a dedicated
  process. Use `logger.eval_device` for its policy device and
  `env.eval_render_gpu_ids` for its EGL render workers; when the latter is
  left null, eval reuses `env.render_gpu_ids`.
- Minimal CUDA containers often lack the NVIDIA EGL/GLVND userspace stack.
  Before debugging TorchRL, verify that `libEGL_nvidia`,
  `libnvidia-eglcore`, `libGLX_nvidia`, and
  `/usr/share/glvnd/egl_vendor.d/10_nvidia.json` are visible in the runtime.
- Full-parameter fine-tuning of the 7B model requires sharded training
  (FSDP) and a multi-GPU inference/training split with explicit weight
  synchronization. That topology should be sized on the target hardware
  (profile the sync split before committing to async overlap or
  colocation); it is the next step for this script, not covered by this
  configuration.

Before any RL run, validate the checkpoint loading path: evaluate the SFT
checkpoint greedily (50 cycled trials/task) and compare against the
SimpleVLA-RL paper's SFT numbers (see the policy section above).
