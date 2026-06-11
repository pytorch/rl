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

Per-iteration accounting: `groups_per_iter` initial states x `group_size`
rollouts, each contributing up to `max_outer_steps` decisions; the dynamic
sampling filter makes the effective optimization batch variable.

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

```bash
export MUJOCO_GL=egl ROBOT_PLATFORM=LIBERO
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
  `env.num_envs` with the available cores first.
- Full-parameter fine-tuning of the 7B model requires sharded training
  (FSDP) and a multi-GPU inference/training split with explicit weight
  synchronization. That topology should be sized on the target hardware
  (profile the sync split before committing to async overlap or
  colocation); it is the next step for this script, not covered by this
  configuration.

Before any RL run, validate the checkpoint loading path: evaluate the SFT
checkpoint greedily (50 cycled trials/task) and compare against the
SimpleVLA-RL paper's SFT numbers (see the policy section above).
