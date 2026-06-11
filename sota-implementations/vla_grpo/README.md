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

## Scaling up

The LIBERO / OpenVLA-OFT (7B) configuration of the same recipe lands in
follow-up versions of this script (LIBERO env adapter, vendored token-OFT
policy, multi-GPU topology).
