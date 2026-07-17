# Off-policy data-parallel validation

This suite validates TorchRL's Ray-owned replay, asynchronous collection,
multi-rank learner execution, and direct learner-to-collector weight publication
for DQN, SAC, DDPG, and TD3.

All four algorithms use transition replay. Each inner Ray collector installs a
postprocessor through `collector_kwargs` that materializes and flattens `[B, T]`
rollouts to `[B * T]` before direct insertion. The suite asserts that replay
`write_count` equals the collector's transition count and that sampled batches
have shape `[N]`. Grouped sequence replay is not inferred from rollout shape;
sequence-aware workloads must configure and validate it explicitly.

Two asynchronous evaluators periodically load an exact learner-weight snapshot.
The metric evaluator runs termination-aware deterministic episodes without
rendering. A separate single-environment diagnostic rollout records
`evaluation/video` to W&B; for locomotion tasks it continues until the configured
video horizon after the agent becomes unhealthy, rather than producing a
one-frame video when an early policy falls immediately. Rendering is kept out
of the 100,000 training environments. Both evaluators default to
`mujoco-torch` on `cuda:7`, avoiding the CPU physics bottleneck and unavailable
headless EGL/OpenGL support.

The continuous environment is selected by `algorithm.environment_name`. The
`humanoid_smoke` and `humanoid_scale` profiles run the substantially more
complex Humanoid task while retaining flat transition replay and the same
Ray/Gloo/NCCL topology.

Collection remains stochastic after replay prefill. SAC samples its policy,
TD3 and DDPG apply persistent Gaussian action noise, and DQN retains nonzero
epsilon-greedy exploration. Evaluation always uses the policy without the DDPG
or TD3 exploration module. Replay samples log action mean, standard deviation,
absolute maximum, and saturation fraction so a deterministic or saturated
collector policy is visible during a run.

The scale profile keeps Ray as the replay owner and uses the stack's distributed
Gloo tensor transport for collector inserts and learner samples. This avoids
serializing 25,000-transition GPU-collector batches through Ray pickle while
retaining the same replay service and ownership topology.

The continuous scale profile uses four GPU collector actors. Each actor owns a
compiled `mujoco-torch` Hopper environment with 25,000 parallel environments.
Four additional GPU actors form the NCCL learner group, giving 100,000 total
environments and four learner ranks on an eight-GPU node.

The learning-scale profiles collect 200 million transitions, or 2,000 policy
decisions per environment. They do not use a separate random-action phase.
Instead, the stochastic collection policy fills replay with ten million
transitions (100 decisions per environment) before optimization starts. This
separates learner startup from action selection and avoids distributed random
warmup accounting against a shared replay write counter. Requiring a full
1,000-step trajectory from every environment would delay learning until 100
million transitions, so the prefill deliberately targets early terminations
while persistent exploration continues throughout training.

Scale replay retains the most recent 20 million transitions. A global batch of
16,384 gives each learner rank 4,096 samples, and 25 optimizer steps per roughly
100,000 collected frames preserve a sample update-to-data ratio of 4.096 while
reducing small-batch synchronization overhead. Metric evaluation uses 64
episodes of up to 1,000 steps every ten million frames; video is recorded every
20 million frames.

## Installation

Install the experiment-only dependencies into the active environment:

```bash
uv pip install --python /root/venv/bin/python \
  -r sota-implementations/offpolicy-dp/requirements.txt
```

Configure W&B before launching. Credentials are intentionally not read by the
scripts and must never be stored in this directory.

## Individual runs

Run the planned 100,000-frame DQN validation:

```bash
python sota-implementations/offpolicy-dp/train.py \
  algorithm=dqn profile=dqn
```

Run a reduced SAC smoke test:

```bash
python sota-implementations/offpolicy-dp/train.py \
  algorithm=sac profile=smoke_continuous
```

Run a 100,000-environment, 200-million-transition continuous experiment by
choosing `sac`, `ddpg`, or `td3`:

```bash
python sota-implementations/offpolicy-dp/train.py \
  algorithm=sac profile=scale
```

Run rendered Humanoid DDPG validation:

```bash
python sota-implementations/offpolicy-dp/train.py \
  algorithm=ddpg profile=humanoid_smoke
python sota-implementations/offpolicy-dp/train.py \
  algorithm=ddpg profile=humanoid_scale
```

The Humanoid profiles use `frame_skip=1` so health termination is checked after
every physics step. This prevents an unstable state in a 25,000-environment
collector batch from advancing through a five-substep action before reset. The
scale run is lengthened to 2,000 decisions per environment to compensate for
the finer control interval; it is not directly comparable to the standard
five-substep Humanoid benchmark.

Hydra overrides can reduce or expand any resource or training setting. Runtime
summaries and Hydra output are written under `/root/artifacts/offpolicy-dp`.

## Suite launcher

The launcher uses a fresh Python process for every algorithm so Ray resources
are released between runs. It records all failures instead of stopping at the
first one.

```bash
python sota-implementations/offpolicy-dp/run_suite.py --mode smoke
python sota-implementations/offpolicy-dp/run_suite.py --mode full
```

The `all` mode runs reduced smokes first and then the planned full runs.

## Success checks

Every run fails if optimization never starts, replay writes stop early, a
metric becomes non-finite, the learner's published model version differs from
its optimization count, or sampled replay data never observes an updated
policy version. W&B additionally records collection and optimization
throughput, replay activity, policy-version statistics, and terminal returns
sampled from replay. Continuous runs also record replay action dispersion and
saturation to verify that the collection policy remains stochastic.

Learner metrics are logged with slash-delimited namespaces. W&B therefore
places losses under `loss`, predicted and target values under `value`, gradient
and step metrics under `optimization`, and entropy/temperature under `policy`,
rather than leaving these histories in the generic Charts section.
