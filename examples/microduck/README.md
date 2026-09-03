# MicroDuck examples

[MicroDuck](https://github.com/pollen-robotics/microduck) is a small
open-hardware biped from Pollen Robotics. Its walking model and meshes live in
[`microduck_rl`](https://github.com/pollen-robotics/microduck_rl) and are not
vendored in TorchRL. Pass `env.download=true` to the PPO script, `--download`
to the gait script (or `MicroDuckEnv(download=True)`) to fetch a pinned commit
into `~/.cache/torchrl/microduck`, or point the scripts at an existing checkout
with `env.microduck_root=/path/to/microduck_rl` (`--microduck-root` for the
gait script) or `export MICRODUCK_RL_ROOT=...`. Without any of these the env
raises an error listing the options.

| File | What it does | Needs |
| --- | --- | --- |
| [`ppo_mujoco.py`](ppo_mujoco.py) | Recurrent PPO on `torchrl.envs.MicroDuckEnv`, configured by Hydra from [`config.yaml`](config.yaml); native MuJoCo, MJX or `mujoco-torch` | `mujoco`, the `utils` extra (+ `mujoco-mjx`/`jax` or `mujoco-torch`) |
| [`heuristic_gait.py`](heuristic_gait.py) | Closed-form walking gait as a TensorDict policy, contact-based gait metrics, `rlrender` policy | `mujoco` |
| [`ppo_mjlab.py`](ppo_mjlab.py) | PPO on the upstream `Mjlab-Velocity-Flat-MicroDuck` task through `MJLabWrapper` | MJLab, `mjlab_microduck`, CUDA |

## The task

`MicroDuckEnv` is a commanded longitudinal-velocity task written once against
`MujocoEnv`, so `backend="mujoco"`, `"mjx"` and `"mujoco-torch"` share the
observation, action, reward and termination definitions. The 14 actions are
normalized offsets around the `STAND` keyframe targets, applied at 50 Hz. The
53-value observation holds projected gravity, body angular velocity, measured
and commanded body-frame forward velocity, joint errors and velocities, a
fixed-frequency gait clock and the previous action; the command is also
exposed under `commanded_x_velocity`.

The reward follows the mjlab velocity-task recipe, with every term weighted
per second and multiplied by the control period: Gaussian tracking of the
commanded velocity (lateral velocity tracked to zero) and of a zero yaw rate,
a Gaussian uprightness term, a nominal-pose term that is tight when standing
and loose when walking, and contact-based gait terms that are active under a
nonzero command: foot air time inside a 0.125 to 0.3 s swing window,
swing-foot height toward a 2 cm clearance target, agreement between foot
contacts and the gait clock, and a penalty for keeping both feet planted.
Small costs act on vertical and roll/pitch base motion, joint velocity and
action rate; a fall costs a fixed penalty and terminates the episode. Weights
are class attributes on `MicroDuckEnv`.

The gait terms and the 0.1 m/s tracking std exist because of a failure mode
seen in training: with a looser tracking term and no double-support penalty,
a from-scratch policy learned to stand perfectly still under every command
within 600k transitions (full survival, 3 mm displacement, tracking error
equal to the command) and never left that optimum. Standing under a nonzero
command must earn clearly less than stepping.
`command_range` samples the command from an interval and
`warm_start_velocity` launches a fraction of resets already moving forward,
so an untrained policy sees locomotion states early.

The upstream walking MJCF reuses detailed render meshes as collision geoms.
Their convex-hull edge pairs make the accelerated backends run out of memory,
so the env swaps collision-class meshes for tight box proxies at load time
(`low_cost_collisions=True`). Visual meshes and the checkout are untouched.
Contact-based gait metrics are available through `env.foot_contacts()` and
`env.foot_heights()` on every backend.

## Closed-form gait

`MicroDuckGaitActor` in `heuristic_gait.py` is a `TensorDictModuleBase` that
combines a bilateral phase oscillator with hip and ankle pitch feedback,
reading only the env observation. The same module drives the baseline through
`env.rollout(steps, actor)`, seeds the PPO actor and serves as the `rlrender`
policy. `gait_metrics` judges a rollout of an env built with
`diagnostics=True` from foot contacts: it counts as walking only when both
feet alternate swing phases in single support, torso pitch stays bounded and
the mean forward speed points along the command. Forward motion alone is not
accepted, since a planted-foot controller can move forward by pitching.

```bash
uv run --with mujoco python examples/microduck/heuristic_gait.py \
  --microduck-root "$MICRODUCK_RL_ROOT" --num-seeds 20 \
  --render-checkpoint microduck_gait.pt
```

## Recurrent PPO

The policy is a GRU backbone shared by the actor and the critic. With
`policy.head=gaussian` the actor is a plain Gaussian head trained from
scratch; with `policy.head=gait-residual` it adds a bounded, zero-initialized
residual to the closed-form gait, so training starts from a walking
controller. Data flows through standard TorchRL components:

1. a `Collector` with `trajs_per_batch=1` writes every finished episode as a
   whole, unpadded sequence into a `TensorDictReplayBuffer`;
2. GAE runs once over the buffer in recurrent mode;
3. `SliceSampler` draws whole episodes for the PPO minibatches;
4. the buffer is emptied and in-flight episodes dropped before collecting
   again with the updated policy.

`KLAdaptiveLR` keeps the mean policy KL near `ppo.target_kl`. Every
`evaluation.interval` iterations one `torchrl.collectors.Evaluator` per
velocity command runs `evaluation.num_episodes` deterministic episodes, and the
checkpoint that ranks best on survival, direction, forward speed and return, in
that order, is kept.

```bash
WANDB_BASE_URL=https://api.wandb.ai \
uv run --extra utils --with mujoco --with wandb python examples/microduck/ppo_mujoco.py \
  env.microduck_root="$MICRODUCK_RL_ROOT" env.num_envs=8 \
  ppo.total_transitions=2000000 logger.entity=YOUR_ENTITY
```

[`config.yaml`](config.yaml) holds every setting and each one is a Hydra
override; the `utils` extra provides Hydra. `logger.entity` is required for
W&B so runs never land in a default workspace; use `logger.backend=csv` or
`logger.backend=null` for a local run. Set
`env.task.commanded_x_velocity=[0.0,0.03,0.06]` to train a command
distribution (`env.task` mirrors the fields of `torchrl.envs.MicroDuckTask`),
and `smoke=true` for a pipeline check. Checkpoints are unified TorchRL
checkpoints written with `save_render_checkpoint`, which `rlrender` and
`policy.init_from` read directly.

### Backends

Pass `env.backend=mjx` or `env.backend=mujoco-torch` to change only the
physics. `env.compile_step=true` compiles the `mujoco-torch` step; with the fixes in
[pytorch/rl#4202](https://github.com/pytorch/rl/pull/4202) and
[vmoens/mujoco-torch#85](https://github.com/vmoens/mujoco-torch/pull/85) the
compiled eight-environment MicroDuck step runs at roughly 180-190 transitions/s
on an Apple-silicon CPU after a one-off compile of about a minute, against
about 45 transitions/s in eager mode.

Measured on an Apple-silicon CPU with random actions at 50 Hz control:

| Backend | Batch | Transitions/s |
| --- | ---: | ---: |
| native MuJoCo, `ParallelEnv` | 16 workers | ~3,500 |
| MJX, jit + vmap | 1,024 envs | ~700 |
| MJX, jit + vmap | 2,048 envs | ~500 |

MJX slows down as envs fall, because its contact solver cost grows with the
number of active contacts, and it does not scale past about a thousand envs on
this CPU. Native MuJoCo in processes is the right choice for a laptop; MJX and
compiled `mujoco-torch` are the right choice on a GPU host.

## Results of the validation runs

The following runs used the gait-residual head with the earlier directional
reward at 125 Hz control, before the reward redesign described above. They
validated the pipeline rather than the task.

Three CPU runs (8 parallel native envs, Apple silicon) on the personal W&B
project
[`vmoens/torchrl-microduck-ppo`](https://wandb.ai/vmoens/torchrl-microduck-ppo):

| Run | Settings | Transitions | Collect / train throughput |
| --- | --- | ---: | ---: |
| [`microduck-gru-ppo-2m`](https://wandb.ai/vmoens/torchrl-microduck-ppo/runs/pwoj4cy8) | lr 3e-4, 10 epochs, command 0.03 | 2M | ~1,700 / ~14,000 transitions/s |
| [`microduck-gru-ppo-lr1e-4-ep5`](https://wandb.ai/vmoens/torchrl-microduck-ppo/runs/7uxqg9d7) | lr 1e-4, 5 epochs, command 0.03 (current defaults) | 1M | ~1,600 / ~13,000 transitions/s |
| [`microduck-gru-ppo-cmd-0-3-6`](https://wandb.ai/vmoens/torchrl-microduck-ppo/runs/04w5so9d) | lr 1e-4, 5 epochs, commands 0, 0.03, 0.06 | 1M | ~1,650 / ~13,000 transitions/s |

Deterministic evaluation of the best checkpoints over seeds 0-7 and 500
steps, with the requested command passed at reset:

| Policy | 0.00 m/s: displacement, tracking error | 0.03 m/s: displacement, tracking error | 0.06 m/s: displacement, tracking error |
| --- | --- | --- | --- |
| gait prior (transition 0) | +0.001 m, 0.0005 | +0.128 m, 0.020 | +0.128 m, 0.028 |
| `lr1e-4-ep5` best | +0.002 m, 0.0006 | +0.132 m, 0.021 | +0.132 m, 0.028 |
| `cmd-0-3-6` best | +0.001 m, 0.0005 | +0.126 m, 0.020 | +0.126 m, 0.028 |
| `2m` best | +0.001 m, 0.0005 | +0.130 m, 0.021 | +0.130 m, 0.028 |

Every policy survives all 500 steps for every command. What the runs show:

- The collector-to-buffer pipeline behaves as designed: every iteration holds
  40 complete 500-step episodes, GAE runs once over them, and the buffer is
  emptied before the next collection.
- The closed-form prior is already close to this reward's ceiling for a
  0.03 m/s command (about 944 of a possible 950 per episode), so PPO has little
  to gain and mostly preserves the prior; displacement moved from 0.128 to
  0.132 m at best.
- With the 0.05 exploration scale, a 3e-4 learning rate over ten epochs
  produced per-update KL divergences between 1 and 4 in the first iterations,
  and `KLAdaptiveLR` drove the learning rate to its floor. A 1e-4 learning rate
  with five epochs kept the KL inside the target band from the first update,
  which is why those are now the defaults.
- That reward paid for velocity along the commanded direction, not for
  matching its magnitude, and the gait prior only reads the command's sign, so
  every policy behaved identically at 0.03 and 0.06 m/s. The current reward
  tracks the command magnitude and rewards stepping through foot contacts.

## Training from scratch

The Gaussian head learns to walk without the closed-form prior once the
exploration noise is wide enough to lift a foot and the reward pays for
stepping rather than for standing. Four earlier from-scratch attempts
converged to standing still under every command; the `diagnostics=True`
reward breakdown showed why (the phase term paid half credit with both feet
planted, and the default 0.35 rad action scale with a 0.3 initial standard
deviation never broke ground contact). The recipe that walks is:

- `env.task.action_scale=1.0 policy.initial_policy_scale=1.0`: one radian of position
  target per unit action and an initial policy standard deviation of one, so
  the untrained policy actually swings its legs;
- dense contact shaping in `MicroDuckEnv`: single-support credit only when the
  clock's swing foot is airborne, a dense swing-height term, and a penalty for
  standing on both feet under a nonzero command;
- a velocity command in `[0.1, 0.3]` m/s with a warm start on half of the
  resets (`env.task.warm_start_velocity=[0.05,0.25]
  env.task.warm_start_fraction=0.5`) and
  0.25 rad of joint noise at reset, so episodes start away from the standing
  fixed point;
- PPO with 32,768 transitions per update, 5 epochs, 64 whole episodes per
  minibatch, `ppo.learning_rate=3e-4` under the KL-adaptive schedule
  (`ppo.target_kl=0.01`) and `ppo.entropy_coeff=0.01`.

```bash
WANDB_BASE_URL=https://api.wandb.ai \
uv run --extra utils --with mujoco --with wandb python examples/microduck/ppo_mujoco.py \
  env.microduck_root="$MICRODUCK_RL_ROOT" env.num_envs=16 env.parallel=true \
  policy.head=gaussian policy.initial_policy_scale=1.0 \
  env.task.action_scale=1.0 env.task.command_range=[0.1,0.3] \
  env.task.warm_start_velocity=[0.05,0.25] env.task.warm_start_fraction=0.5 \
  env.task.joint_reset_noise_scale=0.25 \
  env.task.gait_frequency_hz=1.0 env.task.gait_frequency_per_mps=5.0 \
  ppo.transitions_per_update=32768 ppo.epochs=5 ppo.minibatch_trajectories=64 \
  ppo.learning_rate=3e-4 ppo.target_kl=0.01 ppo.entropy_coeff=0.01 \
  ppo.total_transitions=10000000 evaluation.interval=10 \
  evaluation.latest_checkpoint_path=microduck_ppo_latest.ckpt \
  logger.entity=YOUR_ENTITY
```

The baseline run
[`9jgfqj8p`](https://wandb.ai/vmoens/torchrl-microduck-ppo/runs/9jgfqj8p)
(16 native workers on Apple silicon, about 3,500 collected transitions/s,
without the two gait clock flags that the ablations below added) stood still
until 2M transitions, started stepping around 3M and walked 1.5 m in 10 s
under every command by 7.5M with full survival.

### Ablations

Each variant below changed one setting of the baseline and trained for 5M
transitions from scratch; `S0` and `S1` instead continued the baseline
checkpoint taken at 7.46M transitions for 5M more. Evaluation is
deterministic over seeds 0-7 and 500 steps (10 s) at 0.1, 0.2 and 0.3 m/s.
"Speed error" is the mean absolute difference between the body-frame forward
speed and the command, averaged over the three commands; "displacement" is
the world-frame distance covered along the initial heading, which these runs
logged. The example now evaluates and ranks checkpoints by the mean
body-frame forward speed instead, for the reason given below.

| Run | Change | Speed error (m/s) | Displacement at 0.1 / 0.2 / 0.3 m/s (m) | Forward speed at 0.1 / 0.2 / 0.3 m/s (m/s) |
| --- | --- | ---: | --- | --- |
| [`9jgfqj8p`](https://wandb.ai/vmoens/torchrl-microduck-ppo/runs/9jgfqj8p) at 5M | baseline | 0.109 | +0.99 / +0.95 / +0.89 | 0.10 / 0.10 / 0.10 |
| [`9jgfqj8p`](https://wandb.ai/vmoens/torchrl-microduck-ppo/runs/9jgfqj8p) at 7.46M | baseline | 0.080 | +1.72 / +1.62 / +1.52 | 0.19 / 0.19 / 0.19 |
| [`vwk0hgu2`](https://wandb.ai/vmoens/torchrl-microduck-ppo/runs/vwk0hgu2) A1 | 16 episodes per minibatch | 0.087 | -0.21 / -0.20 / -0.39 | 0.17 / 0.17 / 0.17 |
| [`kkz2ctnm`](https://wandb.ai/vmoens/torchrl-microduck-ppo/runs/kkz2ctnm) A2 | gait clock 1 Hz + 5 Hz per m/s | 0.076 | +0.22 / +1.01 / +1.64 | 0.11 / 0.15 / 0.17 |
| [`iirlbbkv`](https://wandb.ai/vmoens/torchrl-microduck-ppo/runs/iirlbbkv) A3 | lateral and vertical velocity observed | 0.087 | +0.03 / -0.00 / +0.11 | 0.19 / 0.19 / 0.19 |
| [`fdit1i9n`](https://wandb.ai/vmoens/torchrl-microduck-ppo/runs/fdit1i9n) A4 | entropy coefficient 0.003 | 0.103 | -0.25 / -0.21 / -0.20 | 0.12 / 0.12 / 0.12 |
| [`4ae0vl4t`](https://wandb.ai/vmoens/torchrl-microduck-ppo/runs/4ae0vl4t) S0 | baseline continued to 12.5M | 0.085 | -0.39 / -0.61 / -0.59 | 0.22 / 0.23 / 0.24 |
| [`ahqvrivb`](https://wandb.ai/vmoens/torchrl-microduck-ppo/runs/ahqvrivb) S1 | continued to 12.5M with phase-contact 1.5, swing height 1.0, tracking 4.0 | 0.053 | +0.13 / +0.50 / +0.46 | 0.15 / 0.19 / 0.21 |

Every run survives all 500 steps for every command. What the ablations show:

- Two changes make the speed follow the command. From scratch, the
  command-scaled gait clock (A2) is the only variant that does it, which is
  why the recipe above passes `env.task.gait_frequency_hz=1.0
  env.task.gait_frequency_per_mps=5.0` (the `MicroDuckEnv.speed_range_task`
  preset in Python). Once a gait exists, halving the contact
  shaping and doubling the tracking weight (S1) cuts the speed error from
  0.085 to 0.053 m/s where continuing unchanged (S0) plateaus. Every other
  policy settles on a single gait at 0.12-0.24 m/s whatever the command.
- Smaller minibatches (A1) learn fastest early but end at the same speed
  error as the baseline; a lower entropy coefficient (A4) and extra velocity
  observations (A3) do not help.
- World-frame displacement understates walking: the deterministic rollouts
  cover 2.1-2.7 m of path but turn by 50-160 degrees in 10 s, because the
  yaw-rate term (Gaussian with standard deviation 0.71 rad/s) costs at most
  about 0.006 per step against 0.03 for velocity tracking, and body-frame
  tracking is blind to heading. Judge speed by the tracking error; a tighter
  yaw-rate term or a heading-error term is the next reward change to try.

## Rendering

`rlrender` reconstructs the env and policy from factories in these files.
The MuJoCo WASM viewer uses the original scene with its meshes, so playback
looks like the real robot even though physics ran with box proxies. Install
Node through the `mujoco_wasm` extra if the machine has none.

```bash
uv run --extra rendering --extra mujoco_wasm --with mujoco rlrender \
  --ckpt microduck_gait.pt \
  --policy examples/microduck/heuristic_gait.py:make_render_policy \
  --env examples/microduck/heuristic_gait.py:make_env \
  --env-kwargs "{\"microduck_root\":\"$MICRODUCK_RL_ROOT\",\"commanded_x_velocity\":0.03}" \
  --render-backend env --no-auto-load-policy --max-steps 500 --fps 50 \
  --format ipynb --out microduck_gait.ipynb \
  --notebook-render-backend mujoco-wasm --notebook-rollout-mode both \
  --mujoco-model-path "$MICRODUCK_RL_ROOT/src/mjlab_microduck/robot/microduck/scene_walk.xml" \
  --mujoco-qpos-key qpos --overwrite
```

A PPO checkpoint is a unified TorchRL checkpoint whose policy is the actor,
so no state-dict key is needed:

```bash
uv run --extra rendering --extra mujoco_wasm --with mujoco rlrender \
  --ckpt microduck_ppo_best.ckpt \
  --policy examples/microduck/ppo_mujoco.py:make_render_policy \
  --env examples/microduck/ppo_mujoco.py:make_env \
  --deterministic \
  --env-kwargs "{\"microduck_root\":\"$MICRODUCK_RL_ROOT\",\"num_envs\":1,\"cfg\":{\"task\":{\"commanded_x_velocity\":[0.03]}}}" \
  --render-backend null --max-steps 500 \
  --format ipynb --out microduck_ppo.ipynb \
  --notebook-render-backend mujoco-wasm --notebook-rollout-mode both \
  --mujoco-model-path "$MICRODUCK_RL_ROOT/src/mjlab_microduck/robot/microduck/scene_walk.xml" \
  --mujoco-qpos-key qpos --overwrite
```

Open either notebook with
`uv run --extra rendering --extra mujoco_wasm --extra notebook --with mujoco jupyter lab <file>`.
The generated notebook exposes a `live_env_kwargs` cell, so a different
command can be rolled out in the kernel without regenerating it.

## MJLab

`ppo_mjlab.py` trains the upstream `Mjlab-Velocity-Flat-MicroDuck` task, with
its BAM actuator model, observations, rewards, curricula and randomization
intact, through `MJLabWrapper(log_extras=True)` so mjlab's episode metrics are
logged alongside the PPO losses. It requires a CUDA GPU and the pinned
`microduck_rl` environment; see the module docstring for the `uv run` command.
