# MicroDuck examples

[MicroDuck](https://github.com/pollen-robotics/microduck) is a small
open-hardware biped from Pollen Robotics. Its walking model and meshes live in
[`microduck_rl`](https://github.com/pollen-robotics/microduck_rl) and are not
vendored in TorchRL: point the scripts at a checkout with
`--microduck-root /path/to/microduck_rl` or `export MICRODUCK_RL_ROOT=...`.

| File | What it does | Needs |
| --- | --- | --- |
| [`ppo_mujoco.py`](ppo_mujoco.py) | Recurrent PPO on `torchrl.envs.MicroDuckEnv`; native MuJoCo, MJX or `mujoco-torch` | `mujoco` (+ `mujoco-mjx`/`jax` or `mujoco-torch`) |
| [`heuristic_gait.py`](heuristic_gait.py) | Closed-form walking gait, contact-based gait metrics, gait search, `rlrender` policy | `mujoco` |
| [`ppo_mjlab.py`](ppo_mjlab.py) | PPO on the upstream `Mjlab-Velocity-Flat-MicroDuck` task through `MJLabWrapper` | MJLab, `mjlab_microduck`, CUDA |
| [`microduck_ppo.ipynb`](microduck_ppo.ipynb) | Interactive version of the MuJoCo pipeline with native and WASM rendering | `rendering` and `mujoco_wasm` extras |

## The task

`MicroDuckEnv` is a commanded longitudinal-velocity task written once against
`MujocoEnv`, so `backend="mujoco"`, `"mjx"` and `"mujoco-torch"` share the
observation, action, reward and termination definitions. The 14 actions are
normalized offsets around the `STAND` keyframe targets. The 53-value
observation holds projected gravity, body angular velocity, measured and
commanded body-frame forward velocity, joint errors and velocities, a
fixed-frequency gait clock and the previous action; the command is also
exposed under `commanded_x_velocity`. Nonzero commands are rewarded for
velocity along the commanded direction plus an alive bonus, a zero command
for Gaussian velocity tracking and a nominal pose; uprightness and height
stabilize, small costs discourage lateral drift, roll/yaw rate, joint velocity
and action rate, and a fall costs a fixed penalty and terminates the episode.

The upstream walking MJCF reuses detailed render meshes as collision geoms.
Their convex-hull edge pairs make the accelerated backends run out of memory,
so the env swaps collision-class meshes for tight box proxies at load time
(`low_cost_collisions=True`). Visual meshes and the checkout are untouched.
Contact-based gait metrics are available through `env.foot_contacts()` and
`env.foot_heights()` on every backend.

## Closed-form gait

`heuristic_gait.py` combines a bilateral phase oscillator with hip and ankle
pitch feedback. The same `gait_action` function drives the baseline, seeds the
PPO actor and serves as the `rlrender` policy. Rollouts are judged from foot
contacts: a rollout counts as walking only when both feet alternate swing
phases while the other foot is in single support, torso pitch stays bounded,
and the robot moves in the commanded direction. Displacement alone is not
accepted, since a planted-foot controller can move forward by pitching.

```bash
uv run --with mujoco python examples/microduck/heuristic_gait.py \
  --microduck-root "$MICRODUCK_RL_ROOT" --num-seeds 20 \
  --render-checkpoint microduck_gait.pt
```

Add `--search-candidates 128 --search-num-seeds 8` to run a gait-constrained
random search around the defaults; candidates are ranked by worst-case
survival and bilateral stepping before speed.

## Recurrent PPO

The policy is a GRU backbone shared by the actor and the critic. The actor head
adds a bounded, zero-initialized residual to the closed-form gait, so training
starts from a validated walking controller. Data flows through standard TorchRL
components:

1. a `Collector` with `trajs_per_batch=1` writes every finished episode as a
   whole, unpadded sequence into a `TensorDictReplayBuffer`;
2. GAE runs once over the buffer in recurrent mode;
3. `SliceSampler` draws whole episodes for the PPO minibatches;
4. the buffer is emptied and in-flight episodes dropped before collecting
   again with the updated policy.

`KLAdaptiveLR` keeps the mean policy KL near `--target-kl`. Deterministic,
fixed-seed evaluation runs every `--evaluation-interval` iterations and keeps
the checkpoint that ranks best on survival, direction, displacement and
return, in that order.

```bash
WANDB_BASE_URL=https://api.wandb.ai \
uv run --with mujoco --with wandb python examples/microduck/ppo_mujoco.py \
  --microduck-root "$MICRODUCK_RL_ROOT" \
  --num-envs 8 --total-transitions 2000000 \
  --best-checkpoint-path microduck_ppo_best.pt \
  --wandb-entity YOUR_ENTITY
```

`--wandb-entity` is required whenever logging is enabled; use
`--wandb-mode disabled` for a local run. Repeat `--commanded-x-velocity` to
train a command distribution, and `--smoke` for a pipeline check.

### Backends

Pass `--backend mjx` or `--backend mujoco-torch` to change only the physics.
`--compile-step` compiles the `mujoco-torch` step; with the fixes in
[pytorch/rl#4202](https://github.com/pytorch/rl/pull/4202) and
[vmoens/mujoco-torch#85](https://github.com/vmoens/mujoco-torch/pull/85) the
compiled eight-environment MicroDuck step runs at roughly 180-190 transitions/s
on an Apple-silicon CPU after a one-off compile of about a minute, against
about 45 transitions/s in eager mode. Native MuJoCo with eight serial envs
collects around 1,000 transitions/s on the same machine and remains the
default for laptop training; `--parallel` switches it to `ParallelEnv`.

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
  --render-backend env --no-auto-load-policy --max-steps 500 --fps 125 \
  --format ipynb --out microduck_gait.ipynb \
  --notebook-render-backend mujoco-wasm --notebook-rollout-mode both \
  --mujoco-model-path "$MICRODUCK_RL_ROOT/src/mjlab_microduck/robot/microduck/scene_walk.xml" \
  --mujoco-qpos-key qpos --overwrite
```

A PPO checkpoint stores the actor under the `actor` key:

```bash
uv run --extra rendering --extra mujoco_wasm --with mujoco rlrender \
  --ckpt microduck_ppo_best.pt \
  --policy examples/microduck/ppo_mujoco.py:make_render_policy \
  --env examples/microduck/ppo_mujoco.py:make_env \
  --state-dict-key actor --deterministic \
  --env-kwargs "{\"microduck_root\":\"$MICRODUCK_RL_ROOT\",\"num_envs\":1,\"commanded_x_velocity\":0.03}" \
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
