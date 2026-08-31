# MuJoCo examples

## MicroDuck PPO

[`ppo_microduck.py`](ppo_microduck.py) defines one commanded longitudinal-
velocity task across the native MuJoCo, MJX, and `mujoco-torch` backends. An
exact zero command requests standing; positive and negative commands request
forward and backward motion in the duck's body frame. The 50-value policy
observation contains projected gravity, body angular velocity, measured and
commanded longitudinal velocity, joint position error, joint velocity, and the
previous action. The commanded velocity is also exposed directly in the
TensorDict.

[MuJoCo free-joint linear velocity is world-frame, while angular velocity is
body-frame](https://mujoco.readthedocs.io/en/3.3.0/overview.html#floating-objects).
The task rotates the measured linear velocity into the duck's body frame before
computing the observation, tracking reward, or lateral-drift cost.

The reward prioritizes smooth signed velocity tracking and uses uprightness and
height only as stabilizers. It penalizes lateral drift, roll/yaw rate, joint
velocity, and action rate. A nominal-pose reward is smoothly gated to zero away
from the zero-velocity command, so it supports standing without fighting a
gait. Velocity error does not terminate an episode; only a physical fall or
non-finite state does. The normalized 14-joint position action and its
`0.35`-radian scale are unchanged from the stand feasibility task.

The upstream walking MJCF uses detailed render meshes for foot and
self-collision geoms. Convex collision preprocessing for the two roughly
10,000-edge soles creates about 107 million edge pairs per environment and was
the source of the accelerated-backend memory blow-up. At load time, the task
now replaces only collision-class meshes with tight axis-aligned box proxies;
visual meshes and the source checkout remain unchanged. The same patched
physics scene is passed to all three backends.

The same script trains every command combination. Its long-run defaults are 10
million complete-trajectory transitions, a 16,384-transition replay buffer, 10
PPO epochs, an actor/value model with a shared GRU backbone, deterministic
evaluation, and best-checkpoint retention. The collector is attached directly
to the replay buffer: it holds incomplete episodes internally and writes each
finished episode as one contiguous sequence. `SliceSampler` then samples whole
episodes for the recurrent update, and the on-policy buffer is erased after the
10 epochs.

```bash
uv run --with mujoco --with wandb --with psutil \
  python examples/mujoco/ppo_microduck.py \
  --microduck-root /path/to/microduck_rl
```

Repeat `--commanded-x-velocity` to select the reset command distribution; the
default is `-0.3`, `0.0`, and `0.3` m/s. W&B records collection/inference and
training throughput, process and device telemetry, reward and episodic return,
tracking error, survival and episode length, gradient norm, ESS, clip fraction,
KL, entropy, and all PPO losses. Use `--wandb-mode disabled` for a local run
without tracking.

On an Apple-silicon CPU after the collision fix, a zero-action physics-only
benchmark measured the following steady-state rates. These figures select
native MuJoCo as the default laptop training backend; they are not
cross-hardware simulator benchmarks.

| backend | environments | transitions/s |
| --- | ---: | ---: |
| native MuJoCo | 1 | 4,236 |
| native MuJoCo | 8 | 3,731 |
| MJX | 4 | 1,069 |
| MJX | 8 | 1,221 |
| `mujoco-torch` | 8 | 39 |

The `mujoco-torch` measurement used its current main branch because release
0.2.0 does not import with the current MuJoCo enum layout. It used less memory
than MJX in the isolated check, but was much slower in eager CPU execution.
With the 128-unit GRU included, native MuJoCo collection measured about 974,
1,777, 1,969, and 1,866 transitions/s for 1, 4, 8, and 16 environments,
respectively. Eight environments are therefore the script default on this
machine; increasing the batch further did not improve end-to-end collection.

The accompanying [`microduck_ppo.ipynb`](microduck_ppo.ipynb) trains the task
locally and renders the policy with native MuJoCo or the interactive MuJoCo
WASM viewer. `evaluate_policy()` runs deterministic fixed-command episodes over
multiple seeds and reports return, tracking error, survival, episode length,
and signed displacement. Passing `evaluation_env` and `evaluation_interval` to
`train_ppo()` retains the best evaluated actor and critic; optionally pass
`best_checkpoint_path` to persist them. Transition 0 is eligible so a regressing
training run cannot overwrite a better initial policy. The actor starts with
zero deterministic actions and a `0.2` exploration scale.

[Watch the local PPO before/after rollout](assets/microduck_ppo_before_after.mp4).
This earlier stand-only feasibility video uses native MuJoCo on CPU, an
identical `STAND` reset, and deterministic actions. The initial policy
terminates at step 108; the PPO checkpoint terminates at step 143. It validates
the local training and rendering path, not the current commanded-locomotion
reward.

That feasibility run used eight environments, 240 iterations of 128 steps, two
PPO epochs, 256-sample minibatches, a linearly decayed `1e-4` learning rate, an
entropy coefficient of `1e-4`, and a critic coefficient of `0.5`. Evaluation
every 20 iterations selected iteration 60. Across eight 500-step evaluation
streams, mean step reward changed from 2.740 to 2.801 and terminations from 32
to 24. This is not a benchmark, and the compact defaults are not validated
training hyperparameters.

No standing, forward-walking, or backward-walking policy is claimed solved by
the current defaults. Train and evaluate all three fixed commands before using
the example as locomotion evidence.

A fixed zero-command feasibility run used eight native MuJoCo CPU environments,
240 iterations of 128 steps, two PPO epochs, 256-sample minibatches, linear
learning-rate decay from `1e-4`, an entropy coefficient of `1e-4`, and a critic
coefficient of `0.5`. The best checkpoint was iteration 120. Across controlled
seeds 100–103, episode lengths were 122, 330, 251, and 459 steps: 0/4 survived
the 500-step horizon. This is evidence that the revised initialization and
zero-command pose gate improve some standing rollouts, but standing is not yet
robust, so fixed-command walking training has not started.
