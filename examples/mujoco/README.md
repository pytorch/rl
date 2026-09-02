# MuJoCo examples

## MicroDuck locomotion

[`ppo_microduck.py`](ppo_microduck.py) defines one longitudinal-velocity task
across the native MuJoCo, MJX, and `mujoco-torch` backends. The default task is
a forward command of `0.03 m/s`. An exact zero command requests standing, and a
negative command requests backward motion in the duck's body frame.

The 53-value policy observation contains projected gravity, body angular
velocity, measured and commanded longitudinal velocity, joint position error,
joint velocity, gait phase sine and cosine, gait ramp, and the previous action.
The commanded velocity is also exposed directly in the TensorDict.

[MuJoCo free-joint linear velocity is world-frame, while angular velocity is
body-frame](https://mujoco.readthedocs.io/en/3.3.0/overview.html#floating-objects).
The task rotates the measured linear velocity into the duck's body frame before
computing the observation, reward, or lateral-drift cost.

For locomotion, each alive step receives a constant reward plus signed forward
velocity. This makes displacement accumulated while upright the primary
objective. A `-10` terminal cost prevents a brief forward lunge from competing
with a complete episode. Uprightness and height stabilize the gait, while small
costs discourage lateral drift, roll/yaw rate, joint velocity, and action rate.
The exact zero-command task retains a Gaussian velocity-tracking reward and a
nominal-pose term. Only a physical fall or non-finite state terminates an
episode.

The normalized 14-joint position action and its `0.35`-radian scale are
unchanged from the stand feasibility task. The upstream walking MJCF uses
detailed render meshes for foot and self-collision geoms. Convex collision
preprocessing for the two roughly 10,000-edge soles creates about 107 million
edge pairs per environment, which caused the accelerated-backend memory
blow-up. At load time, the task replaces only collision-class meshes with tight
axis-aligned box proxies; visual meshes and the source checkout remain
unchanged. The same patched physics scene is passed to all three backends.

### Closed-form walking controller

[`heuristic_microduck.py`](heuristic_microduck.py) provides a no-gradient
walking baseline for the same model, collision proxies, position-control
interface, reset perturbations, and fall conditions. It combines a bilateral
phase oscillator over hip, knee, ankle, and lateral targets with hip and ankle
pitch feedback. Its action remains an offset around the MJCF `STAND` target,
exactly like the PPO policy.

The default parameters completed all 100 fixed 500-step evaluation rollouts
with `0.02` reset noise. Every rollout satisfied the walking gate: both feet
entered at least four distinct swing phases, the opposite foot remained in
contact for at least four consecutive control steps per phase, torso pitch
remained below `0.056 rad`, and the robot moved forward. The worst rollout had
seven left and six right swing phases, at least 56 and 53 left/right
single-support steps, and foot-site heights above `0.007 m` for both feet. Mean
signed displacement was `0.1280 m` over four seconds, the minimum was
`0.1200 m`, and mean body-forward speed was `0.0355 m/s`.

Displacement and survival alone are not accepted as locomotion metrics. A
planted-foot controller can move forward by pitching its torso, so the script
reports contact-derived swing phases, single-support duration, foot height, and
maximum pitch and ranks those constraints before speed.

```bash
uv run --with mujoco python examples/mujoco/heuristic_microduck.py \
  --microduck-root /path/to/microduck_rl \
  --commanded-x-velocity 0.03 \
  --num-seeds 100 \
  --render-checkpoint microduck_heuristic_policy.pt
```

The script can also perform gait-constrained random search around the validated
controller. Candidates are ranked by worst-case survival, repeated bilateral
swing phases, walking success, and bounded pitch before forward speed:

```bash
uv run --with mujoco python examples/mujoco/heuristic_microduck.py \
  --microduck-root /path/to/microduck_rl \
  --commanded-x-velocity 0.03 \
  --search-candidates 128 \
  --search-num-seeds 8
```

### PPO from a walking gait prior

The PPO actor starts at the closed-form walking controller and learns a bounded
residual. A six-feature basis (bias, gait sine and cosine, pitch, pitch rate,
and longitudinal velocity) feeds one linear `6 x 14` map. Together with its
action bias and per-joint residual scale, the actor has 112 trainable
parameters. Initializing that map to zero makes transition 0 exactly the robust
closed-form gait. A separate two-layer critic prevents value gradients from
changing the actor representation.

The long-run defaults use complete trajectories, a 16,384-transition replay
buffer, up to 10 PPO epochs with `0.01` approximate-KL early stopping,
trajectory minibatches, a fixed `3e-5` learning rate, and a `0.01` initial
policy scale. `gamma=lambda=1` matches the finite-horizon objective. Evaluation
runs after every collection over seeds 0-7 plus three historically fragile
seeds. The current checkpoint score ranks survival, displacement, and return;
that protects against a fast fall but does not detect planted feet. Transition
0 remains eligible so a regressing run cannot replace its gait prior.

The bundled PPO checkpoint predates the contact-derived walking gate and must
not be used as evidence of walking. Its earlier evaluation measured survival
and displacement only, which cannot distinguish stepping from a controlled
forward pitch. Retrain and select a new checkpoint with the corrected gait and
contact metrics before making PPO locomotion claims.

```bash
uv run --with mujoco --with wandb --with psutil \
  python examples/mujoco/ppo_microduck.py \
  --microduck-root /path/to/microduck_rl \
  --commanded-x-velocity 0.03 \
  --total-transitions 5000000 \
  --best-checkpoint-path microduck_locomotion_ppo.pt \
  --wandb-entity YOUR_ENTITY
```

Repeat `--commanded-x-velocity` to train a command distribution. Use
`--wandb-mode disabled` for a local run without tracking. When logging is
enabled, `--wandb-entity` is required so W&B cannot silently use a different
default team or personal workspace.

### Why the original PPO task did not learn to walk

The original task asked for `0.3 m/s`, while the feasible gait demonstrated
only about `0.011 m/s`. More importantly, its per-step Gaussian tracking reward
made pitching forward and falling a strong local optimum. On the same forward
command, a zero-action fall lasted 113 steps, moved `0.097 m`, and earned
`1.613` reward per step; the stable controller lasted 500 steps, moved
`0.042 m`, and earned only `1.245` reward per step. With `gamma=0.99`, their discounted
returns were close enough that PPO had weak long-horizon pressure to preserve
balance.

The original 14-dimensional policy also began near zero action, so it had to
discover both a periodic gait and balance through delayed credit. Full signed
command sampling diluted that already sparse signal, the shared recurrent
actor/critic let critic gradients alter policy features, and linear learning-
rate decay reached zero even when the task remained unsolved. Selecting the
largest raw return further rewarded some short forward falls. The revised task
addresses each observed failure directly: a feasible forward curriculum,
explicit phase, a safe gait prior, bounded residual learning, separate actor
and critic, undiscounted finite-horizon credit, a terminal cost, and
survival-first checkpoint selection.

The earlier end-to-end 10-million-transition run remains useful negative
evidence: none of its fixed-command evaluation rollouts survived 500 steps, and
the policy often collapsed toward the same direction for every command. It
does not establish that locomotion is impossible; it establishes that the old
reward, initialization, and selection rule were misaligned with the desired
behavior.

### Rendering and notebooks

The closed-form policy can be opened as a live RLRender notebook. The notebook
stores one rollout for immediate playback and can reconstruct the environment
and policy to collect another rollout in the kernel:

```bash
MICRODUCK_RL_ROOT=/path/to/microduck_rl

uv run --extra rendering --with mujoco rlrender \
  --ckpt microduck_heuristic_policy.pt \
  --policy examples.mujoco.heuristic_microduck:make_render_policy \
  --env examples.mujoco.ppo_microduck:make_env \
  --env-kwargs "{\"microduck_root\":\"$MICRODUCK_RL_ROOT\",\"backend\":\"mujoco\",\"commanded_x_velocity\":0.03,\"num_envs\":1}" \
  --render-backend env \
  --no-auto-load-policy \
  --max-steps 500 \
  --fps 125 \
  --format ipynb \
  --out microduck_heuristic_policy.ipynb \
  --notebook-render-backend mujoco-wasm \
  --notebook-rollout-mode both \
  --mujoco-model-path "$MICRODUCK_RL_ROOT/src/mjlab_microduck/robot/microduck/scene_walk.xml" \
  --mujoco-qpos-key qpos \
  --overwrite

uv run --extra rendering --extra notebook --with mujoco \
  jupyter lab microduck_heuristic_policy.ipynb
```

A trained checkpoint stores the actor under the `actor` key. Render it with:

```bash
MICRODUCK_RL_ROOT=/path/to/microduck_rl

uv run --extra rendering --with mujoco rlrender \
  --ckpt microduck_locomotion_ppo.pt \
  --policy examples/mujoco/ppo_microduck.py:make_render_policy \
  --env examples/mujoco/ppo_microduck.py:make_env \
  --state-dict-key actor \
  --env-kwargs "{\"microduck_root\":\"$MICRODUCK_RL_ROOT\",\"backend\":\"mujoco\",\"commanded_x_velocity\":0.03,\"num_envs\":1}" \
  --render-backend null \
  --deterministic \
  --max-steps 500 \
  --format ipynb \
  --out microduck_locomotion_ppo.ipynb \
  --notebook-render-backend mujoco-wasm \
  --notebook-rollout-mode both \
  --mujoco-model-path "$MICRODUCK_RL_ROOT/src/mjlab_microduck/robot/microduck/scene_walk.xml" \
  --mujoco-qpos-key qpos \
  --overwrite

uv run --extra rendering --extra notebook --with mujoco \
  jupyter lab microduck_locomotion_ppo.ipynb
```

The `nodejs-wheel` dependency belongs to the `rendering` extra because it is
required by the MuJoCo-WASM stack. The final Jupyter command therefore needs no
additional `--with nodejs-wheel` argument.

The accompanying [`microduck_ppo.ipynb`](microduck_ppo.ipynb) exposes the same
training and deterministic fixed-seed evaluation functions for interactive
experiments.

### Backend throughput notes

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
With the phase-residual actor and periodic deterministic evaluation, native
MuJoCo collects approximately 1,000 training transitions/s on this machine;
the small actor makes PPO updates much faster than collection.
