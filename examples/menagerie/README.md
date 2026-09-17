# PPO on MuJoCo Menagerie robots, rendered with rlrender

`ppo.py` trains a PPO policy on any [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie)
robot through `MenagerieEnv` and writes a checkpoint that `rlrender` plays back
as a video, a GIF or a notebook. Two tasks:

| task | robots | what the policy learns |
|---|---|---|
| `hold_pose` | any | `MenagerieEnv.hold_pose_task()`: stay at the `home` keyframe under a control cost |
| `walk` | position-controlled quadrupeds | track a body-frame velocity command with the reward of MuJoCo Playground's Go1 joystick task |

The walk task is two transforms over the raw env state: `HomeOffsetActions`
maps a normalized action to joint targets around the home pose, and
`QuadrupedJoystick` assembles the observation, the reward and the fall
termination. Both live in `ppo.py`; `make_env` and `make_policy` are the
factories `rlrender` imports.

## Setup

```bash
pip install mujoco-menagerie   # robots download into its cache with --download
uv run --extra rendering rlrender --help   # MP4 output needs torchcodec and FFmpeg
```

Without the package, point `TORCHRL_MUJOCO_MENAGERIE_PATH` at a checkout of
`mujoco_menagerie`; `--entry` selects the XML by stem (`scene`, `scene_mjx`).

## Train

Stand (any robot, here the Unitree Go2's MJX scene, about two minutes on a
laptop CPU):

```bash
python examples/menagerie/ppo.py --task hold_pose --robot unitree_go2 --entry scene_mjx \
    --download --fall-height 0.15 --num-envs 4 --frames 200000 --ckpt go2_stand.ckpt
```

Walk (the MJX scene has position servos, which the task needs):

```bash
python examples/menagerie/ppo.py --task walk --robot unitree_go2 --entry scene_mjx \
    --download --num-envs 6 --frames 6000000 --command-range 1.0 0.5 0.8 --ckpt go2_walk.ckpt
```

`--command-range` draws a fresh command per episode; `--command` (0.5 m/s
forward by default) is what the policy tracks at evaluation. Every batch logs
the reward per step, the termination rate and the mean forward speed, and
rewrites the checkpoint.

`--smoke` runs one tiny update on one env, which is what the examples CI does
on the UR5e.

## Render

`rlrender` rebuilds the env and the policy from the checkpoint through the
factories in `ppo.py`; the checkpoint records the robot, the task and the
network, so only render options are needed. Frames come from the scene's
first camera (`camera_id=0`, the Go2's `track` camera) at the requested size;
`--fps 50` is real time for the 20 ms control step.

```bash
rlrender --ckpt go2_walk.ckpt \
    --policy examples/menagerie/ppo.py:make_policy \
    --env examples/menagerie/ppo.py:make_env \
    --env-kwargs '{"camera_id": 0, "render_width": 640, "render_height": 480, "fixed_command": true}' \
    --deterministic --from-pixels --render-backend pixels \
    --max-steps 500 --fps 50 --format mp4 --out go2_walk.mp4 --overwrite
```

`--format gif` writes an animated GIF through Pillow when FFmpeg is missing,
and `--no-auto-load-policy` renders the freshly initialised policy for a
before-and-after comparison.

A notebook with the saved rollout, a cell that collects a fresh one in the
kernel, the MP4 preview and an interactive MuJoCo WASM viewer of the trajectory
(Node.js needed; `--mujoco-model-path` is the scene the checkpoint trained on):

```bash
rlrender --ckpt go2_walk.ckpt \
    --policy examples/menagerie/ppo.py:make_policy \
    --env examples/menagerie/ppo.py:make_env \
    --env-kwargs '{"camera_id": 0, "render_width": 640, "render_height": 480, "fixed_command": true}' \
    --deterministic --from-pixels --render-backend pixels \
    --max-steps 500 --fps 50 --format ipynb --out go2_walk.ipynb \
    --notebook-render-backend mujoco-wasm --notebook-rollout-mode both \
    --mujoco-model-path "$(python -c 'import mujoco_menagerie as mm; print(mm.get("unitree_go2").xml("scene_mjx"))')" \
    --mujoco-qpos-key qpos --overwrite
```

Notebooks, checkpoints and videos are generated artifacts and are not
committed; the commands above regenerate them.

## Results

`backend="mujoco"` with worker processes, one seed; laptop rows on an Apple M1 Pro. On a headless machine without a display, render with `--render-backend null` and finish the video on a machine with FFmpeg.

| run | command | frames | wall time | outcome |
|---|---|---|---|---|
| Go2 `hold_pose` | `--fall-height 0.15 --num-envs 4 --frames 200000` | 200k | 2 min | reward per step 1.10 to 1.23 (max 1.5); deterministic return 699 over 500 steps versus 606 untrained |
| Go2 `walk` | `--command-range 1.0 0.5 0.8 --num-envs 6 --frames 6000000` | 6M | 30 min | at the 0.5 m/s command: 0.45 m/s, 19 to 20 touchdowns per foot in 10 s, 0.24 s flight, 0.08 m swing height, 85% diagonal-pair timing, no falls |
| Go2 `walk`, Mac Studio (20 workers) | `--command-range 1.0 0.5 0.8 --num-envs 20 --frames 12000000 --steps-per-batch 1024 --minibatch 4096` | 12M | 40 min | at the 0.5 m/s command: 0.48 m/s, 17 to 18 touchdowns per foot with all four feet loaded evenly (53 to 63% contact each), 0.25 s flight, 0.09 m swing height, no falls |

Two things the walk numbers depend on. Without the `feet_stuck` cost the
reference terms let PPO settle, after about 5M frames, on a three-legged gait
that keeps one foot in the air for the whole episode: it tracks the command
about as well and avoids the slip and clearance costs. The cost on air time
beyond half a second closes that loophole. And a fixed training command
reaches a clean trot sooner (0.51 m/s with all four feet at 2M frames) but
specializes to it; `--command-range` costs some tracking accuracy for a
policy that follows any command.
