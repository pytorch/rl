# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Profile a TorchRL collector running a MuJoCo Playground environment.

MuJoCo Playground is a collection of JAX-based MJX environments spanning
locomotion, manipulation, and dm_control suite tasks.  This script mirrors
``profile_isaaclab_collector.py`` and additionally supports saving a rendered
visualization after profiling is complete.

Usage::

    python profile_mujoco_playground_collector.py --env-name CartpoleBalance

To also save a rendered video::

    python profile_mujoco_playground_collector.py \\
        --env-name CartpoleBalance \\
        --batch-size 8 \\
        --render \\
        --render-steps 120 \\
        --render-output cartpole.mp4

Video export requires ``imageio``; if unavailable, frames are saved as
individual PNG files in a directory of the same name.
"""

from __future__ import annotations

import argparse
import logging
import os
import typing
from pathlib import Path

import numpy as np
import torch

from tensordict import TensorDictBase
from torchrl._utils import logger as torchrl_logger
from torchrl.collectors import Collector
from torchrl.envs import MujocoPlaygroundEnv
from torchrl.envs.libs.jax_utils import _tensordict_to_object

os.environ.setdefault("TORCHRL_PROFILING", "1")


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

parser = argparse.ArgumentParser(
    description="Profile a TorchRL Collector with a MuJoCo Playground environment."
)
parser.add_argument(
    "--env-name",
    default="CartpoleBalance",
    help="MuJoCo Playground environment name (see MujocoPlaygroundEnv.available_envs).",
)
parser.add_argument(
    "--batch-size",
    default=16,
    type=int,
    help=(
        "Number of environments to simulate in parallel via JAX vmap. "
        "Pass 0 for a scalar (unbatched) environment."
    ),
)
parser.add_argument(
    "--frames-per-batch",
    default=8192,
    type=int,
    help="Frames requested from the collector per rollout.",
)
parser.add_argument(
    "--profile-rollouts",
    default=5,
    type=int,
    help="Number of collector rollouts observed by torch.profiler.",
)
parser.add_argument(
    "--warmup-rollouts",
    default=1,
    type=int,
    help="Rollouts skipped by the profiler schedule before active recording.",
)
parser.add_argument(
    "--output-dir",
    default=Path("mujoco_playground_profiles"),
    type=Path,
    help="Directory where Chrome trace JSON files (and optional render output) are written.",
)
parser.add_argument("--seed", default=0, type=int, help="Random seed.")
parser.add_argument(
    "--device",
    default="cuda:0",
    help="Torch device passed to MujocoPlaygroundEnv.",
)
parser.add_argument(
    "--policy-mode",
    default="rand_action",
    choices=["rand_action", "fixed_zero"],
    help="Policy used by the collector during profiling.",
)
parser.add_argument(
    "--track-traj-ids",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Track collector trajectory ids in emitted batches.",
)
parser.add_argument(
    "--trust-policy",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Trust the policy output and skip selected collector safety checks.",
)
parser.add_argument(
    "--activities",
    nargs="+",
    default=["cpu", "cuda"],
    choices=["cpu", "cuda"],
    help="Profiler activities.",
)
parser.add_argument(
    "--record-shapes",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record tensor shapes in the profiler trace.",
)
parser.add_argument(
    "--profile-memory",
    action="store_true",
    help="Record memory events in the profiler trace.",
)
parser.add_argument(
    "--with-stack",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Record Python stack traces in the profiler trace.",
)
parser.add_argument(
    "--log-level",
    default="INFO",
    choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    help="Python logging level.",
)

# ---------------------------------------------------------------------------
# Visualization arguments
# ---------------------------------------------------------------------------
parser.add_argument(
    "--render",
    action="store_true",
    help=(
        "After profiling, collect a short rollout and save a rendered video "
        "(requires the underlying env to support rendering via env._env.render)."
    ),
)
parser.add_argument(
    "--render-steps",
    default=100,
    type=int,
    help="Number of steps to collect for the visualization rollout.",
)
parser.add_argument(
    "--render-height",
    default=240,
    type=int,
    help="Rendered frame height in pixels.",
)
parser.add_argument(
    "--render-width",
    default=320,
    type=int,
    help="Rendered frame width in pixels.",
)
parser.add_argument(
    "--render-camera",
    default=None,
    help="Camera name to use for rendering (None = environment default).",
)
parser.add_argument(
    "--render-output",
    default=None,
    type=Path,
    help=(
        "Output path for the rendered video (.mp4) or PNG frames (directory). "
        "Defaults to <output-dir>/render.mp4."
    ),
)
parser.add_argument(
    "--render-fps",
    default=30,
    type=int,
    help="Frames per second for the output video.",
)

args_cli = parser.parse_args()


# ---------------------------------------------------------------------------
# Helper classes and functions
# ---------------------------------------------------------------------------


class FixedZeroPolicy:
    """Policy that always returns a zero action tensor."""

    def __init__(self, env: MujocoPlaygroundEnv):
        self.action_key = env.action_key
        self.action_spec = env.action_spec

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        action = self.action_spec.zero()
        if isinstance(action, TensorDictBase):
            tensordict.update(action)
        else:
            tensordict.set(self.action_key, action)
        return tensordict


def make_env(env_name: str, batch_size: int, device: str) -> MujocoPlaygroundEnv:
    """Create a :class:`~torchrl.envs.MujocoPlaygroundEnv`.

    Args:
        env_name (str): environment name from :attr:`MujocoPlaygroundEnv.available_envs`.
        batch_size (int): number of parallel JAX vmap environments.
            Pass ``0`` for a scalar (unbatched) environment.
        device (str): torch device string.

    Returns:
        MujocoPlaygroundEnv: the constructed environment.
    """
    bs = [batch_size] if batch_size > 0 else []
    # Force the 'jax' implementation to avoid potential Warp-related
    # compatibility issues.
    return MujocoPlaygroundEnv(
        env_name,
        batch_size=bs,
        device=torch.device(device),
        config_overrides={"impl": "jax"},
    )


def make_policy(
    env: MujocoPlaygroundEnv, policy_mode: typing.Literal["rand_action", "fixed_zero"]
) -> typing.Callable:
    """Return a callable policy for the given mode.

    Args:
        env (MujocoPlaygroundEnv): the environment whose specs define the policy.
        policy_mode (str): ``"rand_action"`` or ``"fixed_zero"``.

    Returns:
        callable: a policy callable accepting and returning a TensorDict.
    """
    if policy_mode == "rand_action":
        return env.rand_action
    if policy_mode == "fixed_zero":
        return FixedZeroPolicy(env)
    raise ValueError(f"Unknown policy mode {policy_mode!r}.")


def save_visualization(env: MujocoPlaygroundEnv, args: argparse.Namespace) -> None:
    """Collect a short rollout and save a rendered visualization.

    Converts TensorDict states back to JAX State objects, calls the
    underlying environment's ``render`` method, then writes the frames
    to an MP4 file (via ``imageio``) or to individual PNG files.

    Args:
        env (MujocoPlaygroundEnv): the environment to render.
        args (argparse.Namespace): parsed command-line arguments.
    """
    import jax

    torchrl_logger.info(
        f"Collecting {args.render_steps} steps for visualization rollout."
    )
    with torch.no_grad():
        td = env.rollout(args.render_steps)

    batched = len(env.batch_size) > 0

    # Build a single-element state example so _tensordict_to_object can
    # infer field structure without a batch dimension.
    if batched:
        single_example = jax.tree_util.tree_map(lambda x: x[0], env._state_example)
    else:
        single_example = env._state_example

    torchrl_logger.info("Converting TensorDict states to JAX State objects.")
    states = []
    for t in range(args.render_steps):
        if batched:
            step_td = td[0, t].get("state")
        else:
            step_td = td[t].get("state")
        jax_state = _tensordict_to_object(step_td, single_example)
        states.append(jax_state)

    torchrl_logger.info(
        f"Rendering {len(states)} frames "
        f"({args.render_height}x{args.render_width}px)."
    )
    frames = env._env.render(
        states,
        height=args.render_height,
        width=args.render_width,
        camera=args.render_camera,
    )
    frames_np = [np.asarray(f) for f in frames]

    render_output = args.render_output
    if render_output is None:
        render_output = args.output_dir / "render.mp4"

    render_output = Path(render_output)

    try:
        import imageio

        render_output.parent.mkdir(parents=True, exist_ok=True)
        imageio.mimwrite(str(render_output), frames_np, fps=args.render_fps)
        torchrl_logger.info(f"Video saved to {render_output}.")
    except ImportError:
        torchrl_logger.warning(
            "imageio not found; saving frames as individual PNG files. "
            "Install imageio for MP4 export: pip install imageio[ffmpeg]."
        )
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError(
                "Neither imageio nor Pillow is available. "
                "Install at least one: pip install imageio[ffmpeg] or pip install pillow."
            ) from exc

        frame_dir = render_output.with_suffix("")
        frame_dir.mkdir(parents=True, exist_ok=True)
        for i, frame in enumerate(frames_np):
            Image.fromarray(frame).save(frame_dir / f"frame_{i:04d}.png")
        torchrl_logger.info(f"{len(frames_np)} PNG frames saved to {frame_dir}/.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    logging.basicConfig(level=getattr(logging, args_cli.log_level))
    torch.manual_seed(args_cli.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args_cli.seed)

    total_rollouts = args_cli.warmup_rollouts + args_cli.profile_rollouts
    args_cli.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = args_cli.output_dir / "collector_worker_{worker_idx}.json"
    worker_0_trace_path = args_cli.output_dir / "collector_worker_0.json"

    torchrl_logger.info(
        f"Creating MuJoCo Playground env '{args_cli.env_name}' "
        f"(batch_size={args_cli.batch_size}, device={args_cli.device})."
    )
    env = make_env(args_cli.env_name, args_cli.batch_size, args_cli.device)
    env.set_seed(args_cli.seed)
    policy = make_policy(env, args_cli.policy_mode)

    collector = Collector(
        env,
        policy,
        frames_per_batch=args_cli.frames_per_batch,
        total_frames=-1,
        trust_policy=args_cli.trust_policy,
        track_traj_ids=args_cli.track_traj_ids,
    )
    collector.enable_profile(
        workers=[0],
        num_rollouts=args_cli.profile_rollouts,
        warmup_rollouts=args_cli.warmup_rollouts,
        save_path=trace_path,
        activities=args_cli.activities,
        record_shapes=args_cli.record_shapes,
        profile_memory=args_cli.profile_memory,
        with_stack=args_cli.with_stack,
    )

    try:
        for idx, batch in enumerate(collector):
            torchrl_logger.info(
                f"Rollout {idx}: batch_size={tuple(batch.batch_size)}, "
                f"frames={batch.numel()}."
            )
            if idx + 1 >= total_rollouts:
                break
    finally:
        collector.disable_profile()
        collector.shutdown(close_env=False)

    torchrl_logger.info(f"Trace written to {worker_0_trace_path}.")

    if args_cli.render:
        save_visualization(env, args_cli)

    env.close()


if __name__ == "__main__":
    main()
