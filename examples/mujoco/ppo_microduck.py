# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Backend-neutral MicroDuck velocity task and a compact TorchRL PPO loop.

The MicroDuck MJCF is shared by all three :class:`~torchrl.envs.MujocoEnv`
backends. This example supplies a small commanded locomotion task around that
model: normalized joint-position actions, a 50-value proprioceptive
observation, and signed longitudinal-velocity tracking. A zero velocity command
requests standing, while positive and negative velocities command forward and
backward motion. It intentionally does not reproduce the richer walking task from
``microduck_rl`` (sensor history, delays, curricula, domain randomization, and
the BAM actuator model).

Run a short CPU training job from a TorchRL checkout::

    python examples/mujoco/ppo_microduck.py \
        --microduck-root /path/to/microduck_rl --smoke

Use ``--backend mjx`` or ``--backend mujoco-torch`` to keep the same task and
change only the simulator backend. The accompanying notebook demonstrates
native rendering and the interactive MuJoCo WASM viewer.
"""

from __future__ import annotations

import argparse
import importlib.util
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Literal

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn

from torchrl import torchrl_logger
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.envs import EnvBase, ExplorationType, MujocoEnv, set_exploration_type
from torchrl.envs.utils import step_mdp
from torchrl.modules import ProbabilisticActor, TanhNormal, ValueOperator
from torchrl.objectives import ClipPPOLoss
from torchrl.objectives.value import GAE

_has_mujoco = importlib.util.find_spec("mujoco") is not None

Backend = Literal["mujoco", "mjx", "mujoco-torch"]
NUM_JOINTS = 14
OBSERVATION_DIM = 3 + 3 + 2 + NUM_JOINTS * 3
VELOCITY_TRACKING_STD = 0.25


def resolve_microduck_scene(
    microduck_root: str | Path | None = None,
) -> Path:
    """Resolve MicroDuck's ``scene_walk.xml`` from a checkout or installation.

    Args:
        microduck_root: ``microduck_rl`` checkout, its Python package directory,
            or the scene XML itself. When omitted, an installed
            ``mjlab_microduck`` package and nearby ``microduck_rl`` checkouts are
            considered.

    Returns:
        Absolute path to ``scene_walk.xml``.

    Raises:
        FileNotFoundError: If the MicroDuck MJCF cannot be located.
    """
    candidates: list[Path] = []
    if microduck_root is not None:
        candidates.append(Path(microduck_root).expanduser())

    spec = importlib.util.find_spec("mjlab_microduck")
    if spec is not None and spec.origin is not None:
        candidates.append(Path(spec.origin).resolve().parent)

    cwd = Path.cwd()
    candidates.extend((cwd / "microduck_rl", cwd.parent / "microduck_rl"))
    suffixes = (
        Path("scene_walk.xml"),
        Path("robot/microduck/scene_walk.xml"),
        Path("mjlab_microduck/robot/microduck/scene_walk.xml"),
        Path("src/mjlab_microduck/robot/microduck/scene_walk.xml"),
    )
    attempted = []
    for candidate in candidates:
        paths = (
            (candidate,)
            if candidate.suffix == ".xml"
            else (candidate / suffix for suffix in suffixes)
        )
        for path in paths:
            attempted.append(path)
            if path.is_file():
                return path.resolve()
    detail = "\n".join(f"  - {path}" for path in attempted)
    raise FileNotFoundError(
        "Could not find MicroDuck's scene_walk.xml. Pass --microduck-root or "
        "install microduck_rl. Tried:\n" + detail
    )


def _load_stand_metadata(scene_path: Path) -> tuple[torch.Tensor, ...]:
    if not _has_mujoco:
        raise ImportError(
            "MicroDuckVelocityEnv requires the `mujoco` package to read MJCF "
            "metadata, including when another physics backend is selected."
        )
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    if (model.nq, model.nv, model.nu) != (21, 20, NUM_JOINTS):
        raise ValueError(
            "Expected the 14-actuator MicroDuck walking model with "
            f"(nq, nv, nu)=(21, 20, 14), got {(model.nq, model.nv, model.nu)}."
        )
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "STAND")
    if key_id < 0:
        raise ValueError("MicroDuck MJCF must define a `STAND` keyframe.")

    home_qpos = torch.from_numpy(model.key_qpos[key_id].copy())
    home_ctrl = torch.from_numpy(model.key_ctrl[key_id].copy())
    joint_ids = torch.from_numpy(model.actuator_trnid[:, 0].copy()).long()
    if (joint_ids < 0).any():
        raise ValueError("Every MicroDuck actuator must target a joint.")
    joint_limited = torch.from_numpy(model.jnt_limited[joint_ids.numpy()].copy()).bool()
    joint_range = torch.from_numpy(model.jnt_range[joint_ids.numpy()].copy())
    joint_low = torch.where(joint_limited, joint_range[:, 0], home_ctrl - torch.pi)
    joint_high = torch.where(joint_limited, joint_range[:, 1], home_ctrl + torch.pi)
    return home_qpos, home_ctrl, joint_low, joint_high


def _projected_gravity(quaternion: torch.Tensor) -> torch.Tensor:
    """Rotate world-frame gravity into the body frame for wxyz quaternions."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            -2.0 * (x * z - w * y),
            -2.0 * (y * z + w * x),
            -(1.0 - 2.0 * (x.square() + y.square())),
        ),
        dim=-1,
    )


def _body_frame_linear_velocity(
    quaternion: torch.Tensor,
    world_velocity: torch.Tensor,
) -> torch.Tensor:
    """Rotate a world-frame velocity into the body frame for wxyz quaternions."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w = quaternion[..., :1]
    vector = -quaternion[..., 1:]
    twice_cross = 2.0 * torch.cross(vector, world_velocity, dim=-1)
    return world_velocity + w * twice_cross + torch.cross(vector, twice_cross, dim=-1)


def _body_forward_vector(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the body x-axis expressed in the world frame."""
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    w, x, y, z = quaternion.unbind(-1)
    return torch.stack(
        (
            1.0 - 2.0 * (y.square() + z.square()),
            2.0 * (x * y + w * z),
            2.0 * (x * z - w * y),
        ),
        dim=-1,
    )


class MicroDuckVelocityEnv(MujocoEnv):
    """A compact commanded-velocity task for the MicroDuck MJCF.

    Actions are normalized offsets around the actuator targets in the
    ``STAND`` keyframe. Observations concatenate projected gravity, base
    angular velocity, measured and commanded body-frame longitudinal velocity,
    joint-position error, joint velocity, and the previous action. The reward
    prioritizes signed velocity tracking while uprightness and target height
    stabilize the motion.

    Args:
        microduck_root: ``microduck_rl`` checkout, package directory, or
            ``scene_walk.xml`` path.
        backend: MuJoCo physics backend.
        commanded_x_velocity: A fixed body-frame longitudinal velocity command,
            or a sequence sampled uniformly at every reset. The default exact
            zero and signed commands train standing, forward, and backward
            motion in one task.
        action_scale: Maximum position-target offset in radians for a unit
            normalized action.
        kwargs: Forwarded to :class:`~torchrl.envs.MujocoEnv`.
    """

    FRAME_SKIP = 4
    RESET_NOISE_SCALE = 0.02

    def __init__(
        self,
        microduck_root: str | Path | None = None,
        *,
        backend: Backend = "mujoco",
        commanded_x_velocity: float | Sequence[float] = (-0.3, 0.0, 0.3),
        action_scale: float = 0.35,
        **kwargs: Any,
    ):
        scene_path = resolve_microduck_scene(microduck_root)
        home_qpos, home_ctrl, joint_low, joint_high = _load_stand_metadata(scene_path)
        super().__init__(
            xml_path=scene_path,
            patch_xml=False,
            backend=backend,
            **kwargs,
        )
        self.scene_path = scene_path
        self.action_scale = float(action_scale)
        self._home_qpos = home_qpos.to(device=self.device, dtype=self.dtype)
        self._home_ctrl = home_ctrl.to(device=self.device, dtype=self.dtype)
        self._joint_low = joint_low.to(device=self.device, dtype=self.dtype)
        self._joint_high = joint_high.to(device=self.device, dtype=self.dtype)
        self._target_height = self._home_qpos[2].clone()
        self._previous_action = torch.zeros(
            self.num_envs, NUM_JOINTS, dtype=self.dtype, device=self.device
        )
        self._observation_action = self._previous_action.clone()
        command_values = torch.as_tensor(
            commanded_x_velocity, dtype=self.dtype, device=self.device
        )
        if command_values.ndim == 0:
            command_values = command_values.unsqueeze(0)
        if command_values.ndim != 1 or command_values.numel() == 0:
            raise ValueError(
                "commanded_x_velocity must be a scalar or a non-empty 1-D sequence."
            )
        if not torch.isfinite(command_values).all():
            raise ValueError("commanded_x_velocity values must be finite.")
        self._command_values = command_values
        self._commanded_x_velocity = torch.zeros(
            self.num_envs, 1, dtype=self.dtype, device=self.device
        )
        self.action_spec = Bounded(
            low=-1.0,
            high=1.0,
            shape=(self.num_envs, NUM_JOINTS),
            dtype=self.dtype,
            device=self.device,
        )

    def _make_obs_spec(self) -> Composite:
        return Composite(
            observation=Unbounded(
                shape=(self.num_envs, OBSERVATION_DIM),
                dtype=self.dtype,
                device=self.device,
            ),
            commanded_x_velocity=Unbounded(
                shape=(self.num_envs, 1),
                dtype=self.dtype,
                device=self.device,
            ),
            shape=(self.num_envs,),
            device=self.device,
        )

    def _build_obs_dict(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        observation = super()._build_obs_dict(state)
        observation["commanded_x_velocity"] = self._commanded_x_velocity.clone()
        return observation

    def _make_obs(self, state: TensorDictBase) -> torch.Tensor:
        qpos = state["qpos"].to(self.dtype)
        qvel = state["qvel"].to(self.dtype)
        body_velocity = _body_frame_linear_velocity(qpos[..., 3:7], qvel[..., :3])
        return torch.cat(
            (
                _projected_gravity(qpos[..., 3:7]),
                qvel[..., 3:6],
                body_velocity[..., :1],
                self._commanded_x_velocity,
                qpos[..., 7:] - self._home_qpos[7:],
                qvel[..., 6:],
                self._observation_action,
            ),
            dim=-1,
        )

    def _sample_initial_state(
        self,
        n: int,
        tensordict: TensorDictBase | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del tensordict
        qpos = self._home_qpos.unsqueeze(0).expand(n, -1).clone()
        qvel = torch.zeros(
            n, self._backend.nv, dtype=self._backend.qvel0.dtype, device=self.device
        )
        qpos = qpos.to(dtype=self._backend.qpos0.dtype)
        if self.reset_noise_scale > 0:
            qpos[..., :2] += torch.empty_like(qpos[..., :2]).uniform_(
                -self.reset_noise_scale,
                self.reset_noise_scale,
                generator=self.rng,
            )
            qpos[..., 7:] += torch.empty_like(qpos[..., 7:]).uniform_(
                -self.reset_noise_scale,
                self.reset_noise_scale,
                generator=self.rng,
            )
            qvel += torch.empty_like(qvel).uniform_(
                -self.reset_noise_scale,
                self.reset_noise_scale,
                generator=self.rng,
            )
        return qpos, qvel

    def _sample_command(
        self,
        tensordict: TensorDictBase | None,
    ) -> torch.Tensor:
        if tensordict is not None and "commanded_x_velocity" in tensordict.keys():
            command = tensordict["commanded_x_velocity"].to(
                device=self.device, dtype=self.dtype
            )
            if command.shape == self.batch_size:
                command = command.unsqueeze(-1)
            if command.shape != self.batch_size + torch.Size([1]):
                raise ValueError(
                    "A reset commanded_x_velocity must have shape "
                    f"{tuple(self.batch_size + torch.Size([1]))}, got "
                    f"{tuple(command.shape)}."
                )
            if not torch.isfinite(command).all():
                raise ValueError("A reset commanded_x_velocity must be finite.")
            return command
        indices = torch.randint(
            self._command_values.numel(),
            (self.num_envs,),
            generator=self.rng,
            device=self.device,
        )
        return self._command_values[indices].unsqueeze(-1)

    def _prepare_ctrl(self, action: torch.Tensor) -> torch.Tensor:
        action = action.clamp(-1.0, 1.0)
        target = self._home_ctrl + self.action_scale * action
        return target.clamp(self._joint_low, self._joint_high)

    def _compute_reward(
        self,
        state: TensorDictBase,
        action: torch.Tensor,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        del state
        qpos = next_state["qpos"].to(self.dtype)
        qvel = next_state["qvel"].to(self.dtype)
        projected_gravity = _projected_gravity(qpos[..., 3:7])
        # MuJoCo free-joint linear velocities are world-frame. Angular
        # velocities are already expressed in the local body frame.
        body_velocity = _body_frame_linear_velocity(qpos[..., 3:7], qvel[..., :3])
        velocity_tracking_reward = torch.exp(
            -(
                (body_velocity[..., 0] - self._commanded_x_velocity.squeeze(-1))
                / VELOCITY_TRACKING_STD
            ).square()
        )
        upright = (-projected_gravity[..., 2]).clamp(-1.0, 1.0)
        upright_reward = torch.exp(-4.0 * (1.0 - upright).square())
        height_reward = torch.exp(
            -((qpos[..., 2] - self._target_height) / 0.03).square()
        )
        lateral_velocity_cost = body_velocity[..., 1].square()
        roll_yaw_velocity_cost = qvel[..., [3, 5]].square().mean(dim=-1)
        joint_velocity_cost = qvel[..., 6:].square().mean(dim=-1)
        action_rate_cost = (action - self._previous_action).square().mean(dim=-1)
        return (
            2.0 * velocity_tracking_reward
            + 0.5 * upright_reward
            + 0.25 * height_reward
            - 0.1 * lateral_velocity_cost
            - 0.02 * roll_yaw_velocity_cost
            - 0.002 * joint_velocity_cost
            - 0.02 * action_rate_cost
        ).unsqueeze(-1)

    def _compute_done(
        self,
        state: TensorDictBase,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        del state
        qpos = next_state["qpos"].to(self.dtype)
        qvel = next_state["qvel"].to(self.dtype)
        upright = -_projected_gravity(qpos[..., 3:7])[..., 2]
        finite = torch.isfinite(qpos).all(dim=-1) & torch.isfinite(qvel).all(dim=-1)
        return (
            (qpos[..., 2] < 0.55 * self._target_height) | (upright < 0.35) | ~finite
        ).unsqueeze(-1)

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        self._observation_action = tensordict["action"].to(self.dtype)
        result = super()._step(tensordict)
        self._previous_action = self._observation_action.clone()
        return result

    def _on_reset_all(self, tensordict: TensorDictBase | None = None) -> None:
        self._previous_action.zero_()
        self._observation_action.zero_()
        self._commanded_x_velocity = self._sample_command(tensordict)

    def _on_reset_mask(
        self,
        mask: torch.Tensor,
        tensordict: TensorDictBase | None = None,
    ) -> None:
        mask = mask.unsqueeze(-1) if mask.ndim == 1 else mask
        self._previous_action = torch.where(
            mask, torch.zeros_like(self._previous_action), self._previous_action
        )
        self._observation_action = torch.where(
            mask, torch.zeros_like(self._observation_action), self._observation_action
        )
        self._commanded_x_velocity = torch.where(
            mask, self._sample_command(tensordict), self._commanded_x_velocity
        )

    def _index_extra_state(self, index: slice | torch.Tensor) -> dict[str, Any]:
        return {
            "previous_action": self._previous_action[index].clone(),
            "commanded_x_velocity": self._commanded_x_velocity[index].clone(),
        }

    def _load_indexed_extra_state(self, state: dict[str, Any]) -> None:
        self._previous_action = state["previous_action"].clone()
        self._observation_action = self._previous_action.clone()
        self._commanded_x_velocity = state["commanded_x_velocity"].clone()

    def _set_indexed_extra_state(
        self,
        index: slice | torch.Tensor,
        source: MujocoEnv,
    ) -> None:
        if not isinstance(source, MicroDuckVelocityEnv):
            raise TypeError(
                "MicroDuckVelocityEnv snapshots can only be restored from the same task."
            )
        self._previous_action[index] = source._previous_action.to(self.device)
        self._observation_action[index] = source._observation_action.to(self.device)
        self._commanded_x_velocity[index] = source._commanded_x_velocity.to(self.device)


def make_env(
    microduck_root: str | Path | None = None,
    *,
    backend: Backend = "mujoco",
    commanded_x_velocity: float | Sequence[float] = (-0.3, 0.0, 0.3),
    num_envs: int = 4,
    device: torch.device | str = "cpu",
    seed: int = 0,
    parallel: bool = False,
) -> EnvBase:
    """Build a batched MicroDuck commanded-velocity task.

    Native MuJoCo uses :class:`~torchrl.envs.SerialEnv` by default for a
    notebook-friendly, multiprocessing-free batch. MJX and ``mujoco-torch``
    batch directly in their respective array frameworks.
    """
    kwargs = {
        "microduck_root": microduck_root,
        "backend": backend,
        "commanded_x_velocity": commanded_x_velocity,
        "num_envs": num_envs,
        "device": torch.device(device),
        "seed": seed,
        "max_episode_steps": 500,
    }
    if backend == "mujoco":
        kwargs["parallel"] = parallel
    return MicroDuckVelocityEnv(**kwargs)


def make_models(
    env: EnvBase,
    *,
    device: torch.device | str = "cpu",
    hidden_size: int = 128,
) -> tuple[ProbabilisticActor, ValueOperator]:
    """Create the actor and critic used by the compact PPO example."""
    device = torch.device(device)
    action_dim = env.action_spec_unbatched.shape[-1]
    actor_net = nn.Sequential(
        nn.LazyLinear(hidden_size, device=device),
        nn.Tanh(),
        nn.Linear(hidden_size, hidden_size, device=device),
        nn.Tanh(),
        nn.Linear(hidden_size, 2 * action_dim, device=device),
        NormalParamExtractor(),
    )
    actor = ProbabilisticActor(
        module=TensorDictModule(
            actor_net,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        ),
        spec=env.action_spec,
        in_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        distribution_kwargs={
            "low": env.action_spec_unbatched.space.low.to(device),
            "high": env.action_spec_unbatched.space.high.to(device),
        },
        return_log_prob=True,
    ).to(device)
    critic = ValueOperator(
        nn.Sequential(
            nn.LazyLinear(hidden_size, device=device),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size, device=device),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, device=device),
        ),
        in_keys=["observation"],
    ).to(device)
    with torch.no_grad():
        fake_tensordict = env.fake_tensordict().to(device)
        actor(fake_tensordict)
        critic(fake_tensordict)
    return actor, critic


@torch.no_grad()
def evaluate_policy(
    env: MicroDuckVelocityEnv,
    policy: ProbabilisticActor,
    *,
    commanded_x_velocities: Sequence[float] = (-0.3, 0.0, 0.3),
    seeds: Sequence[int] = (0, 1, 2),
    steps: int = 500,
) -> list[dict[str, float]]:
    """Evaluate deterministic fixed-command episodes from controlled seeds.

    Each result reports return, tracking error, survival, episode length, and
    displacement along the duck's initial heading. Keeping results separated
    by signed command prevents aggregate reward from hiding a wrong-way policy.

    Args:
        env: A single-environment task used only for evaluation.
        policy: Policy to evaluate with deterministic actions.
        commanded_x_velocities: Fixed signed commands to evaluate.
        seeds: Reset seeds evaluated for every command.
        steps: Maximum number of steps in each episode.

    Returns:
        One metric dictionary per command and seed.
    """
    if env.num_envs != 1:
        raise ValueError("MicroDuck evaluation expects a single environment.")
    if steps < 1 or not commanded_x_velocities or not seeds:
        raise ValueError("Evaluation commands, seeds, and steps must be non-empty.")

    policy_device = next(policy.parameters()).device
    was_training = policy.training
    policy.eval()
    results = []
    try:
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            for command in commanded_x_velocities:
                for seed in seeds:
                    env.set_seed(seed)
                    reset_input = TensorDict(
                        {
                            "commanded_x_velocity": torch.full(
                                (1, 1),
                                float(command),
                                dtype=env.dtype,
                                device=env.device,
                            )
                        },
                        batch_size=env.batch_size,
                        device=env.device,
                    )
                    tensordict = env.reset(reset_input)
                    start_state = env.get_state()
                    start_qpos = start_state["qpos"][0].to(env.dtype)
                    initial_forward = _body_forward_vector(start_qpos[3:7])
                    episode_return = 0.0
                    tracking_error = 0.0
                    terminated = False
                    episode_length = 0
                    for _ in range(steps):
                        policy_input = tensordict.to(policy_device)
                        policy(policy_input)
                        tensordict["action"] = policy_input["action"].to(env.device)
                        transition = env.step(tensordict)
                        body_x_velocity = transition["next", "observation"][..., 6]
                        episode_return += float(transition["next", "reward"].sum())
                        tracking_error += float(
                            (body_x_velocity - float(command)).abs().sum()
                        )
                        terminated = bool(transition["next", "terminated"].any())
                        done = bool(transition["next", "done"].any())
                        episode_length += 1
                        tensordict = step_mdp(transition, keep_other=True)
                        if done:
                            break
                    end_qpos = env.get_state()["qpos"][0].to(env.dtype)
                    displacement = torch.dot(
                        end_qpos[:2] - start_qpos[:2], initial_forward[:2]
                    )
                    results.append(
                        {
                            "commanded_x_velocity": float(command),
                            "seed": float(seed),
                            "episode_return": episode_return,
                            "tracking_error": tracking_error / episode_length,
                            "survived": float(not terminated),
                            "episode_length": float(episode_length),
                            "signed_displacement": float(displacement),
                        }
                    )
    finally:
        policy.train(was_training)
    return results


def train_ppo(
    env: EnvBase,
    actor: ProbabilisticActor,
    critic: ValueOperator,
    *,
    iterations: int = 10,
    rollout_steps: int = 64,
    epochs: int = 4,
    minibatch_size: int = 128,
    learning_rate: float = 3e-4,
    evaluation_env: MicroDuckVelocityEnv | None = None,
    evaluation_interval: int | None = None,
    evaluation_commands: Sequence[float] = (-0.3, 0.0, 0.3),
    evaluation_seeds: Sequence[int] = (0, 1, 2),
    evaluation_steps: int = 500,
    best_checkpoint_path: str | Path | None = None,
) -> list[dict[str, float]]:
    """Train ``actor`` and ``critic`` with a small synchronous PPO loop.

    The function is intentionally compact enough for a notebook. It returns
    per-iteration metrics instead of owning an experiment tracker. When an
    evaluation environment and interval are supplied, deterministic fixed-
    command evaluation selects and restores the best policy. The same best
    actor and critic state can optionally be persisted to disk.
    """
    if min(iterations, rollout_steps, epochs, minibatch_size) < 1:
        raise ValueError("PPO loop sizes must all be positive.")
    if evaluation_interval is not None and evaluation_interval < 1:
        raise ValueError("evaluation_interval must be positive when provided.")
    if evaluation_interval is not None and evaluation_env is None:
        raise ValueError("evaluation_interval requires an evaluation_env.")
    if best_checkpoint_path is not None and evaluation_interval is None:
        raise ValueError("best_checkpoint_path requires periodic evaluation.")
    device = next(actor.parameters()).device
    advantage = GAE(
        gamma=0.99,
        lmbda=0.95,
        value_network=critic,
        average_gae=True,
        device=device,
    )
    loss_module = ClipPPOLoss(
        actor_network=actor,
        critic_network=critic,
        clip_epsilon=0.2,
        entropy_bonus=True,
        entropy_coeff=0.001,
        critic_coeff=1.0,
        loss_critic_type="smooth_l1",
        normalize_advantage=True,
    )
    optimizer = torch.optim.Adam(loss_module.parameters(), lr=learning_rate)
    history = []
    best_evaluation_return = -float("inf")
    best_actor_state = None
    best_critic_state = None
    checkpoint_path = (
        Path(best_checkpoint_path) if best_checkpoint_path is not None else None
    )
    for iteration in range(iterations):
        with torch.no_grad():
            batch = env.rollout(
                rollout_steps,
                actor,
                auto_reset=True,
                break_when_any_done=False,
                return_contiguous=True,
            ).to(device)
            advantage(batch)
        flat_batch = batch.reshape(-1)
        losses = []
        for _ in range(epochs):
            order = torch.randperm(flat_batch.numel(), device=device)
            for indices in order.split(minibatch_size):
                sample = flat_batch[indices]
                loss_values = loss_module(sample)
                total_loss = (
                    loss_values["loss_objective"]
                    + loss_values["loss_critic"]
                    + loss_values["loss_entropy"]
                )
                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                nn.utils.clip_grad_norm_(loss_module.parameters(), 1.0)
                optimizer.step()
                losses.append(total_loss.detach())
        metrics = {
            "iteration": float(iteration + 1),
            "reward": float(batch["next", "reward"].mean()),
            "loss": float(torch.stack(losses).mean()),
        }
        should_evaluate = evaluation_interval is not None and (
            (iteration + 1) % evaluation_interval == 0 or iteration + 1 == iterations
        )
        if should_evaluate:
            evaluation = evaluate_policy(
                evaluation_env,
                actor,
                commanded_x_velocities=evaluation_commands,
                seeds=evaluation_seeds,
                steps=evaluation_steps,
            )
            mean_return = sum(row["episode_return"] for row in evaluation) / len(
                evaluation
            )
            metrics.update(
                {
                    "evaluation_return": mean_return,
                    "evaluation_tracking_error": sum(
                        row["tracking_error"] for row in evaluation
                    )
                    / len(evaluation),
                    "evaluation_survival_rate": sum(
                        row["survived"] for row in evaluation
                    )
                    / len(evaluation),
                    "evaluation_episode_length": sum(
                        row["episode_length"] for row in evaluation
                    )
                    / len(evaluation),
                }
            )
            for row in evaluation:
                torchrl_logger.info(
                    "MicroDuck evaluation command=%+.3f seed=%d: "
                    "return=%+.4f tracking_error=%.4f survived=%d "
                    "length=%d displacement=%+.4f",
                    row["commanded_x_velocity"],
                    int(row["seed"]),
                    row["episode_return"],
                    row["tracking_error"],
                    int(row["survived"]),
                    int(row["episode_length"]),
                    row["signed_displacement"],
                )
            if mean_return > best_evaluation_return:
                best_evaluation_return = mean_return
                best_actor_state = deepcopy(actor.state_dict())
                best_critic_state = deepcopy(critic.state_dict())
                if checkpoint_path is not None:
                    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {
                            "iteration": iteration + 1,
                            "evaluation_return": mean_return,
                            "evaluation": evaluation,
                            "actor": best_actor_state,
                            "critic": best_critic_state,
                        },
                        checkpoint_path,
                    )
        history.append(metrics)
        torchrl_logger.info(
            "MicroDuck PPO iteration %d: reward=%+.4f loss=%.4f",
            iteration + 1,
            metrics["reward"],
            metrics["loss"],
        )
    if best_actor_state is not None and best_critic_state is not None:
        actor.load_state_dict(best_actor_state)
        critic.load_state_dict(best_critic_state)
    return history


@torch.no_grad()
def collect_qpos_trajectory(
    env: MicroDuckVelocityEnv,
    policy: ProbabilisticActor,
    *,
    steps: int = 300,
    commanded_x_velocity: float = 0.0,
) -> TensorDict:
    """Collect a single-env qpos trajectory for native or WASM rendering."""
    if env.num_envs != 1:
        raise ValueError("qpos rendering expects a single MicroDuckVelocityEnv.")
    reset_input = TensorDict(
        {
            "commanded_x_velocity": torch.full(
                (1, 1),
                float(commanded_x_velocity),
                dtype=env.dtype,
                device=env.device,
            )
        },
        batch_size=env.batch_size,
        device=env.device,
    )
    tensordict = env.reset(reset_input)
    qpos = [env.get_state()["qpos"].squeeze(0).cpu()]
    for _ in range(steps):
        policy(tensordict)
        transition = env.step(tensordict)
        qpos.append(env.get_state()["qpos"].squeeze(0).cpu())
        tensordict = step_mdp(transition, keep_other=True)
        if bool(transition["next", "done"].any()):
            break
    trajectory = torch.stack(qpos)
    return TensorDict({"qpos": trajectory}, batch_size=(trajectory.shape[0],))


def parse_args(args: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--microduck-root", type=Path)
    parser.add_argument(
        "--backend",
        choices=("mujoco", "mjx", "mujoco-torch"),
        default="mujoco",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--rollout-steps", type=int, default=64)
    parser.add_argument(
        "--commanded-x-velocity",
        action="append",
        type=float,
        dest="commanded_x_velocities",
        help=(
            "Velocity command to sample at reset. Repeat for multiple commands; "
            "defaults to -0.3, 0.0, and 0.3 m/s."
        ),
    )
    parser.add_argument(
        "--evaluation-interval",
        type=int,
        help="Run deterministic fixed-command evaluation every N iterations.",
    )
    parser.add_argument("--evaluation-steps", type=int, default=500)
    parser.add_argument("--best-checkpoint-path", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(args)


def main(args: argparse.Namespace) -> None:
    if args.smoke:
        args.iterations = 1
        args.rollout_steps = 8
        args.num_envs = 1
        args.evaluation_interval = None
    torch.manual_seed(args.seed)
    commands = args.commanded_x_velocities or (-0.3, 0.0, 0.3)
    env = make_env(
        args.microduck_root,
        backend=args.backend,
        commanded_x_velocity=commands,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
    )
    evaluation_env = None
    try:
        actor, critic = make_models(env, device=args.device)
        if args.evaluation_interval is not None:
            evaluation_env = MicroDuckVelocityEnv(
                args.microduck_root,
                backend=args.backend,
                commanded_x_velocity=commands,
                num_envs=1,
                device=torch.device(args.device),
                seed=args.seed,
                max_episode_steps=500,
            )
        train_ppo(
            env,
            actor,
            critic,
            iterations=args.iterations,
            rollout_steps=args.rollout_steps,
            evaluation_env=evaluation_env,
            evaluation_interval=args.evaluation_interval,
            evaluation_commands=commands,
            evaluation_steps=args.evaluation_steps,
            best_checkpoint_path=args.best_checkpoint_path,
        )
    finally:
        if evaluation_env is not None:
            evaluation_env.close()
        env.close()


if __name__ == "__main__":
    main(parse_args())
