# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Train MicroDuck skills, then a waypoint policy, using PPOTrainer at both levels.

Run from the repository root with ``python -m examples.microduck.train_skills``.
The tutorial imports the same task and network definitions without launching workers.
"""

from __future__ import annotations

import argparse
import functools as ft
import json
from pathlib import Path

import torch

from examples.microduck.ppo_mujoco import (
    evaluation_metrics,
    make_env,
    make_evaluator,
    make_models,
    make_render_policy,
    make_tasks,
    save_checkpoint,
)
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule
from torch.distributions import Categorical
from torchrl.checkpoint import Checkpoint, GlobalRNGState
from torchrl.collectors import Evaluator
from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase, MicroDuckController, MicroDuckEnv
from torchrl.envs.transforms import ClosedLoopMultiAction
from torchrl.modules import MLP, ProbabilisticActor
from torchrl.record.loggers import CSVLogger
from torchrl.render import load_checkpoint
from torchrl.trainers.algorithms import PPOTrainer

# These ordered definitions are also saved in walker.ckpt. Never reorder them
# when loading weights: task_id indexes a learned embedding.
SKILL_PRESETS = [
    {"preset": "standing_task"},
    {"preset": "tracking_task", "speed": 0.2},
    {"preset": "tracking_task", "speed": -0.2},
    {"preset": "sidestep_task", "speed": 0.15},
    {"preset": "sidestep_task", "speed": -0.15},
    {"preset": "jump_task", "weight": 3.0},
]


class WaypointMicroDuck(MicroDuckEnv):
    """Tutorial task: approach (0.5, 0.3) metres, terminating on arrival or a fall."""

    goal = (0.5, 0.3)

    def _make_obs_spec(self) -> Composite:
        spec = super()._make_obs_spec()
        spec["observation"] = Unbounded(
            (self.num_envs, self.OBSERVATION_DIM + 6),
            dtype=self.dtype,
            device=self.device,
        )
        return spec

    def _build_obs_dict(self, state: TensorDictBase) -> dict[str, torch.Tensor]:
        obs = super()._build_obs_dict(state)
        qpos = state["qpos"].to(self.dtype)
        delta = qpos.new_tensor(self.goal) - qpos[..., :2]
        obs["observation"] = torch.cat(
            (obs["observation"], delta, qpos[..., 3:7]), dim=-1
        )
        return obs

    def _compute_reward(
        self,
        state: TensorDictBase,
        action: torch.Tensor,
        next_state: TensorDictBase,
    ) -> torch.Tensor:
        before, after = state["qpos"][..., :2], next_state["qpos"][..., :2]
        goal = after.new_tensor(self.goal)
        old_distance = (before - goal).norm(dim=-1, keepdim=True)
        distance = (after - goal).norm(dim=-1, keepdim=True)
        fallen = super()._compute_done(state, next_state)
        return (
            10 * (old_distance - distance)
            + (distance < 0.05).to(self.dtype)
            - fallen.to(self.dtype)
            - 0.001
        )

    def _compute_done(
        self, state: TensorDictBase, next_state: TensorDictBase
    ) -> torch.Tensor:
        position = next_state["qpos"][..., :2]
        distance = (position - position.new_tensor(self.goal)).norm(
            dim=-1, keepdim=True
        )
        return super()._compute_done(state, next_state) | (distance < 0.05)


def make_navigation_models(
    env: EnvBase,
) -> tuple[ProbabilisticActor, TensorDictModule]:
    """A categorical skill selector and a separate local value MLP."""
    obs_dim = env.observation_spec["observation"].shape[-1]
    actor = ProbabilisticActor(
        TensorDictModule(
            MLP(
                in_features=obs_dim,
                out_features=env.full_action_spec[env.action_key].n,
                num_cells=[64, 64],
            ),
            in_keys=["observation"],
            out_keys=["logits"],
        ),
        in_keys=["logits"],
        out_keys=[env.action_key],
        spec=env.full_action_spec_unbatched,
        distribution_class=Categorical,
        return_log_prob=True,
    )
    critic = TensorDictModule(
        MLP(in_features=obs_dim, out_features=1, num_cells=[64, 64]),
        in_keys=["observation"],
        out_keys=["state_value"],
    )
    return actor, critic


def navigation_metrics(trajectories: TensorDictBase) -> dict[str, float]:
    lengths = trajectories["collector", "mask"].sum(-1)
    distance = trajectories["next", "observation"][..., -6:-4].norm(dim=-1)
    final_distance = distance.gather(-1, (lengths - 1).unsqueeze(-1))
    return {
        "arrival_rate": (final_distance < 0.05).float().mean().item(),
        "final_distance_m": final_distance.mean().item(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("microduck-training"))
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--low-level-frames", type=int, default=10_000_000)
    parser.add_argument("--high-level-frames", type=int, default=1_000_000)
    parser.add_argument("--walker-checkpoint", type=Path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--smoke", action="store_true", help="64 steps per stage, one simulator"
    )
    args = parser.parse_args()
    if args.smoke:
        args.num_envs = 1
        args.low_level_frames = args.high_level_frames = 64
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.manual_seed(0)
    (output_dir / "run.json").write_text(json.dumps(vars(args), default=str, indent=2))

    if args.walker_checkpoint:
        payload = load_checkpoint(args.walker_checkpoint)
    else:
        env_config = {
            "backend": "mujoco",
            "device": "cpu",
            "download": True,
            "num_envs": args.num_envs,
            "parallel": args.num_envs > 1,
            "max_episode_steps": 32 if args.smoke else 500,
            "action_scale": 1.0,
            "tasks": SKILL_PRESETS,
        }
        policy_kwargs = {
            "hidden_size": 32 if args.smoke else 128,
            "policy_head": "gaussian",
            "initial_policy_scale": 1.0,
        }
        env = make_env(env_config)
        actor, critic = make_models(env, **policy_kwargs)
        low_trainer = PPOTrainer.from_env(
            env,
            actor=actor,
            critic=critic,
            total_frames=args.low_level_frames,
            frames_per_batch=64 if args.smoke else args.num_envs * 1024,
            minibatch_size=64 if args.smoke else args.num_envs * 128,
            sub_traj_len=64 if args.smoke else 128,
            gae_kwargs={"average_gae": True, "group_key": "task_id"},
            loss_kwargs={"entropy_coeff": 0.01},
            num_epochs=1 if args.smoke else 5,
            progress_bar=not args.smoke,
            logger=CSVLogger("low_level", log_dir=str(output_dir)),
            log_interval=1,
            log_timings=True,
            checkpoint=Checkpoint(rng=GlobalRNGState()),
            save_trainer_file=output_dir / "low_level.trainer",
            save_trainer_interval=100_000,
        )
        try:
            if args.resume:
                low_trainer.load_from_file(output_dir / "low_level.trainer")
            low_trainer.train()
            path = save_checkpoint(
                output_dir / "walker.ckpt",
                actor,
                critic,
                transitions=low_trainer.collected_frames,
                policy_kwargs=policy_kwargs,
                metrics={},
                config={"env": env_config},
            )
            payload = load_checkpoint(path)
        finally:
            low_trainer.collector.shutdown()

    model_env = make_env(
        checkpoint=payload,
        cfg={"backend": "mujoco", "device": "cpu", "parallel": False},
        download=True,
        num_envs=1,
    )
    try:
        walker = make_render_policy(model_env, checkpoint=payload)
        walker.load_state_dict(payload["model_state_dict"])
        walker.eval().requires_grad_(False)
    finally:
        model_env.close()
    skill_tasks = torch.stack(make_tasks(payload["config"]["env"]["tasks"]))

    skill_results = []
    jump_index = list(MicroDuckEnv.REWARD_TERMS).index("jump")
    for task_id, task in enumerate(skill_tasks.unbind(0)):
        evaluator = make_evaluator(
            make_env(
                checkpoint=payload,
                cfg={"backend": "mujoco", "task_id": task_id, "diagnostics": True},
                download=True,
                num_envs=1,
                parallel=False,
            ),
            walker,
            label=str(task_id),
            jumping=bool(task.reward_weights[jump_index] > 0),
            num_episodes=1 if args.smoke else 8,
            steps=32 if args.smoke else 500,
        )
        try:
            skill_results.append(evaluator.evaluate())
        finally:
            evaluator.shutdown()
    (output_dir / "skills.json").write_text(
        json.dumps(evaluation_metrics(skill_results), indent=2)
    )

    task_factory = ft.partial(
        WaypointMicroDuck,
        download=True,
        backend="mujoco",
        tasks=MicroDuckEnv.standing_task(),
        action_scale=payload["config"]["env"]["action_scale"],
        max_episode_steps=64 if args.smoke else 500,
        seed=0,
    )
    controller = MicroDuckController(
        walker,
        skill_tasks,
        group_key=None,
        reset_key=None,
        control_period_s=0.02,
    )
    training_env = ClosedLoopMultiAction.from_env(
        task_factory(num_envs=args.num_envs, parallel=args.num_envs > 1),
        controller,
        steps=5,
    )
    actor, critic = make_navigation_models(training_env)
    high_trainer = PPOTrainer.from_env(
        training_env,
        actor=actor,
        critic=critic,
        total_frames=args.high_level_frames,
        frames_per_batch=32 if args.smoke else args.num_envs * 128,
        minibatch_size=256,
        num_epochs=1 if args.smoke else 4,
        loss_kwargs={"entropy_coeff": 0.01},
        gamma=0.99,
        lmbda=0.95,
        progress_bar=not args.smoke,
        logger=CSVLogger("high_level", log_dir=str(output_dir)),
        log_interval=1,
        log_timings=True,
        checkpoint=Checkpoint(rng=GlobalRNGState()),
        save_trainer_file=output_dir / "high_level.trainer",
        save_trainer_interval=10_000,
    )
    try:
        # A high-level checkpoint must be paired with its original walker.
        if args.resume and args.walker_checkpoint:
            high_trainer.load_from_file(output_dir / "high_level.trainer")
        high_trainer.train()
    finally:
        high_trainer.collector.shutdown()

    evaluator = Evaluator(
        ClosedLoopMultiAction.from_env(task_factory(), controller, steps=5),
        actor,
        num_trajectories=1 if args.smoke else 32,
        max_steps=20 if args.smoke else 100,
        metrics_fn=navigation_metrics,
    )
    try:
        (output_dir / "navigation.json").write_text(
            json.dumps(evaluator.evaluate(), indent=2)
        )
    finally:
        evaluator.shutdown()


if __name__ == "__main__":
    main()
