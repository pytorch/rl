"""
MicroDuck skills in an external game
====================================

What you will learn
-------------------

Load a frozen locomotion policy, give each duck independent controller memory,
and train a skill selector with TorchRL's collector and PPOTrainer. Football
provides a short walkthrough of the optional
`six-game zoo <https://github.com/vmoens/rl-zoo/blob/codex/microduck-games/docs/catalog.md>`_.
Game rules, scenes, opponents and recipes live there. Skills, locomotion
training, controllers and reusable training components remain in TorchRL.

Install the matching source revisions listed in the zoo's README. This
tutorial runs on native MuJoCo, downloads the pinned robot assets and walker,
and completes a small pipeline check. It does not demonstrate learned football.
When the optional zoo is absent, the documentation renders without executing
the game. The dedicated integration CI installs it and executes these cells.
"""

from __future__ import annotations

import importlib.util

import torch

from torchrl.collectors import Evaluator
from torchrl.envs import load_microduck_walker, microduck_skill_env
from torchrl.envs.utils import check_env_specs, ExplorationType, set_exploration_type

_has_zoo = importlib.util.find_spec("torchrl_zoo") is not None
if _has_zoo:
    from torchrl_zoo.microduck import MicroDuckFootballEnv
    from torchrl_zoo.microduck.football import (
        football_metrics,
        make_env,
        make_models,
        make_trainer,
    )

# %%
# Load the published controller
# -----------------------------
#
# The checkpoint's ordered tasks define the selector's action indices. This
# historical seven-skill bundle must keep its matching walker; the nine-skill
# recipes use a separate immutable bundle. Loading validates the hash and
# action scale and freezes the actor. It never imports training examples.

walker_url = (
    "https://huggingface.co/torchrl/microduck-skills/resolve/"
    "8c31e2696520c402980a723b37d025e594197d9d/walker.ckpt"
)
walker_hash = "cbb5023d70bac278b3d066914289698c4ca53c6c8502ee39f5285fe51cf62c31"

if _has_zoo:
    torch.manual_seed(0)
    walker, tasks = load_microduck_walker(
        walker_url, sha256=walker_hash, action_scale=1.0
    )
    env = microduck_skill_env(
        MicroDuckFootballEnv(
            download=True,
            backend="mujoco",
            players_per_team=1,
            num_envs=1,
            parallel=False,
            max_episode_steps=30,
            action_scale=1.0,
        ),
        walker,
        tasks,
        steps=5,
    )
    check_env_specs(env)

# %%
# Collect, optimize and evaluate
# -----------------------------
#
# Each action chooses one skill per duck. The controller executes five physics
# control steps, sums rewards, stops at a terminal transition and maintains
# each duck's recurrent memory. The zoo factory constructs a TorchRL Collector,
# ClipPPOLoss and PPOTrainer with the matching multi-agent keys. PPOTrainer
# owns the optimization loop; zoo hooks handle game metrics and evaluation.
# Truncations bootstrap values, while true game endings stop bootstrapping.

if _has_zoo:
    config = {
        "env": {
            "download": True,
            "backend": "mujoco",
            "num_envs": 1,
            "parallel": False,
            "players_per_team": 1,
            "max_episode_steps": 30,
        }
    }
    actor, critic = make_models(env, hidden_size=16, depth=1)
    evaluator = Evaluator(
        make_env(config),
        actor,
        num_trajectories=1,
        max_steps=6,
        metrics_fn=football_metrics,
        reward_keys=("next", "agents", "reward"),
        log_prefix="evaluation",
    )
    trainer = make_trainer(
        env,
        actor,
        critic,
        total_frames=40,
        frames_per_batch=20,
        minibatch_size=10,
        epochs=1,
        evaluator=evaluator,
        evaluation_interval=1,
    )
    try:
        trainer.train()
        metrics = trainer.game_hooks.history[-1]
    finally:
        evaluator.shutdown()
        trainer.collector.shutdown()

# %%
# Render and continue from a recipe
# ---------------------------------
#
# Rendering uses normal importable factories. The zoo's full recipe exports
# portable best/latest actors and a separate resumable trainer checkpoint.
# A historical actor export is a warm start; it contains no optimizer history.
# Native MuJoCo resumes training at a fresh episode boundary.

if _has_zoo:
    filmed = make_env(config, from_pixels=True, render_width=160, render_height=90)
    try:
        with torch.no_grad(), set_exploration_type(ExplorationType.DETERMINISTIC):
            rollout = filmed.rollout(3, actor)
        frames = rollout["next", "pixels"]
    finally:
        filmed.close()

# %%
# The common game command composes Hydra groups::
#
#    python -m torchrl_zoo.microduck.train game=football observations=state \
#        algorithm=ppo runtime=macbook env.download=true smoke=true
#
# Render a full recipe's selector export with the existing rlrender CLI::
#
#    rlrender --ckpt microduck_football_best.ckpt \
#        --env torchrl_zoo.microduck.football:make_env \
#        --policy torchrl_zoo.microduck.football:make_render_policy \
#        --env-kwargs '{"download": true, "num_envs": 1, "parallel": false}' \
#        --format mp4 --out football.mp4
#
# Conclusion
# ----------
#
# The game can evolve independently while consuming TorchRL's installed skill
# controller, collection, PPO training, checkpoint and rendering components.
# Check the catalog's evidence labels before treating a game as evaluated.
#
# Further reading
# ---------------
#
# - :doc:`microduck_skills` for locomotion training and skill composition.
# - :doc:`multiagent_ppo` for PPO's multi-agent data layout.
# - `Zoo recipes and game rules <https://github.com/vmoens/rl-zoo>`_.
