"""
Render policy rollouts with rlrender
====================================

**Author**: `TorchRL contributors <https://github.com/pytorch/rl>`_

.. _rlrender_tuto:

This tutorial shows how to describe a renderable policy and environment with two
small factory functions, then call :func:`torchrl.render.render_policy` or the
``rlrender`` command to create a reproducible artifact.
"""

#####################################
# What you will learn
# -------------------
#
# This tutorial covers three pieces:
#
# - a policy factory that receives :class:`~torchrl.render.RenderPolicySpec`,
# - an environment factory that receives :class:`~torchrl.render.RenderEnvSpec`,
# - a :class:`~torchrl.render.RenderConfig` that can also be expressed as CLI
#   flags.
#
# The example writes JSONL so it works without optional video dependencies. Use
# ``uv run --extra rendering rlrender ...`` when writing MP4, GIF, PNG frames, or
# YAML-backed configs.

from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from torchrl.envs import PendulumEnv
from torchrl.render import render_policy, RenderConfig, RenderEnvSpec, RenderPolicySpec


#####################################
# Factory functions
# -----------------
#
# The environment factory should return a TorchRL environment when possible.
# ``rlrender`` adds a :class:`~torchrl.envs.transforms.StepCounter` when
# ``max_steps`` is set, so the factory can stay close to the training setup.


def make_env(spec: RenderEnvSpec):
    return PendulumEnv(device=spec.device)


#####################################
# The policy factory can load arbitrary project state. For this short example,
# it returns a deterministic zero-action policy and ignores the empty checkpoint.


def make_policy(spec: RenderPolicySpec):
    def policy(tensordict):
        tensordict.set("action", torch.zeros(1, device=spec.device))
        return tensordict

    return policy


#####################################
# Programmatic rendering
# ----------------------
#
# The same configuration can be provided by a Python object or command-line
# flags. The JSONL format stores metadata and rollout events without requiring
# image or video encoders.

with tempfile.TemporaryDirectory() as tmpdir:
    tmpdir = Path(tmpdir)
    ckpt = tmpdir / "policy.pt"
    torch.save({}, ckpt)
    out = tmpdir / "render.jsonl"
    config = RenderConfig(
        ckpt=ckpt,
        policy=make_policy,
        env=make_env,
        max_steps=3,
        format="jsonl",
        out=out,
        auto_load_policy=False,
        overwrite=True,
    )
    result = render_policy(config)
    artifact_path = result.artifact_path
    metadata = result.metadata


#####################################
# Equivalent command-line shape
# -----------------------------
#
# In a project, place the two factories in an importable module and call:
#
# .. code-block:: bash
#
#     rlrender \
#       --ckpt ./policy.pt \
#       --policy project.render:make_policy \
#       --env project.render:make_env \
#       --max-steps 500 \
#       --num-trajs 4 \
#       --format mp4 \
#       --out ./renders/policy.mp4
#
# ``rlrender`` imports trusted Python code and loads trusted checkpoints by
# design. Only run it with factories and checkpoints you would execute directly.

#####################################
# Conclusion and further reading
# ------------------------------
#
# ``rlrender`` is a thin application layer over reusable TorchRL APIs. The MVP
# captures frames from TensorDict pixel entries or ``env.render()`` and writes
# metadata with every artifact. For API details, see :ref:`ref_render` and the
# recorder utilities in :ref:`Environment-Recorders`.
