# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import torch

from torchrl.render.artifacts import write_render_artifact
from torchrl.render.checkpoint import (
    checkpoint_hash,
    infer_state_dict,
    load_checkpoint,
    save_render_checkpoint,
)
from torchrl.render.config import (
    CameraLayout,
    EnvBackendName,
    ExplorationMode,
    FrameBundle,
    key_to_string,
    parse_nested_key,
    RenderBackendName,
    RenderConfig,
    RenderEnvSpec,
    RenderFormat,
    RenderPolicySpec,
    RenderResult,
)
from torchrl.render.env import (
    add_step_counter,
    make_render_env,
    normalize_env,
    seed_env,
)
from torchrl.render.import_utils import call_with_supported_kwargs, import_from_string
from torchrl.render.policy import (
    load_render_policy,
    normalize_policy,
    TensorDictPolicyAdapter,
)
from torchrl.render.rollout import collect_render_rollouts

__all__ = [
    "CameraLayout",
    "EnvBackendName",
    "ExplorationMode",
    "FrameBundle",
    "RenderBackendName",
    "RenderConfig",
    "RenderEnvSpec",
    "RenderFormat",
    "RenderPolicySpec",
    "RenderResult",
    "TensorDictPolicyAdapter",
    "add_step_counter",
    "call_with_supported_kwargs",
    "checkpoint_hash",
    "collect_render_rollouts",
    "import_from_string",
    "infer_state_dict",
    "key_to_string",
    "load_checkpoint",
    "load_render_policy",
    "make_render_env",
    "normalize_env",
    "normalize_policy",
    "parse_nested_key",
    "render_policy",
    "save_render_checkpoint",
    "seed_env",
    "write_render_artifact",
]


def render_policy(config: RenderConfig) -> RenderResult:
    """Renders a policy according to ``config`` and writes the requested artifact.

    Args:
        config: Render configuration.

    Returns:
        The render result with trajectories, metadata, and artifact paths.
    """
    device = torch.device(config.policy_device or config.device)
    checkpoint = load_checkpoint(config.ckpt, map_location=device)
    digest = checkpoint_hash(config.ckpt)
    env = make_render_env(config, checkpoint=checkpoint)
    try:
        policy = load_render_policy(
            config, env, checkpoint=checkpoint, checkpoint_digest=digest
        )
        result = collect_render_rollouts(env, policy, config)
        return write_render_artifact(result, config)
    finally:
        close = getattr(env, "close", None)
        if callable(close):
            close()
