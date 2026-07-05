# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

import torch
from tensordict import TensorDictBase

from torchrl._utils import logger as torchrl_logger
from torchrl.render.checkpoint import checkpoint_hash, infer_state_dict, load_checkpoint
from torchrl.render.config import RenderConfig, RenderPolicySpec
from torchrl.render.import_utils import call_with_supported_kwargs, import_from_string

__all__ = ["TensorDictPolicyAdapter", "load_render_policy", "normalize_policy"]


class TensorDictPolicyAdapter:
    """Adapts plain tensor policies to a TensorDict policy callable.

    Args:
        policy: Policy object or callable.
        obs_key: Observation key used for tensor-only policies.
        action_key: Action key written when tensor actions are returned.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.render.policy import TensorDictPolicyAdapter
        >>> def policy(obs):
        ...     if not torch.is_tensor(obs):
        ...         raise TypeError("expected tensor input")
        ...     return obs + 1
        >>> adapter = TensorDictPolicyAdapter(policy, "obs", ("agent", "action"))
        >>> td = TensorDict({"obs": torch.zeros(1)}, [])
        >>> adapter(td).get(("agent", "action"))
        tensor([1.])
    """

    def __init__(self, policy: Any, obs_key: Any, action_key: Any) -> None:
        self.policy = policy
        self.obs_key = obs_key
        self.action_key = action_key

    def __call__(self, tensordict: TensorDictBase) -> TensorDictBase:
        try:
            output = self.policy(tensordict)
        except Exception as td_err:
            try:
                output = self.policy(tensordict.get(self.obs_key))
            except Exception:
                raise td_err
        if isinstance(output, TensorDictBase):
            return output
        if isinstance(output, dict):
            tensordict.update(output)
            return tensordict
        tensordict.set(self.action_key, output)
        return tensordict


def load_render_policy(
    config: RenderConfig,
    env: Any | None = None,
    *,
    checkpoint: Any | None = None,
    checkpoint_digest: str | None = None,
) -> Any:
    """Builds and prepares a policy for rendering.

    Args:
        config: Render configuration.
        env: Optional environment used to expose specs to the policy factory.

    Keyword Args:
        checkpoint: Checkpoint payload already loaded from ``config.ckpt``.
            When ``None``, the checkpoint is loaded here.
        checkpoint_digest: SHA256 digest of the checkpoint file. When ``None``,
            the digest is computed here.

    Returns:
        A TensorDict-compatible policy callable or module.
    """
    factory = (
        import_from_string(config.policy)
        if isinstance(config.policy, str)
        else config.policy
    )
    if not callable(factory):
        raise TypeError(
            f"Policy factory must be callable, got {type(factory).__name__}."
        )
    device = torch.device(config.policy_device or config.device)
    if checkpoint is None:
        checkpoint = load_checkpoint(config.ckpt, map_location=device)
    digest = checkpoint_digest or checkpoint_hash(config.ckpt)
    env_specs = getattr(env, "specs", None)
    spec = RenderPolicySpec(
        ckpt_path=config.ckpt,
        checkpoint=checkpoint,
        checkpoint_hash=digest,
        device=device,
        env_specs=env_specs,
        policy_kwargs=dict(config.policy_kwargs),
        config=config,
    )
    kwargs = {
        "spec": spec,
        "ckpt_path": config.ckpt,
        "checkpoint": checkpoint,
        "device": device,
        "env": env,
        "env_specs": env_specs,
        "config": config,
        "policy_kwargs": dict(config.policy_kwargs),
        **config.policy_kwargs,
    }
    policy = call_with_supported_kwargs(factory, spec, kwargs)
    if config.auto_load_policy:
        _load_state_dict_if_possible(policy, checkpoint, config)
    if config.policy_eval:
        _set_policy_eval(policy)
    return normalize_policy(policy, config)


def normalize_policy(
    policy: Any, config: RenderConfig
) -> TensorDictPolicyAdapter | Any:
    """Normalizes a policy into a TensorDict-compatible callable."""
    if isinstance(policy, TensorDictPolicyAdapter):
        return policy
    return TensorDictPolicyAdapter(policy, config.obs_key, config.action_key)


def _load_state_dict_if_possible(
    policy: Any, checkpoint: Any, config: RenderConfig
) -> None:
    loader = getattr(policy, "load_state_dict", None)
    if loader is None:
        return
    try:
        state_dict = infer_state_dict(
            checkpoint, key=config.state_dict_key or config.checkpoint_key
        )
    except Exception as err:
        torchrl_logger.warning(
            "rlrender could not infer a state dict for automatic policy loading: %s",
            err,
        )
        return
    loader(state_dict, strict=config.strict_load)


def _set_policy_eval(policy: Any) -> None:
    eval_method = getattr(policy, "eval", None)
    if callable(eval_method):
        eval_method()
        return
    train_method = getattr(policy, "train", None)
    if callable(train_method):
        train_method(False)
