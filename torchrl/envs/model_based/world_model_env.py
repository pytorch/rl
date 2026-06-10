# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey
from torchrl.envs.common import EnvBase
from torchrl.envs.model_based.common import ModelBasedEnvBase

if TYPE_CHECKING:
    from torchrl.modules import WorldModel


class WorldModelEnv(ModelBasedEnvBase):
    """A generic environment wrapper around a :class:`~torchrl.modules.WorldModel`.

    Wraps a :class:`~torchrl.modules.WorldModel` so it can be driven through
    the standard :class:`~torchrl.envs.EnvBase` API and rolled out with
    :meth:`~torchrl.envs.EnvBase.rollout`. The world model owns prediction
    (encoder, dynamics, reward / done heads, optional decoder); this env owns
    the rollout contract (reset, step, done handling, spec validation).

    Use this class instead of writing a bespoke rollout loop on the world
    model itself. The env semantics — including how
    :meth:`~torchrl.envs.EnvBase.rollout` propagates state via
    :func:`~torchrl.envs.utils.step_mdp` and how it terminates on ``done`` —
    are then shared with every other TorchRL env and stay consistent across
    real and imagined rollouts.

    The env steps in latent space: it does **not** rerun the world model's
    encoder on every step. The caller is expected to seed the latent state on
    reset, typically by calling :meth:`WorldModel.encode` on an observation
    tensordict and passing the result as the ``tensordict`` argument to
    :meth:`~torchrl.envs.EnvBase.reset` or
    :meth:`~torchrl.envs.EnvBase.rollout`.

    Specs are taken from a reference env so that the imagined env presents
    the same action / reward / done specs as the real one. The observation
    spec defaults to the latent representation (under ``latent_key``); pass
    ``observation_spec=`` to override (e.g. when a decoder is present and the
    env should expose decoded observations).

    Args:
        world_model (WorldModel): the prediction module that the env drives.
            Its :attr:`~torchrl.modules.WorldModel.step_module` is used as
            the underlying ``world_model`` argument of
            :class:`~torchrl.envs.model_based.ModelBasedEnvBase`.
        base_env (EnvBase): a reference env whose action / reward / done
            specs are copied into the imagined env. The reference env is not
            stepped — only its specs are read.

    Keyword Args:
        observation_spec (TensorSpec, optional): override for the observation
            spec. When ``None``, the env exposes the latent state under
            ``latent_key`` with shape inferred from ``base_env``.
        batch_size (torch.Size, optional): batch size for the env. Defaults
            to ``base_env.batch_size``.
        device (torch.device, optional): device for the env. Defaults to
            ``base_env.device``.
        latent_key (NestedKey, optional): the key under which the latent
            state is stored. Defaults to ``"latent"``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.envs.model_based import WorldModelEnv
        >>> from torchrl.modules import WorldModel
        >>> base_env = GymEnv("Pendulum-v1")
        >>> obs_dim = base_env.observation_spec["observation"].shape[-1]
        >>> action_dim = base_env.action_spec.shape[-1]
        >>> latent_dim = 4
        >>> encoder = TensorDictModule(
        ...     torch.nn.Linear(obs_dim, latent_dim),
        ...     in_keys=["observation"], out_keys=["latent"],
        ... )
        >>> dynamics = TensorDictModule(
        ...     torch.nn.Linear(latent_dim + action_dim, latent_dim),
        ...     in_keys=["latent", "action"], out_keys=[("next", "latent")],
        ... )
        >>> reward_head = TensorDictModule(
        ...     torch.nn.Linear(latent_dim, 1),
        ...     in_keys=[("next", "latent")], out_keys=[("next", "reward")],
        ... )
        >>> world_model = WorldModel(encoder, dynamics, reward_head)
        >>> wm_env = WorldModelEnv(world_model, base_env=base_env, batch_size=[3])
        >>> # Seed the env with a starting latent and roll it out.
        >>> obs_td = TensorDict(
        ...     {"observation": torch.randn(3, obs_dim)}, batch_size=[3]
        ... )
        >>> start_td = world_model.encode(obs_td)
        >>> rollout = wm_env.rollout(max_steps=5, tensordict=start_td)
        >>> rollout.shape
        torch.Size([3, 5])
    """

    def __init__(
        self,
        world_model: WorldModel,
        base_env: EnvBase,
        *,
        observation_spec=None,
        batch_size: int | torch.Size | Sequence[int] | None = None,
        device: torch.device | str | None = None,
        latent_key: NestedKey = "latent",
    ) -> None:
        if batch_size is None:
            batch_size = (
                base_env.batch_size if len(base_env.batch_size) > 0 else torch.Size([1])
            )
        else:
            batch_size = (
                torch.Size(batch_size)
                if not isinstance(batch_size, torch.Size)
                else batch_size
            )
        if device is None:
            device = base_env.device

        # ModelBasedEnvBase calls ``self.world_model(td)`` inside ``_step``. We
        # want the step-only sequence (dynamics + heads, no encoder) so that
        # each env step advances latent state without rerunning the encoder.
        super().__init__(
            world_model.step_module,
            device=device,
            batch_size=batch_size,
            # Imagined trajectories may legitimately set ``done=True`` on the
            # first step (e.g. when the done_head predicts termination at the
            # very start); the env should not validate that away.
            allow_done_after_reset=True,
        )
        # ``self.world_model`` is the step-only sequence (set by ``super``).
        # Keep a reference to the full :class:`WorldModel` for introspection
        # (encode/decode access from outside) and ``latent_key`` for ``_reset``.
        self._world_model = world_model
        self.latent_key = latent_key

        # Spec wiring: action / reward / done are copied directly from the
        # reference env. The observation spec defaults to the latent
        # representation but can be overridden by the caller.
        self.action_spec = base_env.action_spec.expand(self.batch_size).clone()
        self.reward_spec = base_env.reward_spec.expand(self.batch_size).clone()
        self.done_spec = base_env.done_spec.expand(self.batch_size).clone()

        if observation_spec is not None:
            self.observation_spec = observation_spec
        else:
            self.observation_spec = base_env.observation_spec.expand(
                self.batch_size
            ).clone()

    def _step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Run one imagined step and remap WorldModel's ``("next", *)`` outputs to root.

        :class:`WorldModel` writes its outputs under ``("next", *)`` keys
        (matching the layout of real env step outputs), but
        :class:`~torchrl.envs.EnvBase`'s outer ``step`` machinery expects
        ``_step`` to return *root* keys and then re-wraps them under
        ``("next", *)`` itself. Without this remap, ``EnvBase`` would not
        find the reward / observation / done at the keys it expects.

        Spec iteration walks the leaf keys (``include_nested=True,
        leaves_only=True``) so that nested observation / reward / done specs
        — common in multi-agent or hierarchical envs — are preserved
        end-to-end.
        """
        out = self.world_model(tensordict.copy())
        next_td = out.get("next", default=None)

        result = TensorDict(batch_size=self.batch_size, device=self.device)
        if next_td is None:
            return result

        # Observation: leaf keys present in the world model's "next" subtree
        # are forwarded. Absent keys are skipped — EnvBase will spec-check
        # what's missing and surface a clear error.
        for key in self.observation_spec.keys(include_nested=True, leaves_only=True):
            value = next_td.get(key, default=None)
            if value is not None:
                result.set(key, value)

        # Reward / done leaves: forward what the world model wrote; fall
        # back to the spec's zero tensor when the model did not emit the key.
        for key in self.full_reward_spec.keys(include_nested=True, leaves_only=True):
            value = next_td.get(key, default=None)
            if value is None:
                value = self.full_reward_spec[key].zero()
            result.set(key, value)
        for key in self.full_done_spec.keys(include_nested=True, leaves_only=True):
            value = next_td.get(key, default=None)
            if value is None:
                value = self.full_done_spec[key].zero()
            result.set(key, value)
        return result

    def _reset(
        self, tensordict: TensorDictBase | None = None, **kwargs
    ) -> TensorDictBase:
        """Reset the imagined env from a caller-supplied latent state.

        The caller is expected to provide a tensordict carrying the initial
        latent under :attr:`latent_key` (typically obtained by encoding a real
        observation via :meth:`WorldModel.encode`). Resetting without a
        tensordict is not supported because there is no canonical
        "observation distribution" to sample a fresh latent from.
        """
        if tensordict is None or self.latent_key not in tensordict.keys(
            include_nested=True
        ):
            raise RuntimeError(
                f"WorldModelEnv._reset requires a tensordict carrying the "
                f"initial latent under key {self.latent_key!r}. Typical "
                f"usage: encode a real observation with "
                f"`world_model.encode(obs_td)` and pass the result as the "
                f"`tensordict=` argument to `env.reset(...)` or "
                f"`env.rollout(...)`."
            )
        out = tensordict.copy()
        # Done flags are zeros at reset; the done head (if any) will
        # determine termination on subsequent steps via ``_step``. Drawing
        # them from ``full_done_spec.zero()`` keeps shapes / dtypes / nested
        # done structure (e.g. multi-agent dones) consistent with the spec.
        zero_done = self.full_done_spec.zero()
        for key in zero_done.keys(include_nested=True, leaves_only=True):
            out.set(key, zero_done.get(key))
        return out
