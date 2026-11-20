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
from torchrl.data.tensor_specs import Composite, TensorSpec, Unbounded
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
            spec. When ``None`` and ``base_env`` declares ``latent_key`` (for
            instance a reference env that only carries specs), its
            observation spec is reused. When ``None`` and ``base_env`` does
            not declare it (a real env exposing raw observations), the env
            exposes only the latent state under ``latent_key``, with its spec
            inferred by running the world model's encoder once on
            ``base_env.fake_tensordict()`` (the world model must then already
            be on ``device``). A non-composite override is stored under
            ``"observation"``, as :class:`~torchrl.envs.EnvBase` does. When an
            override does not contain ``latent_key``, the latent spec is added
            to it so that the latent state is always carried from one imagined
            step to the next.
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
        >>> from torchrl.modules import MLP, WorldModel
        >>> base_env = GymEnv("Pendulum-v1")
        >>> obs_dim = base_env.observation_spec["observation"].shape[-1]
        >>> action_dim = base_env.action_spec.shape[-1]
        >>> latent_dim = 4
        >>> encoder = TensorDictModule(
        ...     torch.nn.Linear(obs_dim, latent_dim),
        ...     in_keys=["observation"], out_keys=["latent"],
        ... )
        >>> dynamics = TensorDictModule(
        ...     MLP(in_features=latent_dim + action_dim, out_features=latent_dim, depth=0),
        ...     in_keys=["latent", "action"], out_keys=[("next", "latent")],
        ... )
        >>> reward_head = TensorDictModule(
        ...     torch.nn.Linear(latent_dim, 1),
        ...     in_keys=[("next", "latent")], out_keys=[("next", "reward")],
        ... )
        >>> world_model = WorldModel(encoder, dynamics, reward_head)
        >>> wm_env = WorldModelEnv(world_model, base_env=base_env, batch_size=[3])
        >>> wm_env.observation_spec["latent"].shape
        torch.Size([3, 4])
        >>> wm_env.action_spec.shape
        torch.Size([3, 1])
        >>> obs_td = TensorDict(
        ...     {"observation": torch.randn(3, obs_dim)}, batch_size=[3]
        ... )
        >>> start_td = world_model.encode(obs_td)
        >>> rollout = wm_env.rollout(max_steps=5, tensordict=start_td)
        >>> rollout.shape
        torch.Size([3, 5])
        >>> bool((rollout["next", "latent"][:, -1] != start_td["latent"]).any())
        True
    """

    def __init__(
        self,
        world_model: WorldModel,
        base_env: EnvBase,
        *,
        observation_spec: TensorSpec | None = None,
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

        self.action_spec = self._expand_base_spec(base_env.action_spec, base_env)
        self.reward_spec = self._expand_base_spec(base_env.reward_spec, base_env)
        self.done_spec = self._expand_base_spec(base_env.done_spec, base_env)

        if isinstance(observation_spec, Composite):
            observation_spec = observation_spec.clone()
        elif observation_spec is not None:
            observation_spec = Composite(
                observation=observation_spec.clone(),
                shape=self.batch_size,
                device=self.device,
            )
        elif latent_key in base_env.observation_spec.keys(include_nested=True):
            observation_spec = self._expand_base_spec(
                base_env.observation_spec, base_env
            )
        else:
            observation_spec = Composite(shape=self.batch_size, device=self.device)
        if latent_key not in observation_spec.keys(include_nested=True):
            observation_spec[latent_key] = self._latent_spec(
                world_model, base_env, latent_key
            )
        self.observation_spec = observation_spec

    def _expand_base_spec(self, spec: TensorSpec, base_env: EnvBase) -> TensorSpec:
        """Expand a reference-env spec to ``(*self.batch_size, *feature_shape)``.

        The feature shape is what remains of the spec shape once the reference
        env's own batch dimensions are dropped, so an unbatched reference env
        with an action of shape ``(1,)`` gives ``(*self.batch_size, 1)``.
        """
        feature_shape = spec.shape[len(base_env.batch_size) :]
        return spec.expand(*self.batch_size, *feature_shape).clone()

    def _latent_spec(
        self, world_model: WorldModel, base_env: EnvBase, latent_key: NestedKey
    ) -> TensorSpec:
        """Build the spec of the latent state stepped by the world model.

        The reference env's own spec is used when it already declares
        ``latent_key``. Otherwise the encoder is run once on a fake tensordict
        of the reference env, expanded to this env's batch size when the
        reference env is unbatched and moved to this env's device, to read the
        latent's feature shape and dtype. The pass runs without gradients, in
        eval mode and with a forked RNG, and every encoder submodule gets its
        training flag back afterwards, so running statistics, dropout masks
        and the global seed are left untouched.

        Raises:
            ValueError: if the encoder cannot produce ``latent_key`` from the
                fake tensordict, for instance because it reads a key the
                reference env does not declare or lives on another device.
        """
        base_observation_spec = base_env.full_observation_spec
        if latent_key in base_observation_spec.keys(include_nested=True):
            return self._expand_base_spec(base_observation_spec[latent_key], base_env)
        fake_tensordict = base_env.fake_tensordict()
        if not base_env.batch_size:
            fake_tensordict = fake_tensordict.expand(self.batch_size)
        fake_tensordict = fake_tensordict.to(self.device)
        encoder = world_model.encoder
        training_flags = [(module, module.training) for module in encoder.modules()]
        rng_devices = (
            [self.device]
            if self.device is not None and self.device.type == "cuda"
            else []
        )
        encoder.eval()
        try:
            with torch.no_grad(), torch.random.fork_rng(devices=rng_devices):
                latent = world_model.encode(fake_tensordict).get(latent_key)
        except (KeyError, RuntimeError) as err:
            raise ValueError(
                f"WorldModelEnv could not infer the spec of {latent_key!r} by "
                f"running the world model's encoder on base_env.fake_tensordict(). "
                f"Pass an observation_spec that contains {latent_key!r}, or check "
                f"that the world model is on the env device ({self.device})."
            ) from err
        finally:
            for module, training in training_flags:
                module.training = training
        feature_shape = latent.shape[fake_tensordict.batch_dims :]
        return Unbounded(
            shape=(*self.batch_size, *feature_shape),
            dtype=latent.dtype,
            device=self.device,
        )

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
