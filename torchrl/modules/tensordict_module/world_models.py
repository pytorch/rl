# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential


class WorldModel(TensorDictModuleBase):
    """A general, composable world model for model-based RL.

    ``WorldModel`` wraps an encoder, a dynamics model, a reward head, and
    optionally a done head and a decoder into a single TensorDict-native
    module. It owns *prediction and composition* — encoding observations,
    advancing latent state, predicting rewards and termination — and exposes
    named shortcuts (:meth:`encode`, :meth:`step`, :meth:`decode`) so each
    component can be invoked individually.

    Rollout semantics live elsewhere: wrap a ``WorldModel`` in
    :class:`~torchrl.envs.model_based.WorldModelEnv` (or another
    :class:`~torchrl.envs.model_based.ModelBasedEnvBase` subclass) and use
    :meth:`~torchrl.envs.EnvBase.rollout` to generate imagined trajectories.
    This keeps env-level concerns — reset/step contract, ``done`` handling,
    spec validation — out of the prediction module and avoids forking a
    second rollout implementation with subtly different semantics.

    The module is key-driven: each component communicates through named
    TensorDict keys defined by its ``in_keys`` / ``out_keys``. No specific
    latent representation, observation space, or dynamics architecture is
    assumed.

    Args:
        encoder (TensorDictModule): maps an observation to a latent
            representation, e.g. ``obs -> latent``.
        dynamics (TensorDictModule): advances the latent state given an
            action, e.g. ``(latent, action) -> ("next", latent)``.
        reward_head (TensorDictModule): predicts the reward from the next
            latent, e.g. ``("next", latent) -> ("next", "reward")``.
        done_head (TensorDictModule, optional): predicts the done flag, e.g.
            ``("next", latent) -> ("next", "done")``. When provided,
            :meth:`rollout` can terminate early when any trajectory is done.
        decoder (TensorDictModule, optional): reconstructs an observation from
            a latent, e.g. ``latent -> obs_recon``. Required to call
            :meth:`decode`.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import WorldModel
        >>> obs_dim, latent_dim, action_dim = 8, 4, 2
        >>> encoder = TensorDictModule(
        ...     torch.nn.Linear(obs_dim, latent_dim),
        ...     in_keys=["observation"],
        ...     out_keys=["latent"],
        ... )
        >>> dynamics = TensorDictModule(
        ...     torch.nn.Linear(latent_dim + action_dim, latent_dim),
        ...     in_keys=["latent", "action"],
        ...     out_keys=[("next", "latent")],
        ... )
        >>> reward_head = TensorDictModule(
        ...     torch.nn.Linear(latent_dim, 1),
        ...     in_keys=[("next", "latent")],
        ...     out_keys=[("next", "reward")],
        ... )
        >>> world_model = WorldModel(encoder, dynamics, reward_head)
        >>> td = TensorDict(
        ...     {"observation": torch.randn(2, obs_dim), "action": torch.randn(2, action_dim)},
        ...     batch_size=[2],
        ... )
        >>> out = world_model(td)
        >>> out.keys()
        dict_keys(['observation', 'action', 'latent', 'next'])

    """

    def __init__(
        self,
        encoder: TensorDictModule,
        dynamics: TensorDictModule,
        reward_head: TensorDictModule,
        *,
        done_head: TensorDictModule | None = None,
        decoder: TensorDictModule | None = None,
    ) -> None:
        super().__init__()
        self.encoder = encoder
        self.dynamics = dynamics
        self.reward_head = reward_head
        self.done_head = done_head
        self.decoder = decoder

        # Full forward sequence (encode + step).
        full_modules = [encoder, dynamics, reward_head]
        if done_head is not None:
            full_modules.append(done_head)
        if decoder is not None:
            full_modules.append(decoder)
        self._full_seq = TensorDictSequential(*full_modules)
        self.in_keys = self._full_seq.in_keys
        self.out_keys = self._full_seq.out_keys

        # Step sequence (dynamics + heads, no encoder).
        step_modules = [dynamics, reward_head]
        if done_head is not None:
            step_modules.append(done_head)
        if decoder is not None:
            step_modules.append(decoder)
        self._step_seq = TensorDictSequential(*step_modules)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Run the full pipeline: encoder -> dynamics -> reward_head -> [done_head] -> [decoder]."""
        return self._full_seq(tensordict)

    def encode(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Encode an observation into the latent space."""
        return self.encoder(tensordict)

    def step(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Take one imagined step: dynamics -> reward_head -> [done_head] -> [decoder].

        The encoder is *not* called; the tensordict must already contain the
        current latent state as produced by :meth:`encode` or a previous call
        to :meth:`step`.
        """
        return self._step_seq(tensordict)

    def decode(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Decode a latent back to observation space.

        Raises:
            RuntimeError: if no ``decoder`` was provided at construction.
        """
        if self.decoder is None:
            raise RuntimeError(
                "WorldModel.decode requires a decoder module. Pass decoder= at construction."
            )
        return self.decoder(tensordict)

    @property
    def step_module(self) -> TensorDictSequential:
        """The step-only sequence (dynamics + heads, no encoder).

        Exposed as a public attribute so :class:`~torchrl.envs.model_based.WorldModelEnv`
        and other model-based env wrappers can drive the world model in latent
        space, one step at a time, without rerunning the encoder on every step.
        """
        return self._step_seq


class WorldModelWrapper(TensorDictSequential):
    """World model wrapper.

    This module wraps together a transition model and a reward model.
    The transition model is used to predict an imaginary world state.
    The reward model is used to predict the reward of the imagined transition.

    Args:
        transition_model (TensorDictModule): a transition model that generates a new world states.
        reward_model (TensorDictModule): a reward model, that reads the world state and returns a reward.

    """

    def __init__(
        self, transition_model: TensorDictModule, reward_model: TensorDictModule
    ):
        super().__init__(transition_model, reward_model)

    def get_transition_model_operator(self) -> TensorDictModule:
        """Returns a transition operator that maps either an observation to a world state or a world state to the next world state."""
        return self.module[0]

    def get_reward_operator(self) -> TensorDictModule:
        """Returns a reward operator that maps a world state to a reward."""
        return self.module[1]
