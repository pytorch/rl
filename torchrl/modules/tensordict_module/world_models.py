# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from collections.abc import Callable

import torch
from tensordict import TensorDictBase
from tensordict.nn import TensorDictModule, TensorDictModuleBase, TensorDictSequential


class WorldModel(TensorDictModuleBase):
    """A general, composable world model for model-based RL.

    ``WorldModel`` wraps an encoder, a dynamics model, a reward head, and
    optionally a done head and a decoder into a single TensorDict-native
    module. It exposes named shortcuts for encoding, decoding, and stepping,
    and provides a ``rollout`` method whose output matches the layout produced
    by :meth:`~torchrl.envs.EnvBase.rollout`, making imagined trajectories
    directly compatible with replay buffers, value estimators, and loss
    modules.

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

    def rollout(
        self,
        start_td: TensorDictBase,
        policy: Callable[[TensorDictBase], TensorDictBase],
        horizon: int,
        *,
        break_when_any_done: bool = True,
    ) -> TensorDictBase:
        """Run an imagined rollout for up to ``horizon`` steps.

        At each step the policy is queried for an action, then the world model
        advances the latent state via :meth:`step`. State propagation between
        steps uses :func:`~torchrl.envs.utils.step_mdp`, which moves all
        ``("next", key)`` entries to the root level — identical to how
        :class:`~torchrl.envs.EnvBase` advances a real MDP.

        The returned TensorDict has shape ``(*start_td.batch_size, t)`` where
        ``t <= horizon``. Each entry along the time dimension contains the
        full transition: current latent, action, next latent, predicted
        reward, and optionally done/decoder outputs. This layout matches
        :meth:`~torchrl.envs.EnvBase.rollout`, so the result can be stored
        directly in a :class:`~torchrl.data.TensorDictReplayBuffer`, sampled
        by :class:`~torchrl.data.SliceSampler`, or consumed by any TorchRL
        loss module.

        Args:
            start_td (TensorDictBase): initial tensordict containing the
                starting latent state (and any other keys required by the
                policy and dynamics).
            policy (callable): a callable that maps a ``TensorDictBase`` to a
                ``TensorDictBase`` with an action key added in-place or as a
                copy.
            horizon (int): maximum number of imagined steps.
            break_when_any_done (bool, optional): if ``True`` and a
                ``done_head`` is present, the rollout terminates as soon as
                any trajectory in the batch is done. Default: ``True``.

        Returns:
            TensorDictBase: shape ``(*start_td.batch_size, t)``.

        Examples:
            >>> import torch
            >>> from tensordict import TensorDict
            >>> from tensordict.nn import TensorDictModule
            >>> from torchrl.modules import WorldModel
            >>> obs_dim, latent_dim, action_dim = 8, 4, 2
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
            >>> policy = TensorDictModule(
            ...     torch.nn.Linear(latent_dim, action_dim),
            ...     in_keys=["latent"], out_keys=["action"],
            ... )
            >>> start_td = TensorDict(
            ...     {"observation": torch.randn(3, obs_dim)}, batch_size=[3]
            ... )
            >>> start_td = world_model.encode(start_td)
            >>> rollout_td = world_model.rollout(start_td, policy, horizon=5)
            >>> rollout_td.shape
            torch.Size([3, 5])
        """
        # Lazy import to avoid circular dependency:
        # torchrl.modules.tensordict_module -> torchrl.envs.utils ->
        # torchrl.modules.tensordict_module.exploration
        from torchrl.envs.utils import step_mdp

        td = start_td.copy()
        outputs = []
        for _ in range(horizon):
            td = policy(td)
            td = self._step_seq(td)
            outputs.append(td.copy())
            done = td.get(("next", "done"), default=None)
            if break_when_any_done and self.done_head is not None and done is not None:
                if done.any():
                    break
            td = step_mdp(td, keep_other=True)
        return torch.stack(outputs, dim=len(start_td.batch_size))


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
