# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.utils import NestedKey

from torchrl._utils import _maybe_record_function_decorator
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import distance_loss


class WorldModelLoss(LossModule):
    """A general loss module for model-based world models.

    ``WorldModelLoss`` evaluates a :class:`~torchrl.modules.WorldModel` on a
    batch of real transitions and returns a :class:`~tensordict.TensorDict`
    containing one or more named sub-losses.  All sub-losses are optional and
    controlled via the ``losses`` argument:

    - ``"reward"``: MSE / L1 between the predicted reward and the ground-truth
      reward stored in the replay buffer.
    - ``"done"``: MSE / L1 between the predicted done flag and the ground-truth
      done flag.
    - ``"reconstruction"``: MSE / L1 between the decoder's reconstructed
      observation and the original observation.
    - ``"latent"``: MSE / L1 between a predicted next-latent key and a
      target next-latent key.  Useful for deterministic world models; for
      VAE / RSSM-style KL losses use
      :class:`~torchrl.objectives.DreamerModelLoss` instead.

    The ground-truth reward and done tensors are read from the input
    tensordict, renamed to ``("next", true_reward)`` / ``("next", true_done)``
    before the world model is called, so that the world model can freely write
    its predictions under ``("next", reward)`` / ``("next", done)``.

    Args:
        world_model (WorldModel): the world model to evaluate.
        losses (list of str, optional): which sub-losses to compute.
            Any subset of ``["reward", "done", "reconstruction", "latent"]``.
            Defaults to ``["reward"]``.
        reward_loss (str, optional): loss function for the reward head.
            Passed to :func:`~torchrl.objectives.utils.distance_loss`.
            Default: ``"l2"``.
        done_loss (str, optional): loss function for the done head.
            Default: ``"l2"``.
        reconstruction_loss (str, optional): loss function for the decoder.
            Default: ``"l2"``.
        latent_loss (str, optional): loss function for the latent prediction.
            Default: ``"l2"``.
        reward_weight (float, optional): scalar weight applied to
            ``loss_reward``. Default: ``1.0``.
        done_weight (float, optional): scalar weight applied to
            ``loss_done``. Default: ``1.0``.
        reconstruction_weight (float, optional): scalar weight applied to
            ``loss_reconstruction``. Default: ``1.0``.
        latent_weight (float, optional): scalar weight applied to
            ``loss_latent``. Default: ``1.0``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.modules import WorldModel
        >>> from torchrl.objectives import WorldModelLoss
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
        >>> loss_module = WorldModelLoss(world_model, losses=["reward"])
        >>> batch = TensorDict(
        ...     {
        ...         "observation": torch.randn(4, obs_dim),
        ...         "action": torch.randn(4, action_dim),
        ...         "next": {"reward": torch.randn(4, 1)},
        ...     },
        ...     batch_size=[4],
        ... )
        >>> loss_td = loss_module(batch)
        >>> loss_td.keys()
        dict_keys(['loss_reward'])
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using
        ``'.set_keys(key_name=key_value)'`` and their default values.

        Attributes:
            reward (NestedKey): The predicted reward written by the world
                model's reward head. Defaults to ``"reward"``.
            true_reward (NestedKey): Temporary key used to store the
                ground-truth reward before the world model is called.
                Defaults to ``"true_reward"``.
            done (NestedKey): The predicted done flag written by the world
                model's done head. Defaults to ``"done"``.
            true_done (NestedKey): Temporary key used to store the
                ground-truth done flag. Defaults to ``"true_done"``.
            terminated (NestedKey): The predicted terminated flag.
                Defaults to ``"terminated"``.
            true_terminated (NestedKey): Temporary key for the ground-truth
                terminated flag. Defaults to ``"true_terminated"``.
            observation (NestedKey): The original observation used to compute
                the reconstruction loss. Defaults to ``"observation"``.
            reconstructed_observation (NestedKey): The observation
                reconstructed by the decoder. Defaults to
                ``"reconstructed_observation"``.
            predicted_latent (NestedKey): The next-latent predicted by the
                dynamics model, used for the latent loss.
                Defaults to ``"predicted_latent"``.
            target_latent (NestedKey): The target next-latent (e.g. produced
                by running the encoder on the next observation).
                Defaults to ``"target_latent"``.
        """

        reward: NestedKey = "reward"
        true_reward: NestedKey = "true_reward"
        done: NestedKey = "done"
        true_done: NestedKey = "true_done"
        terminated: NestedKey = "terminated"
        true_terminated: NestedKey = "true_terminated"
        observation: NestedKey = "observation"
        reconstructed_observation: NestedKey = "reconstructed_observation"
        predicted_latent: NestedKey = "predicted_latent"
        target_latent: NestedKey = "target_latent"

    tensor_keys: _AcceptedKeys
    default_keys = _AcceptedKeys

    _VALID_LOSSES = frozenset({"reward", "done", "reconstruction", "latent"})

    def __init__(
        self,
        world_model,
        *,
        losses: list[Literal["reward", "done", "reconstruction", "latent"]]
        | None = None,
        reward_loss: str = "l2",
        done_loss: str = "l2",
        reconstruction_loss: str = "l2",
        latent_loss: str = "l2",
        reward_weight: float = 1.0,
        done_weight: float = 1.0,
        reconstruction_weight: float = 1.0,
        latent_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.world_model = world_model
        if losses is None:
            losses = ["reward"]
        unknown = set(losses) - self._VALID_LOSSES
        if unknown:
            raise ValueError(
                f"Unknown loss type(s): {unknown}. "
                f"Valid choices are {sorted(self._VALID_LOSSES)}."
            )
        self.losses = list(losses)
        self.reward_loss = reward_loss
        self.done_loss = done_loss
        self.reconstruction_loss = reconstruction_loss
        self.latent_loss = latent_loss
        self.reward_weight = reward_weight
        self.done_weight = done_weight
        self.reconstruction_weight = reconstruction_weight
        self.latent_weight = latent_weight

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    @_maybe_record_function_decorator("world_model_loss/forward")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Compute the world model loss on a batch of real transitions.

        Args:
            tensordict (TensorDictBase): a batch of real transitions containing
                at minimum the keys consumed by the world model.

        Returns:
            TensorDictBase: a scalar TensorDict with keys ``"loss_reward"``,
            ``"loss_done"``, ``"loss_reconstruction"``, and/or
            ``"loss_latent"`` depending on the active ``losses``.
        """
        tensordict = tensordict.copy()

        # Save ground-truth labels before the world model overwrites them.
        if "reward" in self.losses:
            tensordict.rename_key_(
                ("next", self.tensor_keys.reward),
                ("next", self.tensor_keys.true_reward),
            )
        if "done" in self.losses:
            tensordict.rename_key_(
                ("next", self.tensor_keys.done),
                ("next", self.tensor_keys.true_done),
                safe=True,
            )
            tensordict.rename_key_(
                ("next", self.tensor_keys.terminated),
                ("next", self.tensor_keys.true_terminated),
                safe=True,
            )

        tensordict = self.world_model(tensordict)

        out: dict[str, torch.Tensor] = {}

        if "reward" in self.losses:
            pred_reward = tensordict.get(("next", self.tensor_keys.reward))
            true_reward = tensordict.get(("next", self.tensor_keys.true_reward))
            loss_r = distance_loss(pred_reward, true_reward, self.reward_loss).mean()
            out["loss_reward"] = self.reward_weight * loss_r

        if "done" in self.losses:
            pred_done = tensordict.get(("next", self.tensor_keys.done))
            true_done = tensordict.get(("next", self.tensor_keys.true_done))
            loss_d = distance_loss(
                pred_done.float(), true_done.float(), self.done_loss
            ).mean()
            out["loss_done"] = self.done_weight * loss_d

        if "reconstruction" in self.losses:
            obs = tensordict.get(self.tensor_keys.observation)
            recon = tensordict.get(self.tensor_keys.reconstructed_observation)
            loss_recon = distance_loss(recon, obs, self.reconstruction_loss).mean()
            out["loss_reconstruction"] = self.reconstruction_weight * loss_recon

        if "latent" in self.losses:
            pred_latent = tensordict.get(self.tensor_keys.predicted_latent)
            target_latent = tensordict.get(self.tensor_keys.target_latent)
            loss_lat = distance_loss(
                pred_latent, target_latent, self.latent_loss
            ).mean()
            out["loss_latent"] = self.latent_weight * loss_lat

        td_out = TensorDict(out)
        self._clear_weakrefs(tensordict, td_out)
        return td_out
