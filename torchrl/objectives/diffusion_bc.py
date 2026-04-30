# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass

import torch
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import TensorDictModule
from tensordict.utils import NestedKey

from torchrl.objectives.common import LossModule


class DiffusionBCLoss(LossModule):
    """Behavioural Cloning loss for diffusion-based policies.

    Implements the ε-prediction (noise-prediction) denoising loss from
    `Diffusion Policy: Visuomotor Policy Learning via Action Diffusion
    <https://arxiv.org/abs/2303.04137>`_ (Chi et al., RSS 2023).

    Given a batch of (observation, clean_action) pairs from a demonstration
    dataset, the loss:

    1. Samples a random diffusion timestep ``t`` for each item in the batch.
    2. Corrupts the clean action with Gaussian noise via the DDPM forward
       process: ``noisy_action = sqrt(ᾱ_t) * action + sqrt(1 - ᾱ_t) * ε``.
    3. Asks the score network to predict the noise ``ε``.
    4. Returns the MSE between the predicted and actual noise.

    This loss is designed to be used together with
    :class:`~torchrl.modules.DiffusionActor`.  The actor's inner
    :class:`~torchrl.modules.tensordict_module.actors._DDPMModule` is
    accessed via ``actor_network.module`` and its ``add_noise`` method is
    used for step 2.

    Args:
        actor_network (TensorDictModule): a :class:`~torchrl.modules.DiffusionActor`
            (or any :class:`~tensordict.nn.TensorDictModule` whose ``.module``
            exposes ``add_noise(clean_action, t)`` and a
            ``score_network`` attribute).

    Keyword Args:
        reduction (str, optional): Specifies the reduction to apply to the
            output: ``"none"`` | ``"mean"`` | ``"sum"``.  Defaults to
            ``"mean"``.

    .. note::
        The tensordict passed to :meth:`forward` must contain:

        * ``self.tensor_keys.action`` — the *clean* (demonstration) action.
        * ``self.tensor_keys.observation`` — the conditioning observation.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.modules import DiffusionActor
        >>> from torchrl.objectives import DiffusionBCLoss
        >>> actor = DiffusionActor(action_dim=2, obs_dim=4, num_steps=10)
        >>> loss_fn = DiffusionBCLoss(actor)
        >>> td = TensorDict(
        ...     {
        ...         "observation": torch.randn(8, 4),
        ...         "action": torch.randn(8, 2),
        ...     },
        ...     batch_size=[8],
        ... )
        >>> loss_td = loss_fn(td)
        >>> loss_td["loss_diffusion_bc"].backward()
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys.

        Attributes:
            action (NestedKey): Key for the clean demonstration action.
                Defaults to ``"action"``.
            observation (NestedKey): Key for the conditioning observation.
                Defaults to ``"observation"``.
        """

        action: NestedKey = "action"
        observation: NestedKey = "observation"

    actor_network: TensorDictModule
    actor_network_params: TensorDictParams | None
    target_actor_network_params: TensorDictParams | None

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    def __init__(
        self,
        actor_network: TensorDictModule,
        *,
        reduction: str = "mean",
    ) -> None:
        self._in_keys = None
        self._out_keys = None
        super().__init__()
        self.convert_to_functional(actor_network, "actor_network")
        self.reduction = reduction

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._in_keys = [
                self.tensor_keys.observation,
                self.tensor_keys.action,
            ]
        return self._in_keys

    @in_keys.setter
    def in_keys(self, value):
        self._in_keys = value

    @property
    def out_keys(self):
        if self._out_keys is None:
            self._out_keys = ["loss_diffusion_bc"]
        return self._out_keys

    @out_keys.setter
    def out_keys(self, value):
        self._out_keys = value

    def forward(self, tensordict: TensorDictBase) -> TensorDict:
        """Compute the diffusion BC loss.

        Args:
            tensordict (TensorDictBase): input data containing observations
                and clean demonstration actions.

        Returns:
            TensorDict with key ``"loss_diffusion_bc"``.
        """
        clean_action = tensordict[self.tensor_keys.action]
        observation = tensordict[self.tensor_keys.observation]

        batch_size = clean_action.shape[0]
        device = clean_action.device

        with self.actor_network_params.to_module(self.actor_network):
            # Access the underlying _DDPMModule
            ddpm = self.actor_network.module

            # Sample a random timestep per batch element
            t = torch.randint(0, ddpm.num_steps, (batch_size,), device=device)

            # Forward diffusion: corrupt the clean action
            noisy_action, noise = ddpm.add_noise(clean_action, t)

            # Build the score network input: (noisy_action || observation || t)
            t_float = t.float().unsqueeze(-1)  # (B, 1)
            model_input = torch.cat([noisy_action, observation, t_float], dim=-1)

            # Predict the noise
            predicted_noise = ddpm.score_network(model_input)

        # ε-prediction MSE loss
        loss = torch.nn.functional.mse_loss(
            predicted_noise, noise, reduction=self.reduction
        )

        return TensorDict({"loss_diffusion_bc": loss}, batch_size=[])
