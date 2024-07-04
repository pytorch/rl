# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from dataclasses import dataclass

import torch

import torch.autograd as autograd
from tensordict import TensorDict, TensorDictBase, TensorDictParams
from tensordict.nn import dispatch, TensorDictModule
from tensordict.utils import NestedKey

from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import _reduce


class GAILLoss(LossModule):
    r"""TorchRL implementation of the Generative Adversarial Imitation Learning (GAIL) loss.

    Presented in `"Generative Adversarial Imitation Learning" <https://arxiv.org/pdf/1606.03476>`

    Args:
        discriminator_network (TensorDictModule): stochastic actor

    Keyword Args:
        use_grad_penalty (bool, optional): Whether to use gradient penalty. Default: ``False``.
        gp_lambda (float, optional): Gradient penalty lambda. Default: ``10``.
        reduction (str, optional): Specifies the reduction to apply to the output:
            ``"none"`` | ``"mean"`` | ``"sum"``. ``"none"``: no reduction will be applied,
            ``"mean"``: the sum of the output will be divided by the number of
            elements in the output, ``"sum"``: the output will be summed. Default: ``"mean"``.
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            action (NestedKey): The input tensordict key where the action is expected.
                Defaults to ``"action"``.
            observation (NestedKey): The tensordict key where the observation is expected.
                Defaults to ``"observation"``.
        """

        action: NestedKey = "action"
        observation: NestedKey = "observation"
        discriminator_pred: NestedKey = "d_logits"

    default_keys = _AcceptedKeys()

    discriminator_network: TensorDictModule
    discriminator_network_params: TensorDictParams

    def __init__(
        self,
        discriminator_network: TensorDictModule,
        *,
        use_grad_penalty: bool = False,
        gp_lambda: float = 10,
        reduction: str = None,
    ) -> None:
        self._in_keys = None
        self._out_keys = None
        if reduction is None:
            reduction = "mean"
        super().__init__()

        # Discriminator Network
        self.convert_to_functional(
            discriminator_network,
            "discriminator_network",
            create_target_params=False,
        )
        self.loss_function = torch.nn.BCELoss()
        self.use_grad_penalty = use_grad_penalty
        self.gp_lambda = gp_lambda

        self.reduction = reduction

    def _set_in_keys(self):
        keys = self.discriminator_network.in_keys
        keys = set(keys)
        self._in_keys = sorted(keys, key=str)

    def _forward_value_estimator_keys(self, **kwargs) -> None:
        pass

    @property
    def in_keys(self):
        if self._in_keys is None:
            self._set_in_keys()
        return self._in_keys

    @in_keys.setter
    def in_keys(self, values):
        self._in_keys = values

    @property
    def out_keys(self):
        if self._out_keys is None:
            keys = ["loss"]
            self._out_keys = keys
        return self._out_keys

    @out_keys.setter
    def out_keys(self, values):
        self._out_keys = values

    @dispatch
    def forward(
        self, tensordict: TensorDictBase, collection_tensordict: TensorDictBase
    ) -> TensorDictBase:
        """Compute the GAIL discriminator loss."""
        expert_tensordict = tensordict.clone(False)
        expert_input = expert_tensordict.select(*self.in_keys).detach()

        collection_tensordict = collection_tensordict.clone(False)
        collection_input = collection_tensordict.select(*self.in_keys).detach()

        combined_inputs = torch.cat([expert_input, collection_input], dim=0)

        # create labels
        collection_bs = collection_tensordict.batch_size[0]
        expert_bs = expert_tensordict.batch_size[0]
        fake_labels = torch.zeros((collection_bs, 1), dtype=torch.float32).to(
            collection_tensordict.device
        )
        real_labels = torch.ones((expert_bs, 1), dtype=torch.float32).to(
            expert_tensordict.device
        )

        with self.discriminator_network_params.to_module(self.discriminator_network):
            d_logits = self.discriminator_network(combined_inputs).get(
                self.tensor_keys.discriminator_pred
            )

        expert_preds, collection_preds = torch.split(
            d_logits, [expert_bs, collection_bs], dim=0
        )

        expert_loss = self.loss_function(expert_preds, real_labels)
        collection_loss = self.loss_function(collection_preds, fake_labels)

        loss = expert_loss + collection_loss
        out = {"loss": loss}
        if not self.use_grad_penalty:
            obs = collection_tensordict.get(self.tensor_keys.observation)
            acts = collection_tensordict.get(self.tensor_keys.action)
            obs_e = expert_tensordict.get(self.tensor_keys.observation)
            acts_e = expert_tensordict.get(self.tensor_keys.action)

            obs = obs[:expert_bs]
            acts = acts[:expert_bs]

            obss_noise = (
                torch.distributions.Uniform(0.0, 1.0)
                .sample(obs_e.shape)
                .to(tensordict.device)
            )
            acts_noise = (
                torch.distributions.Uniform(0.0, 1.0)
                .sample(acts_e.shape)
                .to(tensordict.device)
            )
            obss_mixture = obss_noise * obs + (1 - obss_noise) * obs_e
            acts_mixture = acts_noise * acts + (1 - acts_noise) * acts_e
            obss_mixture.requires_grad_(True)
            acts_mixture.requires_grad_(True)

            pg_input_td = TensorDict(
                {
                    self.tensor_keys.observation: obss_mixture,
                    self.tensor_keys.action: acts_mixture,
                },
                [],
            )

            with self.discriminator_network_params.to_module(
                self.discriminator_network
            ):
                d_logits_mixture = self.discriminator_network(pg_input_td).get(
                    self.tensor_keys.discriminator_pred
                )

            gradients = torch.cat(
                autograd.grad(
                    outputs=d_logits_mixture,
                    inputs=(obss_mixture, acts_mixture),
                    grad_outputs=torch.ones(
                        d_logits_mixture.size(), device=tensordict.device
                    ),
                    create_graph=True,
                    retain_graph=True,
                    only_inputs=True,
                ),
                dim=-1,
            )

            gp_loss = self.gp_lambda * torch.mean(
                (torch.linalg.norm(gradients, dim=-1) - 1) ** 2
            )

            loss += gp_loss
            out["gp_loss"] = gp_loss
        loss = _reduce(loss, reduction=self.reduction)

        td_out = TensorDict(out, [])
        return td_out
