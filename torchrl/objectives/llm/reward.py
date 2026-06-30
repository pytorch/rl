# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import contextlib

from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import NestedKey, TensorClass, TensorDictBase
from tensordict.nn import TensorDictModule
from torchrl.objectives.common import LossModule


def reward_model_loss(
    chosen_scores: torch.Tensor,
    rejected_scores: torch.Tensor,
    reduction: Literal["mean", "sum", "none"],
) -> torch.Tensor:
    r"""Compute the Bradley-Terry pairwise reward-model loss.

    The loss is computed as ``-log_sigmoid(chosen_scores - rejected_scores)``. It is
    small when the reward model assigns a higher score to the chosen response than to
    the rejected one, and large otherwise.

    .. math::

        \text{loss} = -\log\sigma(r_\theta(x, y_c) - r_\theta(x, y_r))

    Args:
        chosen_scores (torch.Tensor): the scalar scores assigned to the chosen
            responses. Must have shape ``[B]``.
        rejected_scores (torch.Tensor): the scalar scores assigned to the rejected
            responses. Must have shape ``[B]``.
        reduction (Literal["mean", "sum", "none"]): the reduction to apply to the loss.

    Returns:
        The Bradley-Terry loss.

    References:
        - Ralph Allan Bradley, Milton E. Terry, 1952. "Rank Analysis of Incomplete Block
          Designs: I. The Method of Paired Comparisons".
    """
    loss = -torch.nn.functional.logsigmoid(chosen_scores - rejected_scores)
    if reduction == "mean":
        return loss.mean()
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    raise ValueError(f"Invalid reduction: {reduction}.")


class RewardModelLossOutput(TensorClass["nocast"]):
    """Reward-model loss output.

    Attributes:
        loss_reward_model (torch.Tensor): the Bradley-Terry pairwise loss.
        loss_center (torch.Tensor | None): the score-centering regularization loss.
            Only present when ``center_coeff`` is set on the loss. Defaults to ``None``.
        accuracy (torch.Tensor | None): the fraction of pairs for which the chosen
            score exceeds the rejected score. This is a detached metric for logging
            and is not differentiable. Defaults to ``None``.

    .. note::
        The differentiable total loss is the sum of the ``loss_`` prefixed fields,
        i.e. ``loss_reward_model`` (plus ``loss_center`` when set). ``accuracy`` is a
        detached diagnostic and should not be backpropagated:

            >>> loss_fn = RewardModelLoss(score_network)
            >>> loss_output = loss_fn(td)
            >>> loss = loss_output.loss_reward_model
            >>> if loss_output.loss_center is not None:
            ...     loss = loss + loss_output.loss_center
            >>> loss.backward()
    """

    loss_reward_model: torch.Tensor
    loss_center: torch.Tensor | None = None
    accuracy: torch.Tensor | None = None


class RewardModelLoss(LossModule):
    r"""Bradley-Terry reward-model training loss for RLHF.

    Trains a scalar reward model from pairwise human preference data. Given a prompt and
    two responses (a ``chosen`` one preferred by an annotator and a ``rejected`` one),
    the model is encouraged to assign a higher score to the chosen response. This is the
    reward-modelling stage that precedes policy optimization in RLHF pipelines.

    The loss is model-agnostic: ``score_network`` can wrap any backbone that maps a
    tokenized response to a single scalar score, for example a Hugging Face
    ``AutoModelForSequenceClassification`` with ``num_labels=1`` or
    :class:`~torchrl.modules.models.llm.GPT2RewardModel`.

    Args:
        score_network (TensorDictModule, optional): a module mapping a (chosen or
            rejected) sub-tensordict to a per-sequence scalar score written under the
            ``score`` key. The same network is applied to the chosen and rejected
            inputs (weight sharing). If ``None``, the chosen and rejected scores are
            expected to already be present under the ``score`` key of their respective
            sub-tensordicts. Defaults to ``None``.

    Keyword Args:
        reduction (Literal["mean", "sum", "none"], optional): the reduction to apply to
            the loss. Defaults to ``"mean"``.
        center_coeff (float, optional): if set, adds a centering regularization term
            ``center_coeff * (chosen_score**2 + rejected_score**2)`` that discourages
            the reward model from drifting to large magnitudes. Defaults to ``None``
            (disabled).
        device (torch.device or str, optional): if provided, chosen/rejected
            sub-tensordicts are moved to this device before scoring and loss
            computation. The ``score_network`` is expected to already be on this
            device. Defaults to ``None``.

    .. note::
        The input tensordict is expected to contain the following keys by default:
            - ``"chosen"``: a sub-tensordict with the chosen response inputs (e.g.
              ``input_ids`` / ``attention_mask``), or the chosen ``score`` directly
              when ``score_network`` is ``None``.
            - ``"rejected"``: the corresponding sub-tensordict for the rejected
              response.

        These keys (and the ``score`` output key) can be customized using the
        :meth:`~torchrl.objectives.common.LossModule.set_keys` method.

    .. seealso:: :class:`~torchrl.modules.models.llm.GPT2RewardModel` for a ready-made
        GPT2-based reward-model backbone, and
        :class:`~torchrl.data.llm.reward.PairwiseDataset` for a pairwise preference
        dataset.

    References:
        - Nisan Stiennon, Long Ouyang, Jeff Wu, Daniel M. Ziegler, Ryan Lowe, Chelsea
          Voss, Alec Radford, Dario Amodei, Paul Christiano, 2020.
          `"Learning to summarize from human feedback" <https://arxiv.org/abs/2009.01325>`_
        - Long Ouyang et al., 2022.
          `"Training language models to follow instructions with human feedback" <https://arxiv.org/abs/2203.02155>`_

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from tensordict.nn import TensorDictModule
        >>> from torchrl.objectives.llm.reward import RewardModelLoss
        >>>
        >>> # A toy score network mapping input_ids to a scalar score per sequence.
        >>> class Scorer(torch.nn.Module):
        ...     def __init__(self, vocab_size=128, embed_dim=8):
        ...         super().__init__()
        ...         self.embed = torch.nn.Embedding(vocab_size, embed_dim)
        ...         self.head = torch.nn.Linear(embed_dim, 1)
        ...
        ...     def forward(self, input_ids):
        ...         return self.head(self.embed(input_ids).mean(-2))
        >>>
        >>> score_network = TensorDictModule(
        ...     Scorer(), in_keys=["input_ids"], out_keys=["score"]
        ... )
        >>> loss_fn = RewardModelLoss(score_network=score_network)
        >>> data = TensorDict(
        ...     chosen=TensorDict(
        ...         input_ids=torch.randint(0, 128, (4, 16)), batch_size=[4]
        ...     ),
        ...     rejected=TensorDict(
        ...         input_ids=torch.randint(0, 128, (4, 16)), batch_size=[4]
        ...     ),
        ...     batch_size=[4],
        ... )
        >>> loss_vals = loss_fn(data)
        >>> print(f"Reward model loss: {loss_vals.loss_reward_model.item():.4f}")
        >>> print(f"Accuracy: {loss_vals.accuracy.item():.4f}")
        >>> loss_vals.loss_reward_model.backward()
    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            chosen (NestedKey): The input tensordict key where the chosen response
                sub-tensordict is expected. Defaults to ``"chosen"``.
            rejected (NestedKey): The input tensordict key where the rejected response
                sub-tensordict is expected. Defaults to ``"rejected"``.
            score (NestedKey): The key (within each chosen/rejected sub-tensordict)
                where the per-sequence scalar score is written by ``score_network`` or
                read from when ``score_network`` is ``None``. Defaults to ``"score"``.
        """

        chosen: NestedKey = "chosen"
        rejected: NestedKey = "rejected"
        score: NestedKey = "score"

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys

    def __init__(
        self,
        score_network: TensorDictModule | None = None,
        *,
        reduction: Literal["mean", "sum", "none"] = "mean",
        center_coeff: float | None = None,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        self.score_network = score_network
        self.reduction = reduction
        self.center_coeff = center_coeff
        self.device = torch.device(device) if device is not None else None
        self._set_in_keys()

    def _set_in_keys(self) -> None:
        """Sets the input keys for the loss module."""
        self.in_keys = [self.tensor_keys.chosen, self.tensor_keys.rejected]
        self.out_keys = []  # Loss modules typically don't have out_keys

    def _score(self, tensordict: TensorDictBase, key: NestedKey) -> torch.Tensor:
        """Run the score network on a sub-tensordict and return a per-sequence score."""
        sub_td = tensordict.get(key, default=None)
        if sub_td is None:
            raise KeyError(
                f"Could not find the sub-tensordict at key {key!r} in the input "
                f"tensordict with keys {set(tensordict.keys())}."
            )
        if self.device is not None:
            sub_td = sub_td.to(self.device)
        if self.score_network is not None:
            with (
                torch.device(self.device)
                if self.device is not None
                else contextlib.nullcontext()
            ):
                sub_td = self.score_network(sub_td)
        score = sub_td.get(self.tensor_keys.score, default=None)
        if score is None:
            raise KeyError(
                f"Could not find the score at key {self.tensor_keys.score!r} under "
                f"{key!r}. If score_network is None, the scores must be precomputed."
            )
        # reduce a trailing singleton dimension (e.g. [B, 1] -> [B])
        if score.ndim > 1 and score.shape[-1] == 1:
            score = score.squeeze(-1)
        return score

    def forward(self, tensordict: TensorDictBase) -> RewardModelLossOutput:
        chosen_score = self._score(tensordict, self.tensor_keys.chosen)
        rejected_score = self._score(tensordict, self.tensor_keys.rejected)
        if chosen_score.shape != rejected_score.shape:
            raise ValueError(
                f"Chosen and rejected scores have different shapes: "
                f"{chosen_score.shape=} vs {rejected_score.shape=}."
            )

        loss = reward_model_loss(chosen_score, rejected_score, self.reduction)

        loss_center = None
        if self.center_coeff is not None:
            center = chosen_score.pow(2) + rejected_score.pow(2)
            if self.reduction == "mean":
                center = center.mean()
            elif self.reduction == "sum":
                center = center.sum()
            loss_center = self.center_coeff * center

        with torch.no_grad():
            accuracy = (chosen_score > rejected_score).float().mean()

        return RewardModelLossOutput(
            loss_reward_model=loss,
            loss_center=loss_center,
            accuracy=accuracy,
        )
