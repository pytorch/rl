# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Base class for Vision-Language-Action (VLA) policies."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import TensorDictBase
from tensordict.nn import InteractionType, TensorDictModuleBase
from tensordict.nn.probabilistic import interaction_type
from tensordict.utils import NestedKey
from torch import distributions as torch_dist

from torchrl.data.vla.schema import (
    ACTION_CHUNK_KEY,
    ACTION_TOKENS_KEY,
    IMAGE_KEY,
    INSTRUCTION_KEY,
    STATE_KEY,
)

__all__ = ["VLAWrapperBase"]

ActionHead = Literal["continuous", "tokens"]
LogProbsMode = Literal["sequence", "token"]
SamplingMode = Literal["greedy", "sample"]


class VLAWrapperBase(TensorDictModuleBase):
    """Base class for Vision-Language-Action policies.

    A VLA policy maps multimodal robot observations -- one or more camera
    images, optional proprioceptive state, and a natural-language instruction --
    to a short *action chunk*. This base owns the TensorDict key contract and
    the :meth:`forward` / :meth:`get_dist` orchestration; concrete policies only
    implement the prediction hooks :meth:`_predict_chunk` (continuous head) and
    :meth:`_predict_logits` (discrete-token head).

    Two action heads are supported via ``action_head``:

    - ``"continuous"``: :meth:`forward` writes a continuous action chunk of
      shape ``[*B, chunk_size, action_dim]`` under ``action_chunk``;
    - ``"tokens"``: :meth:`forward` writes discrete action tokens
      ``[*B, chunk_size, action_dim]`` under ``action_tokens`` and their
      log-probabilities under ``log_probs``; :meth:`get_dist` returns
      the token distribution for log-prob/entropy-based RL fine-tuning.
      With ``log_probs_mode="sequence"`` (default) the log-probabilities are
      summed over the chunk (one scalar per sample, the ``sample_log_prob``
      contract of the PPO losses); with ``log_probs_mode="token"`` they are
      per-token, shaped ``[*B, chunk_size, action_dim]`` (the groundwork for
      token-level DAPO-style ratios).

    Keys are configurable through :meth:`set_keys`. The wrapper is a
    :class:`~tensordict.nn.TensorDictModuleBase`, so it composes with the
    standard collectors, losses and transforms.

    Keyword Args:
        action_dim (int): the dimensionality of a single action.
        chunk_size (int): the action-chunk horizon ``H``.
        action_head (str): ``"continuous"`` (default) or ``"tokens"``.
        vocab_size (int, optional): number of action-token bins per dimension
            (required for the ``"tokens"`` head).
        use_state (bool): whether to read the proprioceptive state.
            Defaults to ``True``.
        default_interaction_type (InteractionType): how the ``"tokens"`` head
            turns logits into action tokens when no exploration context is
            active. The forward consults the ambient
            :func:`~torchrl.envs.utils.exploration_type` (set by collectors
            and :func:`~torchrl.envs.utils.set_exploration_type`):
            ``InteractionType.RANDOM`` samples, every other type (and this
            default) takes the argmax. This mirrors
            :class:`~torchrl.modules.tensordict_module.ProbabilisticActor`, so
            the same code drives rollouts (the collector's ``exploration_type``
            defaults to ``RANDOM``) and greedy evaluation
            (``with set_exploration_type(ExplorationType.DETERMINISTIC):``)
            without mutating the policy. Defaults to
            ``InteractionType.DETERMINISTIC`` (argmax). Ignored by the
            continuous head.
        log_probs_mode (str): ``"sequence"`` (default; one summed
            log-probability per sample) or ``"token"`` (per-token
            log-probabilities, ``[*B, chunk_size, action_dim]``) for the
            ``"tokens"`` head.
        mode (str, optional): backward-compatible alias for
            ``default_interaction_type``. ``"sample"`` maps to
            ``InteractionType.RANDOM`` and ``"greedy"`` maps to
            ``InteractionType.DETERMINISTIC``. Defaults to ``None``.

    .. note::
        This base deliberately does **not** inherit from the text-generation
        :class:`~torchrl.modules.llm.LLMWrapperBase`: a VLA policy emits robot
        actions, not text, so it carries only the small multimodal-to-action
        contract.

    .. seealso:: :class:`~torchrl.modules.vla.TinyVLA` (reference policy).
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys for a VLA policy."""

        image: NestedKey = IMAGE_KEY
        state: NestedKey = STATE_KEY
        instruction: NestedKey = INSTRUCTION_KEY
        action_chunk: NestedKey = ACTION_CHUNK_KEY
        action_tokens: NestedKey = ACTION_TOKENS_KEY
        log_probs: NestedKey = "log_probs"

    def __init__(
        self,
        *,
        action_dim: int,
        chunk_size: int,
        action_head: ActionHead = "continuous",
        vocab_size: int | None = None,
        use_state: bool = True,
        default_interaction_type: InteractionType = InteractionType.DETERMINISTIC,
        log_probs_mode: LogProbsMode = "sequence",
        mode: SamplingMode | None = None,
    ) -> None:
        super().__init__()
        if action_head not in ("continuous", "tokens"):
            raise ValueError(
                f"action_head must be 'continuous' or 'tokens', got {action_head!r}."
            )
        if action_head == "tokens" and not vocab_size:
            raise ValueError("vocab_size must be set for the 'tokens' action head.")
        if mode is not None:
            if mode == "sample":
                default_interaction_type = InteractionType.RANDOM
            elif mode == "greedy":
                default_interaction_type = InteractionType.DETERMINISTIC
            else:
                raise ValueError(f"mode must be 'greedy' or 'sample', got {mode!r}.")
        if not isinstance(default_interaction_type, InteractionType):
            raise ValueError(
                "default_interaction_type must be an InteractionType, got "
                f"{default_interaction_type!r}."
            )
        if log_probs_mode not in ("sequence", "token"):
            raise ValueError(
                f"log_probs_mode must be 'sequence' or 'token', got {log_probs_mode!r}."
            )
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.action_head = action_head
        self.vocab_size = vocab_size
        self.use_state = bool(use_state)
        self.default_interaction_type = default_interaction_type
        self.log_probs_mode = log_probs_mode
        self._tensor_keys = self._AcceptedKeys()
        self._update_keys()

    @property
    def tensor_keys(self) -> _AcceptedKeys:
        return self._tensor_keys

    def set_keys(self, **kwargs) -> VLAWrapperBase:
        """Set the tensordict key names used by the policy (see ``_AcceptedKeys``)."""
        for key, value in kwargs.items():
            if key not in self._AcceptedKeys.__dataclass_fields__:
                raise ValueError(
                    f"{key!r} is not an accepted key. Accepted keys are "
                    f"{list(self._AcceptedKeys.__dataclass_fields__)}."
                )
            if value is not None:
                setattr(self._tensor_keys, key, value)
        self._update_keys()
        return self

    def _update_keys(self) -> None:
        in_keys = [self._tensor_keys.image]
        if self.use_state:
            in_keys.append(self._tensor_keys.state)
        in_keys.append(self._tensor_keys.instruction)
        self.in_keys = in_keys
        if self.action_head == "continuous":
            self.out_keys = [self._tensor_keys.action_chunk]
        else:
            self.out_keys = [
                self._tensor_keys.action_tokens,
                self._tensor_keys.log_probs,
            ]

    # -- hook implemented by concrete policies ---------------------------------
    def _predict(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Return the raw head output for a batch of observations.

        Shape ``[*B, chunk_size * action_dim]`` for the continuous head, or
        ``[*B, chunk_size * action_dim * vocab_size]`` for the token head; the
        base reshapes it into an action chunk or per-token logits.
        """
        raise NotImplementedError

    def _action_logits(self, tensordict: TensorDictBase) -> torch.Tensor:
        return self._predict(tensordict).unflatten(
            -1, (self.chunk_size, self.action_dim, self.vocab_size)
        )

    # -- TensorDict API --------------------------------------------------------
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.action_head == "continuous":
            chunk = self._predict(tensordict).unflatten(
                -1, (self.chunk_size, self.action_dim)
            )
            tensordict.set(self._tensor_keys.action_chunk, chunk)
        else:
            dist = self.get_dist(tensordict)
            logits = (
                dist.base_dist.logits
                if isinstance(dist, torch_dist.Independent)
                else dist.logits
            )
            # consult the ambient exploration context (set by the collector
            # during rollout, or by set_exploration_type at eval); fall back
            # to the policy's default when no context is active
            interaction = interaction_type()
            if interaction is None:
                interaction = self.default_interaction_type
            tokens = (
                dist.sample()
                if interaction == InteractionType.RANDOM
                else logits.argmax(-1)
            )
            tensordict.set(self._tensor_keys.action_tokens, tokens)
            tensordict.set(self._tensor_keys.log_probs, dist.log_prob(tokens))
        return tensordict

    def get_dist(self, tensordict: TensorDictBase) -> torch_dist.Distribution:
        """Return the action-token distribution.

        Only defined for the ``"tokens"`` action head: a
        :class:`~torch.distributions.Categorical` over the vocabulary. With
        ``log_probs_mode="sequence"`` (default) it is wrapped in
        :class:`~torch.distributions.Independent` over the
        ``(chunk_size, action_dim)`` token dims, so ``log_prob`` returns one
        *sequence-level* log-probability per sample. This is the contract
        PPO-style objectives expect: token RL fine-tuning works directly with
        :class:`~torchrl.objectives.ClipPPOLoss` (pass
        ``critic_network=None``, ``entropy_bonus=False`` and remap the keys
        via ``set_keys``). With ``log_probs_mode="token"``, the bare
        per-token :class:`~torch.distributions.Categorical` is returned and
        ``log_prob`` is per token, ``[*B, chunk_size, action_dim]`` -- one
        importance ratio per token for DAPO-style objectives.
        """
        if self.action_head != "tokens":
            raise RuntimeError(
                "get_dist is only defined for the 'tokens' action head; the "
                "'continuous' head is a deterministic regressor."
            )
        dist = torch_dist.Categorical(logits=self._action_logits(tensordict))
        if self.log_probs_mode == "sequence":
            return torch_dist.Independent(dist, 2)
        return dist
