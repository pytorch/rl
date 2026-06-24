# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""A small reference Vision-Language-Action policy for tests and tutorials."""
from __future__ import annotations

import hashlib

import torch
from tensordict import TensorDictBase
from tensordict.nn import InteractionType
from torch import nn

from torchrl.data.utils import DEVICE_TYPING
from torchrl.data.vla import ActionTokenizerBase
from torchrl.modules.models.models import ConvNet, MLP

from torchrl.modules.vla.common import (
    ActionHead,
    InputMode,
    LogProbsMode,
    OutputMode,
    SamplingMode,
    VLAWrapperBase,
)

__all__ = ["TinyVLA"]


class TinyVLA(VLAWrapperBase):
    """A tiny, dependency-free reference VLA policy for CI and tutorials.

    ``TinyVLA`` fuses a small convolutional image encoder, an optional
    proprioceptive-state MLP, and a hashed language-instruction embedding into a
    trunk that feeds either a continuous action-chunk head or a discrete
    action-token head (see :class:`~torchrl.modules.vla.VLAWrapperBase`). It is
    intentionally small and CPU-friendly -- a stand-in to exercise the VLA data
    pipeline, losses and collectors end-to-end, **not** a competitive policy.

    The language instruction is embedded by hashing the instruction string to an
    embedding-table index (a deterministic, tokenizer-free stand-in), so the
    policy is genuinely language-conditioned without any external dependency.

    .. note::
        ``TinyVLA`` expects observations with a single leading batch dimension
        (``image`` shaped ``[B, C, H, W]``). When training on chunked windows,
        flatten the time dimension into the batch first.

    Keyword Args:
        action_dim (int): the dimensionality of a single action.
        chunk_size (int): the action-chunk horizon ``H``.
        action_head (str): ``"continuous"`` (default) or ``"tokens"``.
        vocab_size (int): action-token bins per dimension (token head).
            Defaults to ``256``.
        use_state (bool): whether to read the proprioceptive state.
            Defaults to ``True``.
        hidden_dim (int): width of the fused trunk. Defaults to ``128``.
        text_vocab (int): size of the hashed instruction embedding table.
            Defaults to ``256``.
        text_dim (int): instruction-embedding dimension. Defaults to ``32``.
        default_interaction_type (InteractionType): token-head readout when no
            exploration context is active (``RANDOM`` samples, else argmax);
            the forward otherwise follows the ambient
            :func:`~torchrl.envs.utils.exploration_type`. Defaults to
            ``InteractionType.DETERMINISTIC``. See
            :class:`~torchrl.modules.vla.VLAWrapperBase`.
        mode (str, optional): backward-compatible alias for
            ``default_interaction_type``. Defaults to ``None``.
        device (DEVICE_TYPING, optional): device to move the parameters to.

    Examples:
        >>> import torch
        >>> from tensordict import NonTensorStack, TensorDict
        >>> from torchrl.modules.vla import TinyVLA
        >>> policy = TinyVLA(action_dim=7, chunk_size=4)
        >>> td = TensorDict(
        ...     {
        ...         "observation": {
        ...             "image": torch.zeros(2, 3, 16, 16, dtype=torch.uint8),
        ...             "state": torch.zeros(2, 5),
        ...         },
        ...         "language_instruction": NonTensorStack("pick", "place"),
        ...     },
        ...     batch_size=[2],
        ... )
        >>> policy(td)["vla_action", "chunk"].shape
        torch.Size([2, 4, 7])
    """

    def __init__(
        self,
        *,
        action_dim: int,
        chunk_size: int,
        action_head: ActionHead = "continuous",
        vocab_size: int = 256,
        use_state: bool = True,
        hidden_dim: int = 128,
        text_vocab: int = 256,
        text_dim: int = 32,
        default_interaction_type: InteractionType = InteractionType.DETERMINISTIC,
        log_probs_mode: LogProbsMode = "sequence",
        mode: SamplingMode | None = None,
        device: DEVICE_TYPING | None = None,
        input_mode: InputMode = "canonical",
        output_mode: OutputMode | None = None,
        action_tokenizer: ActionTokenizerBase | None = None,
        return_log_probs: bool | None = None,
        return_logits: bool = False,
        logits_only: bool = False,
        inplace: bool | str | None = True,
        num_samples: int | None = None,
    ) -> None:
        super().__init__(
            action_dim=action_dim,
            chunk_size=chunk_size,
            action_head=action_head,
            vocab_size=vocab_size,
            use_state=use_state,
            input_mode=input_mode,
            output_mode=output_mode,
            action_tokenizer=action_tokenizer,
            default_interaction_type=default_interaction_type,
            log_probs_mode=log_probs_mode,
            return_log_probs=return_log_probs,
            return_logits=return_logits,
            logits_only=logits_only,
            inplace=inplace,
            num_samples=num_samples,
            mode=mode,
            device=device,
        )
        self.hidden_dim = int(hidden_dim)
        self.text_vocab = int(text_vocab)
        self.image_encoder = nn.Sequential(
            ConvNet(num_cells=[16, 32], kernel_sizes=3, strides=2, paddings=1),
            nn.LazyLinear(hidden_dim),
            nn.ReLU(),
        )
        self.state_encoder = (
            MLP(out_features=hidden_dim, num_cells=[hidden_dim]) if use_state else None
        )
        self.text_embedding = nn.Embedding(text_vocab, text_dim)
        self.trunk = MLP(out_features=hidden_dim, num_cells=[hidden_dim])
        out_features = chunk_size * action_dim
        if action_head == "tokens":
            out_features *= vocab_size
        self.head = nn.LazyLinear(out_features)
        if device is not None:
            self.to(device)

    def _hash_text(self, strings: list[str], device: torch.device) -> torch.Tensor:
        indices = [
            int(hashlib.sha1(s.encode("utf-8")).hexdigest(), 16) % self.text_vocab
            for s in strings
        ]
        return torch.tensor(indices, dtype=torch.long, device=device)

    def _instruction_strings(self, tensordict: TensorDictBase, batch: int) -> list[str]:
        instruction = self._get_instruction(tensordict)
        data = getattr(instruction, "tolist", lambda: instruction)()
        if isinstance(data, str):
            data = [data] * batch
        elif not isinstance(data, list):
            data = [str(data)] * batch
        return data

    def _features(self, tensordict: TensorDictBase) -> torch.Tensor:
        image = self._get_image(tensordict)
        if image.dtype == torch.uint8:
            image = image.float() / 255.0
        else:
            image = image.float()
        batch = image.shape[0]
        feats = [self.image_encoder(image)]
        if self.use_state:
            feats.append(self.state_encoder(self._get_state(tensordict).float()))
        strings = self._instruction_strings(tensordict, batch)
        feats.append(self.text_embedding(self._hash_text(strings, image.device)))
        return self.trunk(torch.cat(feats, dim=-1))

    def _predict(self, tensordict: TensorDictBase) -> torch.Tensor:
        return self.head(self._features(tensordict))
