# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""TRL Interoperability Adapters for TorchRL.

This module provides thin adapters for interoperability between
TorchRL and Hugging Face ``trl``:

* :class:`TorchRLBufferDataset` — wraps a :class:`~torchrl.data.ReplayBuffer` as a
  ``torch.utils.data.IterableDataset`` and can expose a Hugging Face
  :class:`datasets.IterableDataset` for trainers that require the
  ``datasets`` interface.

* :class:`HFRewardModelWrapper` — wraps a Hugging Face reward model (e.g., one
  trained via ``trl.RewardTrainer``) as a
  :class:`~tensordict.nn.TensorDictModuleBase` so it can be plugged into any
  TorchRL training loop, including GRPO recipes, without any boilerplate.

Importing this module does not require ``datasets``, ``transformers``, or
``trl``. The optional ``datasets`` dependency is checked only when
:meth:`TorchRLBufferDataset.as_hf_dataset` is called.
"""

from __future__ import annotations

import contextlib
import importlib.util
from collections.abc import Iterator
from typing import TYPE_CHECKING, Any

import torch
from tensordict import NonTensorData, NonTensorStack, TensorDictBase
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import NestedKey
from torch import nn

from torchrl._utils import logger as torchrl_logger
from torchrl.data.replay_buffers import ReplayBuffer

_has_datasets = importlib.util.find_spec("datasets") is not None

if TYPE_CHECKING:
    from datasets import IterableDataset as HFIterableDataset
else:
    HFIterableDataset = Any

__all__ = [
    "HFRewardModelWrapper",
    "TorchRLBufferDataset",
]


# ---------------------------------------------------------------------------
# TorchRLBufferDataset — TorchRL -> TRL
# ---------------------------------------------------------------------------


class TorchRLBufferDataset(torch.utils.data.IterableDataset):
    """An :class:`torch.utils.data.IterableDataset` backed by a TorchRL :class:`~torchrl.data.ReplayBuffer`.

    The PyTorch dataset can be consumed directly by
    :class:`transformers.Trainer`. Trainers such as :class:`trl.GRPOTrainer`
    that require a Hugging Face :class:`datasets.IterableDataset` can consume
    the object returned by :meth:`as_hf_dataset`.

    Each sampling call draws ``batch_size`` entries from the replay buffer and
    yields them individually as flat ``dict[str, Any]`` objects. By default an
    iterator samples one replay batch. Set ``num_batches=None`` for an
    unbounded online stream; consumers of such a stream must impose their own
    step limit.

    .. note::
        This class implements :class:`torch.utils.data.IterableDataset` (no
        ``__len__``), which is the safest choice for online / infinite replay
        buffers.  If you need a finite dataset with a known length, iterate
        for a fixed number of steps yourself and collect the results.

    Args:
        replay_buffer (:class:`~torchrl.data.ReplayBuffer`): the TorchRL
            replay buffer to wrap.
        batch_size (int): number of samples to draw from the buffer per
            internal sampling call.  Each yielded item is one *individual*
            sample (no leading batch dimension).

    Keyword Args:
        keys (list of :class:`~tensordict.NestedKey`, optional): if provided,
            only these keys are included in the yielded dicts.  Nested keys
            are serialised as ``"key0.key1"`` strings so they remain
            compatible with HuggingFace collators.  Defaults to ``None``
            (all leaf keys, with nested keys flattened).
        device (torch.device or str, optional): if provided, all tensors are
            moved to this device before yielding.  Defaults to ``None``
            (tensors stay on their current device).
        num_batches (int or None, optional): number of replay batches sampled
            by each iterator. ``None`` produces an unbounded stream. Defaults
            to ``1``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.data import ReplayBuffer, ListStorage
        >>> from torchrl.modules.llm.trl_interop import TorchRLBufferDataset
        >>>
        >>> rb = ReplayBuffer(storage=ListStorage(100), batch_size=4)
        >>> for _ in range(10):
        ...     _ = rb.add(TensorDict(
        ...         {"input_ids": torch.randint(0, 100, (8,)),
        ...          "attention_mask": torch.ones(8, dtype=torch.long)},
        ...         batch_size=[],
        ...     ))
        >>>
        >>> dataset = TorchRLBufferDataset(rb, batch_size=4)
        >>> sample = next(iter(dataset))
        >>> sample["input_ids"].shape
        torch.Size([8])

    .. seealso::
        :class:`HFRewardModelWrapper` for the reverse direction (TRL -> TorchRL).
    """

    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        batch_size: int,
        *,
        keys: list[NestedKey] | None = None,
        device: torch.device | str | None = None,
        num_batches: int | None = 1,
    ) -> None:
        if not isinstance(replay_buffer, ReplayBuffer):
            raise TypeError(
                f"replay_buffer must be a torchrl ReplayBuffer instance, "
                f"got {type(replay_buffer).__name__}."
            )
        if not isinstance(batch_size, int) or batch_size <= 0:
            raise ValueError(
                f"batch_size must be a positive integer, got {batch_size!r}."
            )
        if num_batches is not None and (
            not isinstance(num_batches, int) or num_batches <= 0
        ):
            raise ValueError(
                f"num_batches must be a positive integer or None, got {num_batches!r}."
            )
        self._replay_buffer = replay_buffer
        self._batch_size = batch_size
        self._keys = keys
        self._device = torch.device(device) if device is not None else None
        self._num_batches = num_batches

    # ------------------------------------------------------------------
    # IterableDataset protocol
    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict[str, Any]]:
        """Sample replay batches and yield individual, flattened samples."""
        batch_index = 0
        while self._num_batches is None or batch_index < self._num_batches:
            batch: TensorDictBase = self._replay_buffer.sample(self._batch_size)
            batch_keys: list[NestedKey] = []
            for key in batch.keys(include_nested=True):
                value = batch.get(key)
                if not isinstance(value, TensorDictBase) or isinstance(
                    value, NonTensorStack
                ):
                    batch_keys.append(key)

            if self._keys is not None:
                key_list = self._keys
            else:
                key_list = batch_keys

            available_keys: list[NestedKey] = []
            batch_key_set = set(batch_keys)
            for key in key_list:
                if key in batch_key_set:
                    available_keys.append(key)
                else:
                    torchrl_logger.warning(
                        f"TorchRLBufferDataset: key {key!r} not found in sampled "
                        "TensorDict -- skipping."
                    )

            for i in range(batch.batch_size[0]):
                sample_td = batch[i]
                out: dict[str, Any] = {}
                for key in available_keys:
                    value = sample_td.get(key)
                    if value is None:
                        continue
                    if isinstance(value, NonTensorData):
                        value = value.data
                    elif self._device is not None and hasattr(value, "to"):
                        value = value.to(self._device)
                    if isinstance(key, str):
                        str_key: str = key
                    else:
                        str_key = ".".join(str(k) for k in key)
                    out[str_key] = value
                yield out
            batch_index += 1

    def as_hf_dataset(self) -> HFIterableDataset:
        """Return a Hugging Face iterable dataset backed by this adapter.

        The returned object is accepted by current ``trl`` trainers, which
        require :class:`datasets.Dataset` or
        :class:`datasets.IterableDataset` rather than a PyTorch iterable
        dataset. The replay samples must still contain the schema required by
        the selected trainer, such as a top-level ``"prompt"`` field for
        :class:`trl.GRPOTrainer`.

        Returns:
            A :class:`datasets.IterableDataset` that yields the same samples
            as this adapter.

        Raises:
            ImportError: if the optional ``datasets`` package is unavailable.
        """
        if not _has_datasets:
            raise ImportError(
                "TorchRLBufferDataset.as_hf_dataset requires the optional "
                "'datasets' dependency. Install it with `pip install datasets`."
            )
        from datasets import IterableDataset

        return IterableDataset.from_generator(self.__iter__)

    def __repr__(self) -> str:
        keys_repr = self._keys if self._keys is not None else "<all>"
        return (
            f"{self.__class__.__name__}("
            f"replay_buffer={self._replay_buffer!r}, "
            f"batch_size={self._batch_size}, "
            f"keys={keys_repr}, "
            f"device={self._device}, "
            f"num_batches={self._num_batches})"
        )


# ---------------------------------------------------------------------------
# HFRewardModelWrapper — TRL -> TorchRL
# ---------------------------------------------------------------------------


class _HFRewardModule(nn.Module):
    """Inner nn.Module that calls the HF reward model and returns a scalar reward tensor.

    This is kept separate from :class:`HFRewardModelWrapper` so that
    :class:`~tensordict.nn.TensorDictModuleBase` controls all TensorDict
    bookkeeping while this class focuses purely on the HF model call.
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        inference_mode: bool,
    ) -> None:
        super().__init__()
        self.model = model
        self._inference_mode = inference_mode

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Call the HF reward model and return a 1-D reward tensor of shape ``[B]``."""
        if self._inference_mode:
            ctx = torch.inference_mode()
        else:
            ctx = contextlib.nullcontext()

        with ctx:
            if attention_mask is None:
                out = self.model(input_ids=input_ids)
            else:
                out = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

        # HF reward models typically return a ModelOutput with a ``logits``
        # attribute of shape [B, 1] or [B].  Handle both shapes.
        if hasattr(out, "logits"):
            rewards: torch.Tensor = out.logits
        elif hasattr(out, "rewards"):
            rewards = out.rewards
        elif isinstance(out, torch.Tensor):
            rewards = out
        else:
            raise RuntimeError(
                "HFRewardModelWrapper: could not extract reward from model output. "
                f"Got type {type(out).__name__}. Expected an output with a 'logits' "
                "or 'rewards' attribute, or a bare torch.Tensor."
            )

        expected_shape = input_ids.shape[:-1]
        if rewards.shape == expected_shape + (1,):
            rewards = rewards.squeeze(-1)
        elif rewards.shape != expected_shape:
            raise RuntimeError(
                "HFRewardModelWrapper: expected one scalar reward per input "
                f"with shape {expected_shape} or {expected_shape + (1,)}, got "
                f"{rewards.shape}."
            )
        return rewards.float()


class HFRewardModelWrapper(TensorDictModuleBase):
    """A :class:`~tensordict.nn.TensorDictModuleBase` that wraps a Hugging Face reward model.

    This adapter allows any HuggingFace sequence-classification / reward model
    (e.g., one trained via ``trl.RewardTrainer`` or
    ``transformers.AutoModelForSequenceClassification``) to be used as a reward
    signal inside a TorchRL training loop, including GRPO / PPO recipes.

    On :meth:`forward`, the wrapper:

    1. Reads ``token_key`` (input token ids) and ``attention_mask_key`` from
       the incoming :class:`~tensordict.TensorDictBase`.
    2. Calls the wrapped HF model with ``input_ids`` and ``attention_mask``.
    3. Extracts the scalar reward (``logits`` or ``rewards`` attribute, squeezed
       to shape ``[B]``).
    4. Writes the reward to ``reward_key`` in the output TensorDict.

    Args:
        model (nn.Module): a Hugging Face reward model.  Typically an instance
            of ``AutoModelForSequenceClassification`` or any model whose
            ``forward`` accepts ``input_ids`` + ``attention_mask`` and returns
            an output with a ``logits`` attribute of shape ``[B, 1]`` or ``[B]``.

    Keyword Args:
        token_key (:class:`~tensordict.NestedKey`, optional): key from which
            ``input_ids`` are read.  Defaults to ``("tokens", "full")``,
            matching the :class:`~torchrl.modules.llm.policies.Tokens` layout
            used by TorchRL LLM wrappers.
        attention_mask_key (:class:`~tensordict.NestedKey` or ``None``, optional):
            key from which the attention mask is read.  Pass ``None`` to omit
            the attention mask (model must support this).  Defaults to
            ``("masks", "all_attention_mask")``.
        reward_key (:class:`~tensordict.NestedKey`, optional): key under which
            the scalar reward tensor (shape ``[B]``) is written.  Defaults to
            ``"reward"``.
        inference_mode (bool, optional): if ``True``, the model forward pass is
            wrapped in :func:`torch.inference_mode`, which disables gradient
            computation and is more memory-efficient.  Set to ``False`` (the
            default) when you need gradients (e.g. for a differentiable critic
            loss in PPO / GRPO).  Defaults to ``False``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.modules.llm.trl_interop import HFRewardModelWrapper
        >>>
        >>> class DummyRewardModel(torch.nn.Module):
        ...     def forward(self, input_ids, attention_mask=None):
        ...         class Out:
        ...             logits = torch.randn(input_ids.shape[0], 1)
        ...         return Out()
        >>>
        >>> wrapper = HFRewardModelWrapper(DummyRewardModel())
        >>> td = TensorDict(
        ...     {
        ...         "tokens": {"full": torch.randint(0, 1000, (2, 16))},
        ...         "masks": {"all_attention_mask": torch.ones(2, 16, dtype=torch.long)},
        ...     },
        ...     batch_size=[2],
        ... )
        >>> result = wrapper(td)
        >>> result["reward"].shape
        torch.Size([2])

    .. seealso::
        :class:`TorchRLBufferDataset` for the reverse direction (TorchRL -> TRL).
    """

    def __init__(
        self,
        model: nn.Module,
        *,
        token_key: NestedKey = ("tokens", "full"),
        attention_mask_key: NestedKey | None = ("masks", "all_attention_mask"),
        reward_key: NestedKey = "reward",
        inference_mode: bool = False,
    ) -> None:
        in_keys: list[NestedKey] = [token_key]
        if attention_mask_key is not None:
            in_keys.append(attention_mask_key)

        out_keys: list[NestedKey] = [reward_key]

        super().__init__()

        self.in_keys = in_keys
        self.out_keys = out_keys

        self._token_key = token_key
        self._attention_mask_key = attention_mask_key
        self._reward_key = reward_key

        self._reward_module = _HFRewardModule(model, inference_mode=inference_mode)

    # ------------------------------------------------------------------
    # Public accessors
    # ------------------------------------------------------------------

    @property
    def model(self) -> nn.Module:
        """The underlying HF reward model."""
        return self._reward_module.model

    # ------------------------------------------------------------------
    # TensorDictModuleBase forward
    # ------------------------------------------------------------------

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Read tokens from ``tensordict``, call the reward model, write reward back.

        Args:
            tensordict (:class:`~tensordict.TensorDictBase`): input data.
                Must contain the keys specified by ``token_key`` and, if not
                ``None``, ``attention_mask_key``.

        Returns:
            The same (modified in-place) :class:`~tensordict.TensorDictBase`
            with the scalar reward tensor written to ``reward_key``.
        """
        input_ids: torch.Tensor = tensordict.get(self._token_key)
        if input_ids is None:
            raise KeyError(
                f"HFRewardModelWrapper: token_key {self._token_key!r} not found "
                "in the input TensorDict."
            )

        attention_mask: torch.Tensor | None = None
        if self._attention_mask_key is not None:
            attention_mask = tensordict.get(self._attention_mask_key, default=None)
            if attention_mask is None:
                raise KeyError(
                    "HFRewardModelWrapper: attention_mask_key "
                    f"{self._attention_mask_key!r} not found in the input TensorDict."
                )

        reward = self._reward_module(input_ids, attention_mask)

        tensordict.set(self._reward_key, reward)
        return tensordict
