# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Base class for Vision-Language-Action (VLA) policies."""
from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Literal

import torch
from tensordict import TensorDict, TensorDictBase
from tensordict.nn import InteractionType, TensorDictModuleBase
from tensordict.nn.probabilistic import interaction_type
from tensordict.utils import NestedKey
from torch import distributions as torch_dist

from torchrl.data.vla.containers import VLAAction, VLAObservation
from torchrl.data.vla.schema import (
    ACTION_CHUNK_KEY,
    ACTION_KEY,
    ACTION_TOKENS_KEY,
    IMAGE_KEY,
    INSTRUCTION_KEY,
    STATE_KEY,
    VLA_ACTION_KEY,
)
from torchrl.data.vla.tokenizers import ActionTokenizerBase

__all__ = ["VLAWrapperBase"]

ActionHead = Literal["continuous", "tokens"]
InputMode = Literal["canonical", "preprocessed"]
LogProbsMode = Literal["sequence", "token"]
OutputMode = Literal["chunk", "tokens", "both"]
SamplingMode = Literal["greedy", "sample"]


class VLAWrapperBase(TensorDictModuleBase):
    """Base class for TensorDict-native Vision-Language-Action policies.

    A VLA policy maps images, optional proprioceptive state, and a language
    instruction to either a continuous action chunk or discrete action tokens.
    Outputs are stored in a structured :class:`~torchrl.data.vla.VLAAction`
    container under ``"vla_action"`` by default. Its fields are ordinary nested
    TensorDict keys, e.g. ``("vla_action", "chunk")`` for continuous chunks.

    Keyword Args:
        action_dim (int): The dimensionality of a single action.
        chunk_size (int): The action-chunk horizon.
        action_head (str): ``"continuous"`` or ``"tokens"``.
        input_mode (str): ``"canonical"`` reads raw VLA keys. ``"preprocessed"``
            reads a :class:`~torchrl.data.vla.VLAObservation` or
            :class:`~tensordict.TensorDictBase` from ``observation_key``.
        output_mode (str, optional): ``"chunk"``, ``"tokens"`` or ``"both"``.
            Defaults to ``"chunk"`` for continuous heads and ``"tokens"`` for
            token heads.
        return_vla_action_container (bool): whether to write the structured
            :class:`~torchrl.data.vla.VLAAction` object at the VLA action root
            key. When ``False``, only its plain TensorDict fields are written.
            Defaults to ``True``.
        vocab_size (int, optional): Number of action-token bins, required for
            token heads.
        action_tokenizer (ActionTokenizerBase, optional): Token/chunk codec used
            when ``output_mode`` asks for both representations.
        return_log_probs (bool, optional): Whether token ``forward`` writes
            log-probabilities. Defaults to ``True`` for token heads.
        return_logits (bool): Whether token ``forward`` writes ``action_logits``.
        logits_only (bool): Whether token ``forward`` returns logits without
            sampling actions by default. A per-call ``logits_only=True`` argument
            also enables this path.
        log_probs_mode (str): ``"sequence"`` returns one summed log-probability
            per sample; ``"token"`` returns per-token log-probabilities.
        use_state (bool): Whether canonical mode reads the state key.
        default_interaction_type (InteractionType): Token readout when no
            exploration context is active.
        mode (str, optional): Backward-compatible alias mapping ``"sample"`` to
            ``InteractionType.RANDOM`` and ``"greedy"`` to deterministic.
        inplace (bool | "empty" | None): Output TensorDict behavior. ``True``
            updates the input, ``False`` returns a new output TensorDict, and
            ``"empty"`` returns an empty TensorDict populated with outputs.
        num_samples (int, optional): Number of token samples to draw per input.

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
        >>> out = policy(td)
        >>> out["vla_action"].chunk.shape
        torch.Size([2, 4, 7])
        >>> out["vla_action", "chunk"].shape
        torch.Size([2, 4, 7])
    """

    @dataclass
    class _AcceptedKeys:
        """Configurable tensordict keys for a VLA policy."""

        image: NestedKey = IMAGE_KEY
        wrist_image: NestedKey | None = ("observation", "wrist_image")
        state: NestedKey | None = STATE_KEY
        instruction: NestedKey = INSTRUCTION_KEY
        observation: NestedKey = "vla_observation"
        action: NestedKey = ACTION_KEY
        vla_action: NestedKey = VLA_ACTION_KEY
        action_chunk: NestedKey = ACTION_CHUNK_KEY
        action_tokens: NestedKey = ACTION_TOKENS_KEY
        action_logits: NestedKey = (VLA_ACTION_KEY, "logits")
        action_mask: NestedKey = (VLA_ACTION_KEY, "mask")
        log_probs: NestedKey = (VLA_ACTION_KEY, "log_probs")

    def __init__(
        self,
        model=None,
        *,
        processor=None,
        input_mode: InputMode = "canonical",
        output_mode: OutputMode | None = None,
        return_vla_action_container: bool = True,
        action_dim: int,
        chunk_size: int,
        action_head: ActionHead = "continuous",
        vocab_size: int | None = None,
        action_tokenizer: ActionTokenizerBase | None = None,
        action_kwargs: dict | None = None,
        return_log_probs: bool | None = None,
        return_logits: bool = False,
        logits_only: bool = False,
        use_state: bool = True,
        default_interaction_type: InteractionType = InteractionType.DETERMINISTIC,
        log_probs_mode: LogProbsMode = "sequence",
        mode: SamplingMode | None = None,
        image_key: NestedKey = IMAGE_KEY,
        wrist_image_key: NestedKey | None = ("observation", "wrist_image"),
        state_key: NestedKey | None = STATE_KEY,
        instruction_key: NestedKey = INSTRUCTION_KEY,
        observation_key: NestedKey = "vla_observation",
        action_key: NestedKey = ACTION_KEY,
        vla_action_key: NestedKey = VLA_ACTION_KEY,
        action_chunk_key: NestedKey | None = None,
        action_tokens_key: NestedKey | None = None,
        action_logits_key: NestedKey | None = None,
        action_mask_key: NestedKey | None = None,
        log_probs_key: NestedKey | None = None,
        inplace: Literal[True, False, "empty"] | None = True,
        device: torch.device | str | None = None,
        num_samples: int | None = None,
    ) -> None:
        super().__init__()
        if action_head not in ("continuous", "tokens"):
            raise ValueError(
                f"action_head must be 'continuous' or 'tokens', got {action_head!r}."
            )
        if input_mode not in ("canonical", "preprocessed"):
            raise ValueError(
                f"input_mode must be 'canonical' or 'preprocessed', got {input_mode!r}."
            )
        if output_mode is None:
            output_mode = "chunk" if action_head == "continuous" else "tokens"
        if output_mode not in ("chunk", "tokens", "both"):
            raise ValueError(
                f"output_mode must be 'chunk', 'tokens' or 'both', got {output_mode!r}."
            )
        if action_head == "tokens" and vocab_size is None:
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
        if inplace not in (True, False, "empty", None):
            raise ValueError(
                "inplace must be True, False, 'empty' or None, got " f"{inplace!r}."
            )
        if num_samples is not None and int(num_samples) < 1:
            raise ValueError(f"num_samples must be >= 1, got {num_samples}.")
        self.model = model
        self.processor = processor
        self.action_dim = int(action_dim)
        self.chunk_size = int(chunk_size)
        self.action_head = action_head
        self.input_mode = input_mode
        self.output_mode = output_mode
        self.return_vla_action_container = bool(return_vla_action_container)
        self.vocab_size = None if vocab_size is None else int(vocab_size)
        self.action_tokenizer = action_tokenizer
        self.action_kwargs = {} if action_kwargs is None else dict(action_kwargs)
        self.return_log_probs = (
            action_head == "tokens"
            if return_log_probs is None
            else bool(return_log_probs)
        )
        self.return_logits = bool(return_logits)
        self.logits_only = bool(logits_only)
        self.use_state = bool(use_state)
        self.default_interaction_type = default_interaction_type
        self.log_probs_mode = log_probs_mode
        self.inplace = True if inplace is None else inplace
        self.device = None if device is None else torch.device(device)
        self.num_samples = None if num_samples is None else int(num_samples)
        if action_chunk_key is None:
            action_chunk_key = self._vla_field_key(vla_action_key, "chunk")
        if action_tokens_key is None:
            action_tokens_key = self._vla_field_key(vla_action_key, "tokens")
        if action_logits_key is None:
            action_logits_key = self._vla_field_key(vla_action_key, "logits")
        if action_mask_key is None:
            action_mask_key = self._vla_field_key(vla_action_key, "mask")
        if log_probs_key is None:
            log_probs_key = self._vla_field_key(vla_action_key, "log_probs")
        self._tensor_keys = self._AcceptedKeys(
            image=image_key,
            wrist_image=wrist_image_key,
            state=state_key,
            instruction=instruction_key,
            observation=observation_key,
            action=action_key,
            vla_action=vla_action_key,
            action_chunk=action_chunk_key,
            action_tokens=action_tokens_key,
            action_logits=action_logits_key,
            action_mask=action_mask_key,
            log_probs=log_probs_key,
        )
        self._update_keys()

    @property
    def tensor_keys(self) -> _AcceptedKeys:
        return self._tensor_keys

    @staticmethod
    def _vla_field_key(vla_action_key: NestedKey, field: str) -> NestedKey:
        if isinstance(vla_action_key, str):
            return (vla_action_key, field)
        return (*vla_action_key, field)

    def _is_vla_field_key(self, key: NestedKey, field: str) -> bool:
        return key == self._vla_field_key(self._tensor_keys.vla_action, field)

    def set_keys(self, **kwargs) -> VLAWrapperBase:
        """Set the tensordict key names used by the policy."""
        old_vla_action_key = self._tensor_keys.vla_action
        vla_field_keys = (
            ("action_chunk", "chunk"),
            ("action_tokens", "tokens"),
            ("action_logits", "logits"),
            ("action_mask", "mask"),
            ("log_probs", "log_probs"),
        )
        old_default_field_keys = {
            key: self._vla_field_key(old_vla_action_key, field)
            for key, field in vla_field_keys
        }
        for key, value in kwargs.items():
            if key not in self._AcceptedKeys.__dataclass_fields__:
                raise ValueError(
                    f"{key!r} is not an accepted key. Accepted keys are "
                    f"{list(self._AcceptedKeys.__dataclass_fields__)}."
                )
            setattr(self._tensor_keys, key, value)
        if "vla_action" in kwargs:
            for key, field in vla_field_keys:
                if key not in kwargs and (
                    getattr(self._tensor_keys, key) == old_default_field_keys[key]
                ):
                    setattr(
                        self._tensor_keys,
                        key,
                        self._vla_field_key(self._tensor_keys.vla_action, field),
                    )
        self._update_keys()
        return self

    def _update_keys(self) -> None:
        if self.input_mode == "canonical":
            in_keys = [self._tensor_keys.image]
            if self.use_state and self._tensor_keys.state is not None:
                in_keys.append(self._tensor_keys.state)
            in_keys.append(self._tensor_keys.instruction)
        else:
            in_keys = [self._tensor_keys.observation]
        self.in_keys = in_keys
        out_keys = []
        if self.action_head == "tokens" and self.logits_only:
            out_keys.append(self._tensor_keys.action_logits)
            self.out_keys = out_keys
            return
        if self.output_mode in ("chunk", "both"):
            out_keys.append(self._tensor_keys.action_chunk)
        if self.output_mode in ("tokens", "both"):
            out_keys.append(self._tensor_keys.action_tokens)
        if self.action_head == "tokens" and self.return_log_probs:
            out_keys.append(self._tensor_keys.log_probs)
        if self.action_head == "tokens" and self.return_logits:
            out_keys.append(self._tensor_keys.action_logits)
        self.out_keys = out_keys

    # -- input helpers -----------------------------------------------------
    def _preprocessed_observation(self, tensordict: TensorDictBase):
        obs = tensordict.get(self._tensor_keys.observation)
        if isinstance(obs, VLAObservation) and obs.preprocessed is not None:
            return obs.preprocessed
        return obs

    def _get_image(self, tensordict: TensorDictBase) -> torch.Tensor:
        if self.input_mode == "canonical":
            return tensordict.get(self._tensor_keys.image)
        obs = self._preprocessed_observation(tensordict)
        if isinstance(obs, VLAObservation):
            if obs.images is None or obs.images.image is None:
                raise KeyError("preprocessed VLAObservation has no primary image.")
            return obs.images.image
        if isinstance(obs, TensorDictBase):
            value = obs.get(self._tensor_keys.image, None)
            if value is None:
                value = obs.get("image")
            return value
        raise TypeError(
            "preprocessed input must be a VLAObservation or TensorDictBase, got "
            f"{type(obs)}."
        )

    def _get_state(self, tensordict: TensorDictBase) -> torch.Tensor:
        if self.input_mode == "canonical":
            return tensordict.get(self._tensor_keys.state)
        obs = self._preprocessed_observation(tensordict)
        if isinstance(obs, VLAObservation):
            return obs.state
        if isinstance(obs, TensorDictBase):
            value = None
            if self._tensor_keys.state is not None:
                value = obs.get(self._tensor_keys.state, None)
            if value is None:
                value = obs.get("state")
            return value
        raise TypeError(
            "preprocessed input must be a VLAObservation or TensorDictBase, got "
            f"{type(obs)}."
        )

    def _get_instruction(self, tensordict: TensorDictBase):
        if self.input_mode == "canonical":
            return tensordict.get(self._tensor_keys.instruction)
        obs = self._preprocessed_observation(tensordict)
        if isinstance(obs, VLAObservation):
            return obs.instruction
        if isinstance(obs, TensorDictBase):
            value = obs.get(self._tensor_keys.instruction, None)
            if value is None:
                value = obs.get("instruction")
            return value
        raise TypeError(
            "preprocessed input must be a VLAObservation or TensorDictBase, got "
            f"{type(obs)}."
        )

    # -- hooks implemented by concrete policies ---------------------------
    def _predict(self, tensordict: TensorDictBase) -> torch.Tensor:
        """Return flattened continuous chunks or token logits."""
        raise NotImplementedError

    def _predict_chunk(self, tensordict: TensorDictBase) -> torch.Tensor:
        return self._predict(tensordict).unflatten(
            -1, (self.chunk_size, self.action_dim)
        )

    def _predict_logits(self, tensordict: TensorDictBase) -> torch.Tensor:
        return self._predict(tensordict).unflatten(
            -1, (self.chunk_size, self.action_dim, self.vocab_size)
        )

    def _action_logits(self, tensordict: TensorDictBase) -> torch.Tensor:
        return self._predict_logits(tensordict)

    # -- output helpers ----------------------------------------------------
    def _output_tensordict(
        self, tensordict: TensorDictBase, out: TensorDictBase, tensordict_out=None
    ) -> TensorDictBase:
        if tensordict_out is None:
            if self.inplace is True:
                tensordict_out = tensordict
            elif self.inplace is False:
                tensordict_out = out
            else:
                tensordict_out = TensorDict(
                    {}, batch_size=out.batch_size, device=out.device
                )
        if tensordict_out is not out:
            vla_action_key = self._tensor_keys.vla_action
            if out.get(vla_action_key, None) is not None:
                tensordict_out.set(vla_action_key, out.get(vla_action_key))
            vla_action_prefix = (
                (vla_action_key,)
                if isinstance(vla_action_key, str)
                else tuple(vla_action_key)
            )
            for key in out.keys(True, True):
                key_tuple = (key,) if isinstance(key, str) else tuple(key)
                if key_tuple[: len(vla_action_prefix)] == vla_action_prefix:
                    continue
                tensordict_out.set(key, out.get(key))
            return tensordict_out
        return out

    def _set_action(
        self,
        out: TensorDictBase,
        *,
        chunk: torch.Tensor | None = None,
        tokens: torch.Tensor | None = None,
        logits: torch.Tensor | None = None,
        log_probs: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> None:
        if self.return_vla_action_container:
            action = VLAAction(
                chunk=chunk,
                tokens=tokens,
                logits=logits,
                log_probs=log_probs,
                mask=mask,
                batch_size=out.batch_size,
                device=out.device,
            )
            out.set(self._tensor_keys.vla_action, action)
        if chunk is not None and (
            not self.return_vla_action_container
            or not self._is_vla_field_key(self._tensor_keys.action_chunk, "chunk")
        ):
            out.set(self._tensor_keys.action_chunk, chunk)
        if tokens is not None and (
            not self.return_vla_action_container
            or not self._is_vla_field_key(self._tensor_keys.action_tokens, "tokens")
        ):
            out.set(self._tensor_keys.action_tokens, tokens)
        if logits is not None and (
            not self.return_vla_action_container
            or not self._is_vla_field_key(self._tensor_keys.action_logits, "logits")
        ):
            out.set(self._tensor_keys.action_logits, logits)
        if log_probs is not None and (
            not self.return_vla_action_container
            or not self._is_vla_field_key(self._tensor_keys.log_probs, "log_probs")
        ):
            out.set(self._tensor_keys.log_probs, log_probs)
        if mask is not None and (
            not self.return_vla_action_container
            or not self._is_vla_field_key(self._tensor_keys.action_mask, "mask")
        ):
            out.set(self._tensor_keys.action_mask, mask)

    def _dist_from_logits(
        self, logits: torch.Tensor, mask: torch.Tensor | None = None
    ) -> torch_dist.Distribution:
        if mask is not None:
            logits = logits.masked_fill(~mask.to(torch.bool), -torch.inf)
        dist = torch_dist.Categorical(logits=logits)
        if self.log_probs_mode == "sequence":
            return torch_dist.Independent(dist, 2)
        return dist

    def _sample_tokens(
        self, dist: torch_dist.Distribution, logits: torch.Tensor
    ) -> torch.Tensor:
        interaction = interaction_type()
        if interaction is None:
            interaction = self.default_interaction_type
        batch_ndim = logits.ndim - 3
        greedy = logits.argmax(-1)
        if self.num_samples is None:
            return dist.sample() if interaction == InteractionType.RANDOM else greedy
        if interaction == InteractionType.RANDOM:
            tokens = dist.sample((self.num_samples,))
            return tokens.movedim(0, batch_ndim)
        return greedy.unsqueeze(batch_ndim).expand(
            *greedy.shape[:batch_ndim],
            self.num_samples,
            *greedy.shape[batch_ndim:],
        )

    def _log_prob_tokens(
        self, dist: torch_dist.Distribution, tokens: torch.Tensor, logits: torch.Tensor
    ) -> torch.Tensor:
        if self.num_samples is None or tokens.ndim == logits.ndim - 1:
            return dist.log_prob(tokens)
        batch_ndim = logits.ndim - 3
        tokens_for_dist = tokens.movedim(batch_ndim, 0)
        log_probs = dist.log_prob(tokens_for_dist)
        return log_probs.movedim(0, batch_ndim)

    # -- TensorDict API ----------------------------------------------------
    def forward(
        self,
        tensordict: TensorDictBase,
        *,
        tensordict_out: TensorDictBase | None = None,
        logits_only: bool = False,
        **kwargs,
    ) -> TensorDictBase:
        if self.action_head == "continuous":
            chunk = self._predict_chunk(tensordict)
            tokens = None
            if self.output_mode in ("tokens", "both"):
                if self.action_tokenizer is None:
                    raise RuntimeError(
                        "output_mode requires action tokens but no action_tokenizer "
                        "was provided."
                    )
                tokens = self.action_tokenizer.encode(chunk)
            out = TensorDict({}, batch_size=chunk.shape[:-2], device=chunk.device)
            self._set_action(
                out,
                chunk=chunk if self.output_mode in ("chunk", "both") else None,
                tokens=tokens,
            )
            return self._output_tensordict(tensordict, out, tensordict_out)

        logits = self._action_logits(tensordict)
        mask = tensordict.get(self._tensor_keys.action_mask, None)
        dist = self._dist_from_logits(logits, mask)
        logits_only = self.logits_only or logits_only
        if logits_only:
            out = TensorDict({}, batch_size=logits.shape[:-3], device=logits.device)
            self._set_action(out, logits=logits, mask=mask)
            return self._output_tensordict(tensordict, out, tensordict_out)

        tokens = self._sample_tokens(dist, logits)
        log_probs = self._log_prob_tokens(dist, tokens, logits)
        chunk = None
        if self.output_mode in ("chunk", "both"):
            if self.action_tokenizer is None:
                raise RuntimeError(
                    "output_mode requires decoded chunks but no action_tokenizer "
                    "was provided."
                )
            chunk = self.action_tokenizer.decode(tokens)
        logits_out = logits if self.return_logits else None
        mask_out = mask
        if logits_out is not None and self.num_samples is not None:
            batch_ndim = logits.ndim - 3
            logits_out = logits.unsqueeze(batch_ndim).expand(
                *logits.shape[:batch_ndim],
                self.num_samples,
                *logits.shape[batch_ndim:],
            )
        if mask_out is not None and self.num_samples is not None:
            batch_ndim = logits.ndim - 3
            mask_out = mask.unsqueeze(batch_ndim).expand(
                *mask.shape[:batch_ndim],
                self.num_samples,
                *mask.shape[batch_ndim:],
            )
        out = TensorDict({}, batch_size=tokens.shape[:-2], device=tokens.device)
        self._set_action(
            out,
            chunk=chunk,
            tokens=tokens if self.output_mode in ("tokens", "both") else None,
            logits=logits_out,
            log_probs=log_probs if self.return_log_probs else None,
            mask=mask_out,
        )
        return self._output_tensordict(tensordict, out, tensordict_out)

    def get_dist(
        self,
        tensordict: TensorDictBase,
        *,
        tensordict_out: TensorDictBase | None = None,
        logits_key: NestedKey | None = None,
        mask_key: NestedKey | None = None,
        **kwargs,
    ) -> torch_dist.Distribution:
        """Return the action-token distribution for loss-time recomputation."""
        if self.action_head != "tokens":
            raise RuntimeError(
                "get_dist is only defined for the 'tokens' action head; the "
                "'continuous' head is a deterministic regressor."
            )
        logits_key = (
            self._tensor_keys.action_logits if logits_key is None else logits_key
        )
        mask_key = self._tensor_keys.action_mask if mask_key is None else mask_key
        logits = None
        if tensordict_out is not None:
            logits = tensordict_out.get(logits_key, None)
        if logits is None:
            logits = tensordict.get(logits_key, None)
        if logits is None:
            logits = self._action_logits(tensordict)
        mask = None
        if mask_key is not None:
            if tensordict_out is not None:
                mask = tensordict_out.get(mask_key, None)
            if mask is None:
                mask = tensordict.get(mask_key, None)
        return self._dist_from_logits(logits, mask)

    def log_prob(
        self,
        tensordict: TensorDictBase,
        *,
        action_key: NestedKey | None = None,
        log_probs_key: NestedKey | None = None,
        **kwargs,
    ) -> TensorDictBase:
        """Recompute and write token log-probabilities for stored actions."""
        if self.action_head != "tokens":
            raise RuntimeError("log_prob is only defined for token VLA policies.")
        action_key = (
            self._tensor_keys.action_tokens if action_key is None else action_key
        )
        log_probs_key = (
            self._tensor_keys.log_probs if log_probs_key is None else log_probs_key
        )
        tokens = tensordict.get(action_key)
        logits = tensordict.get(self._tensor_keys.action_logits, None)
        if logits is None:
            logits = self._action_logits(tensordict)
        mask = tensordict.get(self._tensor_keys.action_mask, None)
        dist = self._dist_from_logits(logits, mask)
        log_probs = self._log_prob_tokens(dist, tokens, logits)
        tensordict.set(log_probs_key, log_probs)
        action = tensordict.get(self._tensor_keys.vla_action, None)
        if isinstance(action, VLAAction):
            action.log_probs = log_probs
        return tensordict

    def get_new_version(self, **kwargs) -> VLAWrapperBase:
        """Return a shallow wrapper copy with altered runtime parameters."""
        new = copy.copy(self)
        new._modules = self._modules.copy()
        new._parameters = self._parameters.copy()
        new._buffers = self._buffers.copy()
        new._tensor_keys = copy.copy(self._tensor_keys)
        for key, value in kwargs.items():
            if key.endswith("_key"):
                field = key[: -len("_key")]
                if field in self._AcceptedKeys.__dataclass_fields__:
                    setattr(new._tensor_keys, field, value)
                    continue
            if not hasattr(new, key):
                raise TypeError(f"{type(self).__name__} has no parameter {key!r}.")
            setattr(new, key, value)
        new._update_keys()
        return new
