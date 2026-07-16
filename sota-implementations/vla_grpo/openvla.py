# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""OpenVLA-OFT policy wrappers for the VLA GRPO recipe.

The token wrapper covers the vendored SimpleVLA-RL token-OFT model (see
``openvla_oft/``): one forward emits the whole action chunk (parallel decoding
-- ``chunk_size * action_dim`` tokens in a single language-model pass, no
autoregressive loop), sampled from the 256-way categorical over the
action-token window at the tail of the LLaMA-2 vocabulary.

The module also provides a separate continuous L1-head wrapper for the official
OpenVLA-OFT checkpoint family. That path is deterministic and intended for
reference/evaluation rollouts; the token GRPO training semantics stay
unchanged.

The wrappers own ALL model-side preprocessing (prompt construction, image
transforms) and, for token heads, the temperature scaling. Both rollout and
loss-time recomputation go through the same :meth:`get_dist`, so with
identical weights the PPO importance ratio is exactly 1 -- the temperature
contract a T != 1 rollout requires.
"""
from __future__ import annotations

import importlib.util
import io
import json
import os
from typing import Literal

import numpy as np
import torch
from tensordict import TensorDictBase
from tensordict.nn import InteractionType, TensorDictSequential
from tensordict.utils import NestedKey, unravel_key
from torch import nn
from torch.nn.utils.rnn import pad_sequence
from torchrl._utils import logger as torchrl_logger

from torchrl.data.vla import (
    ACTION_CHUNK_KEY,
    OpenVLAImagePreprocessor,
    VocabTailActionTokenizer,
)
from torchrl.envs.transforms import ActionTokenizerTransform
from torchrl.envs.transforms._base import Transform
from torchrl.modules.vla import VLAWrapperBase

from torchrl.modules.vla.common import LogProbsMode

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_timm = importlib.util.find_spec("timm") is not None
_has_pil = importlib.util.find_spec("PIL") is not None
_has_huggingface_hub = importlib.util.find_spec("huggingface_hub") is not None
_hf_hub_download = None

# the special empty token LLaMA-2 emits after "Out:" at training time
_EMPTY_TOKEN_ID = 29871

PROMPT_TEMPLATE = "In: What action should the robot take to {instruction}?\nOut:"

# OpenVLA-OFT center-crop: ``crop_scale`` is an AREA fraction (the eval-time
# augmentation match, ``experiments/robot/openvla_utils.py``). The linear crop
# is its square root, so a 0.9 area crop keeps ~94.87% of each side -- NOT 90%.
# Cropping at a linear 0.9 (0.81 area) zooms ~5% harder than training and
# miscalibrates the policy's learned pixel->action mapping.
_CROP_AREA_SCALE = 0.9
_CROP_LINEAR_SCALE = _CROP_AREA_SCALE**0.5

# OpenVLA-OFT eval feeds the policy a 224x224 image, resized from the rendered
# frame with a high-quality (lanczos3) filter and JPEG-round-tripped so the
# pixels carry the same compression artifacts as the JPEG-encoded RLDS training
# data (experiments/robot/openvla_utils.py:resize_image_for_policy). Skipping
# either step hands the model out-of-distribution (too-clean / wrong-scaled)
# pixels, costing closed-loop precision.
_OPENVLA_IMAGE_SIZE = 224
_JPEG_QUALITY = 95
_OPENVLA_INPUT_IDS_KEY = ("openvla", "input_ids")
_OPENVLA_ATTENTION_MASK_KEY = ("openvla", "attention_mask")
_OPENVLA_PIXEL_VALUES_KEY = ("openvla", "pixel_values")


class _MLPResNetBlock(nn.Module):
    """Residual MLP block used by the OpenVLA-OFT L1 action head."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.ffn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim), nn.ReLU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.ffn(x) + x


class _MLPResNet(nn.Module):
    """Small residual MLP used by the OpenVLA-OFT L1 action head."""

    def __init__(
        self,
        *,
        num_blocks: int,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
    ) -> None:
        super().__init__()
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.mlp_resnet_blocks = nn.ModuleList(
            [_MLPResNetBlock(hidden_dim) for _ in range(num_blocks)]
        )
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(self.layer_norm1(x)))
        for block in self.mlp_resnet_blocks:
            x = block(x)
        return self.fc2(self.layer_norm2(x))


class _L1RegressionActionHead(nn.Module):
    """Continuous OpenVLA-OFT L1 action head."""

    def __init__(
        self,
        *,
        input_dim: int = 4096,
        hidden_dim: int = 4096,
        action_dim: int = 7,
    ) -> None:
        super().__init__()
        from openvla_oft.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

        self.action_dim = int(action_dim)
        self.chunk_size = int(NUM_ACTIONS_CHUNK)
        self.model = _MLPResNet(
            num_blocks=2,
            input_dim=input_dim * ACTION_DIM,
            hidden_dim=hidden_dim,
            output_dim=action_dim,
        )

    def predict_action(self, actions_hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size = actions_hidden_states.shape[0]
        rearranged = actions_hidden_states.reshape(batch_size, self.chunk_size, -1)
        return self.model(rearranged)


def _hf_hub_download_fn():
    global _hf_hub_download
    if not _has_huggingface_hub:
        raise ImportError("Hugging Face Hub is required to download OpenVLA weights.")
    if _hf_hub_download is None:
        from huggingface_hub import hf_hub_download

        _hf_hub_download = hf_hub_download
    return _hf_hub_download


def _load_dataset_statistics(spec: str) -> dict:
    """Load an OpenVLA ``dataset_statistics.json`` from a local path or HF repo.

    ``spec`` is either a filesystem path to the json or a HuggingFace repo id
    that ships a ``dataset_statistics.json`` at its root.
    """
    if os.path.isfile(spec):
        path = spec
    else:
        path = _hf_hub_download_fn()(spec, "dataset_statistics.json")
    with open(path) as f:
        return json.load(f)


class OpenVLAInputTransform(Transform):
    """Build OpenVLA prompt tokens and image tensors from canonical VLA inputs.

    The transform reads TorchRL's canonical VLA observation keys and writes the
    preprocessed prompt/image entries consumed by :class:`OpenVLAModelTransform`.
    It is intentionally a :class:`~torchrl.envs.Transform`: the same object can
    be used in an environment, a replay-buffer transform, or a
    :class:`~tensordict.nn.TensorDictSequential` policy stack.
    """

    def __init__(
        self,
        processor,
        *,
        use_wrist_image: bool = False,
        center_crop: bool = False,
        image_backend: Literal[
            "torchvision", "torch_reference", "pil", "tensorflow"
        ] = "torch_reference",
        image_key: NestedKey = ("observation", "image"),
        wrist_image_key: NestedKey | None = ("observation", "wrist_image"),
        instruction_key: NestedKey = "language_instruction",
        input_ids_key: NestedKey = _OPENVLA_INPUT_IDS_KEY,
        attention_mask_key: NestedKey = _OPENVLA_ATTENTION_MASK_KEY,
        pixel_values_key: NestedKey = _OPENVLA_PIXEL_VALUES_KEY,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        image_key = unravel_key(image_key)
        wrist_image_key = (
            None if wrist_image_key is None else unravel_key(wrist_image_key)
        )
        instruction_key = unravel_key(instruction_key)
        input_ids_key = unravel_key(input_ids_key)
        attention_mask_key = unravel_key(attention_mask_key)
        pixel_values_key = unravel_key(pixel_values_key)
        in_keys = [image_key, instruction_key]
        if use_wrist_image and wrist_image_key is not None:
            in_keys.append(wrist_image_key)
        super().__init__(
            in_keys=in_keys,
            out_keys=[
                input_ids_key,
                attention_mask_key,
                pixel_values_key,
            ],
        )
        self.processor = processor
        self.use_wrist_image = bool(use_wrist_image)
        self.center_crop = bool(center_crop)
        self.image_backend = image_backend
        self.image_key = image_key
        self.wrist_image_key = wrist_image_key
        self.instruction_key = instruction_key
        self.input_ids_key = input_ids_key
        self.attention_mask_key = attention_mask_key
        self.pixel_values_key = pixel_values_key
        self.device = None if device is None else torch.device(device)
        self.dtype = dtype
        self._prompt_cache: dict[str, torch.Tensor] = {}
        self._image_preprocessor = self._make_image_preprocessor(
            processor, image_backend
        )
        if self._image_preprocessor is not None:
            self.image_backend = self._image_preprocessor.backend

    def _to_device(
        self, tensor: torch.Tensor, *, dtype: torch.dtype | None = None
    ) -> torch.Tensor:
        device = self.device
        if device is None and dtype is None:
            return tensor
        return tensor.to(device=device, dtype=dtype)

    def _to_pil(self, image: torch.Tensor):
        from PIL import Image

        array = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        pil = Image.fromarray(array).convert("RGB")
        size = _OPENVLA_IMAGE_SIZE
        if pil.size != (size, size):
            pil = pil.resize((size, size), Image.LANCZOS)
        buffer = io.BytesIO()
        pil.save(buffer, format="JPEG", quality=_JPEG_QUALITY)
        buffer.seek(0)
        pil = Image.open(buffer).convert("RGB")
        if self.center_crop:
            width, height = pil.size
            crop_w = int(round(width * _CROP_LINEAR_SCALE))
            crop_h = int(round(height * _CROP_LINEAR_SCALE))
            left = (width - crop_w) // 2
            top = (height - crop_h) // 2
            pil = pil.crop((left, top, left + crop_w, top + crop_h)).resize(
                (width, height), Image.LANCZOS
            )
        return pil

    def _instructions(self, tensordict: TensorDictBase, batch: int) -> list[str]:
        instruction = tensordict.get(self.instruction_key)
        data = getattr(instruction, "tolist", lambda: instruction)()
        if isinstance(data, str):
            data = [data] * batch
        elif len(data) != batch:
            flat_data = []
            stack = list(data)
            while stack:
                item = stack.pop(0)
                if isinstance(item, str):
                    flat_data.append(item)
                elif isinstance(item, (list, tuple)):
                    stack[:0] = list(item)
                else:
                    flat_data.append(item)
            if len(flat_data) == 1:
                data = flat_data * batch
            else:
                data = flat_data
        return [str(item) for item in data]

    def _prompt_input_ids(self, instruction: str) -> torch.Tensor | None:
        tokenizer = getattr(self.processor, "tokenizer", None)
        if not callable(tokenizer):
            return None
        prompt = PROMPT_TEMPLATE.format(instruction=instruction.lower())
        input_ids = self._prompt_cache.get(prompt)
        if input_ids is None:
            tokenized = tokenizer(prompt, return_tensors="pt")
            input_ids = tokenized.input_ids.squeeze(0).to(
                dtype=torch.long, device="cpu"
            )
            if input_ids.numel() == 0 or int(input_ids[-1]) != _EMPTY_TOKEN_ID:
                input_ids = torch.cat(
                    (input_ids, torch.tensor([_EMPTY_TOKEN_ID], dtype=torch.long))
                )
            self._prompt_cache[prompt] = input_ids
        return input_ids

    @staticmethod
    def _image_processor_config(
        image_processor,
    ) -> tuple[int, torch.Tensor, torch.Tensor] | None:
        input_sizes = getattr(image_processor, "input_sizes", None)
        normalize_params = getattr(image_processor, "tvf_normalize_params", None)
        if (
            input_sizes is None
            or normalize_params is None
            or not input_sizes
            or len(input_sizes) != len(normalize_params)
            or any(
                tuple(input_size[-2:]) != (_OPENVLA_IMAGE_SIZE, _OPENVLA_IMAGE_SIZE)
                for input_size in input_sizes
            )
        ):
            return None
        return (
            int(input_sizes[0][-1]),
            torch.as_tensor(
                [params["mean"] for params in normalize_params],
                dtype=torch.float32,
            ),
            torch.as_tensor(
                [params["std"] for params in normalize_params],
                dtype=torch.float32,
            ),
        )

    def _make_image_preprocessor(
        self,
        processor,
        image_backend: Literal["torchvision", "torch_reference", "pil", "tensorflow"],
    ) -> OpenVLAImagePreprocessor | None:
        image_processor = getattr(processor, "image_processor", None)
        if image_processor is None:
            return None
        config = self._image_processor_config(image_processor)
        if config is None:
            return None
        size, mean, std = config
        try:
            return OpenVLAImagePreprocessor(
                size=size,
                jpeg_quality=_JPEG_QUALITY,
                center_crop=self.center_crop,
                backend=image_backend,
                mean=mean,
                std=std,
            )
        except ImportError as err:
            if image_backend not in ("torchvision", "torch_reference"):
                raise
            torchrl_logger.warning(
                "OpenVLA image backend %r is unavailable (%s); falling back to "
                "'pil', which does not preserve reference preprocessing semantics.",
                image_backend,
                err,
            )
            return OpenVLAImagePreprocessor(
                size=size,
                jpeg_quality=_JPEG_QUALITY,
                center_crop=self.center_crop,
                backend="pil",
                mean=mean,
                std=std,
            )

    def _pixel_values_from_images(self, images: torch.Tensor) -> torch.Tensor | None:
        if self._image_preprocessor is None:
            return None
        if images.device.type != "cpu":
            images = images.cpu()
        return self._image_preprocessor(images)

    def _preprocess(self, images, wrist_images, instructions):
        pad_token_id = self.processor.tokenizer.pad_token_id
        input_ids_list = [
            self._prompt_input_ids(instruction) for instruction in instructions
        ]
        if all(input_ids is not None for input_ids in input_ids_list):
            pixel_values = self._pixel_values_from_images(images)
            if pixel_values is not None:
                if wrist_images is not None:
                    wrist_pixel_values = self._pixel_values_from_images(wrist_images)
                    if wrist_pixel_values is None:
                        return self._preprocess_slow(images, wrist_images, instructions)
                    pixel_values = torch.cat((pixel_values, wrist_pixel_values), dim=1)
                input_ids = pad_sequence(
                    input_ids_list, batch_first=True, padding_value=pad_token_id
                )
                attention_mask = input_ids.ne(pad_token_id).long()
                return (
                    self._to_device(input_ids),
                    self._to_device(attention_mask),
                    self._to_device(pixel_values, dtype=self.dtype),
                )
        return self._preprocess_slow(images, wrist_images, instructions)

    def _preprocess_slow(self, images, wrist_images, instructions):
        pad_token_id = self.processor.tokenizer.pad_token_id
        input_ids_list, pixel_values_list = [], []
        for index, instruction in enumerate(instructions):
            prompt = PROMPT_TEMPLATE.format(instruction=instruction.lower())
            image = self._to_pil(images[index])
            features = self.processor(prompt, image)
            input_ids = features["input_ids"]
            pixel_values = [features["pixel_values"]]
            if wrist_images is not None:
                wrist = self.processor(prompt, self._to_pil(wrist_images[index]))
                pixel_values.append(wrist["pixel_values"])
            if not torch.all(input_ids[:, -1] == _EMPTY_TOKEN_ID):
                input_ids = torch.cat(
                    (input_ids, torch.tensor([[_EMPTY_TOKEN_ID]], dtype=torch.long)),
                    dim=1,
                )
            input_ids_list.append(input_ids.squeeze(0))
            pixel_values_list.append(torch.cat(pixel_values, dim=1))
        input_ids = pad_sequence(
            input_ids_list, batch_first=True, padding_value=pad_token_id
        )
        attention_mask = input_ids.ne(pad_token_id).long()
        pixel_values = torch.cat(pixel_values_list, dim=0)
        return (
            self._to_device(input_ids),
            self._to_device(attention_mask),
            self._to_device(pixel_values, dtype=self.dtype),
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        images = tensordict.get(self.image_key)
        batch_dims = images.shape[:-3]
        if images.ndim == 3:
            images = images.unsqueeze(0)
        else:
            images = images.reshape(-1, *images.shape[-3:])
        wrist_images = None
        if self.use_wrist_image:
            wrist_images = tensordict.get(self.wrist_image_key)
            if wrist_images.ndim == 3:
                wrist_images = wrist_images.unsqueeze(0)
            else:
                wrist_images = wrist_images.reshape(-1, *wrist_images.shape[-3:])
        instructions = self._instructions(tensordict, images.shape[0])
        input_ids, attention_mask, pixel_values = self._preprocess(
            images, wrist_images, instructions
        )
        if batch_dims:
            input_ids = input_ids.reshape(*batch_dims, input_ids.shape[-1])
            attention_mask = attention_mask.reshape(
                *batch_dims, attention_mask.shape[-1]
            )
            pixel_values = pixel_values.reshape(*batch_dims, *pixel_values.shape[-3:])
        tensordict.set(self.input_ids_key, input_ids)
        tensordict.set(self.attention_mask_key, attention_mask)
        tensordict.set(self.pixel_values_key, pixel_values)
        return tensordict

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        return self.forward(next_tensordict)


class OpenVLAModelTransform(Transform):
    """Map preprocessed OpenVLA inputs to action-token logits."""

    def __init__(
        self,
        model,
        processor,
        *,
        chunk_size: int,
        action_dim: int,
        num_bins: int,
        temperature: float = 1.0,
        top_k: int | None = None,
        micro_batch_size: int | None = None,
        image_key: NestedKey = ("observation", "image"),
        input_ids_key: NestedKey = _OPENVLA_INPUT_IDS_KEY,
        attention_mask_key: NestedKey = _OPENVLA_ATTENTION_MASK_KEY,
        pixel_values_key: NestedKey = _OPENVLA_PIXEL_VALUES_KEY,
        logits_key: NestedKey = ("vla_action", "logits"),
    ) -> None:
        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        if top_k is not None and int(top_k) < 1:
            raise ValueError(f"top_k must be >= 1 when provided, got {top_k}.")
        if micro_batch_size is not None and int(micro_batch_size) < 1:
            raise ValueError(
                f"micro_batch_size must be >= 1 when provided, got {micro_batch_size}."
            )
        image_key = unravel_key(image_key)
        input_ids_key = unravel_key(input_ids_key)
        attention_mask_key = unravel_key(attention_mask_key)
        pixel_values_key = unravel_key(pixel_values_key)
        logits_key = unravel_key(logits_key)
        super().__init__(
            in_keys=[
                image_key,
                input_ids_key,
                attention_mask_key,
                pixel_values_key,
            ],
            out_keys=[logits_key],
        )
        self.model = model
        self.processor = processor
        self.chunk_size = int(chunk_size)
        self.action_dim = int(action_dim)
        self.num_bins = int(num_bins)
        self.temperature = float(temperature)
        self.top_k = int(top_k) if top_k is not None else None
        self.micro_batch_size = (
            int(micro_batch_size) if micro_batch_size is not None else None
        )
        self.image_key = image_key
        self.input_ids_key = input_ids_key
        self.attention_mask_key = attention_mask_key
        self.pixel_values_key = pixel_values_key
        self.logits_key = logits_key

    def _window_logits(self, input_ids, attention_mask, pixel_values):
        from openvla_oft.constants import IGNORE_INDEX

        model = self.model
        pad_token_id = self.processor.tokenizer.pad_token_id
        labels = input_ids.clone()
        labels[:] = IGNORE_INDEX
        num_prompt_tokens = input_ids.ne(pad_token_id).sum(dim=1) - 1
        input_ids, attention_mask = model._prepare_input_for_action_prediction(
            input_ids, attention_mask
        )
        labels = model._prepare_labels_for_action_prediction(labels, input_ids)
        padding_mask = (~input_ids.ne(pad_token_id)).int()
        sorted_indices = torch.argsort(
            padding_mask, dim=1, descending=False, stable=True
        )
        input_ids = torch.gather(input_ids, 1, sorted_indices)
        attention_mask = torch.gather(attention_mask, 1, sorted_indices)
        labels = torch.gather(labels, 1, sorted_indices)
        input_embeddings = model.get_input_embeddings()(input_ids)
        all_actions_mask = model._process_action_masks(labels)
        language_embeddings = input_embeddings[~all_actions_mask].reshape(
            input_embeddings.shape[0], -1, input_embeddings.shape[2]
        )
        projected_patch_embeddings = model._process_vision_features(
            pixel_values, language_embeddings, False
        )
        num_patches = (
            model.vision_backbone.get_num_patches()
            * model.vision_backbone.get_num_images_in_input()
        )
        input_embeddings = input_embeddings * ~all_actions_mask.unsqueeze(-1)
        (
            multimodal_embeddings,
            multimodal_attention_mask,
        ) = model._build_multimodal_attention(
            input_embeddings, projected_patch_embeddings, attention_mask
        )
        output = model.language_model(
            input_ids=None,
            attention_mask=multimodal_attention_mask,
            inputs_embeds=multimodal_embeddings,
            use_cache=False,
            return_dict=True,
        )
        batch_size = output.logits.shape[0]
        device = output.logits.device
        start = (num_prompt_tokens.to(device) + num_patches).unsqueeze(1)
        offsets = torch.arange(
            self.chunk_size * self.action_dim, device=device
        ).unsqueeze(0)
        positions = start + offsets
        logits = output.logits[
            torch.arange(batch_size, device=device).unsqueeze(-1), positions, :
        ]
        return logits[..., model.vocab_size - self.num_bins : model.vocab_size]

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        image = tensordict.get(self.image_key)
        batch_dims = image.shape[:-3]
        input_ids = tensordict.get(self.input_ids_key)
        attention_mask = tensordict.get(self.attention_mask_key)
        pixel_values = tensordict.get(self.pixel_values_key)
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        input_ids = input_ids.reshape(-1, input_ids.shape[-1]).to(device)
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[-1]).to(device)
        pixel_values = pixel_values.reshape(-1, *pixel_values.shape[-3:]).to(
            device=device, dtype=dtype
        )
        micro_batch_size = self.micro_batch_size
        if micro_batch_size is None or input_ids.shape[0] <= micro_batch_size:
            window = self._window_logits(input_ids, attention_mask, pixel_values)
        else:
            window = torch.cat(
                [
                    self._window_logits(
                        input_ids[start : start + micro_batch_size],
                        attention_mask[start : start + micro_batch_size],
                        pixel_values[start : start + micro_batch_size],
                    )
                    for start in range(0, input_ids.shape[0], micro_batch_size)
                ],
                dim=0,
            )
        window = window.float() / self.temperature
        if self.top_k is not None and self.top_k < window.shape[-1]:
            indices = torch.argsort(window, dim=-1, descending=True, stable=True)[
                ..., : self.top_k
            ]
            values = torch.gather(window, -1, indices)
            masked = torch.full_like(window, -torch.inf)
            window = masked.scatter(-1, indices, values)
        logits = window.reshape(
            *batch_dims, self.chunk_size, self.action_dim, self.num_bins
        )
        tensordict.set(self.logits_key, logits)
        return tensordict

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        return self.forward(next_tensordict)


class GripperPostProcessTransform(Transform):
    """Apply the SimpleVLA/OpenVLA gripper convention to decoded actions.

    SimpleVLA decodes the gripper scalar, maps it through ``2 * g - 1``,
    binarizes by sign, then optionally inverts for LIBERO. Keeping this as a
    transform makes the action-tokenizer a pure token<->continuous codec and
    lets env-side decode and policy-side decode share the same postprocess.
    """

    def __init__(
        self,
        *,
        action_key: NestedKey = ACTION_CHUNK_KEY,
        out_key: NestedKey | None = None,
        rescale: bool = True,
        binarize: bool = True,
        threshold: float = 0.0,
        invert: bool = False,
    ) -> None:
        action_key = unravel_key(action_key)
        out_key = action_key if out_key is None else unravel_key(out_key)
        self.action_key = action_key
        self.out_key = out_key
        self.rescale = bool(rescale)
        self.binarize = bool(binarize)
        self.threshold = float(threshold)
        self.invert = bool(invert)
        super().__init__(
            in_keys=[action_key],
            out_keys=[out_key],
            in_keys_inv=[action_key],
            out_keys_inv=[out_key],
        )

    def postprocess(self, actions: torch.Tensor) -> torch.Tensor:
        """Return actions with SimpleVLA/OpenVLA gripper post-processing."""
        gripper = actions[..., -1]
        if self.rescale:
            gripper = 2.0 * gripper - 1.0
        if self.binarize:
            gripper = (gripper > self.threshold).to(actions.dtype) * 2.0 - 1.0
        if self.invert:
            gripper = -gripper
        actions = actions.clone()
        actions[..., -1] = gripper
        return actions

    def _postprocess(self, actions: torch.Tensor) -> torch.Tensor:
        return self.postprocess(actions)

    def _apply_transform(self, actions: torch.Tensor) -> torch.Tensor:
        return self.postprocess(actions)

    def _inv_apply_transform(self, actions: torch.Tensor) -> torch.Tensor:
        return self.postprocess(actions)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Postprocess actions when present and ignore observation-only calls."""
        return self._call(tensordict)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        actions = next_tensordict.get(self.action_key, None)
        if actions is None:
            return next_tensordict
        next_tensordict.set(self.out_key, self.postprocess(actions))
        return next_tensordict


def register_openvla_oft() -> None:
    """Register the vendored token-OFT classes with the transformers Auto* APIs."""
    from openvla_oft.configuration_prismatic import OpenVLAConfig
    from openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
    from openvla_oft.processing_prismatic import (
        PrismaticImageProcessor,
        PrismaticProcessor,
    )
    from transformers import (
        AutoConfig,
        AutoImageProcessor,
        AutoModelForVision2Seq,
        AutoProcessor,
    )

    # The vendored modeling code predates the attention-implementation
    # dispatch added in recent transformers (>=4.5x), which probes
    # ``_supports_sdpa`` / ``_supports_flash_attn_*`` class attributes during
    # ``PreTrainedModel.__init__``. Declaring no support routes the model to
    # eager attention -- exactly the path OpenVLA was trained and evaluated
    # with -- so the numerics (and the SFT success rate) are preserved while
    # the vendored code stays verbatim.
    for attr in (
        "_supports_sdpa",
        "_supports_flash_attn_2",
        "_supports_flash_attn_3",
        "_supports_flex_attn",
        "_supports_attention_backend",
    ):
        if not hasattr(OpenVLAForActionPrediction, attr):
            setattr(OpenVLAForActionPrediction, attr, False)

    try:
        AutoConfig.register("openvla", OpenVLAConfig)
        AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
        AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
        AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)
    except ValueError:
        # already registered
        pass


class OpenVLAOFTWrapper(VLAWrapperBase):
    """Token-head OpenVLA-OFT policy speaking the canonical VLA TensorDict schema.

    Reads ``("observation", "image")`` (uint8 ``[*B, 3, H, W]``), optionally
    ``("observation", "wrist_image")``, and ``language_instruction``; writes
    ``("vla_action", "tokens")`` (``[*B, chunk_size, action_dim]`` ids in the
    256-way action-token *window*) and their ``("vla_action", "log_probs")``.
    Decode the tokens to environment actions with :attr:`action_tokenizer`
    (e.g. through :class:`~torchrl.envs.transforms.ActionTokenizerTransform`).

    Args:
        model: a vendored ``OpenVLAForActionPrediction`` (token variant).
        processor: the matching ``PrismaticProcessor``.

    Keyword Args:
        unnorm_key (str, optional): dataset key of the normalization
            statistics (e.g. ``"libero_spatial_no_noops"``). Defaults to the
            checkpoint's single key when unambiguous.
        temperature (float, optional): sampling temperature, applied
            identically when sampling rollout actions and when recomputing
            log-probabilities at loss time. Defaults to ``1.0``.
        top_k (int, optional): if provided, restrict each action-token
            categorical to its top-k logits before sampling and log-prob
            recomputation. This keeps exploration local while preserving the
            rollout/loss distribution contract. Defaults to ``None`` (full
            categorical).
        micro_batch_size (int, optional): maximum number of observations in
            each model forward. This can reproduce single-environment
            inference while the surrounding collector remains batched.
            Defaults to ``None`` (one model forward for the complete batch).
        default_interaction_type (InteractionType, optional): token readout
            when no exploration context is active (``RANDOM`` samples, else
            argmax); the forward otherwise follows the ambient
            :func:`~torchrl.envs.utils.exploration_type`. Defaults to
            ``InteractionType.DETERMINISTIC``. See
            :class:`~torchrl.modules.vla.VLAWrapperBase`.
        log_probs_mode (str, optional): ``"sequence"`` or ``"token"``. See
            :class:`~torchrl.modules.vla.VLAWrapperBase`. Defaults to
            ``"sequence"``.
        output_mode (str, optional): ``"tokens"`` keeps the default token-head
            output, ``"chunk"`` decodes to continuous action chunks, and
            ``"both"`` writes both representations. Defaults to ``"tokens"``.
        use_wrist_image (bool, optional): read
            ``("observation", "wrist_image")`` as the second camera (the
            checkpoint must have been trained with two input images).
            Defaults to ``False``.
        center_crop (bool, optional): center-crop the images to 90% area
            before the processor resize, as in SimpleVLA-RL evaluation with
            augmented training. Defaults to ``False``.
        image_backend (str, optional): backend passed to
            :class:`~torchrl.data.vla.OpenVLAImagePreprocessor` for the fast
            batched image path. ``"torch_reference"`` follows the LIBERO
            reference operation order and interpolation semantics without a
            TensorFlow dependency; ``"tensorflow"`` runs the reference
            implementation, ``"torchvision"`` selects the faster bicubic path,
            and ``"pil"`` is a lightweight debugging path. Defaults to
            ``"torch_reference"``.
        gripper_binarize (bool, optional): binarize the decoded gripper action
            to +/-1. The model emits a continuous gripper value but robots
            (LIBERO/robosuite) need a firm open/close, so without this the
            gripper half-actuates and never secures a grasp. Defaults to
            ``False``.
        gripper_binarize_threshold (float, optional): threshold used when
            ``gripper_binarize=True`` after the SimpleVLA ``2 * g - 1``
            remap. A zero threshold here is therefore equivalent to a ``0.5``
            threshold on the raw decoded gripper scalar. Defaults to ``0.0``.
        gripper_invert (bool, optional): flip the gripper open/close sign after
            (optional) binarization, for checkpoints whose convention is
            opposite the environment's. Defaults to ``False``.
    """

    def __init__(
        self,
        model,
        processor,
        *,
        unnorm_key: str | None = None,
        temperature: float = 1.0,
        top_k: int | None = None,
        micro_batch_size: int | None = None,
        default_interaction_type: InteractionType = InteractionType.DETERMINISTIC,
        log_probs_mode: LogProbsMode = "sequence",
        output_mode: Literal["chunk", "tokens", "both"] | None = None,
        use_wrist_image: bool = False,
        center_crop: bool = False,
        image_backend: Literal[
            "torchvision", "torch_reference", "pil", "tensorflow"
        ] = "torch_reference",
        gripper_binarize: bool = False,
        gripper_binarize_threshold: float = 0.0,
        gripper_invert: bool = False,
        return_vla_action_container: bool = True,
    ) -> None:
        from openvla_oft.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        if top_k is not None and int(top_k) < 1:
            raise ValueError(f"top_k must be >= 1 when provided, got {top_k}.")
        num_bins = len(model.bin_centers) + 1
        if unnorm_key is None and model.norm_stats is not None:
            if len(model.norm_stats) != 1:
                raise ValueError(
                    "the checkpoint carries statistics for several datasets; "
                    f"pass unnorm_key explicitly (options: {sorted(model.norm_stats)})."
                )
            unnorm_key = next(iter(model.norm_stats))
        action_tokenizer = None
        if model.norm_stats is not None and unnorm_key is not None:
            action_tokenizer = VocabTailActionTokenizer.from_norm_stats(
                model.norm_stats,
                unnorm_key,
                num_bins=num_bins,
            )
        super().__init__(
            action_dim=ACTION_DIM,
            chunk_size=NUM_ACTIONS_CHUNK,
            action_head="tokens",
            vocab_size=num_bins,
            action_tokenizer=action_tokenizer,
            use_state=False,
            default_interaction_type=default_interaction_type,
            log_probs_mode=log_probs_mode,
            output_mode=output_mode,
            return_vla_action_container=return_vla_action_container,
        )
        self.model = model
        self.processor = processor
        self.temperature = float(temperature)
        self.top_k = int(top_k) if top_k is not None else None
        self.micro_batch_size = (
            int(micro_batch_size) if micro_batch_size is not None else None
        )
        self.use_wrist_image = bool(use_wrist_image)
        self.center_crop = bool(center_crop)
        self.image_backend = image_backend
        self.gripper_binarize = bool(gripper_binarize)
        self.gripper_binarize_threshold = float(gripper_binarize_threshold)
        self.gripper_invert = bool(gripper_invert)
        self.num_bins = num_bins
        self.unnorm_key = unnorm_key
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.input_transform = OpenVLAInputTransform(
            processor,
            use_wrist_image=use_wrist_image,
            center_crop=center_crop,
            image_backend=image_backend,
            image_key=self.tensor_keys.image,
            wrist_image_key=self.tensor_keys.wrist_image,
            instruction_key=self.tensor_keys.instruction,
            device=device,
            dtype=dtype,
        )
        self.model_transform = OpenVLAModelTransform(
            model,
            processor,
            chunk_size=NUM_ACTIONS_CHUNK,
            action_dim=ACTION_DIM,
            num_bins=num_bins,
            temperature=temperature,
            top_k=top_k,
            micro_batch_size=micro_batch_size,
            image_key=self.tensor_keys.image,
            logits_key=self.tensor_keys.action_logits,
        )
        self.policy_stack = TensorDictSequential(
            self.input_transform,
            self.model_transform,
        )
        self.gripper_postprocess = GripperPostProcessTransform(
            action_key=self.tensor_keys.action_chunk,
            rescale=True,
            binarize=gripper_binarize,
            threshold=gripper_binarize_threshold,
            invert=gripper_invert,
        )
        self.decode_stack = None
        if self.action_tokenizer is not None:
            self.decode_stack = TensorDictSequential(
                ActionTokenizerTransform(
                    self.action_tokenizer,
                    in_key=self.tensor_keys.action_chunk,
                    out_key=self.tensor_keys.action_tokens,
                    mode="decode",
                ),
                self.gripper_postprocess,
            )
        self._prompt_cache = self.input_transform._prompt_cache
        self._image_preprocessor = self.input_transform._image_preprocessor
        self.image_backend = self.input_transform.image_backend
        if self.use_wrist_image:
            self.in_keys = [*self.in_keys, ("observation", "wrist_image")]

    # -- loading ----------------------------------------------------------
    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        dataset_statistics: str | None = None,
        **kwargs,
    ) -> OpenVLAOFTWrapper:
        """Load a SimpleVLA-RL token-OFT checkpoint (e.g. ``Haozhan72/...``).

        Args:
            dataset_statistics (str, optional): a path to an OpenVLA
                ``dataset_statistics.json`` or a HF repo id shipping one, whose
                action-normalization stats are merged into ``model.norm_stats``.
                The SimpleVLA-RL LIBERO checkpoints omit their fine-tuning
                dataset's stats (only the base pretraining datasets remain), so
                point this at the matching official OFT release (e.g.
                ``moojink/openvla-7b-oft-finetuned-libero-spatial``) -- it is the
                same LIBERO data, hence the same normalization.
        """
        from openvla_oft.modeling_prismatic import OpenVLAForActionPrediction
        from openvla_oft.processing_prismatic import PrismaticProcessor

        from transformers import AutoConfig

        register_openvla_oft()
        # The SimpleVLA-RL token-OFT checkpoints ship a base-Prismatic
        # config.json that omits the OFT architecture flags the modeling code
        # reads. ``use_proprio`` is the only one read unconditionally; this
        # single-image token variant has no proprio projector, so default it
        # to False (which skips the projector and the dependent proprio_dim).
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=False
        )
        if not hasattr(config, "use_proprio"):
            config.use_proprio = False
        model = OpenVLAForActionPrediction.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            # the vendored model implements only eager attention
            attn_implementation="eager",
        )
        if dataset_statistics is not None:
            extra = _load_dataset_statistics(dataset_statistics)
            if getattr(model, "norm_stats", None) is None:
                model.norm_stats = {}
            model.norm_stats.update(extra)
        if device is not None:
            model = model.to(device)
        model.eval()
        processor = PrismaticProcessor.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=False
        )
        return cls(model, processor, **kwargs)

    def make_action_tokenizer(self) -> VocabTailActionTokenizer:
        """Return the matching window-id tokenizer (tokens -> env actions)."""
        if self.action_tokenizer is None:
            raise RuntimeError(
                "OpenVLA-OFT action statistics are missing; cannot build an "
                "action tokenizer."
            )
        return self.action_tokenizer

    # -- preprocessing (shared by rollout and loss-time recompute) ---------
    def _to_pil(self, image: torch.Tensor):
        return self.input_transform._to_pil(image)

    def _instructions(self, tensordict: TensorDictBase, batch: int) -> list[str]:
        return self.input_transform._instructions(tensordict, batch)

    def _prompt_input_ids(self, instruction: str) -> torch.Tensor | None:
        return self.input_transform._prompt_input_ids(instruction)

    @staticmethod
    def _image_processor_config(
        image_processor,
    ) -> tuple[int, torch.Tensor, torch.Tensor] | None:
        input_sizes = getattr(image_processor, "input_sizes", None)
        normalize_params = getattr(image_processor, "tvf_normalize_params", None)
        if (
            input_sizes is None
            or normalize_params is None
            or len(input_sizes) != 1
            or len(normalize_params) != 1
            or tuple(input_sizes[0][-2:]) != (_OPENVLA_IMAGE_SIZE, _OPENVLA_IMAGE_SIZE)
        ):
            return None
        return (
            int(input_sizes[0][-1]),
            torch.as_tensor(normalize_params[0]["mean"], dtype=torch.float32),
            torch.as_tensor(normalize_params[0]["std"], dtype=torch.float32),
        )

    def _make_image_preprocessor(
        self,
        processor,
        image_backend: Literal["torchvision", "torch_reference", "pil", "tensorflow"],
    ) -> OpenVLAImagePreprocessor | None:
        return self.input_transform._make_image_preprocessor(processor, image_backend)

    def _pixel_values_from_images(self, images: torch.Tensor) -> torch.Tensor | None:
        return self.input_transform._pixel_values_from_images(images)

    def _preprocess(self, images, wrist_images, instructions):
        return self.input_transform._preprocess(images, wrist_images, instructions)

    def _preprocess_slow(self, images, wrist_images, instructions):
        return self.input_transform._preprocess_slow(images, wrist_images, instructions)

    # -- the parallel-decoding forward -------------------------------------
    def _window_logits(self, input_ids, attention_mask, pixel_values):
        return self.model_transform._window_logits(
            input_ids, attention_mask, pixel_values
        )

    def _action_logits(self, tensordict: TensorDictBase) -> torch.Tensor:
        out = self.policy_stack(tensordict.clone(False))
        return out.get(self.tensor_keys.action_logits)

    def forward(
        self,
        tensordict: TensorDictBase,
        *,
        tensordict_out: TensorDictBase | None = None,
        logits_only: bool = False,
        **kwargs,
    ) -> TensorDictBase:
        out = super().forward(
            tensordict,
            tensordict_out=tensordict_out,
            logits_only=logits_only,
            **kwargs,
        )
        if self.action_head != "tokens":
            return out
        chunk = out.get(self.tensor_keys.action_chunk, None)
        if chunk is None:
            return out
        chunk = self.gripper_postprocess.postprocess(chunk)
        out.set(self.tensor_keys.action_chunk, chunk)
        action = out.get(self.tensor_keys.vla_action, None)
        if hasattr(action, "chunk"):
            action.chunk = chunk
        return out


class OpenVLAOFTL1Wrapper(OpenVLAOFTWrapper):
    """Continuous L1-head OpenVLA-OFT policy for reference/evaluation rollouts.

    This wrapper targets the official OpenVLA-OFT LIBERO checkpoints that use
    two camera images and an 8-D proprio vector. It reads the canonical TorchRL
    LIBERO observation, converts the default 9-D quaternion state to the
    OpenVLA 8-D axis-angle state when needed, normalizes proprio using the
    checkpoint statistics, and writes a continuous ``("vla_action", "chunk")``.
    """

    def __init__(
        self,
        model,
        processor,
        action_head: nn.Module,
        proprio_projector: nn.Module | None,
        *,
        unnorm_key: str | None = None,
        default_interaction_type: InteractionType = InteractionType.DETERMINISTIC,
        use_proprio: bool = True,
        use_wrist_image: bool = True,
        center_crop: bool = True,
        image_backend: Literal[
            "torchvision", "torch_reference", "pil", "tensorflow"
        ] = "torch_reference",
        gripper_binarize: bool = True,
        gripper_binarize_threshold: float = 0.0,
        gripper_invert: bool = True,
        return_vla_action_container: bool = True,
    ) -> None:
        from openvla_oft.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

        if unnorm_key is None and model.norm_stats is not None:
            if len(model.norm_stats) != 1:
                raise ValueError(
                    "the checkpoint carries statistics for several datasets; "
                    f"pass unnorm_key explicitly (options: {sorted(model.norm_stats)})."
                )
            unnorm_key = next(iter(model.norm_stats))
        if model.norm_stats is not None and unnorm_key not in model.norm_stats:
            no_noops_key = f"{unnorm_key}_no_noops"
            if no_noops_key in model.norm_stats:
                unnorm_key = no_noops_key
        VLAWrapperBase.__init__(
            self,
            action_dim=ACTION_DIM,
            chunk_size=NUM_ACTIONS_CHUNK,
            action_head="continuous",
            use_state=use_proprio,
            default_interaction_type=default_interaction_type,
            return_vla_action_container=return_vla_action_container,
        )
        self.model = model
        self.processor = processor
        self.action_head_module = action_head
        self.proprio_projector = proprio_projector
        self.unnorm_key = unnorm_key
        self.use_proprio = bool(use_proprio)
        self.use_wrist_image = bool(use_wrist_image)
        self.center_crop = bool(center_crop)
        self.image_backend = image_backend
        self.gripper_binarize = bool(gripper_binarize)
        self.gripper_binarize_threshold = float(gripper_binarize_threshold)
        self.gripper_invert = bool(gripper_invert)
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        self.input_transform = OpenVLAInputTransform(
            processor,
            use_wrist_image=use_wrist_image,
            center_crop=center_crop,
            image_backend=image_backend,
            image_key=self.tensor_keys.image,
            wrist_image_key=self.tensor_keys.wrist_image,
            instruction_key=self.tensor_keys.instruction,
            device=device,
            dtype=dtype,
        )
        self.gripper_postprocess = GripperPostProcessTransform(
            action_key=self.tensor_keys.action_chunk,
            rescale=True,
            binarize=gripper_binarize,
            threshold=gripper_binarize_threshold,
            invert=gripper_invert,
        )
        self._prompt_cache = self.input_transform._prompt_cache
        self._image_preprocessor = self.input_transform._image_preprocessor
        self.image_backend = self.input_transform.image_backend
        if self.use_wrist_image:
            self.in_keys = [*self.in_keys, ("observation", "wrist_image")]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_model_name_or_path: str,
        *,
        torch_dtype: torch.dtype = torch.bfloat16,
        device: torch.device | str | None = None,
        dataset_statistics: str | None = None,
        action_head_file: str = "action_head--150000_checkpoint.pt",
        proprio_projector_file: str = "proprio_projector--150000_checkpoint.pt",
        use_proprio: bool = True,
        num_images_in_input: int = 2,
        **kwargs,
    ) -> OpenVLAOFTL1Wrapper:
        """Load an official continuous-head OpenVLA-OFT checkpoint."""
        from openvla_oft.constants import ACTION_DIM
        from openvla_oft.modeling_prismatic import (
            OpenVLAForActionPrediction,
            ProprioProjector,
        )
        from openvla_oft.processing_prismatic import PrismaticProcessor
        from openvla_oft.train_utils import load_component_state_dict
        from transformers import AutoConfig

        register_openvla_oft()
        config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=False
        )
        config.use_proprio = False
        config.proprio_dim = 8
        model = OpenVLAForActionPrediction.from_pretrained(
            pretrained_model_name_or_path,
            config=config,
            torch_dtype=torch_dtype,
            low_cpu_mem_usage=True,
            trust_remote_code=False,
            attn_implementation="eager",
        )
        model.vision_backbone.set_num_images_in_input(int(num_images_in_input))
        if dataset_statistics is not None:
            model.norm_stats = _load_dataset_statistics(dataset_statistics)
        if device is not None:
            model = model.to(device)
        model.eval()
        processor = PrismaticProcessor.from_pretrained(
            pretrained_model_name_or_path, trust_remote_code=False
        )
        device = next(model.parameters()).device
        dtype = next(model.parameters()).dtype
        action_head = _L1RegressionActionHead(
            input_dim=model.llm_dim,
            hidden_dim=model.llm_dim,
            action_dim=ACTION_DIM,
        ).to(device=device, dtype=dtype)
        action_head.load_state_dict(
            load_component_state_dict(
                _resolve_component_checkpoint(
                    pretrained_model_name_or_path,
                    action_head_file,
                    "action_head",
                )
            )
        )
        action_head.eval()
        proprio_projector = None
        if use_proprio:
            proprio_projector = ProprioProjector(
                llm_dim=model.llm_dim, proprio_dim=8
            ).to(device=device, dtype=dtype)
            proprio_projector.load_state_dict(
                load_component_state_dict(
                    _resolve_component_checkpoint(
                        pretrained_model_name_or_path,
                        proprio_projector_file,
                        "proprio_projector",
                    )
                )
            )
            proprio_projector.eval()
        return cls(
            model,
            processor,
            action_head,
            proprio_projector,
            use_proprio=use_proprio,
            **kwargs,
        )

    def _openvla_state(self, state: torch.Tensor) -> torch.Tensor:
        if state.shape[-1] == 8:
            return state
        if state.shape[-1] != 9:
            raise ValueError(
                "OpenVLA L1 proprio expects an 8-D axis-angle state or the "
                f"default 9-D LIBERO quaternion state, got shape {state.shape}."
            )
        position = state[..., :3]
        quat = state[..., 3:7]
        gripper = state[..., 7:]
        xyz = quat[..., :3]
        w = quat[..., 3].clamp(-1.0, 1.0)
        den = (1.0 - w.square()).clamp_min(0.0).sqrt()
        angle = 2.0 * torch.acos(w)
        axis_angle = torch.where(
            den.unsqueeze(-1) > 1e-6,
            xyz * angle.unsqueeze(-1) / den.clamp_min(1e-6).unsqueeze(-1),
            torch.zeros_like(xyz),
        )
        return torch.cat((position, axis_angle, gripper), dim=-1)

    def _normalize_proprio(self, state: torch.Tensor) -> np.ndarray:
        if not self.use_proprio:
            raise RuntimeError("OpenVLA L1 proprio normalization requires state input.")
        stats = self.model.norm_stats[self.unnorm_key]["proprio"]
        if "q01" in stats:
            low = torch.as_tensor(stats["q01"], dtype=state.dtype, device=state.device)
            high = torch.as_tensor(stats["q99"], dtype=state.dtype, device=state.device)
        else:
            low = torch.as_tensor(stats["min"], dtype=state.dtype, device=state.device)
            high = torch.as_tensor(stats["max"], dtype=state.dtype, device=state.device)
        mask = torch.as_tensor(
            stats.get("mask", np.ones(low.shape, dtype=bool)),
            dtype=torch.bool,
            device=state.device,
        )
        normalized = 2.0 * (state - low) / (high - low).clamp_min(1e-8) - 1.0
        normalized = torch.where(mask, normalized, state)
        return normalized.clamp(-1.0, 1.0).detach().cpu().numpy().astype(np.float32)

    def _sort_padding_for_predict_action(self, input_ids, attention_mask):
        pad_token_id = self.processor.tokenizer.pad_token_id
        padding_mask = (~input_ids.ne(pad_token_id)).int()
        sorted_indices = torch.argsort(
            padding_mask, dim=1, descending=True, stable=True
        )
        return (
            torch.gather(input_ids, 1, sorted_indices),
            torch.gather(attention_mask, 1, sorted_indices),
        )

    def _postprocess_actions(self, actions: torch.Tensor) -> torch.Tensor:
        return self.gripper_postprocess.postprocess(actions)

    def _predict_chunk(self, tensordict: TensorDictBase) -> torch.Tensor:
        images = tensordict.get(self.tensor_keys.image)
        batch_dims = images.shape[:-3]
        images = images.reshape(-1, *images.shape[-3:])
        wrist_images = None
        if self.use_wrist_image:
            wrist_images = tensordict.get(("observation", "wrist_image"))
            wrist_images = wrist_images.reshape(-1, *wrist_images.shape[-3:])
        instructions = self._instructions(tensordict, images.shape[0])
        input_ids, attention_mask, pixel_values = self._preprocess(
            images, wrist_images, instructions
        )
        input_ids, attention_mask = self._sort_padding_for_predict_action(
            input_ids, attention_mask
        )
        proprio = None
        if self.use_proprio:
            state = self._get_state(tensordict).reshape(images.shape[0], -1).float()
            proprio = self._normalize_proprio(self._openvla_state(state))
        chunks = []
        for index in range(images.shape[0]):
            row_proprio = None if proprio is None else proprio[index : index + 1]
            with torch.no_grad():
                actions, _ = self.model.predict_action(
                    input_ids=input_ids[index : index + 1],
                    pixel_values=pixel_values[index : index + 1],
                    attention_mask=attention_mask[index : index + 1],
                    unnorm_key=self.unnorm_key,
                    proprio=row_proprio,
                    proprio_projector=self.proprio_projector,
                    action_head=self.action_head_module,
                    noisy_action_projector=None,
                    use_film=False,
                )
            actions = np.asarray(actions, dtype=np.float32)
            if actions.ndim == 3 and actions.shape[0] == 1:
                actions = actions[0]
            chunks.append(torch.as_tensor(actions, device=images.device))
        chunk = torch.stack(chunks, dim=0).reshape(
            *batch_dims, self.chunk_size, self.action_dim
        )
        return self._postprocess_actions(chunk)

    def _predict(self, tensordict: TensorDictBase) -> torch.Tensor:
        return self._predict_chunk(tensordict).flatten(-2)


def _resolve_component_checkpoint(
    pretrained_model_name_or_path: str, filename_or_path: str, pattern: str
) -> str:
    if os.path.isfile(filename_or_path):
        return filename_or_path
    if os.path.isdir(pretrained_model_name_or_path):
        exact = os.path.join(pretrained_model_name_or_path, filename_or_path)
        if os.path.isfile(exact):
            return exact
        matches = [
            os.path.join(pretrained_model_name_or_path, filename)
            for filename in os.listdir(pretrained_model_name_or_path)
            if pattern in filename and "checkpoint" in filename
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one {pattern!r} checkpoint in "
                f"{pretrained_model_name_or_path!r}, found {len(matches)}."
            )
        return matches[0]
    return _hf_hub_download_fn()(pretrained_model_name_or_path, filename_or_path)
