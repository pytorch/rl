# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""OpenVLA-OFT (token variant) policy wrapper for the VLA GRPO recipe.

Wraps the vendored SimpleVLA-RL token-OFT model (see ``openvla_oft/``) as a
:class:`~torchrl.modules.vla.VLAWrapperBase`: one forward emits the whole
action chunk (parallel decoding -- ``chunk_size * action_dim`` tokens in a
single language-model pass, no autoregressive loop), sampled from the 256-way
categorical over the action-token window at the tail of the LLaMA-2
vocabulary.

The wrapper owns ALL model-side preprocessing (prompt construction, image
transforms) and the temperature scaling, and both rollout sampling and
loss-time recomputation go through the same :meth:`get_dist`, so with
identical weights the PPO importance ratio is exactly 1 -- the temperature
contract a T != 1 rollout requires.
"""
from __future__ import annotations

import importlib.util
import io

import numpy as np
import torch
from tensordict import TensorDictBase
from tensordict.nn import InteractionType
from torch.nn.utils.rnn import pad_sequence

from torchrl.data.vla import VocabTailActionTokenizer
from torchrl.modules.vla import VLAWrapperBase

from torchrl.modules.vla.common import LogProbsMode

_has_transformers = importlib.util.find_spec("transformers") is not None
_has_timm = importlib.util.find_spec("timm") is not None
_has_pil = importlib.util.find_spec("PIL") is not None

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


def _load_dataset_statistics(spec: str) -> dict:
    """Load an OpenVLA ``dataset_statistics.json`` from a local path or HF repo.

    ``spec`` is either a filesystem path to the json or a HuggingFace repo id
    that ships a ``dataset_statistics.json`` at its root.
    """
    import json
    import os

    if os.path.isfile(spec):
        path = spec
    else:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(spec, "dataset_statistics.json")
    with open(path) as f:
        return json.load(f)


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
    ``action_tokens`` (``[*B, chunk_size, action_dim]`` ids in the 256-way
    action-token *window*) and their ``log_probs``. Decode the tokens to
    environment actions with :meth:`action_tokenizer` (e.g. through
    :class:`~torchrl.envs.transforms.ActionTokenizerTransform`).

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
        default_interaction_type (InteractionType, optional): token readout
            when no exploration context is active (``RANDOM`` samples, else
            argmax); the forward otherwise follows the ambient
            :func:`~torchrl.envs.utils.exploration_type`. Defaults to
            ``InteractionType.DETERMINISTIC``. See
            :class:`~torchrl.modules.vla.VLAWrapperBase`.
        log_probs_mode (str, optional): ``"sequence"`` or ``"token"``. See
            :class:`~torchrl.modules.vla.VLAWrapperBase`. Defaults to
            ``"sequence"``.
        use_wrist_image (bool, optional): read
            ``("observation", "wrist_image")`` as the second camera (the
            checkpoint must have been trained with two input images).
            Defaults to ``False``.
        center_crop (bool, optional): center-crop the images to 90% area
            before the processor resize, as in SimpleVLA-RL evaluation with
            augmented training. Defaults to ``False``.
        gripper_binarize (bool, optional): binarize the decoded gripper action
            to +/-1 by sign. The model emits a continuous gripper value but
            robots (LIBERO/robosuite) need a firm open/close, so without this
            the gripper half-actuates and never secures a grasp. Defaults to
            ``False``.
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
        default_interaction_type: InteractionType = InteractionType.DETERMINISTIC,
        log_probs_mode: LogProbsMode = "sequence",
        use_wrist_image: bool = False,
        center_crop: bool = False,
        gripper_binarize: bool = False,
        gripper_invert: bool = False,
    ) -> None:
        from openvla_oft.constants import ACTION_DIM, NUM_ACTIONS_CHUNK

        if temperature <= 0:
            raise ValueError(f"temperature must be > 0, got {temperature}.")
        num_bins = len(model.bin_centers) + 1
        super().__init__(
            action_dim=ACTION_DIM,
            chunk_size=NUM_ACTIONS_CHUNK,
            action_head="tokens",
            vocab_size=num_bins,
            use_state=False,
            default_interaction_type=default_interaction_type,
            log_probs_mode=log_probs_mode,
        )
        self.model = model
        self.processor = processor
        self.temperature = float(temperature)
        self.use_wrist_image = bool(use_wrist_image)
        self.center_crop = bool(center_crop)
        self.gripper_binarize = bool(gripper_binarize)
        self.gripper_invert = bool(gripper_invert)
        self.num_bins = num_bins
        if unnorm_key is None and model.norm_stats is not None:
            if len(model.norm_stats) != 1:
                raise ValueError(
                    "the checkpoint carries statistics for several datasets; "
                    f"pass unnorm_key explicitly (options: {sorted(model.norm_stats)})."
                )
            unnorm_key = next(iter(model.norm_stats))
        self.unnorm_key = unnorm_key
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

    def action_tokenizer(self) -> VocabTailActionTokenizer:
        """The matching window-id tokenizer (decode tokens -> env actions)."""
        return VocabTailActionTokenizer.from_norm_stats(
            self.model.norm_stats,
            self.unnorm_key,
            num_bins=self.num_bins,
            gripper_binarize=self.gripper_binarize,
            gripper_invert=self.gripper_invert,
        )

    # -- preprocessing (shared by rollout and loss-time recompute) ---------
    def _to_pil(self, image: torch.Tensor):
        from PIL import Image

        array = image.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
        pil = Image.fromarray(array).convert("RGB")
        # Match OpenVLA-OFT's eval chain: resize to 224 (lanczos), JPEG
        # round-trip to mirror the RLDS training compression, THEN the
        # area-0.9 center crop -- in that order.
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
            # LANCZOS to mirror the reference tf.image.resize(method="lanczos3")
            pil = pil.crop((left, top, left + crop_w, top + crop_h)).resize(
                (width, height), Image.LANCZOS
            )
        return pil

    def _instructions(self, tensordict: TensorDictBase, batch: int) -> list[str]:
        instruction = tensordict.get(self.tensor_keys.instruction)
        data = getattr(instruction, "tolist", lambda: instruction)()
        if isinstance(data, str):
            data = [data] * batch
        return [str(item) for item in data]

    def _preprocess(self, images, wrist_images, instructions):
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
        device = next(self.model.parameters()).device
        dtype = next(self.model.parameters()).dtype
        return (
            input_ids.to(device),
            attention_mask.to(device),
            pixel_values.to(device=device, dtype=dtype),
        )

    # -- the parallel-decoding forward -------------------------------------
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
        # _prepare_input_for_action_prediction appends the action placeholders
        # and stop token AFTER the padding of shorter rows: sort the padding
        # to the end of each row (input ids, attention mask AND labels with
        # the same permutation, as in the vendored generate_action_verl) so
        # that the action block sits at num_prompt_tokens for every row.
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
        # restrict to the action-token window at the tail of the (true,
        # unpadded) vocabulary: the policy is a num_bins-way categorical
        window = logits[..., model.vocab_size - self.num_bins : model.vocab_size]
        return window

    def _action_logits(self, tensordict: TensorDictBase) -> torch.Tensor:
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
        window = self._window_logits(input_ids, attention_mask, pixel_values)
        window = window.float() / self.temperature
        return window.reshape(
            *batch_dims, self.chunk_size, self.action_dim, self.num_bins
        )
