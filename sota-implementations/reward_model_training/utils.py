# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Helper functions for the RLHF reward-model training recipe.

The recipe is model-agnostic: any Hugging Face ``AutoModelForSequenceClassification``
with ``num_labels=1`` can be used as the reward-model backbone. A small from-config
model and a synthetic preference dataset are used when ``model.name`` /
``data.dataset_name`` are left empty, which keeps the CI smoke test hermetic (no
download, no ``datasets`` dependency).
"""
from __future__ import annotations

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl.data import (
    LazyTensorStorage,
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
)
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer


class _RewardModel(nn.Module):
    """Maps ``(input_ids, attention_mask)`` to a single scalar score per sequence."""

    def __init__(self, hf_model: nn.Module) -> None:
        super().__init__()
        self.model = hf_model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits  # shape [B, num_labels=1]


def make_reward_model(cfg, device: torch.device) -> TensorDictModule:
    """Build the score network: an HF sequence-classification model with a 1-d head.

    When ``cfg.model.name`` is empty, a tiny GPT2-style model is built from scratch
    (random weights, no download) so the recipe can run hermetically in CI.
    """
    name = cfg.model.name
    if name:
        hf_model = AutoModelForSequenceClassification.from_pretrained(
            name, num_labels=1
        )
    else:
        config = AutoConfig.for_model(
            "gpt2",
            num_labels=1,
            n_layer=2,
            n_head=2,
            n_embd=64,
            vocab_size=256,
            n_positions=max(int(cfg.data.max_length), 32),
            # keep the special token ids within the tiny vocab
            bos_token_id=0,
            eos_token_id=0,
        )
        hf_model = AutoModelForSequenceClassification.from_config(config)

    # Sequence-classification models need a pad token to locate the final token.
    if hf_model.config.pad_token_id is None:
        hf_model.config.pad_token_id = (
            hf_model.config.eos_token_id
            if hf_model.config.eos_token_id is not None
            else 0
        )

    score_network = TensorDictModule(
        _RewardModel(hf_model),
        in_keys=["input_ids", "attention_mask"],
        out_keys=["score"],
    )
    return score_network.to(device)


def make_tokenizer(cfg):
    """Return the tokenizer matching the model, or ``None`` in synthetic mode."""
    name = cfg.model.name
    if not name:
        return None
    tokenizer = AutoTokenizer.from_pretrained(name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def get_vocab_size(tokenizer, score_network: TensorDictModule) -> int:
    if tokenizer is not None:
        return len(tokenizer)
    return score_network.module.model.config.vocab_size


def _pairwise_td(
    chosen_ids: torch.Tensor,
    rejected_ids: torch.Tensor,
    chosen_mask: torch.Tensor | None = None,
    rejected_mask: torch.Tensor | None = None,
) -> TensorDict:
    """Pack tokenized chosen/rejected responses into the loss input layout."""
    n = chosen_ids.shape[0]
    if chosen_mask is None:
        chosen_mask = torch.ones_like(chosen_ids)
    if rejected_mask is None:
        rejected_mask = torch.ones_like(rejected_ids)
    return TensorDict(
        {
            "chosen": TensorDict(
                {"input_ids": chosen_ids, "attention_mask": chosen_mask},
                batch_size=[n],
            ),
            "rejected": TensorDict(
                {"input_ids": rejected_ids, "attention_mask": rejected_mask},
                batch_size=[n],
            ),
        },
        batch_size=[n],
    )


def make_dataset(cfg, tokenizer, split: str, vocab_size: int) -> TensorDict:
    """Build a pairwise preference dataset as a single batched ``TensorDict``.

    The returned tensordict has ``"chosen"`` and ``"rejected"`` sub-tensordicts, each
    carrying ``input_ids`` / ``attention_mask`` -- exactly the keys
    :class:`~torchrl.objectives.llm.RewardModelLoss` expects by default.
    """
    dataset_name = cfg.data.dataset_name
    max_length = int(cfg.data.max_length)

    if not dataset_name:
        # Hermetic synthetic dataset (no download, no ``datasets`` dependency).
        n = int(cfg.data.synthetic_size)
        gen = torch.Generator().manual_seed(int(cfg.seed) + (split == "train"))
        chosen_ids = torch.randint(0, vocab_size, (n, max_length), generator=gen)
        rejected_ids = torch.randint(0, vocab_size, (n, max_length), generator=gen)
        return _pairwise_td(chosen_ids, rejected_ids)

    # Real preference data. ``datasets`` is imported lazily so the synthetic/CI path
    # never requires it.
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    max_samples = cfg.data.max_samples
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    chosen_texts, rejected_texts = [], []
    for sample in ds:
        prompt = sample.get("prompt", "")
        sep = "\n" if prompt else ""
        chosen_texts.append(prompt + sep + sample["chosen"])
        rejected_texts.append(prompt + sep + sample["rejected"])

    tok_kwargs = {
        "max_length": max_length,
        "padding": "max_length",
        "truncation": True,
        "return_tensors": "pt",
    }
    chosen_tok = tokenizer(chosen_texts, **tok_kwargs)
    rejected_tok = tokenizer(rejected_texts, **tok_kwargs)
    return _pairwise_td(
        chosen_tok["input_ids"],
        rejected_tok["input_ids"],
        chosen_tok["attention_mask"],
        rejected_tok["attention_mask"],
    )


def make_replay_buffer(
    data: TensorDict, batch_size: int, device: torch.device
) -> TensorDictReplayBuffer:
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(data.shape[0], device=device),
        sampler=SamplerWithoutReplacement(drop_last=True),
        batch_size=batch_size,
    )
    rb.extend(data)
    return rb


def make_optimizer(cfg, score_network: TensorDictModule) -> torch.optim.Optimizer:
    _maybe_freeze_backbone(score_network, cfg.optim.freeze_frac)
    params = [p for p in score_network.parameters() if p.requires_grad]
    return torch.optim.AdamW(
        params, lr=cfg.optim.lr, weight_decay=cfg.optim.weight_decay
    )


def _maybe_freeze_backbone(score_network: TensorDictModule, freeze_frac: float) -> None:
    """Best-effort freezing of the first ``freeze_frac`` of transformer layers.

    Freezing the lower layers of the backbone is a common efficiency trick for
    reward-model fine-tuning. This is best-effort: if the backbone layer list cannot
    be located for the given architecture, no layer is frozen.
    """
    if not freeze_frac or freeze_frac <= 0:
        return
    base = score_network.module.model
    # The base transformer is exposed via ``base_model`` on HF models.
    transformer = getattr(base, "base_model", base)
    layers = None
    for attr in ("h", "layers", "layer", "block"):
        candidate = getattr(transformer, attr, None)
        if candidate is not None and len(candidate) > 0:
            layers = candidate
            break
    if layers is None:
        return
    num_freeze = int(freeze_frac * len(layers))
    for layer in layers[:num_freeze]:
        layer.requires_grad_(False)


def log_metrics(logger, metrics: dict, step: int) -> None:
    if logger is None:
        return
    for key, value in metrics.items():
        logger.log_scalar(key, value, step)
