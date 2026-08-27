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

import importlib.util

import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn
from torchrl._utils import logger as torchrl_logger
from torchrl.data import (
    SamplerWithoutReplacement,
    TensorDictReplayBuffer,
    TensorStorage,
)
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    PreTrainedTokenizerBase,
)

_has_datasets = importlib.util.find_spec("datasets") is not None

_TOKENIZE_CHUNK_SIZE = 1024


class _RewardModel(nn.Module):
    """Maps ``(input_ids, attention_mask)`` to a single scalar score per sequence."""

    def __init__(self, hf_model: nn.Module):
        super().__init__()
        self.model = hf_model

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        out = self.model(input_ids=input_ids, attention_mask=attention_mask)
        return out.logits  # shape [B, num_labels=1]


def make_reward_model(
    cfg, device: torch.device, tokenizer: PreTrainedTokenizerBase | None = None
) -> TensorDictModule:
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

    # Reward-model training deliberately runs with dropout disabled (eval mode):
    # chosen and rejected would otherwise be scored under independent dropout
    # masks, adding pure noise to the score difference the Bradley-Terry loss is
    # built on, and the exported scorer must behave at training time exactly as it
    # will at inference. Eval mode covers both nn.Dropout modules and the
    # functional dropout paths that read self.training. from_pretrained already
    # returns eval mode; this makes the from_config (synthetic) path consistent.
    hf_model.eval()

    # Sequence-classification models locate the final non-pad token by comparing
    # input_ids against config.pad_token_id (the attention mask is not used for
    # pooling), so the config value must match the id the tokenizer actually pads
    # with -- otherwise scores are silently read from the wrong position.
    pad_token_id = tokenizer.pad_token_id if tokenizer is not None else None
    if pad_token_id is None:
        pad_token_id = hf_model.config.pad_token_id
    if pad_token_id is None:
        eos_token_id = hf_model.config.eos_token_id
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0] if eos_token_id else None
        pad_token_id = eos_token_id if eos_token_id is not None else 0
    hf_model.config.pad_token_id = pad_token_id

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

    Real datasets are tokenized in bounded chunks so the transient peak (raw text
    plus tokenizer buffers) stays independent of dataset size; only the final
    token tensors scale with the number of pairs.
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
    if not _has_datasets:
        raise ImportError(
            "Loading a Hugging Face preference dataset requires the optional "
            "`datasets` dependency. Install the recipe requirements with "
            "`pip install -r requirements.txt`."
        )
    from datasets import load_dataset

    ds = load_dataset(dataset_name, split=split)
    max_samples = cfg.data.max_samples
    if max_samples is not None:
        ds = ds.select(range(min(int(max_samples), len(ds))))

    tok_kwargs = {
        "max_length": max_length,
        "padding": "max_length",
        "truncation": True,
        "return_tensors": "pt",
    }
    chosen_ids, chosen_masks, rejected_ids, rejected_masks = [], [], [], []
    num_dropped = 0
    for start in range(0, len(ds), _TOKENIZE_CHUNK_SIZE):
        chunk = ds.select(range(start, min(start + _TOKENIZE_CHUNK_SIZE, len(ds))))
        chosen_texts, rejected_texts = [], []
        for sample in chunk:
            prompt = sample.get("prompt", "")
            sep = "\n" if prompt else ""
            chosen_text = prompt + sep + sample["chosen"]
            rejected_text = prompt + sep + sample["rejected"]
            # Identical pairs carry no preference signal: they contribute a
            # constant log(2) to the loss with exactly cancelling gradients and
            # deflate the reported accuracy.
            if chosen_text == rejected_text:
                num_dropped += 1
                continue
            chosen_texts.append(chosen_text)
            rejected_texts.append(rejected_text)
        if not chosen_texts:
            continue
        chosen_tok = tokenizer(chosen_texts, **tok_kwargs)
        rejected_tok = tokenizer(rejected_texts, **tok_kwargs)
        # Truncation can collapse pairs whose shared prompt reaches max_length
        # into identical token sequences; drop those too.
        keep = (chosen_tok["input_ids"] != rejected_tok["input_ids"]).any(-1)
        num_dropped += int((~keep).sum())
        chosen_ids.append(chosen_tok["input_ids"][keep])
        chosen_masks.append(chosen_tok["attention_mask"][keep])
        rejected_ids.append(rejected_tok["input_ids"][keep])
        rejected_masks.append(rejected_tok["attention_mask"][keep])
    if num_dropped:
        torchrl_logger.info(
            f"Dropped {num_dropped} pair(s) with identical chosen/rejected "
            f"sequences from the {split!r} split."
        )
    if not chosen_ids:
        raise ValueError(
            f"No usable preference pairs remain in the {split!r} split after "
            "dropping identical chosen/rejected sequences. Check the dataset "
            "fields and consider increasing data.max_length."
        )
    return _pairwise_td(
        torch.cat(chosen_ids),
        torch.cat(rejected_ids),
        torch.cat(chosen_masks),
        torch.cat(rejected_masks),
    )


def make_replay_buffer(data: TensorDict, batch_size: int) -> TensorDictReplayBuffer:
    return TensorDictReplayBuffer(
        storage=TensorStorage(data.cpu(), device="cpu"),
        sampler=SamplerWithoutReplacement(drop_last=True),
        batch_size=batch_size,
    )


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
    # Locate the transformer block list: the longest ModuleList in the backbone.
    # A recursive search covers nested layouts such as OPT (decoder.layers) and
    # BERT-style encoders (encoder.layer) that a top-level attribute probe misses.
    layers = None
    for module in transformer.modules():
        if isinstance(module, nn.ModuleList) and (
            layers is None or len(module) > len(layers)
        ):
            layers = module
    if layers is None or len(layers) == 0:
        torchrl_logger.warning(
            "optim.freeze_frac is set but no transformer block list could be "
            "located on the backbone; no layer was frozen."
        )
        return
    num_freeze = int(freeze_frac * len(layers))
    for layer in layers[:num_freeze]:
        layer.requires_grad_(False)


def log_metrics(logger, metrics: dict, step: int) -> None:
    if logger is None:
        return
    for key, value in metrics.items():
        logger.log_scalar(key, value, step)
