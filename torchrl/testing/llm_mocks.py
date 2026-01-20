# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Shared test fixtures and mock infrastructure for LLM tests."""
from __future__ import annotations

import torch


class MockTransformerConfig:
    """Mock config to mimic transformers model config."""

    def __init__(self, vocab_size: int, max_position_embeddings: int = 2048):
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = vocab_size  # For simplicity


class MockTransformerOutput:
    """Mock output that mimics transformers model output with dict-like access."""

    def __init__(self, logits):
        self.logits = logits

    def __getitem__(self, key):
        """Allow dict-like access for compatibility."""
        if key == "logits":
            return self.logits
        raise KeyError(f"Key {key} not found in model output")


class MockTransformerModel(torch.nn.Module):
    """Mock transformer model that mimics the structure of HuggingFace models."""

    def __init__(self, vocab_size: int, device: torch.device | str | int = "cpu"):
        super().__init__()
        device = torch.device(device)
        self.config = MockTransformerConfig(vocab_size)
        # Simple embedding layer that maps tokens to logits
        self.embedding = torch.nn.Embedding(vocab_size, vocab_size, device=device)
        self.device = device

    def forward(self, input_ids, attention_mask=None, **kwargs):
        """Forward pass that returns logits in the expected format."""
        # Get embeddings (which we'll use as logits for simplicity)
        logits = self.embedding(input_ids.to(self.device) % self.config.vocab_size)
        # Return output object similar to transformers models
        return MockTransformerOutput(logits)

    def get_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")
