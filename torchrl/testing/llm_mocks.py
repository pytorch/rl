# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Shared test fixtures and mock infrastructure for LLM tests."""
from __future__ import annotations

import random
import string

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
        logits = self.embedding(input_ids % self.config.vocab_size)
        # Return output object similar to transformers models
        return MockTransformerOutput(logits)

    def get_tokenizer(self):
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")


class DummyStrDataLoader:
    """A dummy dataloader that yields random strings for LLM testing."""

    def __init__(self, batch_size=0):
        if isinstance(batch_size, tuple):
            batch_size = torch.Size(batch_size).numel()
        self.batch_size = batch_size

    def generate_random_string(self, length=10):
        """Generate a random string of a given length."""
        return "".join(random.choice(string.ascii_lowercase) for _ in range(length))

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_size == 0:
            return {"text": self.generate_random_string()}
        else:
            return {
                "query": [self.generate_random_string() for _ in range(self.batch_size)]
            }


class DummyTensorDataLoader:
    """A dummy dataloader that yields random token tensors for LLM testing."""

    def __init__(self, batch_size=0, max_length=10, padding=False):
        if isinstance(batch_size, tuple):
            batch_size = torch.Size(batch_size).numel()
        self.batch_size = batch_size
        self.max_length = max_length
        self.padding = padding

    def generate_random_tensor(self):
        """Generate a tensor of random int64 values."""
        length = random.randint(1, self.max_length)
        rt = torch.randint(1, 10000, (length,))
        return rt

    def pad_tensor(self, tensor):
        """Pad a tensor to the maximum length."""
        padding_length = self.max_length - len(tensor)
        return torch.cat((torch.zeros(padding_length, dtype=torch.int64), tensor))

    def __iter__(self):
        return self

    def __next__(self):
        if self.batch_size == 0:
            tensor = self.generate_random_tensor()
            tokens = self.pad_tensor(tensor) if self.padding else tensor
        else:
            tensors = [self.generate_random_tensor() for _ in range(self.batch_size)]
            if self.padding:
                tensors = [self.pad_tensor(tensor) for tensor in tensors]
                tokens = torch.stack(tensors)
            else:
                tokens = tensors
        return {"tokens": tokens, "attention_mask": tokens != 0}
