# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Test fixtures for service tests that need to be importable by Ray workers."""

from __future__ import annotations

from typing import Any


class SimpleService:
    """A simple service for testing."""

    def __init__(self, value: int = 0):
        self.value = value

    def get_value(self):
        return self.value

    def set_value(self, value: int):
        self.value = value

    def getattr(self, val: str, **kwargs) -> Any:
        if "default" in kwargs:
            default = kwargs["default"]
            return getattr(self, val, default)
        return getattr(self, val)


class TokenizerService:
    """Mock tokenizer service."""

    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size

    def encode(self, text: str):
        return [hash(c) % self.vocab_size for c in text]

    def decode(self, tokens: list):
        return "".join([str(t) for t in tokens])

    def getattr(self, val: str, **kwargs) -> Any:
        if "default" in kwargs:
            default = kwargs["default"]
            return getattr(self, val, default)
        return getattr(self, val)
