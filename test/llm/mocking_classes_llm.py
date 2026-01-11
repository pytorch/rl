# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import random
import string

import torch


class DummyStrDataLoader:
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
