from __future__ import annotations

import torch

from torchrl.data.utils import CloudpickleWrapper


def decorate_thread_sub_func(func, num_threads):
    """Decorate a function to assert that the number of threads is correct."""

    def new_func(*args, **kwargs):
        assert torch.get_num_threads() == num_threads
        return func(*args, **kwargs)

    return CloudpickleWrapper(new_func)
