# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch


def check_finite(tensor: torch.Tensor):
    """Raise an error if a tensor has non-finite elements."""
    if not tensor.isfinite().all():
        raise ValueError("Encountered a non-finite tensor.")


def _init_first(fun):
    def new_fun(self, *args, **kwargs):
        if not self.initialized:
            self._init()
        return fun(self, *args, **kwargs)

    return new_fun


class _set_missing_tolerance:
    """Context manager to change the transform tolerance to missing values."""

    def __init__(self, transform, mode):
        self.transform = transform
        self.mode = mode

    def __enter__(self):
        self.exit_mode = self.transform.missing_tolerance
        if self.mode != self.exit_mode:
            self.transform.set_missing_tolerance(self.mode)

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.mode != self.exit_mode:
            self.transform.set_missing_tolerance(self.exit_mode)


def _get_reset(reset_key, tensordict):
    _reset = tensordict.get(reset_key, None)
    # reset key must be unraveled already
    parent_td = (
        tensordict.get(reset_key[:-1], None)
        if isinstance(reset_key, tuple)
        else tensordict
    )
    if parent_td is None:
        # we do this just in case the nested td wasn't found
        parent_td = tensordict
    if _reset is None:
        _reset = torch.ones(
            (),
            dtype=torch.bool,
            device=parent_td.device,
        ).expand(parent_td.batch_size)
    if _reset.ndim > parent_td.ndim:
        _reset = _reset.flatten(parent_td.ndim, -1).any(-1)
    return _reset
