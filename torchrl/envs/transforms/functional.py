# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List

from torch import Tensor


# copied from torchvision
def _get_image_num_channels(img: Tensor) -> int:
    if img.ndim == 2:
        return 1
    elif img.ndim > 2:
        return img.shape[-3]

    raise TypeError("Input ndim should be 2 or more. Got {}".format(img.ndim))


def _assert_channels(img: Tensor, permitted: List[int]) -> None:
    c = _get_image_num_channels(img)
    if c not in permitted:
        raise TypeError(
            f"Input image tensor permitted channel values are {permitted}, but found "
            f"{c} (full shape: {img.shape})"
        )


def rgb_to_grayscale(img: Tensor, num_output_channels: int = 1) -> Tensor:
    """Turns an RGB image into grayscale."""
    if img.ndim < 3:
        raise TypeError(
            "Input image tensor should have at least 3 dimensions, but found"
            "{}".format(img.ndim)
        )
    _assert_channels(img, [3])

    if num_output_channels not in (1, 3):
        raise ValueError("num_output_channels should be either 1 or 3")

    r, g, b = img.unbind(dim=-3)
    l_img = (0.2989 * r + 0.587 * g + 0.114 * b).to(img.dtype)
    l_img = l_img.unsqueeze(dim=-3)

    if num_output_channels == 3:
        return l_img.expand(img.shape)

    return l_img
