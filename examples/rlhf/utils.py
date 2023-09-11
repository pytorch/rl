# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from contextlib import nullcontext

import torch
import torch._dynamo
from hydra.utils import to_absolute_path


def resolve_name_or_path(name_or_path):
    """Hydra changes the working directory, so we need to absolutify paths."""
    if name_or_path.startswith("./") or name_or_path.startswith("/"):
        return to_absolute_path(name_or_path)
    return name_or_path


def get_file_logger(name, filename, level=logging.DEBUG):
    """
    Set up logger that will log to the given filename.
    """
    logger = logging.getLogger(name)
    handler = logging.FileHandler(filename)
    handler.setFormatter(
        # logging.Formatter("%(asctime)s, %(name)s %(levelname)s %(message)s")
        logging.Formatter("%(asctime)s - %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup(device, dtype):
    """
    Set manual seed, configure backend and autocasting.
    """
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch._dynamo.config.cache_size_limit = 256

    if "cuda" not in device:
        return nullcontext()

    return torch.amp.autocast(device_type="cuda", dtype=getattr(torch, dtype))
