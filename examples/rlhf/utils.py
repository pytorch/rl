# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import argparse
import logging
from contextlib import nullcontext
from pathlib import Path

import torch
import torch._dynamo
import yaml


def load_config(path):
    """Load config from specified path.
    
    Useful in notebooks where argparse can cause problems.
    """
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config


def load_and_update_config(path):
    """Loads config from specified path and allows values to be overridden with command line arguments."""
    config = load_config(path)

    parser = argparse.ArgumentParser()
    for key, value in _yield_nested_items(config):
        parser.add_argument(f"--{key}", type=type(value))

    args = parser.parse_args()
    for key, _ in _yield_nested_items(config):
        value = getattr(args, key)
        if value is not None:
            _set_nested_key(config, key, value)

    return config


def _yield_nested_items(d, sep=".", prefix=""):
    for key, value in d.items():
        if isinstance(value, dict):
            yield from _yield_nested_items(value, sep=sep, prefix=f"{prefix}{key}{sep}")
        else:
            yield f"{prefix}{key}", value


def _set_nested_key(d, k, v, sep="."):
    keys = k.split(sep)
    for key in keys[:-1]:
        d = d[key]
    d[keys[-1]] = v


def get_file_logger(name, filename, level=logging.DEBUG):
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
    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    torch._dynamo.config.cache_size_limit = 256

    if "cuda" not in device:
        return nullcontext()

    return torch.amp.autocast(device_type="cuda", dtype=getattr(torch, dtype))
