import math
import os
from contextlib import nullcontext
from pathlib import Path

import torch

HERE = Path(__file__).parent
import argparse
import logging

import yaml


def load_and_update_config(path):
    """
    Loads config from specified path and allows values to be overridden with command
    line arguments
    """
    with open(path) as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    parser = argparse.ArgumentParser()
    for key, value in config.items():
        parser.add_argument(f"--{key}", type=type(value))

    args = parser.parse_args()
    for key in config:
        value = getattr(args, key)
        if value is not None:
            config[key] = value

    return config


def get_file_logger(name, filename, level=logging.DEBUG):
    logger = logging.getLogger(name)
    handler = logging.FileHandler(filename)
    handler.setFormatter(
        logging.Formatter("%(asctime)s, %(name)s %(levelname)s %(message)s")
    )
    logger.addHandler(handler)
    logger.setLevel(level)
    return logger


def setup(config):
    os.makedirs(config["out_dir"], exist_ok=True)

    torch.manual_seed(1337)
    torch.backends.cuda.matmul.allow_tf32 = True  # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True  # allow tf32 on cudnn
    # for later use in torch.autocast
    device_type = "cuda" if "cuda" in config["device"] else "cpu"
    # note: float16 data type will automatically use a GradScaler
    ptdtype = {
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
    }[config["dtype"]]

    ctx = (
        nullcontext()
        if device_type == "cpu"
        else torch.amp.autocast(device_type=device_type, dtype=ptdtype)
    )
    return ctx


def create_lr_scheduler(config):

    # learning rate decay scheduler (cosine with warmup)
    def scheduler(it):
        # 1) linear warmup for warmup_iters steps
        if it < config["warmup_iters"]:
            return config["learning_rate"] * it / config["warmup_iters"]
        # 2) if it > lr_decay_iters, return min learning rate
        if it > config["lr_decay_iters"]:
            return config["min_lr"]
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - config["warmup_iters"]) / (
            config["lr_decay_iters"] - config["warmup_iters"]
        )
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # coeff ranges 0..1
        return config["min_lr"] + coeff * (config["learning_rate"] - config["min_lr"])

    def fixed_lr(_):
        return config["learning_rate"]

    if config["decay_lr"]:
        return scheduler
    return fixed_lr
