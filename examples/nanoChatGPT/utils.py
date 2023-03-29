import argparse
import os

import torch
import yaml
from torch.distributed import destroy_process_group, init_process_group


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


def init_ddp(ddp_backend, device=None):
    ddp = int(os.environ.get("RANK", -1)) != -1  # is this a ddp run?
    if ddp and device != "cpu":
        init_process_group(backend=ddp_backend)
        ddp_rank = int(os.environ["RANK"])
        ddp_local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{ddp_local_rank}"
        torch.cuda.set_device(device)

        seed_offset = ddp_rank  # each process gets a different seed
        ddp_configs = {
            "is_ddp": ddp,
            "ddp_rank": ddp_rank,
            "ddp_local_rank": ddp_local_rank,
            "world_size": world_size,
            "master_process": (ddp_rank == 0),
            "device": device,
            "seed_offset": seed_offset,
        }
    else:
        # if not ddp, we are running on a single gpu, and one process
        ddp_configs = {
            "is_ddp": ddp,
            "world_size": 1,
            "master_process": True,
            "seed_offset": 0,
            "ddp_local_rank": None,
            "device": device,
        }
    return ddp_configs


def close_ddp():
    destroy_process_group()
