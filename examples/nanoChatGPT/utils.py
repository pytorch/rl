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
