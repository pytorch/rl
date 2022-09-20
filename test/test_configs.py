import argparse

import pytest
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from torchrl.modules import TensorDictModule

from mocking_classes import ContinuousActionVecMockEnv

def make_env():
    def fun():
        return ContinuousActionVecMockEnv()
    return fun

@pytest.mark.parametrize("file", [
    "examples/configs/collectors/async_sync.yaml",
    "examples/configs/collectors/sync_single.yaml",
    "examples/configs/collectors/sync_sync.yaml",
])
def test_collector_configs(file):
    create_env = make_env()
    policy = TensorDictModule(
        nn.Linear(7, 7),
        in_keys=["observation"],
        out_keys=["action"]
    )

    yaml_read = OmegaConf.load(file)
    cfg = OmegaConf.create(yaml_read)

    if cfg.num_workers == 0:
        create_env_fn = create_env
    else:
        create_env_fn = [create_env, ] * cfg.num_workers
    instantiate(cfg.collector, policy=policy, create_env_fn=create_env_fn)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
