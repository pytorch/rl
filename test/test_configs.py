import argparse

import pytest

try:
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    _has_hydra = True
except ImportError:
    _has_hydra = False
from mocking_classes import ContinuousActionVecMockEnv
from torch import nn
from torchrl.modules import TensorDictModule


def make_env():
    def fun():
        return ContinuousActionVecMockEnv()

    return fun


@pytest.fixture(scope="session", autouse=True)
def init_hydra(request):
    hydra.initialize("../examples/configs/")
    request.addfinalizer(GlobalHydra.instance().clear)


@pytest.mark.skipif(not _has_hydra, reason="No hydra found")
@pytest.mark.parametrize(
    "file,num_workers",
    [
        ("async_sync", 2),
        ("sync_single", 0),
        ("sync_sync", 2),
    ],
)
def test_collector_configs(file, num_workers):
    create_env = make_env()
    policy = TensorDictModule(
        nn.Linear(7, 7), in_keys=["observation"], out_keys=["action"]
    )

    cfg = hydra.compose(
        "config", overrides=[f"collector={file}", f"num_workers={num_workers}"]
    )

    if cfg.num_workers == 0:
        create_env_fn = create_env
    else:
        create_env_fn = [
            create_env,
        ] * cfg.num_workers
    collector = instantiate(cfg.collector, policy=policy, create_env_fn=create_env_fn)
    for data in collector:
        assert data.numel() == 200
        break
    collector.shutdown()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
