import argparse

import pytest
import torch.cuda

try:
    import hydra
    from hydra.core.global_hydra import GlobalHydra
    from hydra.utils import instantiate

    _has_hydra = True
except ImportError:
    _has_hydra = False
from mocking_classes import ContinuousActionVecMockEnv
from torch import nn
from torchrl.envs.libs.dm_control import _has_dmc
from torchrl.envs.libs.gym import _has_gym
from torchrl.modules import TensorDictModule


def make_env():
    def fun():
        return ContinuousActionVecMockEnv()

    return fun


@pytest.mark.skipif(not _has_hydra, reason="No hydra found")
class TestConfigs:
    @pytest.fixture(scope="class", autouse=True)
    def init_hydra(self, request):
        GlobalHydra.instance().clear()
        hydra.initialize("../examples/configs/")
        request.addfinalizer(GlobalHydra.instance().clear)

    @pytest.mark.parametrize(
        "file,num_workers",
        [
            ("async_sync", 2),
            ("sync_single", 0),
            ("sync_sync", 2),
        ],
    )
    def test_collector_configs(self, file, num_workers):
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
        collector = instantiate(
            cfg.collector, policy=policy, create_env_fn=create_env_fn
        )
        for data in collector:
            assert data.numel() == 200
            break
        collector.shutdown()

    @pytest.mark.skipif(not _has_gym, reason="No gym found")
    @pytest.mark.skipif(not _has_dmc, reason="No gym found")
    @pytest.mark.parametrize(
        "file,from_pixels",
        [
            ("cartpole", True),
            ("cartpole", False),
            ("halfcheetah", True),
            ("halfcheetah", False),
            ("cheetah", True),
            # ("cheetah",False), # processes fail -- to be investigated
        ],
    )
    def test_env_configs(self, file, from_pixels):
        if from_pixels and torch.cuda.device_count() == 0:
            return pytest.skip("not testing pixel rendering without gpu")

        cfg = hydra.compose(
            "config", overrides=[f"env={file}", f"++env.env.from_pixels={from_pixels}"]
        )

        env = instantiate(cfg.env)

        tensordict = env.rollout(3)
        if from_pixels:
            assert "next_pixels" in tensordict.keys()
            assert tensordict["next_pixels"].shape[-1] == 3
        env.rollout(3)
        env.close()
        del env

    @pytest.mark.skipif(not _has_gym, reason="No gym found")
    @pytest.mark.skipif(not _has_dmc, reason="No gym found")
    @pytest.mark.parametrize(
        "env_file,transform_file",
        [
            ["cartpole", "pixels"],
            ["halfcheetah", "pixels"],
            # ["cheetah", "pixels"],
            ["cartpole", "state"],
            ["halfcheetah", "state"],
            ["cheetah", "state"],
        ],
    )
    def test_transforms_configs(self, env_file, transform_file):
        if transform_file == "state":
            from_pixels = False
        else:
            if torch.cuda.device_count() == 0:
                return pytest.skip("not testing pixel rendering without gpu")
            from_pixels = True
        cfg = hydra.compose(
            "config",
            overrides=[
                f"env={env_file}",
                f"++env.env.from_pixels={from_pixels}",
                f"transforms={transform_file}",
            ],
        )

        env = instantiate(cfg.env)
        transforms = [instantiate(transform) for transform in cfg.transforms]
        for t in transforms:
            env.append_transform(t)
        env.rollout(3)
        env.close()
        del env


@pytest.mark.skipif(not _has_hydra, reason="No hydra found")
@pytest.mark.parametrize(
    "file",
    [
        "circular",
        "prioritized",
    ],
)
@pytest.mark.parametrize(
    "size",
    [
        "10",
        None,
    ],
)
def test_replaybuffer(file, size):
    args = [f"replay_buffer={file}"]
    if size is not None:
        args += [f"replay_buffer.size={size}"]
    cfg = hydra.compose("config", overrides=args)
    replay_buffer = instantiate(cfg.replay_buffer)
    assert replay_buffer._capacity == replay_buffer._storage.size


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
