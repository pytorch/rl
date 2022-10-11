# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch.cuda

try:
    import hydra
    from hydra.utils import instantiate

    _has_hydra = True
except ImportError:
    _has_hydra = False

from torchrl.envs.libs.dm_control import _has_dmc
from torchrl.envs.libs.gym import _has_gym

from test.test_configs.test_configs_common import init_hydra  # noqa: F401


@pytest.mark.skipif(not _has_hydra, reason="No hydra found")
@pytest.mark.skipif(not _has_gym, reason="No gym found")
@pytest.mark.skipif(not _has_dmc, reason="No gym found")
class TestEnvAndTransform:
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
