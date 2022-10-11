# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
from torch import nn

try:
    import hydra
    from hydra.utils import instantiate

    _has_hydra = True
except ImportError:
    _has_hydra = False

from torchrl.modules import TensorDictModule

from test.test_configs.test_configs_common import init_hydra  # noqa: F401
from ..mocking_classes import ContinuousActionVecMockEnv


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
    create_env = lambda: ContinuousActionVecMockEnv()
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
