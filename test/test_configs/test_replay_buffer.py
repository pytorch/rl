# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

try:
    import hydra
    from hydra.utils import instantiate

    _has_hydra = True
except ImportError:
    _has_hydra = False

from test.test_configs.test_configs_common import init_hydra  # noqa: F401


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
def test_replay_buffer(file, size):
    args = [f"replay_buffer={file}"]
    if size is not None:
        args += [f"replay_buffer.size={size}"]
    cfg = hydra.compose("config", overrides=args)
    replay_buffer = instantiate(cfg.replay_buffer)
    assert replay_buffer._capacity == replay_buffer._storage.max_size
