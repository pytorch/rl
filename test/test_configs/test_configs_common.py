# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest

try:
    import hydra
    from hydra.core.global_hydra import GlobalHydra
except Exception:
    pass


@pytest.fixture(autouse=True)
def init_hydra():
    GlobalHydra.instance().clear()
    hydra.initialize("../../examples/configs/")
    yield
    GlobalHydra.instance().clear()
