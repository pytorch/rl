# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for torch.compile compatibility of collectors."""
from __future__ import annotations

import functools
import sys

import pytest
import torch
from packaging import version
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.collectors import Collector, MultiAsyncCollector, MultiSyncCollector
from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)
IS_WINDOWS = sys.platform == "win32"

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:`torch.jit.script_method` is deprecated:DeprecationWarning"
    ),
]


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.5.0"), reason="requires Torch >= 2.5.0"
)
@pytest.mark.skipif(IS_WINDOWS, reason="windows is not supported for compile tests.")
@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="torch.compile is not supported on Python 3.14+"
)
class TestCompile:
    @pytest.mark.parametrize(
        "collector_cls",
        # Clearing compiled policies causes segfault on machines with cuda
        [Collector, MultiAsyncCollector, MultiSyncCollector]
        if not torch.cuda.is_available()
        else [Collector],
    )
    @pytest.mark.parametrize("compile_policy", [True, {}, {"mode": "default"}])
    @pytest.mark.parametrize(
        "device", [torch.device("cuda:0" if torch.cuda.is_available() else "cpu")]
    )
    def test_compiled_policy(self, collector_cls, compile_policy, device):
        policy = TensorDictModule(
            nn.Linear(7, 7, device=device), in_keys=["observation"], out_keys=["action"]
        )
        make_env = functools.partial(ContinuousActionVecMockEnv, device=device)
        if collector_cls is Collector:
            torch._dynamo.reset_code_caches()
            collector = Collector(
                make_env(),
                policy,
                frames_per_batch=10,
                total_frames=30,
                compile_policy=compile_policy,
            )
            assert collector.compiled_policy
        else:
            collector = collector_cls(
                [make_env] * 2,
                policy,
                frames_per_batch=10,
                total_frames=30,
                compile_policy=compile_policy,
            )
            assert collector.compiled_policy
        try:
            for data in collector:
                assert data is not None
        finally:
            collector.shutdown()
            del collector

    @pytest.mark.gpu
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
    @pytest.mark.parametrize(
        "collector_cls",
        [Collector],
    )
    @pytest.mark.parametrize("cudagraph_policy", [True, {}, {"warmup": 10}])
    def test_cudagraph_policy(self, collector_cls, cudagraph_policy):
        device = torch.device("cuda:0")
        policy = TensorDictModule(
            nn.Linear(7, 7, device=device), in_keys=["observation"], out_keys=["action"]
        )
        make_env = functools.partial(ContinuousActionVecMockEnv, device=device)
        if collector_cls is Collector:
            collector = Collector(
                make_env(),
                policy,
                frames_per_batch=30,
                total_frames=120,
                cudagraph_policy=cudagraph_policy,
                device=device,
            )
            assert collector.cudagraphed_policy
        else:
            collector = collector_cls(
                [make_env] * 2,
                policy,
                frames_per_batch=30,
                total_frames=120,
                cudagraph_policy=cudagraph_policy,
                device=device,
            )
            assert collector.cudagraphed_policy
        try:
            for data in collector:
                assert data is not None
        finally:
            collector.shutdown()
            del collector


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
