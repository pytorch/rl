# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import copy
import importlib.util

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.collectors import Collector

_has_torch_geometric = importlib.util.find_spec("torch_geometric") is not None


@pytest.mark.skipif(not _has_torch_geometric, reason="torch_geometric not installed")
class TestTorchGeometric:
    """Tests for torch_geometric compatibility with torchrl (issue #2679).

    The primary concern is that torch_geometric modules override __deepcopy__
    in a way that conflicts with torchrl's collector parameter-mapping logic.
    """

    def _make_pyg_module(self, in_features=10, hidden=32, out_features=4):
        from torch_geometric.nn import Linear as PyGLinear

        class PyGModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.pyg_linear = PyGLinear(in_features, hidden)
                self.head = nn.Linear(hidden, out_features)

            def forward(self, x):
                return self.head(torch.relu(self.pyg_linear(x)))

        return PyGModule()

    def test_deepcopy(self):
        module = self._make_pyg_module()
        module_copy = copy.deepcopy(module)
        x = torch.randn(5, 10)
        out_orig = module(x)
        out_copy = module_copy(x)
        assert out_orig.shape == out_copy.shape == (5, 4)

    def test_deepcopy_meta_device(self):
        """Reproduce the collector's internal deepcopy pattern that triggers #2679."""
        module = self._make_pyg_module()
        param_and_buf = TensorDict.from_module(module, as_module=True)

        with param_and_buf.data.to("meta").to_module(module):
            module_copy = copy.deepcopy(module)

        param_and_buf.to_module(module_copy)

        x = torch.randn(5, 10)
        out = module_copy(x)
        assert out.shape == (5, 4)

    @pytest.mark.skipif(
        not (torch.cuda.is_available() and torch.cuda.device_count()),
        reason="CUDA required for collector device-mapping test",
    )
    def test_collector_with_pyg_policy(self):
        from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv

        in_features = 7
        act_features = 7
        module = self._make_pyg_module(
            in_features=in_features, hidden=32, out_features=act_features
        )
        policy = TensorDictModule(module, in_keys=["observation"], out_keys=["action"])

        collector = Collector(
            create_env_fn=ContinuousActionVecMockEnv,
            policy=policy,
            total_frames=20,
            frames_per_batch=10,
            device="cpu",
            policy_device="cuda:0",
        )
        for data in collector:
            assert "action" in data
            break
        collector.shutdown()

    def test_collector_with_pyg_policy_same_device(self):
        from torchrl.testing.mocking_classes import ContinuousActionVecMockEnv

        in_features = 7
        act_features = 7
        module = self._make_pyg_module(
            in_features=in_features, hidden=32, out_features=act_features
        )
        policy = TensorDictModule(module, in_keys=["observation"], out_keys=["action"])

        collector = Collector(
            create_env_fn=ContinuousActionVecMockEnv,
            policy=policy,
            total_frames=20,
            frames_per_batch=10,
            device="cpu",
        )
        for data in collector:
            assert "action" in data
            break
        collector.shutdown()

    def test_tensordict_module_wrap(self):
        module = self._make_pyg_module()
        td_module = TensorDictModule(
            module, in_keys=["observation"], out_keys=["action"]
        )
        td = TensorDict({"observation": torch.randn(3, 10)})
        out = td_module(td)
        assert "action" in out
        assert out["action"].shape == (3, 4)
