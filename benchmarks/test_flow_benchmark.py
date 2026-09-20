# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
from torch import nn

from torchrl.modules import FlowMatchingModel
from torchrl.testing import get_default_devices


class TestFlow:
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("compiled", [False, True])
    @pytest.mark.parametrize("unroll", [1, 2, 5, 10])
    def test_sampling(self, benchmark, device, compiled, unroll):
        model = FlowMatchingModel(
            nn.Sequential(
                nn.Linear(21, 64, device=device),
                nn.Tanh(),
                nn.Linear(64, 4, device=device),
            ),
            4,
            unroll=unroll,
        )
        observation = torch.randn(64, 16, device=device)
        noise = torch.randn(64, 4, device=device)
        torch._dynamo.reset()
        call = torch.compile(model, fullgraph=True) if compiled else model

        def run():
            action = call(observation, noise)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            return action

        with torch.no_grad():
            run()
            benchmark(run)


if __name__ == "__main__":
    pytest.main([__file__, "--benchmark-only"])
