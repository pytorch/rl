# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Tests for torch.compile compatibility of objectives-related modules."""
from __future__ import annotations

import sys

import pytest
import torch

from packaging import version
from tensordict import TensorDict
from tensordict.nn import ProbabilisticTensorDictModule, set_composite_lp_aggregate

from torchrl.envs.utils import exploration_type, ExplorationType, set_exploration_type

TORCH_VERSION = version.parse(version.parse(torch.__version__).base_version)
IS_WINDOWS = sys.platform == "win32"

pytestmark = [
    pytest.mark.filterwarnings(
        "ignore:`torch.jit.script_method` is deprecated:DeprecationWarning"
    ),
]


@pytest.mark.skipif(
    TORCH_VERSION < version.parse("2.5.0"), reason="requires torch>=2.5"
)
@pytest.mark.skipif(IS_WINDOWS, reason="windows tests do not support compile")
@pytest.mark.skipif(
    sys.version_info >= (3, 14), reason="torch.compile is not supported on Python 3.14+"
)
@set_composite_lp_aggregate(False)
def test_exploration_compile():
    try:
        torch._dynamo.reset_code_caches()
    except Exception:
        # older versions of PT don't have that function
        pass
    m = ProbabilisticTensorDictModule(
        in_keys=["loc", "scale"],
        out_keys=["sample"],
        distribution_class=torch.distributions.Normal,
    )

    # class set_exploration_type_random(set_exploration_type):
    #     __init__ = object.__init__
    #     type = ExplorationType.RANDOM
    it = exploration_type()

    @torch.compile(fullgraph=True)
    def func(t):
        with set_exploration_type(ExplorationType.RANDOM):
            t0 = m(t.clone())
            t1 = m(t.clone())
        return t0, t1

    t = TensorDict(loc=torch.randn(3), scale=torch.rand(3))
    t0, t1 = func(t)
    assert (t0["sample"] != t1["sample"]).any()
    assert it == exploration_type()

    @torch.compile(fullgraph=True)
    def func(t):
        with set_exploration_type(ExplorationType.MEAN):
            t0 = m(t.clone())
            t1 = m(t.clone())
        return t0, t1

    t = TensorDict(loc=torch.randn(3), scale=torch.rand(3))
    t0, t1 = func(t)
    assert (t0["sample"] == t1["sample"]).all()
    assert it == exploration_type()

    @torch.compile(fullgraph=True)
    @set_exploration_type(ExplorationType.RANDOM)
    def func(t):
        t0 = m(t.clone())
        t1 = m(t.clone())
        return t0, t1

    t = TensorDict(loc=torch.randn(3), scale=torch.rand(3))
    t0, t1 = func(t)
    assert (t0["sample"] != t1["sample"]).any()
    assert it == exploration_type()

    @torch.compile(fullgraph=True)
    @set_exploration_type(ExplorationType.MEAN)
    def func(t):
        t0 = m(t.clone())
        t1 = m(t.clone())
        return t0, t1

    t = TensorDict(loc=torch.randn(3), scale=torch.rand(3))
    t0, t1 = func(t)
    assert (t0["sample"] == t1["sample"]).all()
    assert it == exploration_type()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
