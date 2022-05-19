# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pytest
import torch
from _utils_internal import get_available_devices
from scipy.stats import ttest_1samp
from torch import nn
from torchrl.data import NdBoundedTensorSpec
from torchrl.data.tensordict.tensordict import TensorDict
from torchrl.envs.transforms.transforms import gSDENoise
from torchrl.modules import TDModule
from torchrl.modules.distributions import TanhNormal
from torchrl.modules.distributions.continuous import (
    IndependentNormal,
    NormalParamWrapper,
)
from torchrl.modules.models.exploration import gSDEWrapper
from torchrl.modules.td_module.actors import ProbabilisticActor
from torchrl.modules.td_module.exploration import (
    _OrnsteinUhlenbeckProcess,
    OrnsteinUhlenbeckProcessWrapper,
)


@pytest.mark.parametrize("device", get_available_devices())
def test_ou(device, seed=0):
    torch.manual_seed(seed)
    td = TensorDict({"action": torch.randn(3, device=device) / 10}, batch_size=[])
    ou = _OrnsteinUhlenbeckProcess(10.0, mu=2.0, x0=-4, sigma=0.1, sigma_min=0.01)

    tds = []
    for i in range(2000):
        td = ou.add_sample(td)
        tds.append(td.clone())
        td.set_("action", torch.randn(3) / 10)
        if i % 1000 == 0:
            td.zero_()

    tds = torch.stack(tds, 0)

    tset, pval_acc = ttest_1samp(tds.get("action")[950:1000, 0].cpu().numpy(), 2.0)
    tset, pval_reg = ttest_1samp(tds.get("action")[:50, 0].cpu().numpy(), 2.0)
    assert pval_acc > 0.05
    assert pval_reg < 0.1

    tset, pval_acc = ttest_1samp(tds.get("action")[1950:2000, 0].cpu().numpy(), 2.0)
    tset, pval_reg = ttest_1samp(tds.get("action")[1000:1050, 0].cpu().numpy(), 2.0)
    assert pval_acc > 0.05
    assert pval_reg < 0.1


@pytest.mark.parametrize("device", get_available_devices())
def test_ou_wrapper(device, d_obs=4, d_act=6, batch=32, n_steps=100, seed=0):
    torch.manual_seed(seed)
    net = NormalParamWrapper(nn.Linear(d_obs, 2 * d_act)).to(device)
    module = TDModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
    action_spec = NdBoundedTensorSpec(-torch.ones(d_act), torch.ones(d_act), (d_act,))
    policy = ProbabilisticActor(
        spec=action_spec,
        module=module,
        dist_param_keys=["loc", "scale"],
        distribution_class=TanhNormal,
        default_interaction_mode="random",
    ).to(device)
    exploratory_policy = OrnsteinUhlenbeckProcessWrapper(policy)

    tensordict = TensorDict(
        batch_size=[batch],
        source={"observation": torch.randn(batch, d_obs, device=device)},
        device=device,
    )
    out_noexp = []
    out = []
    for i in range(n_steps):
        tensordict_noexp = policy(tensordict.select("observation"))
        tensordict = exploratory_policy(tensordict)
        out.append(tensordict.clone())
        out_noexp.append(tensordict_noexp.clone())
        tensordict.set_("observation", torch.randn(batch, d_obs, device=device))
    out = torch.stack(out, 0)
    out_noexp = torch.stack(out_noexp, 0)
    assert (out_noexp.get("action") != out.get("action")).all()
    assert (out.get("action") <= 1.0).all(), out.get("action").min()
    assert (out.get("action") >= -1.0).all(), out.get("action").max()


@pytest.mark.parametrize("state_dim", [7])
@pytest.mark.parametrize("action_dim", [5, 11])
@pytest.mark.parametrize("gSDE", [True, False])
@pytest.mark.parametrize("safe", [True, False])
@pytest.mark.parametrize("device", get_available_devices())
def test_gsde(state_dim, action_dim, gSDE, device, safe, batch=16, bound=0.1):
    torch.manual_seed(0)
    exploration_mode = "random"
    if gSDE:
        model = torch.nn.LazyLinear(action_dim)
        wrapper = gSDEWrapper(model, action_dim, state_dim).to(device)
        in_keys = ["observation", "_eps_gSDE"]
        module = TDModule(
            wrapper, in_keys=in_keys, out_keys=["loc", "scale", "action", "_eps_gSDE"]
        ).to(device)
        distribution_class = IndependentNormal
        distribution_kwargs = {}
    else:
        in_keys = ["observation"]
        model = torch.nn.LazyLinear(action_dim * 2)
        wrapper = NormalParamWrapper(model)
        module = TDModule(wrapper, in_keys=in_keys, out_keys=["loc", "scale"]).to(
            device
        )
        distribution_class = TanhNormal
        distribution_kwargs = {"min": -bound, "max": bound}
    spec = NdBoundedTensorSpec(
        -torch.ones(action_dim) * bound, torch.ones(action_dim) * bound, (action_dim,)
    ).to(device)

    actor = ProbabilisticActor(
        module=module,
        spec=spec,
        dist_param_keys=["loc", "scale"],
        out_key_sample=["action"],
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        default_interaction_mode=exploration_mode,
        safe=safe,
    )

    td = TensorDict(
        {"observation": torch.randn(batch, state_dim, device=device)},
        [
            batch,
        ],
    )
    if gSDE:
        gSDENoise().reset(td)
        assert "_eps_gSDE" in td.keys()
        assert td.get("_eps_gSDE").device == device
    actor(td)
    assert "action" in td.keys()
    if not safe and gSDE:
        assert not spec.is_in(td.get("action"))
    elif safe and gSDE:
        assert spec.is_in(td.get("action"))

    if not safe:
        action1 = module(td).get("action")
        action2 = actor(td).get("action")
        if gSDE:
            torch.testing.assert_allclose(action1, action2)
        else:
            with pytest.raises(AssertionError):
                torch.testing.assert_allclose(action1, action2)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
