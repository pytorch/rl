import argparse

import pytest
import torch
from _utils_internal import get_available_devices
from scipy.stats import ttest_1samp
from torch import nn
from torchrl.data import NdBoundedTensorSpec
from torchrl.data.tensordict.tensordict import TensorDict
from torchrl.envs.transforms.transforms import gSDENoise
from torchrl.modules import ProbabilisticActor
from torchrl.modules.distributions import TanhNormal
from torchrl.modules.distributions.continuous import (
    IndependentNormal,
    NormalParamWrapper,
)
from torchrl.modules.models.exploration import gSDEWrapper
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
    module = NormalParamWrapper(nn.Linear(d_obs, 2 * d_act)).to(device)
    action_spec = NdBoundedTensorSpec(-torch.ones(d_act), torch.ones(d_act), (d_act,))
    policy = ProbabilisticActor(
        spec=action_spec,
        module=module,
        distribution_class=TanhNormal,
        default_interaction_mode="random",
    ).to(device)
    exploratory_policy = OrnsteinUhlenbeckProcessWrapper(policy)

    tensor_dict = TensorDict(
        batch_size=[batch],
        source={"observation": torch.randn(batch, d_obs, device=device)},
        device=device,
    )
    out_noexp = []
    out = []
    for i in range(n_steps):
        tensor_dict_noexp = policy(tensor_dict.select("observation"))
        tensor_dict = exploratory_policy(tensor_dict)
        out.append(tensor_dict.clone())
        out_noexp.append(tensor_dict_noexp.clone())
        tensor_dict.set_("observation", torch.randn(batch, d_obs, device=device))
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
    if gSDE:
        model = torch.nn.LazyLinear(action_dim)
        wrapper = gSDEWrapper(model, action_dim, state_dim).to(device)
        exploration_mode = "net_output"
        distribution_class = IndependentNormal
        distribution_kwargs = {}
        in_keys = ["observation", "_eps_gSDE"]
    else:
        model = torch.nn.LazyLinear(action_dim * 2)
        wrapper = NormalParamWrapper(model).to(device)
        exploration_mode = "random"
        distribution_class = TanhNormal
        distribution_kwargs = {"min": -bound, "max": bound}
        in_keys = ["observation"]
    spec = NdBoundedTensorSpec(
        -torch.ones(action_dim) * bound, torch.ones(action_dim) * bound, (action_dim,)
    ).to(device)
    actor = ProbabilisticActor(
        module=wrapper,
        spec=spec,
        in_keys=in_keys,
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
        gSDENoise(action_dim, state_dim).reset(td)
        assert "_eps_gSDE" in td.keys()
        assert td.get("_eps_gSDE").device == device
    actor(td)
    assert "action" in td.keys()
    if not safe and gSDE:
        assert not spec.is_in(td.get("action"))
    elif safe and gSDE:
        assert spec.is_in(td.get("action"))


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
