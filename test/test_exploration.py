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
from torchrl.data import CompositeSpec, NdBoundedTensorSpec
from torchrl.data.tensordict.tensordict import TensorDict
from torchrl.envs.transforms.transforms import gSDENoise
from torchrl.envs.utils import set_exploration_mode
from torchrl.modules import TensorDictModule, TensorDictSequential
from torchrl.modules.distributions import TanhNormal
from torchrl.modules.distributions.continuous import (
    IndependentNormal,
    NormalParamWrapper,
)
from torchrl.modules.models.exploration import LazygSDEModule
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.modules.tensordict_module.exploration import (
    _OrnsteinUhlenbeckProcess,
    AdditiveGaussianWrapper,
    OrnsteinUhlenbeckProcessWrapper,
)


@pytest.mark.parametrize("device", get_available_devices())
def test_ou(device, seed=0):
    torch.manual_seed(seed)
    td = TensorDict({"action": torch.randn(3) / 10}, batch_size=[], device=device)
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
    module = TensorDictModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
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
    for _ in range(n_steps):
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


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("spec_origin", ["spec", "policy", None])
class TestAdditiveGaussian:
    def test_additivegaussian_sd(
        self,
        device,
        spec_origin,
        d_obs=4,
        d_act=6,
        batch=32,
        n_steps=100,
        seed=0,
    ):
        torch.manual_seed(seed)
        net = NormalParamWrapper(nn.Linear(d_obs, 2 * d_act)).to(device)
        action_spec = NdBoundedTensorSpec(
            -torch.ones(d_act, device=device),
            torch.ones(d_act, device=device),
            (d_act,),
            device=device,
        )
        module = TensorDictModule(
            net,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
            spec=None,
        )
        policy = ProbabilisticActor(
            spec=CompositeSpec(action=action_spec) if spec_origin is not None else None,
            module=module,
            dist_param_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            default_interaction_mode="random",
        ).to(device)
        given_spec = action_spec if spec_origin == "spec" else None
        exploratory_policy = AdditiveGaussianWrapper(policy, spec=given_spec).to(device)
        if spec_origin is not None:
            sigma_init = (
                action_spec.project(
                    torch.randn(1000000, action_spec.shape[-1], device=device)
                ).std()
                * exploratory_policy.sigma_init
            )
            sigma_end = (
                action_spec.project(
                    torch.randn(1000000, action_spec.shape[-1], device=device)
                ).std()
                * exploratory_policy.sigma_end
            )
        else:
            sigma_init = exploratory_policy.sigma_init
            sigma_end = exploratory_policy.sigma_end
        if spec_origin is None:
            with pytest.raises(
                RuntimeError,
                match="the action spec must be provided to AdditiveGaussianWrapper",
            ):
                exploratory_policy._add_noise(action_spec.rand((100000,)).zero_())
            return
        noisy_action = exploratory_policy._add_noise(
            action_spec.rand((100000,)).zero_()
        )
        if spec_origin is not None:
            assert action_spec.is_in(noisy_action), (
                noisy_action.min(),
                noisy_action.max(),
            )
        assert abs(noisy_action.std() - sigma_init) < 1e-1

        for _ in range(exploratory_policy.annealing_num_steps):
            exploratory_policy.step(1)
        noisy_action = exploratory_policy._add_noise(
            action_spec.rand((100000,)).zero_()
        )
        assert abs(noisy_action.std() - sigma_end) < 1e-1

    def test_additivegaussian_wrapper(
        self, device, spec_origin, d_obs=4, d_act=6, batch=32, n_steps=100, seed=0
    ):
        torch.manual_seed(seed)
        net = NormalParamWrapper(nn.Linear(d_obs, 2 * d_act)).to(device)
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        action_spec = NdBoundedTensorSpec(
            -torch.ones(d_act, device=device),
            torch.ones(d_act, device=device),
            (d_act,),
            device=device,
        )
        policy = ProbabilisticActor(
            spec=action_spec if spec_origin is not None else None,
            module=module,
            dist_param_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            default_interaction_mode="random",
        ).to(device)
        given_spec = action_spec if spec_origin == "spec" else None
        exploratory_policy = AdditiveGaussianWrapper(
            policy, spec=given_spec, safe=False
        ).to(device)

        tensordict = TensorDict(
            batch_size=[batch],
            source={"observation": torch.randn(batch, d_obs, device=device)},
            device=device,
        )
        out_noexp = []
        out = []
        for _ in range(n_steps):
            tensordict_noexp = policy(tensordict.select("observation"))
            tensordict = exploratory_policy(tensordict)
            out.append(tensordict.clone())
            out_noexp.append(tensordict_noexp.clone())
            tensordict.set_("observation", torch.randn(batch, d_obs, device=device))
        out = torch.stack(out, 0)
        out_noexp = torch.stack(out_noexp, 0)
        assert (out_noexp.get("action") != out.get("action")).all()
        if spec_origin is not None:
            assert (out.get("action") <= 1.0).all(), out.get("action").min()
            assert (out.get("action") >= -1.0).all(), out.get("action").max()
            if action_spec is not None:
                assert action_spec.is_in(out.get("action"))


@pytest.mark.parametrize("state_dim", [7])
@pytest.mark.parametrize("action_dim", [5, 11])
@pytest.mark.parametrize("gSDE", [True, False])
@pytest.mark.parametrize("safe", [True, False])
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("exploration_mode", ["random", "mode"])
def test_gsde(
    state_dim, action_dim, gSDE, device, safe, exploration_mode, batch=16, bound=0.1
):
    torch.manual_seed(0)
    if gSDE:
        model = torch.nn.LazyLinear(action_dim, device=device)
        in_keys = ["observation"]
        module = TensorDictSequential(
            TensorDictModule(model, in_keys=in_keys, out_keys=["action"]),
            TensorDictModule(
                LazygSDEModule(device=device),
                in_keys=["action", "observation", "_eps_gSDE"],
                out_keys=["loc", "scale", "action", "_eps_gSDE"],
            ),
        )
        distribution_class = IndependentNormal
        distribution_kwargs = {}
    else:
        in_keys = ["observation"]
        model = torch.nn.LazyLinear(action_dim * 2, device=device)
        wrapper = NormalParamWrapper(model)
        module = TensorDictModule(wrapper, in_keys=in_keys, out_keys=["loc", "scale"])
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
        [batch],
        device=device,
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
        with set_exploration_mode(exploration_mode):
            action1 = module(td).get("action")
        action2 = actor(td).get("action")
        if gSDE or exploration_mode == "mode":
            torch.testing.assert_close(action1, action2)
        else:
            with pytest.raises(AssertionError):
                torch.testing.assert_close(action1, action2)


@pytest.mark.parametrize(
    "state_dim",
    [
        (5,),
        (12,),
        (12, 3),
    ],
)
@pytest.mark.parametrize("action_dim", [5, 12])
@pytest.mark.parametrize("mean", [0, -2])
@pytest.mark.parametrize("std", [1, 2])
@pytest.mark.parametrize("sigma_init", [None, 1.5, 3])
@pytest.mark.parametrize("learn_sigma", [False, True])
@pytest.mark.parametrize("device", get_available_devices())
def test_gsde_init(sigma_init, state_dim, action_dim, mean, std, device, learn_sigma):
    torch.manual_seed(0)
    state = torch.randn(100000, *state_dim, device=device) * std + mean
    action = torch.randn(100000, *state_dim[:-1], action_dim, device=device)
    # lazy
    gsde_lazy = LazygSDEModule(sigma_init=sigma_init, learn_sigma=learn_sigma).to(
        device
    )
    _eps = torch.randn(
        100000, *state_dim[:-1], action_dim, state_dim[-1], device=device
    )
    with set_exploration_mode("random"):
        mu, sigma, action_out, _eps = gsde_lazy(action, state, _eps)
    sigma_init = sigma_init if sigma_init else 1.0
    assert (
        abs(sigma_init - sigma.mean()) < 0.3
    ), f"failed: mean={mean}, std={std}, sigma_init={sigma_init}, actual: {sigma.mean()}"


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
