# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import math
import os
import tempfile
import warnings

import pytest
import torch
from scipy.stats import ttest_1samp
from tensordict import TensorDict

from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from torch import nn
from torchrl._utils import _replace_last

from torchrl.collectors import Collector
from torchrl.data import Bounded, Categorical, Composite, OneHot
from torchrl.envs import SerialEnv
from torchrl.envs.transforms.transforms import gSDENoise, InitTracker, TransformedEnv
from torchrl.envs.utils import ExplorationType, set_exploration_type
from torchrl.modules import SafeModule, SafeSequential
from torchrl.modules.distributions import (
    IndependentNormal,
    NormalParamExtractor,
    TanhNormal,
)
from torchrl.modules.models.exploration import (
    ConsistentDropoutModule,
    LazygSDEModule,
    NoisyLinear,
    reset_noise,
)
from torchrl.modules.tensordict_module.actors import (
    Actor,
    ProbabilisticActor,
    QValueActor,
)
from torchrl.modules.tensordict_module.exploration import (
    _OrnsteinUhlenbeckProcess,
    AdditiveGaussianModule,
    EGreedyModule,
    EGreedyWrapper,
    OrnsteinUhlenbeckProcessModule,
)

from torchrl.testing import get_default_devices
from torchrl.testing.mocking_classes import (
    ContinuousActionVecMockEnv,
    CountingEnvCountModule,
    NestedCountingEnv,
)


class TestEGreedy:
    @pytest.mark.parametrize("eps_init", [0.0, 0.5, 1])
    @pytest.mark.parametrize("module", [True, False])
    @set_exploration_type(InteractionType.RANDOM)
    def test_egreedy(self, eps_init, module):
        torch.manual_seed(0)
        spec = Bounded(1, 1, torch.Size([4]))
        module = torch.nn.Linear(4, 4, bias=False)

        policy = Actor(spec=spec, module=module)
        if module:
            explorative_policy = TensorDictSequential(
                policy, EGreedyModule(eps_init=eps_init, eps_end=eps_init, spec=spec)
            )
        else:
            explorative_policy = EGreedyWrapper(
                policy, eps_init=eps_init, eps_end=eps_init
            )
        td = TensorDict({"observation": torch.zeros(10, 4)}, batch_size=[10])
        action = explorative_policy(td).get("action")
        if eps_init == 0:
            assert (action == 0).all()
        elif eps_init == 1:
            assert (action == 1).all()
        else:
            assert (action == 1).any()
            assert (action == 0).any()
            assert ((action == 1) | (action == 0)).all()

    @pytest.mark.parametrize("eps_init", [0.0, 0.5, 1])
    @pytest.mark.parametrize("module", [True, False])
    @pytest.mark.parametrize("spec_class", ["discrete", "one_hot"])
    def test_egreedy_masked(self, module, eps_init, spec_class):
        torch.manual_seed(0)
        action_size = 4
        batch_size = (3, 4, 2)
        module = torch.nn.Linear(action_size, action_size, bias=False)
        if spec_class == "discrete":
            spec = Categorical(action_size)
        else:
            spec = OneHot(
                action_size,
                shape=(action_size,),
            )
        policy = QValueActor(spec=spec, module=module, action_mask_key="action_mask")
        if module:
            explorative_policy = TensorDictSequential(
                policy,
                EGreedyModule(
                    eps_init=eps_init,
                    eps_end=eps_init,
                    spec=spec,
                    action_mask_key="action_mask",
                ),
            )
        else:
            explorative_policy = EGreedyWrapper(
                policy,
                eps_init=eps_init,
                eps_end=eps_init,
                action_mask_key="action_mask",
            )

        td = TensorDict(
            {"observation": torch.zeros(*batch_size, action_size)},
            batch_size=batch_size,
        )
        with pytest.raises(
            KeyError, match="Action mask key action_mask not found in TensorDict"
        ):
            explorative_policy(td)

        torch.manual_seed(0)
        action_mask = torch.ones(*batch_size, action_size).to(torch.bool)
        td = TensorDict(
            {
                "observation": torch.zeros(*batch_size, action_size),
                "action_mask": action_mask,
            },
            batch_size=batch_size,
        )
        action = explorative_policy(td).get("action")

        torch.manual_seed(0)
        action_mask = torch.randint(high=2, size=(*batch_size, action_size)).to(
            torch.bool
        )
        while not action_mask.any(dim=-1).all() or action_mask.all():
            action_mask = torch.randint(high=2, size=(*batch_size, action_size)).to(
                torch.bool
            )

        td = TensorDict(
            {
                "observation": torch.zeros(*batch_size, action_size),
                "action_mask": action_mask,
            },
            batch_size=batch_size,
        )
        masked_action = explorative_policy(td).get("action")

        if spec_class == "discrete":
            action = spec.to_one_hot(action)
            masked_action = spec.to_one_hot(masked_action)

        assert not (action[~action_mask] == 0).all()
        assert (masked_action[~action_mask] == 0).all()

    def test_no_spec_error(self):
        torch.manual_seed(0)
        action_size = 4
        batch_size = (3, 4, 2)
        module = torch.nn.Linear(action_size, action_size, bias=False)
        spec = OneHot(action_size, shape=(action_size,))
        policy = QValueActor(spec=spec, module=module)
        explorative_policy = TensorDictSequential(
            policy,
            EGreedyModule(spec=None),
        )
        td = TensorDict(
            {
                "observation": torch.zeros(*batch_size, action_size),
            },
            batch_size=batch_size,
        )

        with pytest.raises(
            RuntimeError,
            match="Failed while executing module|spec must be provided to the exploration wrapper",
        ):
            explorative_policy(td)

    @pytest.mark.parametrize("module", [True, False])
    def test_wrong_action_shape(self, module):
        torch.manual_seed(0)
        spec = Bounded(1, 1, torch.Size([4]))
        module = torch.nn.Linear(4, 5, bias=False)

        policy = Actor(spec=spec, module=module)
        if module:
            explorative_policy = TensorDictSequential(policy, EGreedyModule(spec=spec))
        else:
            explorative_policy = EGreedyWrapper(
                policy,
            )
        td = TensorDict({"observation": torch.zeros(10, 4)}, batch_size=[10])
        with pytest.raises(
            ValueError, match="Action spec shape does not match the action shape"
        ):
            explorative_policy(td)


@pytest.mark.parametrize("device", get_default_devices())
class TestOrnsteinUhlenbeckProcess:
    def test_ou_process(self, device, seed=0):
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

    @pytest.mark.parametrize("interface", ["module"])
    def test_ou(
        self, device, interface, d_obs=4, d_act=6, batch=32, n_steps=100, seed=0
    ):
        torch.manual_seed(seed)
        net = nn.Sequential(
            nn.Linear(d_obs, 2 * d_act, device=device), NormalParamExtractor()
        )
        module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        action_spec = Bounded(-torch.ones(d_act), torch.ones(d_act), (d_act,))
        policy = ProbabilisticActor(
            spec=action_spec,
            module=module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            default_interaction_type=InteractionType.RANDOM,
        )

        if interface == "module":
            ou = OrnsteinUhlenbeckProcessModule(spec=action_spec, device=device)
            exploratory_policy = TensorDictSequential(policy, ou)
        else:
            raise NotImplementedError

        tensordict = TensorDict(
            batch_size=[batch],
            source={
                "observation": torch.randn(batch, d_obs, device=device),
                "is_init": torch.ones(batch, 1, dtype=torch.bool, device=device),
            },
            device=device,
        )
        out_noexp = []
        out = []
        for i in range(n_steps):
            tensordict_noexp = policy(
                tensordict.clone().exclude(
                    *(key for key in tensordict.keys() if key.startswith("_"))
                )
            )
            tensordict = exploratory_policy(tensordict.clone())
            if i == 0:
                assert (tensordict[ou.ou.steps_key] == 1).all()
            elif i == n_steps // 2 + 1:
                assert (tensordict[ou.ou.steps_key][: batch // 2] == 1).all()
            else:
                assert not (tensordict[ou.ou.steps_key] == 1).any()

            out.append(tensordict.clone())
            out_noexp.append(tensordict_noexp.clone())
            tensordict.set_("observation", torch.randn(batch, d_obs, device=device))
            tensordict["is_init"][:] = 0
            if i == n_steps // 2:
                tensordict["is_init"][: batch // 2] = 1

        out = torch.stack(out, 0)
        out_noexp = torch.stack(out_noexp, 0)
        assert (out_noexp.get("action") != out.get("action")).all()
        assert (out.get("action") <= 1.0).all(), out.get("action").min()
        assert (out.get("action") >= -1.0).all(), out.get("action").max()

    @pytest.mark.parametrize("parallel_spec", [True, False])
    @pytest.mark.parametrize("probabilistic", [True, False])
    @pytest.mark.parametrize("interface", ["module"])
    def test_collector(self, device, parallel_spec, probabilistic, interface, seed=0):
        torch.manual_seed(seed)
        env = SerialEnv(
            2,
            ContinuousActionVecMockEnv,
        )
        env = TransformedEnv(env.to(device), InitTracker())
        # the module must work with the action spec of a single env or a serial env
        if parallel_spec:
            action_spec = env.action_spec
        else:
            action_spec = ContinuousActionVecMockEnv(device=device).action_spec
        d_act = action_spec.shape[-1]
        if probabilistic:
            net = nn.Sequential(nn.LazyLinear(2 * d_act), NormalParamExtractor()).to(
                device
            )
            module = SafeModule(
                net,
                in_keys=["observation"],
                out_keys=["loc", "scale"],
            )
            policy = ProbabilisticActor(
                module=module,
                in_keys=["loc", "scale"],
                distribution_class=TanhNormal,
                default_interaction_type=InteractionType.RANDOM,
                spec=action_spec,
            ).to(device)
        else:
            net = nn.LazyLinear(d_act).to(device)
            policy = Actor(
                net, in_keys=["observation"], out_keys=["action"], spec=action_spec
            )

        if interface == "module":
            exploratory_policy = TensorDictSequential(
                policy, OrnsteinUhlenbeckProcessModule(spec=action_spec, device=device)
            )
        else:
            raise NotImplementedError
        exploratory_policy(env.reset())
        collector = Collector(
            create_env_fn=env,
            policy=exploratory_policy,
            frames_per_batch=100,
            total_frames=1000,
            device=device,
        )
        for _ in collector:
            # check that we can run the policy
            pass
        return

    @pytest.mark.parametrize("nested_obs_action", [True, False])
    @pytest.mark.parametrize("nested_done", [True, False])
    @pytest.mark.parametrize("is_init_key", ["some"])
    @pytest.mark.parametrize("interface", ["module"])
    def test_nested(
        self,
        device,
        nested_obs_action,
        nested_done,
        is_init_key,
        interface,
        seed=0,
        n_envs=2,
        nested_dim=5,
        frames_per_batch=100,
    ):
        torch.manual_seed(seed)

        env = SerialEnv(
            n_envs,
            lambda: TransformedEnv(
                NestedCountingEnv(
                    nest_obs_action=nested_obs_action,
                    nest_done=nested_done,
                    nested_dim=nested_dim,
                ).to(device),
                InitTracker(init_key=is_init_key),
            ),
        )

        action_spec = env.action_spec
        action_spec.shape[-1]

        policy = TensorDictModule(
            CountingEnvCountModule(action_spec=action_spec),
            in_keys=[("data", "states") if nested_obs_action else "observation"],
            out_keys=[env.action_key],
        )
        if interface == "module":
            exploratory_policy = TensorDictSequential(
                policy,
                OrnsteinUhlenbeckProcessModule(
                    spec=action_spec.clone(),
                    action_key=env.action_key,
                    is_init_key=is_init_key,
                ).to(device),
            )
        else:
            raise NotImplementedError
        collector = Collector(
            create_env_fn=env,
            policy=exploratory_policy,
            frames_per_batch=frames_per_batch,
            total_frames=1000,
            device=device,
        )
        for _td in collector:
            for done_key in env.done_keys:
                assert (
                    _td[_replace_last(done_key, is_init_key)].shape
                    == _td[done_key].shape
                )
            break

        return

    def test_no_spec_error(self, device):
        with pytest.raises(RuntimeError, match="spec cannot be None."):
            OrnsteinUhlenbeckProcessModule(spec=None).to(device)


@pytest.mark.parametrize("device", get_default_devices())
class TestAdditiveGaussian:
    @pytest.mark.parametrize("spec_origin", ["spec", "policy", None])
    @pytest.mark.parametrize("interface", ["module"])
    def test_additivegaussian_sd(
        self,
        device,
        spec_origin,
        interface,
        d_obs=4,
        d_act=6,
        batch=32,
        n_steps=100,
        seed=0,
    ):
        if interface == "module" and spec_origin != "spec":
            pytest.skip("module raises an error if given spec=None")

        torch.manual_seed(seed)
        action_spec = Bounded(
            -torch.ones(d_act, device=device),
            torch.ones(d_act, device=device),
            (d_act,),
            device=device,
        )
        if interface == "module":
            exploratory_policy = AdditiveGaussianModule(action_spec, device=device)
        else:
            net = nn.Sequential(
                nn.Linear(d_obs, 2 * d_act, device=device), NormalParamExtractor()
            )
            module = SafeModule(
                net,
                in_keys=["observation"],
                out_keys=["loc", "scale"],
                spec=None,
            )
            policy = ProbabilisticActor(
                spec=Composite(action=action_spec) if spec_origin is not None else None,
                module=module,
                in_keys=["loc", "scale"],
                distribution_class=TanhNormal,
                default_interaction_type=InteractionType.RANDOM,
            )
            given_spec = action_spec if spec_origin == "spec" else None
            exploratory_policy = TensorDictModule(
                policy, AdditiveGaussianModule(spec=given_spec, device=device)
            )
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
            class_name = (
                "AdditiveGaussianModule"
                if interface == "module"
                else "AdditiveGaussianWrapper"
            )
            with pytest.raises(
                RuntimeError,
                match=f"the action spec must be provided to {class_name}",
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

    @pytest.mark.parametrize("spec_origin", ["spec", "policy", None])
    @pytest.mark.parametrize("interface", ["module"])
    def test_additivegaussian(
        self,
        device,
        spec_origin,
        interface,
        d_obs=4,
        d_act=6,
        batch=32,
        n_steps=100,
        seed=0,
    ):
        if interface == "module" and spec_origin != "spec":
            pytest.skip("module raises an error if given spec=None")

        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(d_obs, 2 * d_act), NormalParamExtractor()).to(
            device
        )
        module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        action_spec = Bounded(
            -torch.ones(d_act, device=device),
            torch.ones(d_act, device=device),
            (d_act,),
            device=device,
        )
        policy = ProbabilisticActor(
            spec=action_spec if spec_origin is not None else None,
            module=module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            default_interaction_type=InteractionType.RANDOM,
        ).to(device)
        given_spec = action_spec if spec_origin == "spec" else None
        if interface == "module":
            exploratory_policy = TensorDictSequential(
                policy, AdditiveGaussianModule(spec=given_spec).to(device)
            )
        else:
            raise NotImplementedError

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

    @pytest.mark.parametrize("parallel_spec", [True, False])
    @pytest.mark.parametrize("interface", ["module"])
    def test_collector(self, device, parallel_spec, interface, seed=0):
        torch.manual_seed(seed)
        env = SerialEnv(
            2,
            ContinuousActionVecMockEnv,
        )
        env = env.to(device)
        # the module must work with the action spec of a single env or a serial env
        if parallel_spec:
            action_spec = env.action_spec
        else:
            action_spec = ContinuousActionVecMockEnv(device=device).action_spec
        d_act = action_spec.shape[-1]
        net = nn.Sequential(nn.LazyLinear(2 * d_act), NormalParamExtractor()).to(device)
        module = SafeModule(
            net,
            in_keys=["observation"],
            out_keys=["loc", "scale"],
        )
        policy = ProbabilisticActor(
            module=module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            default_interaction_type=InteractionType.RANDOM,
            spec=action_spec,
        ).to(device)
        if interface == "module":
            exploratory_policy = TensorDictSequential(
                policy, AdditiveGaussianModule(spec=action_spec).to(device)
            )
        else:
            raise NotImplementedError
        exploratory_policy(env.reset())
        collector = Collector(
            create_env_fn=env,
            policy=exploratory_policy,
            frames_per_batch=100,
            total_frames=1000,
            device=device,
        )
        for _ in collector:
            # check that we can run the policy
            pass
        return

    def test_no_spec_error(self, device):
        # Test that forward() raises error if spec is None and not set via setter
        module = AdditiveGaussianModule(spec=None, device=device)
        action = torch.randn(10, 6, device=device)
        with pytest.raises(RuntimeError, match="spec has not been set"):
            module._add_noise(action)

    def test_delayed_spec_initialization(self, device):
        """Test that spec can be set via property setter after initialization."""
        torch.manual_seed(0)
        d_act = 6
        action_spec = Bounded(
            -torch.ones(d_act, device=device),
            torch.ones(d_act, device=device),
            (d_act,),
            device=device,
        )

        module = AdditiveGaussianModule(spec=None, device=device)
        assert module._spec is None

        # Set spec via property setter
        module.spec = action_spec
        assert module._spec is not None

        # Verify it works with forward
        action = torch.randn(100, d_act, device=device)
        noisy_action = module._add_noise(action)
        assert action_spec.is_in(noisy_action)


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("use_batched_env", [False, True])
def test_set_exploration_modules_spec_from_env(device, use_batched_env):
    """Test set_exploration_modules_spec_from_env helper configures exploration modules."""
    from tensordict.nn import TensorDictSequential
    from torchrl.modules.tensordict_module.exploration import (
        set_exploration_modules_spec_from_env,
    )

    torch.manual_seed(0)

    if use_batched_env:
        env = SerialEnv(2, ContinuousActionVecMockEnv)
        env = env.to(device)
        expected_spec = env.action_spec_unbatched
    else:
        env = ContinuousActionVecMockEnv(device=device)
        expected_spec = env.action_spec
    env.reset()

    d_obs = env.observation_spec["observation"].shape[-1]
    d_act = expected_spec.shape[-1]

    # Create a policy with exploration module that has spec=None
    net = nn.Sequential(
        nn.Linear(d_obs, 2 * d_act, device=device), NormalParamExtractor()
    )
    module = SafeModule(
        net,
        in_keys=["observation"],
        out_keys=["loc", "scale"],
    )
    policy = ProbabilisticActor(
        module=module,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=TanhNormal,
        default_interaction_type=InteractionType.RANDOM,
    ).to(device)
    exploration_module = AdditiveGaussianModule(spec=None, device=device)
    exploratory_policy = TensorDictSequential(policy, exploration_module)

    assert exploration_module._spec is None

    set_exploration_modules_spec_from_env(exploratory_policy, env)

    # Verify spec is set after configuration and matches the environment's action_spec
    assert exploration_module._spec is not None
    if isinstance(exploration_module._spec, Composite):
        assert exploration_module._spec[exploration_module.action_key] == expected_spec
    else:
        assert exploration_module._spec == expected_spec

    td = env.reset()
    result = exploratory_policy(td)
    assert "action" in result.keys()
    env.close()


@pytest.mark.parametrize("state_dim", [7])
@pytest.mark.parametrize("action_dim", [5, 11])
@pytest.mark.parametrize("gSDE", [True, False])
@pytest.mark.parametrize("safe", [True, False])
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize(
    "exploration_type", [InteractionType.RANDOM, InteractionType.DETERMINISTIC]
)
def test_gsde(
    state_dim, action_dim, gSDE, device, safe, exploration_type, batch=16, bound=0.1
):
    torch.manual_seed(0)
    if gSDE:
        model = torch.nn.LazyLinear(action_dim, device=device)
        in_keys = ["observation"]
        module = SafeSequential(
            SafeModule(model, in_keys=in_keys, out_keys=["action"]),
            SafeModule(
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
        wrapper = nn.Sequential(model, NormalParamExtractor())
        module = SafeModule(wrapper, in_keys=in_keys, out_keys=["loc", "scale"])
        distribution_class = TanhNormal
        distribution_kwargs = {"low": -bound, "high": bound}
    spec = Bounded(
        -torch.ones(action_dim) * bound, torch.ones(action_dim) * bound, (action_dim,)
    ).to(device)

    actor = ProbabilisticActor(
        module=module,
        spec=spec,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        distribution_class=distribution_class,
        distribution_kwargs=distribution_kwargs,
        default_interaction_type=exploration_type,
        safe=safe,
    )

    td = TensorDict(
        {"observation": torch.randn(batch, state_dim, device=device)},
        [batch],
        device=device,
    )
    if gSDE:
        td_reset = td.empty()
        gsde = gSDENoise(shape=[batch], reset_key="_reset").to(device)
        gsde._reset(td, td_reset)
        td.update(td_reset)
        assert "_eps_gSDE" in td.keys()
        assert td.get("_eps_gSDE").device == device
    actor(td)
    assert "action" in td.keys()
    if not safe and gSDE:
        assert not spec.is_in(td.get("action"))
    elif safe and gSDE:
        assert spec.is_in(td.get("action"))

    if not safe:
        with set_exploration_type(exploration_type):
            action1 = module(td).get("action")
        action2 = actor(td.exclude("action")).get("action")
        if gSDE or exploration_type in (
            InteractionType.DETERMINISTIC,
            InteractionType.MODE,
        ):
            torch.testing.assert_close(action1, action2)
        else:
            with pytest.raises(AssertionError):
                torch.testing.assert_close(action1, action2)


@pytest.mark.parametrize("state_dim", [(5,), (12,), (12, 3)])
@pytest.mark.parametrize("action_dim", [5, 12])
@pytest.mark.parametrize("mean", [0, -2])
@pytest.mark.parametrize("std", [1, 2])
@pytest.mark.parametrize("sigma_init", [None, 1.5, 3])
@pytest.mark.parametrize("learn_sigma", [False, True])
@pytest.mark.parametrize("device", get_default_devices())
def test_gsde_init(sigma_init, state_dim, action_dim, mean, std, device, learn_sigma):
    torch.manual_seed(0)
    state = torch.randn(10000, *state_dim, device=device) * std + mean
    action = torch.randn(10000, *state_dim[:-1], action_dim, device=device)
    # lazy
    gsde_lazy = LazygSDEModule(sigma_init=sigma_init, learn_sigma=learn_sigma).to(
        device
    )
    _eps = torch.randn(10000, *state_dim[:-1], action_dim, state_dim[-1], device=device)
    with set_exploration_type(InteractionType.RANDOM):
        mu, sigma, action_out, _eps = gsde_lazy(action, state, _eps)
    sigma_init = sigma_init if sigma_init else 1.0
    assert (
        abs(sigma_init - sigma.mean()) < 0.3
    ), f"failed: mean={mean}, std={std}, sigma_init={sigma_init}, actual: {sigma.mean()}"


class TestConsistentDropout:
    @pytest.mark.parametrize("dropout_p", [0.0, 0.1, 0.5])
    @pytest.mark.parametrize("parallel_spec", [False, True])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_consistent_dropout(self, dropout_p, parallel_spec, device):
        """

        This preliminary test seeks to ensure two things for both
        ConsistentDropout and ConsistentDropoutModule:
        1. Rollout transitions generate a dropout mask as desired.
            - We can easily verify the existence of a mask
        2. The dropout mask is correctly applied.
            - We will check with stochastic policies whether or not
            the loc and scale are the same.
        """
        torch.manual_seed(0)

        # NOTE: Please only put a module with one dropout layer.
        # That's how this test is constructed anyways.
        @torch.no_grad()
        def inner_verify_routine(module, env):
            # Perform transitions.
            collector = Collector(
                create_env_fn=env,
                policy=module,
                frames_per_batch=1,
                total_frames=10,
                device=device,
            )
            for frames in collector:
                masks = [
                    (key, value)
                    for key, value in frames.items()
                    if key.startswith("mask_")
                ]
                # Assert rollouts do indeed correctly generate the masks.
                assert len(masks) == 1, (
                    "Expected exactly ONE mask since we only put "
                    f"one dropout module, got {len(masks)}."
                )

                # Verify that the result for this batch is the same.
                # Kind of Monte Carlo, to be honest.
                sentinel_mask = masks[0][1].clone()
                sentinel_outputs = frames.select("loc", "scale").clone()

                desired_dropout_mask = torch.full_like(
                    sentinel_mask, 1 / (1 - dropout_p)
                )
                desired_dropout_mask[sentinel_mask == 0.0] = 0.0
                # As of 15/08/24, :meth:`~torch.nn.functional.dropout`
                # is being used. Never hurts to be safe.
                assert torch.allclose(
                    sentinel_mask, desired_dropout_mask
                ), "Dropout was not scaled properly."

                new_frames = module(frames.clone())
                infer_mask = new_frames[masks[0][0]]
                infer_outputs = new_frames.select("loc", "scale")
                assert (infer_mask == sentinel_mask).all(), "Mask does not match"

                assert all(
                    [
                        torch.allclose(infer_outputs[key], sentinel_outputs[key])
                        for key in ("loc", "scale")
                    ]
                ), (
                    "Outputs do not match:\n "
                    f"{infer_outputs['loc']}\n--- vs ---\n{sentinel_outputs['loc']}"
                    f"{infer_outputs['scale']}\n--- vs ---\n{sentinel_outputs['scale']}"
                )

        env = SerialEnv(
            2,
            ContinuousActionVecMockEnv,
        )
        env = TransformedEnv(env.to(device), InitTracker())
        env = env.to(device)
        # the module must work with the action spec of a single env or a serial env
        if parallel_spec:
            action_spec = env.action_spec
        else:
            action_spec = ContinuousActionVecMockEnv(device=device).action_spec
        d_act = action_spec.shape[-1]

        # NOTE: Please only put a module with one dropout layer.
        # That's how this test is constructed anyways.
        module_td_seq = TensorDictSequential(
            TensorDictModule(
                nn.LazyLinear(2 * d_act), in_keys=["observation"], out_keys=["out"]
            ),
            ConsistentDropoutModule(p=dropout_p, in_keys="out"),
            TensorDictModule(
                NormalParamExtractor(), in_keys=["out"], out_keys=["loc", "scale"]
            ),
        )

        policy_td_seq = ProbabilisticActor(
            module=module_td_seq,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            default_interaction_type=InteractionType.RANDOM,
            spec=action_spec,
        ).to(device)

        # Wake up the policies
        policy_td_seq(env.reset())

        # Test.
        inner_verify_routine(policy_td_seq, env)

    def test_consistent_dropout_primer(self):
        import torch

        from tensordict.nn import TensorDictModule as Mod, TensorDictSequential as Seq
        from torchrl.envs import SerialEnv, StepCounter
        from torchrl.modules import ConsistentDropoutModule, get_primers_from_module

        torch.manual_seed(0)

        m = Seq(
            Mod(
                torch.nn.Linear(7, 4),
                in_keys=["observation"],
                out_keys=["intermediate"],
            ),
            ConsistentDropoutModule(
                p=0.5,
                input_shape=(
                    2,
                    4,
                ),
                in_keys="intermediate",
            ),
            Mod(torch.nn.Linear(4, 7), in_keys=["intermediate"], out_keys=["action"]),
        )
        primer = get_primers_from_module(m)
        env0 = ContinuousActionVecMockEnv().append_transform(StepCounter(5))
        env1 = ContinuousActionVecMockEnv().append_transform(StepCounter(6))
        env = SerialEnv(2, [lambda env=env0: env, lambda env=env1: env])
        env = env.append_transform(primer)
        r = env.rollout(10, m, break_when_any_done=False)
        mask = [k for k in r.keys() if k.startswith("mask")][0]
        assert (r[mask][0, :5] != r[mask][0, 5:6]).any()
        assert (r[mask][0, :4] == r[mask][0, 4:5]).all()

        assert (r[mask][1, :6] != r[mask][1, 6:7]).any()
        assert (r[mask][1, :5] == r[mask][1, 5:6]).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)


@pytest.mark.parametrize("device", get_default_devices())
class TestNoisyLinear:
    """Tests for NoisyLinear layer based on NoisyNet paper specifications."""

    def test_noisy_linear_initialization(self, device):
        """Test that NoisyLinear initializes with correct parameters."""
        in_features, out_features = 10, 5
        layer = NoisyLinear(
            in_features, out_features, device=device, use_exploration_type=True
        )

        # Check that mu and sigma parameters exist
        assert hasattr(layer, "weight_mu")
        assert hasattr(layer, "weight_sigma")
        assert hasattr(layer, "bias_mu")
        assert hasattr(layer, "bias_sigma")

        # Check parameter shapes
        assert layer.weight_mu.shape == (out_features, in_features)
        assert layer.weight_sigma.shape == (out_features, in_features)
        assert layer.bias_mu.shape == (out_features,)
        assert layer.bias_sigma.shape == (out_features,)

        # Check that sigma values are positive
        assert (layer.weight_sigma > 0).all()
        assert (layer.bias_sigma > 0).all()

        # Check initialization ranges (from paper)
        mu_range = 1 / math.sqrt(in_features)
        assert (layer.weight_mu >= -mu_range).all()
        assert (layer.weight_mu <= mu_range).all()
        assert (layer.bias_mu >= -mu_range).all()
        assert (layer.bias_mu <= mu_range).all()

    def test_noisy_linear_exploration_modes(self, device):
        """Test that NoisyLinear behaves differently based on exploration mode."""
        torch.manual_seed(0)
        # Use use_exploration_type=True to enable exploration_type-based control
        layer = NoisyLinear(10, 5, device=device, use_exploration_type=True)
        x = torch.randn(3, 10, device=device)

        # Get outputs in RANDOM exploration mode (with noise)
        with set_exploration_type(ExplorationType.RANDOM):
            y_random_1 = layer(x)
            layer.reset_noise()  # Reset noise
            y_random_2 = layer(x)

        # Get outputs in DETERMINISTIC mode (no noise)
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            y_det_1 = layer(x)
            layer.reset_noise()  # Reset noise
            y_det_2 = layer(x)

        # Random mode outputs should be different due to noise
        assert not torch.allclose(y_random_1, y_random_2, atol=1e-6)

        # Deterministic outputs should be identical (no noise)
        torch.testing.assert_close(y_det_1, y_det_2)

        # Random and deterministic outputs should be different
        assert not torch.allclose(y_random_1, y_det_1, atol=1e-6)

    def test_noise_consistency_within_episode(self, device):
        """Test that noise remains consistent within an episode (no reset)."""
        torch.manual_seed(0)
        layer = NoisyLinear(10, 5, device=device, use_exploration_type=True)
        x = torch.randn(3, 10, device=device)

        with set_exploration_type(ExplorationType.RANDOM):
            # First forward pass
            y1 = layer(x)

            # Multiple forward passes without resetting noise
            y2 = layer(x)
            y3 = layer(x)
            y4 = layer(x)

        # All outputs should be identical (same noise)
        assert torch.allclose(y1, y2, atol=1e-6)
        assert torch.allclose(y1, y3, atol=1e-6)
        assert torch.allclose(y1, y4, atol=1e-6)

    def test_noise_change_after_reset(self, device):
        """Test that noise changes after reset_noise() is called."""
        torch.manual_seed(0)
        layer = NoisyLinear(10, 5, device=device, use_exploration_type=True)
        x = torch.randn(3, 10, device=device)

        with set_exploration_type(ExplorationType.RANDOM):
            # First episode
            y1 = layer(x)

            # Reset noise (simulating new episode)
            layer.reset_noise()
            y2 = layer(x)

            # Reset noise again
            layer.reset_noise()
            y3 = layer(x)

        # Outputs should be different after each reset
        assert not torch.allclose(y1, y2, atol=1e-6)
        assert not torch.allclose(y1, y3, atol=1e-6)
        assert not torch.allclose(y2, y3, atol=1e-6)

    def test_factorized_gaussian_noise(self, device):
        """Test that the noise follows factorized Gaussian distribution."""
        torch.manual_seed(0)
        layer = NoisyLinear(10, 5, device=device, use_exploration_type=True)

        # Get noise samples
        noise_samples = []
        with set_exploration_type(ExplorationType.RANDOM):
            for _ in range(1000):
                layer.reset_noise()
                # Extract the actual noise used
                weight_noise = layer.weight - layer.weight_mu
                noise_samples.append(weight_noise.flatten())

        noise_samples = torch.stack(noise_samples)

        # Check that noise has approximately zero mean
        assert abs(noise_samples.mean()) < 0.1

        # Check that noise has reasonable variance
        noise_std = noise_samples.std()
        expected_std = layer.std_init / math.sqrt(10)  # Based on initialization
        assert 0.5 * expected_std < noise_std < 2.0 * expected_std

    def test_weight_property_behavior(self, device):
        """Test that weight property returns correct values based on exploration mode."""
        torch.manual_seed(0)
        layer = NoisyLinear(10, 5, device=device, use_exploration_type=True)

        # RANDOM exploration mode - should include noise
        with set_exploration_type(ExplorationType.RANDOM):
            layer.reset_noise()
            weight_random = layer.weight
            bias_random = layer.bias

        # Should include noise
        assert not torch.allclose(weight_random, layer.weight_mu, atol=1e-6)
        assert not torch.allclose(bias_random, layer.bias_mu, atol=1e-6)

        # DETERMINISTIC mode - should be exactly the mean weights
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            weight_det = layer.weight
            bias_det = layer.bias

        # Should be exactly the mean weights
        assert torch.allclose(weight_det, layer.weight_mu, atol=1e-6)
        assert torch.allclose(bias_det, layer.bias_mu, atol=1e-6)

    def test_noisy_linear_in_network(self, device):
        """Test NoisyLinear in a complete network setup."""
        torch.manual_seed(0)

        # Create a simple network with NoisyLinear using new behavior
        network = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            NoisyLinear(20, 5, device=device, use_exploration_type=True),
        ).to(device)

        x = torch.randn(3, 10, device=device)

        # RANDOM exploration mode
        with set_exploration_type(ExplorationType.RANDOM):
            y_random_1 = network(x)
            network[-1].reset_noise()  # Reset noise in NoisyLinear layer
            y_random_2 = network(x)

        # DETERMINISTIC mode
        with set_exploration_type(ExplorationType.DETERMINISTIC):
            y_det_1 = network(x)
            y_det_2 = network(x)

        # Random outputs should be different
        assert not torch.allclose(y_random_1, y_random_2, atol=1e-6)

        # Deterministic outputs should be identical
        assert torch.allclose(y_det_1, y_det_2, atol=1e-6)

    def test_noise_reset_function(self, device):
        """Test the reset_noise utility function."""
        torch.manual_seed(0)

        # Create network with multiple NoisyLinear layers using new behavior
        network = nn.Sequential(
            NoisyLinear(10, 20, device=device, use_exploration_type=True),
            nn.ReLU(),
            NoisyLinear(20, 5, device=device, use_exploration_type=True),
        ).to(device)

        x = torch.randn(3, 10, device=device)

        with set_exploration_type(ExplorationType.RANDOM):
            # First forward pass
            network(x)

            # Reset noise using utility function
            reset_noise(network)
            network(x)

            # Check that at least one of the layers has noise
            changed = False
            for module in network.modules():
                if hasattr(module, "weight_mu"):
                    # Check if the actual weights have noise
                    if not torch.allclose(module.weight, module.weight_mu, atol=1e-6):
                        changed = True
                        break

        # In RANDOM mode, there should be noise
        assert changed, "Expected noise to be present in RANDOM exploration mode"

    def test_noisy_linear_gradients(self, device):
        """Test that gradients flow through NoisyLinear parameters."""
        torch.manual_seed(0)
        layer = NoisyLinear(10, 5, device=device, use_exploration_type=True)

        x = torch.randn(3, 10, device=device, requires_grad=True)

        # Gradients should flow in RANDOM mode
        with set_exploration_type(ExplorationType.RANDOM):
            y = layer(x)
            loss = y.sum()

        # Backward pass
        loss.backward()

        # Check that gradients exist for all parameters
        assert layer.weight_mu.grad is not None
        assert layer.weight_sigma.grad is not None
        assert layer.bias_mu.grad is not None
        assert layer.bias_sigma.grad is not None

        # Check that gradients are not zero
        assert not torch.allclose(
            layer.weight_mu.grad, torch.zeros_like(layer.weight_mu.grad)
        )
        assert not torch.allclose(
            layer.weight_sigma.grad, torch.zeros_like(layer.weight_sigma.grad)
        )

    def test_noisy_linear_parameter_learning(self, device):
        """Test that sigma parameters actually learn during training."""
        torch.manual_seed(0)
        layer = NoisyLinear(10, 5, device=device, use_exploration_type=True)

        # Store initial sigma values
        initial_weight_sigma = layer.weight_sigma.clone()
        initial_bias_sigma = layer.bias_sigma.clone()

        # Simple training loop
        optimizer = torch.optim.Adam(layer.parameters(), lr=0.01)
        x = torch.randn(100, 10, device=device)
        target = torch.randn(100, 5, device=device)

        with set_exploration_type(ExplorationType.RANDOM):
            for _ in range(10):
                optimizer.zero_grad()
                layer.reset_noise()  # Reset noise each iteration
                y = layer(x)
                loss = torch.nn.functional.mse_loss(y, target)
                loss.backward()
                optimizer.step()

        # Check that sigma values have changed
        assert not torch.allclose(layer.weight_sigma, initial_weight_sigma, atol=1e-6)
        assert not torch.allclose(layer.bias_sigma, initial_bias_sigma, atol=1e-6)

    def test_noisy_linear_std_init_effect(self, device):
        """Test that different std_init values affect noise magnitude."""
        torch.manual_seed(0)

        # Create layers with different std_init values using new behavior
        layer_small = NoisyLinear(
            10, 5, std_init=0.01, device=device, use_exploration_type=True
        )
        layer_large = NoisyLinear(
            10, 5, std_init=1.0, device=device, use_exploration_type=True
        )

        x = torch.randn(3, 10, device=device)

        # Get multiple samples to measure noise variance
        noise_samples_small = []
        noise_samples_large = []

        with set_exploration_type(ExplorationType.RANDOM):
            for _ in range(10):
                layer_small.reset_noise()
                layer_large.reset_noise()
                y_small = layer_small(x)
                y_large = layer_large(x)
                noise_samples_small.append(y_small)
                noise_samples_large.append(y_large)

        noise_samples_small = torch.stack(noise_samples_small)
        noise_samples_large = torch.stack(noise_samples_large)

        # Calculate noise variance
        noise_var_small = noise_samples_small.var(dim=0).mean()
        noise_var_large = noise_samples_large.var(dim=0).mean()

        # Large std_init should produce larger noise variance
        assert noise_var_large > noise_var_small

    def test_noisy_linear_serialization(self, device):
        """Test that NoisyLinear can be saved and loaded correctly."""
        torch.manual_seed(0)
        layer = NoisyLinear(10, 5, device=device, use_exploration_type=True)

        # Save and load
        with tempfile.NamedTemporaryFile(delete=False) as f:
            filepath = f.name
        # File is now closed, so we can safely work with it on Windows
        try:
            torch.save(layer.state_dict(), filepath)
            layer_loaded = NoisyLinear(10, 5, device=device, use_exploration_type=True)
            layer_loaded.load_state_dict(torch.load(filepath))
        finally:
            os.unlink(filepath)

        # Check that parameters are the same
        assert torch.allclose(layer.weight_mu, layer_loaded.weight_mu, atol=1e-6)
        assert torch.allclose(layer.weight_sigma, layer_loaded.weight_sigma, atol=1e-6)
        assert torch.allclose(layer.bias_mu, layer_loaded.bias_mu, atol=1e-6)
        assert torch.allclose(layer.bias_sigma, layer_loaded.bias_sigma, atol=1e-6)

    def test_noisy_linear_legacy_behavior(self, device):
        """Test that legacy behavior (using self.training) works with use_exploration_type=False."""
        torch.manual_seed(0)
        # Silence the warning by explicitly opting out
        layer = NoisyLinear(10, 5, device=device, use_exploration_type=False)
        x = torch.randn(3, 10, device=device)

        # Training mode - should use noise
        layer.train()
        y_train_1 = layer(x)
        layer.reset_noise()
        y_train_2 = layer(x)

        # Eval mode - should not use noise
        layer.eval()
        y_eval_1 = layer(x)
        layer.reset_noise()
        y_eval_2 = layer(x)

        # Training outputs should be different (noise is on)
        assert not torch.allclose(y_train_1, y_train_2, atol=1e-6)

        # Eval outputs should be identical (noise is off)
        torch.testing.assert_close(y_eval_1, y_eval_2)

    def test_noisy_linear_deprecation_warning(self, device):
        """Test that FutureWarning is raised when use_exploration_type is None."""
        # Should emit FutureWarning when use_exploration_type is not specified
        with pytest.warns(FutureWarning, match="exploration_type"):
            NoisyLinear(10, 5, device=device)

        # Should NOT emit warning when use_exploration_type is explicitly set
        with warnings.catch_warnings():
            warnings.simplefilter("error", FutureWarning)
            # These should not raise FutureWarning
            NoisyLinear(10, 5, device=device, use_exploration_type=True)
            NoisyLinear(10, 5, device=device, use_exploration_type=False)
