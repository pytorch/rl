# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import os

import pytest
import torch
from scipy.stats import ttest_1samp
from tensordict import TensorDict

from tensordict.nn import InteractionType, TensorDictModule, TensorDictSequential
from torch import nn
from torchrl._utils import _replace_last

from torchrl.collectors import SyncDataCollector
from torchrl.data import Bounded, Categorical, Composite, OneHot
from torchrl.envs import SerialEnv
from torchrl.envs.transforms.transforms import gSDENoise, InitTracker, TransformedEnv
from torchrl.envs.utils import set_exploration_type
from torchrl.modules import SafeModule, SafeSequential
from torchrl.modules.distributions import (
    IndependentNormal,
    NormalParamExtractor,
    TanhNormal,
)
from torchrl.modules.models.exploration import ConsistentDropoutModule, LazygSDEModule
from torchrl.modules.tensordict_module.actors import (
    Actor,
    ProbabilisticActor,
    QValueActor,
)
from torchrl.modules.tensordict_module.exploration import (
    _OrnsteinUhlenbeckProcess,
    AdditiveGaussianModule,
    AdditiveGaussianWrapper,
    EGreedyModule,
    EGreedyWrapper,
    OrnsteinUhlenbeckProcessModule,
    OrnsteinUhlenbeckProcessWrapper,
)

if os.getenv("PYTORCH_TEST_FBCODE"):
    from pytorch.rl.test._utils_internal import get_default_devices
    from pytorch.rl.test.mocking_classes import (
        ContinuousActionVecMockEnv,
        CountingEnvCountModule,
        NestedCountingEnv,
    )
else:
    from _utils_internal import get_default_devices
    from mocking_classes import (
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
        with pytest.raises(KeyError, match="Action mask key action_mask not found in"):
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

    def test_no_spec_error(
        self,
    ):
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
            RuntimeError, match="spec must be provided to the exploration wrapper."
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

    @pytest.mark.parametrize("interface", ["module", "wrapper"])
    def test_ou(
        self, device, interface, d_obs=4, d_act=6, batch=32, n_steps=100, seed=0
    ):
        torch.manual_seed(seed)
        net = nn.Sequential(nn.Linear(d_obs, 2 * d_act), NormalParamExtractor()).to(
            device
        )
        module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        action_spec = Bounded(-torch.ones(d_act), torch.ones(d_act), (d_act,))
        policy = ProbabilisticActor(
            spec=action_spec,
            module=module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            default_interaction_type=InteractionType.RANDOM,
        ).to(device)

        if interface == "module":
            ou = OrnsteinUhlenbeckProcessModule(spec=action_spec).to(device)
            exploratory_policy = TensorDictSequential(policy, ou)
        else:
            exploratory_policy = OrnsteinUhlenbeckProcessWrapper(policy)
            ou = exploratory_policy

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
    @pytest.mark.parametrize("interface", ["module", "wrapper"])
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
                policy, OrnsteinUhlenbeckProcessModule(spec=action_spec).to(device)
            )
        else:
            exploratory_policy = OrnsteinUhlenbeckProcessWrapper(policy)
        exploratory_policy(env.reset())
        collector = SyncDataCollector(
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
    @pytest.mark.parametrize("interface", ["module", "wrapper"])
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
        d_act = action_spec.shape[-1]

        net = nn.LazyLinear(d_act).to(device)
        policy = TensorDictModule(
            CountingEnvCountModule(action_spec=action_spec),
            in_keys=[("data", "states") if nested_obs_action else "observation"],
            out_keys=[env.action_key],
        )
        if interface == "module":
            exploratory_policy = TensorDictSequential(
                policy,
                OrnsteinUhlenbeckProcessModule(
                    spec=action_spec, action_key=env.action_key, is_init_key=is_init_key
                ).to(device),
            )
        else:
            exploratory_policy = OrnsteinUhlenbeckProcessWrapper(
                policy,
                spec=action_spec,
                action_key=env.action_key,
                is_init_key=is_init_key,
            )
        collector = SyncDataCollector(
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
    @pytest.mark.parametrize("interface", ["module", "wrapper"])
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
            exploratory_policy = AdditiveGaussianModule(action_spec).to(device)
        else:
            net = nn.Sequential(nn.Linear(d_obs, 2 * d_act), NormalParamExtractor()).to(
                device
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
            ).to(device)
            given_spec = action_spec if spec_origin == "spec" else None
            exploratory_policy = AdditiveGaussianWrapper(policy, spec=given_spec).to(
                device
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
    @pytest.mark.parametrize("interface", ["module", "wrapper"])
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

    @pytest.mark.parametrize("parallel_spec", [True, False])
    @pytest.mark.parametrize("interface", ["module", "wrapper"])
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
            exploratory_policy = AdditiveGaussianWrapper(policy, safe=False)
        exploratory_policy(env.reset())
        collector = SyncDataCollector(
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
        with pytest.raises(RuntimeError, match="spec cannot be None."):
            AdditiveGaussianModule(spec=None).to(device)


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
@pytest.mark.parametrize(
    "device",
    [torch.device("cuda:0") if torch.cuda.device_count() else torch.device("cpu")],
)
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
            collector = SyncDataCollector(
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
