# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import functools
import itertools
from dataclasses import asdict

import pytest
import torch
from _objectives_common import (
    _has_functorch,
    _has_transformers,
    FUNCTORCH_ERR,
    LossModuleTestBase,
    make_functional_with_buffers,
    MARLEnv,
)

from packaging import version as pack_version
from tensordict import assert_allclose_td, TensorDict
from tensordict.nn import (
    composite_lp_aggregate,
    CompositeDistribution,
    InteractionType,
    NormalParamExtractor,
    ProbabilisticTensorDictModule,
    ProbabilisticTensorDictModule as ProbMod,
    ProbabilisticTensorDictSequential,
    ProbabilisticTensorDictSequential as ProbSeq,
    set_composite_lp_aggregate,
    TensorDictModule,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
    WrapModule,
)
from torch import autograd, nn

from torchrl._utils import rl_warnings
from torchrl.data import Bounded, Composite, Unbounded
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules.models.models import MLP
from torchrl.modules.tensordict_module.actors import (
    ActorValueOperator,
    ProbabilisticActor,
    ValueOperator,
)
from torchrl.objectives import A2CLoss, ClipPPOLoss, KLPENPPOLoss, PPOLoss
from torchrl.objectives.reinforce import ReinforceLoss
from torchrl.objectives.utils import _sum_td_features, ValueEstimators
from torchrl.objectives.value.advantages import (
    GAE,
    TD1Estimator,
    TDLambdaEstimator,
    VTrace,
)

from torchrl.testing import (  # noqa
    call_value_nets as _call_value_nets,
    dtype_fixture,
    get_available_devices,
    get_default_devices,
    PENDULUM_VERSIONED,
)


@pytest.mark.skipif(not _has_transformers, reason="requires transformers lib")
class TestPPO(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        action_key=None,
        observation_key="observation",
        sample_log_prob_key=None,
        composite_action_dist=False,
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = nn.Sequential(nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor())
        if composite_action_dist:
            if action_key is None:
                action_key = ("action", "action1")
            else:
                action_key = (action_key, "action1")
            action_spec = Composite({action_key: {"action1": action_spec}})
            distribution_class = functools.partial(
                CompositeDistribution,
                distribution_map={
                    "action1": TanhNormal,
                },
                name_map={
                    "action1": action_key,
                },
                log_prob_key=sample_log_prob_key,
            )
            module_out_keys = [
                ("params", "action1", "loc"),
                ("params", "action1", "scale"),
            ]
            actor_in_keys = ["params"]
        else:
            if action_key is None:
                action_key = "action"
            distribution_class = TanhNormal
            module_out_keys = actor_in_keys = ["loc", "scale"]
        module = TensorDictModule(
            net, in_keys=[observation_key], out_keys=module_out_keys
        )
        actor = ProbabilisticActor(
            module=module,
            distribution_class=distribution_class,
            in_keys=actor_in_keys,
            out_keys=[action_key],
            spec=action_spec,
            return_log_prob=True,
            log_prob_key=sample_log_prob_key,
        )
        return actor.to(device)

    def _create_mock_value(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        out_keys=None,
        observation_key="observation",
    ):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=[observation_key],
            out_keys=out_keys,
        )
        return value.to(device)

    def _create_mock_actor_value(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        composite_action_dist=False,
        sample_log_prob_key="action_log_prob",
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        if composite_action_dist:
            action_spec = Composite({"action": {"action1": action_spec}})
        base_layer = nn.Linear(obs_dim, 5)
        net = nn.Sequential(
            base_layer, nn.Linear(5, 2 * action_dim), NormalParamExtractor()
        )
        if composite_action_dist:
            distribution_class = functools.partial(
                CompositeDistribution,
                distribution_map={
                    "action1": TanhNormal,
                },
                name_map={
                    "action1": ("action", "action1"),
                },
                log_prob_key=sample_log_prob_key,
            )
            module_out_keys = [
                ("params", "action1", "loc"),
                ("params", "action1", "scale"),
            ]
            actor_in_keys = ["params"]
        else:
            distribution_class = TanhNormal
            module_out_keys = actor_in_keys = ["loc", "scale"]
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=module_out_keys
        )
        actor = ProbabilisticActor(
            module=module,
            distribution_class=distribution_class,
            in_keys=actor_in_keys,
            spec=action_spec,
            return_log_prob=True,
        )
        module = nn.Sequential(base_layer, nn.Linear(5, 1))
        value = ValueOperator(
            module=module,
            in_keys=["observation"],
        )
        return actor.to(device), value.to(device)

    def _create_mock_actor_value_shared(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        composite_action_dist=False,
        sample_log_prob_key="action_log_prob",
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        if composite_action_dist:
            action_spec = Composite({"action": {"action1": action_spec}})
        base_layer = nn.Linear(obs_dim, 5)
        common = TensorDictModule(
            base_layer, in_keys=["observation"], out_keys=["hidden"]
        )
        net = nn.Sequential(nn.Linear(5, 2 * action_dim), NormalParamExtractor())
        if composite_action_dist:
            distribution_class = functools.partial(
                CompositeDistribution,
                distribution_map={
                    "action1": TanhNormal,
                },
                name_map={
                    "action1": ("action", "action1"),
                },
                log_prob_key=sample_log_prob_key,
            )
            module_out_keys = [
                ("params", "action1", "loc"),
                ("params", "action1", "scale"),
            ]
            actor_in_keys = ["params"]
        else:
            distribution_class = TanhNormal
            module_out_keys = actor_in_keys = ["loc", "scale"]
        module = TensorDictModule(net, in_keys=["hidden"], out_keys=module_out_keys)
        actor_head = ProbabilisticActor(
            module=module,
            distribution_class=distribution_class,
            in_keys=actor_in_keys,
            spec=action_spec,
            return_log_prob=True,
        )
        module = nn.Linear(5, 1)
        value_head = ValueOperator(
            module=module,
            in_keys=["hidden"],
        )
        model = ActorValueOperator(common, actor_head, value_head).to(device)
        return model, model.get_policy_operator(), model.get_value_operator()

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=0, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_ppo(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        observation_key="observation",
        action_key="action",
        reward_key="reward",
        done_key="done",
        terminated_key="terminated",
        sample_log_prob_key="action_log_prob",
        composite_action_dist=False,
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        loc_key = "loc"
        scale_key = "scale"
        loc = torch.randn(batch, 4, device=device)
        scale = torch.rand(batch, 4, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                observation_key: obs,
                "next": {
                    observation_key: next_obs,
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward,
                },
                action_key: {"action1": action} if composite_action_dist else action,
                sample_log_prob_key: torch.randn_like(action[..., 1]) / 10,
            },
            device=device,
        )
        if composite_action_dist:
            td[("params", "action1", loc_key)] = loc
            td[("params", "action1", scale_key)] = scale
        else:
            td[loc_key] = loc
            td[scale_key] = scale
        return td

    def _create_seq_mock_data_ppo(
        self,
        batch=2,
        T=4,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        sample_log_prob_key=None,
        action_key=None,
        composite_action_dist=False,
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = torch.ones(batch, T, dtype=torch.bool, device=device)
        action = action.masked_fill_(~mask.unsqueeze(-1), 0.0)
        params_mean = torch.randn_like(action) / 10
        params_scale = torch.rand_like(action) / 10
        loc = params_mean.masked_fill_(~mask.unsqueeze(-1), 0.0)
        scale = params_scale.masked_fill_(~mask.unsqueeze(-1), 0.0)
        if sample_log_prob_key is None:
            if composite_action_dist:
                sample_log_prob_key = ("action", "action1_log_prob")
            else:
                # conforming to composite_lp_aggregate(False)
                sample_log_prob_key = "action_log_prob"

        if action_key is None:
            if composite_action_dist:
                action_key = ("action", "action1")
            else:
                action_key = "action"
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "terminated": terminated,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                action_key: action,
                sample_log_prob_key: (
                    torch.randn_like(action[..., 1]) / 10
                ).masked_fill_(~mask, 0.0),
            },
            device=device,
            names=[None, "time"],
        )
        if composite_action_dist:
            td[("params", "action1", "loc")] = loc
            td[("params", "action1", "scale")] = scale
        else:
            td["loc"] = loc
            td["scale"] = scale

        return td

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    def test_reset_parameters_recursive(self, loss_class):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = loss_class(actor, value)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    @pytest.mark.parametrize("functional", [True, False])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo(
        self,
        loss_class,
        device,
        gradient_mode,
        advantage,
        td_est,
        functional,
        composite_action_dist,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(
            device=device, composite_action_dist=composite_action_dist
        )

        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            functional=functional,
            device=device,
        )
        if composite_action_dist:
            loss_fn.set_keys(
                action=("action", "action1"),
                sample_log_prob=[("action", "action1_log_prob")],
            )
            if advantage is not None:
                advantage.set_keys(sample_log_prob=[("action", "action1_log_prob")])
        if advantage is not None:
            assert not composite_lp_aggregate()
            advantage(td)
        else:
            if td_est is not None:
                loss_fn.make_value_estimator(td_est)

        loss = loss_fn(td)
        if isinstance(loss_fn, KLPENPPOLoss):
            if composite_action_dist:
                kl = loss.pop("kl_approx")
            else:
                kl = loss.pop("kl")
            assert (kl != 0).any()

        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        counter = 0
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert ("actor" in name) or ("target_" in name)
                assert ("critic" not in name) or ("target_" in name)
        assert counter == 2

        value.zero_grad()
        loss_objective.backward()
        counter = 0
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        assert counter == 2
        actor.zero_grad()

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    @pytest.mark.parametrize("functional", [True, False])
    def test_ppo_composite_no_aggregate(
        self, loss_class, device, gradient_mode, advantage, td_est, functional
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(device=device, composite_action_dist=True)

        actor = self._create_mock_actor(
            device=device,
            composite_action_dist=True,
        )
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            functional=functional,
            device=device,
        )
        loss_fn.set_keys(
            action=("action", "action1"),
            sample_log_prob=[("action", "action1_log_prob")],
        )
        if advantage is not None:
            advantage.set_keys(sample_log_prob=[("action", "action1_log_prob")])
            advantage(td)
        else:
            if td_est is not None:
                loss_fn.make_value_estimator(td_est)

        loss = loss_fn(td)
        if isinstance(loss_fn, KLPENPPOLoss):
            kl = loss.pop("kl_approx")
            assert (kl != 0).any()
        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        counter = 0
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert ("actor" in name) or ("target_" in name)
                assert ("critic" not in name) or ("target_" in name)
        assert counter == 2

        value.zero_grad()
        loss_objective.backward()
        counter = 0
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        assert counter == 2
        actor.zero_grad()

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("gradient_mode", (True,))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_state_dict(
        self, loss_class, device, gradient_mode, composite_action_dist
    ):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            device=device,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            device=device,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_shared(self, loss_class, device, advantage, composite_action_dist):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(
            device=device, composite_action_dist=composite_action_dist
        )

        actor, value = self._create_mock_actor_value(
            device=device, composite_action_dist=composite_action_dist
        )
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9,
                value_network=value,
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError
        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            separate_losses=True,
            device=device,
        )

        if advantage is not None:
            if composite_action_dist:
                advantage.set_keys(sample_log_prob=[("action", "action1_log_prob")])
            advantage(td)

        if composite_action_dist:
            loss_fn.set_keys(
                action=("action", "action1"),
                sample_log_prob=[("action", "action1_log_prob")],
            )
        loss = loss_fn(td)

        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        counter = 0
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert ("actor" in name) or ("target_" in name)
                assert ("critic" not in name) or ("target_" in name)
        assert counter == 2

        value.zero_grad()
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        counter = 0
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        actor.zero_grad()
        assert counter == 4

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize(
        "advantage",
        (
            "gae",
            "vtrace",
            "td",
            "td_lambda",
        ),
    )
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("separate_losses", [True, False])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_shared_seq(
        self,
        loss_class,
        device,
        advantage,
        separate_losses,
        composite_action_dist,
    ):
        """Tests PPO with shared module with and without passing twice across the common module."""
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(
            device=device, composite_action_dist=composite_action_dist
        )

        model, actor, value = self._create_mock_actor_value_shared(
            device=device, composite_action_dist=composite_action_dist
        )
        value2 = value[-1]  # prune the common module
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9,
                value_network=value,
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
            )
        else:
            raise NotImplementedError
        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            separate_losses=separate_losses,
            entropy_coeff=0.0,
            device=device,
        )

        loss_fn2 = loss_class(
            actor,
            value2,
            loss_critic_type="l2",
            separate_losses=separate_losses,
            entropy_coeff=0.0,
            device=device,
        )

        if advantage is not None:
            if composite_action_dist:
                advantage.set_keys(sample_log_prob=[("action", "action1_log_prob")])
            advantage(td)

        if composite_action_dist:
            loss_fn.set_keys(
                action=("action", "action1"),
                sample_log_prob=[("action", "action1_log_prob")],
            )
            loss_fn2.set_keys(
                action=("action", "action1"),
                sample_log_prob=[("action", "action1_log_prob")],
            )

        loss = loss_fn(td).exclude("entropy")
        if composite_action_dist:
            loss = loss.exclude("composite_entropy")

        sum(val for key, val in loss.items() if key.startswith("loss_")).backward()
        grad = TensorDict(dict(model.named_parameters()), []).apply(
            lambda x: x.grad.clone()
        )
        loss2 = loss_fn2(td).exclude("entropy")
        if composite_action_dist:
            loss2 = loss2.exclude("composite_entropy")

        model.zero_grad()
        sum(val for key, val in loss2.items() if key.startswith("loss_")).backward()
        grad2 = TensorDict(dict(model.named_parameters()), []).apply(
            lambda x: x.grad.clone()
        )
        if composite_action_dist and loss_class is KLPENPPOLoss:
            # KL computation for composite dist is based on randomly
            # sampled data, thus will not be the same.
            # Similarly, objective loss depends on the KL, so ir will
            # not be the same either.
            # Finally, gradients will be different too.
            loss.pop("kl", None)
            loss2.pop("kl", None)
            loss.pop("loss_objective", None)
            loss2.pop("loss_objective", None)
            assert_allclose_td(loss, loss2)
        else:
            assert_allclose_td(loss, loss2)
            assert_allclose_td(grad, grad2)
        model.zero_grad()

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found, {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_diff(
        self, loss_class, device, gradient_mode, advantage, composite_action_dist
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(
            device=device, composite_action_dist=composite_action_dist
        )

        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            device=device,
        )

        params = TensorDict.from_module(loss_fn, as_module=True)

        # fill params with zero
        def zero_param(p):
            if isinstance(p, nn.Parameter):
                p.data.zero_()

        params.apply(zero_param, filter_empty=True)

        # assert len(list(floss_fn.parameters())) == 0
        with params.to_module(loss_fn):
            if advantage is not None:
                if composite_action_dist:
                    advantage.set_keys(sample_log_prob=[("action", "action1_log_prob")])
                advantage(td)
            if composite_action_dist:
                loss_fn.set_keys(
                    action=("action", "action1"),
                    sample_log_prob=[("action", "action1_log_prob")],
                )
            loss = loss_fn(td)

        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        for name, p in params.items(True, True):
            if isinstance(name, tuple):
                name = "-".join(name)
            if not isinstance(p, nn.Parameter):
                continue
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert "actor" in name
                assert "critic" not in name

        for p in params.values(True, True):
            p.grad = None
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        for name, other_p in named_parameters:
            p = params.get(tuple(name.split(".")))
            assert other_p.shape == p.shape
            assert other_p.dtype == p.dtype
            assert other_p.device == p.device
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "critic" in name
        for param in params.values(True, True):
            param.grad = None

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize(
        "td_est",
        [
            ValueEstimators.TD1,
            ValueEstimators.TD0,
            ValueEstimators.GAE,
            ValueEstimators.VTrace,
            ValueEstimators.TDLambda,
        ],
    )
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_tensordict_keys(self, loss_class, td_est, composite_action_dist):
        assert not composite_lp_aggregate()
        actor = self._create_mock_actor(composite_action_dist=composite_action_dist)
        value = self._create_mock_value()

        loss_fn = loss_class(actor, value, loss_critic_type="l2")

        default_keys = {
            "advantage": "advantage",
            "value_target": "value_target",
            "value": "state_value",
            "sample_log_prob": "action_log_prob"
            if not composite_action_dist
            else ("action", "action1_log_prob"),
            "action": "action" if not composite_action_dist else ("action", "action1"),
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value_key = "state_value_test"
        value = self._create_mock_value(out_keys=[value_key])
        loss_fn = loss_class(actor, value, loss_critic_type="l2")

        key_mapping = {
            "advantage": ("advantage", "advantage_new"),
            "value_target": ("value_target", "value_target_new"),
            "value": ("value", value_key),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_ppo_tensordict_keys_run(self, loss_class, advantage, td_est):
        """Test PPO loss module with non-default tensordict keys."""
        torch.manual_seed(self.seed)
        gradient_mode = True
        tensor_keys = {
            "advantage": "advantage_test",
            "value_target": "value_target_test",
            "value": "state_value_test",
            "sample_log_prob": "action_log_prob_test",
            "action": "action_test",
        }

        td = self._create_seq_mock_data_ppo(
            sample_log_prob_key=tensor_keys["sample_log_prob"],
            action_key=tensor_keys["action"],
        )
        actor = self._create_mock_actor(
            sample_log_prob_key=tensor_keys["sample_log_prob"],
            action_key=tensor_keys["action"],
        )
        value = self._create_mock_value(out_keys=[tensor_keys["value"]])

        if advantage == "gae":
            advantage = GAE(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
                differentiable=gradient_mode,
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9,
                value_network=value,
                differentiable=gradient_mode,
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
                differentiable=gradient_mode,
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = loss_class(actor, value, loss_critic_type="l2")
        loss_fn.set_keys(**tensor_keys)
        if advantage is not None:
            # collect tensordict key names for the advantage module
            adv_keys = {
                key: value
                for key, value in tensor_keys.items()
                if key in asdict(GAE._AcceptedKeys()).keys()
            }
            advantage.set_keys(**adv_keys)
            advantage(td)
        else:
            if td_est is not None:
                loss_fn.make_value_estimator(td_est)

        loss = loss_fn(td)

        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        counter = 0
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert ("actor" in name) or ("target" in name)
                assert ("critic" not in name) or ("target" in name)
        assert counter == 2

        value.zero_grad()
        loss_objective.backward()
        counter = 0
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                counter += 1
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target" in name)
                assert ("critic" in name) or ("target" in name)
        assert counter == 2
        actor.zero_grad()

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("sample_log_prob_key", ["samplelogprob", "samplelogprob2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    @pytest.mark.parametrize(
        "composite_action_dist",
        [False],
    )
    def test_ppo_notensordict(
        self,
        loss_class,
        action_key,
        sample_log_prob_key,
        observation_key,
        reward_key,
        done_key,
        terminated_key,
        composite_action_dist,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_ppo(
            observation_key=observation_key,
            action_key=action_key,
            sample_log_prob_key=sample_log_prob_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
            composite_action_dist=composite_action_dist,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key,
            sample_log_prob_key=sample_log_prob_key,
            composite_action_dist=composite_action_dist,
            action_key=action_key,
        )
        value = self._create_mock_value(observation_key=observation_key)

        loss = loss_class(actor_network=actor, critic_network=value)
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
            sample_log_prob=sample_log_prob_key,
        )

        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            sample_log_prob_key: td.get(sample_log_prob_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
        }
        if loss_class is KLPENPPOLoss:
            loc_key = "params" if composite_action_dist else "loc"
            scale_key = "params" if composite_action_dist else "scale"
            kwargs.update({loc_key: td.get(loc_key), scale_key: td.get(scale_key)})

        td = TensorDict(kwargs, td.batch_size, names=["time"]).unflatten_keys("_")

        # setting the seed for each loss so that drawing the random samples from
        # value network leads to same numbers for both runs
        torch.manual_seed(self.seed)
        beta = getattr(loss, "beta", None)
        if beta is not None:
            beta = beta.clone()
        loss_val = loss(**kwargs)
        torch.manual_seed(self.seed)
        if beta is not None:

            loss.beta = beta.clone()
        loss_val_td = loss(td)

        for i, out_key in enumerate(loss.out_keys):
            torch.testing.assert_close(
                loss_val_td.get(out_key), loss_val[i], msg=out_key
            )

        # test select
        torch.manual_seed(self.seed)
        if beta is not None:
            loss.beta = beta.clone()
        loss.select_out_keys("loss_objective", "loss_critic")
        if torch.__version__ >= "2.0.0":
            loss_obj, loss_crit = loss(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_obj, loss_crit = loss(**kwargs)
            return
        assert loss_obj == loss_val_td.get("loss_objective")
        assert loss_crit == loss_val_td.get("loss_critic")

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_reduction(self, reduction, loss_class, composite_action_dist):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_seq_mock_data_ppo(
            device=device, composite_action_dist=composite_action_dist
        )
        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        advantage = GAE(
            gamma=0.9,
            lmbda=0.9,
            value_network=value,
        )
        loss_fn = loss_class(
            actor,
            value,
            loss_critic_type="l2",
            reduction=reduction,
        )
        advantage(td)
        if composite_action_dist:
            loss_fn.set_keys(
                action=("action", "action1"),
                sample_log_prob=[("action", "action1_log_prob")],
            )
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss_"):
                    assert loss[key].shape == td.shape, key
        else:
            for key in loss.keys():
                if not key.startswith("loss_"):
                    continue
                assert loss[key].shape == torch.Size([])

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("clip_value", [True, False, None, 0.5, torch.tensor(0.5)])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_ppo_value_clipping(
        self, clip_value, loss_class, device, composite_action_dist
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(
            device=device, composite_action_dist=composite_action_dist
        )
        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        advantage = GAE(
            gamma=0.9,
            lmbda=0.9,
            value_network=value,
        )

        if isinstance(clip_value, bool) and loss_class is not ClipPPOLoss:
            with pytest.raises(
                ValueError,
                match=f"clip_value must be a float or a scalar tensor, got {clip_value}.",
            ):
                loss_fn = loss_class(
                    actor,
                    value,
                    loss_critic_type="l2",
                    clip_value=clip_value,
                    device=device,
                )

        else:
            loss_fn = loss_class(
                actor,
                value,
                loss_critic_type="l2",
                clip_value=clip_value,
                device=device,
            )
            advantage(td)
            if composite_action_dist:
                loss_fn.set_keys(
                    action=("action", "action1"),
                    sample_log_prob=[("action", "action1_log_prob")],
                )

            value = td.pop(loss_fn.tensor_keys.value)

            if clip_value:
                # Test it fails without value key
                with pytest.raises(
                    KeyError,
                    match=f"clip_value is set to {loss_fn.clip_value}, but the key "
                    "state_value was not found in the input tensordict. "
                    "Make sure that the.*passed to PPO exists in "
                    "the input tensordict.",
                ):
                    loss = loss_fn(td)

            # Add value back to td
            td.set(loss_fn.tensor_keys.value, value)

            # Test it works with value key
            loss = loss_fn(td)
            assert "loss_critic" in loss.keys()

    def test_ppo_composite_dists(self):
        d = torch.distributions

        make_params = TensorDictModule(
            lambda: (
                torch.ones(4),
                torch.ones(4),
                torch.ones(4),
                torch.ones(4),
                torch.ones(4),
                torch.ones(4),
                torch.ones(4, 2),
                torch.ones(4, 2),
                torch.ones(4, 10) / 10,
                torch.zeros(4, 10),
                torch.ones(4, 10),
            ),
            in_keys=[],
            out_keys=[
                ("params", "gamma1", "concentration"),
                ("params", "gamma1", "rate"),
                ("params", "gamma2", "concentration"),
                ("params", "gamma2", "rate"),
                ("params", "gamma3", "concentration"),
                ("params", "gamma3", "rate"),
                ("params", "Kumaraswamy", "concentration0"),
                ("params", "Kumaraswamy", "concentration1"),
                ("params", "mixture", "logits"),
                ("params", "mixture", "loc"),
                ("params", "mixture", "scale"),
            ],
        )

        def mixture_constructor(logits, loc, scale):
            return d.MixtureSameFamily(
                d.Categorical(logits=logits), d.Normal(loc=loc, scale=scale)
            )

        dist_constructor = functools.partial(
            CompositeDistribution,
            distribution_map={
                "gamma1": d.Gamma,
                "gamma2": d.Gamma,
                "gamma3": d.Gamma,
                "Kumaraswamy": d.Kumaraswamy,
                "mixture": mixture_constructor,
            },
            name_map={
                "gamma1": ("agent0", "action", "action1", "sub_action1"),
                "gamma2": ("agent0", "action", "action1", "sub_action2"),
                "gamma3": ("agent0", "action", "action2"),
                "Kumaraswamy": ("agent1", "action"),
                "mixture": ("agent2"),
            },
        )
        policy = ProbSeq(
            make_params,
            ProbabilisticTensorDictModule(
                in_keys=["params"],
                out_keys=[
                    ("agent0", "action", "action1", "sub_action1"),
                    ("agent0", "action", "action1", "sub_action2"),
                    ("agent0", "action", "action2"),
                    ("agent1", "action"),
                    ("agent2"),
                ],
                distribution_class=dist_constructor,
                return_log_prob=True,
                default_interaction_type=InteractionType.RANDOM,
            ),
        )
        # We want to make sure there is no warning
        td = policy(TensorDict(batch_size=[4]))
        assert isinstance(
            policy.get_dist(td).log_prob(td),
            TensorDict,
        )
        assert isinstance(
            policy.log_prob(td),
            TensorDict,
        )
        value_operator = Seq(
            WrapModule(
                lambda td: td.set("state_value", torch.ones((*td.shape, 1))),
                out_keys=["state_value"],
            )
        )
        for cls in (PPOLoss, ClipPPOLoss, KLPENPPOLoss):
            data = policy(TensorDict(batch_size=[4]))
            data.set(
                "next",
                TensorDict(
                    reward=torch.randn(4, 1), done=torch.zeros(4, 1, dtype=torch.bool)
                ),
            )
            scalar_entropy = 0.07
            ppo = cls(policy, value_operator, entropy_coeff=scalar_entropy)
            ppo.set_keys(
                action=[
                    ("agent0", "action", "action1", "sub_action1"),
                    ("agent0", "action", "action1", "sub_action2"),
                    ("agent0", "action", "action2"),
                    ("agent1", "action"),
                    ("agent2"),
                ],
                sample_log_prob=[
                    ("agent0", "action", "action1", "sub_action1_log_prob"),
                    ("agent0", "action", "action1", "sub_action2_log_prob"),
                    ("agent0", "action", "action2_log_prob"),
                    ("agent1", "action_log_prob"),
                    ("agent2_log_prob"),
                ],
            )
            loss = ppo(data)
            composite_entropy = loss["composite_entropy"]
            entropy = _sum_td_features(composite_entropy)
            expected_loss = -(scalar_entropy * entropy).mean()  # batch mean
            torch.testing.assert_close(
                loss["loss_entropy"], expected_loss, rtol=1e-5, atol=1e-7
            )
            loss.sum(reduce=True)

            # keep per-head entropies instead of the aggregated tensor
            set_composite_lp_aggregate(False).set()
            coef_map = {
                ("agent0", "action", "action1", "sub_action1_log_prob"): 0.02,
                ("agent0", "action", "action1", "sub_action2_log_prob"): 0.01,
                ("agent0", "action", "action2_log_prob"): 0.01,
                ("agent1", "action_log_prob"): 0.01,
                "agent2_log_prob": 0.01,
            }
            ppo_weighted = cls(policy, value_operator, entropy_coeff=coef_map)
            ppo_weighted.set_keys(
                action=[
                    ("agent0", "action", "action1", "sub_action1"),
                    ("agent0", "action", "action1", "sub_action2"),
                    ("agent0", "action", "action2"),
                    ("agent1", "action"),
                    ("agent2"),
                ],
                sample_log_prob=[
                    ("agent0", "action", "action1", "sub_action1_log_prob"),
                    ("agent0", "action", "action1", "sub_action2_log_prob"),
                    ("agent0", "action", "action2_log_prob"),
                    ("agent1", "action_log_prob"),
                    ("agent2_log_prob"),
                ],
            )
            loss = ppo_weighted(data)
            composite_entropy = loss["composite_entropy"]

            # sanity check: loss_entropy is scalar + finite
            assert loss["loss_entropy"].ndim == 0
            assert torch.isfinite(loss["loss_entropy"])
            # Check individual loss is computed with the right weights
            expected_loss = 0.0
            for i, (_, head_entropy) in enumerate(
                composite_entropy.items(include_nested=True, leaves_only=True)
            ):
                expected_loss -= (
                    coef_map[list(coef_map.keys())[i]] * head_entropy
                ).mean()
            torch.testing.assert_close(
                loss["loss_entropy"], expected_loss, rtol=1e-5, atol=1e-7
            )

    def test_ppo_marl_aggregate(self):
        env = MARLEnv()

        def primer(td):
            params = TensorDict(
                agents=TensorDict(
                    dirich=TensorDict(
                        concentration=env.action_spec["agents", "dirich"].one()
                    ),
                    categ=TensorDict(logits=env.action_spec["agents", "categ"].one()),
                    batch_size=env.action_spec["agents"].shape,
                ),
                batch_size=td.batch_size,
            )
            td.set("params", params)
            return td

        policy = ProbabilisticTensorDictSequential(
            primer,
            env.make_composite_dist(),
            # return_composite=True,
        )
        output = policy(env.fake_tensordict())
        assert output.shape == env.batch_size
        assert (
            output["agents", "dirich_log_prob"].shape == env.batch_size + env.n_agents
        )
        assert output["agents", "categ_log_prob"].shape == env.batch_size + env.n_agents

        output["advantage"] = output["next", "agents", "reward"].clone()
        output["value_target"] = output["next", "agents", "reward"].clone()
        critic = TensorDictModule(
            lambda obs: obs.new_zeros((*obs.shape[:-1], 1)),
            in_keys=list(env.full_observation_spec.keys(True, True)),
            out_keys=["state_value"],
        )
        ppo = ClipPPOLoss(actor_network=policy, critic_network=critic)
        ppo.set_keys(action=list(env.full_action_spec.keys(True, True)))
        assert isinstance(ppo.tensor_keys.action, list)
        ppo(output)

    def _make_entropy_loss(self, entropy_coeff):
        actor, critic = self._create_mock_actor_value()
        return PPOLoss(actor, critic, entropy_coeff=entropy_coeff)

    def test_weighted_entropy_scalar(self):
        loss = self._make_entropy_loss(entropy_coeff=0.5)
        entropy = torch.tensor(2.0)
        out = loss._weighted_loss_entropy(entropy)
        torch.testing.assert_close(out, torch.tensor(-1.0))

    def test_weighted_entropy_mapping(self):
        coef = {("head_0", "action_log_prob"): 0.3, ("head_1", "action_log_prob"): 0.7}
        loss = self._make_entropy_loss(entropy_coeff=coef)
        entropy = TensorDict(
            {
                "head_0": {"action_log_prob": torch.tensor(1.0)},
                "head_1": {"action_log_prob": torch.tensor(2.0)},
            },
            [],
        )
        out = loss._weighted_loss_entropy(entropy)
        expected = -(
            coef[("head_0", "action_log_prob")] * 1.0
            + coef[("head_1", "action_log_prob")] * 2.0
        )
        torch.testing.assert_close(out, torch.tensor(expected))

    def test_weighted_entropy_mapping_missing_key(self):
        loss = self._make_entropy_loss(entropy_coeff={"head_not_present": 0.5})
        entropy = TensorDict({"head_0": {"action_log_prob": torch.tensor(1.0)}}, [])
        with pytest.raises(KeyError):
            loss._weighted_loss_entropy(entropy)

    def test_critic_loss_tensordict(self):
        # Creates a dummy actor.
        actor, _ = self._create_mock_actor_value()

        # Creates a critic that produces a tensordict of values.
        class CompositeValueNetwork(nn.Module):
            def forward(self, _) -> tuple[torch.Tensor, torch.Tensor]:
                return torch.tensor([0.0]), torch.tensor([0.0])

        critic = TensorDictModule(
            CompositeValueNetwork(),
            in_keys=["state"],
            out_keys=[("state_value", "value_0"), ("state_value", "value_1")],
        )

        # Creates the loss and its input tensordict.
        loss = ClipPPOLoss(actor, critic, loss_critic_type="l2", clip_value=0.1)
        td = TensorDict(
            {
                "state": torch.tensor([0.0]),
                "value_target": TensorDict(
                    {"value_0": torch.tensor([-1.0]), "value_1": torch.tensor([2.0])}
                ),
                # Log an existing 'state_value' for the 'clip_fraction'
                "state_value": TensorDict(
                    {"value_0": torch.tensor([0.0]), "value_1": torch.tensor([0.0])}
                ),
            },
            batch_size=(1,),
        )

        critic_loss, clip_fraction, explained_variance = loss.loss_critic(td)

        assert isinstance(critic_loss, TensorDict)
        assert "value_0" in critic_loss.keys() and "value_1" in critic_loss.keys()
        torch.testing.assert_close(critic_loss["value_0"], torch.tensor([1.0]))
        torch.testing.assert_close(critic_loss["value_1"], torch.tensor([4.0]))

        assert isinstance(clip_fraction, TensorDict)
        assert isinstance(explained_variance, TensorDict)


class TestA2C(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        action_key="action",
        observation_key="observation",
        composite_action_dist=False,
        sample_log_prob_key=None,
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        if composite_action_dist:
            action_spec = Composite({action_key: {"action1": action_spec}})
        net = nn.Sequential(nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor())
        if composite_action_dist:
            distribution_class = functools.partial(
                CompositeDistribution,
                distribution_map={
                    "action1": TanhNormal,
                },
                name_map={
                    "action1": (action_key, "action1"),
                },
            )
            module_out_keys = [
                ("params", "action1", "loc"),
                ("params", "action1", "scale"),
            ]
            actor_in_keys = ["params"]
        else:
            distribution_class = TanhNormal
            module_out_keys = actor_in_keys = ["loc", "scale"]
        module = TensorDictModule(
            net, in_keys=[observation_key], out_keys=module_out_keys
        )
        actor = ProbabilisticActor(
            module=module,
            in_keys=actor_in_keys,
            out_keys=[action_key],
            spec=action_spec,
            distribution_class=distribution_class,
            return_log_prob=True,
            log_prob_key=sample_log_prob_key,
        )
        return actor.to(device)

    def _create_mock_value(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        out_keys=None,
        observation_key="observation",
    ):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=[observation_key],
            out_keys=out_keys,
        )
        return value.to(device)

    def _create_mock_common_layer_setup(
        self,
        n_obs=3,
        n_act=4,
        ncells=4,
        batch=2,
        n_hidden=2,
        T=10,
        composite_action_dist=False,
    ):
        common_net = MLP(
            num_cells=ncells,
            in_features=n_obs,
            depth=3,
            out_features=n_hidden,
        )
        actor_net = MLP(
            num_cells=ncells,
            in_features=n_hidden,
            depth=1,
            out_features=2 * n_act,
        )
        value_net = MLP(
            in_features=n_hidden,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        batch = [batch, T]
        action = torch.randn(*batch, n_act)
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": {"action1": action} if composite_action_dist else action,
                "action_log_prob": torch.randn(*batch),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
                "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                "next": {
                    "obs": torch.randn(*batch, n_obs),
                    "reward": torch.randn(*batch, 1),
                    "done": torch.zeros(*batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                },
            },
            batch,
            names=[None, "time"],
        )
        common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])

        if composite_action_dist:
            distribution_class = functools.partial(
                CompositeDistribution,
                distribution_map={
                    "action1": TanhNormal,
                },
                name_map={
                    "action1": ("action", "action1"),
                },
            )
            module_out_keys = [
                ("params", "action1", "loc"),
                ("params", "action1", "scale"),
            ]
            actor_in_keys = ["params"]
        else:
            distribution_class = TanhNormal
            module_out_keys = actor_in_keys = ["loc", "scale"]

        actor = ProbSeq(
            common,
            Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
            Mod(NormalParamExtractor(), in_keys=["param"], out_keys=module_out_keys),
            ProbMod(
                in_keys=actor_in_keys,
                out_keys=["action"],
                distribution_class=distribution_class,
            ),
        )
        critic = Seq(
            common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"])
        )
        actor(td.clone())
        critic(td.clone())
        return actor, critic, common, td

    def _create_seq_mock_data_a2c(
        self,
        batch=2,
        T=4,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        action_key="action",
        observation_key="observation",
        reward_key="reward",
        done_key="done",
        terminated_key="terminated",
        sample_log_prob_key="action_log_prob",
        composite_action_dist=False,
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
        action = action.masked_fill_(~mask.unsqueeze(-1), 0.0)
        params_mean = torch.randn_like(action) / 10
        params_scale = torch.rand_like(action) / 10
        loc = params_mean.masked_fill_(~mask.unsqueeze(-1), 0.0)
        scale = params_scale.masked_fill_(~mask.unsqueeze(-1), 0.0)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                observation_key: obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    observation_key: next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                action_key: {"action1": action} if composite_action_dist else action,
                sample_log_prob_key: torch.randn_like(action[..., 1]).masked_fill_(
                    ~mask, 0.0
                )
                / 10,
            },
            device=device,
            names=[None, "time"],
        )
        if composite_action_dist:
            td[("params", "action1", "loc")] = loc
            td[("params", "action1", "scale")] = scale
        else:
            td["loc"] = loc
            td["scale"] = scale
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = A2CLoss(actor, value)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    @pytest.mark.parametrize("functional", (True, False))
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c(
        self,
        device,
        gradient_mode,
        advantage,
        td_est,
        functional,
        composite_action_dist,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_a2c(
            device=device, composite_action_dist=composite_action_dist
        )

        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = A2CLoss(
            actor,
            value,
            loss_critic_type="l2",
            functional=functional,
        )

        def set_requires_grad(tensor, requires_grad):
            tensor.requires_grad = requires_grad
            return tensor

        # Check error is raised when actions require grads
        if composite_action_dist:
            td["action"].apply_(lambda x: set_requires_grad(x, True))
        else:
            td["action"].requires_grad = True
        with pytest.raises(
            RuntimeError,
            match="tensordict stored action requires grad.",
        ):
            _ = loss_fn._log_probs(td)
        if composite_action_dist:
            td["action"].apply_(lambda x: set_requires_grad(x, False))
        else:
            td["action"].requires_grad = False

        td = td.exclude(loss_fn.tensor_keys.value_target)
        if advantage is not None:
            advantage.set_keys(
                sample_log_prob=actor.log_prob_keys
                if composite_action_dist
                else "action_log_prob"
            )
            advantage(td)
        elif td_est is not None:
            loss_fn.make_value_estimator(td_est)
        loss = loss_fn(td)

        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert ("actor" in name) or ("target_" in name)
                assert ("critic" not in name) or ("target_" in name)

        value.zero_grad()
        for n, p in loss_fn.named_parameters():
            assert p.grad is None or p.grad.norm() == 0, n
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        actor.zero_grad()

        # test reset
        loss_fn.reset()

    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_state_dict(self, device, gradient_mode, composite_action_dist):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        loss_fn = A2CLoss(actor, value, loss_critic_type="l2")
        sd = loss_fn.state_dict()
        loss_fn2 = A2CLoss(actor, value, loss_critic_type="l2")
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("separate_losses", [False, True])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_separate_losses(self, separate_losses, composite_action_dist):
        torch.manual_seed(self.seed)
        actor, critic, common, td = self._create_mock_common_layer_setup(
            composite_action_dist=composite_action_dist
        )
        loss_fn = A2CLoss(
            actor_network=actor,
            critic_network=critic,
            separate_losses=separate_losses,
        )

        def set_requires_grad(tensor, requires_grad):
            tensor.requires_grad = requires_grad
            return tensor

        # Check error is raised when actions require grads
        if composite_action_dist:
            td["action"].apply_(lambda x: set_requires_grad(x, True))
        else:
            td["action"].requires_grad = True
        with pytest.raises(
            RuntimeError,
            match="tensordict stored action requires grad.",
        ):
            _ = loss_fn._log_probs(td)
        if composite_action_dist:
            td["action"].apply_(lambda x: set_requires_grad(x, False))
        else:
            td["action"].requires_grad = False

        td = td.exclude(loss_fn.tensor_keys.value_target)
        loss = loss_fn(td)
        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if separate_losses:
                if p.grad is not None and p.grad.norm() > 0.0:
                    assert "actor" not in name
                    assert "critic" in name
                if p.grad is None:
                    assert ("actor" in name) or ("target_" in name)
                    assert ("critic" not in name) or ("target_" in name)
            else:
                if p.grad is not None and p.grad.norm() > 0.0:
                    assert ("actor" in name) or ("critic" in name)
                if p.grad is None:
                    assert ("actor" in name) or ("critic" in name)

        critic.zero_grad()
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        actor.zero_grad()

        # test reset
        loss_fn.reset()

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found, {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_diff(self, device, gradient_mode, advantage, composite_action_dist):
        if pack_version.parse(torch.__version__) > pack_version.parse("1.14"):
            raise pytest.skip("make_functional_with_buffers needs to be changed")
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_a2c(
            device=device, composite_action_dist=composite_action_dist
        )

        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = A2CLoss(actor, value, loss_critic_type="l2")

        floss_fn, params, buffers = make_functional_with_buffers(loss_fn)

        if advantage is not None:
            advantage(td)
        loss = floss_fn(params, buffers, td)
        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        for (name, _), p in zip(named_parameters, params):
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" not in name
                assert "critic" in name
            if p.grad is None:
                assert ("actor" in name) or ("target_" in name)
                assert ("critic" not in name) or ("target_" in name)

        for param in params:
            param.grad = None
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        for (name, _), p in zip(named_parameters, params):
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        for param in params:
            param.grad = None

    @pytest.mark.parametrize(
        "td_est",
        [
            ValueEstimators.TD1,
            ValueEstimators.TD0,
            ValueEstimators.GAE,
            ValueEstimators.VTrace,
            ValueEstimators.TDLambda,
        ],
    )
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_tensordict_keys(self, td_est, composite_action_dist):
        actor = self._create_mock_actor(composite_action_dist=composite_action_dist)
        value = self._create_mock_value()

        loss_fn = A2CLoss(actor, value, loss_critic_type="l2")

        default_keys = {
            "advantage": "advantage",
            "value_target": "value_target",
            "value": "state_value",
            "action": "action",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
            "sample_log_prob": "action_log_prob",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value = self._create_mock_value(out_keys=["value_state_test"])

        loss_fn = A2CLoss(actor, value, loss_critic_type="l2")

        key_mapping = {
            "advantage": ("advantage", "advantage_test"),
            "value_target": ("value_target", "value_target_test"),
            "value": ("value", "value_state_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize(
        "td_est",
        [
            ValueEstimators.GAE,
            ValueEstimators.VTrace,
        ],
    )
    @pytest.mark.parametrize("advantage", ("gae", "vtrace", None))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_tensordict_keys_run(
        self, device, advantage, td_est, composite_action_dist
    ):
        """Test A2C loss module with non-default tensordict keys."""
        torch.manual_seed(self.seed)
        gradient_mode = True
        advantage_key = "advantage_test"
        value_target_key = "value_target_test"
        value_key = "state_value_test"
        action_key = "action_test"
        reward_key = "reward_test"
        sample_log_prob_key = "action_log_prob_test"
        done_key = ("done", "test")
        terminated_key = ("terminated", "test")

        td = self._create_seq_mock_data_a2c(
            device=device,
            action_key=action_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
            sample_log_prob_key=sample_log_prob_key,
            composite_action_dist=composite_action_dist,
        )

        actor = self._create_mock_actor(
            device=device,
            sample_log_prob_key=sample_log_prob_key,
            composite_action_dist=composite_action_dist,
            action_key=action_key,
        )
        value = self._create_mock_value(device=device, out_keys=[value_key])
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
            )
        elif advantage == "vtrace":
            advantage = VTrace(
                gamma=0.9,
                value_network=value,
                actor_network=actor,
                differentiable=gradient_mode,
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = A2CLoss(actor, value, loss_critic_type="l2")
        loss_fn.set_keys(
            advantage=advantage_key,
            value_target=value_target_key,
            value=value_key,
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=done_key,
            sample_log_prob=sample_log_prob_key,
        )

        if advantage is not None:
            advantage.set_keys(
                advantage=advantage_key,
                value_target=value_target_key,
                value=value_key,
                reward=reward_key,
                done=done_key,
                terminated=terminated_key,
                sample_log_prob=sample_log_prob_key,
            )
            advantage(td)
        else:
            if td_est is not None:
                loss_fn.make_value_estimator(td_est)

        loss = loss_fn(td)
        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" not in name
            if p.grad is None:
                assert ("actor" in name) or ("target_" in name)
                assert ("critic" not in name) or ("target_" in name)

        value.zero_grad()
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert ("actor" not in name) or ("target_" in name)
                assert ("critic" in name) or ("target_" in name)
        actor.zero_grad()

        # test reset
        loss_fn.reset()

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    @pytest.mark.parametrize(
        "composite_action_dist",
        [
            False,
        ],
    )
    def test_a2c_notensordict(
        self,
        action_key,
        observation_key,
        reward_key,
        done_key,
        terminated_key,
        composite_action_dist,
    ):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(
            observation_key=observation_key, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(observation_key=observation_key)
        td = self._create_seq_mock_data_a2c(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
            composite_action_dist=composite_action_dist,
        )

        loss = A2CLoss(actor, value)
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
        )

        kwargs = {
            observation_key: td.get(observation_key),
            f"next_{observation_key}": td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            action_key: td.get(action_key),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        loss_val = loss(**kwargs)
        loss_val_td = loss(td)

        torch.testing.assert_close(loss_val_td.get("loss_objective"), loss_val[0])
        torch.testing.assert_close(loss_val_td.get("loss_critic"), loss_val[1])
        # don't test entropy and loss_entropy, since they depend on a random sample
        # from distribution
        assert len(loss_val) == 4
        # test select
        torch.manual_seed(self.seed)
        loss.select_out_keys("loss_objective", "loss_critic")
        if torch.__version__ >= "2.0.0":
            loss_objective, loss_critic = loss(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_obj, loss_crit = loss(**kwargs)
            return
        assert loss_objective == loss_val_td["loss_objective"]
        assert loss_critic == loss_val_td["loss_critic"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_reduction(self, reduction, composite_action_dist):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_seq_mock_data_a2c(
            device=device, composite_action_dist=composite_action_dist
        )
        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        advantage = GAE(
            gamma=0.9,
            lmbda=0.9,
            value_network=value,
        )
        loss_fn = A2CLoss(
            actor,
            value,
            loss_critic_type="l2",
            reduction=reduction,
        )
        advantage(td)
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss_"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss_"):
                    continue
                assert loss[key].shape == torch.Size([])

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("clip_value", [True, None, 0.5, torch.tensor(0.5)])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_a2c_value_clipping(self, clip_value, device, composite_action_dist):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_a2c(
            device=device, composite_action_dist=composite_action_dist
        )
        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        value = self._create_mock_value(device=device)
        advantage = GAE(
            gamma=0.9,
            lmbda=0.9,
            value_network=value,
        )

        if isinstance(clip_value, bool):
            with pytest.raises(
                ValueError,
                match=f"clip_value must be a float or a scalar tensor, got {clip_value}.",
            ):
                loss_fn = A2CLoss(
                    actor,
                    value,
                    loss_critic_type="l2",
                    clip_value=clip_value,
                )
        else:
            loss_fn = A2CLoss(
                actor,
                value,
                loss_critic_type="l2",
                clip_value=clip_value,
            )
            advantage(td)

            value = td.pop(loss_fn.tensor_keys.value)

            if clip_value:
                # Test it fails without value key
                with pytest.raises(
                    KeyError,
                    match=f"clip_value is set to {clip_value}, but the key "
                    "state_value was not found in the input tensordict. "
                    "Make sure that the value_key passed to A2C exists in "
                    "the input tensordict.",
                ):
                    loss = loss_fn(td)

            # Add value back to td
            td.set(loss_fn.tensor_keys.value, value)

            # Test it works with value key
            loss = loss_fn(td)
            assert "loss_critic" in loss.keys()


class TestReinforce(LossModuleTestBase):
    seed = 0

    def test_reset_parameters_recursive(self):
        n_obs = 3
        n_act = 5
        value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=["observation"])
        net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor_net = ProbabilisticActor(
            module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            in_keys=["loc", "scale"],
            spec=Unbounded(n_act),
        )
        loss_fn = ReinforceLoss(
            actor_net,
            critic_network=value_net,
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("gradient_mode", [True, False])
    @pytest.mark.parametrize("advantage", ["gae", "td", "td_lambda", None])
    @pytest.mark.parametrize(
        "td_est",
        [
            ValueEstimators.TD1,
            ValueEstimators.TD0,
            ValueEstimators.GAE,
            ValueEstimators.TDLambda,
            None,
        ],
    )
    @pytest.mark.parametrize(
        "delay_value,functional", [[False, True], [False, False], [True, True]]
    )
    def test_reinforce_value_net(
        self, advantage, gradient_mode, delay_value, td_est, functional
    ):
        n_obs = 3
        n_act = 5
        batch = 4
        gamma = 0.9
        value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=["observation"])
        net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor_net = ProbabilisticActor(
            module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            in_keys=["loc", "scale"],
            spec=Unbounded(n_act),
        )
        if advantage == "gae":
            advantage = GAE(
                gamma=gamma,
                lmbda=0.9,
                value_network=value_net,
                differentiable=gradient_mode,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=gamma,
                value_network=value_net,
                differentiable=gradient_mode,
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9,
                lmbda=0.9,
                value_network=value_net,
                differentiable=gradient_mode,
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = ReinforceLoss(
            actor_net,
            critic_network=value_net,
            delay_value=delay_value,
            functional=functional,
        )

        td = TensorDict(
            {
                "observation": torch.randn(batch, n_obs),
                "next": {
                    "observation": torch.randn(batch, n_obs),
                    "reward": torch.randn(batch, 1),
                    "done": torch.zeros(batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(batch, 1, dtype=torch.bool),
                },
                "action": torch.randn(batch, n_act),
            },
            [batch],
            names=["time"],
        )
        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ):
            if advantage is not None:
                params = TensorDict.from_module(value_net)
                if delay_value:
                    target_params = loss_fn.target_critic_network_params
                else:
                    target_params = None
                advantage(td, params=params, target_params=target_params)
            elif td_est is not None:
                loss_fn.make_value_estimator(td_est)
            loss_td = loss_fn(td)
            autograd.grad(
                loss_td.get("loss_actor"),
                actor_net.parameters(),
                retain_graph=True,
            )
            autograd.grad(
                loss_td.get("loss_value"),
                value_net.parameters(),
                retain_graph=True,
            )
            with pytest.raises(RuntimeError, match="One of the "):
                autograd.grad(
                    loss_td.get("loss_actor"),
                    value_net.parameters(),
                    retain_graph=True,
                    allow_unused=False,
                )
            with pytest.raises(RuntimeError, match="One of the "):
                autograd.grad(
                    loss_td.get("loss_value"),
                    actor_net.parameters(),
                    retain_graph=True,
                    allow_unused=False,
                )

    @pytest.mark.parametrize(
        "td_est",
        [
            ValueEstimators.TD1,
            ValueEstimators.TD0,
            ValueEstimators.GAE,
            ValueEstimators.TDLambda,
        ],
    )
    def test_reinforce_tensordict_keys(self, td_est):
        n_obs = 3
        n_act = 5
        value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=["observation"])
        net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor_net = ProbabilisticActor(
            module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            in_keys=["loc", "scale"],
            spec=Unbounded(n_act),
        )

        loss_fn = ReinforceLoss(
            actor_net,
            critic_network=value_net,
        )

        default_keys = {
            "advantage": "advantage",
            "value_target": "value_target",
            "value": "state_value",
            "sample_log_prob": "action_log_prob",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value_net = ValueOperator(
            nn.Linear(n_obs, 1), in_keys=["observation"], out_keys=["state_value_test"]
        )

        loss_fn = ReinforceLoss(
            actor_net,
            critic_network=value_net,
        )

        key_mapping = {
            "advantage": ("advantage", "advantage_test"),
            "value_target": ("value_target", "value_target_test"),
            "value": ("value", "state_value_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize()
    def _create_mock_common_layer_setup(
        self, n_obs=3, n_act=4, ncells=4, batch=2, n_hidden=2, T=10
    ):
        common_net = MLP(
            num_cells=ncells,
            in_features=n_obs,
            depth=3,
            out_features=n_hidden,
        )
        actor_net = MLP(
            num_cells=ncells,
            in_features=n_hidden,
            depth=2,
            out_features=2 * n_act,
        )
        value_net = MLP(
            in_features=n_hidden,
            num_cells=ncells,
            depth=2,
            out_features=1,
        )
        batch = [batch, T]
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": torch.randn(*batch, n_act),
                "action_log_prob": torch.randn(*batch),
                "done": torch.zeros(*batch, 1, dtype=torch.bool),
                "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                "next": {
                    "obs": torch.randn(*batch, n_obs),
                    "reward": torch.randn(*batch, 1),
                    "done": torch.zeros(*batch, 1, dtype=torch.bool),
                    "terminated": torch.zeros(*batch, 1, dtype=torch.bool),
                },
            },
            batch,
            names=[None, "time"],
        )
        common = Mod(common_net, in_keys=["obs"], out_keys=["hidden"])
        actor = ProbSeq(
            common,
            Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
            Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
            ProbMod(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                distribution_class=TanhNormal,
            ),
        )
        critic = Seq(
            common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"])
        )
        return actor, critic, common, td

    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_reinforce_tensordict_separate_losses(self, separate_losses):
        torch.manual_seed(self.seed)
        actor, critic, common, td = self._create_mock_common_layer_setup()
        loss_fn = ReinforceLoss(
            actor_network=actor,
            critic_network=critic,
            separate_losses=separate_losses,
        )

        loss = loss_fn(td)

        assert all(
            (p.grad is None) or (p.grad == 0).all()
            for p in loss_fn.critic_network_params.values(True, True)
        )
        assert all(
            (p.grad is None) or (p.grad == 0).all()
            for p in loss_fn.actor_network_params.values(True, True)
        )
        # check that losses are independent
        for k in loss.keys():
            # can't sum over loss_actor as it is a scalar ()
            if not k.startswith("loss") or k == "loss_actor":
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_value":
                common_layers_no = len(list(common.parameters()))
                if separate_losses:
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(True, True)
                    )
                    common_layers = itertools.islice(
                        loss_fn.critic_network_params.values(True, True),
                        common_layers_no,
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all() for p in common_layers
                    )
                    critic_layers = itertools.islice(
                        loss_fn.critic_network_params.values(True, True),
                        common_layers_no,
                        None,
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all() for p in critic_layers
                    )
                else:
                    common_layers = itertools.islice(
                        loss_fn.critic_network_params.values(True, True),
                        common_layers_no,
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all() for p in common_layers
                    )
                    actor_layers = itertools.islice(
                        loss_fn.actor_network_params.values(True, True),
                        common_layers_no,
                        None,
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all() for p in actor_layers
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.critic_network_params.values(True, True)
                    )

            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_reinforce_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        n_obs = 3
        n_act = 5
        batch = 4
        value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=[observation_key])
        net = nn.Sequential(nn.Linear(n_obs, 2 * n_act), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=[observation_key], out_keys=["loc", "scale"]
        )
        actor_net = ProbabilisticActor(
            module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            in_keys=["loc", "scale"],
            spec=Unbounded(n_act),
        )
        loss = ReinforceLoss(actor_network=actor_net, critic_network=value_net)
        loss.set_keys(
            reward=reward_key,
            done=done_key,
            action=action_key,
            terminated=terminated_key,
        )

        observation = torch.randn(batch, n_obs)
        action = torch.randn(batch, n_act)
        next_reward = torch.randn(batch, 1)
        next_observation = torch.randn(batch, n_obs)
        next_done = torch.zeros(batch, 1, dtype=torch.bool)
        next_terminated = torch.zeros(batch, 1, dtype=torch.bool)

        kwargs = {
            action_key: action,
            observation_key: observation,
            f"next_{reward_key}": next_reward,
            f"next_{done_key}": next_done,
            f"next_{terminated_key}": next_terminated,
            f"next_{observation_key}": next_observation,
        }
        td = TensorDict(kwargs, [batch]).unflatten_keys("_")
        loss_val = loss(**kwargs)
        loss_val_td = loss(td)
        torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
        torch.testing.assert_close(loss_val_td.get("loss_value"), loss_val[1])
        # test select
        torch.manual_seed(self.seed)
        loss.select_out_keys("loss_actor")
        if torch.__version__ >= "2.0.0":
            loss_actor = loss(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_actor = loss(**kwargs)
            return
        assert loss_actor == loss_val_td["loss_actor"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_reinforce_reduction(self, reduction):
        torch.manual_seed(self.seed)
        actor, critic, common, td = self._create_mock_common_layer_setup()
        loss_fn = ReinforceLoss(
            actor_network=actor,
            critic_network=critic,
            reduction=reduction,
        )
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss_"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss_"):
                    continue
                assert loss[key].shape == torch.Size([])

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("clip_value", [True, None, 0.5, torch.tensor(0.5)])
    def test_reinforce_value_clipping(self, clip_value, device):
        torch.manual_seed(self.seed)
        actor, critic, common, td = self._create_mock_common_layer_setup()
        actor = actor.to(device)
        critic = critic.to(device)
        td = td.to(device)
        advantage = GAE(
            gamma=0.9,
            lmbda=0.9,
            value_network=critic,
        )
        if isinstance(clip_value, bool):
            with pytest.raises(
                ValueError,
                match=f"clip_value must be a float or a scalar tensor, got {clip_value}.",
            ):
                loss_fn = ReinforceLoss(
                    actor_network=actor,
                    critic_network=critic,
                    clip_value=clip_value,
                )
                return
        else:
            loss_fn = ReinforceLoss(
                actor_network=actor,
                critic_network=critic,
                clip_value=clip_value,
            )
            advantage(td)

            value = td.pop(loss_fn.tensor_keys.value)

            if clip_value:
                # Test it fails without value key
                with pytest.raises(
                    KeyError,
                    match=f"clip_value is set to {loss_fn.clip_value}, but the key "
                    "state_value was not found in the input tensordict. "
                    "Make sure that the value_key passed to Reinforce exists in "
                    "the input tensordict.",
                ):
                    loss = loss_fn(td)

            # Add value back to td
            td.set(loss_fn.tensor_keys.value, value)

            # Test it works with value key
            loss = loss_fn(td)
            assert "loss_value" in loss.keys()
