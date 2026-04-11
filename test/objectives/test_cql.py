# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib

import numpy as np
import pytest
import torch
from _objectives_common import _check_td_steady, LossModuleTestBase

from tensordict import assert_allclose_td, TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn

from torchrl._utils import rl_warnings
from torchrl.data import Bounded, Categorical, Composite, OneHot
from torchrl.data.postprocs.postprocs import MultiStep
from torchrl.modules import QValueActor
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
from torchrl.objectives import CQLLoss, DiscreteCQLLoss, DQNLoss
from torchrl.objectives.utils import SoftUpdate, ValueEstimators

from torchrl.testing import (  # noqa
    call_value_nets as _call_value_nets,
    dtype_fixture,
    get_available_devices,
    get_default_devices,
    PENDULUM_VERSIONED,
)


class TestCQL(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = nn.Sequential(nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=module,
            in_keys=["loc", "scale"],
            spec=action_spec,
            distribution_class=TanhNormal,
        )
        return actor.to(device)

    def _create_mock_qvalue(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        qvalue = ValueOperator(
            module=module,
            in_keys=["observation", "action"],
        )
        return qvalue.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_cql(
        self, batch=16, obs_dim=3, action_dim=4, atoms=None, device="cpu"
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
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "done": done,
                    "terminated": terminated,
                    "reward": reward,
                },
                "action": action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_cql(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
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
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            names=[None, "time"],
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        loss_fn = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, True))
    @pytest.mark.parametrize("max_q_backup", [True, False])
    @pytest.mark.parametrize("deterministic_backup", [True, False])
    @pytest.mark.parametrize("with_lagrange", [True, False])
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_cql(
        self,
        delay_actor,
        delay_qvalue,
        max_q_backup,
        deterministic_backup,
        with_lagrange,
        device,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_cql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return

        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys()

        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            if k == "loss_alpha_prime" and not with_lagrange:
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_actor_bc":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_cql":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            elif k == "loss_alpha_prime":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()
            assert all(
                (p.grad is None) or (p.grad == 0).all()
                for p in loss_fn.actor_network_params.values(
                    include_nested=True, leaves_only=True
                )
            )
            assert all(
                (p.grad is None) or (p.grad == 0).all()
                for p in loss_fn.qvalue_network_params.values(
                    include_nested=True, leaves_only=True
                )
            )

        sum(
            [item for name, item in loss.items() if name.startswith("loss_")]
        ).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            if not name.startswith("target_"):
                assert (
                    p.grad is not None and p.grad.norm() > 0.0
                ), f"parameter {name} (shape: {p.shape}) has a null gradient"
            else:
                assert (
                    p.grad is None or p.grad.norm() == 0.0
                ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

    @pytest.mark.parametrize("delay_actor", (True,))
    @pytest.mark.parametrize("delay_qvalue", (True,))
    @pytest.mark.parametrize(
        "max_q_backup",
        [
            True,
        ],
    )
    @pytest.mark.parametrize(
        "deterministic_backup",
        [
            True,
        ],
    )
    @pytest.mark.parametrize(
        "with_lagrange",
        [
            True,
        ],
    )
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("td_est", [None])
    def test_cql_deactivate_vmap(
        self,
        delay_actor,
        delay_qvalue,
        max_q_backup,
        deterministic_backup,
        with_lagrange,
        device,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_cql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        torch.manual_seed(0)
        loss_fn_vmap = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
            deactivate_vmap=False,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_vmap.make_value_estimator(td_est)
            return

        if td_est is not None:
            loss_fn_vmap.make_value_estimator(td_est)
        tdc = td.clone()
        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            torch.manual_seed(1)
            loss_vmap = loss_fn_vmap(td)
        td = tdc

        torch.manual_seed(0)
        loss_fn_no_vmap = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
            deactivate_vmap=True,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_no_vmap.make_value_estimator(td_est)
            return

        if td_est is not None:
            loss_fn_no_vmap.make_value_estimator(td_est)

        with pytest.raises(
            NotImplementedError,
            match="This implementation is not supported for torch<2.7",
        ) if torch.__version__ < "2.7" else contextlib.nullcontext():
            with _check_td_steady(td), pytest.warns(
                UserWarning, match="No target network updater"
            ) if rl_warnings() else contextlib.nullcontext():
                torch.manual_seed(1)
                loss_no_vmap = loss_fn_no_vmap(td)
            assert_allclose_td(loss_vmap, loss_no_vmap)

    @pytest.mark.parametrize("delay_actor", (True,))
    @pytest.mark.parametrize("delay_qvalue", (True,))
    @pytest.mark.parametrize("max_q_backup", [True])
    @pytest.mark.parametrize("deterministic_backup", [True])
    @pytest.mark.parametrize("with_lagrange", [True])
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("td_est", [None])
    def test_cql_qvalfromlist(
        self,
        delay_actor,
        delay_qvalue,
        max_q_backup,
        deterministic_backup,
        with_lagrange,
        device,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_cql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue0 = self._create_mock_qvalue(device=device)
        qvalue1 = self._create_mock_qvalue(device=device)

        loss_fn_single = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue0,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        loss_fn_mult = CQLLoss(
            actor_network=actor,
            qvalue_network=[qvalue0, qvalue1],
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        # Check that all params have the same shape
        p2 = dict(loss_fn_mult.named_parameters())
        for key, val in loss_fn_single.named_parameters():
            assert val.shape == p2[key].shape
        assert len(dict(loss_fn_single.named_parameters())) == len(p2)

    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("max_q_backup", [True])
    @pytest.mark.parametrize("deterministic_backup", [True])
    @pytest.mark.parametrize("with_lagrange", [True])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_cql_state_dict(
        self,
        delay_actor,
        delay_qvalue,
        max_q_backup,
        deterministic_backup,
        with_lagrange,
        device,
    ):
        if delay_actor or delay_qvalue:
            pytest.skip("incompatible config")

        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True

        loss_fn = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            **kwargs,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            **kwargs,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("action_dim", [1, 2, 4, 8])
    def test_cql_target_entropy_auto(self, action_dim):
        """Regression test for target_entropy='auto' should be -dim(A)."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(action_dim=action_dim)
        qvalue = self._create_mock_qvalue(action_dim=action_dim)

        loss_fn = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        # target_entropy="auto" should compute -action_dim
        assert (
            loss_fn.target_entropy.item() == -action_dim
        ), f"target_entropy should be -{action_dim}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("max_q_backup", [True, False])
    @pytest.mark.parametrize("deterministic_backup", [True, False])
    @pytest.mark.parametrize("with_lagrange", [True, False])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_cql_batcher(
        self,
        n,
        delay_actor,
        delay_qvalue,
        max_q_backup,
        deterministic_backup,
        with_lagrange,
        device,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_cql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True

        loss_fn = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            max_q_backup=max_q_backup,
            deterministic_backup=deterministic_backup,
            with_lagrange=with_lagrange,
            **kwargs,
        )

        ms = MultiStep(gamma=0.9, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)
        with pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            with _check_td_steady(ms_td):
                loss_ms = loss_fn(ms_td)
            assert loss_fn.tensor_keys.priority in ms_td.keys()

            with torch.no_grad():
                torch.manual_seed(0)  # log-prob is computed with a random action
                np.random.seed(0)
                loss = loss_fn(td)
            if n == 1:
                assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
                _loss = sum(
                    [item for name, item in loss.items() if name.startswith("loss_")]
                )
                _loss_ms = sum(
                    [item for name, item in loss_ms.items() if name.startswith("loss_")]
                )
                assert (
                    abs(_loss - _loss_ms) < 1e-3
                ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
            else:
                with pytest.raises(AssertionError):
                    assert_allclose_td(loss, loss_ms)
            sum(
                [item for name, item in loss_ms.items() if name.startswith("loss_")]
            ).backward()
            named_parameters = loss_fn.named_parameters()
            for name, p in named_parameters:
                if not name.startswith("target_"):
                    assert (
                        p.grad is not None and p.grad.norm() > 0.0
                    ), f"parameter {name} (shape: {p.shape}) has a null gradient"
                else:
                    assert (
                        p.grad is None or p.grad.norm() == 0.0
                    ), f"target parameter {name} (shape: {p.shape}) has a non-null gradient"

            # Check param update effect on targets
            target_actor = [
                p.clone()
                for p in loss_fn.target_actor_network_params.values(
                    include_nested=True, leaves_only=True
                )
            ]
            target_qvalue = [
                p.clone()
                for p in loss_fn.target_qvalue_network_params.values(
                    include_nested=True, leaves_only=True
                )
            ]
            for p in loss_fn.parameters():
                if p.requires_grad:
                    p.data += torch.randn_like(p)
            target_actor2 = [
                p.clone()
                for p in loss_fn.target_actor_network_params.values(
                    include_nested=True, leaves_only=True
                )
            ]
            target_qvalue2 = [
                p.clone()
                for p in loss_fn.target_qvalue_network_params.values(
                    include_nested=True, leaves_only=True
                )
            ]
            if loss_fn.delay_actor:
                assert all(
                    (p1 == p2).all() for p1, p2 in zip(target_actor, target_actor2)
                )
            else:
                assert not any(
                    (p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2)
                )
            if loss_fn.delay_qvalue:
                assert all(
                    (p1 == p2).all() for p1, p2 in zip(target_qvalue, target_qvalue2)
                )
            else:
                assert not any(
                    (p1 == p2).any() for p1, p2 in zip(target_qvalue, target_qvalue2)
                )

            # check that policy is updated after parameter update
            parameters = [p.clone() for p in actor.parameters()]
            for p in loss_fn.parameters():
                if p.requires_grad:
                    p.data += torch.randn_like(p)
            assert all(
                (p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters())
            )

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_cql_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_cql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = CQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            delay_actor=False,
            delay_qvalue=False,
            reduction=reduction,
            scalar_output_mode="exclude" if reduction == "none" else None,
        )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])


class TestDiscreteCQL(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        action_spec_type,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        is_nn_module=False,
        action_value_key=None,
    ):
        # Actor
        if action_spec_type == "one_hot":
            action_spec = OneHot(action_dim)
        elif action_spec_type == "categorical":
            action_spec = Categorical(action_dim)
        else:
            raise ValueError(f"Wrong action spec type: {action_spec_type}")

        module = nn.Linear(obs_dim, action_dim)
        if is_nn_module:
            return module.to(device)
        actor = QValueActor(
            spec=Composite(
                {
                    "action": action_spec,
                    (
                        "action_value" if action_value_key is None else action_value_key
                    ): None,
                    "chosen_action_value": None,
                },
                shape=[],
            ),
            action_space=action_spec_type,
            module=module,
            action_value_key=action_value_key,
        ).to(device)
        return actor

    def _create_mock_data_dcql(
        self,
        action_spec_type,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        action_key="action",
        action_value_key="action_value",
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim)
        next_obs = torch.randn(batch, obs_dim)

        action_value = torch.randn(batch, action_dim)
        action = (action_value == action_value.max(-1, True)[0]).to(torch.long)

        if action_spec_type == "categorical":
            action_value = torch.max(action_value, -1, keepdim=True)[0]
            action = torch.argmax(action, -1, keepdim=False)
        reward = torch.randn(batch, 1)
        done = torch.zeros(batch, 1, dtype=torch.bool)
        terminated = torch.zeros(batch, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "done": done,
                    "terminated": terminated,
                    "reward": reward,
                },
                action_key: action,
                action_value_key: action_value,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_dcql(
        self,
        action_spec_type,
        batch=2,
        T=4,
        obs_dim=3,
        action_dim=4,
        device="cpu",
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]

        action_value = torch.randn(batch, T, action_dim, device=device)
        action = (action_value == action_value.max(-1, True)[0]).to(torch.long)

        # action_value = action_value.unsqueeze(-1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
        if action_spec_type == "categorical":
            action_value = torch.max(action_value, -1, keepdim=True)[0]
            action = torch.argmax(action, -1, keepdim=False)
            action = action.masked_fill_(~mask, 0.0)
        else:
            action = action.masked_fill_(~mask.unsqueeze(-1), 0.0)
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
                "action": action,
                "action_value": action_value.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            names=[None, "time"],
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor(
            action_spec_type="one_hot",
        )
        loss_fn = DiscreteCQLLoss(actor)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_dcql(self, delay_value, device, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        td = self._create_mock_data_dcql(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DiscreteCQLLoss(actor, loss_function="l2", delay_value=delay_value)
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)
        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(td):
            loss = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys(True)

        if delay_value:
            SoftUpdate(loss_fn, eps=0.5)

        sum([item for key, item in loss.items() if key.startswith("loss")]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = loss_fn.target_value_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_value2 = loss_fn.target_value_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    def test_dcql_state_dict(self, delay_value, device, action_spec_type):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DiscreteCQLLoss(actor, loss_function="l2", delay_value=delay_value)
        sd = loss_fn.state_dict()
        loss_fn2 = DiscreteCQLLoss(actor, loss_function="l2", delay_value=delay_value)
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    def test_dcql_batcher(self, n, delay_value, device, action_spec_type, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )

        td = self._create_seq_mock_data_dcql(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DiscreteCQLLoss(actor, loss_function="l2", delay_value=delay_value)

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)
        ms_td = ms(td.clone())

        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        if delay_value:
            SoftUpdate(loss_fn, eps=0.5)

        with torch.no_grad():
            loss = loss_fn(td)
        if n == 1:
            assert_allclose_td(td, ms_td.select(*td.keys(True, True)))
            _loss = sum([item for key, item in loss.items() if key.startswith("loss_")])
            _loss_ms = sum(
                [item for key, item in loss_ms.items() if key.startswith("loss_")]
            )
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum(
            [item for key, item in loss_ms.items() if key.startswith("loss_")]
        ).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = loss_fn.target_value_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_value2 = loss_fn.target_value_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_dcql_tensordict_keys(self, td_est):
        torch.manual_seed(self.seed)
        action_spec_type = "one_hot"
        actor = self._create_mock_actor(action_spec_type=action_spec_type)
        loss_fn = DQNLoss(actor, delay_value=True)

        default_keys = {
            "value_target": "value_target",
            "value": "chosen_action_value",
            "priority": "td_error",
            "action_value": "action_value",
            "action": "action",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(loss_fn, default_keys=default_keys)

        loss_fn = DiscreteCQLLoss(actor)
        key_mapping = {
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, action_value_key="chosen_action_value_2"
        )
        loss_fn = DiscreteCQLLoss(actor)
        key_mapping = {
            "value": ("value", "chosen_action_value_2"),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_spec_type", ("categorical", "one_hot"))
    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_dcql_tensordict_run(self, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        tensor_keys = {
            "action_value": "action_value_test",
            "action": "action_test",
            "priority": "priority_test",
        }
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type,
            action_value_key=tensor_keys["action_value"],
        )
        td = self._create_mock_data_dcql(
            action_spec_type=action_spec_type,
            action_key=tensor_keys["action"],
            action_value_key=tensor_keys["action_value"],
        )

        loss_fn = DiscreteCQLLoss(actor, loss_function="l2")
        loss_fn.set_keys(**tensor_keys)

        SoftUpdate(loss_fn, eps=0.5)

        if td_est is not None:
            loss_fn.make_value_estimator(td_est)
        with _check_td_steady(td):
            _ = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys()

    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_dcql_notensordict(
        self, observation_key, reward_key, done_key, terminated_key
    ):
        n_obs = 3
        n_action = 4
        action_spec = OneHot(n_action)
        module = nn.Linear(n_obs, n_action)  # a simple value model
        actor = QValueActor(
            spec=action_spec,
            action_space="one_hot",
            module=module,
            in_keys=[observation_key],
        )
        loss = DiscreteCQLLoss(actor)

        SoftUpdate(loss, eps=0.5)

        loss.set_keys(reward=reward_key, done=done_key, terminated=terminated_key)
        # define data
        observation = torch.randn(n_obs)
        next_observation = torch.randn(n_obs)
        action = action_spec.rand()
        next_reward = torch.randn(1)
        next_done = torch.zeros(1, dtype=torch.bool)
        next_terminated = torch.zeros(1, dtype=torch.bool)
        kwargs = {
            observation_key: observation,
            f"next_{observation_key}": next_observation,
            f"next_{reward_key}": next_reward,
            f"next_{done_key}": next_done,
            f"next_{terminated_key}": next_terminated,
            "action": action,
        }
        td = TensorDict(kwargs, []).unflatten_keys("_")
        loss_val = loss(**kwargs)

        loss_val_td = loss(td)

        torch.testing.assert_close(loss_val_td.get(loss.out_keys[0]), loss_val[0])
        torch.testing.assert_close(loss_val_td.get(loss.out_keys[1]), loss_val[1])

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_dcql_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        actor = self._create_mock_actor(action_spec_type="one_hot", device=device)
        td = self._create_mock_data_dcql(action_spec_type="one_hot", device=device)
        loss_fn = DiscreteCQLLoss(
            actor, loss_function="l2", delay_value=False, reduction=reduction
        )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])
