# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import itertools

import numpy as np
import pytest
import torch
from _objectives_common import (
    _check_td_steady,
    _has_functorch,
    FUNCTORCH_ERR,
    LossModuleTestBase,
)

from tensordict import assert_allclose_td, TensorDict
from tensordict.nn import (
    NormalParamExtractor,
    ProbabilisticTensorDictModule as ProbMod,
    ProbabilisticTensorDictSequential as ProbSeq,
    TensorDictModule,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from torch import nn

from torchrl._utils import rl_warnings
from torchrl.data import Bounded, OneHot
from torchrl.data.postprocs.postprocs import MultiStep
from torchrl.modules import OneHotCategorical
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules.models.models import MLP
from torchrl.modules.tensordict_module.actors import ProbabilisticActor, ValueOperator
from torchrl.objectives import DiscreteIQLLoss, IQLLoss
from torchrl.objectives.common import LossModule
from torchrl.objectives.utils import SoftUpdate, ValueEstimators

from torchrl.testing import (  # noqa
    call_value_nets as _call_value_nets,
    dtype_fixture,
    get_available_devices,
    get_default_devices,
    PENDULUM_VERSIONED,
)


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestIQL(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = nn.Sequential(nn.Linear(obs_dim, 2 * action_dim), NormalParamExtractor())
        module = TensorDictModule(
            net, in_keys=[observation_key], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=module,
            in_keys=["loc", "scale"],
            spec=action_spec,
            distribution_class=TanhNormal,
        )
        return actor.to(device)

    def _create_mock_qvalue(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        out_keys=None,
        observation_key="observation",
        action_key="action",
    ):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        qvalue = ValueOperator(
            module=module,
            in_keys=[observation_key, action_key],
            out_keys=out_keys,
        )
        return qvalue.to(device)

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
            module=module, in_keys=[observation_key], out_keys=out_keys
        )
        return value.to(device)

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
            depth=1,
            out_features=2 * n_act,
        )
        value_net = MLP(
            in_features=n_hidden,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        qvalue_net = MLP(
            in_features=n_hidden + n_act,
            num_cells=ncells,
            depth=1,
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
        value = Seq(
            common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"])
        )
        qvalue = Seq(
            common,
            Mod(
                qvalue_net,
                in_keys=["hidden", "action"],
                out_keys=["state_action_value"],
            ),
        )
        qvalue(actor(td.clone()))
        value(td.clone())
        return actor, value, qvalue, common, td

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_iql(
        self,
        batch=16,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        observation_key="observation",
        action_key="action",
        done_key="done",
        terminated_key="terminated",
        reward_key="reward",
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
                observation_key: obs,
                "next": {
                    observation_key: next_obs,
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward,
                },
                action_key: action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_iql(
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
        value = self._create_mock_value()
        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("temperature", [0.0, 0.1, 1.0, 10.0])
    @pytest.mark.parametrize("expectile", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_iql(
        self,
        num_qvalue,
        device,
        temperature,
        expectile,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
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
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
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
            elif k == "loss_value":
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
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(
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
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

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

    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("temperature", [0.1])
    @pytest.mark.parametrize("expectile", [0.1])
    @pytest.mark.parametrize("td_est", [None])
    def test_iql_deactivate_vmap(
        self,
        num_qvalue,
        device,
        temperature,
        expectile,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        torch.manual_seed(0)
        loss_fn_vmap = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
            deactivate_vmap=False,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_vmap.make_value_estimator(td_est)

        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            torch.manual_seed(1)
            loss_vmap = loss_fn_vmap(td)

        torch.manual_seed(0)
        loss_fn_no_vmap = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
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

    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("temperature", [0.0])
    @pytest.mark.parametrize("expectile", [0.1])
    def test_iql_state_dict(
        self,
        num_qvalue,
        device,
        temperature,
        expectile,
    ):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
        )
        sd = loss_fn.state_dict()
        loss_fn2 = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_iql_separate_losses(self, separate_losses):
        torch.manual_seed(self.seed)
        actor, value, qvalue, common, td = self._create_mock_common_layer_setup()
        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            separate_losses=separate_losses,
        )
        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

            assert loss_fn.tensor_keys.priority in td.keys()

            # check that losses are independent
            for k in loss.keys():
                if not k.startswith("loss"):
                    continue
                loss[k].sum().backward(retain_graph=True)
                common_layers_no = len(list(common.parameters()))
                if k == "loss_actor":
                    if separate_losses:
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.value_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                    else:
                        common_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        value_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in value_layers
                        )
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
                elif k == "loss_value":
                    if separate_losses:
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.actor_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                    else:
                        common_layers = itertools.islice(
                            loss_fn.actor_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        actor_layers = itertools.islice(
                            loss_fn.actor_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in actor_layers
                        )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    if separate_losses:
                        common_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        value_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in value_layers
                        )
                    else:
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.value_network_params.values(
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
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.value_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    if separate_losses:
                        common_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        qvalue_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in qvalue_layers
                        )
                    else:
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.qvalue_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("temperature", [0.0, 0.1, 1.0, 10.0])
    @pytest.mark.parametrize("expectile", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_iql_batcher(
        self,
        n,
        num_qvalue,
        temperature,
        expectile,
        device,
        gamma=0.9,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)
        with _check_td_steady(ms_td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        # Remove warnings
        SoftUpdate(loss_fn, eps=0.5)

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
        target_qvalue = [
            p.clone()
            for p in loss_fn.target_qvalue_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_qvalue2 = [
            p.clone()
            for p in loss_fn.target_qvalue_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
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
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_iql_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        value = self._create_mock_value()

        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
        )

        default_keys = {
            "priority": "td_error",
            "log_prob": "_log_prob",
            "action": "action",
            "state_action_value": "state_action_value",
            "value": "state_value",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value = self._create_mock_value(out_keys=["value_test"])
        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
        )

        key_mapping = {
            "value": ("value", "value_test"),
            "done": ("done", "done_test"),
            "terminated": ("terminated", "terminated_test"),
            "reward": ("reward", ("reward", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_iql_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_iql(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(observation_key=observation_key)
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
            action_key=action_key,
            out_keys=["state_action_value"],
        )
        value = self._create_mock_value(observation_key=observation_key)

        loss = IQLLoss(actor_network=actor, qvalue_network=qvalue, value_network=value)
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
        )

        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        with pytest.warns(
            UserWarning,
            match="No target network updater has been associated with this loss module",
        ) if rl_warnings() else contextlib.nullcontext():
            loss_val = loss(**kwargs)
            loss_val_td = loss(td)
            assert len(loss_val) == 4
            torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
            torch.testing.assert_close(loss_val_td.get("loss_qvalue"), loss_val[1])
            torch.testing.assert_close(loss_val_td.get("loss_value"), loss_val[2])
            torch.testing.assert_close(loss_val_td.get("entropy"), loss_val[3])
            # test select
            torch.manual_seed(self.seed)
            loss.select_out_keys("loss_actor", "loss_value")
            if torch.__version__ >= "2.0.0":
                loss_actor, loss_value = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, loss_value = loss(**kwargs)
                return
            assert loss_actor == loss_val_td["loss_actor"]
            assert loss_value == loss_val_td["loss_value"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_iql_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = IQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            reduction=reduction,
            scalar_output_mode="exclude" if reduction == "none" else None,
        )
        loss_fn.make_value_estimator()
        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
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


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestDiscreteIQL(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        action_key="action",
    ):
        # Actor
        action_spec = OneHot(action_dim)
        net = nn.Linear(obs_dim, action_dim)
        module = TensorDictModule(net, in_keys=[observation_key], out_keys=["logits"])
        actor = ProbabilisticActor(
            spec=action_spec,
            module=module,
            in_keys=["logits"],
            out_keys=[action_key],
            distribution_class=OneHotCategorical,
            return_log_prob=False,
        )
        return actor.to(device)

    def _create_mock_qvalue(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        out_keys=None,
        observation_key="observation",
    ):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim, action_dim)

            def forward(self, obs):
                return self.linear(obs)

        module = ValueClass()
        qvalue = ValueOperator(
            module=module, in_keys=[observation_key], out_keys=["state_action_value"]
        )
        return qvalue.to(device)

    def _create_mock_value(
        self,
        batch=2,
        obs_dim=3,
        device="cpu",
        out_keys=None,
        observation_key="observation",
    ):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module, in_keys=[observation_key], out_keys=out_keys
        )
        return value.to(device)

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
            depth=1,
            out_features=2 * n_act,
        )
        value_net = MLP(
            in_features=n_hidden,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        qvalue_net = MLP(
            in_features=n_hidden,
            num_cells=ncells,
            depth=1,
            out_features=n_act,
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
            Mod(actor_net, in_keys=["hidden"], out_keys=["logits"]),
            ProbMod(
                in_keys=["logits"],
                out_keys=["action"],
                distribution_class=OneHotCategorical,
            ),
        )
        value = Seq(
            common, Mod(value_net, in_keys=["hidden"], out_keys=["state_value"])
        )
        qvalue = Seq(
            common,
            Mod(
                qvalue_net,
                in_keys=["hidden"],
                out_keys=["state_action_value"],
            ),
        )
        qvalue(actor(td.clone()))
        value(td.clone())
        return actor, value, qvalue, common, td

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_discrete_iql(
        self,
        batch=16,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        observation_key="observation",
        action_key="action",
        done_key="done",
        terminated_key="terminated",
        reward_key="reward",
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        next_obs = torch.randn(batch, obs_dim, device=device)
        if atoms:
            raise NotImplementedError
        else:
            action_value = torch.randn(batch, action_dim, device=device)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)
        reward = torch.randn(batch, 1, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
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
                action_key: action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_discrete_iql(
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
            action_value = torch.randn(batch, T, action_dim, device=device)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)

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
        value = self._create_mock_value()
        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            action_space="one-hot",
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("temperature", [0.0, 0.1, 1.0, 10.0])
    @pytest.mark.parametrize("expectile", [0.1, 0.5])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_discrete_iql(
        self,
        num_qvalue,
        device,
        temperature,
        expectile,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_discrete_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
            action_space="one-hot",
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
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
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
            elif k == "loss_value":
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
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(
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
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

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

    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("temperature", [0.0])
    @pytest.mark.parametrize("expectile", [0.1])
    def test_discrete_iql_state_dict(
        self,
        num_qvalue,
        device,
        temperature,
        expectile,
    ):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
            action_space="one-hot",
        )
        sd = loss_fn.state_dict()
        loss_fn2 = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
            action_space="one-hot",
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_discrete_iql_separate_losses(self, separate_losses):
        torch.manual_seed(self.seed)
        actor, value, qvalue, common, td = self._create_mock_common_layer_setup()
        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            separate_losses=separate_losses,
            action_space="one-hot",
        )
        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

            assert loss_fn.tensor_keys.priority in td.keys()

            # check that losses are independent
            for k in loss.keys():
                if not k.startswith("loss"):
                    continue
                loss[k].sum().backward(retain_graph=True)
                common_layers_no = len(list(common.parameters()))
                if k == "loss_actor":
                    if separate_losses:
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.value_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                    else:
                        common_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        value_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in value_layers
                        )
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
                elif k == "loss_value":
                    if separate_losses:
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.actor_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                    else:
                        common_layers = itertools.islice(
                            loss_fn.actor_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        actor_layers = itertools.islice(
                            loss_fn.actor_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in actor_layers
                        )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    if separate_losses:
                        common_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        value_layers = itertools.islice(
                            loss_fn.value_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in value_layers
                        )
                    else:
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.value_network_params.values(
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
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.value_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    if separate_losses:
                        common_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                        )
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in common_layers
                        )
                        qvalue_layers = itertools.islice(
                            loss_fn.qvalue_network_params.values(True, True),
                            common_layers_no,
                            None,
                        )
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in qvalue_layers
                        )
                    else:
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.qvalue_network_params.values(
                                include_nested=True, leaves_only=True
                            )
                        )
                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("temperature", [0.0, 0.1, 1.0, 10.0])
    @pytest.mark.parametrize("expectile", [0.1, 0.5])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_discrete_iql_batcher(
        self,
        n,
        num_qvalue,
        temperature,
        expectile,
        device,
        gamma=0.9,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_discrete_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            temperature=temperature,
            expectile=expectile,
            loss_function="l2",
            action_space="one-hot",
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)
        with _check_td_steady(ms_td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        SoftUpdate(loss_fn, eps=0.5)

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
        target_qvalue = [
            p.clone()
            for p in loss_fn.target_qvalue_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_qvalue2 = [
            p.clone()
            for p in loss_fn.target_qvalue_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
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
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_discrete_iql_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        value = self._create_mock_value()

        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            action_space="one-hot",
        )

        default_keys = {
            "priority": "td_error",
            "log_prob": "_log_prob",
            "action": "action",
            "state_action_value": "state_action_value",
            "value": "state_value",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value = self._create_mock_value(out_keys=["value_test"])
        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            action_space="one-hot",
        )

        key_mapping = {
            "value": ("value", "value_test"),
            "done": ("done", "done_test"),
            "terminated": ("terminated", "terminated_test"),
            "reward": ("reward", ("reward", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_discrete_iql_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_discrete_iql(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(observation_key=observation_key)
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
            out_keys=["state_action_value"],
        )
        value = self._create_mock_value(observation_key=observation_key)

        loss = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            action_space="one-hot",
        )
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
        )

        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        with pytest.warns(
            UserWarning,
            match="No target network updater has been associated with this loss module",
        ) if rl_warnings() else contextlib.nullcontext():
            loss_val = loss(**kwargs)
            loss_val_td = loss(td)
            assert len(loss_val) == 4
            torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
            torch.testing.assert_close(loss_val_td.get("loss_qvalue"), loss_val[1])
            torch.testing.assert_close(loss_val_td.get("loss_value"), loss_val[2])
            torch.testing.assert_close(loss_val_td.get("entropy"), loss_val[3])
            # test select
            torch.manual_seed(self.seed)
            loss.select_out_keys("loss_actor", "loss_value")
            if torch.__version__ >= "2.0.0":
                loss_actor, loss_value = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, loss_value = loss(**kwargs)
                return
            assert loss_actor == loss_val_td["loss_actor"]
            assert loss_value == loss_val_td["loss_value"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_discrete_iql_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_discrete_iql(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

        loss_fn = DiscreteIQLLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            action_space="one-hot",
            reduction=reduction,
            scalar_output_mode="exclude" if reduction == "none" else None,
        )
        loss_fn.make_value_estimator()
        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
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


@pytest.mark.parametrize("create_target_params", [True, False])
@pytest.mark.parametrize(
    "cast", [None, torch.float, torch.double, *get_default_devices()]
)
def test_param_buffer_types(create_target_params, cast):
    class MyLoss(LossModule):
        actor_network: TensorDictModule
        actor_network_params: TensorDict
        target_actor_network_params: TensorDict

        def __init__(self, actor_network):
            super().__init__()
            self.convert_to_functional(
                actor_network,
                "actor_network",
                create_target_params=create_target_params,
            )

        def _forward_value_estimator_keys(self, **kwargs) -> None:
            pass

    actor_module = TensorDictModule(
        nn.Sequential(nn.Linear(3, 4), nn.BatchNorm1d(4)),
        in_keys=["obs"],
        out_keys=["action"],
    )
    loss = MyLoss(actor_module)

    LossModuleTestBase.reset_parameters_recursive_test(loss)

    if create_target_params:
        SoftUpdate(loss, eps=0.5)

    if cast is not None:
        loss.to(cast)
    for name in ("weight", "bias"):
        param = loss.actor_network_params["module", "0", name]
        assert isinstance(param, nn.Parameter)
        target = loss.target_actor_network_params["module", "0", name]
        if create_target_params:
            assert target.data_ptr() != param.data_ptr()
        else:
            assert target.data_ptr() == param.data_ptr()
        assert param.requires_grad
        assert not target.requires_grad
        if cast is not None:
            if isinstance(cast, torch.dtype):
                assert param.dtype == cast
                assert target.dtype == cast
            else:
                assert param.device == cast
                assert target.device == cast

    if create_target_params:
        assert (
            loss.actor_network_params["module", "0", "weight"].data.data_ptr()
            != loss.target_actor_network_params["module", "0", "weight"].data.data_ptr()
        )
        assert (
            loss.actor_network_params["module", "0", "bias"].data.data_ptr()
            != loss.target_actor_network_params["module", "0", "bias"].data.data_ptr()
        )
    else:
        assert (
            loss.actor_network_params["module", "0", "weight"].data.data_ptr()
            == loss.target_actor_network_params["module", "0", "weight"].data.data_ptr()
        )
        assert (
            loss.actor_network_params["module", "0", "bias"].data.data_ptr()
            == loss.target_actor_network_params["module", "0", "bias"].data.data_ptr()
        )

    assert loss.actor_network_params["module", "0", "bias"].requires_grad
    assert not loss.target_actor_network_params["module", "0", "bias"].requires_grad
    assert not isinstance(
        loss.actor_network_params["module", "1", "running_mean"], nn.Parameter
    )
    assert not isinstance(
        loss.target_actor_network_params["module", "1", "running_mean"], nn.Parameter
    )
    assert not isinstance(
        loss.actor_network_params["module", "1", "running_var"], nn.Parameter
    )
    assert not isinstance(
        loss.target_actor_network_params["module", "1", "running_var"], nn.Parameter
    )
