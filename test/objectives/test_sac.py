# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib
import functools
import itertools
from copy import deepcopy

import numpy as np
import pytest
import torch
from _objectives_common import (
    _check_td_steady,
    _has_functorch,
    FUNCTORCH_ERR,
    LossModuleTestBase,
)

from tensordict import assert_allclose_td, TensorDict, TensorDictBase
from tensordict.nn import (
    CompositeDistribution,
    NormalParamExtractor,
    ProbabilisticTensorDictModule as ProbMod,
    ProbabilisticTensorDictSequential as ProbSeq,
    TensorDictModule,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from tensordict.nn.distributions.composite import _add_suffix
from torch import nn

from torchrl._utils import rl_warnings
from torchrl.data import (
    Bounded,
    Composite,
    LazyTensorStorage,
    OneHot,
    TensorDictPrioritizedReplayBuffer,
    Unbounded,
)
from torchrl.data.postprocs.postprocs import MultiStep
from torchrl.modules import OneHotCategorical
from torchrl.modules.distributions.continuous import TanhDelta, TanhNormal
from torchrl.modules.models.models import MLP
from torchrl.modules.tensordict_module.actors import (
    ActorCriticOperator,
    ProbabilisticActor,
    ValueOperator,
)
from torchrl.objectives import CrossQLoss, DiscreteSACLoss, SACLoss
from torchrl.objectives.deprecated import DoubleREDQLoss_deprecated, REDQLoss_deprecated
from torchrl.objectives.redq import REDQLoss
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
@pytest.mark.parametrize("version", [1, 2])
class TestSAC(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        action_key="action",
        composite_action_dist=False,
        return_action_spec=False,
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
            distribution_class=distribution_class,
            in_keys=actor_in_keys,
            out_keys=[action_key],
            spec=action_spec,
        )
        assert actor.log_prob_keys
        actor = actor.to(device)
        if return_action_spec:
            return actor, action_spec
        return actor

    def _create_mock_qvalue(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        action_key="action",
        out_keys=None,
    ):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                if isinstance(act, TensorDictBase):
                    act = act.get("action1")
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
        observation_key="observation",
        out_keys=None,
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
        composite_action_dist=False,
    ):
        class QValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(n_hidden + n_act, n_hidden)
                self.relu = nn.ReLU()
                self.linear2 = nn.Linear(n_hidden, 1)

            def forward(self, obs, act):
                if isinstance(act, TensorDictBase):
                    act = act.get("action1")
                return self.linear2(self.relu(self.linear1(torch.cat([obs, act], -1))))

        common = MLP(
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
        qvalue = QValueClass()
        batch = [batch]
        action = torch.randn(*batch, n_act)
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": {"action1": action} if composite_action_dist else action,
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
        )
        common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
        if composite_action_dist:
            distribution_class = functools.partial(
                CompositeDistribution,
                distribution_map={
                    "action1": TanhNormal,
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
        qvalue_head = Mod(
            qvalue, in_keys=["hidden", "action"], out_keys=["state_action_value"]
        )
        qvalue = Seq(common, qvalue_head)
        return actor, qvalue, common, td

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_sac(
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
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_sac(
        self,
        batch=8,
        T=4,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
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
                "action": {"action1": action} if composite_action_dist else action,
            },
            names=[None, "time"],
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self, version):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        if version == 1:
            value = self._create_mock_value()
        else:
            value = None
        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
        )
        self.reset_parameters_recursive_test(loss_fn)

    def test_sac_list_qvalue_networks(self, version):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac()
        actor = self._create_mock_actor()
        qvalue1 = self._create_mock_qvalue()
        qvalue2 = self._create_mock_qvalue()
        if version == 1:
            value = self._create_mock_value()
        else:
            value = None
        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=[qvalue1, qvalue2],
            value_network=value,
            num_qvalue_nets=2,
        )
        with pytest.warns(
            UserWarning, match="No target network updater has been associated"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)
        assert "loss_qvalue" in loss.keys()

    @pytest.mark.parametrize("delay_value", (True, False))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_sac(
        self,
        delay_value,
        delay_actor,
        delay_qvalue,
        num_qvalue,
        device,
        version,
        td_est,
        composite_action_dist,
    ):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")

        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(
            device=device, composite_action_dist=composite_action_dist
        )

        # For composite action distributions, we need to pass the action_spec
        # explicitly because ProbabilisticActor doesn't preserve it properly
        if composite_action_dist:
            actor, action_spec = self._create_mock_actor(
                device=device,
                composite_action_dist=composite_action_dist,
                return_action_spec=True,
            )
        else:
            actor = self._create_mock_actor(
                device=device, composite_action_dist=composite_action_dist
            )
            action_spec = None
        qvalue = self._create_mock_qvalue(device=device)
        if version == 1:
            value = self._create_mock_value(device=device)
        else:
            value = None

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True
        if delay_value:
            kwargs["delay_value"] = True

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            action_spec=action_spec,
            **kwargs,
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
                if version == 1:
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
            elif k == "loss_value" and version == 1:
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
                if version == 1:
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
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                if version == 1:
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

    @pytest.mark.parametrize("delay_value", (True,))
    @pytest.mark.parametrize("delay_actor", (True,))
    @pytest.mark.parametrize("delay_qvalue", (True,))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", [None])
    @pytest.mark.parametrize("composite_action_dist", [False])
    def test_sac_deactivate_vmap(
        self,
        delay_value,
        delay_actor,
        delay_qvalue,
        num_qvalue,
        device,
        version,
        td_est,
        composite_action_dist,
    ):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")

        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(
            device=device, composite_action_dist=composite_action_dist
        )

        actor = self._create_mock_actor(
            device=device, composite_action_dist=composite_action_dist
        )
        qvalue = self._create_mock_qvalue(device=device)
        if version == 1:
            value = self._create_mock_value(device=device)
        else:
            value = None

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True
        if delay_value:
            kwargs["delay_value"] = True

        torch.manual_seed(0)
        loss_fn_vmap = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            deactivate_vmap=False,
            **kwargs,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_vmap.make_value_estimator(td_est)

        tdc = td.clone()
        torch.manual_seed(0)
        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss_vmap = loss_fn_vmap(td)
        td = tdc
        torch.manual_seed(0)
        loss_fn_no_vmap = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            deactivate_vmap=True,
            **kwargs,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_no_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_no_vmap.make_value_estimator(td_est)

        torch.manual_seed(0)
        with pytest.raises(
            NotImplementedError,
            match="This implementation is not supported for torch<2.7",
        ) if torch.__version__ < "2.7" else contextlib.nullcontext():
            with _check_td_steady(td), pytest.warns(
                UserWarning, match="No target network updater"
            ) if rl_warnings() else contextlib.nullcontext():
                loss_no_vmap = loss_fn_no_vmap(td)
            assert_allclose_td(loss_vmap, loss_no_vmap)

    @pytest.mark.parametrize("delay_value", (True, False))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_sac_state_dict(
        self,
        delay_value,
        delay_actor,
        delay_qvalue,
        num_qvalue,
        device,
        version,
        composite_action_dist,
    ):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")

        torch.manual_seed(self.seed)

        # For composite action distributions, we need to pass the action_spec
        # explicitly because ProbabilisticActor doesn't preserve it properly
        if composite_action_dist:
            actor, action_spec = self._create_mock_actor(
                device=device,
                composite_action_dist=composite_action_dist,
                return_action_spec=True,
            )
        else:
            actor = self._create_mock_actor(
                device=device, composite_action_dist=composite_action_dist
            )
            action_spec = None
        qvalue = self._create_mock_qvalue(device=device)
        if version == 1:
            value = self._create_mock_value(device=device)
        else:
            value = None

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True
        if delay_value:
            kwargs["delay_value"] = True

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            action_spec=action_spec,
            **kwargs,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            action_spec=action_spec,
            **kwargs,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("separate_losses", [False, True])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_sac_separate_losses(
        self,
        device,
        separate_losses,
        version,
        composite_action_dist,
        n_act=4,
    ):
        torch.manual_seed(self.seed)
        actor, qvalue, common, td = self._create_mock_common_layer_setup(
            n_act=n_act, composite_action_dist=composite_action_dist
        )

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            action_spec=Unbounded(shape=(n_act,)),
            num_qvalue_nets=1,
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
                elif k == "loss_qvalue":
                    common_layers_no = len(list(common.parameters()))
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
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
                            for p in loss_fn.qvalue_network_params.values(True, True)
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
                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_value", (True, False))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_sac_batcher(
        self,
        n,
        delay_value,
        delay_actor,
        delay_qvalue,
        num_qvalue,
        device,
        version,
        composite_action_dist,
    ):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_sac(
            device=device, composite_action_dist=composite_action_dist
        )

        # For composite action distributions, we need to pass the action_spec
        # explicitly because ProbabilisticActor doesn't preserve it properly
        if composite_action_dist:
            actor, action_spec = self._create_mock_actor(
                device=device,
                composite_action_dist=composite_action_dist,
                return_action_spec=True,
            )
        else:
            actor = self._create_mock_actor(
                device=device, composite_action_dist=composite_action_dist
            )
            action_spec = None
        qvalue = self._create_mock_qvalue(device=device)
        if version == 1:
            value = self._create_mock_value(device=device)
        else:
            value = None

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True
        if delay_value:
            kwargs["delay_value"] = True

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            action_spec=action_spec,
            **kwargs,
        )

        ms = MultiStep(gamma=0.9, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)
        with pytest.warns(
            UserWarning,
            match="No target network updater has been associated with this loss module",
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
            if version == 1:
                target_value = [
                    p.clone()
                    for p in loss_fn.target_value_network_params.values(
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
            if version == 1:
                target_value2 = [
                    p.clone()
                    for p in loss_fn.target_value_network_params.values(
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
            if version == 1:
                if loss_fn.delay_value:
                    assert all(
                        (p1 == p2).all() for p1, p2 in zip(target_value, target_value2)
                    )
                else:
                    assert not any(
                        (p1 == p2).any() for p1, p2 in zip(target_value, target_value2)
                    )

            # check that policy is updated after parameter update
            parameters = [p.clone() for p in actor.parameters()]
            for p in loss_fn.parameters():
                if p.requires_grad:
                    p.data += torch.randn_like(p)
            assert all(
                (p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters())
            )

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_sac_tensordict_keys(self, td_est, version, composite_action_dist):
        td = self._create_mock_data_sac(composite_action_dist=composite_action_dist)

        # For composite action distributions, we need to pass the action_spec
        # explicitly because ProbabilisticActor doesn't preserve it properly
        if composite_action_dist:
            actor, action_spec = self._create_mock_actor(
                composite_action_dist=composite_action_dist,
                return_action_spec=True,
            )
        else:
            actor = self._create_mock_actor(composite_action_dist=composite_action_dist)
            action_spec = None
        qvalue = self._create_mock_qvalue()
        if version == 1:
            value = self._create_mock_value()
        else:
            value = None

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=2,
            loss_function="l2",
            action_spec=action_spec,
        )

        default_keys = {
            "priority": "td_error",
            "value": "state_value",
            "state_action_value": "state_action_value",
            "action": "action",
            "log_prob": "action_log_prob",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value = self._create_mock_value()
        loss_fn = SACLoss(
            actor,
            value,
            loss_function="l2",
        )

        key_mapping = {
            "value": ("value", "state_value_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_sac_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key, version
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key, action_key=action_key
        )
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
            action_key=action_key,
            out_keys=["state_action_value"],
        )
        if version == 1:
            value = self._create_mock_value(observation_key=observation_key)
        else:
            value = None

        loss = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
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

        # setting the seed for each loss so that drawing the random samples from value network
        # leads to same numbers for both runs
        torch.manual_seed(self.seed)
        with pytest.warns(
            UserWarning, match="No target network updater"
        ) if rl_warnings() else contextlib.nullcontext():
            loss_val = loss(**kwargs)

        torch.manual_seed(self.seed)

        SoftUpdate(loss, eps=0.5)

        loss_val_td = loss(td)

        if version == 1:
            assert len(loss_val) == 6
        elif version == 2:
            assert len(loss_val) == 5

        torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
        torch.testing.assert_close(loss_val_td.get("loss_qvalue"), loss_val[1])
        torch.testing.assert_close(loss_val_td.get("loss_alpha"), loss_val[2])
        torch.testing.assert_close(loss_val_td.get("alpha"), loss_val[3])
        torch.testing.assert_close(loss_val_td.get("entropy"), loss_val[4])
        if version == 1:
            torch.testing.assert_close(loss_val_td.get("loss_value"), loss_val[5])
        # test select
        torch.manual_seed(self.seed)
        loss.select_out_keys("loss_actor", "loss_alpha")
        if torch.__version__ >= "2.0.0":
            loss_actor, loss_alpha = loss(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_actor, loss_alpha = loss(**kwargs)
            return
        assert loss_actor == loss_val_td["loss_actor"]
        assert loss_alpha == loss_val_td["loss_alpha"]

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_sac_terminating(
        self, action_key, observation_key, reward_key, done_key, terminated_key, version
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key, action_key=action_key
        )
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
            action_key=action_key,
            out_keys=["state_action_value"],
        )
        if version == 1:
            value = self._create_mock_value(observation_key=observation_key)
        else:
            value = None

        loss = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            skip_done_states=True,
        )
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
        )

        torch.manual_seed(self.seed)

        SoftUpdate(loss, eps=0.5)

        done = td.get(("next", done_key))
        while not (done.any() and not done.all()):
            done.bernoulli_(0.1)
        obs_nan = td.get(("next", terminated_key))
        obs_nan[done.squeeze(-1)] = float("nan")

        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": done,
            f"next_{terminated_key}": obs_nan,
            f"next_{observation_key}": td.get(("next", observation_key)),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")
        assert loss(td).isfinite().all()

    def test_state_dict(self, version):
        if version == 1:
            pytest.skip("Test not implemented for version 1.")
        model = torch.nn.Linear(3, 4)
        actor_module = TensorDictModule(model, in_keys=["obs"], out_keys=["logits"])
        policy = ProbabilisticActor(
            module=actor_module,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=TanhDelta,
        )
        value = ValueOperator(module=model, in_keys=["obs"], out_keys="value")

        loss = SACLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        state = loss.state_dict()

        loss = SACLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        loss.load_state_dict(state)

        # with an access in between
        loss = SACLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        loss.target_entropy
        state = loss.state_dict()

        loss = SACLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        loss.load_state_dict(state)

    @pytest.mark.parametrize("action_dim", [1, 2, 4, 8])
    def test_sac_target_entropy_auto(self, version, action_dim):
        """Regression test for issue #3291: target_entropy='auto' should be -dim(A)."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(action_dim=action_dim)
        qvalue = self._create_mock_qvalue(action_dim=action_dim)
        if version == 1:
            value = self._create_mock_value(action_dim=action_dim)
        else:
            value = None

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
        )
        # target_entropy="auto" should compute -action_dim
        assert (
            loss_fn.target_entropy.item() == -action_dim
        ), f"target_entropy should be -{action_dim}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("target_entropy", [-1.0, -2.0, -5.0, 0.0])
    def test_sac_target_entropy_explicit(self, version, target_entropy):
        """Regression test for explicit target_entropy values."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        if version == 1:
            value = self._create_mock_value()
        else:
            value = None

        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            target_entropy=target_entropy,
        )
        assert (
            loss_fn.target_entropy.item() == target_entropy
        ), f"target_entropy should be {target_entropy}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    @pytest.mark.parametrize("composite_action_dist", [True, False])
    def test_sac_reduction(self, reduction, version, composite_action_dist):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_sac(
            device=device, composite_action_dist=composite_action_dist
        )
        # For composite action distributions, we need to pass the action_spec
        # explicitly because ProbabilisticActor doesn't preserve it properly
        if composite_action_dist:
            actor, action_spec = self._create_mock_actor(
                device=device,
                composite_action_dist=composite_action_dist,
                return_action_spec=True,
            )
        else:
            actor = self._create_mock_actor(
                device=device, composite_action_dist=composite_action_dist
            )
            action_spec = None
        qvalue = self._create_mock_qvalue(device=device)
        if version == 1:
            value = self._create_mock_value(device=device)
        else:
            value = None
        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            loss_function="l2",
            delay_qvalue=False,
            delay_actor=False,
            delay_value=False,
            reduction=reduction,
            action_spec=action_spec,
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

    def test_sac_prioritized_weights(self, version):
        """Test SAC with prioritized replay buffer weighted loss reduction."""
        if version != 2:
            pytest.skip("Test not implemented for version 1.")
        torch.manual_seed(42)
        n_obs = 4
        n_act = 2
        batch_size = 32
        buffer_size = 100

        # Actor network
        actor_net = nn.Sequential(
            nn.Linear(n_obs, 64),
            nn.ReLU(),
            nn.Linear(64, 2 * n_act),
            NormalParamExtractor(),
        )
        actor_module = TensorDictModule(
            actor_net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=actor_module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=True,
            spec=Bounded(
                low=-torch.ones(n_act), high=torch.ones(n_act), shape=(n_act,)
            ),
        )

        # Q-value network
        qvalue_net = MLP(in_features=n_obs + n_act, out_features=1, num_cells=[64, 64])
        qvalue = ValueOperator(module=qvalue_net, in_keys=["observation", "action"])

        # Value network for SAC v1
        value_net = MLP(in_features=n_obs, out_features=1, num_cells=[64, 64])
        value = ValueOperator(module=value_net, in_keys=["observation"])

        # Create SAC loss
        loss_fn = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=2,
        )
        SoftUpdate(loss_fn, eps=0.5)
        loss_fn.make_value_estimator()

        # Create prioritized replay buffer
        rb = TensorDictPrioritizedReplayBuffer(
            alpha=0.7,
            beta=0.9,
            storage=LazyTensorStorage(buffer_size),
            batch_size=batch_size,
            priority_key="td_error",
        )

        # Create initial data
        initial_data = TensorDict(
            {
                "observation": torch.randn(buffer_size, n_obs),
                "action": torch.randn(buffer_size, n_act).clamp(-1, 1),
                ("next", "observation"): torch.randn(buffer_size, n_obs),
                ("next", "reward"): torch.randn(buffer_size, 1),
                ("next", "done"): torch.zeros(buffer_size, 1, dtype=torch.bool),
                ("next", "terminated"): torch.zeros(buffer_size, 1, dtype=torch.bool),
            },
            batch_size=[buffer_size],
        )
        rb.extend(initial_data)

        # Sample (weights should all be identical initially)
        sample1 = rb.sample()
        assert "priority_weight" in sample1.keys()
        weights1 = sample1["priority_weight"]
        assert torch.allclose(weights1, weights1[0], atol=1e-5)

        # Run loss to get priorities
        loss_fn(sample1)
        assert "td_error" in sample1.keys()

        # Update replay buffer with new priorities
        rb.update_tensordict_priority(sample1)

        # Sample again - weights should now be non-equal
        sample2 = rb.sample()
        weights2 = sample2["priority_weight"]
        assert weights2.std() > 1e-5

        # Run loss again with varied weights
        loss_out2 = loss_fn(sample2)
        assert torch.isfinite(loss_out2["loss_qvalue"])

        # Verify weighted vs unweighted differ
        loss_fn_no_weights = SACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=2,
            use_prioritized_weights=False,
        )
        SoftUpdate(loss_fn_no_weights, eps=0.5)
        loss_fn_no_weights.make_value_estimator()
        loss_fn_no_weights.qvalue_network_params = loss_fn.qvalue_network_params
        loss_fn_no_weights.target_qvalue_network_params = (
            loss_fn.target_qvalue_network_params
        )
        loss_fn_no_weights.actor_network_params = loss_fn.actor_network_params
        loss_fn_no_weights.value_network_params = loss_fn.value_network_params
        loss_fn_no_weights.target_value_network_params = (
            loss_fn.target_value_network_params
        )

        loss_out_no_weights = loss_fn_no_weights(sample2)
        # Weighted and unweighted should differ (in general)
        assert torch.isfinite(loss_out_no_weights["loss_qvalue"])


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestDiscreteSAC(LossModuleTestBase):
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
            module=module, in_keys=[observation_key], out_keys=["action_value"]
        )
        return qvalue.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_sac(
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
            action_value = torch.randn(batch, atoms, action_dim).softmax(-2)
            action = (
                (action_value[..., 0, :] == action_value[..., 0, :].max(-1, True)[0])
                .to(torch.long)
                .to(device)
            )
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

    def _create_seq_mock_data_sac(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action_value = torch.randn(
                batch, T, atoms, action_dim, device=device
            ).softmax(-2)
            action = (
                action_value[..., 0, :] == action_value[..., 0, :].max(-1, True)[0]
            ).to(torch.long)
        else:
            action_value = torch.randn(batch, T, action_dim, device=device)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)

        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
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
                "action_value": action_value.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            names=[None, "time"],
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            action_space="one-hot",
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("target_entropy_weight", [0.01, 0.5, 0.99])
    @pytest.mark.parametrize("target_entropy", ["auto", 1.0, 0.1, 0.0])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_discrete_sac(
        self,
        delay_qvalue,
        num_qvalue,
        device,
        target_entropy_weight,
        target_entropy,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        kwargs = {}
        if delay_qvalue:
            kwargs["delay_qvalue"] = True

        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            num_qvalue_nets=num_qvalue,
            target_entropy_weight=target_entropy_weight,
            target_entropy=target_entropy,
            loss_function="l2",
            action_space="one-hot",
            **kwargs,
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

    @pytest.mark.parametrize("delay_qvalue", (True,))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("target_entropy_weight", [0.5])
    @pytest.mark.parametrize("target_entropy", ["auto"])
    @pytest.mark.parametrize("td_est", [None])
    def test_discrete_sac_deactivate_vmap(
        self,
        delay_qvalue,
        num_qvalue,
        device,
        target_entropy_weight,
        target_entropy,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        kwargs = {}
        if delay_qvalue:
            kwargs["delay_qvalue"] = True

        torch.manual_seed(0)
        loss_fn_vmap = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            num_qvalue_nets=num_qvalue,
            target_entropy_weight=target_entropy_weight,
            target_entropy=target_entropy,
            loss_function="l2",
            action_space="one-hot",
            deactivate_vmap=False,
            **kwargs,
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
        loss_fn_no_vmap = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            num_qvalue_nets=num_qvalue,
            target_entropy_weight=target_entropy_weight,
            target_entropy=target_entropy,
            loss_function="l2",
            action_space="one-hot",
            deactivate_vmap=True,
            **kwargs,
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

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("target_entropy_weight", [0.5])
    @pytest.mark.parametrize("target_entropy", ["auto"])
    def test_discrete_sac_state_dict(
        self,
        delay_qvalue,
        num_qvalue,
        device,
        target_entropy_weight,
        target_entropy,
    ):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        kwargs = {}
        if delay_qvalue:
            kwargs["delay_qvalue"] = True

        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            num_qvalue_nets=num_qvalue,
            target_entropy_weight=target_entropy_weight,
            target_entropy=target_entropy,
            loss_function="l2",
            action_space="one-hot",
            **kwargs,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            num_qvalue_nets=num_qvalue,
            target_entropy_weight=target_entropy_weight,
            target_entropy=target_entropy,
            loss_function="l2",
            action_space="one-hot",
            **kwargs,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("action_dim", [2, 4, 8])
    @pytest.mark.parametrize("target_entropy_weight", [0.5, 0.98])
    def test_discrete_sac_target_entropy_auto(self, action_dim, target_entropy_weight):
        """Regression test for target_entropy='auto' in DiscreteSACLoss."""
        import numpy as np

        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(action_dim=action_dim)
        qvalue = self._create_mock_qvalue(action_dim=action_dim)

        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=action_dim,
            target_entropy_weight=target_entropy_weight,
            action_space="one-hot",
        )
        # target_entropy="auto" should compute -log(1/num_actions) * target_entropy_weight
        expected = -float(np.log(1.0 / action_dim) * target_entropy_weight)
        assert (
            abs(loss_fn.target_entropy.item() - expected) < 1e-5
        ), f"target_entropy should be {expected}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("target_entropy_weight", [0.01, 0.5, 0.99])
    @pytest.mark.parametrize("target_entropy", ["auto", 1.0, 0.1, 0.0])
    def test_discrete_sac_batcher(
        self,
        n,
        delay_qvalue,
        num_qvalue,
        device,
        target_entropy_weight,
        target_entropy,
        gamma=0.9,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_sac(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        kwargs = {}
        if delay_qvalue:
            kwargs["delay_qvalue"] = True
        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            target_entropy_weight=target_entropy_weight,
            target_entropy=target_entropy,
            action_space="one-hot",
            **kwargs,
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
            assert all((p1 == p2).all() for p1, p2 in zip(target_actor, target_actor2))
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
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_discrete_sac_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()

        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            loss_function="l2",
            action_space="one-hot",
        )

        default_keys = {
            "priority": "td_error",
            "value": "state_value",
            "action": "action",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }
        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        qvalue = self._create_mock_qvalue()
        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            loss_function="l2",
            action_space="one-hot",
        )

        key_mapping = {
            "value": ("value", "state_value_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_discrete_sac_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key, action_key=action_key
        )
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
        )

        loss = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec[action_key].space.n,
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
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            loss_val = loss(**kwargs)
            loss_val_td = loss(td)

            torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
            torch.testing.assert_close(loss_val_td.get("loss_qvalue"), loss_val[1])
            torch.testing.assert_close(loss_val_td.get("loss_alpha"), loss_val[2])
            torch.testing.assert_close(loss_val_td.get("alpha"), loss_val[3])
            torch.testing.assert_close(loss_val_td.get("entropy"), loss_val[4])
            # test select
            torch.manual_seed(self.seed)
            loss.select_out_keys("loss_actor", "loss_alpha")
            if torch.__version__ >= "2.0.0":
                loss_actor, loss_alpha = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, loss_alpha = loss(**kwargs)
                return
            assert loss_actor == loss_val_td["loss_actor"]
            assert loss_alpha == loss_val_td["loss_alpha"]

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_discrete_sac_terminating(
        self, action_key, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key, action_key=action_key
        )
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
        )

        loss = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec[action_key].space.n,
            action_space="one-hot",
            skip_done_states=True,
        )
        loss.set_keys(
            action=action_key,
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
        )

        SoftUpdate(loss, eps=0.5)

        torch.manual_seed(0)
        done = td.get(("next", done_key))
        while not (done.any() and not done.all()):
            done = done.bernoulli_(0.1)
        obs_none = td.get(("next", observation_key))
        obs_none[done.squeeze(-1)] = float("nan")
        kwargs = {
            action_key: td.get(action_key),
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": done,
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": obs_none,
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")
        assert loss(td).isfinite().all()

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_discrete_sac_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_sac(device=device)
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        loss_fn = DiscreteSACLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_actions=actor.spec["action"].space.n,
            loss_function="l2",
            action_space="one-hot",
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


class TestCrossQ(LossModuleTestBase):
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
            out_keys=[action_key],
        )
        return actor.to(device)

    def _create_mock_qvalue(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        action_key="action",
        out_keys=None,
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

    def _create_mock_common_layer_setup(
        self, n_obs=3, n_act=4, ncells=4, batch=2, n_hidden=2
    ):
        common = MLP(
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
        qvalue = MLP(
            in_features=n_hidden + n_act,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        batch = [batch]
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": torch.randn(*batch, n_act),
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
        )
        common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
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
        qvalue_head = Mod(
            qvalue, in_keys=["hidden", "action"], out_keys=["state_action_value"]
        )
        qvalue = Seq(common, qvalue_head)
        return actor, qvalue, common, td

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_crossq(
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

    def _create_seq_mock_data_crossq(
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
        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_crossq(
        self,
        num_qvalue,
        device,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_crossq(device=device)
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td):
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

    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_crossq_deactivate_vmap(
        self,
        num_qvalue,
        device,
        td_est,
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_crossq(device=device)
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        torch.manual_seed(0)
        loss_fn_vmap = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            deactivate_vmap=False,
        )

        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_vmap.make_value_estimator(td_est)

        tdc = td.clone()
        with _check_td_steady(td):
            torch.manual_seed(1)
            loss_vmap = loss_fn_vmap(td)

        td = tdc

        torch.manual_seed(0)
        loss_fn_no_vmap = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
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
            with _check_td_steady(td):
                torch.manual_seed(1)
                loss_no_vmap = loss_fn_no_vmap(td)
            assert_allclose_td(loss_vmap, loss_no_vmap)

    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_crossq_state_dict(
        self,
        num_qvalue,
        device,
    ):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
        )
        sd = loss_fn.state_dict()
        loss_fn2 = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_crossq_separate_losses(
        self,
        separate_losses,
        device,
    ):
        n_act = 4
        torch.manual_seed(self.seed)
        actor, qvalue, common, td = self._create_mock_common_layer_setup(n_act=n_act)

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            action_spec=Unbounded(shape=(n_act,)),
            num_qvalue_nets=1,
            separate_losses=separate_losses,
        )
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
                common_layers_no = len(list(common.parameters()))
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                if separate_losses:
                    common_layers = itertools.islice(
                        loss_fn.qvalue_network_params.values(True, True),
                        common_layers_no,
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all() for p in common_layers
                    )
                    qvalue_layers = itertools.islice(
                        loss_fn.qvalue_network_params.values(True, True),
                        common_layers_no,
                        None,
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all() for p in qvalue_layers
                    )
                else:
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.qvalue_network_params.values(True, True)
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
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_crossq_batcher(
        self,
        n,
        num_qvalue,
        device,
    ):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_crossq(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
        )

        ms = MultiStep(gamma=0.9, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)

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
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        target_actor2 = [
            p.clone()
            for p in loss_fn.target_actor_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]

        assert not any((p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2))

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_crossq_tensordict_keys(self, td_est):

        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=2,
            loss_function="l2",
        )

        default_keys = {
            "priority": "td_error",
            "state_action_value": "state_action_value",
            "action": "action",
            "log_prob": "_log_prob",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        qvalue = self._create_mock_qvalue()
        loss_fn = CrossQLoss(
            actor,
            qvalue,
            loss_function="l2",
        )

        key_mapping = {
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_crossq_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_crossq(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key, action_key=action_key
        )
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
            action_key=action_key,
            out_keys=["state_action_value"],
        )

        loss = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
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

        # setting the seed for each loss so that drawing the random samples from value network
        # leads to same numbers for both runs
        torch.manual_seed(self.seed)
        loss_val = loss(**kwargs)

        torch.manual_seed(self.seed)

        loss_val_td = loss(td)
        assert len(loss_val) == 5

        torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
        torch.testing.assert_close(loss_val_td.get("loss_qvalue"), loss_val[1])
        torch.testing.assert_close(loss_val_td.get("loss_alpha"), loss_val[2])
        torch.testing.assert_close(loss_val_td.get("alpha"), loss_val[3])
        torch.testing.assert_close(loss_val_td.get("entropy"), loss_val[4])

        # test select
        torch.manual_seed(self.seed)
        loss.select_out_keys("loss_actor", "loss_alpha")
        if torch.__version__ >= "2.0.0":
            loss_actor, loss_alpha = loss(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_actor, loss_alpha = loss(**kwargs)
            return
        assert loss_actor == loss_val_td["loss_actor"]
        assert loss_alpha == loss_val_td["loss_alpha"]

    def test_state_dict(
        self,
    ):

        model = torch.nn.Linear(3, 4)
        actor_module = TensorDictModule(model, in_keys=["obs"], out_keys=["logits"])
        policy = ProbabilisticActor(
            module=actor_module,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=TanhDelta,
        )
        value = ValueOperator(module=model, in_keys=["obs"], out_keys="value")

        loss = CrossQLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        state = loss.state_dict()

        loss = CrossQLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        loss.load_state_dict(state)

        # with an access in between
        loss = CrossQLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        loss.target_entropy
        state = loss.state_dict()

        loss = CrossQLoss(
            actor_network=policy,
            qvalue_network=value,
            action_spec=Unbounded(shape=(2,)),
        )
        loss.load_state_dict(state)

    @pytest.mark.parametrize("action_dim", [1, 2, 4, 8])
    def test_crossq_target_entropy_auto(self, action_dim):
        """Regression test for target_entropy='auto' should be -dim(A)."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(action_dim=action_dim)
        qvalue = self._create_mock_qvalue(action_dim=action_dim)

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        # target_entropy="auto" should compute -action_dim
        assert (
            loss_fn.target_entropy.item() == -action_dim
        ), f"target_entropy should be -{action_dim}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("target_entropy", [-1.0, -2.0, -5.0, 0.0])
    def test_crossq_target_entropy_explicit(self, target_entropy):
        """Regression test for issue #3309: explicit target_entropy should work."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            target_entropy=target_entropy,
        )
        assert (
            loss_fn.target_entropy.item() == target_entropy
        ), f"target_entropy should be {target_entropy}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_crossq_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_crossq(device=device)
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = CrossQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
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


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestREDQ(LossModuleTestBase):
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
            distribution_class=TanhNormal,
            return_log_prob=True,
            spec=action_spec,
            out_keys=[action_key],
        )
        return actor.to(device)

    def _create_mock_qvalue(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key="observation",
        action_key="action",
        out_keys=None,
    ):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        qvalue = ValueOperator(
            module=module, in_keys=[observation_key, action_key], out_keys=out_keys
        )
        return qvalue.to(device)

    def _create_mock_common_layer_setup(
        self, n_obs=3, n_act=4, ncells=4, batch=2, n_hidden=2
    ):
        common = MLP(
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
        value = MLP(
            in_features=n_hidden + n_act,
            num_cells=ncells,
            depth=1,
            out_features=1,
        )
        batch = [batch]
        td = TensorDict(
            {
                "obs": torch.randn(*batch, n_obs),
                "action": torch.randn(*batch, n_act),
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
        )
        common = Mod(common, in_keys=["obs"], out_keys=["hidden"])
        actor = ProbSeq(
            common,
            Mod(actor_net, in_keys=["hidden"], out_keys=["param"]),
            Mod(NormalParamExtractor(), in_keys=["param"], out_keys=["loc", "scale"]),
            ProbMod(
                in_keys=["loc", "scale"],
                out_keys=["action"],
                distribution_class=TanhNormal,
                return_log_prob=True,
            ),
        )
        value_head = Mod(
            value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
        )
        value = Seq(common, value_head)
        value(actor(td))
        return actor, value, common, td

    def _create_shared_mock_actor_qvalue(
        self, batch=2, obs_dim=3, action_dim=4, hidden_dim=5, device="cpu"
    ):
        class CommonClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim, hidden_dim)

            def forward(self, obs):
                return self.linear(obs)

        class ActorClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Sequential(
                    nn.Linear(hidden_dim, 2 * action_dim), NormalParamExtractor()
                )

            def forward(self, hidden):
                return self.linear(hidden)

        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(hidden_dim + action_dim, 1)

            def forward(self, hidden, act):
                return self.linear(torch.cat([hidden, act], -1))

        common = TensorDictModule(
            CommonClass(), in_keys=["observation"], out_keys=["hidden"]
        )
        actor_subnet = ProbabilisticActor(
            TensorDictModule(
                ActorClass(), in_keys=["hidden"], out_keys=["loc", "scale"]
            ),
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=True,
        )
        qvalue_subnet = ValueOperator(ValueClass(), in_keys=["hidden", "action"])
        model = ActorCriticOperator(common, actor_subnet, qvalue_subnet)
        return model.to(device)

    def _create_mock_data_redq(
        self,
        batch=16,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        observation_key="observation",
        action_key="action",
        reward_key="reward",
        done_key="done",
        terminated_key="terminated",
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

    def _create_seq_mock_data_redq(
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
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
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
        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_redq(self, delay_qvalue, num_qvalue, device, td_est):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with (
            pytest.warns(
                UserWarning,
                match="No target network updater has been associated with this loss module",
            )
            if delay_qvalue and rl_warnings()
            else contextlib.nullcontext()
        ):
            with _check_td_steady(td):
                loss = loss_fn(td)

            # check td is left untouched
            assert loss_fn.tensor_keys.priority in td.keys()

            # check that losses are independent
            for k in loss.keys():
                if not k.startswith("loss"):
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

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_redq_state_dict(self, delay_qvalue, num_qvalue, device):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("action_dim", [1, 2, 4, 8])
    def test_redq_target_entropy_auto(self, action_dim):
        """Regression test for target_entropy='auto' should be -dim(A)."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(action_dim=action_dim)
        qvalue = self._create_mock_qvalue(action_dim=action_dim)

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        # target_entropy="auto" should compute -action_dim
        assert (
            loss_fn.target_entropy.item() == -action_dim
        ), f"target_entropy should be -{action_dim}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_redq_separate_losses(self, separate_losses):
        torch.manual_seed(self.seed)

        actor, qvalue, common, td = self._create_mock_common_layer_setup()

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            target_entropy=0.0,
            separate_losses=separate_losses,
        )

        with pytest.warns(
            UserWarning,
            match="No target network updater has been associated with this loss module",
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

            # check that losses are independent
            for k in loss.keys():
                if not k.startswith("loss"):
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
                elif k == "loss_qvalue":
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(
                            include_nested=True, leaves_only=True
                        )
                    )
                    if separate_losses:
                        common_layers_no = len(list(common.parameters()))
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
                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_redq_deprecated_separate_losses(self, separate_losses):
        torch.manual_seed(self.seed)

        actor, qvalue, common, td = self._create_mock_common_layer_setup()

        loss_fn = REDQLoss_deprecated(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
            target_entropy=0.0,
            separate_losses=separate_losses,
        )

        with pytest.warns(
            UserWarning,
            match="No target network updater has been associated with this loss module",
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

        SoftUpdate(loss_fn, eps=0.5)

        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
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
            elif k == "loss_qvalue":
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(
                        include_nested=True, leaves_only=True
                    )
                )
                if separate_losses:
                    common_layers_no = len(list(common.parameters()))
                    common_layers = itertools.islice(
                        loss_fn.qvalue_network_params.values(True, True),
                        common_layers_no,
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all() for p in common_layers
                    )
                    qvalue_layers = itertools.islice(
                        loss_fn.qvalue_network_params.values(True, True),
                        common_layers_no,
                        None,
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all() for p in qvalue_layers
                    )
                else:
                    assert not any(
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
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_redq_shared(self, delay_qvalue, num_qvalue, device):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq(device=device)

        actor_critic = self._create_shared_mock_actor_qvalue(device=device)
        actor = actor_critic.get_policy_operator()
        qvalue = actor_critic.get_critic_operator()

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
            target_entropy=0.0,
        )

        if delay_qvalue:
            target_updater = SoftUpdate(loss_fn, tau=0.05)

        with _check_td_steady(td):
            loss = loss_fn(td)

        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(True, True)
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(True, True)
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(True, True)
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(True, True)
                    if isinstance(p, nn.Parameter)
                )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(True, True)
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params.values(True, True)
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_redq_batched(self, delay_qvalue, num_qvalue, device, td_est):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = REDQLoss(
            actor_network=deepcopy(actor),
            qvalue_network=deepcopy(qvalue),
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        loss_class_deprec = (
            REDQLoss_deprecated if not delay_qvalue else DoubleREDQLoss_deprecated
        )
        loss_fn_deprec = loss_class_deprec(
            actor_network=deepcopy(actor),
            qvalue_network=deepcopy(qvalue),
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_deprec.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_deprec.make_value_estimator(td_est)

        td_clone1 = td.clone()
        td_clone2 = td.clone()
        torch.manual_seed(0)
        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_qvalue and rl_warnings()
            else contextlib.nullcontext()
        ):
            with _check_td_steady(td_clone1):
                loss_fn(td_clone1)

            torch.manual_seed(0)
            with _check_td_steady(td_clone2):
                loss_fn_deprec(td_clone2)

        # TODO: find a way to compare the losses: problem is that we sample actions either sequentially or in batch,
        #  so setting seed has little impact

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_default_devices())
    def test_redq_batcher(self, n, delay_qvalue, num_qvalue, device, gamma=0.9):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_redq(device=device)
        assert td.names == td.get("next").names

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)

        td_clone = td.clone()
        assert td_clone.names == td_clone.get("next").names
        ms_td = ms(td_clone)
        assert ms_td.names == ms_td.get("next").names

        torch.manual_seed(0)
        np.random.seed(0)

        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_qvalue and rl_warnings()
            else contextlib.nullcontext()
        ):
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
            target_actor = loss_fn.target_actor_network_params.clone().values(
                include_nested=True, leaves_only=True
            )
            target_qvalue = loss_fn.target_qvalue_network_params.clone().values(
                include_nested=True, leaves_only=True
            )
            for p in loss_fn.parameters():
                if p.requires_grad:
                    p.data += torch.randn_like(p)
            target_actor2 = loss_fn.target_actor_network_params.clone().values(
                include_nested=True, leaves_only=True
            )
            target_qvalue2 = loss_fn.target_qvalue_network_params.clone().values(
                include_nested=True, leaves_only=True
            )
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
            actorp_set = set(actor.parameters())
            loss_fnp_set = set(loss_fn.parameters())
            assert len(actorp_set.intersection(loss_fnp_set)) == len(actorp_set)
            parameters = [p.clone() for p in actor.parameters()]
            for p in loss_fn.parameters():
                if p.requires_grad:
                    p.data += torch.randn_like(p)
            assert all(
                (p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters())
            )

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_redq_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
        )

        default_keys = {
            "priority": "td_error",
            "action": "action",
            "value": "state_value",
            "sample_log_prob": "action_log_prob",
            "state_action_value": "state_action_value",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }
        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        qvalue = self._create_mock_qvalue()
        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            loss_function="l2",
        )

        key_mapping = {
            "value": ("value", "state_value_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_key", ["action", "action2"])
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    @pytest.mark.parametrize("deprec", [True, False])
    def test_redq_notensordict(
        self, action_key, observation_key, reward_key, done_key, terminated_key, deprec
    ):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq(
            action_key=action_key,
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )

        actor = self._create_mock_actor(
            observation_key=observation_key,
            action_key=action_key,
        )
        qvalue = self._create_mock_qvalue(
            observation_key=observation_key,
            action_key=action_key,
            out_keys=["state_action_value"],
        )

        if deprec:
            cls = REDQLoss_deprecated
        else:
            cls = REDQLoss
        loss = cls(
            actor_network=actor,
            qvalue_network=qvalue,
        )
        if deprec:
            loss.set_keys(
                action=action_key,
                reward=reward_key,
                done=done_key,
                terminated=terminated_key,
                log_prob=_add_suffix(action_key, "_log_prob"),
            )
        else:
            loss.set_keys(
                action=action_key,
                reward=reward_key,
                done=done_key,
                terminated=terminated_key,
                sample_log_prob=_add_suffix(action_key, "_log_prob"),
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

        torch.manual_seed(self.seed)
        with pytest.warns(
            UserWarning,
            match="No target network updater has been associated with this loss module",
        ) if rl_warnings() else contextlib.nullcontext():
            loss_val = loss(**kwargs)
            torch.manual_seed(self.seed)
            loss_val_td = loss(td)

            torch.testing.assert_close(loss_val_td.get("loss_actor"), loss_val[0])
            torch.testing.assert_close(loss_val_td.get("loss_qvalue"), loss_val[1])
            torch.testing.assert_close(loss_val_td.get("loss_alpha"), loss_val[2])
            torch.testing.assert_close(loss_val_td.get("alpha"), loss_val[3])
            torch.testing.assert_close(loss_val_td.get("entropy"), loss_val[4])
            if not deprec:
                torch.testing.assert_close(
                    loss_val_td.get("state_action_value_actor"), loss_val[5]
                )
                torch.testing.assert_close(
                    loss_val_td.get("action_log_prob_actor"), loss_val[6]
                )
                torch.testing.assert_close(
                    loss_val_td.get("next.state_value"), loss_val[7]
                )
                torch.testing.assert_close(loss_val_td.get("target_value"), loss_val[8])
            # test select
            torch.manual_seed(self.seed)
            loss.select_out_keys("loss_actor", "loss_alpha")
            if torch.__version__ >= "2.0.0":
                loss_actor, loss_alpha = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, loss_alpha = loss(**kwargs)
                return
            assert loss_actor == loss_val_td["loss_actor"]
            assert loss_alpha == loss_val_td["loss_alpha"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    @pytest.mark.parametrize("deprecated_loss", [True, False])
    def test_redq_reduction(self, reduction, deprecated_loss):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_redq(device=device)
        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        if deprecated_loss:
            loss_fn = REDQLoss_deprecated(
                actor_network=actor,
                qvalue_network=qvalue,
                loss_function="l2",
                delay_qvalue=False,
                reduction=reduction,
            )
        else:
            loss_fn = REDQLoss(
                actor_network=actor,
                qvalue_network=qvalue,
                loss_function="l2",
                delay_qvalue=False,
                reduction=reduction,
                scalar_output_mode="exclude" if reduction == "none" else None,
            )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape[-1] == td.shape[0]
        else:
            for key in loss.keys():
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])
