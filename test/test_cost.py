# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import functools
import operator
import warnings
from copy import deepcopy

from packaging import version as pack_version
from tensordict.nn import InteractionType

_has_functorch = True
try:
    import functorch as ft  # noqa

    make_functional_with_buffers = ft.make_functional_with_buffers
    FUNCTORCH_ERR = ""
except ImportError as err:
    _has_functorch = False
    FUNCTORCH_ERR = str(err)

import numpy as np
import pytest
import torch
from _utils_internal import dtype_fixture, get_available_devices  # noqa
from mocking_classes import ContinuousActionConvMockEnv
from tensordict.nn import get_functional, NormalParamExtractor, TensorDictModule

# from torchrl.data.postprocs.utils import expand_as_right
from tensordict.tensordict import assert_allclose_td, TensorDict
from torch import autograd, nn
from torchrl.data import (
    BoundedTensorSpec,
    CompositeSpec,
    DiscreteTensorSpec,
    MultiOneHotDiscreteTensorSpec,
    OneHotDiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
)
from torchrl.data.postprocs.postprocs import MultiStep
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.modules import (
    DistributionalQValueActor,
    OneHotCategorical,
    QValueActor,
    SafeModule,
    SafeProbabilisticModule,
    SafeProbabilisticTensorDictSequential,
    SafeSequential,
    WorldModelWrapper,
)
from torchrl.modules.distributions.continuous import NormalParamWrapper, TanhNormal
from torchrl.modules.models.model_based import (
    DreamerActor,
    ObsDecoder,
    ObsEncoder,
    RSSMPosterior,
    RSSMPrior,
    RSSMRollout,
)
from torchrl.modules.models.models import MLP
from torchrl.modules.tensordict_module.actors import (
    Actor,
    ActorCriticOperator,
    ActorValueOperator,
    ProbabilisticActor,
    ValueOperator,
)
from torchrl.modules.utils import Buffer
from torchrl.objectives import (
    A2CLoss,
    ClipPPOLoss,
    DDPGLoss,
    DiscreteSACLoss,
    DistributionalDQNLoss,
    DQNLoss,
    DreamerActorLoss,
    DreamerModelLoss,
    DreamerValueLoss,
    IQLLoss,
    KLPENPPOLoss,
    PPOLoss,
    SACLoss,
    TD3Loss,
)
from torchrl.objectives.common import LossModule
from torchrl.objectives.deprecated import DoubleREDQLoss_deprecated, REDQLoss_deprecated
from torchrl.objectives.redq import REDQLoss
from torchrl.objectives.reinforce import ReinforceLoss
from torchrl.objectives.utils import (
    HardUpdate,
    hold_out_net,
    SoftUpdate,
    ValueEstimators,
)
from torchrl.objectives.value.advantages import GAE, TD1Estimator, TDLambdaEstimator
from torchrl.objectives.value.functional import (
    _transpose_time,
    generalized_advantage_estimate,
    td0_advantage_estimate,
    td1_advantage_estimate,
    td_lambda_advantage_estimate,
    vec_generalized_advantage_estimate,
    vec_td1_advantage_estimate,
    vec_td_lambda_advantage_estimate,
)
from torchrl.objectives.value.utils import (
    _custom_conv1d,
    _get_num_per_traj,
    _get_num_per_traj_init,
    _inv_pad_sequence,
    _make_gammas_tensor,
    _split_and_pad_sequence,
)


class _check_td_steady:
    def __init__(self, td):
        self.td_clone = td.clone()
        self.td = td

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert (
            self.td.select(*self.td_clone.keys()) == self.td_clone
        ).all(), "Some keys have been modified in the tensordict!"


def get_devices():
    devices = [torch.device("cpu")]
    for i in range(torch.cuda.device_count()):
        devices += [torch.device(f"cuda:{i}")]
    return devices


class TestDQN:
    seed = 0

    def _create_mock_actor(
        self,
        action_spec_type,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        is_nn_module=False,
    ):
        # Actor
        if action_spec_type == "one_hot":
            action_spec = OneHotDiscreteTensorSpec(action_dim)
        elif action_spec_type == "categorical":
            action_spec = DiscreteTensorSpec(action_dim)
        # elif action_spec_type == "nd_bounded":
        #     action_spec = BoundedTensorSpec(
        #         -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        #     )
        else:
            raise ValueError(f"Wrong {action_spec_type}")

        module = nn.Linear(obs_dim, action_dim)
        if is_nn_module:
            return module.to(device)
        actor = QValueActor(
            spec=CompositeSpec(
                action=action_spec,
                action_value=None,
                chosen_action_value=None,
                shape=[],
            ),
            action_space=action_spec_type,
            module=module,
        ).to(device)
        return actor

    def _create_mock_distributional_actor(
        self,
        action_spec_type,
        batch=2,
        obs_dim=3,
        action_dim=4,
        atoms=5,
        vmin=1,
        vmax=5,
        is_nn_module=False,
    ):
        # Actor
        var_nums = None
        if action_spec_type == "mult_one_hot":
            action_spec = MultiOneHotDiscreteTensorSpec(
                [action_dim // 2, action_dim // 2]
            )
            var_nums = action_spec.nvec
        elif action_spec_type == "one_hot":
            action_spec = OneHotDiscreteTensorSpec(action_dim)
        elif action_spec_type == "categorical":
            action_spec = DiscreteTensorSpec(action_dim)
        else:
            raise ValueError(f"Wrong {action_spec_type}")
        support = torch.linspace(vmin, vmax, atoms, dtype=torch.float)
        module = MLP(obs_dim, (atoms, action_dim))
        # TODO: Fails tests with
        # TypeError: __init__() missing 1 required keyword-only argument: 'support'
        # DistributionalQValueActor initializer expects additional inputs.
        # if is_nn_module:
        #     return module
        actor = DistributionalQValueActor(
            spec=CompositeSpec(
                action=action_spec,
                action_value=None,
                shape=[],
            ),
            module=module,
            support=support,
            action_space=action_spec_type,
            var_nums=var_nums,
        )
        return actor

    def _create_mock_data_dqn(
        self,
        action_spec_type,
        batch=2,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
    ):
        # create a tensordict
        obs = torch.randn(batch, obs_dim)
        next_obs = torch.randn(batch, obs_dim)
        if atoms:
            action_value = torch.randn(batch, atoms, action_dim).softmax(-2)
            action = (
                action_value[..., 0, :] == action_value[..., 0, :].max(-1, True)[0]
            ).to(torch.long)
        else:
            action_value = torch.randn(batch, action_dim)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)

        if action_spec_type == "categorical":
            action_value = torch.max(action_value, -1, keepdim=True)[0]
            action = torch.argmax(action, -1, keepdim=False)
        reward = torch.randn(batch, 1)
        done = torch.zeros(batch, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "done": done,
                    "reward": reward,
                },
                "action": action,
                "action_value": action_value,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_dqn(
        self,
        action_spec_type,
        batch=2,
        T=4,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
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

        # action_value = action_value.unsqueeze(-1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
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
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action,
                "action_value": action_value.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
        )
        return td

    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_dqn(self, delay_value, device, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DQNLoss(actor, loss_function="l2", delay_value=delay_value)
        if td_est is ValueEstimators.GAE:
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)
        with _check_td_steady(td):
            loss = loss_fn(td)
        assert loss_fn.priority_key in td.keys()

        sum([item for _, item in loss.items()]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = loss_fn.target_value_network_params.clone()
        for p in loss_fn.parameters():
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

    @pytest.mark.parametrize("n", range(4))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    def test_dqn_batcher(self, n, delay_value, device, action_spec_type, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )

        td = self._create_seq_mock_data_dqn(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DQNLoss(actor, loss_function="l2", delay_value=delay_value)

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)
        ms_td = ms(td.clone())

        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.priority_key in ms_td.keys()

        with torch.no_grad():
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*td.keys(True, True)))
            _loss = sum([item for _, item in loss.items()])
            _loss_ms = sum([item for _, item in loss_ms.items()])
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum([item for _, item in loss_ms.items()]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = loss_fn.target_value_network_params.clone()
        for p in loss_fn.parameters():
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

    @pytest.mark.parametrize("atoms", range(4, 10))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_devices())
    @pytest.mark.parametrize(
        "action_spec_type", ("mult_one_hot", "one_hot", "categorical")
    )
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_distributional_dqn(
        self, atoms, delay_value, device, action_spec_type, td_est, gamma=0.9
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_distributional_actor(
            action_spec_type=action_spec_type, atoms=atoms
        ).to(device)

        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type, atoms=atoms
        ).to(device)
        loss_fn = DistributionalDQNLoss(actor, gamma=gamma, delay_value=delay_value)

        if td_est not in (None, ValueEstimators.TD0):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        elif td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td):
            loss = loss_fn(td)
        assert loss_fn.priority_key in td.keys()

        sum([item for _, item in loss.items()]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = loss_fn.target_value_network_params.clone()
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_value2 = loss_fn.target_value_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            for key, val in target_value.flatten_keys(",").items():
                if "support" in key:
                    continue
                assert not (val == target_value2[tuple(key.split(","))]).any(), key

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))


class TestDDPG:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = BoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = nn.Linear(obs_dim, action_dim)
        actor = Actor(
            spec=action_spec,
            module=module,
        )
        return actor.to(device)

    def _create_mock_value(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        value = ValueOperator(
            module=module,
            in_keys=["observation", "action"],
        )
        return value.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_ddpg(
        self, batch=8, obs_dim=3, action_dim=4, atoms=None, device="cpu"
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
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "done": done,
                    "reward": reward,
                },
                "action": action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_ddpg(
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
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            device=device,
        )
        return td

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("delay_actor,delay_value", [(False, False), (True, True)])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_ddpg(self, delay_actor, delay_value, device, td_est):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_ddpg(device=device)
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
            delay_actor=delay_actor,
            delay_value=delay_value,
        )
        if td_est is ValueEstimators.GAE:
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td):
            loss = loss_fn(td)

        assert all(
            (p.grad is None) or (p.grad == 0).all()
            for p in loss_fn.value_network_params.values(True, True)
        )
        assert all(
            (p.grad is None) or (p.grad == 0).all()
            for p in loss_fn.actor_network_params.values(True, True)
        )
        # check that losses are independent
        for k in loss.keys():
            if not k.startswith("loss"):
                continue
            loss[k].sum().backward(retain_graph=True)
            if k == "loss_actor":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(True, True)
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(True, True)
                )
            elif k == "loss_value":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(True, True)
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params.values(True, True)
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

        # check overall grad
        sum([item for _, item in loss.items()]).backward()
        parameters = list(actor.parameters()) + list(value.parameters())
        for p in parameters:
            assert p.grad.norm() > 0.0

        # Check param update effect on targets
        target_actor = [p.clone() for p in loss_fn.target_actor_network_params.values()]
        target_value = [p.clone() for p in loss_fn.target_value_network_params.values()]
        _i = -1
        for _i, p in enumerate(loss_fn.parameters()):
            p.data += torch.randn_like(p)
        assert _i >= 0
        target_actor2 = [
            p.clone() for p in loss_fn.target_actor_network_params.values()
        ]
        target_value2 = [
            p.clone() for p in loss_fn.target_value_network_params.values()
        ]
        if loss_fn.delay_actor:
            assert all((p1 == p2).all() for p1, p2 in zip(target_actor, target_actor2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_actor, target_actor2)
            )
        if loss_fn.delay_value:
            assert all((p1 == p2).all() for p1, p2 in zip(target_value, target_value2))
        else:
            assert not any(
                (p1 == p2).any() for p1, p2 in zip(target_value, target_value2)
            )

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("n", list(range(4)))
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("delay_actor,delay_value", [(False, False), (True, True)])
    def test_ddpg_batcher(self, n, delay_actor, delay_value, device, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_seq_mock_data_ddpg(device=device)
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
            delay_actor=delay_actor,
            delay_value=delay_value,
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)
        ms_td = ms(td.clone())
        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        with torch.no_grad():
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
            _loss = sum([item for _, item in loss.items()])
            _loss_ms = sum([item for _, item in loss_ms.items()])
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum([item for _, item in loss_ms.items()]).backward()
        parameters = list(actor.parameters()) + list(value.parameters())
        for p in parameters:
            assert p.grad.norm() > 0.0


class TestTD3:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = BoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = nn.Linear(obs_dim, action_dim)
        actor = Actor(
            spec=action_spec,
            module=module,
        )
        return actor.to(device)

    def _create_mock_value(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim, 1)

            def forward(self, obs, act):
                return self.linear(torch.cat([obs, act], -1))

        module = ValueClass()
        value = ValueOperator(
            module=module,
            in_keys=["observation", "action"],
        )
        return value.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_td3(
        self, batch=8, obs_dim=3, action_dim=4, atoms=None, device="cpu"
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
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "done": done,
                    "reward": reward,
                },
                "action": action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_td3(
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
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next": {
                    "observation": next_obs * mask.to(obs.dtype),
                    "reward": reward * mask.to(obs.dtype),
                    "done": done,
                },
                "collector": {"mask": mask},
                "action": action * mask.to(obs.dtype),
            },
            device=device,
        )
        return td

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "delay_actor, delay_qvalue", [(False, False), (True, True)]
    )
    @pytest.mark.parametrize("policy_noise", [0.1, 1.0])
    @pytest.mark.parametrize("noise_clip", [0.1, 1.0])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_td3(
        self,
        delay_actor,
        delay_qvalue,
        device,
        policy_noise,
        noise_clip,
        td_est,
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_td3(device=device)
        loss_fn = TD3Loss(
            actor,
            value,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        if td_est is ValueEstimators.GAE:
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)
        with _check_td_steady(td):
            loss = loss_fn(td)

        assert all(
            (p.grad is None) or (p.grad == 0).all()
            for p in loss_fn.qvalue_network_params.values(True, True)
        )
        assert all(
            (p.grad is None) or (p.grad == 0).all()
            for p in loss_fn.actor_network_params.values(True, True)
        )
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
                )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

        sum([item for _, item in loss.items()]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("n", list(range(4)))
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("delay_actor,delay_qvalue", [(False, False), (True, True)])
    @pytest.mark.parametrize("policy_noise", [0.1, 1.0])
    @pytest.mark.parametrize("noise_clip", [0.1, 1.0])
    def test_td3_batcher(
        self, n, delay_actor, delay_qvalue, device, policy_noise, noise_clip, gamma=0.9
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_seq_mock_data_td3(device=device)
        loss_fn = TD3Loss(
            actor,
            value,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            delay_qvalue=delay_qvalue,
            delay_actor=delay_actor,
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)

        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.priority_key in ms_td.keys()

        with torch.no_grad():
            torch.manual_seed(0)  # log-prob is computed with a random action
            np.random.seed(0)
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
            _loss = sum([item for _, item in loss.items()])
            _loss_ms = sum([item for _, item in loss_ms.items()])
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)

        sum([item for _, item in loss_ms.items()]).backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has null gradient"

        # Check param update effect on targets
        target_actor = loss_fn.target_actor_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        target_qvalue = loss_fn.target_qvalue_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_actor2 = loss_fn.target_actor_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        target_qvalue2 = loss_fn.target_qvalue_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
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
        actorp_set = set(actor.parameters())
        loss_fnp_set = set(loss_fn.parameters())
        assert len(actorp_set.intersection(loss_fnp_set)) == len(actorp_set)
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))


@pytest.mark.parametrize("version", [1, 2])
class TestSAC:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = BoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
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

    def _create_mock_value(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=["observation"],
        )
        return value.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_sac(
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
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "done": done,
                    "reward": reward,
                },
                "action": action,
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
            action = torch.randn(batch, T, atoms, action_dim, device=device).clamp(
                -1, 1
            )
        else:
            action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = torch.ones(batch, T, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            device=device,
        )
        return td

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("delay_value", (True, False))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_sac(
        self,
        delay_value,
        delay_actor,
        delay_qvalue,
        num_qvalue,
        device,
        version,
        td_est,
    ):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")

        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(device=device)

        actor = self._create_mock_actor(device=device)
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
            **kwargs,
        )

        if td_est is ValueEstimators.GAE:
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td):
            loss = loss_fn(td)
        assert loss_fn.priority_key in td.keys()

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

        sum([item for _, item in loss.items()]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("n", list(range(4)))
    @pytest.mark.parametrize("delay_value", (True, False))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_sac_batcher(
        self,
        n,
        delay_value,
        delay_actor,
        delay_qvalue,
        num_qvalue,
        device,
        version,
    ):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_sac(device=device)

        actor = self._create_mock_actor(device=device)
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
            **kwargs,
        )

        ms = MultiStep(gamma=0.9, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)
        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.priority_key in ms_td.keys()

        with torch.no_grad():
            torch.manual_seed(0)  # log-prob is computed with a random action
            np.random.seed(0)
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
            _loss = sum([item for _, item in loss.items()])
            _loss_ms = sum([item for _, item in loss_ms.items()])
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum([item for _, item in loss_ms.items()]).backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has null gradient"

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
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))


class TestDiscreteSAC:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = OneHotDiscreteTensorSpec(action_dim)
        net = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        module = SafeModule(net, in_keys=["observation"], out_keys=["logits"])
        actor = ProbabilisticActor(
            spec=action_spec,
            module=module,
            in_keys=["logits"],
            out_keys=["action"],
            distribution_class=OneHotCategorical,
            return_log_prob=False,
        )
        return actor.to(device)

    def _create_mock_qvalue(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim, action_dim)

            def forward(self, obs):
                return self.linear(obs)

        module = ValueClass()
        qvalue = ValueOperator(
            module=module,
            in_keys=["observation"],
        )
        return qvalue.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_sac(
        self, batch=16, obs_dim=3, action_dim=4, atoms=None, device="cpu"
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
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "done": done,
                    "reward": reward,
                },
                "action": action,
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
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "action_value": action_value.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
        )
        return td

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_available_devices())
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
            **kwargs,
        )
        if td_est is ValueEstimators.GAE:
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td):
            loss = loss_fn(td)
        assert loss_fn.priority_key in td.keys()

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

        sum([item for _, item in loss.items()]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("n", list(range(4)))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [2])
    @pytest.mark.parametrize("device", get_available_devices())
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
            **kwargs,
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)
        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.priority_key in ms_td.keys()

        with torch.no_grad():
            torch.manual_seed(0)  # log-prob is computed with a random action
            np.random.seed(0)
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
            _loss = sum([item for _, item in loss.items()])
            _loss_ms = sum([item for _, item in loss_ms.items()])
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum([item for _, item in loss_ms.items()]).backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has null gradient"

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
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestREDQ:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = BoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        actor = ProbabilisticActor(
            module=module,
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=True,
            spec=action_spec,
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
                self.linear = NormalParamWrapper(nn.Linear(hidden_dim, 2 * action_dim))

            def forward(self, hidden):
                return self.linear(hidden)

        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(hidden_dim + action_dim, 1)

            def forward(self, hidden, act):
                return self.linear(torch.cat([hidden, act], -1))

        common = SafeModule(CommonClass(), in_keys=["observation"], out_keys=["hidden"])
        actor_subnet = ProbabilisticActor(
            SafeModule(ActorClass(), in_keys=["hidden"], out_keys=["loc", "scale"]),
            in_keys=["loc", "scale"],
            distribution_class=TanhNormal,
            return_log_prob=True,
        )
        qvalue_subnet = ValueOperator(ValueClass(), in_keys=["hidden", "action"])
        model = ActorCriticOperator(common, actor_subnet, qvalue_subnet)
        return model.to(device)

    def _create_mock_data_redq(
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
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "done": done,
                    "reward": reward,
                },
                "action": action,
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
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            device=device,
        )
        return td

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
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
        if td_est is ValueEstimators.GAE:
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td):
            loss = loss_fn(td)

        # check td is left untouched
        assert loss_fn.priority_key in td.keys()

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

        sum([item for _, item in loss.items()]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
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
            target_updater = SoftUpdate(loss_fn)

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

        # check td is left untouched
        assert loss_fn.priority_key in td.keys()

        sum([item for _, item in loss.items()]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

        # modify params and check that expanded values are updated
        for p in loss_fn.parameters():
            p.data *= 0

        counter = 0
        for key, p in loss_fn.qvalue_network_params.items(True, True):
            if not isinstance(key, tuple):
                key = (key,)
            if not isinstance(p, nn.Parameter):
                counter += 1
                key = "_sep_".join(["qvalue_network", *key])
                mapped_param = next(
                    (k for k, val in loss_fn._param_maps.items() if val == key)
                )
                assert (p == getattr(loss_fn, mapped_param)).all()
                assert (p == 0).all()
        assert counter == len(loss_fn._actor_network_params.keys(True, True))
        assert counter == len(loss_fn.actor_network_params.keys(True, True))

        # check that params of the original actor are those of the loss_fn
        for p in actor.parameters():
            assert p in set(loss_fn.parameters())

        if delay_qvalue:
            # test that updating with target updater resets the targets of qvalue to 0
            target_updater.step()

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
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
        if td_est is ValueEstimators.GAE:
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
        )
        if td_est is ValueEstimators.GAE:
            with pytest.raises(NotImplementedError):
                loss_fn_deprec.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_deprec.make_value_estimator(td_est)

        td_clone1 = td.clone()
        td_clone2 = td.clone()
        torch.manual_seed(0)
        with _check_td_steady(td_clone1):
            loss_fn(td_clone1)

        torch.manual_seed(0)
        with _check_td_steady(td_clone2):
            loss_fn_deprec(td_clone2)

        # TODO: find a way to compare the losses: problem is that we sample actions either sequentially or in batch,
        #  so setting seed has little impact

    @pytest.mark.parametrize("n", list(range(4)))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_redq_batcher(self, n, delay_qvalue, num_qvalue, device, gamma=0.9):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_redq(device=device)

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
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)

        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.priority_key in ms_td.keys()

        with torch.no_grad():
            torch.manual_seed(0)  # log-prob is computed with a random action
            np.random.seed(0)
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
            _loss = sum([item for _, item in loss.items()])
            _loss_ms = sum([item for _, item in loss_ms.items()])
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum([item for _, item in loss_ms.items()]).backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has null gradient"

        # Check param update effect on targets
        target_actor = loss_fn.target_actor_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        target_qvalue = loss_fn.target_qvalue_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_actor2 = loss_fn.target_actor_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
        target_qvalue2 = loss_fn.target_qvalue_network_params.clone().values(
            include_nested=True, leaves_only=True
        )
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
        actorp_set = set(actor.parameters())
        loss_fnp_set = set(loss_fn.parameters())
        assert len(actorp_set.intersection(loss_fnp_set)) == len(actorp_set)
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))


class TestPPO:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = BoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        actor = ProbabilisticActor(
            module=module,
            distribution_class=TanhNormal,
            in_keys=["loc", "scale"],
            spec=action_spec,
        )
        return actor.to(device)

    def _create_mock_value(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=["observation"],
        )
        return value.to(device)

    def _create_mock_actor_value(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = BoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        base_layer = nn.Linear(obs_dim, 5)
        net = NormalParamWrapper(
            nn.Sequential(base_layer, nn.Linear(5, 2 * action_dim))
        )
        module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        actor = ProbabilisticActor(
            module=module,
            distribution_class=TanhNormal,
            in_keys=["loc", "scale"],
            spec=action_spec,
        )
        module = nn.Sequential(base_layer, nn.Linear(5, 1))
        value = ValueOperator(
            module=module,
            in_keys=["observation"],
        )
        return actor.to(device), value.to(device)

    def _create_mock_actor_value_shared(
        self, batch=2, obs_dim=3, action_dim=4, device="cpu"
    ):
        # Actor
        action_spec = BoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        base_layer = nn.Linear(obs_dim, 5)
        common = TensorDictModule(
            base_layer, in_keys=["observation"], out_keys=["hidden"]
        )
        net = nn.Sequential(nn.Linear(5, 2 * action_dim), NormalParamExtractor())
        module = SafeModule(net, in_keys=["hidden"], out_keys=["loc", "scale"])
        actor_head = ProbabilisticActor(
            module=module,
            distribution_class=TanhNormal,
            in_keys=["loc", "scale"],
            spec=action_spec,
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
        self, batch=2, obs_dim=3, action_dim=4, atoms=None, device="cpu"
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
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "done": done,
                    "reward": reward,
                },
                "action": action,
                "sample_log_prob": torch.randn_like(action[..., 1]) / 10,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_ppo(
        self, batch=2, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
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
        mask = torch.ones(batch, T, dtype=torch.bool, device=device)
        params_mean = torch.randn_like(action) / 10
        params_scale = torch.rand_like(action) / 10
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "sample_log_prob": (torch.randn_like(action[..., 1]) / 10).masked_fill_(
                    ~mask, 0.0
                ),
                "loc": params_mean.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "scale": params_scale.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            device=device,
        )
        return td

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_ppo(self, loss_class, device, gradient_mode, advantage, td_est):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(device=device)

        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
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

        loss_fn = loss_class(actor, value, loss_critic_type="l2")
        if advantage is not None:
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
                assert "actor" in name
                assert "critic" not in name
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
                assert "actor" not in name
                assert "critic" in name
        assert counter == 2
        actor.zero_grad()

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("advantage", ("gae", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_ppo_shared(self, loss_class, device, advantage):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(device=device)

        actor, value = self._create_mock_actor_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
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
        )

        if advantage is not None:
            advantage(td)
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
                assert "actor" in name
                assert "critic" not in name
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
                assert "actor" not in name
                assert "critic" in name
        actor.zero_grad()
        assert counter == 4

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize(
        "advantage",
        (
            "gae",
            "td",
            "td_lambda",
        ),
    )
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("separate_losses", [True, False])
    def test_ppo_shared_seq(self, loss_class, device, advantage, separate_losses):
        """Tests PPO with shared module with and without passing twice across the common module."""
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(device=device)

        model, actor, value = self._create_mock_actor_value_shared(device=device)
        value2 = value[-1]  # prune the common module
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9,
                lmbda=0.9,
                value_network=value,
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
            entropy_coef=0.0,
        )

        loss_fn2 = loss_class(
            actor,
            value2,
            loss_critic_type="l2",
            separate_losses=separate_losses,
            entropy_coef=0.0,
        )

        if advantage is not None:
            advantage(td)
        loss = loss_fn(td).exclude("entropy")
        sum(val for key, val in loss.items() if key.startswith("loss_")).backward()
        grad = TensorDict(dict(model.named_parameters()), []).apply(
            lambda x: x.grad.clone()
        )
        loss2 = loss_fn2(td).exclude("entropy")
        model.zero_grad()
        sum(val for key, val in loss2.items() if key.startswith("loss_")).backward()
        grad2 = TensorDict(dict(model.named_parameters()), []).apply(
            lambda x: x.grad.clone()
        )
        assert_allclose_td(loss, loss2)
        assert_allclose_td(grad, grad2)
        model.zero_grad()

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found, {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_ppo_diff(self, loss_class, device, gradient_mode, advantage):
        if pack_version.parse(torch.__version__) > pack_version.parse("1.14"):
            raise pytest.skip("make_functional_with_buffers needs to be changed")
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(device=device)

        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
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

        loss_fn = loss_class(actor, value, gamma=0.9, loss_critic_type="l2")

        floss_fn, params, buffers = make_functional_with_buffers(loss_fn)
        # fill params with zero
        for p in params:
            p.data.zero_()
        # assert len(list(floss_fn.parameters())) == 0
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
                assert "actor" in name
                assert "critic" not in name

        for param in params:
            param.grad = None
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()

        for (name, other_p), p in zip(named_parameters, params):
            assert other_p.shape == p.shape
            assert other_p.dtype == p.dtype
            assert other_p.device == p.device
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "critic" in name
        for param in params:
            param.grad = None


class TestA2C:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = BoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        actor = ProbabilisticActor(
            module=module,
            in_keys=["loc", "scale"],
            spec=action_spec,
            distribution_class=TanhNormal,
        )
        return actor.to(device)

    def _create_mock_value(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=["observation"],
        )
        return value.to(device)

    def _create_seq_mock_data_a2c(
        self, batch=2, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
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
        mask = ~torch.zeros(batch, T, dtype=torch.bool, device=device)
        params_mean = torch.randn_like(action) / 10
        params_scale = torch.rand_like(action) / 10
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "sample_log_prob": torch.randn_like(action[..., 1]).masked_fill_(
                    ~mask, 0.0
                )
                / 10,
                "loc": params_mean.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "scale": params_scale.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            device=device,
        )
        return td

    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_a2c(self, device, gradient_mode, advantage, td_est):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_a2c(device=device)

        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
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

        loss_fn = A2CLoss(actor, value, loss_critic_type="l2")

        # Check error is raised when actions require grads
        td["action"].requires_grad = True
        with pytest.raises(
            RuntimeError,
            match="tensordict stored action require grad.",
        ):
            _ = loss_fn._log_probs(td)
        td["action"].requires_grad = False

        td = td.exclude(loss_fn.value_target_key)
        if advantage is not None:
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
                assert "actor" in name
                assert "critic" not in name

        value.zero_grad()
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "critic" in name
        actor.zero_grad()

        # test reset
        loss_fn.reset()

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not found, {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "td", "td_lambda", None))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_a2c_diff(self, device, gradient_mode, advantage):
        if pack_version.parse(torch.__version__) > pack_version.parse("1.14"):
            raise pytest.skip("make_functional_with_buffers needs to be changed")
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_a2c(device=device)

        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, differentiable=gradient_mode
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
                assert "actor" in name
                assert "critic" not in name

        for param in params:
            param.grad = None
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        for (name, _), p in zip(named_parameters, params):
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "critic" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "critic" in name
        for param in params:
            param.grad = None


class TestReinforce:
    @pytest.mark.parametrize("delay_value", [True, False])
    @pytest.mark.parametrize("gradient_mode", [True, False])
    @pytest.mark.parametrize("advantage", ["gae", "td", "td_lambda", None])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_reinforce_value_net(self, advantage, gradient_mode, delay_value, td_est):
        n_obs = 3
        n_act = 5
        batch = 4
        gamma = 0.9
        value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=["observation"])
        net = NormalParamWrapper(nn.Linear(n_obs, 2 * n_act))
        module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
        actor_net = ProbabilisticActor(
            module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            in_keys=["loc", "scale"],
            spec=UnboundedContinuousTensorSpec(n_act),
        )
        if advantage == "gae":
            advantage = GAE(
                gamma=gamma,
                lmbda=0.9,
                value_network=get_functional(value_net),
                differentiable=gradient_mode,
            )
        elif advantage == "td":
            advantage = TD1Estimator(
                gamma=gamma,
                value_network=get_functional(value_net),
                differentiable=gradient_mode,
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimator(
                gamma=0.9,
                lmbda=0.9,
                value_network=get_functional(value_net),
                differentiable=gradient_mode,
            )
        elif advantage is None:
            pass
        else:
            raise NotImplementedError

        loss_fn = ReinforceLoss(
            actor_net,
            critic=value_net,
            delay_value=delay_value,
        )

        td = TensorDict(
            {
                "observation": torch.randn(batch, n_obs),
                "next": {
                    "observation": torch.randn(batch, n_obs),
                    "reward": torch.randn(batch, 1),
                    "done": torch.zeros(batch, 1, dtype=torch.bool),
                },
                "action": torch.randn(batch, n_act),
            },
            [batch],
        )

        params = TensorDict(value_net.state_dict(), []).unflatten_keys(".")
        if advantage is not None:
            advantage(td, params=params)
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


@pytest.mark.parametrize("device", get_available_devices())
class TestDreamer:
    def _create_world_model_data(
        self, batch_size, temporal_length, rssm_hidden_dim, state_dim
    ):
        td = TensorDict(
            {
                "state": torch.zeros(batch_size, temporal_length, state_dim),
                "belief": torch.zeros(batch_size, temporal_length, rssm_hidden_dim),
                "pixels": torch.randn(batch_size, temporal_length, 3, 64, 64),
                "next": {
                    "pixels": torch.randn(batch_size, temporal_length, 3, 64, 64),
                    "reward": torch.randn(batch_size, temporal_length, 1),
                    "done": torch.zeros(batch_size, temporal_length, dtype=torch.bool),
                },
                "action": torch.randn(batch_size, temporal_length, 64),
            },
            [batch_size, temporal_length],
        )
        return td

    def _create_actor_data(
        self, batch_size, temporal_length, rssm_hidden_dim, state_dim
    ):
        td = TensorDict(
            {
                "state": torch.randn(batch_size, temporal_length, state_dim),
                "belief": torch.randn(batch_size, temporal_length, rssm_hidden_dim),
                "reward": torch.randn(batch_size, temporal_length, 1),
            },
            [batch_size, temporal_length],
        )
        return td

    def _create_value_data(
        self, batch_size, temporal_length, rssm_hidden_dim, state_dim
    ):
        td = TensorDict(
            {
                "state": torch.randn(batch_size * temporal_length, state_dim),
                "belief": torch.randn(batch_size * temporal_length, rssm_hidden_dim),
                "lambda_target": torch.randn(batch_size * temporal_length, 1),
            },
            [batch_size * temporal_length],
        )
        return td

    def _create_world_model_model(self, rssm_hidden_dim, state_dim, mlp_num_units=200):
        mock_env = TransformedEnv(ContinuousActionConvMockEnv(pixel_shape=[3, 64, 64]))
        default_dict = {
            "state": UnboundedContinuousTensorSpec(state_dim),
            "belief": UnboundedContinuousTensorSpec(rssm_hidden_dim),
        }
        mock_env.append_transform(
            TensorDictPrimer(random=False, default_value=0, **default_dict)
        )

        obs_encoder = ObsEncoder()
        obs_decoder = ObsDecoder()

        rssm_prior = RSSMPrior(
            hidden_dim=rssm_hidden_dim,
            rnn_hidden_dim=rssm_hidden_dim,
            state_dim=state_dim,
            action_spec=mock_env.action_spec,
        )
        rssm_posterior = RSSMPosterior(hidden_dim=rssm_hidden_dim, state_dim=state_dim)

        # World Model and reward model
        rssm_rollout = RSSMRollout(
            SafeModule(
                rssm_prior,
                in_keys=["state", "belief", "action"],
                out_keys=[
                    ("next", "prior_mean"),
                    ("next", "prior_std"),
                    "_",
                    ("next", "belief"),
                ],
            ),
            SafeModule(
                rssm_posterior,
                in_keys=[("next", "belief"), ("next", "encoded_latents")],
                out_keys=[
                    ("next", "posterior_mean"),
                    ("next", "posterior_std"),
                    ("next", "state"),
                ],
            ),
        )
        reward_module = MLP(
            out_features=1, depth=2, num_cells=mlp_num_units, activation_class=nn.ELU
        )
        # World Model and reward model
        world_modeler = SafeSequential(
            SafeModule(
                obs_encoder,
                in_keys=[("next", "pixels")],
                out_keys=[("next", "encoded_latents")],
            ),
            rssm_rollout,
            SafeModule(
                obs_decoder,
                in_keys=[("next", "state"), ("next", "belief")],
                out_keys=[("next", "reco_pixels")],
            ),
        )
        reward_module = SafeModule(
            reward_module,
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=[("next", "reward")],
        )
        world_model = WorldModelWrapper(world_modeler, reward_module)

        with torch.no_grad():
            td = mock_env.rollout(10)
            td = td.unsqueeze(0).to_tensordict()
            td["state"] = torch.zeros((1, 10, state_dim))
            td["belief"] = torch.zeros((1, 10, rssm_hidden_dim))
            world_model(td)
        return world_model

    def _create_mb_env(self, rssm_hidden_dim, state_dim, mlp_num_units=200):
        mock_env = TransformedEnv(ContinuousActionConvMockEnv(pixel_shape=[3, 64, 64]))
        default_dict = {
            "state": UnboundedContinuousTensorSpec(state_dim),
            "belief": UnboundedContinuousTensorSpec(rssm_hidden_dim),
        }
        mock_env.append_transform(
            TensorDictPrimer(random=False, default_value=0, **default_dict)
        )

        rssm_prior = RSSMPrior(
            hidden_dim=rssm_hidden_dim,
            rnn_hidden_dim=rssm_hidden_dim,
            state_dim=state_dim,
            action_spec=mock_env.action_spec,
        )
        reward_module = MLP(
            out_features=1, depth=2, num_cells=mlp_num_units, activation_class=nn.ELU
        )
        transition_model = SafeSequential(
            SafeModule(
                rssm_prior,
                in_keys=["state", "belief", "action"],
                out_keys=[
                    "_",
                    "_",
                    "state",
                    "belief",
                ],
            ),
        )
        reward_model = SafeModule(
            reward_module,
            in_keys=["state", "belief"],
            out_keys=["reward"],
        )
        model_based_env = DreamerEnv(
            world_model=WorldModelWrapper(
                transition_model,
                reward_model,
            ),
            prior_shape=torch.Size([state_dim]),
            belief_shape=torch.Size([rssm_hidden_dim]),
        )
        model_based_env.set_specs_from_env(mock_env)
        with torch.no_grad():
            model_based_env.rollout(3)
        return model_based_env

    def _create_actor_model(self, rssm_hidden_dim, state_dim, mlp_num_units=200):
        mock_env = TransformedEnv(ContinuousActionConvMockEnv(pixel_shape=[3, 64, 64]))
        default_dict = {
            "state": UnboundedContinuousTensorSpec(state_dim),
            "belief": UnboundedContinuousTensorSpec(rssm_hidden_dim),
        }
        mock_env.append_transform(
            TensorDictPrimer(random=False, default_value=0, **default_dict)
        )

        actor_module = DreamerActor(
            out_features=mock_env.action_spec.shape[0],
            depth=4,
            num_cells=mlp_num_units,
            activation_class=nn.ELU,
        )
        actor_model = SafeProbabilisticTensorDictSequential(
            SafeModule(
                actor_module,
                in_keys=["state", "belief"],
                out_keys=["loc", "scale"],
            ),
            SafeProbabilisticModule(
                in_keys=["loc", "scale"],
                out_keys="action",
                default_interaction_type=InteractionType.RANDOM,
                distribution_class=TanhNormal,
            ),
        )
        with torch.no_grad():
            td = TensorDict(
                {
                    "state": torch.randn(1, 2, state_dim),
                    "belief": torch.randn(1, 2, rssm_hidden_dim),
                },
                batch_size=[1],
            )
            actor_model(td)
        return actor_model

    def _create_value_model(self, rssm_hidden_dim, state_dim, mlp_num_units=200):
        value_model = SafeModule(
            MLP(
                out_features=1,
                depth=3,
                num_cells=mlp_num_units,
                activation_class=nn.ELU,
            ),
            in_keys=["state", "belief"],
            out_keys=["state_value"],
        )
        with torch.no_grad():
            td = TensorDict(
                {
                    "state": torch.randn(1, 2, state_dim),
                    "belief": torch.randn(1, 2, rssm_hidden_dim),
                },
                batch_size=[1],
            )
            value_model(td)
        return value_model

    @pytest.mark.parametrize("lambda_kl", [0, 1.0])
    @pytest.mark.parametrize("lambda_reco", [0, 1.0])
    @pytest.mark.parametrize("lambda_reward", [0, 1.0])
    @pytest.mark.parametrize("reco_loss", ["l2", "smooth_l1"])
    @pytest.mark.parametrize("reward_loss", ["l2", "smooth_l1"])
    @pytest.mark.parametrize("free_nats", [-1000, 1000])
    @pytest.mark.parametrize("delayed_clamp", [False, True])
    def test_dreamer_world_model(
        self,
        device,
        lambda_reward,
        lambda_kl,
        lambda_reco,
        reward_loss,
        reco_loss,
        delayed_clamp,
        free_nats,
    ):
        tensordict = self._create_world_model_data(2, 3, 10, 5).to(device)
        world_model = self._create_world_model_model(10, 5).to(device)
        loss_module = DreamerModelLoss(
            world_model,
            lambda_reco=lambda_reco,
            lambda_kl=lambda_kl,
            lambda_reward=lambda_reward,
            reward_loss=reward_loss,
            reco_loss=reco_loss,
            delayed_clamp=delayed_clamp,
            free_nats=free_nats,
        )
        loss_td, _ = loss_module(tensordict)
        for loss_str, lmbda in zip(
            ["loss_model_kl", "loss_model_reco", "loss_model_reward"],
            [lambda_kl, lambda_reco, lambda_reward],
        ):
            assert loss_td.get(loss_str) is not None
            assert loss_td.get(loss_str).shape == torch.Size([1])
            if lmbda == 0:
                assert loss_td.get(loss_str) == 0
            else:
                assert loss_td.get(loss_str) > 0

        loss = (
            loss_td.get("loss_model_kl")
            + loss_td.get("loss_model_reco")
            + loss_td.get("loss_model_reward")
        )
        loss.backward()
        grad_total = 0.0
        for name, param in loss_module.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )
                if not valid_gradients:
                    raise ValueError(f"Invalid gradients for {name}")
                gsq = param.grad.pow(2).sum()
                grad_total += gsq.item()
        grad_is_zero = grad_total == 0
        if free_nats < 0:
            lambda_kl_corr = lambda_kl
        else:
            # we expect the kl loss to have 0 grad
            lambda_kl_corr = 0
        if grad_is_zero and (lambda_kl_corr or lambda_reward or lambda_reco):
            raise ValueError(
                f"Gradients are zero: lambdas={(lambda_kl_corr, lambda_reward, lambda_reco)}"
            )
        elif grad_is_zero:
            assert not (lambda_kl_corr or lambda_reward or lambda_reco)
        loss_module.zero_grad()

    @pytest.mark.parametrize("imagination_horizon", [3, 5])
    @pytest.mark.parametrize("discount_loss", [True, False])
    def test_dreamer_env(self, device, imagination_horizon, discount_loss):
        mb_env = self._create_mb_env(10, 5).to(device)
        rollout = mb_env.rollout(3)
        assert rollout.shape == torch.Size([3])
        # test reconstruction
        with pytest.raises(ValueError, match="No observation decoder provided"):
            mb_env.decode_obs(rollout)
        mb_env.obs_decoder = SafeModule(
            nn.LazyLinear(4, device=device),
            in_keys=["state"],
            out_keys=["reco_observation"],
        )
        # reconstruct
        mb_env.decode_obs(rollout)
        assert "reco_observation" in rollout.keys()
        # second pass
        tensordict = mb_env.decode_obs(mb_env.reset(), compute_latents=True)
        assert "reco_observation" in tensordict.keys()

    @pytest.mark.parametrize("imagination_horizon", [3, 5])
    @pytest.mark.parametrize("discount_loss", [True, False])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_dreamer_actor(self, device, imagination_horizon, discount_loss, td_est):
        tensordict = self._create_actor_data(2, 3, 10, 5).to(device)
        mb_env = self._create_mb_env(10, 5).to(device)
        actor_model = self._create_actor_model(10, 5).to(device)
        value_model = self._create_value_model(10, 5).to(device)
        loss_module = DreamerActorLoss(
            actor_model,
            value_model,
            mb_env,
            imagination_horizon=imagination_horizon,
            discount_loss=discount_loss,
        )
        if td_est is ValueEstimators.GAE:
            with pytest.raises(NotImplementedError):
                loss_module.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_module.make_value_estimator(td_est)
        loss_td, fake_data = loss_module(tensordict)
        assert not fake_data.requires_grad
        assert fake_data.shape == torch.Size([tensordict.numel(), imagination_horizon])
        if discount_loss:
            assert loss_module.discount_loss

        assert loss_td.get("loss_actor") is not None
        loss = loss_td.get("loss_actor")
        loss.backward()
        grad_is_zero = True
        for name, param in loss_module.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )
                grad_is_zero = (
                    grad_is_zero and torch.sum(torch.pow((param.grad), 2)) == 0
                )
                if not valid_gradients:
                    raise ValueError(f"Invalid gradients for {name}")
        if grad_is_zero:
            raise ValueError("Gradients are zero")
        loss_module.zero_grad()

    @pytest.mark.parametrize("discount_loss", [True, False])
    def test_dreamer_value(self, device, discount_loss):
        tensordict = self._create_value_data(2, 3, 10, 5).to(device)
        value_model = self._create_value_model(10, 5).to(device)
        loss_module = DreamerValueLoss(value_model, discount_loss=discount_loss)
        loss_td, fake_data = loss_module(tensordict)
        assert loss_td.get("loss_value") is not None
        assert not fake_data.requires_grad
        loss = loss_td.get("loss_value")
        loss.backward()
        grad_is_zero = True
        for name, param in loss_module.named_parameters():
            if param.grad is not None:
                valid_gradients = not (
                    torch.isnan(param.grad).any() or torch.isinf(param.grad).any()
                )
                grad_is_zero = (
                    grad_is_zero and torch.sum(torch.pow((param.grad), 2)) == 0
                )
                if not valid_gradients:
                    raise ValueError(f"Invalid gradients for {name}")
        if grad_is_zero:
            raise ValueError("Gradients are zero")
        loss_module.zero_grad()


class TestIQL:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = BoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        module = SafeModule(net, in_keys=["observation"], out_keys=["loc", "scale"])
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

    def _create_mock_value(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=["observation"],
        )
        return value.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_iql(
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
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {
                    "observation": next_obs,
                    "done": done,
                    "reward": reward,
                },
                "action": action,
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
        mask = torch.ones(batch, T, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "done": done,
                    "reward": reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
                },
                "collector": {"mask": mask},
                "action": action.masked_fill_(~mask.unsqueeze(-1), 0.0),
            },
            device=device,
        )
        return td

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
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
        if td_est is ValueEstimators.GAE:
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td):
            loss = loss_fn(td)
        assert loss_fn.priority_key in td.keys()

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

        sum([item for _, item in loss.items()]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.skipif(
        not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
    )
    @pytest.mark.parametrize("n", list(range(4)))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("temperature", [0.0, 0.1, 1.0, 10.0])
    @pytest.mark.parametrize("expectile", [0.1, 0.5, 1.0])
    @pytest.mark.parametrize("device", get_available_devices())
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
        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.priority_key in ms_td.keys()

        with torch.no_grad():
            torch.manual_seed(0)  # log-prob is computed with a random action
            np.random.seed(0)
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys(True, True))))
            _loss = sum([item for _, item in loss.items()])
            _loss_ms = sum([item for _, item in loss_ms.items()])
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum([item for _, item in loss_ms.items()]).backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has null gradient"

        # Check param update effect on targets
        target_qvalue = [
            p.clone()
            for p in loss_fn.target_qvalue_network_params.values(
                include_nested=True, leaves_only=True
            )
        ]
        for p in loss_fn.parameters():
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
            p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))


def test_hold_out():
    net = torch.nn.Linear(3, 4)
    x = torch.randn(1, 3)
    x_rg = torch.randn(1, 3, requires_grad=True)
    y = net(x)
    assert y.requires_grad
    with hold_out_net(net):
        y = net(x)
        assert not y.requires_grad
        y = net(x_rg)
        assert y.requires_grad

    y = net(x)
    assert y.requires_grad

    # nested case
    with hold_out_net(net):
        y = net(x)
        assert not y.requires_grad
        with hold_out_net(net):
            y = net(x)
            assert not y.requires_grad
            y = net(x_rg)
            assert y.requires_grad

    y = net(x)
    assert y.requires_grad

    # exception
    net = torch.nn.Sequential()
    with hold_out_net(net):
        pass


@pytest.mark.parametrize("mode", ["hard", "soft"])
@pytest.mark.parametrize("value_network_update_interval", [100, 1000])
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize(
    "dtype",
    [
        torch.float64,
        torch.float32,
    ],
)
def test_updater(mode, value_network_update_interval, device, dtype):
    torch.manual_seed(100)

    class custom_module_error(nn.Module):
        def __init__(self):
            super().__init__()
            self.target_params = [torch.randn(3, 4)]
            self.target_error_params = [torch.randn(3, 4)]
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(3, 4, requires_grad=True))]
            )

    module = custom_module_error().to(device)
    with pytest.raises(
        RuntimeError, match="Your module seems to have a target tensor list "
    ):
        if mode == "hard":
            upd = HardUpdate(module, value_network_update_interval)
        elif mode == "soft":
            upd = SoftUpdate(module, 1 - 1 / value_network_update_interval)

    class custom_module(LossModule):
        def __init__(self):
            super().__init__()
            module1 = torch.nn.BatchNorm2d(10).eval()
            self.convert_to_functional(module1, "module1", create_target_params=True)
            module2 = torch.nn.BatchNorm2d(10).eval()
            self.module2 = module2
            iterator_params = self.target_module1_params.values(
                include_nested=True, leaves_only=True
            )
            for target in iterator_params:
                if target.dtype is not torch.int64:
                    target.data.normal_()
                else:
                    target.data += 10

    module = custom_module().to(device).to(dtype)
    _ = module.module1_params
    _ = module.target_module1_params
    if mode == "hard":
        upd = HardUpdate(
            module, value_network_update_interval=value_network_update_interval
        )
    elif mode == "soft":
        upd = SoftUpdate(module, 1 - 1 / value_network_update_interval)
    for _, _v in upd._targets.items(True, True):
        if _v.dtype is not torch.int64:
            _v.copy_(torch.randn_like(_v))
        else:
            _v += 10

    # total dist
    d0 = 0.0
    for (key, source_val) in upd._sources.items(True, True):
        if not isinstance(key, tuple):
            key = (key,)
        key = ("target_" + key[0], *key[1:])
        target_val = upd._targets[key]
        assert target_val.dtype is source_val.dtype, key
        assert target_val.device == source_val.device, key
        if target_val.dtype == torch.long:
            continue
        d0 += (target_val - source_val).norm().item()

    assert d0 > 0
    if mode == "hard":
        for i in range(value_network_update_interval + 1):
            # test that no update is occuring until value_network_update_interval
            d1 = 0.0
            for (key, source_val) in upd._sources.items(True, True):
                if not isinstance(key, tuple):
                    key = (key,)
                key = ("target_" + key[0], *key[1:])
                target_val = upd._targets[key]
                if target_val.dtype == torch.long:
                    continue
                d1 += (target_val - source_val).norm().item()

            assert d1 == d0, i
            assert upd.counter == i
            upd.step()
        assert upd.counter == 0
        # test that a new update has occured
        d1 = 0.0
        for (key, source_val) in upd._sources.items(True, True):
            if not isinstance(key, tuple):
                key = (key,)
            key = ("target_" + key[0], *key[1:])
            target_val = upd._targets[key]
            if target_val.dtype == torch.long:
                continue
            d1 += (target_val - source_val).norm().item()
        assert d1 < d0

    elif mode == "soft":
        upd.step()
        d1 = 0.0
        for (key, source_val) in upd._sources.items(True, True):
            if not isinstance(key, tuple):
                key = (key,)
            key = ("target_" + key[0], *key[1:])
            target_val = upd._targets[key]
            if target_val.dtype == torch.long:
                continue
            d1 += (target_val - source_val).norm().item()
        assert d1 < d0

    upd.init_()
    upd.step()
    d2 = 0.0
    for (key, source_val) in upd._sources.items(True, True):
        if not isinstance(key, tuple):
            key = (key,)
        key = ("target_" + key[0], *key[1:])
        target_val = upd._targets[key]
        if target_val.dtype == torch.long:
            continue
        d2 += (target_val - source_val).norm().item()
    assert d2 < 1e-6


class TestValues:
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("gamma", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    # @pytest.mark.parametrize("random_gamma,rolling_gamma", [[True, False], [True, True], [False, None]])
    @pytest.mark.parametrize("random_gamma,rolling_gamma", [[False, None]])
    def test_tdlambda(self, device, gamma, lmbda, N, T, random_gamma, rolling_gamma):
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool).bernoulli_(0.1)
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)
        if random_gamma:
            gamma = torch.rand_like(reward) * gamma

        r1 = vec_td_lambda_advantage_estimate(
            gamma, lmbda, state_value, next_state_value, reward, done, rolling_gamma
        )
        r2 = td_lambda_advantage_estimate(
            gamma, lmbda, state_value, next_state_value, reward, done, rolling_gamma
        )
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("gamma", [0.1, 0.99])
    @pytest.mark.parametrize("lmbda", [0.1, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 100])
    @pytest.mark.parametrize("feature_dim", [[5], [2, 5]])
    @pytest.mark.parametrize("random_gamma,rolling_gamma", [[False, None]])
    def test_tdlambda_multi(
        self, device, gamma, lmbda, N, T, random_gamma, rolling_gamma, feature_dim
    ):
        torch.manual_seed(0)
        D = feature_dim
        time_dim = -1 - len(D)
        done = torch.zeros(*N, T, *D, device=device, dtype=torch.bool).bernoulli_(0.1)
        reward = torch.randn(*N, T, *D, device=device)
        state_value = torch.randn(*N, T, *D, device=device)
        next_state_value = torch.randn(*N, T, *D, device=device)
        if random_gamma:
            gamma = torch.rand_like(reward) * gamma

        r1 = vec_td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
            rolling_gamma,
            time_dim=time_dim,
        )
        r2 = td_lambda_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
            rolling_gamma,
            time_dim=time_dim,
        )
        if len(D) == 2:
            r3 = torch.cat(
                [
                    vec_td_lambda_advantage_estimate(
                        gamma,
                        lmbda,
                        state_value[..., i : i + 1, j],
                        next_state_value[..., i : i + 1, j],
                        reward[..., i : i + 1, j],
                        done[..., i : i + 1, j],
                        rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                    for j in range(D[1])
                ],
                -1,
            ).unflatten(-1, D)
            r4 = torch.cat(
                [
                    td_lambda_advantage_estimate(
                        gamma,
                        lmbda,
                        state_value[..., i : i + 1, j],
                        next_state_value[..., i : i + 1, j],
                        reward[..., i : i + 1, j],
                        done[..., i : i + 1, j],
                        rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                    for j in range(D[1])
                ],
                -1,
            ).unflatten(-1, D)
        else:
            r3 = torch.cat(
                [
                    vec_td_lambda_advantage_estimate(
                        gamma,
                        lmbda,
                        state_value[..., i : i + 1],
                        next_state_value[..., i : i + 1],
                        reward[..., i : i + 1],
                        done[..., i : i + 1],
                        rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                ],
                -1,
            )
            r4 = torch.cat(
                [
                    td_lambda_advantage_estimate(
                        gamma,
                        lmbda,
                        state_value[..., i : i + 1],
                        next_state_value[..., i : i + 1],
                        reward[..., i : i + 1],
                        done[..., i : i + 1],
                        rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                ],
                -1,
            )

        torch.testing.assert_close(r4, r2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r3, r1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("gamma", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 100])
    @pytest.mark.parametrize("random_gamma,rolling_gamma", [[False, None]])
    def test_td1(self, device, gamma, N, T, random_gamma, rolling_gamma):
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool).bernoulli_(0.1)
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)
        if random_gamma:
            gamma = torch.rand_like(reward) * gamma

        r1 = vec_td1_advantage_estimate(
            gamma, state_value, next_state_value, reward, done, rolling_gamma
        )
        r2 = td1_advantage_estimate(
            gamma, state_value, next_state_value, reward, done, rolling_gamma
        )
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("gamma", [0.1, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5])
    @pytest.mark.parametrize("feature_dim", [[5], [2, 5]])
    @pytest.mark.parametrize("random_gamma,rolling_gamma", [[False, None]])
    def test_td1_multi(
        self, device, gamma, N, T, random_gamma, rolling_gamma, feature_dim
    ):
        torch.manual_seed(0)

        D = feature_dim
        time_dim = -1 - len(D)
        done = torch.zeros(*N, T, *D, device=device, dtype=torch.bool).bernoulli_(0.1)
        reward = torch.randn(*N, T, *D, device=device)
        state_value = torch.randn(*N, T, *D, device=device)
        next_state_value = torch.randn(*N, T, *D, device=device)
        if random_gamma:
            gamma = torch.rand_like(reward) * gamma

        r1 = vec_td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done,
            rolling_gamma,
            time_dim=time_dim,
        )
        r2 = td1_advantage_estimate(
            gamma,
            state_value,
            next_state_value,
            reward,
            done,
            rolling_gamma,
            time_dim=time_dim,
        )
        if len(D) == 2:
            r3 = torch.cat(
                [
                    vec_td1_advantage_estimate(
                        gamma,
                        state_value[..., i : i + 1, j],
                        next_state_value[..., i : i + 1, j],
                        reward[..., i : i + 1, j],
                        done[..., i : i + 1, j],
                        rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                    for j in range(D[1])
                ],
                -1,
            ).unflatten(-1, D)
            r4 = torch.cat(
                [
                    td1_advantage_estimate(
                        gamma,
                        state_value[..., i : i + 1, j],
                        next_state_value[..., i : i + 1, j],
                        reward[..., i : i + 1, j],
                        done[..., i : i + 1, j],
                        rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                    for j in range(D[1])
                ],
                -1,
            ).unflatten(-1, D)
        else:
            r3 = torch.cat(
                [
                    vec_td1_advantage_estimate(
                        gamma,
                        state_value[..., i : i + 1],
                        next_state_value[..., i : i + 1],
                        reward[..., i : i + 1],
                        done[..., i : i + 1],
                        rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                ],
                -1,
            )
            r4 = torch.cat(
                [
                    td1_advantage_estimate(
                        gamma,
                        state_value[..., i : i + 1],
                        next_state_value[..., i : i + 1],
                        reward[..., i : i + 1],
                        done[..., i : i + 1],
                        rolling_gamma,
                        time_dim=-2,
                    )
                    for i in range(D[0])
                ],
                -1,
            )

        torch.testing.assert_close(r4, r2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r3, r1, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("gamma", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("lmbda", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("N", [(1,), (3,), (7, 3)])
    @pytest.mark.parametrize("T", [200, 5, 3])
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_gae(self, device, gamma, lmbda, N, T, dtype, has_done):
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            done = done.bernoulli_(0.1)
        reward = torch.randn(*N, T, 1, device=device, dtype=dtype)
        state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)
        next_state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)

        r1 = vec_generalized_advantage_estimate(
            gamma, lmbda, state_value, next_state_value, reward, done
        )
        r2 = generalized_advantage_estimate(
            gamma, lmbda, state_value, next_state_value, reward, done
        )

        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("N", [(1,), (8,), (7, 3)])
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("has_done", [True, False])
    @pytest.mark.parametrize(
        "gamma_tensor", ["scalar", "tensor", "tensor_single_element"]
    )
    @pytest.mark.parametrize(
        "lmbda_tensor", ["scalar", "tensor", "tensor_single_element"]
    )
    def test_gae_param_as_tensor(
        self, device, N, dtype, has_done, gamma_tensor, lmbda_tensor
    ):
        torch.manual_seed(0)

        gamma = 0.95
        lmbda = 0.90
        T = 200

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            done = done.bernoulli_(0.1)
        reward = torch.randn(*N, T, 1, device=device, dtype=dtype)
        state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)
        next_state_value = torch.randn(*N, T, 1, device=device, dtype=dtype)

        if gamma_tensor == "tensor":
            gamma_vec = torch.full_like(reward, gamma)
        elif gamma_tensor == "tensor_single_element":
            gamma_vec = torch.as_tensor([gamma], device=device)
        else:
            gamma_vec = gamma

        if lmbda_tensor == "tensor":
            lmbda_vec = torch.full_like(reward, lmbda)
        elif gamma_tensor == "tensor_single_element":
            lmbda_vec = torch.as_tensor([lmbda], device=device)
        else:
            lmbda_vec = lmbda

        r1 = vec_generalized_advantage_estimate(
            gamma_vec, lmbda_vec, state_value, next_state_value, reward, done
        )
        r2 = generalized_advantage_estimate(
            gamma, lmbda, state_value, next_state_value, reward, done
        )
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("gamma", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("lmbda", [0.99, 0.5, 0.1])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [100, 3])
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("feature_dim", [[5], [2, 5]])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_gae_multidim(
        self, device, gamma, lmbda, N, T, dtype, has_done, feature_dim
    ):
        D = feature_dim
        time_dim = -1 - len(D)

        torch.manual_seed(0)

        done = torch.zeros(*N, T, *D, device=device, dtype=torch.bool)
        if has_done:
            done = done.bernoulli_(0.1)
        reward = torch.randn(*N, T, *D, device=device, dtype=dtype)
        state_value = torch.randn(*N, T, *D, device=device, dtype=dtype)
        next_state_value = torch.randn(*N, T, *D, device=device, dtype=dtype)

        r1 = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
            time_dim=time_dim,
        )
        r2 = generalized_advantage_estimate(
            gamma,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
            time_dim=time_dim,
        )
        if len(D) == 2:
            r3 = [
                vec_generalized_advantage_estimate(
                    gamma,
                    lmbda,
                    state_value[..., i : i + 1, j],
                    next_state_value[..., i : i + 1, j],
                    reward[..., i : i + 1, j],
                    done[..., i : i + 1, j],
                    time_dim=-2,
                )
                for i in range(D[0])
                for j in range(D[1])
            ]
            r4 = [
                generalized_advantage_estimate(
                    gamma,
                    lmbda,
                    state_value[..., i : i + 1, j],
                    next_state_value[..., i : i + 1, j],
                    reward[..., i : i + 1, j],
                    done[..., i : i + 1, j],
                    time_dim=-2,
                )
                for i in range(D[0])
                for j in range(D[1])
            ]
        else:
            r3 = [
                vec_generalized_advantage_estimate(
                    gamma,
                    lmbda,
                    state_value[..., i : i + 1],
                    next_state_value[..., i : i + 1],
                    reward[..., i : i + 1],
                    done[..., i : i + 1],
                    time_dim=-2,
                )
                for i in range(D[0])
            ]
            r4 = [
                generalized_advantage_estimate(
                    gamma,
                    lmbda,
                    state_value[..., i : i + 1],
                    next_state_value[..., i : i + 1],
                    reward[..., i : i + 1],
                    done[..., i : i + 1],
                    time_dim=-2,
                )
                for i in range(D[0])
            ]

        list3 = list(zip(*r3))
        list4 = list(zip(*r4))
        r3 = [torch.cat(list3[0], -1), torch.cat(list3[1], -1)]
        r4 = [torch.cat(list4[0], -1), torch.cat(list4[1], -1)]
        if len(D) == 2:
            r3 = [r3[0].unflatten(-1, D), r3[1].unflatten(-1, D)]
            r4 = [r4[0].unflatten(-1, D), r4[1].unflatten(-1, D)]
        torch.testing.assert_close(r2, r4, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r3, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(r1, r2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99, 0.1])
    @pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_tdlambda_tensor_gamma(self, device, gamma, lmbda, N, T, has_done):
        """Tests vec_td_lambda_advantage_estimate against itself with
        gamma being a tensor or a scalar

        """
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            done = done.bernoulli_(0.1)
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        gamma_tensor = torch.full((*N, T, 1), gamma, device=device)

        v1 = vec_td_lambda_advantage_estimate(
            gamma, lmbda, state_value, next_state_value, reward, done
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor, lmbda, state_value, next_state_value, reward, done
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        gamma_tensor[..., -1, :] = 0.0

        v1 = vec_td_lambda_advantage_estimate(
            gamma, lmbda, state_value, next_state_value, reward, done
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor, lmbda, state_value, next_state_value, reward, done
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99, 0.1])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_td1_tensor_gamma(self, device, gamma, N, T, has_done):
        """Tests vec_td_lambda_advantage_estimate against itself with
        gamma being a tensor or a scalar

        """
        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            done = done.bernoulli_(0.1)
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        gamma_tensor = torch.full((*N, T, 1), gamma, device=device)

        v1 = vec_td1_advantage_estimate(
            gamma, state_value, next_state_value, reward, done
        )
        v2 = vec_td1_advantage_estimate(
            gamma_tensor, state_value, next_state_value, reward, done
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        gamma_tensor[..., -1, :] = 0.0

        v1 = vec_td1_advantage_estimate(
            gamma, state_value, next_state_value, reward, done
        )
        v2 = vec_td1_advantage_estimate(
            gamma_tensor, state_value, next_state_value, reward, done
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99, 0.1])
    @pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5, 50])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_vectdlambda_tensor_gamma(
        self, device, gamma, lmbda, N, T, dtype_fixture, has_done  # noqa
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a tensor or a scalar

        """

        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            done = done.bernoulli_(0.1)
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        gamma_tensor = torch.full((*N, T, 1), gamma, device=device)

        v1 = td_lambda_advantage_estimate(
            gamma, lmbda, state_value, next_state_value, reward, done
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor, lmbda, state_value, next_state_value, reward, done
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        gamma_tensor[..., -1, :] = 0.0

        v1 = td_lambda_advantage_estimate(
            gamma, lmbda, state_value, next_state_value, reward, done
        )
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor, lmbda, state_value, next_state_value, reward, done
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("gamma", [0.5, 0.99, 0.1])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [3, 5, 50])
    @pytest.mark.parametrize("has_done", [True, False])
    def test_vectd1_tensor_gamma(
        self, device, gamma, N, T, dtype_fixture, has_done  # noqa
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a tensor or a scalar

        """

        torch.manual_seed(0)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            done = done.bernoulli_(0.1)
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        gamma_tensor = torch.full((*N, T, 1), gamma, device=device)

        v1 = td1_advantage_estimate(gamma, state_value, next_state_value, reward, done)
        v2 = vec_td1_advantage_estimate(
            gamma_tensor, state_value, next_state_value, reward, done
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

        # same with last done being true
        done[..., -1, :] = True  # terminating trajectory
        gamma_tensor[..., -1, :] = 0.0

        v1 = td1_advantage_estimate(gamma, state_value, next_state_value, reward, done)
        v2 = vec_td1_advantage_estimate(
            gamma_tensor, state_value, next_state_value, reward, done
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [50, 3])
    @pytest.mark.parametrize("rolling_gamma", [True, False, None])
    @pytest.mark.parametrize("has_done", [True, False])
    @pytest.mark.parametrize("seed", range(1))
    def test_vectdlambda_rand_gamma(
        self, device, lmbda, N, T, rolling_gamma, dtype_fixture, has_done, seed  # noqa
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(seed)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            done = done.bernoulli_(0.1)
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.5 + torch.rand_like(next_state_value) / 2

        v1 = td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
            rolling_gamma,
        )
        if rolling_gamma is False and not done[..., 1:, :][done[..., :-1, :]].all():
            # if a not-done follows a done, then rolling_gamma=False cannot be used
            with pytest.raises(
                NotImplementedError, match="When using rolling_gamma=False"
            ):
                vec_td_lambda_advantage_estimate(
                    gamma_tensor,
                    lmbda,
                    state_value,
                    next_state_value,
                    reward,
                    done,
                    rolling_gamma,
                )
            return
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
            rolling_gamma,
        )
        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("N", [(3,), (7, 3)])
    @pytest.mark.parametrize("T", [50, 3])
    @pytest.mark.parametrize("rolling_gamma", [True, False, None])
    @pytest.mark.parametrize("has_done", [True, False])
    @pytest.mark.parametrize("seed", range(1))
    def test_vectd1_rand_gamma(
        self, device, N, T, rolling_gamma, dtype_fixture, has_done, seed  # noqa
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(seed)

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        if has_done:
            done = done.bernoulli_(0.1)
        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.5 + torch.rand_like(next_state_value) / 2

        v1 = td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done,
            rolling_gamma,
        )
        if rolling_gamma is False and not done[..., 1:, :][done[..., :-1, :]].all():
            # if a not-done follows a done, then rolling_gamma=False cannot be used
            with pytest.raises(
                NotImplementedError, match="When using rolling_gamma=False"
            ):
                vec_td1_advantage_estimate(
                    gamma_tensor,
                    state_value,
                    next_state_value,
                    reward,
                    done,
                    rolling_gamma,
                )
            return
        v2 = vec_td1_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done,
            rolling_gamma,
        )
        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("gamma", [0.99, "rand"])
    @pytest.mark.parametrize("N", [(3,), (3, 7)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    @pytest.mark.parametrize("rolling_gamma", [True, False])
    def test_custom_conv1d_tensor(self, device, gamma, N, T, rolling_gamma):
        """
        Tests the _custom_conv1d logic against a manual for-loop implementation
        """
        torch.manual_seed(0)

        if gamma == "rand":
            gamma = torch.rand(*N, T, 1, device=device)
            rand_gamma = True
        else:
            gamma = torch.full((*N, T, 1), gamma, device=device)
            rand_gamma = False

        values = torch.randn(*N, 1, T, device=device)
        out = torch.zeros(*N, 1, T, device=device)
        if rand_gamma and not rolling_gamma:
            for i in range(T):
                for j in reversed(range(i, T)):
                    out[..., i] = out[..., i] * gamma[..., i, :] + values[..., j]
        else:
            prev_val = 0.0
            for i in reversed(range(T)):
                prev_val = out[..., i] = prev_val * gamma[..., i, :] + values[..., i]

        gammas = _make_gammas_tensor(gamma, T, rolling_gamma)
        gammas = gammas.cumprod(-2)
        out_custom = _custom_conv1d(values.view(-1, 1, T), gammas).reshape(values.shape)

        torch.testing.assert_close(out, out_custom, rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("N", [(3,), (3, 7)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    @pytest.mark.parametrize("rolling_gamma", [True, False])
    def test_successive_traj_tdlambda(self, device, N, T, rolling_gamma):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(0)

        lmbda = torch.rand([]).item()

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        done[..., T // 2 - 1, :] = 1

        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.5 + torch.rand_like(next_state_value) / 2

        v1 = td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
            rolling_gamma,
        )
        v1a = td_lambda_advantage_estimate(
            gamma_tensor[..., : T // 2, :],
            lmbda,
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done[..., : T // 2, :],
            rolling_gamma,
        )
        v1b = td_lambda_advantage_estimate(
            gamma_tensor[..., T // 2 :, :],
            lmbda,
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done[..., T // 2 :, :],
            rolling_gamma,
        )
        torch.testing.assert_close(v1, torch.cat([v1a, v1b], -2), rtol=1e-4, atol=1e-4)

        if not rolling_gamma:
            with pytest.raises(
                NotImplementedError, match="When using rolling_gamma=False"
            ):
                vec_td_lambda_advantage_estimate(
                    gamma_tensor,
                    lmbda,
                    state_value,
                    next_state_value,
                    reward,
                    done,
                    rolling_gamma,
                )
            return
        v2 = vec_td_lambda_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
            rolling_gamma,
        )
        v2a = vec_td_lambda_advantage_estimate(
            gamma_tensor[..., : T // 2, :],
            lmbda,
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done[..., : T // 2, :],
            rolling_gamma,
        )
        v2b = vec_td_lambda_advantage_estimate(
            gamma_tensor[..., T // 2 :, :],
            lmbda,
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done[..., T // 2 :, :],
            rolling_gamma,
        )

        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(v1a, v2a, rtol=1e-4, atol=1e-4)

        torch.testing.assert_close(v1b, v2b, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(v2, torch.cat([v2a, v2b], -2), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("N", [(3,), (3, 7)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    def test_successive_traj_tdadv(
        self,
        device,
        N,
        T,
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(0)

        lmbda = torch.rand([]).item()

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        done[..., T // 2 - 1, :] = 1

        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.5 + torch.rand_like(next_state_value) / 2

        v1 = td0_advantage_estimate(
            gamma_tensor,
            state_value,
            next_state_value,
            reward,
            done,
        )
        v1a = td0_advantage_estimate(
            gamma_tensor[..., : T // 2, :],
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done[..., : T // 2, :],
        )
        v1b = td0_advantage_estimate(
            gamma_tensor[..., T // 2 :, :],
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done[..., T // 2 :, :],
        )
        torch.testing.assert_close(v1, torch.cat([v1a, v1b], -2), rtol=1e-4, atol=1e-4)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("N", [(3,), (3, 7)])
    @pytest.mark.parametrize("T", [3, 5, 200])
    def test_successive_traj_gae(
        self,
        device,
        N,
        T,
    ):
        """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
        with gamma being a random tensor

        """
        torch.manual_seed(0)

        lmbda = torch.rand([]).item()

        done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
        done[..., T // 2 - 1, :] = 1

        reward = torch.randn(*N, T, 1, device=device)
        state_value = torch.randn(*N, T, 1, device=device)
        next_state_value = torch.randn(*N, T, 1, device=device)

        # avoid low values of gamma
        gamma_tensor = 0.95

        v1 = generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
        )[0]
        v1a = generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done[..., : T // 2, :],
        )[0]
        v1b = generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done[..., T // 2 :, :],
        )[0]
        torch.testing.assert_close(v1, torch.cat([v1a, v1b], -2), rtol=1e-4, atol=1e-4)

        v2 = vec_generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value,
            next_state_value,
            reward,
            done,
        )[0]
        v2a = vec_generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value[..., : T // 2, :],
            next_state_value[..., : T // 2, :],
            reward[..., : T // 2, :],
            done[..., : T // 2, :],
        )[0]
        v2b = vec_generalized_advantage_estimate(
            gamma_tensor,
            lmbda,
            state_value[..., T // 2 :, :],
            next_state_value[..., T // 2 :, :],
            reward[..., T // 2 :, :],
            done[..., T // 2 :, :],
        )[0]
        torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(v2, torch.cat([v2a, v2b], -2), rtol=1e-4, atol=1e-4)


@pytest.mark.skipif(
    not _has_functorch,
    reason=f"no vmap allowed without functorch, error: {FUNCTORCH_ERR}",
)
@pytest.mark.parametrize(
    "dest,expected_dtype,expected_device",
    list(
        zip(
            get_available_devices(),
            [torch.float] * len(get_available_devices()),
            get_available_devices(),
        )
    )
    + [
        ["cuda", torch.float, "cuda:0"],
        ["double", torch.double, "cpu"],
        [torch.double, torch.double, "cpu"],
        [torch.half, torch.half, "cpu"],
        ["half", torch.half, "cpu"],
    ],
)
def test_shared_params(dest, expected_dtype, expected_device):
    if torch.cuda.device_count() == 0 and dest == "cuda":
        pytest.skip("no cuda device available")
    module_hidden = torch.nn.Linear(4, 4)
    td_module_hidden = SafeModule(
        module=module_hidden,
        spec=None,
        in_keys=["observation"],
        out_keys=["hidden"],
    )
    module_action = SafeModule(
        NormalParamWrapper(torch.nn.Linear(4, 8)),
        in_keys=["hidden"],
        out_keys=["loc", "scale"],
    )
    td_module_action = ProbabilisticActor(
        module=module_action,
        in_keys=["loc", "scale"],
        out_keys=["action"],
        spec=None,
        distribution_class=TanhNormal,
        return_log_prob=True,
    )
    module_value = torch.nn.Linear(4, 1)
    td_module_value = ValueOperator(module=module_value, in_keys=["hidden"])
    td_module = ActorValueOperator(td_module_hidden, td_module_action, td_module_value)

    class MyLoss(LossModule):
        def __init__(self, actor_network, qvalue_network):
            super().__init__()
            self.convert_to_functional(
                actor_network,
                "actor_network",
                create_target_params=True,
            )
            self.convert_to_functional(
                qvalue_network,
                "qvalue_network",
                3,
                create_target_params=True,
                compare_against=list(actor_network.parameters()),
            )

    actor_network = td_module.get_policy_operator()
    value_network = td_module.get_value_operator()

    loss = MyLoss(actor_network, value_network)
    # modify params
    for p in loss.parameters():
        p.data += torch.randn_like(p)

    assert len(list(loss.parameters())) == 6
    assert (
        len(loss.actor_network_params.keys(include_nested=True, leaves_only=True)) == 4
    )
    assert (
        len(loss.qvalue_network_params.keys(include_nested=True, leaves_only=True)) == 4
    )
    for p in loss.actor_network_params.values(include_nested=True, leaves_only=True):
        assert isinstance(p, nn.Parameter) or isinstance(p, Buffer)
    for i, (key, value) in enumerate(
        loss.qvalue_network_params.items(include_nested=True, leaves_only=True)
    ):
        p1 = value
        p2 = loss.actor_network_params[key]
        assert (p1 == p2).all()
        if i == 1:
            break

    # map module
    if dest == "double":
        loss = loss.double()
    elif dest == "cuda":
        loss = loss.cuda()
    elif dest == "half":
        loss = loss.half()
    else:
        loss = loss.to(dest)

    for p in loss.actor_network_params.values(include_nested=True, leaves_only=True):
        assert isinstance(p, nn.Parameter)
        assert p.dtype is expected_dtype
        assert p.device == torch.device(expected_device)
    for i, (key, qvalparam) in enumerate(
        loss.qvalue_network_params.items(include_nested=True, leaves_only=True)
    ):
        assert qvalparam.dtype is expected_dtype, (key, qvalparam)
        assert qvalparam.device == torch.device(expected_device), key
        assert (qvalparam == loss.actor_network_params[key]).all(), key
        if i == 1:
            break


class TestAdv:
    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
        ],
    )
    def test_dispatch(
        self,
        adv,
        kwargs,
    ):
        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )
        module = adv(
            gamma=0.98,
            value_network=value_net,
            differentiable=False,
            **kwargs,
        )
        kwargs = {
            "obs": torch.randn(1, 10, 3),
            "next_reward": torch.randn(1, 10, 1, requires_grad=True),
            "next_done": torch.zeros(1, 10, 1, dtype=torch.bool),
            "next_obs": torch.randn(1, 10, 3),
        }
        advantage, value_target = module(**kwargs)
        assert advantage.shape == torch.Size([1, 10, 1])
        assert value_target.shape == torch.Size([1, 10, 1])

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
        ],
    )
    def test_diff_reward(
        self,
        adv,
        kwargs,
    ):
        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )
        module = adv(
            gamma=0.98,
            value_network=value_net,
            differentiable=True,
            **kwargs,
        )
        td = TensorDict(
            {
                "obs": torch.randn(1, 10, 3),
                "next": {
                    "obs": torch.randn(1, 10, 3),
                    "reward": torch.randn(1, 10, 1, requires_grad=True),
                    "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                },
            },
            [1, 10],
        )
        td = module(td.clone(False))
        # check that the advantage can't backprop to the value params
        td["advantage"].sum().backward()
        for p in value_net.parameters():
            assert p.grad is None or (p.grad == 0).all()
        # check that rewards have a grad
        assert td["next", "reward"].grad.norm() > 0

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
        ],
    )
    def test_non_differentiable(self, adv, kwargs):
        value_net = TensorDictModule(
            nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
        )
        module = adv(
            gamma=0.98,
            value_network=value_net,
            differentiable=False,
            **kwargs,
        )
        td = TensorDict(
            {
                "obs": torch.randn(1, 10, 3),
                "next": {
                    "obs": torch.randn(1, 10, 3),
                    "reward": torch.randn(1, 10, 1, requires_grad=True),
                    "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                },
            },
            [1, 10],
        )
        td = module(td.clone(False))
        assert td["advantage"].is_leaf

    @pytest.mark.parametrize(
        "adv,kwargs",
        [
            [GAE, {"lmbda": 0.95}],
            [TD1Estimator, {}],
            [TDLambdaEstimator, {"lmbda": 0.95}],
        ],
    )
    @pytest.mark.parametrize("has_value_net", [True, False])
    @pytest.mark.parametrize("skip_existing", [True, False, None])
    def test_skip_existing(
        self,
        adv,
        kwargs,
        has_value_net,
        skip_existing,
    ):
        if has_value_net:
            value_net = TensorDictModule(
                lambda x: torch.zeros(*x.shape[:-1], 1),
                in_keys=["obs"],
                out_keys=["state_value"],
            )
        else:
            value_net = None

        module = adv(
            gamma=0.98,
            value_network=value_net,
            differentiable=True,
            skip_existing=skip_existing,
            **kwargs,
        )
        td = TensorDict(
            {
                "obs": torch.randn(1, 10, 3),
                "state_value": torch.ones(1, 10, 1),
                "next": {
                    "obs": torch.randn(1, 10, 3),
                    "state_value": torch.ones(1, 10, 1),
                    "reward": torch.randn(1, 10, 1, requires_grad=True),
                    "done": torch.zeros(1, 10, 1, dtype=torch.bool),
                },
            },
            [1, 10],
        )
        td = module(td.clone(False))
        if has_value_net and not skip_existing:
            exp_val = 0
        elif has_value_net and skip_existing:
            exp_val = 1
        elif not has_value_net:
            exp_val = 1
        assert (td["state_value"] == exp_val).all()
        # assert (td["next", "state_value"] == exp_val).all()


class TestBase:
    @pytest.mark.parametrize("expand_dim", [None, 2])
    @pytest.mark.parametrize("compare_against", [True, False])
    @pytest.mark.skipif(not _has_functorch, reason="functorch is needed for expansion")
    def test_convert_to_func(self, compare_against, expand_dim):
        class MyLoss(LossModule):
            def __init__(self, compare_against, expand_dim):
                super().__init__()
                module1 = nn.Linear(3, 4)
                module2 = nn.Linear(3, 4)
                module3 = nn.Linear(3, 4)
                module_a = TensorDictModule(
                    nn.Sequential(module1, module2), in_keys=["a"], out_keys=["c"]
                )
                module_b = TensorDictModule(
                    nn.Sequential(module1, module3), in_keys=["b"], out_keys=["c"]
                )
                self.convert_to_functional(module_a, "module_a")
                self.convert_to_functional(
                    module_b,
                    "module_b",
                    compare_against=module_a.parameters() if compare_against else [],
                    expand_dim=expand_dim,
                )

        loss_module = MyLoss(compare_against=compare_against, expand_dim=expand_dim)

        for key in ["module.0.bias", "module.0.weight"]:
            if compare_against:
                assert not loss_module.module_b_params.flatten_keys()[key].requires_grad
            else:
                assert loss_module.module_b_params.flatten_keys()[key].requires_grad
            if expand_dim:
                assert (
                    loss_module.module_b_params.flatten_keys()[key].shape[0]
                    == expand_dim
                )
            else:
                assert (
                    loss_module.module_b_params.flatten_keys()[key].shape[0]
                    != expand_dim
                )

        for key in ["module.1.bias", "module.1.weight"]:
            loss_module.module_b_params.flatten_keys()[key].requires_grad


class TestUtils:
    @pytest.mark.parametrize("B", [None, (1, ), (4, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [1, 10])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_get_num_per_traj_no_stops(self, B, T, device):
        """check _get_num_per_traj when input contains no stops"""
        size = (*B, T) if B else (T,)

        done = torch.zeros(*size, dtype=torch.bool, device=device)
        splits = _get_num_per_traj(done)

        count = functools.reduce(operator.mul, B, 1) if B else 1
        res = torch.full((count,), T, device=device)

        torch.testing.assert_close(splits, res)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_get_num_per_traj(self, B, T, device):
        """check _get_num_per_traj where input contains a stop at half of each trace"""
        size = (*B, T)

        done = torch.zeros(*size, dtype=torch.bool, device=device)
        done[..., T // 2] = True
        splits = _get_num_per_traj(done)

        count = functools.reduce(operator.mul, B, 1)
        res = [T - (T + 1) // 2 + 1, (T + 1) // 2 - 1] * count
        res = torch.as_tensor(res, device=device)

        torch.testing.assert_close(splits, res)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_split_pad_reverse(self, B, T, device):
        """calls _split_and_pad_sequence and reverts it"""
        torch.manual_seed(42)

        size = (*B, T)
        traj = torch.rand(*size, device=device)
        done = torch.zeros(*size, dtype=torch.bool, device=device).bernoulli(0.2)
        splits = _get_num_per_traj(done)

        splitted = _split_and_pad_sequence(traj, splits)
        reversed = _inv_pad_sequence(splitted, splits).reshape(traj.shape)

        torch.testing.assert_close(traj, reversed)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_split_pad_no_stops(self, B, T, device):
        """_split_and_pad_sequence on trajectories without stops should not change input but flatten it along batch dimension"""
        size = (*B, T)
        count = functools.reduce(operator.mul, size, 1)

        traj = torch.arange(0, count, device=device).reshape(size)
        done = torch.zeros(*size, dtype=torch.bool, device=device)

        splits = _get_num_per_traj(done)
        splitted = _split_and_pad_sequence(traj, splits)

        traj_flat = traj.flatten(0, -2)
        torch.testing.assert_close(traj_flat, splitted)

    @pytest.mark.parametrize("device", get_available_devices())
    def test_split_pad_manual(self, device):
        """handcrafted example to test _split_and_pad_seqeunce"""

        traj = torch.as_tensor([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], device=device)
        splits = torch.as_tensor([3, 2, 1, 4], device=device)
        res = torch.as_tensor(
            [[0, 1, 2, 0], [3, 4, 0, 0], [5, 0, 0, 0], [6, 7, 8, 9]], device=device
        )

        splitted = _split_and_pad_sequence(traj, splits)
        torch.testing.assert_close(res, splitted)

    @pytest.mark.parametrize("B", [(1, ), (3, ), (2, 2, ), (1, 2, 8, )])  # fmt: skip
    @pytest.mark.parametrize("T", [5, 100])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_split_pad_reverse_tensordict(self, B, T, device):
        """calls _split_and_pad_sequence and reverts it on tensordict input"""
        torch.manual_seed(42)

        td = TensorDict(
            {
                "observation": torch.arange(T, dtype=torch.float32, device=device)
                .unsqueeze(-1)
                .expand(*B, T, 3),
                "is_init": torch.zeros(
                    *B, T, 1, dtype=torch.bool, device=device
                ).bernoulli(0.3),
            },
            [*B, T],
            device=device,
        )

        is_init = td.get("is_init").squeeze(-1)
        splits = _get_num_per_traj_init(is_init)
        splitted = _split_and_pad_sequence(
            td.select("observation", strict=False), splits
        )

        reversed = _inv_pad_sequence(splitted, splits)
        reversed = reversed.reshape(td.shape)
        torch.testing.assert_close(td["observation"], reversed["observation"])

    def test_timedimtranspose_single(self):
        @_transpose_time
        def fun(a, b, time_dim=-2):
            return a + 1

        x = torch.zeros(10)
        y = torch.ones(10)
        with pytest.raises(RuntimeError):
            z = fun(x, y, time_dim=-3)
        with pytest.raises(RuntimeError):
            z = fun(x, y, time_dim=-2)
        z = fun(x, y, time_dim=-1)
        assert z.shape == torch.Size([10])
        assert (z == 1).all()

        @_transpose_time
        def fun(a, b, time_dim=-2):
            return a + 1, b + 1

        with pytest.raises(RuntimeError):
            z1, z2 = fun(x, y, time_dim=-3)
        with pytest.raises(RuntimeError):
            z1, z2 = fun(x, y, time_dim=-2)
        z1, z2 = fun(x, y, time_dim=-1)
        assert z1.shape == torch.Size([10])
        assert (z1 == 1).all()
        assert z2.shape == torch.Size([10])
        assert (z2 == 2).all()


@pytest.mark.parametrize("updater", [HardUpdate, SoftUpdate])
def test_updater_warning(updater):
    with warnings.catch_warnings():
        dqn = DQNLoss(torch.nn.Linear(3, 4), delay_value=True, action_space="one_hot")
    with pytest.warns(UserWarning):
        dqn.target_value_network_params
    with warnings.catch_warnings():
        updater(dqn)
    with warnings.catch_warnings():
        dqn.target_value_network_params


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
