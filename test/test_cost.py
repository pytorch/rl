# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse
from copy import deepcopy

_has_functorch = True
try:
    import functorch

    make_functional_with_buffers = functorch.make_functional_with_buffers

except ImportError:
    _has_functorch = False
    make_functional_with_buffers = FunctionalModuleWithBuffers._create_from

import numpy as np
import pytest
import torch
from _utils_internal import dtype_fixture, get_available_devices  # noqa
from mocking_classes import ContinuousActionConvMockEnv

# from torchrl.data.postprocs.utils import expand_as_right
from tensordict.tensordict import assert_allclose_td, TensorDict, TensorDictBase
from tensordict.utils import expand_as_right
from torch import autograd, nn
from torchrl.data import (
    CompositeSpec,
    DiscreteTensorSpec,
    MultOneHotDiscreteTensorSpec,
    NdBoundedTensorSpec,
    NdUnboundedContinuousTensorSpec,
    OneHotDiscreteTensorSpec,
)
from torchrl.data.postprocs.postprocs import MultiStep
from torchrl.envs.model_based.dreamer import DreamerEnv
from torchrl.envs.transforms import TensorDictPrimer, TransformedEnv
from torchrl.modules import (
    DistributionalQValueActor,
    ProbabilisticTensorDictModule,
    QValueActor,
    TensorDictModule,
    TensorDictSequential,
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
from torchrl.objectives import (
    A2CLoss,
    ClipPPOLoss,
    DDPGLoss,
    DistributionalDQNLoss,
    DQNLoss,
    DreamerActorLoss,
    DreamerModelLoss,
    DreamerValueLoss,
    KLPENPPOLoss,
    PPOLoss,
    SACLoss,
)
from torchrl.objectives.common import LossModule
from torchrl.objectives.deprecated import DoubleREDQLoss_deprecated, REDQLoss_deprecated
from torchrl.objectives.redq import REDQLoss
from torchrl.objectives.reinforce import ReinforceLoss
from torchrl.objectives.utils import HardUpdate, hold_out_net, SoftUpdate
from torchrl.objectives.value.advantages import GAE, TDEstimate, TDLambdaEstimate
from torchrl.objectives.value.functional import (
    generalized_advantage_estimate,
    td_lambda_advantage_estimate,
    vec_generalized_advantage_estimate,
    vec_td_lambda_advantage_estimate,
)
from torchrl.objectives.value.utils import _custom_conv1d, _make_gammas_tensor


class _check_td_steady:
    def __init__(self, td):
        self.td_clone = td.clone()
        self.td = td

    def __enter__(self):
        pass

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert (self.td.select(*self.td_clone.keys()) == self.td_clone).all()


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
        elif action_spec_type == "nd_bounded":
            action_spec = NdBoundedTensorSpec(
                -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
            )
        else:
            raise ValueError(f"Wrong {action_spec_type}")

        module = nn.Linear(obs_dim, action_dim)
        if is_nn_module:
            return module.to(device)
        actor = QValueActor(
            spec=CompositeSpec(
                action=action_spec, action_value=None, chosen_action_value=None
            ),
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
        if action_spec_type == "mult_one_hot":
            action_spec = MultOneHotDiscreteTensorSpec([atoms] * action_dim)
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
            spec=CompositeSpec(action=action_spec, action_value=None),
            module=module,
            support=support,
            action_space="categorical"
            if isinstance(action_spec, DiscreteTensorSpec)
            else "one_hot",
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
            action = torch.argmax(action, -1, keepdim=True)
        reward = torch.randn(batch, 1)
        done = torch.zeros(batch, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next": {"observation": next_obs},
                "done": done,
                "reward": reward,
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

        if action_spec_type == "categorical":
            action_value = torch.max(action_value, -1, keepdim=True)[0]
            action = torch.argmax(action, -1, keepdim=True)
        reward = torch.randn(batch, T, 1, device=device)
        done = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next": {"observation": next_obs * mask.to(obs.dtype)},
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
                "action_value": action_value
                * expand_as_right(mask.to(obs.dtype).squeeze(-1), action_value),
            },
        )
        return td

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "action_spec_type", ("nd_bounded", "one_hot", "categorical")
    )
    @pytest.mark.parametrize("is_nn_module", (False, True))
    def test_dqn(self, delay_value, device, action_spec_type, is_nn_module):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device, is_nn_module=is_nn_module
        )
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DQNLoss(actor, gamma=0.9, loss_function="l2", delay_value=delay_value)
        with _check_td_steady(td):
            loss = loss_fn(td)
        assert loss_fn.priority_key in td.keys()

        sum([item for _, item in loss.items()]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = [p.clone() for p in loss_fn.target_value_network_params]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_value2 = [p.clone() for p in loss_fn.target_value_network_params]
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

    @pytest.mark.skipif(_has_functorch, reason="functorch installed")
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "action_spec_type", ("nd_bounded", "one_hot", "categorical")
    )
    def test_dqn_nofunctorch(self, delay_value, device, action_spec_type):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DQNLoss(actor, gamma=0.9, loss_function="l2", delay_value=delay_value)
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

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("n", range(4))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "action_spec_type", ("nd_bounded", "one_hot", "categorical")
    )
    def test_dqn_batcher(self, n, delay_value, device, action_spec_type, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )

        td = self._create_seq_mock_data_dqn(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DQNLoss(
            actor, gamma=gamma, loss_function="l2", delay_value=delay_value
        )

        ms = MultiStep(gamma=gamma, n_steps_max=n).to(device)
        ms_td = ms(td.clone())

        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.priority_key in ms_td.keys()

        with torch.no_grad():
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys())))
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
        target_value = [p.clone() for p in loss_fn.target_value_network_params]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_value2 = [p.clone() for p in loss_fn.target_value_network_params]
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

    @pytest.mark.skipif(_has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("n", range(4))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize(
        "action_spec_type", ("nd_bounded", "one_hot", "categorical")
    )
    def test_dqn_batcher_nofunctorch(
        self, n, delay_value, device, action_spec_type, gamma=0.9
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )

        td = self._create_seq_mock_data_dqn(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DQNLoss(
            actor, gamma=gamma, loss_function="l2", delay_value=delay_value
        )

        ms = MultiStep(gamma=gamma, n_steps_max=n).to(device)
        ms_td = ms(td.clone())

        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.priority_key in ms_td.keys()

        with torch.no_grad():
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys())))
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

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("atoms", range(4, 10))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_devices())
    @pytest.mark.parametrize(
        "action_spec_type", ("mult_one_hot", "one_hot", "categorical")
    )
    @pytest.mark.parametrize("is_nn_module", (False, True))
    def test_distributional_dqn(
        self, atoms, delay_value, device, action_spec_type, is_nn_module, gamma=0.9
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_distributional_actor(
            action_spec_type=action_spec_type, atoms=atoms, is_nn_module=is_nn_module
        ).to(device)

        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type, atoms=atoms
        ).to(device)
        loss_fn = DistributionalDQNLoss(actor, gamma=gamma, delay_value=delay_value)

        with _check_td_steady(td):
            loss = loss_fn(td)
        assert loss_fn.priority_key in td.keys()

        sum([item for _, item in loss.items()]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = [p.clone() for p in loss_fn.target_value_network_params]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_value2 = [p.clone() for p in loss_fn.target_value_network_params]
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

    @pytest.mark.skipif(_has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("atoms", range(4, 10))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_devices())
    @pytest.mark.parametrize(
        "action_spec_type", ("mult_one_hot", "one_hot", "categorical")
    )
    def test_distributional_dqn_nofunctorch(
        self, atoms, delay_value, device, action_spec_type, gamma=0.9
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_distributional_actor(
            action_spec_type=action_spec_type, atoms=atoms
        ).to(device)

        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type, atoms=atoms
        ).to(device)
        loss_fn = DistributionalDQNLoss(actor, gamma=gamma, delay_value=delay_value)

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


class TestDDPG:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = NdBoundedTensorSpec(
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
                "next": {"observation": next_obs},
                "done": done,
                "reward": reward,
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
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next": {"observation": next_obs * mask.to(obs.dtype)},
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
            },
            device=device,
        )
        return td

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("delay_actor,delay_value", [(False, False), (True, True)])
    def test_ddpg(self, delay_actor, delay_value, device):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_ddpg(device=device)
        loss_fn = DDPGLoss(
            actor,
            value,
            gamma=0.9,
            loss_function="l2",
            delay_actor=delay_actor,
            delay_value=delay_value,
        )
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
                    for p in loss_fn.value_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
            elif k == "loss_value":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params
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
        target_actor = [p.clone() for p in loss_fn.target_actor_network_params]
        target_value = [p.clone() for p in loss_fn.target_value_network_params]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_actor2 = [p.clone() for p in loss_fn.target_actor_network_params]
        target_value2 = [p.clone() for p in loss_fn.target_value_network_params]
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

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
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
            gamma=gamma,
            loss_function="l2",
            delay_actor=delay_actor,
            delay_value=delay_value,
        )

        ms = MultiStep(gamma=gamma, n_steps_max=n).to(device)
        ms_td = ms(td.clone())
        with _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        with torch.no_grad():
            loss = loss_fn(td)
        if n == 0:
            assert_allclose_td(td, ms_td.select(*list(td.keys())))
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


class TestSAC:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            spec=CompositeSpec(action=action_spec, loc=None, scale=None),
            module=module,
            distribution_class=TanhNormal,
            dist_in_keys=["loc", "scale"],
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
                "next": {"observation": next_obs},
                "done": done,
                "reward": reward,
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
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next": {"observation": next_obs * mask.to(obs.dtype)},
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
            },
            device=device,
        )
        return td

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("delay_value", (True, False))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_sac(self, delay_value, delay_actor, delay_qvalue, num_qvalue, device):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")

        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

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
            gamma=0.9,
            loss_function="l2",
            **kwargs,
        )

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
                    for p in loss_fn.value_network_params
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
            elif k == "loss_value":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params
                )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.value_network_params
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params
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
    @pytest.mark.parametrize("delay_value", (True, False))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_sac_batcher(
        self, n, delay_value, delay_actor, delay_qvalue, num_qvalue, device, gamma=0.9
    ):
        if (delay_actor or delay_qvalue) and not delay_value:
            pytest.skip("incompatible config")
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_sac(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)
        value = self._create_mock_value(device=device)

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
            gamma=0.9,
            loss_function="l2",
            **kwargs,
        )

        ms = MultiStep(gamma=gamma, n_steps_max=n).to(device)

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
            assert_allclose_td(td, ms_td.select(*list(td.keys())))
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
        target_actor = [p.clone() for p in loss_fn.target_actor_network_params]
        target_qvalue = [p.clone() for p in loss_fn.target_qvalue_network_params]
        target_value = [p.clone() for p in loss_fn.target_value_network_params]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_actor2 = [p.clone() for p in loss_fn.target_actor_network_params]
        target_qvalue2 = [p.clone() for p in loss_fn.target_qvalue_network_params]
        target_value2 = [p.clone() for p in loss_fn.target_value_network_params]
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


@pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
class TestREDQ:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            dist_in_keys=["loc", "scale"],
            spec=CompositeSpec(action=action_spec, loc=None, scale=None),
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

        common = TensorDictModule(
            CommonClass(), in_keys=["observation"], out_keys=["hidden"]
        )
        actor_subnet = ProbabilisticActor(
            TensorDictModule(
                ActorClass(), in_keys=["hidden"], out_keys=["loc", "scale"]
            ),
            dist_in_keys=["loc", "scale"],
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
                "next": {"observation": next_obs},
                "done": done,
                "reward": reward,
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
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next": {"observation": next_obs * mask.to(obs.dtype)},
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
            },
            device=device,
        )
        return td

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_redq(self, delay_qvalue, num_qvalue, device):

        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = REDQLoss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            gamma=0.9,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )

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
                    for p in loss_fn.qvalue_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params
                )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.qvalue_network_params
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
            gamma=0.9,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
            target_entropy=0.0,
        )

        if delay_qvalue:
            target_updater = SoftUpdate(loss_fn)
            target_updater.init_()

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
                    for p in loss_fn._qvalue_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn._actor_network_params
                )
            elif k == "loss_qvalue":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn._actor_network_params
                )
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn._qvalue_network_params
                )
            elif k == "loss_alpha":
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn._actor_network_params
                )
                assert all(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn._qvalue_network_params
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
        for p in loss_fn.qvalue_network_params:
            if not isinstance(p, nn.Parameter):
                counter += 1
                assert (p == loss_fn._param_maps[p]).all()
                assert (p == 0).all()
        assert counter == len(loss_fn._actor_network_params)
        assert counter == len(loss_fn.actor_network_params)

        # check that params of the original actor are those of the loss_fn
        for p in actor.parameters():
            assert p in set(loss_fn.parameters())

        if delay_qvalue:
            # test that updating with target updater resets the targets of qvalue to 0
            target_updater.step()

    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    @pytest.mark.parametrize("device", get_available_devices())
    def test_redq_batched(self, delay_qvalue, num_qvalue, device):

        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq(device=device)

        actor = self._create_mock_actor(device=device)
        qvalue = self._create_mock_qvalue(device=device)

        loss_fn = REDQLoss(
            actor_network=deepcopy(actor),
            qvalue_network=deepcopy(qvalue),
            num_qvalue_nets=num_qvalue,
            gamma=0.9,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )

        loss_class_deprec = (
            REDQLoss_deprecated if not delay_qvalue else DoubleREDQLoss_deprecated
        )
        loss_fn_deprec = loss_class_deprec(
            actor_network=deepcopy(actor),
            qvalue_network=deepcopy(qvalue),
            num_qvalue_nets=num_qvalue,
            gamma=0.9,
            loss_function="l2",
        )

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
            gamma=0.9,
            loss_function="l2",
            delay_qvalue=delay_qvalue,
        )

        ms = MultiStep(gamma=gamma, n_steps_max=n).to(device)

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
            assert_allclose_td(td, ms_td.select(*list(td.keys())))
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
        target_actor = [p.clone() for p in loss_fn.target_actor_network_params]
        target_qvalue = [p.clone() for p in loss_fn.target_qvalue_network_params]
        for p in loss_fn.parameters():
            p.data += torch.randn_like(p)
        target_actor2 = [p.clone() for p in loss_fn.target_actor_network_params]
        target_qvalue2 = [p.clone() for p in loss_fn.target_qvalue_network_params]
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
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=module,
            distribution_class=TanhNormal,
            dist_in_keys=["loc", "scale"],
            spec=CompositeSpec(action=action_spec, loc=None, scale=None),
        )
        return actor.to(device)

    def _create_mock_value(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=["observation"],
        )
        return value.to(device)

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
                "next": {"observation": next_obs},
                "done": done,
                "reward": reward,
                "action": action,
                "sample_log_prob": torch.randn_like(action[..., :1]) / 10,
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
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        params_mean = torch.randn_like(action) / 10
        params_scale = torch.rand_like(action) / 10
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next": {"observation": next_obs * mask.to(obs.dtype)},
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
                "sample_log_prob": torch.randn_like(action[..., :1])
                / 10
                * mask.to(obs.dtype),
                "loc": params_mean * mask.to(obs.dtype),
                "scale": params_scale * mask.to(obs.dtype),
            },
            device=device,
        )
        return td

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "td", "td_lambda"))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_ppo(self, loss_class, device, gradient_mode, advantage):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(device=device)

        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, gradient_mode=gradient_mode
            )
        elif advantage == "td":
            advantage = TDEstimate(
                gamma=0.9, value_network=value, gradient_mode=gradient_mode
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimate(
                gamma=0.9, lmbda=0.9, value_network=value, gradient_mode=gradient_mode
            )
        else:
            raise NotImplementedError

        loss_fn = loss_class(
            actor, value, advantage_module=advantage, gamma=0.9, loss_critic_type="l2"
        )

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

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "td", "td_lambda"))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_ppo_diff(self, loss_class, device, gradient_mode, advantage):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo(device=device)

        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, gradient_mode=gradient_mode
            )
        elif advantage == "td":
            advantage = TDEstimate(
                gamma=0.9, value_network=value, gradient_mode=gradient_mode
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimate(
                gamma=0.9, lmbda=0.9, value_network=value, gradient_mode=gradient_mode
            )
        else:
            raise NotImplementedError

        loss_fn = loss_class(
            actor, value, advantage_module=advantage, gamma=0.9, loss_critic_type="l2"
        )

        floss_fn, params, buffers = make_functional_with_buffers(loss_fn)

        loss = floss_fn(params, buffers, td)
        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        if _has_functorch:
            for (name, _), p in zip(named_parameters, params):
                if p.grad is not None and p.grad.norm() > 0.0:
                    assert "actor" not in name
                    assert "critic" in name
                if p.grad is None:
                    assert "actor" in name
                    assert "critic" not in name
        else:
            for key, p in params.flatten_keys(".").items():
                if p.grad is not None and p.grad.norm() > 0.0:
                    assert "actor" not in key
                    assert "value" in key or "critic" in key
                if p.grad is None:
                    assert "actor" in key
                    assert "value" not in key and "critic" not in key

        if _has_functorch:
            for param in params:
                param.grad = None
        else:
            for param in params.flatten_keys(".").values():
                param.grad = None
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        if _has_functorch:
            for (name, _), p in zip(named_parameters, params):
                if p.grad is not None and p.grad.norm() > 0.0:
                    assert "actor" in name
                    assert "critic" not in name
                if p.grad is None:
                    assert "actor" not in name
                    assert "critic" in name
            for param in params:
                param.grad = None
        else:
            for key, p in params.flatten_keys(".").items():
                if p.grad is not None and p.grad.norm() > 0.0:
                    assert "actor" in key
                    assert "value" not in key and "critic" not in key
                if p.grad is None:
                    assert "actor" not in key
                    assert "value" in key or "critic" in key
            for param in params.flatten_keys(".").values():
                param.grad = None


class TestA2C:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor = ProbabilisticActor(
            module=module,
            distribution_class=TanhNormal,
            dist_in_keys=["loc", "scale"],
            spec=CompositeSpec(action=action_spec, loc=None, scale=None),
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
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        params_mean = torch.randn_like(action) / 10
        params_scale = torch.rand_like(action) / 10
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next": {"observation": next_obs * mask.to(obs.dtype)},
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
                "sample_log_prob": torch.randn_like(action[..., :1])
                / 10
                * mask.to(obs.dtype),
                "loc": params_mean * mask.to(obs.dtype),
                "scale": params_scale * mask.to(obs.dtype),
            },
            device=device,
        )
        return td

    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "td", "td_lambda"))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_a2c(self, device, gradient_mode, advantage):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_a2c(device=device)

        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, gradient_mode=gradient_mode
            )
        elif advantage == "td":
            advantage = TDEstimate(
                gamma=0.9, value_network=value, gradient_mode=gradient_mode
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimate(
                gamma=0.9, lmbda=0.9, value_network=value, gradient_mode=gradient_mode
            )
        else:
            raise NotImplementedError

        loss_fn = A2CLoss(
            actor, value, advantage_module=advantage, gamma=0.9, loss_critic_type="l2"
        )

        # Check error is raised when actions require grads
        td["action"].requires_grad = True
        with pytest.raises(
            RuntimeError,
            match="tensordict stored action require grad.",
        ):
            loss = loss_fn._log_probs(td)
        td["action"].requires_grad = False

        # Check error is raised when advantage_diff_key present and does not required grad
        td[loss_fn.advantage_diff_key] = torch.randn_like(td["reward"])
        with pytest.raises(
            RuntimeError,
            match="value_target retrieved from tensordict does not require grad.",
        ):
            loss = loss_fn.loss_critic(td)
        td = td.exclude(loss_fn.advantage_diff_key)
        assert loss_fn.advantage_diff_key not in td.keys()

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

    @pytest.mark.parametrize("gradient_mode", (True, False))
    @pytest.mark.parametrize("advantage", ("gae", "td", "td_lambda"))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_a2c_diff(self, device, gradient_mode, advantage):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_a2c(device=device)

        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        if advantage == "gae":
            advantage = GAE(
                gamma=0.9, lmbda=0.9, value_network=value, gradient_mode=gradient_mode
            )
        elif advantage == "td":
            advantage = TDEstimate(
                gamma=0.9, value_network=value, gradient_mode=gradient_mode
            )
        elif advantage == "td_lambda":
            advantage = TDLambdaEstimate(
                gamma=0.9, lmbda=0.9, value_network=value, gradient_mode=gradient_mode
            )
        else:
            raise NotImplementedError

        loss_fn = A2CLoss(
            actor, value, advantage_module=advantage, gamma=0.9, loss_critic_type="l2"
        )

        floss_fn, params, buffers = make_functional_with_buffers(loss_fn)

        loss = floss_fn(params, buffers, td)
        loss_critic = loss["loss_critic"]
        loss_objective = loss["loss_objective"] + loss.get("loss_entropy", 0.0)
        loss_critic.backward(retain_graph=True)
        # check that grads are independent and non null
        named_parameters = loss_fn.named_parameters()
        if _has_functorch:
            for (name, _), p in zip(named_parameters, params):
                if p.grad is not None and p.grad.norm() > 0.0:
                    assert "actor" not in name
                    assert "critic" in name
                if p.grad is None:
                    assert "actor" in name
                    assert "critic" not in name
        else:
            for key, p in params.flatten_keys(".").items():
                if p.grad is not None and p.grad.norm() > 0.0:
                    assert "actor" not in key
                    assert "value" in key or "critic" in key
                if p.grad is None:
                    assert "actor" in key
                    assert "value" not in key and "critic" not in key

        if _has_functorch:
            for param in params:
                param.grad = None
        else:
            for param in params.flatten_keys(".").values():
                param.grad = None
        loss_objective.backward()
        named_parameters = loss_fn.named_parameters()
        if _has_functorch:
            for (name, _), p in zip(named_parameters, params):
                if p.grad is not None and p.grad.norm() > 0.0:
                    assert "actor" in name
                    assert "critic" not in name
                if p.grad is None:
                    assert "actor" not in name
                    assert "critic" in name
            for param in params:
                param.grad = None
        else:
            for key, p in params.flatten_keys(".").items():
                if p.grad is not None and p.grad.norm() > 0.0:
                    assert "actor" in key
                    assert "value" not in key and "critic" not in key
                if p.grad is None:
                    assert "actor" not in key
                    assert "value" in key or "critic" in key
            for param in params.flatten_keys(".").values():
                param.grad = None


class TestReinforce:
    @pytest.mark.parametrize("delay_value", [True, False])
    @pytest.mark.parametrize("gradient_mode", [True, False])
    @pytest.mark.parametrize("advantage", ["gae", "td", "td_lambda"])
    def test_reinforce_value_net(self, advantage, gradient_mode, delay_value):
        n_obs = 3
        n_act = 5
        batch = 4
        gamma = 0.9
        value_net = ValueOperator(nn.Linear(n_obs, 1), in_keys=["observation"])
        net = NormalParamWrapper(nn.Linear(n_obs, 2 * n_act))
        module = TensorDictModule(
            net, in_keys=["observation"], out_keys=["loc", "scale"]
        )
        actor_net = ProbabilisticActor(
            module,
            distribution_class=TanhNormal,
            return_log_prob=True,
            dist_in_keys=["loc", "scale"],
            spec=CompositeSpec(
                action=NdUnboundedContinuousTensorSpec(n_act), loc=None, scale=None
            ),
        )
        if advantage == "gae":
            advantage_module = GAE(
                gamma=gamma,
                lmbda=0.9,
                value_network=value_net.make_functional_with_buffers(clone=True)[0],
                gradient_mode=gradient_mode,
            )
        elif advantage == "td":
            advantage_module = TDEstimate(
                gamma=gamma,
                value_network=value_net.make_functional_with_buffers(clone=True)[0],
                gradient_mode=gradient_mode,
            )
        elif advantage == "td_lambda":
            advantage_module = TDLambdaEstimate(
                gamma=0.9,
                lmbda=0.9,
                value_network=value_net.make_functional_with_buffers(clone=True)[0],
                gradient_mode=gradient_mode,
            )
        else:
            raise NotImplementedError

        loss_fn = ReinforceLoss(
            actor_net,
            critic=value_net,
            gamma=gamma,
            advantage_module=advantage_module,
            delay_value=delay_value,
        )

        td = TensorDict(
            {
                "reward": torch.randn(batch, 1),
                "observation": torch.randn(batch, n_obs),
                "next": {"observation": torch.randn(batch, n_obs)},
                "done": torch.zeros(batch, 1, dtype=torch.bool),
                "action": torch.randn(batch, n_act),
            },
            [batch],
        )

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
                "next": {"pixels": torch.randn(batch_size, temporal_length, 3, 64, 64)},
                "action": torch.randn(batch_size, temporal_length, 64),
                "reward": torch.randn(batch_size, temporal_length, 1),
                "done": torch.zeros(batch_size, temporal_length, dtype=torch.bool),
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
            "state": NdUnboundedContinuousTensorSpec(state_dim),
            "belief": NdUnboundedContinuousTensorSpec(rssm_hidden_dim),
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
            TensorDictModule(
                rssm_prior,
                in_keys=["state", "belief", "action"],
                out_keys=[
                    ("next", "prior_mean"),
                    ("next", "prior_std"),
                    "_",
                    ("next", "belief"),
                ],
            ),
            TensorDictModule(
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
        world_modeler = TensorDictSequential(
            TensorDictModule(
                obs_encoder,
                in_keys=[("next", "pixels")],
                out_keys=[("next", "encoded_latents")],
            ),
            rssm_rollout,
            TensorDictModule(
                obs_decoder,
                in_keys=[("next", "state"), ("next", "belief")],
                out_keys=[("next", "reco_pixels")],
            ),
        )
        reward_module = TensorDictModule(
            reward_module,
            in_keys=[("next", "state"), ("next", "belief")],
            out_keys=["reward"],
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
            "state": NdUnboundedContinuousTensorSpec(state_dim),
            "belief": NdUnboundedContinuousTensorSpec(rssm_hidden_dim),
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
        transition_model = TensorDictSequential(
            TensorDictModule(
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
        reward_model = TensorDictModule(
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
            "state": NdUnboundedContinuousTensorSpec(state_dim),
            "belief": NdUnboundedContinuousTensorSpec(rssm_hidden_dim),
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
        actor_model = ProbabilisticTensorDictModule(
            TensorDictModule(
                actor_module,
                in_keys=["state", "belief"],
                out_keys=["loc", "scale"],
            ),
            dist_in_keys=["loc", "scale"],
            sample_out_key="action",
            default_interaction_mode="random",
            distribution_class=TanhNormal,
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
        value_model = TensorDictModule(
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
        mb_env.obs_decoder = TensorDictModule(
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
    def test_dreamer_actor(self, device, imagination_horizon, discount_loss):
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
def test_updater(mode, value_network_update_interval, device):
    torch.manual_seed(100)

    class custom_module_error(nn.Module):
        def __init__(self):
            super().__init__()
            self._target_params = [torch.randn(3, 4)]
            self._target_error_params = [torch.randn(3, 4)]
            self.params = nn.ParameterList(
                [nn.Parameter(torch.randn(3, 4, requires_grad=True))]
            )

    module = custom_module_error().to(device)
    with pytest.raises(
        RuntimeError, match="Your module seems to have a _target tensor list "
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
            if _has_functorch:
                iterator_params = self.target_module1_params
                iterator_buffers = self.target_module1_buffers
            else:
                iterator_params = self.target_module1_params.values()
                iterator_buffers = self.target_module1_buffers.values()
            for target in iterator_params:
                target.data.normal_()
            for target in iterator_buffers:
                if target.dtype is not torch.int64:
                    target.data.normal_()
                else:
                    target.data += 10

    module = custom_module().to(device)
    if mode == "hard":
        upd = HardUpdate(
            module, value_network_update_interval=value_network_update_interval
        )
    elif mode == "soft":
        upd = SoftUpdate(module, 1 - 1 / value_network_update_interval)
    upd.init_()
    for _, v in upd._targets.items():
        if isinstance(v, TensorDictBase):
            for _v in v.values():
                if _v.dtype is not torch.int64:
                    _v.copy_(torch.randn_like(_v))
                else:
                    _v += 10
        else:
            for _v in v:
                if _v.dtype is not torch.int64:
                    _v.copy_(torch.randn_like(_v))
                else:
                    _v += 10

    # total dist
    if _has_functorch:
        d0 = sum(
            [
                (target_val[0] - val[0]).norm().item()
                for (_, target_val), (_, val) in zip(
                    upd._targets.items(), upd._sources.items()
                )
            ]
        )
    else:
        d0 = 0.0
        for (_, target_val), (_, val) in zip(
            upd._targets.items(), upd._sources.items()
        ):
            for key in target_val.keys():
                if target_val[key].dtype == torch.long:
                    continue
                d0 += (target_val[key] - val[key]).norm().item()

    assert d0 > 0
    if mode == "hard":
        for i in range(value_network_update_interval + 1):
            # test that no update is occuring until value_network_update_interval
            if _has_functorch:
                d1 = sum(
                    [
                        (target_val[0] - val[0]).norm().item()
                        for (_, target_val), (_, val) in zip(
                            upd._targets.items(), upd._sources.items()
                        )
                    ]
                )
            else:
                d1 = 0.0
                for (_, target_val), (_, val) in zip(
                    upd._targets.items(), upd._sources.items()
                ):
                    for key in target_val.keys():
                        if target_val[key].dtype == torch.long:
                            continue
                        d1 += (target_val[key] - val[key]).norm().item()

            assert d1 == d0, i
            assert upd.counter == i
            upd.step()
        assert upd.counter == 0
        # test that a new update has occured
        if _has_functorch:
            d1 = sum(
                [
                    (target_val[0] - val[0]).norm().item()
                    for (_, target_val), (_, val) in zip(
                        upd._targets.items(), upd._sources.items()
                    )
                ]
            )
        else:
            d1 = 0.0
            for (_, target_val), (_, val) in zip(
                upd._targets.items(), upd._sources.items()
            ):
                for key in target_val.keys():
                    if target_val[key].dtype == torch.long:
                        continue
                    d1 += (target_val[key] - val[key]).norm().item()
        assert d1 < d0

    elif mode == "soft":
        upd.step()
        if _has_functorch:
            d1 = sum(
                [
                    (target_val[0] - val[0]).norm().item()
                    for (_, target_val), (_, val) in zip(
                        upd._targets.items(), upd._sources.items()
                    )
                ]
            )
        else:
            d1 = 0.0
            for (_, target_val), (_, val) in zip(
                upd._targets.items(), upd._sources.items()
            ):
                for key in target_val.keys():
                    if target_val[key].dtype == torch.long:
                        continue
                    d1 += (target_val[key] - val[key]).norm().item()
        assert d1 < d0

    upd.init_()
    upd.step()
    if _has_functorch:
        d2 = sum(
            [
                (target_val[0] - val[0]).norm().item()
                for (_, target_val), (_, val) in zip(
                    upd._targets.items(), upd._sources.items()
                )
            ]
        )
    else:
        d2 = 0.0
        for (_, target_val), (_, val) in zip(
            upd._targets.items(), upd._sources.items()
        ):
            for key in target_val.keys():
                if target_val[key].dtype == torch.long:
                    continue
                d2 += (target_val[key] - val[key]).norm().item()
    assert d2 < 1e-6


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("gamma", [0.1, 0.5, 0.99])
@pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
@pytest.mark.parametrize("N", [(3,), (7, 3)])
@pytest.mark.parametrize("T", [3, 5, 200])
# @pytest.mark.parametrize("random_gamma,rolling_gamma", [[True, False], [True, True], [False, None]])
@pytest.mark.parametrize("random_gamma,rolling_gamma", [[False, None]])
def test_tdlambda(device, gamma, lmbda, N, T, random_gamma, rolling_gamma):
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
@pytest.mark.parametrize("gamma", [0.99, 0.5, 0.1])
@pytest.mark.parametrize("lmbda", [0.99, 0.5, 0.1])
@pytest.mark.parametrize("N", [(3,), (7, 3)])
@pytest.mark.parametrize("T", [200, 5, 3])
@pytest.mark.parametrize("dtype", [torch.float, torch.double])
@pytest.mark.parametrize("dones", [True, False])
def test_gae(device, gamma, lmbda, N, T, dtype, dones):
    torch.manual_seed(0)

    done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
    if dones:
        done = done.bernoulli_(0.1).cumsum(-2).to(torch.bool)
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
@pytest.mark.parametrize("gamma", [0.5, 0.99, 0.1])
@pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
@pytest.mark.parametrize("N", [(3,), (7, 3)])
@pytest.mark.parametrize("T", [3, 5, 200])
def test_tdlambda_tensor_gamma(device, gamma, lmbda, N, T):
    """Tests vec_td_lambda_advantage_estimate against itself with
    gamma being a tensor or a scalar

    """
    torch.manual_seed(0)

    done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
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
@pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
@pytest.mark.parametrize("N", [(3,), (7, 3)])
@pytest.mark.parametrize("T", [3, 5, 50])
def test_vectdlambda_tensor_gamma(device, gamma, lmbda, N, T, dtype_fixture):  # noqa
    """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
    with gamma being a tensor or a scalar

    """

    torch.manual_seed(0)

    done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
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
@pytest.mark.parametrize("lmbda", [0.1, 0.5, 0.99])
@pytest.mark.parametrize("N", [(3,), (7, 3)])
@pytest.mark.parametrize("T", [50, 3])
@pytest.mark.parametrize("rolling_gamma", [True, False, None])
def test_vectdlambda_rand_gamma(
    device, lmbda, N, T, rolling_gamma, dtype_fixture  # noqa
):
    """Tests td_lambda_advantage_estimate against vec_td_lambda_advantage_estimate
    with gamma being a random tensor

    """
    torch.manual_seed(0)

    done = torch.zeros(*N, T, 1, device=device, dtype=torch.bool)
    reward = torch.randn(*N, T, 1, device=device)
    state_value = torch.randn(*N, T, 1, device=device)
    next_state_value = torch.randn(*N, T, 1, device=device)

    # avoid low values of gamma
    gamma_tensor = 0.5 + torch.rand_like(next_state_value) / 2

    v1 = td_lambda_advantage_estimate(
        gamma_tensor, lmbda, state_value, next_state_value, reward, done, rolling_gamma
    )
    v2 = vec_td_lambda_advantage_estimate(
        gamma_tensor, lmbda, state_value, next_state_value, reward, done, rolling_gamma
    )
    torch.testing.assert_close(v1, v2, rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("gamma", [0.99, "rand"])
@pytest.mark.parametrize("N", [(3,), (3, 7)])
@pytest.mark.parametrize("T", [3, 5, 200])
@pytest.mark.parametrize("rolling_gamma", [True, False])
def test_custom_conv1d_tensor(device, gamma, N, T, rolling_gamma):
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


@pytest.mark.skipif(not _has_functorch, reason="no vmap allowed without functorch")
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
    td_module_hidden = TensorDictModule(
        module=module_hidden,
        spec=None,
        in_keys=["observation"],
        out_keys=["hidden"],
    )
    module_action = TensorDictModule(
        NormalParamWrapper(torch.nn.Linear(4, 8)),
        in_keys=["hidden"],
        out_keys=["loc", "scale"],
    )
    td_module_action = ProbabilisticActor(
        module=module_action,
        spec=None,
        dist_in_keys=["loc", "scale"],
        sample_out_key=["action"],
        distribution_class=TanhNormal,
        return_log_prob=True,
    )
    module_value = torch.nn.Linear(4, 1)
    td_module_value = ValueOperator(
        module=module_value,
        in_keys=["hidden"],
    )
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
    assert len(loss.actor_network_params) == 4
    assert len(loss.qvalue_network_params) == 4
    for p in loss.actor_network_params:
        assert isinstance(p, nn.Parameter)
    assert (loss.qvalue_network_params[0] == loss.actor_network_params[0]).all()
    assert (loss.qvalue_network_params[1] == loss.actor_network_params[1]).all()

    # map module
    if dest == "double":
        loss = loss.double()
    elif dest == "cuda":
        loss = loss.cuda()
    elif dest == "half":
        loss = loss.half()
    else:
        loss = loss.to(dest)

    for p in loss.actor_network_params:
        assert isinstance(p, nn.Parameter)
        assert p.dtype is expected_dtype
        assert p.device == torch.device(expected_device)
    assert loss.qvalue_network_params[0].dtype is expected_dtype
    assert loss.qvalue_network_params[1].dtype is expected_dtype
    assert loss.qvalue_network_params[0].device == torch.device(expected_device)
    assert loss.qvalue_network_params[1].device == torch.device(expected_device)
    assert (loss.qvalue_network_params[0] == loss.actor_network_params[0]).all()
    assert (loss.qvalue_network_params[1] == loss.actor_network_params[1]).all()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
