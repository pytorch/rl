import argparse
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn
from torchrl.data import TensorDict, NdBoundedTensorSpec, MultOneHotDiscreteTensorSpec
from torchrl.data.postprocs.postprocs import MultiStep
from _utils_internal import get_available_devices

# from torchrl.data.postprocs.utils import expand_as_right
from torchrl.data.tensordict.tensordict import assert_allclose_td
from torchrl.data.utils import expand_as_right
from torchrl.modules import DistributionalQValueActor, QValueActor
from torchrl.modules.distributions.continuous import TanhNormal, NormalParamWrapper
from torchrl.modules.models.models import MLP
from torchrl.modules.td_module.actors import ValueOperator, Actor, ProbabilisticActor
from torchrl.objectives import (
    DQNLoss,
    DoubleDQNLoss,
    DistributionalDQNLoss,
    DistributionalDoubleDQNLoss,
    DDPGLoss,
    DoubleDDPGLoss,
    SACLoss,
    DoubleSACLoss,
    PPOLoss,
    ClipPPOLoss,
    KLPENPPOLoss,
    GAE,
)
from torchrl.objectives.costs.redq import (
    REDQLoss,
    DoubleREDQLoss,
    REDQLoss_deprecated,
    DoubleREDQLoss_deprecated,
)
from torchrl.objectives.costs.utils import hold_out_net


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

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4,
                           device="cpu"):
        # Actor
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = nn.Linear(obs_dim, action_dim)
        actor = QValueActor(
            spec=action_spec,
            module=module,
        ).to(device)
        return actor

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        # Actor
        action_spec = MultOneHotDiscreteTensorSpec([atoms] * action_dim)
        support = torch.linspace(vmin, vmax, atoms, dtype=torch.float)
        module = MLP(obs_dim, (atoms, action_dim))
        actor = DistributionalQValueActor(
            spec=action_spec,
            module=module,
            support=support,
        )
        return actor

    def _create_mock_data_dqn(self, batch=2, obs_dim=3, action_dim=4,
                              atoms=None, device="cpu"):
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
        reward = torch.randn(batch, 1)
        done = torch.zeros(batch, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next_observation": next_obs,
                "done": done,
                "reward": reward,
                "action": action,
                "action_value": action_value,
            },
        ).to(device)
        return td

    def _create_seq_mock_data_dqn(
        self, batch=2, T=4, obs_dim=3, action_dim=4, atoms=None, device="cpu"
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action_value = torch.randn(batch, T, atoms, action_dim).softmax(-2)
            action = (
                action_value[..., 0, :] == action_value[..., 0, :].max(-1, True)[0]
            ).to(torch.long)
        else:
            action_value = torch.randn(batch, T, action_dim)
            action = (action_value == action_value.max(-1, True)[0]).to(torch.long)
        reward = torch.randn(batch, T, 1)
        done = torch.zeros(batch, T, 1, dtype=torch.bool)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next_observation": next_obs * mask.to(obs.dtype),
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
                "action_value": action_value
                * expand_as_right(mask.to(obs.dtype).squeeze(-1), action_value),
            },
        )
        return td

    @pytest.mark.parametrize("loss_class", (DQNLoss, DoubleDQNLoss))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_dqn(self, loss_class, device):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        td = self._create_mock_data_dqn(device=device)
        loss_fn = loss_class(actor, gamma=0.9, loss_function="l2")
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

    @pytest.mark.parametrize("n", range(4))
    @pytest.mark.parametrize("loss_class", (DQNLoss, DoubleDQNLoss))
    @pytest.mark.parametrize("device", get_available_devices())
    def test_dqn_batcher(self, n, loss_class, device, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device)

        td = self._create_seq_mock_data_dqn(device)
        loss_fn = loss_class(actor, gamma=gamma, loss_function="l2")

        ms = MultiStep(gamma=gamma, n_steps_max=n)
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

    @pytest.mark.parametrize("atoms", range(4, 10))
    @pytest.mark.parametrize(
        "loss_class", (DistributionalDQNLoss, DistributionalDoubleDQNLoss)
    )
    @pytest.mark.parametrize("device", get_devices())
    def test_distributional_dqn(self, atoms, loss_class, device, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_distributional_actor(atoms=atoms).to(device)

        td = self._create_mock_data_dqn(atoms=atoms).to(device)
        loss_fn = loss_class(actor, gamma=gamma)

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


class TestDDPG:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4):
        # Actor
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = nn.Linear(obs_dim, action_dim)
        actor = Actor(
            spec=action_spec,
            module=module,
        )
        return actor

    def _create_mock_value(self, batch=2, obs_dim=3, action_dim=4):
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
        return value

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_ddpg(self, batch=8, obs_dim=3, action_dim=4, atoms=None):
        # create a tensordict
        obs = torch.randn(batch, obs_dim)
        next_obs = torch.randn(batch, obs_dim)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim).clamp(-1, 1)
        reward = torch.randn(batch, 1)
        done = torch.zeros(batch, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next_observation": next_obs,
                "done": done,
                "reward": reward,
                "action": action,
            },
        )
        return td

    def _create_seq_mock_data_ddpg(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim).clamp(-1, 1)
        else:
            action = torch.randn(batch, T, action_dim).clamp(-1, 1)
        reward = torch.randn(batch, T, 1)
        done = torch.zeros(batch, T, 1, dtype=torch.bool)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next_observation": next_obs * mask.to(obs.dtype),
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
            },
        )
        return td

    @pytest.mark.parametrize("loss_class", (DDPGLoss, DoubleDDPGLoss))
    def test_ddpg(self, loss_class):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        td = self._create_mock_data_ddpg()
        loss_fn = loss_class(actor, value, gamma=0.9, loss_function="l2")
        with _check_td_steady(td):
            loss = loss_fn(td)

        # check that loss are independent
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

    @pytest.mark.parametrize("n", list(range(4)))
    @pytest.mark.parametrize("loss_class", (DDPGLoss, DoubleDDPGLoss))
    def test_ddpg_batcher(self, n, loss_class, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor()
        value = self._create_mock_value()

        td = self._create_seq_mock_data_ddpg()
        loss_fn = loss_class(actor, value, gamma=gamma, loss_function="l2")

        ms = MultiStep(gamma=gamma, n_steps_max=n)
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

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4):
        # Actor
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        actor = ProbabilisticActor(
            spec=action_spec,
            module=module,
            distribution_class=TanhNormal,
        )
        return actor

    def _create_mock_qvalue(self, batch=2, obs_dim=3, action_dim=4):
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
        return qvalue

    def _create_mock_value(self, batch=2, obs_dim=3, action_dim=4):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=["observation"],
        )
        return value

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_sac(self, batch=16, obs_dim=3, action_dim=4, atoms=None):
        # create a tensordict
        obs = torch.randn(batch, obs_dim)
        next_obs = torch.randn(batch, obs_dim)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim).clamp(-1, 1)
        reward = torch.randn(batch, 1)
        done = torch.zeros(batch, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next_observation": next_obs,
                "done": done,
                "reward": reward,
                "action": action,
            },
        )
        return td

    def _create_seq_mock_data_sac(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim).clamp(-1, 1)
        else:
            action = torch.randn(batch, T, action_dim).clamp(-1, 1)
        reward = torch.randn(batch, T, 1)
        done = torch.zeros(batch, T, 1, dtype=torch.bool)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next_observation": next_obs * mask.to(obs.dtype),
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
            },
        )
        return td

    @pytest.mark.parametrize("loss_class", (SACLoss, DoubleSACLoss))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    def test_sac(self, loss_class, delay_actor, delay_qvalue, num_qvalue):
        if (delay_actor or delay_qvalue) and loss_class is not DoubleSACLoss:
            pytest.skip("incompatible config")

        torch.manual_seed(self.seed)
        td = self._create_mock_data_sac()

        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        value = self._create_mock_value()

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True

        loss_fn = loss_class(
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

        # check that loss are independent
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

        assert len(set(p for n, p in named_parameters)) == len(list(named_parameters))
        assert len(set(p for n, p in named_buffers)) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("n", list(range(4)))
    @pytest.mark.parametrize("loss_class", (SACLoss, DoubleSACLoss))
    @pytest.mark.parametrize("delay_actor", (True, False))
    @pytest.mark.parametrize("delay_qvalue", (True, False))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    def test_sac_batcher(
        self, n, loss_class, delay_actor, delay_qvalue, num_qvalue, gamma=0.9
    ):
        if (delay_actor or delay_qvalue) and (loss_class is not DoubleSACLoss):
            pytest.skip("incompatible config")
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_sac()

        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()
        value = self._create_mock_value()

        kwargs = {}
        if delay_actor:
            kwargs["delay_actor"] = True
        if delay_qvalue:
            kwargs["delay_qvalue"] = True

        loss_fn = loss_class(
            actor_network=actor,
            qvalue_network=qvalue,
            value_network=value,
            num_qvalue_nets=num_qvalue,
            gamma=0.9,
            loss_function="l2",
            **kwargs,
        )

        ms = MultiStep(gamma=gamma, n_steps_max=n)

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


class TestREDQ:
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4):
        # Actor
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        actor = ProbabilisticActor(
            spec=action_spec,
            module=module,
            distribution_class=TanhNormal,
            return_log_prob=True,
        )
        return actor

    def _create_mock_qvalue(self, batch=2, obs_dim=3, action_dim=4):
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
        return qvalue

    def _create_mock_data_redq(self, batch=16, obs_dim=3, action_dim=4, atoms=None):
        # create a tensordict
        obs = torch.randn(batch, obs_dim)
        next_obs = torch.randn(batch, obs_dim)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim).clamp(-1, 1)
        reward = torch.randn(batch, 1)
        done = torch.zeros(batch, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next_observation": next_obs,
                "done": done,
                "reward": reward,
                "action": action,
            },
        )
        return td

    def _create_seq_mock_data_redq(
        self, batch=8, T=4, obs_dim=3, action_dim=4, atoms=None
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim).clamp(-1, 1)
        else:
            action = torch.randn(batch, T, action_dim).clamp(-1, 1)
        reward = torch.randn(batch, T, 1)
        done = torch.zeros(batch, T, 1, dtype=torch.bool)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next_observation": next_obs * mask.to(obs.dtype),
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
            },
        )
        return td

    @pytest.mark.parametrize("loss_class", (REDQLoss, DoubleREDQLoss))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    def test_redq(self, loss_class, num_qvalue):

        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq()

        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()

        loss_fn = loss_class(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            gamma=0.9,
            loss_function="l2",
        )

        with _check_td_steady(td):
            loss = loss_fn(td)

        # check td is left untouched
        assert loss_fn.priority_key in td.keys()

        # check that loss are independent
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

        assert len(set(p for n, p in named_parameters)) == len(list(named_parameters))
        assert len(set(p for n, p in named_buffers)) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("loss_class", (REDQLoss, DoubleREDQLoss))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    def test_redq_batched(self, loss_class, num_qvalue):

        torch.manual_seed(self.seed)
        td = self._create_mock_data_redq()

        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()

        loss_fn = loss_class(
            actor_network=deepcopy(actor),
            qvalue_network=deepcopy(qvalue),
            num_qvalue_nets=num_qvalue,
            gamma=0.9,
            loss_function="l2",
        )

        loss_class_deprec = (
            REDQLoss_deprecated if loss_class is REDQLoss else DoubleREDQLoss_deprecated
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
            loss1 = loss_fn(td_clone1)

        torch.manual_seed(0)
        with _check_td_steady(td_clone2):
            loss2 = loss_fn_deprec(td_clone2)

        # TODO: find a way to compare the losses: problem is that we sample actions either sequentially or in batch,
        #  so setting seed has little impact

    @pytest.mark.parametrize("n", list(range(4)))
    @pytest.mark.parametrize("loss_class", (REDQLoss, DoubleREDQLoss))
    @pytest.mark.parametrize("num_qvalue", [1, 2, 4, 8])
    def test_redq_batcher(self, n, loss_class, num_qvalue, gamma=0.9):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_redq()

        actor = self._create_mock_actor()
        qvalue = self._create_mock_qvalue()

        loss_fn = loss_class(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=num_qvalue,
            gamma=0.9,
            loss_function="l2",
        )

        ms = MultiStep(gamma=gamma, n_steps_max=n)

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

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4):
        # Actor
        action_spec = NdBoundedTensorSpec(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = NormalParamWrapper(nn.Linear(obs_dim, 2 * action_dim))
        actor = ProbabilisticActor(
            spec=action_spec,
            module=module,
            distribution_class=TanhNormal,
            save_dist_params=True,
        )
        return actor

    def _create_mock_value(self, batch=2, obs_dim=3, action_dim=4):
        module = nn.Linear(obs_dim, 1)
        value = ValueOperator(
            module=module,
            in_keys=["observation"],
        )
        return value

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=0, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_data_ppo(self, batch=2, obs_dim=3, action_dim=4, atoms=None):
        # create a tensordict
        obs = torch.randn(batch, obs_dim)
        next_obs = torch.randn(batch, obs_dim)
        if atoms:
            raise NotImplementedError
        else:
            action = torch.randn(batch, action_dim).clamp(-1, 1)
        reward = torch.randn(batch, 1)
        done = torch.zeros(batch, 1, dtype=torch.bool)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "next_observation": next_obs,
                "done": done,
                "reward": reward,
                "action": action,
                "action_log_prob": torch.randn_like(action[..., :1]) / 10,
            },
        )
        return td

    def _create_seq_mock_data_ppo(
        self, batch=2, T=4, obs_dim=3, action_dim=4, atoms=None
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        if atoms:
            action = torch.randn(batch, T, atoms, action_dim).clamp(-1, 1)
        else:
            action = torch.randn(batch, T, action_dim).clamp(-1, 1)
        reward = torch.randn(batch, T, 1)
        done = torch.zeros(batch, T, 1, dtype=torch.bool)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool)
        params_mean = torch.randn_like(action.repeat(1, 1, 2)) / 10
        params_scale = torch.rand_like(action.repeat(1, 1, 2)) / 10
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next_observation": next_obs * mask.to(obs.dtype),
                "done": done,
                "mask": mask,
                "reward": reward * mask.to(obs.dtype),
                "action": action * mask.to(obs.dtype),
                "action_log_prob": torch.randn_like(action[..., :1])
                / 10
                * mask.to(obs.dtype),
                "action_dist_param_0": params_mean * mask.to(obs.dtype),
                "action_dist_param_1": params_scale * mask.to(obs.dtype),
            },
        )
        return td

    @pytest.mark.parametrize("loss_class", (PPOLoss, ClipPPOLoss, KLPENPPOLoss))
    def test_ppo(self, loss_class):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_ppo()

        actor = self._create_mock_actor()
        value = self._create_mock_value()
        gae = GAE(gamma=0.9, lamda=0.9, critic=value)
        loss_fn = loss_class(
            actor, value, advantage_module=gae, gamma=0.9, loss_critic_type="l2"
        )

        loss = loss_fn(td)
        sum([item for _, item in loss.items()]).backward()
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"


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
    with pytest.raises(
        RuntimeError,
        match="hold_out_net requires the network parameter set to be non-empty.",
    ):
        net = torch.nn.Sequential()
        with hold_out_net(net):
            pass


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
