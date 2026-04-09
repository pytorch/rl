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
from packaging import version as pkg_version

from tensordict import assert_allclose_td, TensorDict
from tensordict.nn import (
    NormalParamExtractor,
    ProbabilisticTensorDictModule as ProbMod,
    ProbabilisticTensorDictSequential as ProbSeq,
    TensorDictModule as Mod,
    TensorDictSequential as Seq,
)
from torch import nn

from torchrl._utils import rl_warnings
from torchrl.data import Bounded, LazyTensorStorage, TensorDictPrioritizedReplayBuffer
from torchrl.data.postprocs.postprocs import MultiStep
from torchrl.modules.distributions.continuous import TanhNormal
from torchrl.modules.models.models import MLP
from torchrl.modules.tensordict_module.actors import Actor, ValueOperator
from torchrl.objectives import DDPGLoss, TD3BCLoss, TD3Loss
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
class TestDDPG(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = nn.Linear(obs_dim, action_dim)
        actor = Actor(
            spec=action_spec,
            module=module,
        )
        return actor.to(device)

    def _create_mock_value(
        self, batch=2, obs_dim=3, action_dim=4, state_dim=8, device="cpu", out_keys=None
    ):
        # Actor
        class ValueClass(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(obs_dim + action_dim + state_dim, 1)

            def forward(self, obs, state, act):
                return self.linear(torch.cat([obs, state, act], -1))

        module = ValueClass()
        value = ValueOperator(
            module=module, in_keys=["observation", "state", "action"], out_keys=out_keys
        )
        return value.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

    def _create_mock_common_layer_setup(
        self, n_obs=3, n_act=4, ncells=4, batch=2, n_hidden=2
    ):
        common = MLP(
            num_cells=ncells,
            in_features=n_obs,
            depth=3,
            out_features=n_hidden,
        )
        actor = MLP(
            num_cells=ncells,
            in_features=n_hidden,
            depth=1,
            out_features=n_act,
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
        actor_head = Mod(actor, in_keys=["hidden"], out_keys=["action"])
        actor = Seq(common, actor_head)
        value_head = Mod(
            value, in_keys=["hidden", "action"], out_keys=["state_action_value"]
        )
        value = Seq(common, value_head)
        value(actor(td))
        return actor, value, common, td

    def _create_mock_data_ddpg(
        self,
        batch=8,
        obs_dim=3,
        action_dim=4,
        state_dim=8,
        atoms=None,
        device="cpu",
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
        state = torch.randn(batch, state_dim, device=device)
        done = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "state": state,
                "next": {
                    "observation": next_obs,
                    "state": state,
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward,
                },
                "action": action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_ddpg(
        self,
        batch=8,
        T=4,
        obs_dim=3,
        action_dim=4,
        state_dim=8,
        atoms=None,
        device="cpu",
        reward_key="reward",
        done_key="done",
        terminated_key="terminated",
    ):
        # create a tensordict
        total_obs = torch.randn(batch, T + 1, obs_dim, device=device)
        total_state = torch.randn(batch, T + 1, state_dim, device=device)
        obs = total_obs[:, :T]
        next_obs = total_obs[:, 1:]
        state = total_state[:, :T]
        next_state = total_state[:, 1:]
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
                "state": state.masked_fill_(~mask.unsqueeze(-1), 0.0),
                "next": {
                    "observation": next_obs.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    "state": next_state.masked_fill_(~mask.unsqueeze(-1), 0.0),
                    done_key: done,
                    terminated_key: terminated,
                    reward_key: reward.masked_fill_(~mask.unsqueeze(-1), 0.0),
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
        value = self._create_mock_value()
        loss_fn = DDPGLoss(actor, value)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("device", get_default_devices())
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
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td), (
            pytest.warns(UserWarning, match="No target network updater has been")
            if (delay_actor or delay_value) and rl_warnings()
            else contextlib.nullcontext()
        ):
            loss = loss_fn(td)

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

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
        sum(
            [item for name, item in loss.items() if name.startswith("loss_")]
        ).backward()
        parameters = list(actor.parameters()) + list(value.parameters())
        for p in parameters:
            assert p.grad.norm() > 0.0

        # Check param update effect on targets
        target_actor = [p.clone() for p in loss_fn.target_actor_network_params.values()]
        target_value = [p.clone() for p in loss_fn.target_value_network_params.values()]
        _i = -1
        for _i, p in enumerate(loss_fn.parameters()):
            if p.requires_grad:
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
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("delay_actor,delay_value", [(False, False), (True, True)])
    def test_ddpg_state_dict(self, delay_actor, delay_value, device):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
            delay_actor=delay_actor,
            delay_value=delay_value,
        )
        state_dict = loss_fn.state_dict()
        loss_fn2 = DDPGLoss(
            actor,
            value,
            loss_function="l2",
            delay_actor=delay_actor,
            delay_value=delay_value,
        )
        loss_fn2.load_state_dict(state_dict)

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_ddpg_separate_losses(
        self,
        device,
        separate_losses,
    ):
        torch.manual_seed(self.seed)
        actor, value, common, td = self._create_mock_common_layer_setup()
        loss_fn = DDPGLoss(
            actor,
            value,
            separate_losses=separate_losses,
        )

        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            loss = loss_fn(td)

        # remove warning
        SoftUpdate(loss_fn, eps=0.5)

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
            common_layers_no = len(list(common.parameters()))
            if k == "loss_actor":
                assert not any(
                    (p.grad is None) or (p.grad == 0).all()
                    for p in loss_fn.actor_network_params.values(True, True)
                )
                if separate_losses:
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.value_network_params.values(True, True)
                    )
                else:
                    common_layers = itertools.islice(
                        loss_fn.value_network_params.values(True, True),
                        common_layers_no,
                    )
                    assert not any(
                        (p.grad is None) or (p.grad == 0).all() for p in common_layers
                    )
                    value_layers = itertools.islice(
                        loss_fn.value_network_params.values(True, True),
                        common_layers_no,
                        None,
                    )
                    assert all(
                        (p.grad is None) or (p.grad == 0).all() for p in value_layers
                    )
            elif k == "loss_value":
                if separate_losses:
                    assert all(
                        (p.grad is None) or (p.grad == 0).all()
                        for p in loss_fn.actor_network_params.values(True, True)
                    )
                else:
                    if separate_losses:
                        assert all(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.actor_network_params.values(True, True)
                        )
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
                        assert not any(
                            (p.grad is None) or (p.grad == 0).all()
                            for p in loss_fn.value_network_params.values(True, True)
                        )
            else:
                raise NotImplementedError(k)
            loss_fn.zero_grad()

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("device", get_default_devices())
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
        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if (delay_value or delay_value) and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

        with torch.no_grad():
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
        parameters = list(actor.parameters()) + list(value.parameters())
        for p in parameters:
            assert p.grad.norm() > 0.0

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_ddpg_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
        )

        default_keys = {
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
            "state_action_value": "state_action_value",
            "priority": "td_error",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
            td_est=td_est,
        )

        value = self._create_mock_value(out_keys=["state_action_value_test"])
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
        )
        key_mapping = {
            "state_action_value": ("value", "state_action_value_test"),
            "reward": ("reward", "reward2"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize(
        "td_est",
        [ValueEstimators.TD0, ValueEstimators.TD1, ValueEstimators.TDLambda, None],
    )
    def test_ddpg_tensordict_run(self, td_est):
        """Test DDPG loss module with non-default tensordict keys."""
        torch.manual_seed(self.seed)
        tensor_keys = {
            "state_action_value": "state_action_value_test",
            "priority": "td_error_test",
            "reward": "reward_test",
            "done": ("done", "test"),
            "terminated": ("terminated", "test"),
        }

        actor = self._create_mock_actor()
        value = self._create_mock_value(out_keys=[tensor_keys["state_action_value"]])
        td = self._create_mock_data_ddpg(
            reward_key="reward_test",
            done_key=("done", "test"),
            terminated_key=("terminated", "test"),
        )
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
        )
        loss_fn.set_keys(**tensor_keys)

        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td), pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            _ = loss_fn(td)

    def test_ddpg_notensordict(self):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        td = self._create_mock_data_ddpg()
        loss = DDPGLoss(actor, value)

        kwargs = {
            "observation": td.get("observation"),
            "next_reward": td.get(("next", "reward")),
            "next_done": td.get(("next", "done")),
            "next_terminated": td.get(("next", "terminated")),
            "next_observation": td.get(("next", "observation")),
            "action": td.get("action"),
            "state": td.get("state"),
            "next_state": td.get(("next", "state")),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            loss_val_td = loss(td)
            loss_val = loss(**kwargs)
            for i, key in enumerate(loss.out_keys):
                torch.testing.assert_close(loss_val_td.get(key), loss_val[i])
            # test select
            loss.select_out_keys("loss_actor", "target_value")
            if torch.__version__ >= "2.0.0":
                loss_actor, target_value = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, target_value = loss(**kwargs)
                return
            assert loss_actor == loss_val_td["loss_actor"]
            assert (target_value == loss_val_td["target_value"]).all()

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_ddpg_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_ddpg(device=device)
        loss_fn = DDPGLoss(
            actor,
            value,
            loss_function="l2",
            delay_actor=False,
            delay_value=False,
            reduction=reduction,
        )
        loss_fn.make_value_estimator()
        loss = loss_fn(td)
        if reduction == "none":
            for key in loss.keys():
                if key.startswith("loss"):
                    assert loss[key].shape == td.shape
        else:
            for key in loss.keys():
                if not key.startswith("loss_"):
                    continue
                assert loss[key].shape == torch.Size([])

    @pytest.mark.xfail(
        pkg_version.parse(torch.__version__) < pkg_version.parse("2.2"),
        reason="Flaky numeric tolerance on PyTorch < 2.2",
    )
    def test_ddpg_prioritized_weights(self):
        """Test DDPG with prioritized replay buffer weighted loss reduction."""
        n_obs = 4
        n_act = 2
        batch_size = 32
        buffer_size = 100

        # Actor network
        actor_net = MLP(in_features=n_obs, out_features=n_act, num_cells=[64, 64])
        actor = ValueOperator(
            module=actor_net,
            in_keys=["observation"],
            out_keys=["action"],
        )

        # Q-value network
        qvalue_net = MLP(in_features=n_obs + n_act, out_features=1, num_cells=[64, 64])
        qvalue = ValueOperator(module=qvalue_net, in_keys=["observation", "action"])

        # Create DDPG loss
        loss_fn = DDPGLoss(actor_network=actor, value_network=qvalue)
        softupdate = SoftUpdate(loss_fn, eps=0.5)
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
        assert torch.isfinite(loss_out2["loss_value"])

        # Verify weighted vs unweighted differ
        loss_fn_no_weights = DDPGLoss(
            actor_network=actor,
            value_network=qvalue,
            use_prioritized_weights=False,
        )
        softupdate = SoftUpdate(loss_fn_no_weights, eps=0.5)
        loss_fn_no_weights.make_value_estimator()
        loss_fn_no_weights.value_network_params = loss_fn.value_network_params
        loss_fn_no_weights.target_value_network_params = (
            loss_fn.target_value_network_params
        )
        loss_fn_no_weights.actor_network_params = loss_fn.actor_network_params
        loss_fn_no_weights.target_actor_network_params = (
            loss_fn.target_actor_network_params
        )

        loss_out_no_weights = loss_fn_no_weights(sample2)
        # Weighted and unweighted should differ (in general)
        assert torch.isfinite(loss_out_no_weights["loss_value"])


@pytest.mark.skipif(
    not _has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestTD3(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        in_keys=None,
        out_keys=None,
        dropout=0.0,
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = nn.Sequential(
            nn.Linear(obs_dim, obs_dim),
            nn.Dropout(dropout),
            nn.Linear(obs_dim, action_dim),
        )
        actor = Actor(
            spec=action_spec, module=module, in_keys=in_keys, out_keys=out_keys
        )
        return actor.to(device)

    def _create_mock_value(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        out_keys=None,
        action_key="action",
        observation_key="observation",
    ):
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
            in_keys=[observation_key, action_key],
            out_keys=out_keys,
        )
        return value.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

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
        return actor, value, common, td

    def _create_mock_data_td3(
        self,
        batch=8,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        action_key="action",
        observation_key="observation",
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
        terminated = torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next": {
                    "observation": next_obs * mask.to(obs.dtype),
                    "reward": reward * mask.to(obs.dtype),
                    "done": done,
                    "terminated": terminated,
                },
                "collector": {"mask": mask},
                "action": action * mask.to(obs.dtype),
            },
            names=[None, "time"],
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = TD3Loss(
            actor,
            value,
            bounds=(-1, 1),
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "delay_actor, delay_qvalue", [(False, False), (True, True)]
    )
    @pytest.mark.parametrize("policy_noise", [0.1, 1.0])
    @pytest.mark.parametrize("noise_clip", [0.1, 1.0])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    @pytest.mark.parametrize("use_action_spec", [True, False])
    @pytest.mark.parametrize("dropout", [0.0, 0.1])
    def test_td3(
        self,
        delay_actor,
        delay_qvalue,
        device,
        policy_noise,
        noise_clip,
        td_est,
        use_action_spec,
        dropout,
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device, dropout=dropout)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_td3(device=device)
        if use_action_spec:
            action_spec = actor.spec
            bounds = None
        else:
            bounds = (-1, 1)
            action_spec = None
        loss_fn = TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            delay_actor=delay_actor,
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
            if (delay_actor or delay_qvalue) and rl_warnings()
            else contextlib.nullcontext()
        ):
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

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("delay_actor, delay_qvalue", [(True, True)])
    @pytest.mark.parametrize("policy_noise", [0.1])
    @pytest.mark.parametrize("noise_clip", [0.1])
    @pytest.mark.parametrize("td_est", [None])
    @pytest.mark.parametrize("use_action_spec", [True])
    @pytest.mark.parametrize("dropout", [0.0])
    def test_td3_deactivate_vmap(
        self,
        delay_actor,
        delay_qvalue,
        device,
        policy_noise,
        noise_clip,
        td_est,
        use_action_spec,
        dropout,
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device, dropout=dropout)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_td3(device=device)
        if use_action_spec:
            action_spec = actor.spec
            bounds = None
        else:
            bounds = (-1, 1)
            action_spec = None
        torch.manual_seed(0)
        loss_fn_vmap = TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_vmap.make_value_estimator(td_est)
        tdc = td.clone()
        with (
            pytest.warns(
                UserWarning,
                match="No target network updater has been associated with this loss module",
            )
            if (delay_actor or delay_qvalue) and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(td):
            torch.manual_seed(1)
            loss_vmap = loss_fn_vmap(td)
        td = tdc
        torch.manual_seed(0)
        loss_fn_no_vmap = TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        if td_est in (ValueEstimators.GAE, ValueEstimators.VTrace):
            with pytest.raises(NotImplementedError):
                loss_fn_no_vmap.make_value_estimator(td_est)
            return
        if td_est is not None:
            loss_fn_no_vmap.make_value_estimator(td_est)
        with (
            pytest.warns(
                UserWarning,
                match="No target network updater has been associated with this loss module",
            )
            if (delay_actor or delay_qvalue) and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(td):
            torch.manual_seed(1)
            loss_no_vmap = loss_fn_no_vmap(td)
        assert_allclose_td(loss_vmap, loss_no_vmap)

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "delay_actor, delay_qvalue", [(False, False), (True, True)]
    )
    @pytest.mark.parametrize("policy_noise", [0.1])
    @pytest.mark.parametrize("noise_clip", [0.1])
    @pytest.mark.parametrize("use_action_spec", [True, False])
    def test_td3_state_dict(
        self,
        delay_actor,
        delay_qvalue,
        device,
        policy_noise,
        noise_clip,
        use_action_spec,
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        if use_action_spec:
            action_spec = actor.spec
            bounds = None
        else:
            bounds = (-1, 1)
            action_spec = None
        loss_fn = TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_td3_separate_losses(
        self,
        device,
        separate_losses,
        n_act=4,
    ):
        torch.manual_seed(self.seed)
        actor, value, common, td = self._create_mock_common_layer_setup(n_act=n_act)
        loss_fn = TD3Loss(
            actor,
            value,
            action_spec=Bounded(shape=(n_act,), low=-1, high=1),
            loss_function="l2",
            separate_losses=separate_losses,
        )
        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
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
                            for p in loss_fn.qvalue_network_params.values(True, True)
                        )

                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("device", get_default_devices())
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
            action_spec=actor.spec,
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

        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if (delay_qvalue or delay_actor) and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        if delay_qvalue or delay_actor:
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
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_td3_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = TD3Loss(
            actor,
            value,
            action_spec=actor.spec,
        )

        default_keys = {
            "priority": "td_error",
            "state_action_value": "state_action_value",
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

        value = self._create_mock_value(out_keys=["state_action_value_test"])
        loss_fn = TD3Loss(
            actor,
            value,
            action_spec=actor.spec,
        )
        key_mapping = {
            "state_action_value": ("value", "state_action_value_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("spec", [True, False])
    @pytest.mark.parametrize("bounds", [True, False])
    def test_constructor(self, spec, bounds):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        action_spec = actor.spec if spec else None
        bounds = (-1, 1) if bounds else None
        if (bounds is not None and action_spec is not None) or (
            bounds is None and action_spec is None
        ):
            with pytest.raises(ValueError, match="but not both"):
                TD3Loss(
                    actor,
                    value,
                    action_spec=action_spec,
                    bounds=bounds,
                )
            return
        TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
        )

    # TODO: test for action_key, atm the action key of the TD3 loss is not configurable,
    # since it is used in it's constructor
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_td3_notensordict(
        self, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(in_keys=[observation_key])
        qvalue = self._create_mock_value(
            observation_key=observation_key, out_keys=["state_action_value"]
        )
        td = self._create_mock_data_td3(
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )
        loss = TD3Loss(actor, qvalue, action_spec=actor.spec)
        loss.set_keys(reward=reward_key, done=done_key, terminated=terminated_key)

        kwargs = {
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
            "action": td.get("action"),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            torch.manual_seed(0)
            loss_val_td = loss(td)
            torch.manual_seed(0)
            loss_val = loss(**kwargs)
            loss_val_reconstruct = TensorDict(dict(zip(loss.out_keys, loss_val)), [])
            assert_allclose_td(loss_val_reconstruct, loss_val_td)

            # test select
            loss.select_out_keys("loss_actor", "loss_qvalue")
            torch.manual_seed(0)
            if torch.__version__ >= "2.0.0":
                loss_actor, loss_qvalue = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, loss_qvalue = loss(**kwargs)
                return

            assert loss_actor == loss_val_td["loss_actor"]
            assert loss_qvalue == loss_val_td["loss_qvalue"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_td3_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_td3(device=device)
        action_spec = actor.spec
        bounds = None
        loss_fn = TD3Loss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            delay_qvalue=False,
            delay_actor=False,
            reduction=reduction,
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

    def test_td3_prioritized_weights(self):
        """Test TD3 with prioritized replay buffer weighted loss reduction."""
        n_obs = 4
        n_act = 2
        batch_size = 32
        buffer_size = 100

        # Actor network
        actor_net = MLP(in_features=n_obs, out_features=n_act, num_cells=[64, 64])
        actor = ValueOperator(
            module=actor_net,
            in_keys=["observation"],
            out_keys=["action"],
        )

        # Q-value network
        qvalue_net = MLP(in_features=n_obs + n_act, out_features=1, num_cells=[64, 64])
        qvalue = ValueOperator(module=qvalue_net, in_keys=["observation", "action"])

        # Create TD3 loss
        loss_fn = TD3Loss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=2,
            action_spec=Bounded(
                low=-torch.ones(n_act), high=torch.ones(n_act), shape=(n_act,)
            ),
        )
        softupdate = SoftUpdate(loss_fn, eps=0.5)
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
        loss_fn_no_weights = TD3Loss(
            actor_network=actor,
            qvalue_network=qvalue,
            num_qvalue_nets=2,
            action_spec=Bounded(
                low=-torch.ones(n_act), high=torch.ones(n_act), shape=(n_act,)
            ),
            use_prioritized_weights=False,
        )
        softupdate = SoftUpdate(loss_fn_no_weights, eps=0.5)
        loss_fn_no_weights.make_value_estimator()
        loss_fn_no_weights.qvalue_network_params = loss_fn.qvalue_network_params
        loss_fn_no_weights.target_qvalue_network_params = (
            loss_fn.target_qvalue_network_params
        )
        loss_fn_no_weights.actor_network_params = loss_fn.actor_network_params
        loss_fn_no_weights.target_actor_network_params = (
            loss_fn.target_actor_network_params
        )

        loss_out_no_weights = loss_fn_no_weights(sample2)
        # Weighted and unweighted should differ (in general)
        assert torch.isfinite(loss_out_no_weights["loss_qvalue"])


class TestTD3BC(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        in_keys=None,
        out_keys=None,
        dropout=0.0,
    ):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        module = nn.Sequential(
            nn.Linear(obs_dim, obs_dim),
            nn.Dropout(dropout),
            nn.Linear(obs_dim, action_dim),
        )
        actor = Actor(
            spec=action_spec, module=module, in_keys=in_keys, out_keys=out_keys
        )
        return actor.to(device)

    def _create_mock_value(
        self,
        batch=2,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        out_keys=None,
        action_key="action",
        observation_key="observation",
    ):
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
            in_keys=[observation_key, action_key],
            out_keys=out_keys,
        )
        return value.to(device)

    def _create_mock_distributional_actor(
        self, batch=2, obs_dim=3, action_dim=4, atoms=5, vmin=1, vmax=5
    ):
        raise NotImplementedError

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
        return actor, value, common, td

    def _create_mock_data_td3bc(
        self,
        batch=8,
        obs_dim=3,
        action_dim=4,
        atoms=None,
        device="cpu",
        action_key="action",
        observation_key="observation",
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

    def _create_seq_mock_data_td3bc(
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
        mask = ~torch.zeros(batch, T, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs * mask.to(obs.dtype),
                "next": {
                    "observation": next_obs * mask.to(obs.dtype),
                    "reward": reward * mask.to(obs.dtype),
                    "done": done,
                    "terminated": terminated,
                },
                "collector": {"mask": mask},
                "action": action * mask.to(obs.dtype),
            },
            names=[None, "time"],
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = TD3BCLoss(
            actor,
            value,
            bounds=(-1, 1),
        )
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "delay_actor, delay_qvalue", [(False, False), (True, True)]
    )
    @pytest.mark.parametrize("policy_noise", [0.1, 1.0])
    @pytest.mark.parametrize("noise_clip", [0.1, 1.0])
    @pytest.mark.parametrize("alpha", [0.1, 6.0])
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    @pytest.mark.parametrize("use_action_spec", [True, False])
    @pytest.mark.parametrize("dropout", [0.0, 0.1])
    def test_td3bc(
        self,
        delay_actor,
        delay_qvalue,
        device,
        policy_noise,
        noise_clip,
        alpha,
        td_est,
        use_action_spec,
        dropout,
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device, dropout=dropout)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_td3bc(device=device)
        if use_action_spec:
            action_spec = actor.spec
            bounds = None
        else:
            bounds = (-1, 1)
            action_spec = None
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            alpha=alpha,
            delay_actor=delay_actor,
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
            if (delay_actor or delay_qvalue) and rl_warnings()
            else contextlib.nullcontext()
        ):
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

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize(
        "delay_actor, delay_qvalue", [(False, False), (True, True)]
    )
    @pytest.mark.parametrize("policy_noise", [0.1])
    @pytest.mark.parametrize("noise_clip", [0.1])
    @pytest.mark.parametrize("alpha", [0.1])
    @pytest.mark.parametrize("use_action_spec", [True, False])
    def test_td3bc_state_dict(
        self,
        delay_actor,
        delay_qvalue,
        device,
        policy_noise,
        noise_clip,
        alpha,
        use_action_spec,
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        if use_action_spec:
            action_spec = actor.spec
            bounds = None
        else:
            bounds = (-1, 1)
            action_spec = None
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            alpha=alpha,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        sd = loss_fn.state_dict()
        loss_fn2 = TD3BCLoss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            alpha=alpha,
            delay_actor=delay_actor,
            delay_qvalue=delay_qvalue,
        )
        loss_fn2.load_state_dict(sd)

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("separate_losses", [False, True])
    def test_td3bc_separate_losses(
        self,
        device,
        separate_losses,
        n_act=4,
    ):
        torch.manual_seed(self.seed)
        actor, value, common, td = self._create_mock_common_layer_setup(n_act=n_act)
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=Bounded(shape=(n_act,), low=-1, high=1),
            loss_function="l2",
            separate_losses=separate_losses,
        )
        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
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
                            for p in loss_fn.qvalue_network_params.values(True, True)
                        )

                else:
                    raise NotImplementedError(k)
                loss_fn.zero_grad()

    @pytest.mark.skipif(not _has_functorch, reason="functorch not installed")
    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("delay_actor,delay_qvalue", [(False, False), (True, True)])
    @pytest.mark.parametrize("policy_noise", [0.1, 1.0])
    @pytest.mark.parametrize("noise_clip", [0.1, 1.0])
    @pytest.mark.parametrize("alpha", [0.1, 6.0])
    def test_td3bc_batcher(
        self,
        n,
        delay_actor,
        delay_qvalue,
        device,
        policy_noise,
        noise_clip,
        alpha,
        gamma=0.9,
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_seq_mock_data_td3bc(device=device)
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=actor.spec,
            policy_noise=policy_noise,
            noise_clip=noise_clip,
            alpha=alpha,
            delay_qvalue=delay_qvalue,
            delay_actor=delay_actor,
        )

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)

        td_clone = td.clone()
        ms_td = ms(td_clone)

        torch.manual_seed(0)
        np.random.seed(0)

        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if (delay_qvalue or delay_actor) and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        if delay_qvalue or delay_actor:
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
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_td3bc_tensordict_keys(self, td_est):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=actor.spec,
        )

        default_keys = {
            "priority": "td_error",
            "state_action_value": "state_action_value",
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

        value = self._create_mock_value(out_keys=["state_action_value_test"])
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=actor.spec,
        )
        key_mapping = {
            "state_action_value": ("value", "state_action_value_test"),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("spec", [True, False])
    @pytest.mark.parametrize("bounds", [True, False])
    def test_constructor(self, spec, bounds):
        actor = self._create_mock_actor()
        value = self._create_mock_value()
        action_spec = actor.spec if spec else None
        bounds = (-1, 1) if bounds else None
        if (bounds is not None and action_spec is not None) or (
            bounds is None and action_spec is None
        ):
            with pytest.raises(ValueError, match="but not both"):
                TD3BCLoss(
                    actor,
                    value,
                    action_spec=action_spec,
                    bounds=bounds,
                )
            return
        TD3BCLoss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
        )

    # TODO: test for action_key, atm the action key of the TD3+BC loss is not configurable,
    # since it is used in it's constructor
    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_td3bc_notensordict(
        self, observation_key, reward_key, done_key, terminated_key
    ):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(in_keys=[observation_key])
        qvalue = self._create_mock_value(
            observation_key=observation_key, out_keys=["state_action_value"]
        )
        td = self._create_mock_data_td3bc(
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )
        loss = TD3BCLoss(actor, qvalue, action_spec=actor.spec)
        loss.set_keys(reward=reward_key, done=done_key, terminated=terminated_key)

        kwargs = {
            observation_key: td.get(observation_key),
            f"next_{reward_key}": td.get(("next", reward_key)),
            f"next_{done_key}": td.get(("next", done_key)),
            f"next_{terminated_key}": td.get(("next", terminated_key)),
            f"next_{observation_key}": td.get(("next", observation_key)),
            "action": td.get("action"),
        }
        td = TensorDict(kwargs, td.batch_size).unflatten_keys("_")

        with pytest.warns(
            UserWarning, match="No target network updater has been"
        ) if rl_warnings() else contextlib.nullcontext():
            torch.manual_seed(0)
            loss_val_td = loss(td)
            torch.manual_seed(0)
            loss_val = loss(**kwargs)
            loss_val_reconstruct = TensorDict(dict(zip(loss.out_keys, loss_val)), [])
            assert_allclose_td(loss_val_reconstruct, loss_val_td)

            # test select
            loss.select_out_keys("loss_actor", "loss_qvalue")
            torch.manual_seed(0)
            if torch.__version__ >= "2.0.0":
                loss_actor, loss_qvalue = loss(**kwargs)
            else:
                with pytest.raises(
                    RuntimeError,
                    match="You are likely using tensordict.nn.dispatch with keyword arguments",
                ):
                    loss_actor, loss_qvalue = loss(**kwargs)
                return

            assert loss_actor == loss_val_td["loss_actor"]
            assert loss_qvalue == loss_val_td["loss_qvalue"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_td3bc_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        actor = self._create_mock_actor(device=device)
        value = self._create_mock_value(device=device)
        td = self._create_mock_data_td3bc(device=device)
        action_spec = actor.spec
        bounds = None
        loss_fn = TD3BCLoss(
            actor,
            value,
            action_spec=action_spec,
            bounds=bounds,
            loss_function="l2",
            delay_qvalue=False,
            delay_actor=False,
            reduction=reduction,
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
