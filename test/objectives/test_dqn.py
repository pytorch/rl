# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import contextlib

import pytest
import torch
from _objectives_common import _check_td_steady, get_devices, LossModuleTestBase

from tensordict import assert_allclose_td, TensorDict
from tensordict.nn import TensorDictModule
from tensordict.utils import unravel_key
from torch import nn

from torchrl._utils import rl_warnings
from torchrl.data import (
    Categorical,
    Composite,
    LazyTensorStorage,
    MultiOneHot,
    OneHot,
    TensorDictPrioritizedReplayBuffer,
)
from torchrl.data.postprocs.postprocs import MultiStep
from torchrl.modules import DistributionalQValueActor, QValueActor, SafeSequential
from torchrl.modules.models import QMixer
from torchrl.modules.models.models import MLP
from torchrl.modules.tensordict_module.actors import QValueModule
from torchrl.objectives import DistributionalDQNLoss, DQNLoss, QMixerLoss
from torchrl.objectives.utils import SoftUpdate, ValueEstimators

from torchrl.testing import (  # noqa
    call_value_nets as _call_value_nets,
    dtype_fixture,
    get_available_devices,
    get_default_devices,
    PENDULUM_VERSIONED,
)


class TestDQN(LossModuleTestBase):
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
        # elif action_spec_type == "nd_bounded":
        #     action_spec = Bounded(
        #         -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        #     )
        else:
            raise ValueError(f"Wrong {action_spec_type}")

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
        action_value_key="action_value",
    ):
        # Actor
        var_nums = None
        if action_spec_type == "mult_one_hot":
            action_spec = MultiOneHot([action_dim // 2, action_dim // 2])
            var_nums = action_spec.nvec
        elif action_spec_type == "one_hot":
            action_spec = OneHot(action_dim)
        elif action_spec_type == "categorical":
            action_spec = Categorical(action_dim)
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
            spec=Composite(
                {
                    "action": action_spec,
                    action_value_key: None,
                },
                shape=[],
            ),
            module=module,
            support=support,
            action_space=action_spec_type,
            var_nums=var_nums,
            action_value_key=action_value_key,
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
        action_key="action",
        action_value_key="action_value",
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
        actor = self._create_mock_actor(action_spec_type="one_hot")
        loss_fn = DQNLoss(actor)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize(
        "delay_value,double_dqn", ([False, False], [True, False], [True, True])
    )
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_dqn(self, delay_value, double_dqn, device, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DQNLoss(
            actor,
            loss_function="l2",
            delay_value=delay_value,
            double_dqn=double_dqn,
        )
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

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

        assert loss_fn.tensor_keys.priority in td.keys()

        sum([item for name, item in loss.items() if name.startswith("loss")]).backward()
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
    def test_dqn_state_dict(self, delay_value, device, action_spec_type):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = DQNLoss(actor, loss_function="l2", delay_value=delay_value)
        sd = loss_fn.state_dict()
        loss_fn2 = DQNLoss(actor, loss_function="l2", delay_value=delay_value)
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
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

        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)
        assert loss_fn.tensor_keys.priority in ms_td.keys()

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

        with torch.no_grad():
            loss = loss_fn(td)
        if n == 1:
            assert_allclose_td(td, ms_td.select(*td.keys(True, True)))
            _loss = sum(
                [item for name, item in loss.items() if name.startswith("loss")]
            )
            _loss_ms = sum(
                [item for name, item in loss_ms.items() if name.startswith("loss")]
            )
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum(
            [item for name, item in loss_ms.items() if name.startswith("loss")]
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
    def test_dqn_tensordict_keys(self, td_est):
        torch.manual_seed(self.seed)
        action_spec_type = "one_hot"
        actor = self._create_mock_actor(action_spec_type=action_spec_type)
        loss_fn = DQNLoss(actor, delay_value=True)

        default_keys = {
            "advantage": "advantage",
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

        loss_fn = DQNLoss(actor, delay_value=True)
        key_mapping = {
            "advantage": ("advantage", "advantage_2"),
            "value_target": ("value_target", ("value_target", "nested")),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, action_value_key="chosen_action_value_2"
        )
        loss_fn = DQNLoss(actor, delay_value=True)
        key_mapping = {
            "value": ("value", "chosen_action_value_2"),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_spec_type", ("categorical", "one_hot"))
    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_dqn_tensordict_run(self, action_spec_type, td_est):
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
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type,
            action_key=tensor_keys["action"],
            action_value_key=tensor_keys["action_value"],
        )

        loss_fn = DQNLoss(actor, loss_function="l2", delay_value=True)
        loss_fn.set_keys(**tensor_keys)

        if td_est is not None:
            loss_fn.make_value_estimator(td_est)

        SoftUpdate(loss_fn, eps=0.5)

        with _check_td_steady(td):
            _ = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys()

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
        loss_fn = DistributionalDQNLoss(
            actor,
            gamma=gamma,
            delay_value=delay_value,
        )

        if td_est not in (None, ValueEstimators.TD0):
            with pytest.raises(NotImplementedError):
                loss_fn.make_value_estimator(td_est)
            return
        elif td_est is not None:
            loss_fn.make_value_estimator(td_est)

        with _check_td_steady(td), (
            pytest.warns(
                UserWarning,
                match="No target network updater has been associated with this loss module",
            )
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ):
            loss = loss_fn(td)

        assert loss_fn.tensor_keys.priority in td.keys()

        sum([item for name, item in loss.items() if name.startswith("loss")]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

        # Check param update effect on targets
        target_value = loss_fn.target_value_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
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

    @pytest.mark.parametrize("observation_key", ["observation", "observation2"])
    @pytest.mark.parametrize("reward_key", ["reward", "reward2"])
    @pytest.mark.parametrize("done_key", ["done", "done2"])
    @pytest.mark.parametrize("terminated_key", ["terminated", "terminated2"])
    def test_dqn_notensordict(
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
        dqn_loss = DQNLoss(actor, delay_value=True)
        dqn_loss.set_keys(reward=reward_key, done=done_key, terminated=terminated_key)
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
        # Disable warning
        SoftUpdate(dqn_loss, eps=0.5)
        loss_val = dqn_loss(**kwargs)
        loss_val_td = dqn_loss(td)
        torch.testing.assert_close(loss_val_td.get("loss"), loss_val)

    def test_distributional_dqn_tensordict_keys(self):
        torch.manual_seed(self.seed)
        action_spec_type = "one_hot"
        atoms = 2
        gamma = 0.9
        actor = self._create_mock_distributional_actor(
            action_spec_type=action_spec_type, atoms=atoms
        )

        loss_fn = DistributionalDQNLoss(actor, gamma=gamma, delay_value=True)

        default_keys = {
            "priority": "td_error",
            "action_value": "action_value",
            "action": "action",
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
            "steps_to_next_obs": "steps_to_next_obs",
        }

        self.tensordict_keys_test(loss_fn, default_keys=default_keys)

    @pytest.mark.parametrize("action_spec_type", ("categorical", "one_hot"))
    @pytest.mark.parametrize("td_est", [ValueEstimators.TD0])
    def test_distributional_dqn_tensordict_run(self, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        atoms = 4
        tensor_keys = {
            "action_value": "action_value_test",
            "action": "action_test",
            "priority": "priority_test",
        }
        actor = self._create_mock_distributional_actor(
            action_spec_type=action_spec_type,
            atoms=atoms,
            action_value_key=tensor_keys["action_value"],
        )
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type,
            atoms=atoms,
            action_key=tensor_keys["action"],
            action_value_key=tensor_keys["action_value"],
        )
        loss_fn = DistributionalDQNLoss(actor, gamma=0.9, delay_value=True)
        loss_fn.set_keys(**tensor_keys)

        loss_fn.make_value_estimator(td_est)

        # remove warnings
        SoftUpdate(loss_fn, eps=0.5)

        with _check_td_steady(td):
            _ = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys()

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_dqn_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        actor = self._create_mock_actor(action_spec_type="categorical", device=device)
        td = self._create_mock_data_dqn(action_spec_type="categorical", device=device)
        loss_fn = DQNLoss(
            actor,
            loss_function="l2",
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
                if not key.startswith("loss"):
                    continue
                assert loss[key].shape == torch.Size([])

    @pytest.mark.parametrize("atoms", range(4, 10))
    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_distributional_dqn_reduction(self, reduction, atoms):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        actor = self._create_mock_distributional_actor(
            action_spec_type="categorical", atoms=atoms
        ).to(device)
        td = self._create_mock_data_dqn(action_spec_type="categorical", device=device)
        loss_fn = DistributionalDQNLoss(
            actor, gamma=0.9, delay_value=False, reduction=reduction
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

    def test_dqn_prioritized_weights(self):
        """Test DQN with prioritized replay buffer weighted loss reduction."""
        n_obs = 4
        n_actions = 3
        batch_size = 32
        buffer_size = 100

        # Create DQN value network using QValueActor
        module = nn.Linear(n_obs, n_actions)
        action_spec = Categorical(n_actions)
        value = QValueActor(
            spec=Composite(
                {
                    "action": action_spec,
                    "action_value": None,
                    "chosen_action_value": None,
                },
                shape=[],
            ),
            action_space="categorical",
            module=module,
        )

        # Create DQN loss
        loss_fn = DQNLoss(
            value_network=value, action_space="categorical", reduction="mean"
        )
        loss_fn.make_value_estimator()
        softupdate = SoftUpdate(loss_fn, eps=0.5)

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
                "action": torch.randint(0, n_actions, (buffer_size,)),
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
        assert torch.isfinite(loss_out2["loss"])

        # Verify manual weighted average matches
        loss_fn_no_reduction = DQNLoss(
            value_network=value,
            action_space="categorical",
            reduction="none",
            use_prioritized_weights=False,
        )
        softupdate = SoftUpdate(loss_fn_no_reduction, eps=0.5)
        loss_fn_no_reduction.make_value_estimator()
        loss_fn_no_reduction.target_value_network_params = (
            loss_fn.target_value_network_params
        )

        loss_elements = loss_fn_no_reduction(sample2)["loss"]
        manual_weighted_loss = (loss_elements * weights2).sum() / weights2.sum()
        assert torch.allclose(loss_out2["loss"], manual_weighted_loss, rtol=1e-4)


class TestQMixer(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(
        self,
        action_spec_type,
        obs_dim=3,
        action_dim=4,
        device="cpu",
        observation_key=("agents", "observation"),
        action_key=("agents", "action"),
        action_value_key=("agents", "action_value"),
        chosen_action_value_key=("agents", "chosen_action_value"),
    ):
        # Actor
        if action_spec_type == "one_hot":
            action_spec = OneHot(action_dim)
        elif action_spec_type == "categorical":
            action_spec = Categorical(action_dim)
        else:
            raise ValueError(f"Wrong {action_spec_type}")

        module = nn.Linear(obs_dim, action_dim).to(device)

        module = TensorDictModule(
            module,
            in_keys=[observation_key],
            out_keys=[action_value_key],
        ).to(device)
        value_module = QValueModule(
            action_value_key=action_value_key,
            out_keys=[
                action_key,
                action_value_key,
                chosen_action_value_key,
            ],
            spec=action_spec,
            action_space=None,
        ).to(device)
        actor = SafeSequential(module, value_module)

        return actor

    def _create_mock_mixer(
        self,
        state_shape=(64, 64, 3),
        n_agents=4,
        device="cpu",
        chosen_action_value_key=("agents", "chosen_action_value"),
        state_key="state",
        global_chosen_action_value_key="chosen_action_value",
    ):
        qmixer = TensorDictModule(
            module=QMixer(
                state_shape=state_shape,
                mixing_embed_dim=32,
                n_agents=n_agents,
                device=device,
            ),
            in_keys=[chosen_action_value_key, state_key],
            out_keys=[global_chosen_action_value_key],
        ).to(device)

        return qmixer

    def _create_mock_data_dqn(
        self,
        action_spec_type,
        batch=(2,),
        T=None,
        n_agents=4,
        obs_dim=3,
        state_shape=(64, 64, 3),
        action_dim=4,
        device="cpu",
        action_key=("agents", "action"),
        action_value_key=("agents", "action_value"),
    ):
        if T is not None:
            batch = batch + (T,)
        # create a tensordict
        obs = torch.randn(*batch, n_agents, obs_dim, device=device)
        state = torch.randn(*batch, *state_shape, device=device)
        next_obs = torch.randn(*batch, n_agents, obs_dim, device=device)
        next_state = torch.randn(*batch, *state_shape, device=device)

        action_value = torch.randn(*batch, n_agents, action_dim, device=device)
        if action_spec_type == "one_hot":
            action = (action_value == action_value.max(dim=-1, keepdim=True)[0]).to(
                torch.long
            )
        elif action_spec_type == "categorical":
            action = torch.argmax(action_value, dim=-1).to(torch.long)

        reward = torch.randn(*batch, 1, device=device)
        done = torch.zeros(*batch, 1, dtype=torch.bool, device=device)
        terminated = torch.zeros(*batch, 1, dtype=torch.bool, device=device)
        td = TensorDict(
            {
                "agents": TensorDict(
                    {"observation": obs},
                    [*batch, n_agents],
                    device=device,
                ),
                "state": state,
                "collector": {
                    "mask": torch.zeros(*batch, dtype=torch.bool, device=device)
                },
                "next": TensorDict(
                    {
                        "agents": TensorDict(
                            {"observation": next_obs},
                            [*batch, n_agents],
                            device=device,
                        ),
                        "state": next_state,
                        "reward": reward,
                        "done": done,
                        "terminated": terminated,
                    },
                    batch_size=batch,
                    device=device,
                ),
            },
            batch_size=batch,
            device=device,
        )
        td.set(action_key, action)
        td.set(action_value_key, action_value)
        if T is not None:
            td.refine_names(None, "time")
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor(action_spec_type="one_hot")
        mixer = self._create_mock_mixer()
        loss_fn = QMixerLoss(actor, mixer)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    @pytest.mark.parametrize("td_est", list(ValueEstimators) + [None])
    def test_qmixer(self, delay_value, device, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        mixer = self._create_mock_mixer(device=device)
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type, device=device
        )
        loss_fn = QMixerLoss(actor, mixer, loss_function="l2", delay_value=delay_value)
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
        assert loss_fn.tensor_keys.priority in td.keys()

        sum([item for name, item in loss.items() if name.startswith("loss")]).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

        # Check param update effect on targets
        target_value = loss_fn.target_local_value_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += 3
        target_value2 = loss_fn.target_local_value_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # Check param update effect on targets
        target_value = loss_fn.target_mixer_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += 3
        target_value2 = loss_fn.target_mixer_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    def test_qmixer_state_dict(self, delay_value, device, action_spec_type):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        mixer = self._create_mock_mixer(device=device)
        loss_fn = QMixerLoss(actor, mixer, loss_function="l2", delay_value=delay_value)
        sd = loss_fn.state_dict()
        loss_fn2 = QMixerLoss(actor, mixer, loss_function="l2", delay_value=delay_value)
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("n", range(1, 4))
    @pytest.mark.parametrize("delay_value", (False, True))
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("action_spec_type", ("one_hot", "categorical"))
    def test_qmix_batcher(self, n, delay_value, device, action_spec_type, gamma=0.9):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type, device=device
        )
        mixer = self._create_mock_mixer(device=device)
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type, T=4, device=device
        )
        loss_fn = QMixerLoss(actor, mixer, loss_function="l2", delay_value=delay_value)

        ms = MultiStep(gamma=gamma, n_steps=n).to(device)
        ms_td = ms(td.clone())

        with (
            pytest.warns(UserWarning, match="No target network updater has been")
            if delay_value and rl_warnings()
            else contextlib.nullcontext()
        ), _check_td_steady(ms_td):
            loss_ms = loss_fn(ms_td)

        if delay_value:
            # remove warning
            SoftUpdate(loss_fn, eps=0.5)

        assert loss_fn.tensor_keys.priority in ms_td.keys()

        with torch.no_grad():
            loss = loss_fn(td)
        if n == 1:
            assert_allclose_td(td, ms_td.select(*td.keys(True, True)))
            _loss = sum(
                [item for name, item in loss.items() if name.startswith("loss")]
            )
            _loss_ms = sum(
                [item for name, item in loss_ms.items() if name.startswith("loss")]
            )
            assert (
                abs(_loss - _loss_ms) < 1e-3
            ), f"found abs(loss-loss_ms) = {abs(loss - loss_ms):4.5f} for n=0"
        else:
            with pytest.raises(AssertionError):
                assert_allclose_td(loss, loss_ms)
        sum(
            [item for name, item in loss_ms.items() if name.startswith("loss")]
        ).backward()
        assert torch.nn.utils.clip_grad.clip_grad_norm_(actor.parameters(), 1.0) > 0.0

        # Check param update effect on targets
        target_value = loss_fn.target_local_value_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += 3
        target_value2 = loss_fn.target_local_value_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # Check param update effect on targets
        target_value = loss_fn.target_mixer_network_params.clone()
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += 3
        target_value2 = loss_fn.target_mixer_network_params.clone()
        if loss_fn.delay_value:
            assert_allclose_td(target_value, target_value2)
        else:
            assert not (target_value == target_value2).any()

        # check that policy is updated after parameter update
        parameters = [p.clone() for p in actor.parameters()]
        for p in loss_fn.parameters():
            if p.requires_grad:
                p.data += torch.randn_like(p)
        assert all((p1 != p2).all() for p1, p2 in zip(parameters, actor.parameters()))

    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_qmix_tensordict_keys(self, td_est):
        torch.manual_seed(self.seed)
        action_spec_type = "one_hot"
        actor = self._create_mock_actor(action_spec_type=action_spec_type)
        mixer = self._create_mock_mixer()
        loss_fn = QMixerLoss(actor, mixer, delay_value=True)

        default_keys = {
            "advantage": "advantage",
            "value_target": "value_target",
            "local_value": ("agents", "chosen_action_value"),
            "global_value": "chosen_action_value",
            "priority": "td_error",
            "action_value": ("agents", "action_value"),
            "action": ("agents", "action"),
            "reward": "reward",
            "done": "done",
            "terminated": "terminated",
        }

        self.tensordict_keys_test(loss_fn, default_keys=default_keys)

        loss_fn = QMixerLoss(actor, mixer, delay_value=True)
        key_mapping = {
            "advantage": ("advantage", "advantage_2"),
            "value_target": ("value_target", ("value_target", "nested")),
            "reward": ("reward", "reward_test"),
            "done": ("done", ("done", "test")),
            "terminated": ("terminated", ("terminated", "test")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

        actor = self._create_mock_actor(
            action_spec_type=action_spec_type,
        )
        mixer = self._create_mock_mixer(
            global_chosen_action_value_key=("some", "nested")
        )
        loss_fn = QMixerLoss(actor, mixer, delay_value=True)
        key_mapping = {
            "global_value": ("value", ("some", "nested")),
        }
        self.set_advantage_keys_through_loss_test(loss_fn, td_est, key_mapping)

    @pytest.mark.parametrize("action_spec_type", ("categorical", "one_hot"))
    @pytest.mark.parametrize(
        "td_est", [ValueEstimators.TD1, ValueEstimators.TD0, ValueEstimators.TDLambda]
    )
    def test_qmix_tensordict_run(self, action_spec_type, td_est):
        torch.manual_seed(self.seed)
        tensor_keys = {
            "action_value": ("other", "action_value_test"),
            "action": ("other", "action"),
            "local_value": ("some", "local_v"),
            "global_value": "global_v",
            "priority": "priority_test",
        }
        actor = self._create_mock_actor(
            action_spec_type=action_spec_type,
            action_value_key=tensor_keys["action_value"],
            action_key=tensor_keys["action"],
            chosen_action_value_key=tensor_keys["local_value"],
        )
        mixer = self._create_mock_mixer(
            chosen_action_value_key=tensor_keys["local_value"],
            global_chosen_action_value_key=tensor_keys["global_value"],
        )
        td = self._create_mock_data_dqn(
            action_spec_type=action_spec_type,
            action_key=tensor_keys["action"],
            action_value_key=tensor_keys["action_value"],
        )

        loss_fn = QMixerLoss(actor, mixer, loss_function="l2", delay_value=True)
        loss_fn.set_keys(**tensor_keys)
        SoftUpdate(loss_fn, eps=0.5)
        if td_est is not None:
            loss_fn.make_value_estimator(td_est)
        with _check_td_steady(td):
            _ = loss_fn(td)
        assert loss_fn.tensor_keys.priority in td.keys()

    @pytest.mark.parametrize(
        "mixer_local_chosen_action_value_key",
        [("agents", "chosen_action_value"), ("other")],
    )
    @pytest.mark.parametrize(
        "mixer_global_chosen_action_value_key",
        ["chosen_action_value", ("nested", "other")],
    )
    def test_mixer_keys(
        self,
        mixer_local_chosen_action_value_key,
        mixer_global_chosen_action_value_key,
        n_agents=4,
        obs_dim=3,
    ):
        torch.manual_seed(0)
        actor = self._create_mock_actor(
            action_spec_type="categorical",
        )
        mixer = self._create_mock_mixer(
            chosen_action_value_key=mixer_local_chosen_action_value_key,
            global_chosen_action_value_key=mixer_global_chosen_action_value_key,
            n_agents=n_agents,
        )

        td = TensorDict(
            {
                "agents": TensorDict(
                    {"observation": torch.zeros(32, n_agents, obs_dim)}, [32, n_agents]
                ),
                "state": torch.zeros(32, 64, 64, 3),
                "next": TensorDict(
                    {
                        "agents": TensorDict(
                            {"observation": torch.zeros(32, n_agents, obs_dim)},
                            [32, n_agents],
                        ),
                        "state": torch.zeros(32, 64, 64, 3),
                        "reward": torch.zeros(32, 1),
                        "done": torch.zeros(32, 1, dtype=torch.bool),
                        "terminated": torch.zeros(32, 1, dtype=torch.bool),
                    },
                    [32],
                ),
            },
            [32],
        )
        td = actor(td)

        loss = QMixerLoss(actor, mixer, delay_value=True)

        SoftUpdate(loss, eps=0.5)

        # Without setting the keys
        if mixer_local_chosen_action_value_key != ("agents", "chosen_action_value"):
            with pytest.raises(KeyError):
                loss(td)
        elif unravel_key(mixer_global_chosen_action_value_key) != "chosen_action_value":
            with pytest.raises(
                KeyError, match='key "chosen_action_value" not found in TensorDict'
            ):
                loss(td)
        else:
            loss(td)

        loss = QMixerLoss(actor, mixer, delay_value=True)

        SoftUpdate(loss, eps=0.5)

        # When setting the key
        loss.set_keys(global_value=mixer_global_chosen_action_value_key)
        if mixer_local_chosen_action_value_key != ("agents", "chosen_action_value"):
            with pytest.raises(
                KeyError
            ):  # The mixer in key still does not match the actor out_key
                loss(td)
        else:
            loss(td)

    def test_dqn_prioritized_weights(self):
        """Test DQN with prioritized replay buffer weighted loss reduction."""
        n_obs = 4
        n_actions = 3
        batch_size = 32
        buffer_size = 100

        # Create DQN value network using QValueActor
        module = nn.Linear(n_obs, n_actions)
        action_spec = Categorical(n_actions)
        value = QValueActor(
            spec=Composite(
                {
                    "action": action_spec,
                    "action_value": None,
                    "chosen_action_value": None,
                },
                shape=[],
            ),
            action_space="categorical",
            module=module,
        )

        # Create DQN loss
        loss_fn = DQNLoss(
            value_network=value, action_space="categorical", reduction="mean"
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
                "action": torch.randint(0, n_actions, (buffer_size,)),
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
        assert torch.isfinite(loss_out2["loss"])

        # Verify manual weighted average matches
        loss_fn_no_reduction = DQNLoss(
            value_network=value,
            action_space="categorical",
            reduction="none",
            use_prioritized_weights=False,
        )
        softupdate = SoftUpdate(loss_fn_no_reduction, eps=0.5)
        loss_fn_no_reduction.make_value_estimator()
        loss_fn_no_reduction.target_value_network_params = (
            loss_fn.target_value_network_params
        )

        loss_elements = loss_fn_no_reduction(sample2)["loss"]
        manual_weighted_loss = (loss_elements * weights2).sum() / weights2.sum()
        assert torch.allclose(loss_out2["loss"], manual_weighted_loss, rtol=1e-4)
