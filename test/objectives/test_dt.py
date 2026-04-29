# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import pytest
import torch
from _objectives_common import LossModuleTestBase

from tensordict import TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule, TensorDictSequential
from torch import nn

from torchrl.data import Bounded
from torchrl.modules.distributions.continuous import TanhDelta, TanhNormal
from torchrl.modules.models.models import MLP
from torchrl.modules.tensordict_module.actors import ProbabilisticActor
from torchrl.objectives import DTLoss, GAILLoss, OnlineDTLoss

from torchrl.testing import (  # noqa
    call_value_nets as _call_value_nets,
    dtype_fixture,
    get_available_devices,
    get_default_devices,
    PENDULUM_VERSIONED,
)


class TestOnlineDT(LossModuleTestBase):
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
            distribution_class=TanhNormal,
            in_keys=["loc", "scale"],
            spec=action_spec,
        )
        return actor.to(device)

    def _create_mock_data_odt(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward2go = torch.randn(batch, 1, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "action": action,
                "reward2go": reward2go,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_odt(
        self, batch=2, T=4, obs_dim=3, action_dim=4, device="cpu"
    ):
        # create a tensordict
        obs = torch.randn(batch, T, obs_dim, device=device)
        action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)
        reward2go = torch.randn(batch, T, 1, device=device)

        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs,
                "reward": reward2go,
                "action": action,
            },
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        loss_fn = OnlineDTLoss(actor)
        self.reset_parameters_recursive_test(loss_fn)

    @pytest.mark.parametrize("device", get_available_devices())
    def test_odt(self, device):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_odt(device=device)

        actor = self._create_mock_actor(device=device)

        loss_fn = OnlineDTLoss(actor)
        loss = loss_fn(td)
        loss_transformer = sum(
            loss[key]
            for key in loss.keys()
            if key.startswith("loss") and key != "loss_alpha"
        )
        loss_alpha = loss["loss_alpha"]
        loss_transformer.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "alpha" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "alpha" in name
        loss_fn.zero_grad()
        loss_alpha.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" not in name
                assert "alpha" in name
            if p.grad is None:
                assert "actor" in name
                assert "alpha" not in name
        loss_fn.zero_grad()

        sum([loss_transformer, loss_alpha]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("device", get_available_devices())
    def test_odt_state_dict(self, device):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)

        loss_fn = OnlineDTLoss(actor)
        sd = loss_fn.state_dict()
        loss_fn2 = OnlineDTLoss(actor)
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("action_dim", [1, 2, 4, 8])
    def test_odt_target_entropy_auto(self, action_dim):
        """Regression test for target_entropy='auto' should be -dim(A)."""
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(action_dim=action_dim)

        loss_fn = OnlineDTLoss(actor)
        # target_entropy="auto" should compute -action_dim
        assert (
            loss_fn.target_entropy.item() == -action_dim
        ), f"target_entropy should be -{action_dim}, got {loss_fn.target_entropy.item()}"

    @pytest.mark.parametrize("device", get_available_devices())
    def test_seq_odt(self, device):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_odt(device=device)

        actor = self._create_mock_actor(device=device)

        loss_fn = OnlineDTLoss(actor)
        loss = loss_fn(td)
        loss_transformer = sum(
            loss[key]
            for key in loss.keys()
            if key.startswith("loss") and key != "loss_alpha"
        )
        loss_alpha = loss["loss_alpha"]
        loss_transformer.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "alpha" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "alpha" in name
        loss_fn.zero_grad()
        loss_alpha.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()
        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" not in name
                assert "alpha" in name
            if p.grad is None:
                assert "actor" in name
                assert "alpha" not in name
        loss_fn.zero_grad()

        sum([loss_transformer, loss_alpha]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    def test_onlinedt_tensordict_keys(self):
        actor = self._create_mock_actor()
        loss_fn = OnlineDTLoss(actor)

        default_keys = {
            "action_pred": "action",
            "action_target": "action",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
        )

    @pytest.mark.parametrize("device", get_default_devices())
    def test_onlinedt_notensordict(self, device):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        td = self._create_mock_data_odt(device=device)
        loss_fn = OnlineDTLoss(actor)

        in_keys = self._flatten_in_keys(loss_fn.in_keys)
        kwargs = dict(td.flatten_keys("_").select(*in_keys))

        torch.manual_seed(0)
        loss_val_td = loss_fn(td)
        torch.manual_seed(0)
        loss_log_likelihood, loss_entropy, loss_alpha, alpha, entropy = loss_fn(
            **kwargs
        )
        torch.testing.assert_close(
            loss_val_td.get("loss_log_likelihood"), loss_log_likelihood
        )
        torch.testing.assert_close(loss_val_td.get("loss_entropy"), loss_entropy)
        torch.testing.assert_close(loss_val_td.get("loss_alpha"), loss_alpha)
        # test select
        torch.manual_seed(0)
        loss_fn.select_out_keys("loss_entropy")
        if torch.__version__ >= "2.0.0":
            loss_entropy = loss_fn(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_entropy = loss_fn(**kwargs)
            return
        assert loss_entropy == loss_val_td["loss_entropy"]

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_onlinedt_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_odt(device=device)
        actor = self._create_mock_actor(device=device)
        loss_fn = OnlineDTLoss(
            actor,
            reduction=reduction,
            scalar_output_mode="exclude" if reduction == "none" else None,
        )
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


class TestDT(LossModuleTestBase):
    seed = 0

    def _create_mock_actor(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # Actor
        action_spec = Bounded(
            -torch.ones(action_dim), torch.ones(action_dim), (action_dim,)
        )
        net = nn.Linear(obs_dim, action_dim)
        module = TensorDictModule(net, in_keys=["observation"], out_keys=["param"])
        actor = ProbabilisticActor(
            module=module,
            distribution_class=TanhDelta,
            in_keys=["param"],
            spec=action_spec,
        )
        return actor.to(device)

    def _create_mock_data_dt(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        reward2go = torch.randn(batch, 1, device=device)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "action": action,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_dt(
        self, batch=2, T=4, obs_dim=3, action_dim=4, device="cpu"
    ):
        # create a tensordict
        obs = torch.randn(batch, T, obs_dim, device=device)
        action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)

        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs,
                "action": action,
            },
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        actor = self._create_mock_actor()
        loss_fn = DTLoss(actor)
        self.reset_parameters_recursive_test(loss_fn)

    def test_dt_tensordict_keys(self):
        actor = self._create_mock_actor()
        loss_fn = DTLoss(actor)

        default_keys = {
            "action_target": "action",
            "action_pred": "action",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
        )

    @pytest.mark.parametrize("device", get_default_devices())
    def test_dt_notensordict(self, device):
        torch.manual_seed(self.seed)
        actor = self._create_mock_actor(device=device)
        td = self._create_mock_data_dt(device=device)
        loss_fn = DTLoss(actor)

        in_keys = self._flatten_in_keys(loss_fn.in_keys)
        kwargs = dict(td.flatten_keys("_").select(*in_keys))

        loss_val_td = loss_fn(td)
        loss_val = loss_fn(**kwargs)
        torch.testing.assert_close(loss_val_td.get("loss"), loss_val)
        # test select
        loss_fn.select_out_keys("loss")
        if torch.__version__ >= "2.0.0":
            loss_actor = loss_fn(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_actor = loss_fn(**kwargs)
            return
        assert loss_actor == loss_val_td["loss"]

    @pytest.mark.parametrize("device", get_available_devices())
    def test_dt(self, device):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_dt(device=device)

        actor = self._create_mock_actor(device=device)

        loss_fn = DTLoss(actor)
        loss = loss_fn(td)
        loss_transformer = loss["loss"]
        loss_transformer.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "alpha" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "alpha" in name
        loss_fn.zero_grad()

        sum([loss_transformer]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("device", get_available_devices())
    def test_dt_state_dict(self, device):
        torch.manual_seed(self.seed)

        actor = self._create_mock_actor(device=device)

        loss_fn = DTLoss(actor)
        sd = loss_fn.state_dict()
        loss_fn2 = DTLoss(actor)
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("device", get_available_devices())
    def test_seq_dt(self, device):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_dt(device=device)

        actor = self._create_mock_actor(device=device)

        loss_fn = DTLoss(actor)
        loss = loss_fn(td)
        loss_transformer = loss["loss"]
        loss_transformer.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "actor" in name
                assert "alpha" not in name
            if p.grad is None:
                assert "actor" not in name
                assert "alpha" in name
        loss_fn.zero_grad()

        sum([loss_transformer]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    def test_dt_reduction(self, reduction):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_dt(device=device)
        actor = self._create_mock_actor(device=device)
        loss_fn = DTLoss(actor, reduction=reduction)
        loss = loss_fn(td)
        if reduction == "none":
            assert loss["loss"].shape == td["action"].shape
        else:
            assert loss["loss"].shape == torch.Size([])


class TestGAIL(LossModuleTestBase):
    seed = 0

    def _create_mock_discriminator(
        self, batch=2, obs_dim=3, action_dim=4, device="cpu"
    ):
        # Discriminator
        body = TensorDictModule(
            MLP(
                in_features=obs_dim + action_dim,
                out_features=32,
                depth=1,
                num_cells=32,
                activation_class=torch.nn.ReLU,
                activate_last_layer=True,
            ),
            in_keys=["observation", "action"],
            out_keys="hidden",
        )
        head = TensorDictModule(
            MLP(
                in_features=32,
                out_features=1,
                depth=0,
                num_cells=32,
                activation_class=torch.nn.Sigmoid,
                activate_last_layer=True,
            ),
            in_keys="hidden",
            out_keys="d_logits",
        )
        discriminator = TensorDictSequential(body, head)

        return discriminator.to(device)

    def _create_mock_data_gail(self, batch=2, obs_dim=3, action_dim=4, device="cpu"):
        # create a tensordict
        obs = torch.randn(batch, obs_dim, device=device)
        action = torch.randn(batch, action_dim, device=device).clamp(-1, 1)
        td = TensorDict(
            batch_size=(batch,),
            source={
                "observation": obs,
                "action": action,
                "collector_action": action,
                "collector_observation": obs,
            },
            device=device,
        )
        return td

    def _create_seq_mock_data_gail(
        self, batch=2, T=4, obs_dim=3, action_dim=4, device="cpu"
    ):
        # create a tensordict
        obs = torch.randn(batch, T, obs_dim, device=device)
        action = torch.randn(batch, T, action_dim, device=device).clamp(-1, 1)

        td = TensorDict(
            batch_size=(batch, T),
            source={
                "observation": obs,
                "action": action,
                "collector_action": action,
                "collector_observation": obs,
            },
            device=device,
        )
        return td

    def test_reset_parameters_recursive(self):
        discriminator = self._create_mock_discriminator()
        loss_fn = GAILLoss(discriminator)
        self.reset_parameters_recursive_test(loss_fn)

    def test_gail_tensordict_keys(self):
        discriminator = self._create_mock_discriminator()
        loss_fn = GAILLoss(discriminator)

        default_keys = {
            "expert_action": "action",
            "expert_observation": "observation",
            "collector_action": "collector_action",
            "collector_observation": "collector_observation",
            "discriminator_pred": "d_logits",
        }

        self.tensordict_keys_test(
            loss_fn,
            default_keys=default_keys,
        )

    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("use_grad_penalty", [True, False])
    @pytest.mark.parametrize("gp_lambda", [0.1, 1.0])
    def test_gail_notensordict(self, device, use_grad_penalty, gp_lambda):
        torch.manual_seed(self.seed)
        discriminator = self._create_mock_discriminator(device=device)
        loss_fn = GAILLoss(
            discriminator, use_grad_penalty=use_grad_penalty, gp_lambda=gp_lambda
        )

        tensordict = self._create_mock_data_gail(device=device)

        in_keys = self._flatten_in_keys(loss_fn.in_keys)
        kwargs = dict(tensordict.flatten_keys("_").select(*in_keys))

        loss_val_td = loss_fn(tensordict)
        if use_grad_penalty:
            loss_val, _ = loss_fn(**kwargs)
        else:
            loss_val = loss_fn(**kwargs)

        torch.testing.assert_close(loss_val_td.get("loss"), loss_val)
        # test select
        loss_fn.select_out_keys("loss")
        if torch.__version__ >= "2.0.0":
            loss_discriminator = loss_fn(**kwargs)
        else:
            with pytest.raises(
                RuntimeError,
                match="You are likely using tensordict.nn.dispatch with keyword arguments",
            ):
                loss_discriminator = loss_fn(**kwargs)
            return
        assert loss_discriminator == loss_val_td["loss"]

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("use_grad_penalty", [True, False])
    @pytest.mark.parametrize("gp_lambda", [0.1, 1.0])
    def test_gail(self, device, use_grad_penalty, gp_lambda):
        torch.manual_seed(self.seed)
        td = self._create_mock_data_gail(device=device)

        discriminator = self._create_mock_discriminator(device=device)

        loss_fn = GAILLoss(
            discriminator, use_grad_penalty=use_grad_penalty, gp_lambda=gp_lambda
        )
        loss = loss_fn(td)
        loss_transformer = loss["loss"]
        loss_transformer.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "discriminator" in name
            if p.grad is None:
                assert "discriminator" not in name
        loss_fn.zero_grad()

        sum([loss_transformer]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("device", get_available_devices())
    def test_gail_state_dict(self, device):
        torch.manual_seed(self.seed)

        discriminator = self._create_mock_discriminator(device=device)

        loss_fn = GAILLoss(discriminator)
        sd = loss_fn.state_dict()
        loss_fn2 = GAILLoss(discriminator)
        loss_fn2.load_state_dict(sd)

    @pytest.mark.parametrize("device", get_available_devices())
    @pytest.mark.parametrize("use_grad_penalty", [True, False])
    @pytest.mark.parametrize("gp_lambda", [0.1, 1.0])
    def test_seq_gail(self, device, use_grad_penalty, gp_lambda):
        torch.manual_seed(self.seed)
        td = self._create_seq_mock_data_gail(device=device)

        discriminator = self._create_mock_discriminator(device=device)

        loss_fn = GAILLoss(
            discriminator, use_grad_penalty=use_grad_penalty, gp_lambda=gp_lambda
        )
        loss = loss_fn(td)
        loss_transformer = loss["loss"]
        loss_transformer.backward(retain_graph=True)
        named_parameters = loss_fn.named_parameters()

        for name, p in named_parameters:
            if p.grad is not None and p.grad.norm() > 0.0:
                assert "discriminator" in name
            if p.grad is None:
                assert "discriminator" not in name
        loss_fn.zero_grad()

        sum([loss_transformer]).backward()
        named_parameters = list(loss_fn.named_parameters())
        named_buffers = list(loss_fn.named_buffers())

        assert len({p for n, p in named_parameters}) == len(list(named_parameters))
        assert len({p for n, p in named_buffers}) == len(list(named_buffers))

        for name, p in named_parameters:
            assert p.grad.norm() > 0.0, f"parameter {name} has a null gradient"

    @pytest.mark.parametrize("reduction", [None, "none", "mean", "sum"])
    @pytest.mark.parametrize("use_grad_penalty", [True, False])
    @pytest.mark.parametrize("gp_lambda", [0.1, 1.0])
    def test_gail_reduction(self, reduction, use_grad_penalty, gp_lambda):
        torch.manual_seed(self.seed)
        device = (
            torch.device("cpu")
            if torch.cuda.device_count() == 0
            else torch.device("cuda")
        )
        td = self._create_mock_data_gail(device=device)
        discriminator = self._create_mock_discriminator(device=device)
        loss_fn = GAILLoss(discriminator, reduction=reduction)
        loss = loss_fn(td)
        if reduction == "none":
            assert loss["loss"].shape == (td["observation"].shape[0], 1)
        else:
            assert loss["loss"].shape == torch.Size([])
