# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from unittest.mock import Mock

import pytest
import torch
from _objectives_common import _has_botorch

from tensordict import TensorDict
from tensordict.nn import TensorDictModule
from torch import nn

from torchrl.data import Composite, Unbounded
from torchrl.envs import EnvBase
from torchrl.envs.model_based.imagined import ImaginedEnv
from torchrl.envs.transforms import MeanActionSelector, TransformedEnv
from torchrl.modules.models.rbf_controller import RBFController
from torchrl.objectives import ExponentialQuadraticCost


class TestRBFController:
    @pytest.mark.parametrize("input_dim", [2, 4])
    @pytest.mark.parametrize("output_dim", [1, 3])
    @pytest.mark.parametrize("n_basis", [5, 10])
    def test_forward_shapes(self, input_dim, output_dim, n_basis):
        max_action = torch.ones(output_dim)
        controller = RBFController(
            input_dim=input_dim,
            output_dim=output_dim,
            max_action=max_action,
            n_basis=n_basis,
        ).double()

        batch_size = 3
        mean = torch.randn(batch_size, input_dim, dtype=torch.float64)
        cov = (
            torch.eye(input_dim, dtype=torch.float64)
            .unsqueeze(0)
            .expand(batch_size, -1, -1)
            * 0.1
        )

        action_mean, action_cov, cross_cov = controller(mean, cov)

        assert action_mean.shape == (batch_size, output_dim)
        assert action_cov.shape == (batch_size, output_dim, output_dim)
        assert cross_cov.shape == (batch_size, input_dim, output_dim)

    def test_action_covariance_is_symmetric(self):
        controller = RBFController(
            input_dim=4, output_dim=2, max_action=1.0, n_basis=5
        ).double()

        mean = torch.randn(2, 4, dtype=torch.float64)
        cov = torch.eye(4, dtype=torch.float64).unsqueeze(0).expand(2, -1, -1) * 0.1

        _, action_cov, _ = controller(mean, cov)

        torch.testing.assert_close(
            action_cov, action_cov.transpose(-2, -1), atol=1e-6, rtol=1e-5
        )

    def test_action_covariance_is_positive_semidefinite(self):
        controller = RBFController(
            input_dim=4, output_dim=2, max_action=1.0, n_basis=5
        ).double()

        mean = torch.randn(2, 4, dtype=torch.float64)
        cov = torch.eye(4, dtype=torch.float64).unsqueeze(0).expand(2, -1, -1) * 0.1

        _, action_cov, _ = controller(mean, cov)

        eigenvalues = torch.linalg.eigvalsh(action_cov)
        assert (
            eigenvalues >= -1e-6
        ).all(), f"Negative eigenvalues found: {eigenvalues}"

    @pytest.mark.parametrize("max_action", [0.5, 1.0, 2.0])
    def test_squash_sin_bounds(self, max_action):
        mean = torch.randn(10, 3, dtype=torch.float64)
        cov = torch.eye(3, dtype=torch.float64).unsqueeze(0).expand(10, -1, -1) * 0.01

        squashed_mean, squashed_cov, cross_cov = RBFController.squash_sin(
            mean, cov, max_action
        )

        assert (squashed_mean.abs() <= max_action + 1e-6).all()
        assert squashed_cov.shape == (10, 3, 3)
        assert cross_cov.shape == (10, 3, 3)

    def test_deterministic_with_zero_variance(self):
        controller = RBFController(
            input_dim=4, output_dim=1, max_action=1.0, n_basis=5
        ).double()

        mean = torch.randn(2, 4, dtype=torch.float64)
        zero_cov = torch.zeros(2, 4, 4, dtype=torch.float64)

        action_mean1, _, _ = controller(mean, zero_cov)
        action_mean2, _, _ = controller(mean, zero_cov)

        torch.testing.assert_close(action_mean1, action_mean2)

    def test_gradients_flow(self):
        controller = RBFController(
            input_dim=4, output_dim=1, max_action=1.0, n_basis=5
        ).double()

        mean = torch.randn(2, 4, dtype=torch.float64)
        cov = torch.eye(4, dtype=torch.float64).unsqueeze(0).expand(2, -1, -1) * 0.1

        action_mean, action_cov, cross_cov = controller(mean, cov)
        loss = action_mean.sum() + action_cov.sum()
        loss.backward()

        for name, param in controller.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"

    def test_as_tensordict_module(self):
        controller = RBFController(
            input_dim=4, output_dim=1, max_action=1.0, n_basis=5
        ).double()

        module = TensorDictModule(
            module=controller,
            in_keys=[("observation", "mean"), ("observation", "var")],
            out_keys=[
                ("action", "mean"),
                ("action", "var"),
                ("action", "cross_covariance"),
            ],
        )

        td = TensorDict(
            {
                ("observation", "mean"): torch.randn(2, 4, dtype=torch.float64),
                ("observation", "var"): torch.eye(4, dtype=torch.float64)
                .unsqueeze(0)
                .expand(2, -1, -1)
                * 0.1,
            },
            batch_size=[2],
        )

        out = module(td)
        assert ("action", "mean") in out.keys(True)
        assert ("action", "var") in out.keys(True)
        assert ("action", "cross_covariance") in out.keys(True)


class TestExponentialQuadraticCost:
    def test_forward_shapes_default(self):
        cost = ExponentialQuadraticCost(reduction="none")

        td = TensorDict(
            {
                ("observation", "mean"): torch.randn(2, 5, 4),
                ("observation", "var"): torch.eye(4)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(2, 5, -1, -1)
                * 0.1,
            },
            batch_size=[2, 5],
        )

        out = cost(td)
        loss = out["loss_cost"]
        assert loss.shape == (2, 5)

    def test_cost_at_target_is_low(self):
        target = torch.zeros(4)
        cost = ExponentialQuadraticCost(target=target, reduction="none")

        td = TensorDict(
            {
                ("observation", "mean"): torch.zeros(1, 4),
                ("observation", "var"): torch.eye(4).unsqueeze(0) * 1e-6,
            },
            batch_size=[1],
        )

        out = cost(td)
        assert out["loss_cost"].item() < 0.01

    def test_cost_far_from_target_is_high(self):
        target = torch.zeros(4)
        cost = ExponentialQuadraticCost(target=target, reduction="none")

        td = TensorDict(
            {
                ("observation", "mean"): torch.ones(1, 4) * 10.0,
                ("observation", "var"): torch.eye(4).unsqueeze(0) * 0.1,
            },
            batch_size=[1],
        )

        out = cost(td)
        assert out["loss_cost"].item() > 0.9

    def test_cost_bounded_zero_one(self):
        cost = ExponentialQuadraticCost(reduction="none")

        td = TensorDict(
            {
                ("observation", "mean"): torch.randn(10, 4),
                ("observation", "var"): torch.eye(4).unsqueeze(0).expand(10, -1, -1)
                * 0.1,
            },
            batch_size=[10],
        )

        out = cost(td)
        loss = out["loss_cost"]
        assert (loss >= -1e-6).all()
        assert (loss <= 1.0 + 1e-6).all()

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_reductions(self, reduction):
        cost = ExponentialQuadraticCost(reduction=reduction)

        td = TensorDict(
            {
                ("observation", "mean"): torch.randn(3, 5, 4),
                ("observation", "var"): torch.eye(4)
                .unsqueeze(0)
                .unsqueeze(0)
                .expand(3, 5, -1, -1)
                * 0.1,
            },
            batch_size=[3, 5],
        )

        out = cost(td)
        loss = out["loss_cost"]

        if reduction == "none":
            assert loss.shape == (3, 5)
        else:
            assert loss.shape == ()

    def test_custom_weights_and_target(self):
        weights = torch.diag(torch.tensor([2.0, 0.5, 1.0, 1.0]))
        target = torch.tensor([1.0, 0.0, 0.0, 0.0])
        cost = ExponentialQuadraticCost(
            target=target, weights=weights, reduction="none"
        )

        td = TensorDict(
            {
                ("observation", "mean"): target.unsqueeze(0),
                ("observation", "var"): torch.eye(4).unsqueeze(0) * 1e-6,
            },
            batch_size=[1],
        )

        out = cost(td)
        assert out["loss_cost"].item() < 0.01

    def test_gradients_flow(self):
        cost = ExponentialQuadraticCost(reduction="mean")

        mean = torch.randn(2, 4, requires_grad=True)
        var = torch.eye(4).unsqueeze(0).expand(2, -1, -1) * 0.1

        td = TensorDict(
            {("observation", "mean"): mean, ("observation", "var"): var},
            batch_size=[2],
        )

        out = cost(td)
        out["loss_cost"].backward()
        assert mean.grad is not None


class TestImaginedEnv:
    @staticmethod
    def _make_dummy_world_model(obs_dim, action_dim):
        class DummyWM(torch.nn.Module):
            def __init__(self, obs_dim):
                super().__init__()
                self.obs_dim = obs_dim

            def forward(self, action, observation):
                mean = observation.get("mean")
                var = (
                    torch.eye(
                        self.obs_dim, device=mean.device, dtype=mean.dtype
                    ).expand(*mean.shape[:-1], -1, -1)
                    * 0.01
                )
                return mean + 0.1, var

        return TensorDictModule(
            DummyWM(obs_dim),
            in_keys=["action", "observation"],
            out_keys=[("next_observation", "mean"), ("next_observation", "var")],
        )

    @staticmethod
    def _make_base_env(obs_dim, action_dim):
        class StubEnv(EnvBase):
            def __init__(self, obs_dim, action_dim):
                super().__init__(batch_size=torch.Size([]))
                self.observation_spec = Composite(
                    observation=Unbounded(shape=(obs_dim,))
                )
                self.action_spec = Unbounded(shape=(action_dim,))
                self.reward_spec = Unbounded(shape=(1,))

            def _reset(self, tensordict=None):
                return TensorDict(
                    {"observation": torch.zeros(obs_dim)},
                    batch_size=self.batch_size,
                )

            def _step(self, tensordict):
                return TensorDict(
                    {
                        "observation": torch.randn(obs_dim),
                        "reward": torch.zeros(1),
                        "done": torch.tensor(False).unsqueeze(0),
                        "terminated": torch.tensor(False).unsqueeze(0),
                    },
                    batch_size=self.batch_size,
                )

            def _set_seed(self, seed):
                pass

        return StubEnv(obs_dim, action_dim)

    def test_creation(self):
        obs_dim, action_dim = 4, 1
        wm = self._make_dummy_world_model(obs_dim, action_dim)
        base_env = self._make_base_env(obs_dim, action_dim)

        env = ImaginedEnv(world_model_module=wm, base_env=base_env)
        assert env.batch_size == torch.Size([1])
        assert ("observation", "mean") in env.observation_spec.keys(True)
        assert ("observation", "var") in env.observation_spec.keys(True)

    def test_creation_with_batch_size(self):
        obs_dim, action_dim = 4, 1
        wm = self._make_dummy_world_model(obs_dim, action_dim)
        base_env = self._make_base_env(obs_dim, action_dim)

        env = ImaginedEnv(world_model_module=wm, base_env=base_env, batch_size=[3])
        assert env.batch_size == torch.Size([3])

    def test_reset_with_observation(self):
        obs_dim, action_dim = 4, 1
        wm = self._make_dummy_world_model(obs_dim, action_dim)
        base_env = self._make_base_env(obs_dim, action_dim)

        env = ImaginedEnv(world_model_module=wm, base_env=base_env)

        reset_td = TensorDict(
            {
                ("observation", "mean"): torch.zeros(1, obs_dim),
                ("observation", "var"): torch.eye(obs_dim).unsqueeze(0) * 1e-3,
            },
            batch_size=[1],
        )

        out = env.reset(reset_td)
        assert ("observation", "mean") in out.keys(True)
        assert ("observation", "var") in out.keys(True)
        assert env.observation_spec.contains(out.select("observation"))

    def test_step(self):
        obs_dim, action_dim = 4, 1
        next_observation_key = (
            "next_observation"  # ("next", "observation") could also be a possibility
        )
        wm = self._make_dummy_world_model(obs_dim, action_dim)
        base_env = self._make_base_env(obs_dim, action_dim)

        env = ImaginedEnv(
            world_model_module=wm,
            base_env=base_env,
            next_observation_key=next_observation_key,
        )

        td = TensorDict(
            {
                ("observation", "mean"): torch.zeros(1, obs_dim),
                ("observation", "var"): torch.eye(obs_dim).unsqueeze(0) * 1e-3,
                ("action", "mean"): torch.zeros(1, action_dim),
                ("action", "var"): torch.zeros(1, action_dim, action_dim),
                ("action", "cross_covariance"): torch.zeros(1, obs_dim, action_dim),
            },
            batch_size=[1],
        )

        out = env.step(td)
        next_td = out["next"]
        assert ("observation", "mean") in next_td.keys(True)
        assert ("observation", "var") in next_td.keys(True)
        assert "done" in next_td.keys()
        assert not next_td["done"].any()
        assert env.observation_spec.contains(next_td.select("observation"))

    def test_never_terminates(self):
        obs_dim, action_dim = 4, 1
        wm = self._make_dummy_world_model(obs_dim, action_dim)
        base_env = self._make_base_env(obs_dim, action_dim)

        env = ImaginedEnv(world_model_module=wm, base_env=base_env)

        td = TensorDict(
            {"done": torch.ones(1, 1, dtype=torch.bool)},
            batch_size=[1],
        )
        assert not env.any_done(td)

    def test_check_env_specs(self):
        obs_dim, action_dim = 4, 1
        wm = self._make_dummy_world_model(obs_dim, action_dim)
        base_env = self._make_base_env(obs_dim, action_dim)

        env = ImaginedEnv(
            world_model_module=wm,
            base_env=base_env,
            next_observation_key="next_observation",
        )

        env.check_env_specs()

        td = TensorDict(
            {
                ("observation", "mean"): torch.zeros(1, obs_dim),
                ("observation", "var"): torch.eye(obs_dim).unsqueeze(0) * 1e-3,
                ("action", "mean"): torch.zeros(1, action_dim),
                ("action", "var"): torch.zeros(1, action_dim, action_dim),
                ("action", "cross_covariance"): torch.zeros(1, obs_dim, action_dim),
            },
            batch_size=[1],
        )
        out = env.step(td)
        assert ("next_observation", "mean") not in out.keys(True)
        assert ("next_observation", "var") not in out.keys(True)


class TestMeanActionSelector:
    @staticmethod
    def _make_base_env(obs_dim, action_dim):
        class StubEnv(EnvBase):
            def __init__(self, obs_dim, action_dim):
                super().__init__(batch_size=torch.Size([]))
                self.observation_spec = Composite(
                    observation=Unbounded(shape=(obs_dim,))
                )
                self.action_spec = Unbounded(shape=(action_dim,))
                self.reward_spec = Unbounded(shape=(1,))

            def _reset(self, tensordict=None):
                return TensorDict(
                    {"observation": torch.zeros(obs_dim)},
                    batch_size=self.batch_size,
                )

            def _step(self, tensordict):
                return TensorDict(
                    {
                        "observation": torch.randn(obs_dim),
                        "reward": torch.zeros(1),
                        "done": torch.tensor(False).unsqueeze(0),
                        "terminated": torch.tensor(False).unsqueeze(0),
                    },
                    batch_size=self.batch_size,
                )

            def _set_seed(self, seed):
                pass

        return StubEnv(obs_dim, action_dim)

    def test_forward_wraps_observation(self):
        transform = MeanActionSelector()
        obs = torch.randn(4)
        td = TensorDict(
            {"observation": obs.clone()},
            batch_size=[],
        )

        out = transform._call(td)
        assert ("observation", "mean") in out.keys(True)
        assert ("observation", "var") in out.keys(True)
        assert out["observation", "var"].shape == (4, 4)
        torch.testing.assert_close(out["observation", "mean"], obs)

    def test_inverse_extracts_action_mean(self):
        transform = MeanActionSelector()
        action_mean = torch.randn(2)
        td = TensorDict(
            {
                ("action", "mean"): action_mean,
                ("action", "var"): torch.eye(2),
            },
            batch_size=[],
        )

        out = transform._inv_call(td)
        assert "action" in out.keys()
        torch.testing.assert_close(out["action"], action_mean)

    def test_with_transformed_env_reset(self):
        obs_dim, action_dim = 4, 1
        base_env = self._make_base_env(obs_dim, action_dim)
        env = TransformedEnv(base_env, MeanActionSelector())

        td = env.reset()
        assert ("observation", "mean") in td.keys(True)
        assert ("observation", "var") in td.keys(True)

    def test_observation_spec_transformed(self):
        obs_dim, action_dim = 4, 1
        base_env = self._make_base_env(obs_dim, action_dim)
        env = TransformedEnv(base_env, MeanActionSelector())

        obs_spec = env.observation_spec
        assert ("observation", "mean") in obs_spec.keys(True)
        assert ("observation", "var") in obs_spec.keys(True)

    def test_zero_variance_on_reset(self):
        obs_dim, action_dim = 4, 1
        base_env = self._make_base_env(obs_dim, action_dim)
        env = TransformedEnv(base_env, MeanActionSelector())

        td = env.reset()
        var = td["observation", "var"]
        torch.testing.assert_close(var, torch.zeros(obs_dim, obs_dim))


@pytest.mark.skipif(not _has_botorch, reason="botorch/gpytorch not installed")
class TestGPWorldModel:
    @staticmethod
    def _make_dispatch_only_model():
        from torchrl.modules.models.gp import GPWorldModel

        model = GPWorldModel.__new__(GPWorldModel)
        nn.Module.__init__(model)
        model.in_keys = [
            ("action", "mean"),
            ("action", "var"),
            ("action", "cross_covariance"),
            ("observation", "mean"),
            ("observation", "var"),
        ]
        model.uncertain_forward = Mock(
            name="uncertain_forward", return_value="uncertain"
        )
        model.deterministic_forward = Mock(
            name="deterministic_forward", return_value="deterministic"
        )
        return model

    def test_creation(self):
        from torchrl.modules.models.gp import GPWorldModel

        model = GPWorldModel(obs_dim=4, action_dim=1)
        assert model.obs_dim == 4
        assert model.action_dim == 1
        assert model.state_action_dim == 5

    def test_fit_and_deterministic_forward(self):
        from torchrl.modules.models.gp import GPWorldModel

        obs_dim, action_dim = 2, 1
        model = GPWorldModel(obs_dim=obs_dim, action_dim=action_dim)

        n_samples = 20
        obs = torch.randn(n_samples, obs_dim).double()
        action = torch.randn(n_samples, action_dim).double()
        next_obs = obs + 0.1 * torch.randn(n_samples, obs_dim).double()

        dataset = TensorDict(
            {
                "observation": obs,
                "action": action,
                ("next", "observation"): next_obs,
            },
            batch_size=[n_samples],
        )

        model.fit(dataset)
        model.eval()

        td = TensorDict(
            {
                ("observation", "mean"): torch.randn(3, obs_dim),
                ("action", "mean"): torch.randn(3, action_dim),
            },
            batch_size=[3],
        )

        forward_td = model.deterministic_forward(td)

        assert forward_td[("next", "observation", "mean")].shape == (3, obs_dim)
        assert forward_td[("next", "observation", "var")].shape == (3, obs_dim, obs_dim)

    def test_uncertain_forward(self):
        from torchrl.modules.models.gp import GPWorldModel

        obs_dim, action_dim = 2, 1
        model = GPWorldModel(obs_dim=obs_dim, action_dim=action_dim)

        n_samples = 20
        obs = torch.randn(n_samples, obs_dim).double()
        action = torch.randn(n_samples, action_dim).double()
        next_obs = obs + 0.1 * torch.randn(n_samples, obs_dim).double()

        dataset = TensorDict(
            {
                "observation": obs,
                "action": action,
                ("next", "observation"): next_obs,
            },
            batch_size=[n_samples],
        )

        model.double()
        model.fit(dataset)
        model.eval()

        batch = 2
        td = TensorDict(
            {
                "observation": {
                    "mean": torch.randn(batch, obs_dim, dtype=torch.float64),
                    "var": torch.eye(obs_dim, dtype=torch.float64)
                    .unsqueeze(0)
                    .expand(batch, -1, -1)
                    * 0.01,
                },
                "action": {
                    "mean": torch.randn(batch, action_dim, dtype=torch.float64),
                    "var": torch.eye(action_dim, dtype=torch.float64)
                    .unsqueeze(0)
                    .expand(batch, -1, -1)
                    * 0.01,
                    "cross_covariance": torch.zeros(
                        batch, obs_dim, action_dim, dtype=torch.float64
                    ),
                },
            },
            batch_size=[batch],
        )

        forward_td = model.uncertain_forward(td)

        mean, var = (
            forward_td[("next", "observation", "mean")],
            forward_td[("next", "observation", "var")],
        )
        assert mean.shape == (batch, obs_dim)
        assert var.shape == (batch, obs_dim, obs_dim)

        torch.testing.assert_close(var, var.transpose(-2, -1), atol=1e-5, rtol=1e-4)

    def test_forward_dispatch_observation_uncertainty(self):
        obs_dim, action_dim = 2, 1
        model = self._make_dispatch_only_model()

        batch = 2
        td = TensorDict(
            {
                "observation": {
                    "mean": torch.randn(batch, obs_dim, dtype=torch.float64),
                    "var": torch.eye(obs_dim, dtype=torch.float64)
                    .unsqueeze(0)
                    .expand(batch, -1, -1)
                    * 0.01,
                },
                "action": {
                    "mean": torch.randn(batch, action_dim, dtype=torch.float64),
                    "var": torch.eye(action_dim, dtype=torch.float64)
                    .unsqueeze(0)
                    .expand(batch, -1, -1)
                    * 0.01,
                    "cross_covariance": torch.zeros(
                        batch, obs_dim, action_dim, dtype=torch.float64
                    ),
                },
            },
            batch_size=[batch],
        )
        assert model.forward(td) == "uncertain"
        model.uncertain_forward.assert_called_once_with(td)
        model.deterministic_forward.assert_not_called()

    @pytest.mark.parametrize(
        "action_key", [("action", "var"), ("action", "cross_covariance")]
    )
    def test_forward_dispatch_action_uncertainty(self, action_key):
        obs_dim, action_dim = 2, 1
        model = self._make_dispatch_only_model()

        action_value = torch.eye(action_dim, dtype=torch.float64).unsqueeze(0) * 0.01
        if action_key == ("action", "cross_covariance"):
            action_value = torch.full(
                (1, obs_dim, action_dim), 0.01, dtype=torch.float64
            )

        td = TensorDict(
            {
                "observation": {
                    "mean": torch.randn(1, obs_dim, dtype=torch.float64),
                    "var": torch.zeros(1, obs_dim, obs_dim, dtype=torch.float64),
                },
                "action": {
                    "mean": torch.randn(1, action_dim, dtype=torch.float64),
                    "var": torch.zeros(1, action_dim, action_dim, dtype=torch.float64),
                    "cross_covariance": torch.zeros(
                        1, obs_dim, action_dim, dtype=torch.float64
                    ),
                },
            },
            batch_size=[1],
        )
        td.set(action_key, action_value)

        assert model.forward(td) == "uncertain"
        model.uncertain_forward.assert_called_once_with(td)
        model.deterministic_forward.assert_not_called()

    def test_forward_dispatch_deterministic_when_covariances_are_zero(self):
        obs_dim, action_dim = 2, 1
        model = self._make_dispatch_only_model()

        td = TensorDict(
            {
                "observation": {
                    "mean": torch.randn(1, obs_dim, dtype=torch.float64),
                    "var": torch.zeros(1, obs_dim, obs_dim, dtype=torch.float64),
                },
                "action": {
                    "mean": torch.randn(1, action_dim, dtype=torch.float64),
                    "var": torch.zeros(1, action_dim, action_dim, dtype=torch.float64),
                    "cross_covariance": torch.zeros(
                        1, obs_dim, action_dim, dtype=torch.float64
                    ),
                },
            },
            batch_size=[1],
        )

        assert model.forward(td) == "deterministic"
        model.deterministic_forward.assert_called_once_with(td)
        model.uncertain_forward.assert_not_called()
