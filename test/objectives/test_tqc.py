# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
import torchrl.objectives.tqc as tqc_objective
from _objectives_common import (
    _check_td_steady as check_td_steady,
    _has_functorch as has_functorch,
    FUNCTORCH_ERR,
)
from packaging import version
from tensordict import TensorDict
from tensordict.nn import NormalParamExtractor, TensorDictModule
from torch import nn

from torchrl.data import Bounded
from torchrl.modules import MLP, ProbabilisticActor, ValueOperator
from torchrl.modules.distributions import TanhNormal
from torchrl.objectives import SoftUpdate, TQCLoss, ValueEstimators


class ConstantQuantiles(nn.Module):
    def __init__(self, quantiles):
        super().__init__()
        self.quantiles = nn.Parameter(torch.tensor(quantiles, dtype=torch.float32))

    def forward(self, observation, action):
        return self.quantiles.expand(*observation.shape[:-1], -1)


@pytest.mark.skipif(
    not has_functorch, reason=f"functorch not installed: {FUNCTORCH_ERR}"
)
class TestTQC:
    def make_actor(self, observation_key="observation"):
        action_spec = Bounded(-1, 1, (2,))
        return ProbabilisticActor(
            TensorDictModule(
                nn.Sequential(nn.Linear(3, 4), NormalParamExtractor()),
                in_keys=[observation_key],
                out_keys=["loc", "scale"],
            ),
            in_keys=["loc", "scale"],
            spec=action_spec,
            distribution_class=TanhNormal,
        )

    def make_data(
        self,
        batch_size=(7,),
        observation_key="observation",
        reward_key="reward",
        done_key="done",
        terminated_key="terminated",
    ):
        def next_key(key):
            return ("next", *key) if isinstance(key, tuple) else ("next", key)

        return TensorDict(
            {
                observation_key: torch.randn(*batch_size, 3),
                "action": torch.randn(*batch_size, 2).tanh(),
                next_key(observation_key): torch.randn(*batch_size, 3),
                next_key(reward_key): torch.randn(*batch_size, 1),
                next_key(done_key): torch.zeros(*batch_size, 1, dtype=torch.bool),
                next_key(terminated_key): torch.zeros(*batch_size, 1, dtype=torch.bool),
            },
            batch_size=batch_size,
        )

    def make_loss(
        self,
        *,
        num_quantiles=5,
        num_qvalue_nets=3,
        top_quantiles_to_drop_per_net=1,
        reduction="mean",
        observation_key="observation",
        skip_done_states=False,
    ):
        actor = self.make_actor(observation_key=observation_key)
        critic = ValueOperator(
            MLP(
                in_features=5,
                out_features=num_quantiles,
                num_cells=[16, 16],
            ),
            in_keys=[observation_key, "action"],
        )
        loss = TQCLoss(
            actor,
            critic,
            num_qvalue_nets=num_qvalue_nets,
            top_quantiles_to_drop_per_net=top_quantiles_to_drop_per_net,
            reduction=reduction,
            skip_done_states=skip_done_states,
            scalar_output_mode="exclude" if reduction == "none" else None,
        )
        loss.make_value_estimator(gamma=0.9)
        SoftUpdate(loss, eps=0.95)
        return loss

    def test_forward_shapes_and_gradient_isolation(self):
        torch.manual_seed(0)
        batch_size = (2, 3)
        loss = self.make_loss(reduction="none")
        data = self.make_data(batch_size=batch_size)

        with check_td_steady(data):
            output = loss(data)
        assert set(output.keys()) == {"loss_actor", "loss_qvalue", "loss_alpha"}
        assert all(value.shape == torch.Size(batch_size) for value in output.values())
        assert data.get(loss.tensor_keys.priority).shape == torch.Size(batch_size)
        assert output.isfinite().all()

        loss.zero_grad(set_to_none=True)
        actor_loss, _ = loss.actor_loss(data)
        actor_loss.mean().backward()
        actor_params = list(loss.actor_network_params.values(True, True))
        critic_params = list(loss.qvalue_network_params.values(True, True))
        assert any(parameter.grad is not None for parameter in actor_params)
        assert all(parameter.grad is None for parameter in critic_params)

        loss.zero_grad(set_to_none=True)
        critic_loss, _ = loss.qvalue_v2_loss(data)
        critic_loss.mean().backward()
        assert all(parameter.grad is None for parameter in actor_params)
        assert any(parameter.grad is not None for parameter in critic_params)

    @pytest.mark.parametrize("all_done", [False, True])
    def test_terminal_nan_is_selected_out(self, all_done):
        loss = self.make_loss(skip_done_states=True)
        data = self.make_data(batch_size=(2, 3))
        if all_done:
            data.get(("next", "done")).fill_(True)
            data.get(("next", "terminated")).fill_(True)
            data.get(("next", "observation")).fill_(float("nan"))
        else:
            data.get(("next", "done"))[0, 0] = True
            data.get(("next", "terminated"))[0, 0] = True
            data.get(("next", "observation"))[0, 0] = float("nan")

        target = loss.compute_target(data)
        assert target.isfinite().all()
        terminal_reward = data.get(("next", "reward"))[0, 0]
        torch.testing.assert_close(
            target[0, 0], terminal_reward.expand_as(target[0, 0])
        )

        output = loss(data)
        total_loss = sum(
            value for key, value in output.items() if key.startswith("loss")
        )
        total_loss.backward()
        gradients = [
            parameter.grad
            for parameter in loss.parameters()
            if parameter.grad is not None
        ]
        assert gradients and all(gradient.isfinite().all() for gradient in gradients)

        if all_done:
            truncated = self.make_data(batch_size=(2, 3))
            torch.manual_seed(0)
            expected = loss.compute_target(truncated)
            truncated.get(("next", "done")).fill_(True)
            torch.manual_seed(0)
            torch.testing.assert_close(loss.compute_target(truncated), expected)

    @pytest.mark.parametrize("deactivate_vmap", [False, True])
    def test_tqc_numerical_contract(self, monkeypatch, deactivate_vmap):
        if deactivate_vmap and version.parse(torch.__version__) < version.parse(
            "2.7.0"
        ):
            pytest.skip("pseudo-vmap requires Torch >= 2.7.0")
        actor = self.make_actor()
        critic = [
            ValueOperator(
                ConstantQuantiles(quantiles), in_keys=["observation", "action"]
            )
            for quantiles in ([0.0, 100.0, 101.0], [1.0, 2.0, 3.0])
        ]
        loss = TQCLoss(
            actor,
            critic,
            num_qvalue_nets=2,
            top_quantiles_to_drop_per_net=1,
            reduction="none",
            scalar_output_mode="exclude",
            deactivate_vmap=deactivate_vmap,
        )
        SoftUpdate(loss, eps=0.95)
        loss.make_value_estimator(gamma=1.0)
        data = self.make_data(batch_size=(1,))
        data.get(("next", "reward")).zero_()
        data.set("steps_to_next_obs", torch.ones(1, 1))

        def sample_with_zero_log_prob(distribution):
            action = distribution.rsample()
            return action, action.new_zeros(action.shape[:-1])

        monkeypatch.setattr(
            tqc_objective,
            "compute_rsample_log_prob",
            sample_with_zero_log_prob,
        )

        target = loss.compute_target(data)
        torch.testing.assert_close(target, torch.tensor([[0.0, 1.0, 2.0, 3.0]]))

        actor_loss = loss.actor_loss(data)[0]
        torch.testing.assert_close(actor_loss, torch.tensor([-34.5]))

        critic_loss, critic_metadata = loss.qvalue_v2_loss(data)
        torch.testing.assert_close(critic_loss, torch.tensor([22.125]))
        torch.testing.assert_close(critic_metadata["td_error"], torch.tensor([66.5]))

        critic_loss.sum().backward()
        critic_gradient = loss.qvalue_network_params.get(("module", "quantiles")).grad
        expected_gradient = torch.tensor(
            [
                [-1 / 24, 1 / 6, 1 / 18],
                [1 / 24, 1 / 24, 1 / 24],
            ]
        )
        torch.testing.assert_close(critic_gradient, expected_gradient)

    def test_nested_keys_and_value_estimator_contract(self):
        observation_key = ("agent", "observation")
        reward_key = ("metrics", "reward")
        done_key = ("flags", "done")
        terminated_key = ("flags", "terminated")
        priority_key = ("replay", "error")
        loss = self.make_loss(observation_key=observation_key)
        loss.set_keys(
            reward=reward_key,
            done=done_key,
            terminated=terminated_key,
            priority=priority_key,
        )
        data = self.make_data(
            batch_size=(5,),
            observation_key=observation_key,
            reward_key=reward_key,
            done_key=done_key,
            terminated_key=terminated_key,
        )
        assert loss(data).isfinite().all()
        assert data.get(priority_key).shape == torch.Size([5])
        with pytest.raises(NotImplementedError, match="TD0"):
            loss.make_value_estimator(ValueEstimators.TD1)

    def test_rejects_negative_drop_count(self):
        with pytest.raises(ValueError, match="greater than or equal"):
            self.make_loss(top_quantiles_to_drop_per_net=-1)

    def test_rejects_dropping_every_atom(self):
        loss = self.make_loss(
            num_quantiles=2,
            num_qvalue_nets=2,
            top_quantiles_to_drop_per_net=2,
        )
        with pytest.raises(ValueError, match="smaller"):
            loss.compute_target(self.make_data())


if __name__ == "__main__":
    pytest.main([__file__])
