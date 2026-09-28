# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch
from tensordict import TensorDict
from tensordict.nn import TensorDictModule, TensorDictSequential

from torchrl.modules import FlowMatchingPolicy, MLP, OneStepPolicy, ValueOperator
from torchrl.objectives import FQLLoss, SoftUpdate
from torchrl.testing import get_default_devices


def make_loss(device, dtype=torch.float32, *, routing=None, **kwargs):
    def network(inputs, outputs):
        return MLP(
            inputs, outputs, num_cells=[8], device=device, layer_kwargs={"dtype": dtype}
        )

    flow = FlowMatchingPolicy(network(6, 2), 2, num_steps=3)
    actor = OneStepPolicy(network(5, 2), 2)
    critics = [
        ValueOperator(network(5, 1), in_keys=["observation", "action"])
        for _ in range(2)
    ]
    if routing:
        observation = ("observations", "state")
        action = ("agent", "action")
        for policy in (flow, actor):
            policy.in_keys = [observation, ("policy", "noise")]
            policy.out_keys = [action]
        for critic in critics:
            critic.in_keys = [
                ("observations", "left"),
                ("observations", "right"),
                action,
            ]
        if routing == "encoder":

            def encoded(policy):
                projection = MLP(3, 3, num_cells=[], device=device)
                with torch.no_grad():
                    projection[0].weight.copy_(torch.eye(3, device=device))
                    projection[0].bias.zero_()
                encoder = TensorDictModule(
                    projection,
                    in_keys=[("observations", "left"), ("observations", "right")],
                    out_keys=[observation],
                )
                return TensorDictSequential(encoder, policy)

            flow, actor = encoded(flow), encoded(actor)
    loss = FQLLoss(flow, actor, critics, **kwargs)
    if routing:
        loss.set_keys(action=action)
    loss.make_value_estimator(gamma=0.9)
    return loss


def make_batch(device, dtype=torch.float32):
    return TensorDict(
        {
            "observation": torch.randn(2, 3, 3, device=device, dtype=dtype),
            "action": torch.randn(2, 3, 2, device=device, dtype=dtype).tanh(),
            ("next", "observation"): torch.randn(2, 3, 3, device=device, dtype=dtype),
            ("next", "reward"): torch.randn(2, 3, 1, device=device, dtype=dtype),
            ("next", "done"): torch.tensor([True, True, False], device=device)
            .expand(2, 3)
            .unsqueeze(-1),
            ("next", "terminated"): torch.tensor([True, False, False], device=device)
            .expand(2, 3)
            .unsqueeze(-1),
            ("collector", "mask"): torch.tensor(
                [True, False, True], device=device
            ).expand(2, 3),
            "priority_weight": torch.tensor(
                [0.5, 1.0, 2.0], device=device, dtype=dtype
            ).expand(2, 3),
        },
        [2, 3],
    )


@pytest.fixture
def fixed_random(monkeypatch):
    # Inductor and eager RNG streams differ; couple their draws for numerical parity.
    def normal_like(tensor):
        return torch.linspace(
            -0.7, 0.8, tensor.numel(), device=tensor.device, dtype=tensor.dtype
        ).reshape(tensor.shape)

    def uniform_like(tensor):
        return torch.full_like(tensor, 0.3)

    def normal_inplace(tensor):
        return tensor.copy_(normal_like(tensor))

    monkeypatch.setattr(torch, "randn_like", normal_like)
    monkeypatch.setattr(torch, "rand_like", uniform_like)
    monkeypatch.setattr(torch.Tensor, "normal_", normal_inplace)


@pytest.mark.parametrize("device", get_default_devices())
class TestFQL:
    @pytest.mark.parametrize("name", ["flow", "actor", "qvalue"])
    @pytest.mark.parametrize("reduction", ["none", "mean", "sum"])
    def test_gradients(self, device, name, reduction):
        torch.manual_seed(7)
        loss = make_loss(device, reduction=reduction)
        SoftUpdate(loss, tau=0.05)
        batch = make_batch(device)
        result = loss(batch)
        result["loss_" + name].sum().backward()
        for group, params in (
            ("flow", loss.flow_policy_params),
            ("actor", loss.actor_network_params),
            ("qvalue", loss.qvalue_network_params),
        ):
            gradients = [p.grad for p in params.values(True, True) if p.requires_grad]
            if group == name:
                assert all(g is not None and g.isfinite().all() for g in gradients)
                assert any(g.count_nonzero() for g in gradients)
            else:
                assert all(g is None for g in gradients)
        assert all(
            p.grad is None for p in loss.target_qvalue_network_params.values(True, True)
        )
        assert not result["target_value"].requires_grad
        assert not batch["td_error"].requires_grad

    @pytest.mark.parametrize("routing", ["nested", "encoder"])
    @pytest.mark.parametrize("compile_loss", [False, True])
    def test_policy_observations(self, device, routing, compile_loss, fixed_random):
        torch.manual_seed(7)
        reference = make_loss(device)
        torch.manual_seed(7)
        routed = make_loss(device, routing=routing)
        for loss in (reference, routed):
            SoftUpdate(loss, tau=0.05)
        batch = make_batch(device)
        data = batch.clone()
        data.rename_key_("action", ("agent", "action"))
        for td in (data, data["next"]):
            observation = td.pop("observation")
            td["observations", "left"] = observation[..., :1]
            td["observations", "right"] = observation[..., 1:]
            if routing == "nested":
                td["observations", "state"] = observation
        original = data.clone()
        expected = reference(batch)
        if compile_loss:
            torch._dynamo.reset()
        call = torch.compile(routed, fullgraph=True) if compile_loss else routed
        actual = call(data)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(data["td_error"], batch["td_error"])
        torch.testing.assert_close(data.exclude("td_error"), original)
        assert "observation" not in routed.in_keys
        assert ("observations", "left") in routed.in_keys
        assert ("next", "observations", "right") in routed.in_keys
        sum(actual[key] for key in routed.out_keys).backward()
        assert all(
            parameter.grad is not None and parameter.grad.isfinite().all()
            for parameter in routed.parameters()
        )
        if routing == "encoder":
            for params in (routed.flow_policy_params, routed.actor_network_params):
                assert params[
                    "module", "0", "module", "0", "weight"
                ].grad.count_nonzero()

    def test_unclamped_distillation(self, device, fixed_random):
        flow = torch.nn.Linear(6, 2, device=device)
        actor = torch.nn.Linear(5, 2, device=device)
        critic = MLP(5, 1, num_cells=[], device=device)
        with torch.no_grad():
            for network in (flow, actor, critic[0]):
                network.weight.zero_()
                network.bias.zero_()
            actor.bias.fill_(2)
            critic[0].weight[..., -2:] = 1
        loss = FQLLoss(
            FlowMatchingPolicy(flow, 2),
            OneStepPolicy(actor, 2),
            [ValueOperator(critic, in_keys=["observation", "action"])],
            num_qvalue_nets=1,
        )
        SoftUpdate(loss, tau=0.05)
        batch = make_batch(device)
        result, metadata = loss.actor_loss(batch)
        target = torch.linspace(-0.7, 0.8, 12, device=device).reshape(2, 3, 2)
        torch.testing.assert_close(
            metadata["distillation_loss"], (2 - target).square().mean(-1)
        )
        torch.testing.assert_close(
            metadata["q_loss"], torch.full((2, 3), -2.0, device=device)
        )
        result.backward()
        assert (actor.bias.grad > 0).all()

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    @pytest.mark.parametrize(
        "aggregation,normalize,reduction",
        [("mean", False, "mean"), ("min", True, "none")],
    )
    def test_compile(
        self, device, dtype, aggregation, normalize, reduction, fixed_random
    ):
        torch.manual_seed(7)
        kwargs = {
            "q_aggregation": aggregation,
            "normalize_q_loss": normalize,
            "reduction": reduction,
        }
        eager = make_loss(device, dtype, **kwargs)
        compiled = make_loss(device, dtype, **kwargs)
        compiled.load_state_dict(eager.state_dict())
        losses = (eager, compiled)
        updaters = [SoftUpdate(loss, tau=0.05) for loss in losses]
        optimizers = [torch.optim.Adam(loss.parameters(), lr=1e-3) for loss in losses]
        torch._dynamo.reset()
        calls = (eager, torch.compile(compiled, fullgraph=True))
        for step in range(3):
            batch = make_batch(device, dtype)
            if step == 0:
                batch = batch.exclude(("collector", "mask"), "priority_weight")
            if step == 2:
                batch["collector", "mask"].zero_()
                batch.pop("priority_weight")
            results, gradients, priorities = [], [], []
            for loss, call, optimizer, updater in zip(
                losses, calls, optimizers, updaters
            ):
                data = batch.clone()
                result = call(data)
                optimizer.zero_grad(set_to_none=True)
                sum(result[key].sum() for key in loss.out_keys).backward()
                gradients.append(tuple(p.grad.clone() for p in loss.parameters()))
                results.append(result.detach())
                priorities.append(data["td_error"])
                optimizer.step()
                updater.step()
            for left, right in (
                (results[0], results[1]),
                (priorities[0], priorities[1]),
                (gradients[0], gradients[1]),
                (eager.state_dict(), compiled.state_dict()),
            ):
                torch.testing.assert_close(
                    left,
                    right,
                    atol=1e-5 if dtype == torch.float32 else 1e-9,
                    rtol=1e-4 if dtype == torch.float32 else 1e-7,
                )
            if step == 2:
                for result in results:
                    for key in eager.out_keys:
                        torch.testing.assert_close(
                            result[key], torch.zeros_like(result[key])
                        )

    def test_random_compile(self, device):
        torch.manual_seed(7)
        loss = make_loss(device)
        updater = SoftUpdate(loss, tau=0.05)
        optimizer = torch.optim.Adam(loss.parameters(), lr=1e-3)
        initial = {key: value.clone() for key, value in loss.state_dict().items()}
        torch._dynamo.reset()
        call = torch.compile(loss, fullgraph=True)
        batch = make_batch(device)
        first = call(batch.clone())["loss_flow"].detach().clone()
        second = call(batch.clone())["loss_flow"].detach()
        assert not torch.equal(first, second)
        for _ in range(3):
            result = call(batch.clone())
            total = sum(result[key] for key in loss.out_keys)
            optimizer.zero_grad(set_to_none=True)
            total.backward()
            assert all(
                p.grad is not None and p.grad.isfinite().all()
                for p in loss.parameters()
            )
            optimizer.step()
            updater.step()
        for prefix in (
            "flow_policy_params",
            "actor_network_params",
            "qvalue_network_params",
            "target_qvalue_network_params",
        ):
            assert any(
                not torch.equal(value, initial[key])
                for key, value in loss.state_dict().items()
                if key.startswith(prefix)
            )


if __name__ == "__main__":
    pytest.main([__file__])
