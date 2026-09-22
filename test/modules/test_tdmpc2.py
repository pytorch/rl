# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import pytest
import torch
from tensordict import TensorDict
from torch import nn

from torchrl.modules import SimplicialNormalization, TdMpc2QEnsemble


def _make_ensemble(
    *,
    num_q=5,
    num_bins=5,
    vmin=-10.0,
    vmax=10.0,
    module_factory=None,
):
    if module_factory is None:
        module_factory = lambda: nn.Linear(3, max(num_bins, 1))
    return TdMpc2QEnsemble(
        [module_factory() for _ in range(num_q)],
        num_bins=num_bins,
        vmin=vmin,
        vmax=vmax,
    )


class TestTdMpc2QEnsemble:
    def test_forward(self):
        batch_size = (2, 3)
        module = _make_ensemble()
        td = TensorDict(
            {
                "latent": torch.randn(*batch_size, 2),
                "action": torch.randn(*batch_size, 1),
                "other": torch.ones(batch_size),
            },
            batch_size=batch_size,
        )

        output = module(td)

        assert output is td
        assert output["q_logits"].shape == (*batch_size, 5, 5)
        assert torch.equal(output["other"], torch.ones(batch_size))

    def test_nested_keys(self):
        module = TdMpc2QEnsemble(
            [nn.Linear(3, 5), nn.Linear(3, 5)],
            num_bins=5,
            vmin=-10.0,
            vmax=10.0,
            in_keys=[("inputs", "latent"), ("inputs", "action")],
            out_keys=[("outputs", "q_logits")],
            q_value_key=("outputs", "q_value"),
        )
        td = TensorDict(
            {
                ("inputs", "latent"): torch.zeros(4, 2),
                ("inputs", "action"): torch.zeros(4, 1),
            },
            batch_size=[4],
        )

        online = module(td.clone(), source="online")
        detached = module(td.clone(), source="detached")
        target = module(td.clone(), source="target")

        assert online["outputs", "q_logits"].shape == (4, 2, 5)
        assert torch.equal(
            online["outputs", "q_logits"], detached["outputs", "q_logits"]
        )
        assert torch.equal(online["outputs", "q_logits"], target["outputs", "q_logits"])

        reduced = module.reduce(td, reduction="min")
        assert reduced["outputs", "q_value"].shape == (4, 1)

    def test_sources(self):
        module = _make_ensemble(num_q=2)
        td = TensorDict(
            {
                "latent": torch.randn(4, 2, requires_grad=True),
                "action": torch.randn(4, 1),
            },
            batch_size=[4],
        )

        module(td, source="detached")["q_logits"].sum().backward()
        assert td["latent"].grad is not None
        assert all(parameter.grad is None for parameter in module.q_params.parameters())

        with torch.no_grad():
            for parameter in module.q_params.parameters():
                parameter.add_(1.0)
        online = module(td.detach().clone(), source="online")["q_logits"]
        target = module(td.detach().clone(), source="target")["q_logits"]
        assert not torch.equal(online, target)

    def test_reduction(self):
        module = _make_ensemble(num_q=2, num_bins=5)
        with torch.no_grad():
            for parameter in module.q_params.parameters():
                parameter.zero_()
            bias = module.q_params.data.get("bias")
            bias.copy_(
                torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 10.0]])
            )

        td = TensorDict(
            {"latent": torch.zeros(3, 2), "action": torch.zeros(3, 1)},
            batch_size=[3],
        )
        output = module.reduce(td, reduction="avg")
        probabilities = torch.softmax(torch.tensor([10.0, 0.0, 0.0, 0.0, 0.0]), dim=-1)
        first = (probabilities * torch.linspace(-10.0, 10.0, 5)).sum()
        probabilities = torch.softmax(torch.tensor([0.0, 0.0, 0.0, 0.0, 10.0]), dim=-1)
        second = (probabilities * torch.linspace(-10.0, 10.0, 5)).sum()
        expected_value = (
            torch.sign(first) * torch.expm1(first.abs())
            + torch.sign(second) * torch.expm1(second.abs())
        ) / 2
        expected = expected_value.expand(3, 1)
        torch.testing.assert_close(output["q_value"], expected)

    def test_target_update(self):
        module = _make_ensemble(num_q=2)
        target_before = {
            key: value.clone()
            for key, value in module.target_q_params.data.items(True, True)
        }
        with torch.no_grad():
            for parameter in module.q_params.parameters():
                parameter.add_(2.0)

        module.soft_update_target(0.25)

        for key, target in module.target_q_params.data.items(True, True):
            expected = target_before[key] * 0.75 + module.q_params.data.get(key) * 0.25
            torch.testing.assert_close(target, expected)

    def test_target_integer_buffers(self):
        module = _make_ensemble(
            num_q=2,
            module_factory=lambda: nn.Sequential(
                nn.Linear(3, 4), nn.BatchNorm1d(4), nn.Linear(4, 5)
            ),
        )
        module.soft_update_target(0.5)
        for key, target in module.target_q_params.data.items(True, True):
            source = module.q_params.data.get(key)
            torch.testing.assert_close(target, source)

    def test_dtype_and_gradients(self):
        module = _make_ensemble(num_q=2).to(dtype=torch.float64)
        td = TensorDict(
            {
                "latent": torch.randn(4, 2, dtype=torch.float64, requires_grad=True),
                "action": torch.randn(4, 1, dtype=torch.float64),
            },
            batch_size=[4],
        )

        output = module(td)["q_logits"]
        output.sum().backward()

        assert output.dtype is torch.float64
        assert all(
            parameter.grad is not None for parameter in module.q_params.parameters()
        )
        assert all(
            parameter.dtype is torch.float64
            for parameter in module.q_params.parameters()
        )
        assert all(
            parameter.dtype is torch.float64
            for parameter in module.target_q_params.buffers()
        )

    def test_state_dict(self):
        module = _make_ensemble(num_q=2)
        with torch.no_grad():
            module.target_q_params.data.get("bias").add_(1.0)
            module.q_params.data.get("weight").mul_(2.0)

        restored = _make_ensemble(num_q=2)
        restored.load_state_dict(module.state_dict())

        for key, value in module.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], value)
        assert not torch.equal(
            restored.q_params.data.get("bias"),
            restored.target_q_params.data.get("bias"),
        )

    def test_validation(self):
        with pytest.raises(ValueError, match="num_bins"):
            _make_ensemble(num_q=2, num_bins=1)

        module = _make_ensemble(num_q=2)
        td = TensorDict({"latent": torch.randn(2, 2)}, batch_size=[2])

        with pytest.raises(KeyError):
            module(td)
        with pytest.raises(ValueError, match="reduction"):
            module.reduce(
                TensorDict(
                    {"latent": torch.randn(2, 2), "action": torch.randn(2, 1)},
                    batch_size=[2],
                ),
                reduction="median",
            )

    def test_invalid_source(self):
        module = _make_ensemble(num_q=2)
        td = TensorDict(
            {"latent": torch.randn(2, 2), "action": torch.randn(2, 1)},
            batch_size=[2],
        )
        with pytest.raises(ValueError, match="source"):
            module(td, source="invalid")


class TestSimplicialNormalization:
    def test_groups(self):
        module = SimplicialNormalization(2)
        output = module(torch.zeros(3, 4))

        assert output.shape == (3, 4)
        torch.testing.assert_close(
            output.reshape(3, 2, 2).sum(-1),
            torch.ones(3, 2),
        )

    def test_invalid_features(self):
        with pytest.raises(ValueError, match="divisible"):
            SimplicialNormalization(3)(torch.zeros(2, 4))


if __name__ == "__main__":
    pytest.main()
