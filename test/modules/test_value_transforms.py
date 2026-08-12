# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import pytest
import torch

import torchrl.modules as modules
from torchrl.modules import (
    ComposeValueTransform,
    functional as F,
    IdentityValueTransform,
    SignedHyperbolicValueTransform,
    SymLogValueTransform,
    ValueTransform,
)


@pytest.mark.parametrize(
    ("transform", "atol", "rtol"),
    [
        (IdentityValueTransform(), 0.0, 0.0),
        (SymLogValueTransform(), 1e-10, 1e-10),
        (SignedHyperbolicValueTransform(), 1e-9, 1e-9),
        (
            ComposeValueTransform(
                SignedHyperbolicValueTransform(), SymLogValueTransform()
            ),
            1e-8,
            1e-8,
        ),
    ],
)
def test_value_transform_round_trip(transform, atol, rtol):
    value = torch.tensor(
        [-1e6, -100.0, -1.0, -1e-6, 0.0, 1e-6, 1.0, 100.0, 1e6],
        dtype=torch.float64,
    )

    transformed = transform(value)
    reconstructed = transform.inverse(transformed)

    torch.testing.assert_close(reconstructed, value, atol=atol, rtol=rtol)
    assert transformed.shape == value.shape
    assert transformed.dtype == value.dtype
    assert transformed.device == value.device


@pytest.mark.parametrize(
    "transform", [SymLogValueTransform(), SignedHyperbolicValueTransform()]
)
def test_value_transform_is_monotonic_and_compresses(transform):
    value = torch.linspace(-1e4, 1e4, 1001, dtype=torch.float64)
    transformed = transform(value)

    assert torch.all(transformed.diff() > 0)
    assert transformed[0].abs() < value[0].abs()
    assert transformed[-1].abs() < value[-1].abs()
    torch.testing.assert_close(transformed, -transform(-value))


def test_value_transform_functional_api():
    value = torch.tensor([-100.0, -1.0, 0.0, 1.0, 100.0])

    torch.testing.assert_close(SymLogValueTransform()(value), F.symlog(value))
    torch.testing.assert_close(F.symexp(F.symlog(value)), value)
    torch.testing.assert_close(
        SignedHyperbolicValueTransform()(value), F.signed_hyperbolic(value)
    )
    torch.testing.assert_close(F.signed_parabolic(F.signed_hyperbolic(value)), value)
    assert modules.symlog is F.symlog
    assert modules.symexp is F.symexp
    assert modules.signed_hyperbolic is F.signed_hyperbolic
    assert modules.signed_parabolic is F.signed_parabolic


def test_compose_value_transform_order():
    first = SignedHyperbolicValueTransform(epsilon=1e-2)
    second = SymLogValueTransform()
    transform = ComposeValueTransform(first, second)
    value = torch.tensor([-10.0, 0.0, 10.0])

    torch.testing.assert_close(transform(value), second(first(value)))
    torch.testing.assert_close(
        transform.inverse(transform(value)),
        first.inverse(second.inverse(transform(value))),
    )


def test_value_transform_gradients():
    value = torch.tensor([-100.0, -1.0, -0.1, 0.0, 0.1, 1.0, 100.0], requires_grad=True)
    transform = ComposeValueTransform(
        SignedHyperbolicValueTransform(), SymLogValueTransform()
    )

    transform(value).sum().backward()

    assert value.grad is not None
    assert torch.isfinite(value.grad).all()
    assert (value.grad > 0).all()


def test_value_transform_errors():
    with pytest.raises(TypeError):
        ValueTransform()
    with pytest.raises(ValueError, match="epsilon must be positive"):
        SignedHyperbolicValueTransform(epsilon=0)
    with pytest.raises(ValueError, match="epsilon must be positive"):
        F.signed_hyperbolic(torch.zeros(1), epsilon=-1)
    with pytest.raises(ValueError, match="at least one"):
        ComposeValueTransform()
    with pytest.raises(TypeError, match="ValueTransform"):
        ComposeValueTransform(torch.nn.Identity())


def test_value_transform_compile():
    transform = ComposeValueTransform(
        SignedHyperbolicValueTransform(), SymLogValueTransform()
    )
    value = torch.tensor([-100.0, 0.0, 100.0])

    compiled = torch.compile(transform, backend="eager", fullgraph=True)
    compiled_inverse = torch.compile(transform.inverse, backend="eager", fullgraph=True)

    torch.testing.assert_close(compiled(value), transform(value))
    torch.testing.assert_close(compiled_inverse(compiled(value)), value)


if __name__ == "__main__":
    pytest.main([__file__])
