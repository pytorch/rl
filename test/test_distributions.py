# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import argparse

import pytest
import torch
import torch.nn.functional as F

from _utils_internal import get_default_devices
from tensordict.tensordict import TensorDictBase
from torch import autograd, nn
from torchrl.modules import (
    NormalParamWrapper,
    OneHotCategorical,
    TanhNormal,
    TruncatedNormal,
)
from torchrl.modules.distributions import Delta, MaskedCategorical, TanhDelta
from torchrl.modules.distributions.continuous import SafeTanhTransform


@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize("div_up", [1, 2])
@pytest.mark.parametrize("div_down", [1, 2])
def test_delta(device, div_up, div_down):
    x = torch.randn(1000000, 4, device=device, dtype=torch.double)
    d = Delta(x)
    assert d.log_prob(d.mode).shape == x.shape[:-1]
    assert (d.log_prob(d.mode) == float("inf")).all()

    x = torch.randn(1000000, 4, device=device, dtype=torch.double)
    d = TanhDelta(x, -1 / div_down, 1.0 / div_up, atol=1e-4, rtol=1e-4)
    xinv = d.transforms[0].inv(d.mode)
    assert d.base_dist._is_equal(xinv).all()
    assert d.log_prob(d.mode).shape == x.shape[:-1]
    assert (d.log_prob(d.mode) == float("inf")).all()

    x = torch.randn(1000000, 4, device=device, dtype=torch.double)
    d = TanhDelta(
        x,
        -torch.ones_like(x) / div_down,
        torch.ones_like(x) / div_up,
        atol=1e-4,
        rtol=1e-4,
    )
    xinv = d.transforms[0].inv(d.mode)
    assert d.base_dist._is_equal(xinv).all()
    assert d.log_prob(d.mode).shape == x.shape[:-1]
    assert (d.log_prob(d.mode) == float("inf")).all()

    x = torch.randn(1000000, 4, device=device)
    d = TanhDelta(x, -torch.ones_like(x), torch.ones_like(x), atol=1e-4, rtol=1e-4)
    xinv = d.transforms[0].inv(d.mode)
    assert d.base_dist._is_equal(xinv).all()
    assert d.log_prob(d.mode).shape == x.shape[:-1]
    assert (d.log_prob(d.mode) == float("inf")).all()


def _map_all(*tensors_or_other, device):
    for t in tensors_or_other:
        if isinstance(t, (torch.Tensor, TensorDictBase)):
            yield t.to(device)
        else:
            yield t


@pytest.mark.parametrize(
    "min", [-torch.ones(3), -1, 3 * torch.tensor([-1.0, -2.0, -0.5]), -0.1]
)
@pytest.mark.parametrize(
    "max", [torch.ones(3), 1, 3 * torch.tensor([1.0, 2.0, 0.5]), 0.1]
)
@pytest.mark.parametrize(
    "vecs",
    [
        (torch.tensor([0.1, 10.0, 5.0]), torch.tensor([0.1, 10.0, 5.0])),
        (torch.zeros(7, 3), torch.ones(7, 3)),
    ],
)
@pytest.mark.parametrize(
    "upscale", [torch.ones(3), 1, 3 * torch.tensor([1.0, 2.0, 0.5]), 3]
)
@pytest.mark.parametrize("shape", [torch.Size([]), torch.Size([3, 4])])
@pytest.mark.parametrize("device", get_default_devices())
def test_tanhnormal(min, max, vecs, upscale, shape, device):
    min, max, vecs, upscale, shape = _map_all(
        min, max, vecs, upscale, shape, device=device
    )
    torch.manual_seed(0)
    d = TanhNormal(
        *vecs,
        upscale=upscale,
        min=min,
        max=max,
    )
    for _ in range(100):
        a = d.rsample(shape)
        assert a.shape[: len(shape)] == shape
        assert (a >= d.min).all()
        assert (a <= d.max).all()
        lp = d.log_prob(a)
        assert torch.isfinite(lp).all()


@pytest.mark.parametrize(
    "min", [-torch.ones(3), -1, 3 * torch.tensor([-1.0, -2.0, -0.5]), -0.1]
)
@pytest.mark.parametrize(
    "max", [torch.ones(3), 1, 3 * torch.tensor([1.0, 2.0, 0.5]), 0.1]
)
@pytest.mark.parametrize(
    "vecs",
    [
        (torch.tensor([0.1, 10.0, 5.0]), torch.tensor([0.1, 10.0, 5.0])),
        (torch.zeros(7, 3), torch.ones(7, 3)),
    ],
)
@pytest.mark.parametrize(
    "upscale", [torch.ones(3), 1, 3 * torch.tensor([1.0, 2.0, 0.5]), 3]
)
@pytest.mark.parametrize("shape", [torch.Size([]), torch.Size([3, 4])])
@pytest.mark.parametrize("device", get_default_devices())
def test_truncnormal(min, max, vecs, upscale, shape, device):
    torch.manual_seed(0)
    min, max, vecs, upscale, shape = _map_all(
        min, max, vecs, upscale, shape, device=device
    )
    d = TruncatedNormal(
        *vecs,
        upscale=upscale,
        min=min,
        max=max,
    )
    for _ in range(100):
        a = d.rsample(shape)
        assert a.shape[: len(shape)] == shape
        assert (a >= d.min).all()
        assert (a <= d.max).all()
        lp = d.log_prob(a)
        assert torch.isfinite(lp).all()


@pytest.mark.parametrize(
    "batch_size",
    [
        (3,),
        (
            5,
            7,
        ),
    ],
)
@pytest.mark.parametrize("device", get_default_devices())
@pytest.mark.parametrize(
    "scale_mapping",
    [
        "exp",
        "biased_softplus_1.0",
        "biased_softplus_0.11",
        "biased_softplus_1.0_1e-6",
        "expln",
        "relu",
        "softplus",
        "raise_error",
    ],
)
def test_normal_mapping(batch_size, device, scale_mapping, action_dim=11, state_dim=3):
    torch.manual_seed(0)
    for _ in range(100):
        module = nn.LazyLinear(2 * action_dim).to(device)
        module = NormalParamWrapper(module, scale_mapping=scale_mapping).to(device)
        if scale_mapping != "raise_error":
            loc, scale = module(torch.randn(*batch_size, state_dim, device=device))
            assert (scale > 0).all()
        else:
            with pytest.raises(
                NotImplementedError, match="Unknown mapping " "raise_error"
            ):
                loc, scale = module(torch.randn(*batch_size, state_dim, device=device))


@pytest.mark.parametrize("shape", [torch.Size([]), torch.Size([3, 4])])
@pytest.mark.parametrize("device", get_default_devices())
def test_categorical(shape, device):
    torch.manual_seed(0)
    for i in range(100):
        logits = i * torch.randn(10)
        dist = OneHotCategorical(logits=logits)
        s = dist.sample(shape)
        assert s.shape[: len(shape)] == shape
        assert s.shape[-1] == logits.shape[-1]
        assert (s.sum(-1) == 1).all()
        assert torch.isfinite(dist.log_prob(s)).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_tanhtrsf(dtype):
    torch.manual_seed(0)
    trsf = SafeTanhTransform()
    some_big_number = (
        torch.randn(10, dtype=dtype).sign() * torch.randn(10, dtype=dtype).pow(2) * 1e6
    )
    some_other_number = trsf(some_big_number)
    assert torch.isfinite(some_other_number).all()
    assert (some_big_number.sign() == some_other_number.sign()).all()

    ones = torch.ones(2, dtype=dtype)
    ones[1] = -1
    some_big_number = trsf.inv(ones)
    assert torch.isfinite(some_big_number).all()
    assert (some_big_number.sign() == ones.sign()).all()


@pytest.mark.parametrize("dtype", [torch.float, torch.double])
def test_tanhtrsf_grad(dtype):
    torch.manual_seed(0)
    trsf = SafeTanhTransform()
    x = torch.randn(100, requires_grad=True)
    y1 = trsf(x)
    y2 = x.tanh()
    g1 = autograd.grad(y1.sum(), x, retain_graph=True)[0]
    g2 = autograd.grad(y2.sum(), x, retain_graph=True)[0]
    torch.testing.assert_close(g1, g2)


class TestMaskedCategorical:
    def test_errs(self):
        with pytest.raises(
            ValueError,
            match="Either `probs` or `logits` must be specified, but not both",
        ):
            MaskedCategorical(
                logits=torch.tensor(()), probs=torch.tensor(()), mask=torch.tensor(())
            )
        with pytest.raises(ValueError, match="must be provided"):
            MaskedCategorical(probs=torch.tensor(()), mask=None)
        with pytest.raises(ValueError, match="must be provided"):
            MaskedCategorical(
                probs=torch.tensor(()), mask=torch.tensor(()), indices=torch.tensor(())
            )

    @pytest.mark.parametrize("neg_inf", [-float(10.0), -float("inf")])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("sparse", [True, False])
    @pytest.mark.parametrize("logits", [True, False])
    def test_construction(self, neg_inf, sparse, logits, device):
        torch.manual_seed(0)
        logits_vals = torch.randn(4, device=device) / 100  # almost equal probabilities
        if logits:
            logits = logits_vals
            probs = None
        else:
            probs = logits_vals.softmax(-1)
            logits = None

        if sparse:
            indices = torch.tensor([0, 2, 3], device=device)
            mask = None
        else:
            mask = torch.tensor([True, False, True, True], device=device)
            indices = None
        dist = MaskedCategorical(
            logits=logits, probs=probs, indices=indices, mask=mask, neg_inf=neg_inf
        )
        for _ in range(10):
            sample = dist.sample((100,))
            assert not (sample == 1).any()
            assert torch.isfinite(dist.log_prob(sample)).all()
            assert sample.device == device

        if neg_inf == -float("inf"):
            assert (dist.log_prob(torch.ones_like(sample)) == neg_inf).all()
        else:
            assert (dist.log_prob(torch.ones_like(sample)) > -float("inf")).all()

    @pytest.mark.parametrize("neg_inf", [-float(10.0), -float("inf")])
    @pytest.mark.parametrize("sparse", [True, False])
    @pytest.mark.parametrize("logits", [True, False])
    def test_backprop(self, neg_inf, sparse, logits):
        torch.manual_seed(0)
        logits_vals = (
            torch.randn(4).div_(100).requires_grad_()
        )  # almost equal probabilities
        if logits:
            logits = logits_vals
            probs = None
        else:
            probs = logits_vals.softmax(-1)
            logits = None

        if sparse:
            indices = torch.tensor([0, 2, 3])
            mask = None
        else:
            mask = torch.tensor([True, False, True, True])
            indices = None
        dist = MaskedCategorical(
            logits=logits, probs=probs, indices=indices, mask=mask, neg_inf=neg_inf
        )
        sample = dist.sample((100,))
        lp = dist.log_prob(sample)
        lp.sum().backward()
        assert logits_vals.grad is not None

    @pytest.mark.parametrize("neg_inf", [-1e20, float("-inf")])
    def test_sample(self, neg_inf: float) -> None:
        torch.manual_seed(0)
        logits = torch.randn(4)
        probs = F.softmax(logits, dim=-1)
        mask = torch.tensor([True, False, True, True])
        ref_probs = probs.masked_fill(~mask, 0.0)
        ref_probs /= ref_probs.sum(dim=-1, keepdim=True)

        dist = MaskedCategorical(
            probs=probs,
            mask=mask,
            neg_inf=neg_inf,
        )
        num_samples = 10000
        samples = dist.sample([num_samples])
        sample_probs = torch.bincount(samples) / num_samples
        torch.testing.assert_close(sample_probs, ref_probs, rtol=1e-5, atol=1e-2)

    @pytest.mark.parametrize("neg_inf", [-1e20, float("-inf")])
    def test_sample_sparse(self, neg_inf: float) -> None:
        torch.manual_seed(0)
        logits = torch.randn(4)
        probs = F.softmax(logits, dim=-1)
        mask = torch.tensor([True, False, True, True])
        indices = torch.tensor([0, 2, 3])
        ref_probs = probs.masked_fill(~mask, 0.0)
        ref_probs /= ref_probs.sum(dim=-1, keepdim=True)

        dist = MaskedCategorical(
            logits=logits,
            indices=indices,
            neg_inf=neg_inf,
        )
        num_samples = 10000
        samples = dist.sample([num_samples])
        sample_probs = torch.bincount(samples) / num_samples
        torch.testing.assert_close(sample_probs, ref_probs, rtol=1e-5, atol=1e-2)


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
