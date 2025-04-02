# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import argparse
import importlib.util
import os

import pytest
import torch
import torch.nn.functional as F

from tensordict import TensorDictBase
from tensordict.nn import NormalParamExtractor
from torch import autograd, nn
from torch.utils._pytree import tree_map
from torchrl.modules import (
    OneHotCategorical,
    OneHotOrdinal,
    Ordinal,
    ReparamGradientStrategy,
    TanhNormal,
    TruncatedNormal,
)
from torchrl.modules.distributions import (
    Delta,
    MaskedCategorical,
    MaskedOneHotCategorical,
    TanhDelta,
)
from torchrl.modules.distributions.continuous import SafeTanhTransform
from torchrl.modules.distributions.discrete import _generate_ordinal_logits

if os.getenv("PYTORCH_TEST_FBCODE"):
    from pytorch.rl.test._utils_internal import get_default_devices
else:
    from _utils_internal import get_default_devices

_has_scipy = importlib.util.find_spec("scipy", None) is not None


@pytest.mark.skipif(torch.__version__ < "2.0", reason="torch 2.0 is required")
@pytest.mark.parametrize("device", get_default_devices())
class TestDelta:
    def test_delta_logprob(self, device):
        x = torch.randn(1000000, 4, device=device, dtype=torch.double)
        d = Delta(x)
        assert d.log_prob(d.mode).shape == x.shape[:-1]
        assert (d.log_prob(d.mode) == float("inf")).all()

    @pytest.mark.parametrize("div_up", [1, 2])
    @pytest.mark.parametrize("div_down", [1, 2])
    def test_tanhdelta_logprob(self, device, div_up, div_down):
        x = torch.randn(1000000, 4, device=device, dtype=torch.double)
        d = TanhDelta(x, -1 / div_down, 1.0 / div_up, atol=1e-4, rtol=1e-4)
        xinv = d.transforms[0].inv(d.mode)
        assert d.base_dist._is_equal(xinv).all()
        assert d.log_prob(d.mode).shape == x.shape[:-1]
        assert (d.log_prob(d.mode) == float("inf")).all()

    @pytest.mark.parametrize("div_up", [1, 2])
    @pytest.mark.parametrize("div_down", [1, 2])
    def test_tanhdelta_inv(self, device, div_up, div_down):
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

    def test_tanhdelta_inv_ones(self, device):
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


class TestTanhNormal:
    @pytest.mark.parametrize(
        "low", [-torch.ones(3), -1, 3 * torch.tensor([-1.0, -2.0, -0.5]), -0.1]
    )
    @pytest.mark.parametrize(
        "high", [torch.ones(3), 1, 3 * torch.tensor([1.0, 2.0, 0.5]), 0.1]
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
    def test_tanhnormal(self, low, high, vecs, upscale, shape, device):
        torch.manual_seed(0)
        low, high, vecs, upscale, shape = _map_all(
            low, high, vecs, upscale, shape, device=device
        )
        torch.manual_seed(0)
        d = TanhNormal(
            *vecs,
            upscale=upscale,
            low=low,
            high=high,
        )
        for _ in range(100):
            a = d.rsample(shape)
            assert a.shape[: len(shape)] == shape
            assert (a >= d.low).all()
            assert (a <= d.high).all()
            lp = d.log_prob(a)
            assert torch.isfinite(lp).all()

    def test_tanhnormal_mode(self):
        # Checks that the std of the mode computed by tanh normal is within a certain range
        # when starting from close points

        torch.manual_seed(0)
        # 10 start points with 1000 jitters around that
        # std of the loc is about 1e-4
        loc = torch.randn(10) + torch.randn(1000, 10) / 10000

        t = TanhNormal(loc=loc, scale=0.5, low=-1, high=1, event_dims=0)

        mode = t.get_mode()
        assert mode.shape == loc.shape
        empirical_mode, empirical_mode_lp = torch.zeros_like(loc), -float("inf")
        for v in torch.arange(-1, 1, step=0.01):
            lp = t.log_prob(v.expand_as(t.loc))
            empirical_mode = torch.where(lp > empirical_mode_lp, v, empirical_mode)
            empirical_mode_lp = torch.where(
                lp > empirical_mode_lp, lp, empirical_mode_lp
            )
        assert abs(empirical_mode - mode).max() < 0.1, abs(empirical_mode - mode).max()
        assert mode.shape == loc.shape
        assert (mode.std(0).max() < 0.1).all(), mode.std(0)

    @pytest.mark.parametrize("event_dims", [0, 1, 2])
    def test_tanhnormal_event_dims(self, event_dims):
        scale = 1
        loc = torch.randn(1, 2, 3, 4)
        t = TanhNormal(loc=loc, scale=scale, event_dims=event_dims)
        sample = t.sample()
        assert sample.shape == loc.shape
        exp_shape = loc.shape[:-event_dims] if event_dims > 0 else loc.shape
        assert t.log_prob(sample).shape == exp_shape, (
            t.log_prob(sample).shape,
            event_dims,
            exp_shape,
        )


class TestTruncatedNormal:
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
    def test_truncnormal(self, min, max, vecs, upscale, shape, device):
        torch.manual_seed(0)
        *vecs, min, max, vecs, upscale = tree_map(
            lambda t: torch.as_tensor(t, device=device),
            (*vecs, min, max, vecs, upscale),
        )
        assert all(t.device == device for t in vecs)
        d = TruncatedNormal(
            *vecs,
            upscale=upscale,
            low=min,
            high=max,
        )
        assert d.device == device
        for _ in range(100):
            a = d.rsample(shape)
            assert a.device == device
            assert a.shape[: len(shape)] == shape
            assert (a >= d.low).all()
            assert (a <= d.high).all()
            lp = d.log_prob(a)
            assert torch.isfinite(lp).all()
        oob_min = d.low.expand((*d.batch_shape, *d.event_shape)) - 1e-2
        assert not torch.isfinite(d.log_prob(oob_min)).any()
        oob_max = d.high.expand((*d.batch_shape, *d.event_shape)) + 1e-2
        assert not torch.isfinite(d.log_prob(oob_max)).any()

    @pytest.mark.skipif(not _has_scipy, reason="scipy not installed")
    def test_truncnormal_against_scipy(self):
        from scipy.stats import truncnorm as sp_truncnorm

        # torchrl version
        x = torch.linspace(-1.5, 2.5, 1000).view(-1, 1, 1)
        mu = torch.tensor([0.0], requires_grad=True)
        sigma = torch.tensor([1.0], requires_grad=True)
        high = 2
        low = -1
        log_pi_x = TruncatedNormal(
            mu, sigma, low=low, high=high, tanh_loc=False
        ).log_prob(x)
        pi_x = torch.exp(log_pi_x)
        log_pi_x.backward(torch.ones_like(log_pi_x))
        # plot pi_x
        x_numpy = x.view(-1).detach().numpy()
        pi_x = pi_x.view(-1).detach()

        # scipy version
        sigma = sigma.item()
        mu = mu.item()

        U, L = (high - mu) / sigma, (low - mu) / sigma
        scipy_truncnorm = sp_truncnorm(L, U, loc=mu, scale=sigma)
        pdf_scypi_truncnorm = scipy_truncnorm.pdf(x_numpy)

        torch.testing.assert_close(
            pi_x, torch.as_tensor(pdf_scypi_truncnorm, dtype=torch.float32)
        )

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
    def test_truncnormal_mode(self, min, max, vecs, upscale, shape, device):
        torch.manual_seed(0)
        min, max, vecs, upscale, shape = _map_all(
            min, max, vecs, upscale, shape, device=device
        )
        d = TruncatedNormal(
            *vecs,
            upscale=upscale,
            low=min,
            high=max,
        )
        assert d.mode is not None
        assert d.entropy() is not None
        assert d.mean is not None


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
        if scale_mapping != "raise_error":
            module = nn.Sequential(
                module, NormalParamExtractor(scale_mapping=scale_mapping)
            ).to(device)
            loc, scale = module(torch.randn(*batch_size, state_dim, device=device))
            assert (scale > 0).all()
        else:
            with pytest.raises(
                NotImplementedError, match="Unknown mapping " "raise_error"
            ):
                module = nn.Sequential(
                    module, NormalParamExtractor(scale_mapping=scale_mapping)
                ).to(device)
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

    @pytest.mark.parametrize("neg_inf", [-1e20, float("-inf")])
    @pytest.mark.parametrize("sparse", [False, True])
    @pytest.mark.parametrize("ndim", [2, 1, 3])
    def test_crossentropy(self, sparse: bool, neg_inf: float, ndim: int):
        torch.manual_seed(0)
        logits = torch.randn(4).log_softmax(dim=-1)
        # probs = logits.exp()
        mask = torch.tensor([True, False, True, True])
        indices = torch.tensor([0, 2, 3])

        if ndim >= 2:
            mask = mask.unsqueeze(0)
            logits = logits.unsqueeze(0)
            indices = indices.unsqueeze(0)
        if ndim == 3:
            mask = mask.unsqueeze(0)
            logits = logits.unsqueeze(0)
            indices = indices.unsqueeze(0)

        dist_ce = MaskedCategorical(
            logits=logits,
            neg_inf=neg_inf,
            mask=mask if not sparse else None,
            indices=indices if sparse else None,
            use_cross_entropy=True,
        )
        dist = MaskedCategorical(
            logits=logits,
            neg_inf=neg_inf,
            mask=mask if not sparse else None,
            indices=indices if sparse else None,
            use_cross_entropy=False,
        )
        data = torch.tensor(0)
        if ndim >= 2:
            data = data.unsqueeze(0)
        if ndim == 3:
            data = data.unsqueeze(0)
        torch.testing.assert_close(dist.log_prob(data), dist_ce.log_prob(data))


class TestOneHotCategorical:
    def test_one_hot(self):
        torch.manual_seed(0)
        logits = torch.randn(1, 10)
        torch.manual_seed(0)
        d = OneHotCategorical(logits=logits)
        s_a = d.sample((1,))
        torch.manual_seed(0)
        d = OneHotCategorical(probs=torch.softmax(logits, -1))
        s_b = d.sample((1,))
        torch.testing.assert_close(s_a, s_b)
        assert s_a.dtype == torch.long
        assert s_b.dtype == torch.long
        assert s_a.sum(-1) == 1
        assert s_b.sum(-1) == 1
        assert s_a.shape[-1] == 10
        assert s_b.shape[-1] == 10

    @pytest.mark.parametrize(
        "reparam",
        (ReparamGradientStrategy.PassThrough, ReparamGradientStrategy.RelaxedOneHot),
    )
    def test_reparam(self, reparam):
        torch.manual_seed(0)
        logits = torch.randn(1, 10, requires_grad=True)
        torch.manual_seed(0)
        d = OneHotCategorical(logits=logits, grad_method=reparam)
        s_a = d.rsample((1,))
        torch.manual_seed(0)
        d = OneHotCategorical(probs=torch.softmax(logits, -1), grad_method=reparam)
        s_b = d.rsample((1,))
        s_a[s_a.detach().bool()].sum().backward()
        assert logits.grad is not None and logits.grad.norm() > 0
        logits.grad = None
        s_b[s_b.detach().bool()].sum().backward()
        assert logits.grad is not None and logits.grad.norm() > 0


class TestMaskedOneHotCategorical:
    def test_errs(self):
        with pytest.raises(
            ValueError,
            match="Either `probs` or `logits` must be specified, but not both",
        ):
            MaskedOneHotCategorical(
                logits=torch.tensor(()), probs=torch.tensor(()), mask=torch.tensor(())
            )
        with pytest.raises(ValueError, match="must be provided"):
            MaskedOneHotCategorical(probs=torch.tensor(()), mask=None)
        with pytest.raises(ValueError, match="must be provided"):
            MaskedOneHotCategorical(
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
        dist = MaskedOneHotCategorical(
            logits=logits, probs=probs, indices=indices, mask=mask, neg_inf=neg_inf
        )
        dist_categ = MaskedCategorical(
            logits=logits, probs=probs, indices=indices, mask=mask, neg_inf=neg_inf
        )
        for _ in range(10):
            sample = dist.sample((100,))
            assert not sample[..., 1].any()
            assert torch.isfinite(dist.log_prob(sample)).all()
            torch.testing.assert_close(
                dist.log_prob(sample), dist_categ.log_prob(sample.argmax(-1))
            )
            assert sample.device == device

        sample_unfeasible = torch.zeros_like(sample)
        sample_unfeasible[..., 1] = 1
        if neg_inf == -float("inf"):
            assert (dist.log_prob(sample_unfeasible) == neg_inf).all()
        else:
            assert (dist.log_prob(sample_unfeasible) > -float("inf")).all()

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
        dist = MaskedOneHotCategorical(
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

        dist = MaskedOneHotCategorical(
            probs=probs,
            mask=mask,
            neg_inf=neg_inf,
        )
        num_samples = 10000
        samples = dist.sample([num_samples]).argmax(-1)
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

        dist = MaskedOneHotCategorical(
            logits=logits,
            indices=indices,
            neg_inf=neg_inf,
        )
        num_samples = 10000
        samples = dist.sample([num_samples]).argmax(-1)
        sample_probs = torch.bincount(samples) / num_samples
        torch.testing.assert_close(sample_probs, ref_probs, rtol=1e-5, atol=1e-2)

    @pytest.mark.parametrize(
        "grad_method",
        [ReparamGradientStrategy.RelaxedOneHot, ReparamGradientStrategy.PassThrough],
    )
    @pytest.mark.parametrize("sparse", [True, False])
    def test_reparam(self, grad_method, sparse):
        torch.manual_seed(0)
        neg_inf = -float("inf")
        logits = torch.randn(100, requires_grad=True)
        probs = F.softmax(logits, dim=-1)
        # mask = torch.tensor([True, False, True, True])
        # indices = torch.tensor([0, 2, 3])
        if sparse:
            indices = torch.randint(100, (70,)).unique().view(-1)
            mask = None
        else:
            mask = torch.zeros(100, dtype=torch.bool).bernoulli_()
            indices = None

        dist = MaskedOneHotCategorical(
            logits=logits,
            indices=indices,
            neg_inf=neg_inf,
            grad_method=grad_method,
            mask=mask,
        )

        s = dist.rsample()
        assert s.shape[-1] == 100
        s[s.detach().bool()].sum().backward()
        assert logits.grad is not None and logits.grad.norm() > 0


class TestOrdinal:
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("logit_shape", [(10,), (1, 1), (10, 10), (5, 10, 20)])
    def test_correct_sampling_shape(
        self, logit_shape: tuple[int, ...], dtype: torch.dtype, device: str
    ) -> None:
        logits = torch.testing.make_tensor(logit_shape, dtype=dtype, device=device)

        sampler = Ordinal(scores=logits)
        actions = sampler.sample()  # type: ignore[no-untyped-call]
        log_probs = sampler.log_prob(actions)  # type: ignore[no-untyped-call]

        expected_log_prob_shape = logit_shape[:-1]
        expected_action_shape = logit_shape[:-1]

        assert actions.size() == torch.Size(expected_action_shape)
        assert log_probs.size() == torch.Size(expected_log_prob_shape)

    @pytest.mark.parametrize("num_categories", [1, 10, 20])
    def test_correct_range(self, num_categories: int) -> None:
        seq_size = 10
        batch_size = 100
        logits = torch.ones((batch_size, seq_size, num_categories))

        sampler = Ordinal(scores=logits)

        actions = sampler.sample()  # type: ignore[no-untyped-call]

        assert actions.min() >= 0
        assert actions.max() < num_categories

    def test_bounded_gradients(self) -> None:
        logits = torch.tensor(
            [[1.0, 0.0, torch.finfo().max], [1.0, 0.0, torch.finfo().min]],
            requires_grad=True,
            dtype=torch.float32,
        )

        sampler = Ordinal(scores=logits)

        actions = sampler.sample()
        log_probs = sampler.log_prob(actions)

        dummy_objective = log_probs.sum()
        dummy_objective.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_generate_ordinal_logits_numerical(self) -> None:
        logits = torch.ones((3, 4))

        ordinal_logits = _generate_ordinal_logits(scores=logits)

        expected_ordinal_logits = torch.tensor(
            [
                [-4.2530, -3.2530, -2.2530, -1.2530],
                [-4.2530, -3.2530, -2.2530, -1.2530],
                [-4.2530, -3.2530, -2.2530, -1.2530],
            ]
        )

        torch.testing.assert_close(
            ordinal_logits, expected_ordinal_logits, atol=1e-4, rtol=1e-6
        )


class TestOneHotOrdinal:
    @pytest.mark.parametrize("dtype", [torch.float, torch.double])
    @pytest.mark.parametrize("device", get_default_devices())
    @pytest.mark.parametrize("logit_shape", [(10,), (10, 10), (5, 10, 20)])
    def test_correct_sampling_shape(
        self, logit_shape: tuple[int, ...], dtype: torch.dtype, device: str
    ) -> None:
        logits = torch.testing.make_tensor(logit_shape, dtype=dtype, device=device)

        sampler = OneHotOrdinal(scores=logits)
        actions = sampler.sample()  # type: ignore[no-untyped-call]
        log_probs = sampler.log_prob(actions)  # type: ignore[no-untyped-call]
        expected_log_prob_shape = logit_shape[:-1]

        expected_action_shape = logit_shape

        assert actions.size() == torch.Size(expected_action_shape)
        assert log_probs.size() == torch.Size(expected_log_prob_shape)

    @pytest.mark.parametrize("num_categories", [2, 10, 20])
    def test_correct_range(self, num_categories: int) -> None:
        seq_size = 10
        batch_size = 100
        logits = torch.ones((batch_size, seq_size, num_categories))

        sampler = OneHotOrdinal(scores=logits)

        actions = sampler.sample()  # type: ignore[no-untyped-call]

        assert torch.all(actions.sum(-1))
        assert actions.shape[-1] == num_categories

    def test_bounded_gradients(self) -> None:
        logits = torch.tensor(
            [[1.0, 0.0, torch.finfo().max], [1.0, 0.0, torch.finfo().min]],
            requires_grad=True,
            dtype=torch.float32,
        )

        sampler = OneHotOrdinal(scores=logits)

        actions = sampler.sample()
        log_probs = sampler.log_prob(actions)

        dummy_objective = log_probs.sum()
        dummy_objective.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)
