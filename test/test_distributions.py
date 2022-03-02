import argparse

import pytest
import torch

from torch import distributions as D
from torchrl.modules import TanhNormal, NormalLogScale, MultivariateNormalCholesky
from torchrl.modules.distributions import TanhDelta, Delta


def test_delta():
    x = torch.randn(1000000, 4)
    d = Delta(x)
    assert d.log_prob(d.mode).shape == x.shape[:-1]
    assert (d.log_prob(d.mode) == float("inf")).all()

    x = torch.randn(1000000, 4)
    d = TanhDelta(x, -1, 1.0, atol=1e-4, rtol=1e-4)
    xinv = d.transforms[0].inv(d.mode)
    assert d.base_dist._is_equal(xinv).all()
    assert d.log_prob(d.mode).shape == x.shape[:-1]
    assert (d.log_prob(d.mode) == float("inf")).all()


def test_categorical():
    raise NotImplementedError


@pytest.mark.parametrize(
    "min", [-torch.ones(3), -1, 3 * torch.tensor([-1.0, -2.0, -0.5]), -3]
)
@pytest.mark.parametrize(
    "max", [torch.ones(3), 1, 3 * torch.tensor([1.0, 2.0, 0.5]), 3]
)
@pytest.mark.parametrize(
    "vec", [torch.tensor([0.1, 1.0, 10.0, 100.0, 5.0, 0.01]), torch.zeros(3, 6)]
)
@pytest.mark.parametrize(
    "upscale", [torch.ones(3), 1, 3 * torch.tensor([1.0, 2.0, 0.5]), 3]
)
@pytest.mark.parametrize(
    "scale_mapping", ["biased_softplus_1.0", "biased_softplus_0.1", "exp"]
)
@pytest.mark.parametrize("shape", [torch.Size([]), torch.Size([3, 4])])
def test_tanhnormal(min, max, vec, upscale, scale_mapping, shape):
    torch.manual_seed(0)
    d = TanhNormal(vec, upscale, min, max, scale_mapping)
    for _ in range(100):
        a = d.rsample(shape)
        assert a.shape[: len(shape)] == shape
        assert (a >= d.min).all()
        assert (a <= d.max).all()

def test_normal():
    vector = torch.randn(3, 4)
    normal = D.Normal(vector[..., :2], vector[..., 2:].exp())
    normal_log_scale = NormalLogScale(vector)
    x = torch.randn(10, 3, 2)
    lp1 = normal.log_prob(x)
    lp2 = normal_log_scale.log_prob(x)
    torch.testing.assert_allclose(lp1, lp2)

def test_mv_normal_chol():
    dim = 4
    vector = torch.randn(11, 14, dtype=torch.double)
    v = vector[..., 4:]
    L = torch.zeros(11, 4, 4, dtype=torch.double)
    idx = torch.ones(4, 4, dtype=torch.bool).tril()
    L[idx.expand_as(L)] = v.reshape(-1)
    diag_elts = L[torch.eye(dim, device=L.device, dtype=torch.bool).expand_as(L)]
    L[torch.eye(dim, device=L.device, dtype=torch.bool).expand_as(L)] = diag_elts.exp()

    cov = L @ L.transpose(-2, -1)
    dist1 = D.MultivariateNormal(vector[:, :4], cov)
    dist2 = MultivariateNormalCholesky(vector)
    x = torch.randn(11, 4, dtype=torch.double)
    torch.testing.assert_allclose(dist1.log_prob(x), dist2.log_prob(x))


if __name__ == "__main__":
    args, unknown = argparse.ArgumentParser().parse_known_args()
    pytest.main([__file__, "--capture", "no", "--exitfirst"] + unknown)

