import pytest
import torch

from torchrl.modules import TanhNormal
from torchrl.modules.distributions import TanhDelta, Delta


def test_delta():
    x = torch.randn(1000000, 4)
    d = Delta(x)
    assert d.log_prob(d.mode).shape == x.shape[:-1]
    assert (d.log_prob(d.mode) == float('inf')).all()

    x = torch.randn(1000000, 4)
    d = TanhDelta(x, -1, 1.0, atol=1e-4, rtol=1e-4)
    xinv = d.transforms[0].inv(d.mode)
    assert d.base_dist._is_equal(xinv).all()
    assert d.log_prob(d.mode).shape == x.shape[:-1]
    assert (d.log_prob(d.mode) == float('inf')).all()


def test_categorical():
    raise NotImplementedError


@pytest.mark.parametrize("min", [-torch.ones(3), -1, 3 * torch.tensor([-1.0, -2.0, -0.5]), -3])
@pytest.mark.parametrize("max", [torch.ones(3), 1, 3 * torch.tensor([1.0, 2.0, 0.5]), 3])
@pytest.mark.parametrize("vec", [torch.tensor([0.1, 1.0, 10.0, 100.0, 5.0, 0.01]), torch.zeros(3, 6)])
@pytest.mark.parametrize("upscale", [torch.ones(3), 1, 3 * torch.tensor([1.0, 2.0, 0.5]), 3])
@pytest.mark.parametrize("scale_mapping", ["biased_softplus_1.0", "biased_softplus_0.1", "exp"])
@pytest.mark.parametrize("shape", [torch.Size([]), torch.Size([3, 4])])
def test_tanhnormal(min, max, vec, upscale, scale_mapping, shape):
    torch.manual_seed(0)
    d = TanhNormal(vec, upscale, min, max, scale_mapping)
    for _ in range(100):
        a = d.rsample(shape)
        assert a.shape[:len(shape)] == shape
        assert (a >= d.min).all()
        assert (a <= d.max).all()


if __name__ == "__main__":
    pytest.main([__file__])
