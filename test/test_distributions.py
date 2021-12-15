from torchrl.modules.distributions import TanhDelta, Delta
import pytest
import torch

def test_delta():
    x = torch.randn(1000000,4)
    d = Delta(x)
    assert d.log_prob(d.mode).shape == x.shape[:-1]
    assert (d.log_prob(d.mode) == float('inf')).all()

    x = torch.randn(1000000,4)
    d = TanhDelta(x, -1, 1.0, atol=1e-4, rtol=1e-4)
    xinv = d.transforms[0].inv(d.mode)
    assert d.base_dist._is_equal(xinv).all()
    assert d.log_prob(d.mode).shape == x.shape[:-1]
    assert (d.log_prob(d.mode) == float('inf')).all()


def test_categorical():
    raise NotImplementedError


def test_tanhnormal():
    raise NotImplementedError



if __name__ == "__main__":
    pytest.main([__file__])
