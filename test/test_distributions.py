import pytest
import torch
from torch import nn

from _utils_internal import get_available_devices
from torchrl.modules import TanhNormal, NormalParamWrapper
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


@pytest.mark.parametrize(
    "min", [-torch.ones(3), -1, 3 * torch.tensor([-1.0, -2.0, -0.5]), -3]
)
@pytest.mark.parametrize(
    "max", [torch.ones(3), 1, 3 * torch.tensor([1.0, 2.0, 0.5]), 3]
)
@pytest.mark.parametrize(
    "vecs", [
        (torch.tensor([0.1, 10.0, 5.0]),
         torch.tensor([0.1, 10.0, 5.0])),
        (torch.zeros(7, 3), torch.ones(7, 3)), ]
)
@pytest.mark.parametrize(
    "upscale", [torch.ones(3), 1, 3 * torch.tensor([1.0, 2.0, 0.5]), 3]
)
@pytest.mark.parametrize("shape", [torch.Size([]), torch.Size([3, 4])])
def test_tanhnormal(min, max, vecs, upscale, shape):
    torch.manual_seed(0)
    d = TanhNormal(*vecs, upscale=upscale, min=min, max=max, )
    for _ in range(100):
        a = d.rsample(shape)
        assert a.shape[: len(shape)] == shape
        assert (a >= d.min).all()
        assert (a <= d.max).all()


@pytest.mark.parametrize("batch_size", [(3,), (5, 7,)])
@pytest.mark.parametrize("device", get_available_devices())
@pytest.mark.parametrize("scale_mapping", ["exp", "biased_softplus_1.0",
                                           "biased_softplus_0.11", "expln",
                                           "relu", "softplus", "raise_error"])
def test_normal_mapping(batch_size, device, scale_mapping, action_dim=11,
                        state_dim=3):
    torch.manual_seed(0)
    for _ in range(100):
        module = nn.LazyLinear(2 * action_dim).to(device)
        module = NormalParamWrapper(module, scale_mapping=scale_mapping).to(
            device)
        if scale_mapping != "raise_error":
            loc, scale = module(
                torch.randn(*batch_size, state_dim, device=device))
            assert (scale > 0).all()
        else:
            with pytest.raises(NotImplementedError, match="Unknown mapping "
                                                          "raise_error"):
                loc, scale = module(
                    torch.randn(*batch_size, state_dim, device=device))



if __name__ == "__main__":
    pytest.main([__file__])
