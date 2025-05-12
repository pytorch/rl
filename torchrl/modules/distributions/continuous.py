# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import weakref
from numbers import Number
from typing import Sequence

import numpy as np
import torch
from packaging import version
from torch import distributions as D, nn
from torch.distributions import constraints
from torch.distributions.transforms import _InverseTransform

from torchrl._utils import safe_is_current_stream_capturing
from torchrl.modules.distributions.truncated_normal import (
    TruncatedNormal as _TruncatedNormal,
)
from torchrl.modules.distributions.utils import (
    _cast_device,
    FasterTransformedDistribution,
    safeatanh_noeps,
    safetanh_noeps,
)

# speeds up distribution construction
D.Distribution.set_default_validate_args(False)

try:
    from torch.compiler import assume_constant_result
except ImportError:
    from torch._dynamo import assume_constant_result

try:
    from torch.compiler import is_compiling
except ImportError:
    from torch._dynamo import is_compiling

TORCH_VERSION = version.parse(torch.__version__).base_version
TORCH_VERSION_PRE_2_6 = version.parse(TORCH_VERSION) < version.parse("2.6.0")


class IndependentNormal(D.Independent):
    """Implements a Normal distribution with location scaling.

    Location scaling prevents the location to be "too far" from 0, which ultimately
    leads to numerically unstable samples and poor gradient computation (e.g. gradient explosion).
    In practice, the location is computed according to

        .. math::
            loc = tanh(loc / upscale) * upscale.

    This behavior can be disabled by switching off the tanh_loc parameter (see below).


    Args:
        loc (torch.Tensor): normal distribution location parameter
        scale (torch.Tensor): normal distribution sigma parameter (squared root of variance)
        upscale (torch.Tensor or number, optional): 'a' scaling factor in the formula:

            .. math::
                loc = tanh(loc / upscale) * upscale.

            Default is 5.0

        tanh_loc (bool, optional): if ``False``, the above formula is used for
            the location scaling, otherwise the raw value
            is kept. Default is ``False``;
    """

    num_params: int = 2

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        upscale: float = 5.0,
        tanh_loc: bool = False,
        event_dim: int = 1,
        **kwargs,
    ):
        self.tanh_loc = tanh_loc
        self.upscale = upscale
        self._event_dim = event_dim
        self._kwargs = kwargs
        super().__init__(D.Normal(loc, scale, **kwargs), event_dim)

    def update(self, loc, scale):
        if self.tanh_loc:
            loc = self.upscale * (loc / self.upscale).tanh()
        super().__init__(D.Normal(loc, scale, **self._kwargs), self._event_dim)

    @property
    def mode(self):
        return self.base_dist.mean

    @property
    def deterministic_sample(self):
        return self.mean


class SafeTanhTransform(D.TanhTransform):
    """TanhTransform subclass that ensured that the transformation is numerically invertible."""

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        return safetanh_noeps(x)

    def _inverse(self, y: torch.Tensor) -> torch.Tensor:
        return safeatanh_noeps(y)

    @property
    def inv(self):
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = _InverseTransform(self)
            if not is_compiling():
                self._inv = weakref.ref(inv)
        return inv


class NormalParamWrapper(nn.Module):  # noqa: D101
    def __init__(
        self,
        operator: nn.Module,
        scale_mapping: str = "biased_softplus_1.0",
        scale_lb: Number = 1e-4,
    ) -> None:
        raise RuntimeError(
            "NormalParamWrapper has been deprecated in favor of `tensordict.nn.NormalParamExtractor`. Use this class instead."
        )


class TruncatedNormal(D.Independent):
    """Implements a Truncated Normal distribution with location scaling.

    Location scaling prevents the location to be "too far" from 0, which ultimately
    leads to numerically unstable samples and poor gradient computation (e.g. gradient explosion).
    In practice, the location is computed according to

        .. math::
            loc = tanh(loc / upscale) * upscale.

    This behavior can be disabled by switching off the tanh_loc parameter (see below).


    Args:
        loc (torch.Tensor): normal distribution location parameter
        scale (torch.Tensor): normal distribution sigma parameter (squared root of variance)
        upscale (torch.Tensor or number, optional): 'a' scaling factor in the formula:

            .. math::
                loc = tanh(loc / upscale) * upscale.

            Default is 5.0

        low (torch.Tensor or number, optional): minimum value of the distribution. Default = -1.0;
        high (torch.Tensor or number, optional): maximum value of the distribution. Default = 1.0;
        tanh_loc (bool, optional): if ``True``, the above formula is used for
            the location scaling, otherwise the raw value is kept.
            Default is ``False``;
    """

    num_params: int = 2

    base_dist: _TruncatedNormal

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.greater_than(1e-6),
    }

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        upscale: torch.Tensor | float = 5.0,
        low: torch.Tensor | float = -1.0,
        high: torch.Tensor | float = 1.0,
        tanh_loc: bool = False,
    ):

        err_msg = "TanhNormal high values must be strictly greater than low values"
        if isinstance(high, torch.Tensor) or isinstance(low, torch.Tensor):
            if not (high > low).all():
                raise RuntimeError(err_msg)
        elif isinstance(high, Number) and isinstance(low, Number):
            if not high > low:
                raise RuntimeError(err_msg)
        else:
            if not all(high > low):
                raise RuntimeError(err_msg)

        if isinstance(high, torch.Tensor):
            self.non_trivial_max = (high != 1.0).any()
        else:
            self.non_trivial_max = high != 1.0

        if isinstance(low, torch.Tensor):
            self.non_trivial_min = (low != -1.0).any()
        else:
            self.non_trivial_min = low != -1.0
        self.tanh_loc = tanh_loc

        self.device = loc.device
        self.upscale = torch.as_tensor(upscale, device=self.device)

        high = torch.as_tensor(high, device=self.device)
        low = torch.as_tensor(low, device=self.device)
        self.low = low
        self.high = high
        self.update(loc, scale)

    @property
    def min(self):
        self._warn_minmax()
        return self.low

    @property
    def max(self):
        self._warn_minmax()
        return self.high

    def update(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        if self.tanh_loc:
            loc = (loc / self.upscale).tanh() * self.upscale
        self.loc = loc
        self.scale = scale

        base_dist = _TruncatedNormal(
            loc,
            scale,
            a=self.low.expand_as(loc),
            b=self.high.expand_as(scale),
            device=self.device,
        )
        super().__init__(base_dist, 1, validate_args=False)

    @property
    def mode(self):
        m = self.base_dist.loc
        a = self.base_dist._non_std_a + self.base_dist._dtype_min_gt_0
        b = self.base_dist._non_std_b - self.base_dist._dtype_min_gt_0
        m = torch.min(torch.stack([m, b], -1), dim=-1)[0]
        return torch.max(torch.stack([m, a], -1), dim=-1)[0]

    @property
    def deterministic_sample(self):
        return self.mean

    def log_prob(self, value, **kwargs):
        above_or_below = (self.low > value) | (self.high < value)
        a = self.base_dist._non_std_a + self.base_dist._dtype_min_gt_0
        a = a.expand_as(value)
        b = self.base_dist._non_std_b - self.base_dist._dtype_min_gt_0
        b = b.expand_as(value)
        value = torch.min(torch.stack([value, b], -1), dim=-1)[0]
        value = torch.max(torch.stack([value, a], -1), dim=-1)[0]
        lp = super().log_prob(value, **kwargs)
        if above_or_below.any():
            if self.event_shape:
                above_or_below = above_or_below.flatten(-len(self.event_shape), -1).any(
                    -1
                )
            lp = torch.masked_fill(
                lp,
                above_or_below.expand_as(lp),
                torch.tensor(-float("inf"), device=lp.device, dtype=lp.dtype),
            )
        return lp


class _PatchedComposeTransform(D.ComposeTransform):
    @property
    def inv(self):
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = _PatchedComposeTransform([p.inv for p in reversed(self.parts)])
            if not is_compiling():
                self._inv = weakref.ref(inv)
                inv._inv = weakref.ref(self)
        return inv


class _PatchedAffineTransform(D.AffineTransform):
    @property
    def inv(self):
        inv = None
        if self._inv is not None:
            inv = self._inv()
        if inv is None:
            inv = _InverseTransform(self)
            if not is_compiling():
                self._inv = weakref.ref(inv)
        return inv


class TanhNormal(FasterTransformedDistribution):
    """Implements a TanhNormal distribution with location scaling.

    Location scaling prevents the location to be "too far" from 0 when a
    ``TanhTransform`` is applied, but ultimately
    leads to numerically unstable samples and poor gradient computation
    (e.g. gradient explosion).
    In practice, with location scaling the location is computed according to

        .. math::
            loc = tanh(loc / upscale) * upscale.


    Args:
        loc (torch.Tensor): normal distribution location parameter
        scale (torch.Tensor): normal distribution sigma parameter (squared root of variance)
        upscale (torch.Tensor or number): 'a' scaling factor in the formula:

            .. math::
                loc = tanh(loc / upscale) * upscale.

        low (torch.Tensor or number, optional): minimum value of the distribution. Default is -1.0;
        high (torch.Tensor or number, optional): maximum value of the distribution. Default is 1.0;
        event_dims (int, optional): number of dimensions describing the action.
            Default is 1. Setting ``event_dims`` to ``0`` will result in a log-probability that has the same shape
            as the input, ``1`` will reduce (sum over) the last dimension, ``2`` the last two etc.
        tanh_loc (bool, optional): if ``True``, the above formula is used for the location scaling, otherwise the raw
            value is kept. Default is ``False``;
        safe_tanh (bool, optional): if ``True``, the Tanh transform is done "safely", to avoid numerical overflows.
            This will currently break with :func:`torch.compile`.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.greater_than(1e-6),
    }

    num_params = 2

    def __init__(
        self,
        loc: torch.Tensor,
        scale: torch.Tensor,
        upscale: torch.Tensor | Number = 5.0,
        low: torch.Tensor | Number = -1.0,
        high: torch.Tensor | Number = 1.0,
        event_dims: int | None = None,
        tanh_loc: bool = False,
        safe_tanh: bool = True,
    ):
        if not isinstance(loc, torch.Tensor):
            loc = torch.as_tensor(loc, dtype=torch.get_default_dtype())
        if not isinstance(scale, torch.Tensor):
            scale = torch.as_tensor(scale, dtype=torch.get_default_dtype())
        if event_dims is None:
            event_dims = min(1, loc.ndim)

        err_msg = "TanhNormal high values must be strictly greater than low values"
        if not is_compiling() and not safe_is_current_stream_capturing():
            if isinstance(high, torch.Tensor) or isinstance(low, torch.Tensor):
                if not (high > low).all():
                    raise RuntimeError(err_msg)
            elif isinstance(high, Number) and isinstance(low, Number):
                if not high > low:
                    raise RuntimeError(err_msg)
            else:
                if not all(high > low):
                    raise RuntimeError(err_msg)

        if not isinstance(high, torch.Tensor):
            high = torch.as_tensor(high, device=loc.device)
        elif high.device != loc.device:
            high = high.to(loc.device)
        if not isinstance(low, torch.Tensor):
            low = torch.as_tensor(low, device=loc.device)
        elif low.device != loc.device:
            low = low.to(loc.device)
        if not is_compiling() and not safe_is_current_stream_capturing():
            self.non_trivial_max = (high != 1.0).any()
            self.non_trivial_min = (low != -1.0).any()
        else:
            self.non_trivial_max = self.non_trivial_min = True

        self.tanh_loc = tanh_loc
        self._event_dims = event_dims

        self.device = loc.device
        self.upscale = (
            upscale
            if not isinstance(upscale, torch.Tensor)
            else upscale.to(self.device)
        )

        low = low.to(loc.device)
        self.low = low
        self.high = high

        if safe_tanh:
            if is_compiling() and TORCH_VERSION_PRE_2_6:
                _err_compile_safetanh()
            t = SafeTanhTransform()
        else:
            t = D.TanhTransform()
        # t = D.TanhTransform()
        if is_compiling() or (self.non_trivial_max or self.non_trivial_min):
            t = _PatchedComposeTransform(
                [
                    t,
                    _PatchedAffineTransform(
                        loc=(high + low) / 2, scale=(high - low) / 2
                    ),
                ]
            )
        self._t = t

        self.update(loc, scale)

    @property
    def min(self):
        self._warn_minmax()
        return self.low

    @property
    def max(self):
        self._warn_minmax()
        return self.high

    def update(self, loc: torch.Tensor, scale: torch.Tensor) -> None:
        if self.tanh_loc:
            loc = (loc / self.upscale).tanh() * self.upscale
            # loc must be rescaled if tanh_loc
            if is_compiling() or (self.non_trivial_max or self.non_trivial_min):
                loc = loc + (self.high - self.low) / 2 + self.low
        self.loc = loc
        self.scale = scale

        if (
            hasattr(self, "base_dist")
            and (self.root_dist.loc.shape == self.loc.shape)
            and (self.root_dist.scale.shape == self.scale.shape)
        ):
            self.root_dist.loc = self.loc
            self.root_dist.scale = self.scale
        else:
            if self._event_dims > 0:
                base = D.Independent(D.Normal(self.loc, self.scale), self._event_dims)
                super().__init__(base, self._t)
            else:
                base = D.Normal(self.loc, self.scale)
                super().__init__(base, self._t)

    @property
    def support(self):
        return D.constraints.real()

    @property
    def root_dist(self):
        bd = self
        while hasattr(bd, "base_dist"):
            bd = bd.base_dist
        return bd

    @property
    def mode(self):
        raise RuntimeError(
            f"The distribution {type(self).__name__} has not analytical mode. "
            f"Use ExplorationMode.DETERMINISTIC to get a deterministic sample from it."
        )

    @property
    def deterministic_sample(self):
        m = self.root_dist.mean
        for t in self.transforms:
            m = t(m)
        return m

    @torch.enable_grad()
    def get_mode(self):
        """Computes an estimation of the mode using the Adam optimizer."""
        # Get starting point
        m = self.sample((1000,)).mean(0)
        m = torch.nn.Parameter(m.clamp(self.low, self.high).detach())
        optim = torch.optim.Adam((m,), lr=1e-2)
        self_copy = type(self)(
            loc=self.loc.detach(),
            scale=self.scale.detach(),
            low=self.low.detach(),
            high=self.high.detach(),
            event_dims=self._event_dims,
            upscale=self.upscale,
            tanh_loc=False,
        )
        for _ in range(200):
            lp = -self_copy.log_prob(m)
            lp.mean().backward()
            mc = m.clone().detach()
            m.grad.clamp_max_(1)
            optim.step()
            optim.zero_grad()
            m.data.clamp_(self_copy.low, self_copy.high)
            nans = m.isnan()
            if nans.any():
                m.data = torch.where(nans, mc, m.data)
            if (m - mc).norm() < 1e-3:
                break
        return m.detach()

    @property
    def mean(self):
        raise NotImplementedError(
            f"{type(self).__name__} does not have a closed form formula for the average. "
            "An estimate of this value can be computed using dist.sample((N,)).mean(dim=0), "
            "where N is a large number of samples."
        )


def uniform_sample_tanhnormal(dist: TanhNormal, size=None) -> torch.Tensor:
    """Defines what uniform sampling looks like for a TanhNormal distribution.

    Args:
        dist (TanhNormal): distribution defining the space where the sampling should occur.
        size (torch.Size): batch-size of the output tensor

    Returns:
         a tensor sampled uniformly in the boundaries defined by the input distribution.

    """
    if size is None:
        size = torch.Size([])
    return torch.rand_like(dist.sample(size)) * (dist.max - dist.min) + dist.min


class Delta(D.Distribution):
    """Delta distribution.

    Args:
        param (torch.Tensor): parameter of the delta distribution;
        atol (number, optional): absolute tolerance to consider that a tensor matches the distribution parameter;
            Default is 1e-6
        rtol (number, optional): relative tolerance to consider that a tensor matches the distribution parameter;
            Default is 1e-6
        batch_shape (torch.Size, optional): batch shape;
        event_shape (torch.Size, optional): shape of the outcome.

    """

    arg_constraints: dict = {}

    def __init__(
        self,
        param: torch.Tensor,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        batch_shape: torch.Size | Sequence[int] = None,
        event_shape: torch.Size | Sequence[int] = None,
    ):
        if batch_shape is None:
            batch_shape = torch.Size([])
        if event_shape is None:
            event_shape = torch.Size([])
        self.update(param)
        self.atol = atol
        self.rtol = rtol
        if not len(batch_shape) and not len(event_shape):
            batch_shape = param.shape[:-1]
            event_shape = param.shape[-1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

    def expand(self, batch_shape: torch.Size, _instance=None):
        if self.batch_shape != tuple(batch_shape):
            return type(self)(
                self.param.expand((*batch_shape, *self.event_shape)),
                atol=self.atol,
                rtol=self.rtol,
            )
        return self

    def update(self, param):
        self.param = param

    def _is_equal(self, value: torch.Tensor) -> torch.Tensor:
        param = self.param.expand_as(value)
        is_equal = abs(value - param) < self.atol + self.rtol * abs(param)
        for i in range(-1, -len(self.event_shape) - 1, -1):
            is_equal = is_equal.all(i)
        return is_equal

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        is_equal = self._is_equal(value)
        out = torch.zeros_like(is_equal, dtype=value.dtype)
        out.masked_fill_(is_equal, np.inf)
        out.masked_fill_(~is_equal, -np.inf)
        return out

    @torch.no_grad()
    def sample(self, size=None) -> torch.Tensor:
        if size is None:
            size = torch.Size([])
        return self.param.expand(*size, *self.param.shape)

    def rsample(self, size=None) -> torch.Tensor:
        if size is None:
            size = torch.Size([])
        return self.param.expand(*size, *self.param.shape)

    @property
    def mode(self) -> torch.Tensor:
        return self.param

    @property
    def deterministic_sample(self):
        return self.mean

    @property
    def mean(self) -> torch.Tensor:
        return self.param


class TanhDelta(FasterTransformedDistribution):
    """Implements a Tanh transformed_in Delta distribution.

    Args:
        param (torch.Tensor): parameter of the delta distribution;
        low (torch.Tensor or number, optional): minimum value of the distribution. Default is -1.0;
        high (torch.Tensor or number, optional): maximum value of the distribution. Default is 1.0;
        event_dims (int, optional): number of dimensions describing the action.
            Default is 1;
        atol (number, optional): absolute tolerance to consider that a tensor matches the distribution parameter;
            Default is 1e-6
        rtol (number, optional): relative tolerance to consider that a tensor matches the distribution parameter;
            Default is 1e-6
        batch_shape (torch.Size, optional): batch shape;
        event_shape (torch.Size, optional): shape of the outcome;

    """

    arg_constraints = {
        "loc": constraints.real,
    }

    def __init__(
        self,
        param: torch.Tensor,
        low: torch.Tensor | float = -1.0,
        high: torch.Tensor | float = 1.0,
        event_dims: int = 1,
        atol: float = 1e-6,
        rtol: float = 1e-6,
        safe: bool = True,
    ):
        minmax_msg = "high value has been found to be equal or less than low value"
        if isinstance(high, torch.Tensor) or isinstance(low, torch.Tensor):
            if is_compiling():
                assert (high > low).all()
            else:
                if not (high > low).all():
                    raise ValueError(minmax_msg)
        elif isinstance(high, Number) and isinstance(low, Number):
            if is_compiling():
                assert high > low
            elif high <= low:
                raise ValueError(minmax_msg)
        else:
            if not all(high > low):
                raise ValueError(minmax_msg)

        if safe:
            if is_compiling():
                _err_compile_safetanh()
            t = SafeTanhTransform()
        else:
            t = torch.distributions.TanhTransform()
        non_trivial_min = is_compiling or (
            (isinstance(low, torch.Tensor) and (low != -1.0).any())
            or (not isinstance(low, torch.Tensor) and low != -1.0)
        )
        non_trivial_max = is_compiling or (
            (isinstance(high, torch.Tensor) and (high != 1.0).any())
            or (not isinstance(high, torch.Tensor) and high != 1.0)
        )
        self.non_trivial = non_trivial_min or non_trivial_max

        self.low = _cast_device(low, param.device)
        self.high = _cast_device(high, param.device)
        loc = self.update(param)

        if self.non_trivial:
            t = _PatchedComposeTransform(
                [
                    t,
                    _PatchedAffineTransform(
                        loc=(self.high + self.low) / 2, scale=(self.high - self.low) / 2
                    ),
                ]
            )
        event_shape = param.shape[-event_dims:]
        batch_shape = param.shape[:-event_dims]
        base = Delta(
            loc,
            atol=atol,
            rtol=rtol,
            batch_shape=batch_shape,
            event_shape=event_shape,
        )

        super().__init__(base, t)

    @property
    def min(self):
        self._warn_minmax()
        return self.low

    @property
    def max(self):
        self._warn_minmax()
        return self.high

    def update(self, net_output: torch.Tensor) -> torch.Tensor | None:
        loc = net_output
        if self.non_trivial:
            device = loc.device
            shift = _cast_device(self.high - self.low, device)
            loc = loc + shift / 2 + _cast_device(self.low, device)
        if hasattr(self, "base_dist"):
            self.base_dist.update(loc)
        else:
            return loc

    @property
    def mode(self) -> torch.Tensor:
        mode = self.base_dist.param
        for t in self.transforms:
            mode = t(mode)
        return mode

    @property
    def deterministic_sample(self):
        return self.mode

    @property
    def mean(self) -> torch.Tensor:
        raise AttributeError("TanhDelta mean has not analytical form.")


def _uniform_sample_delta(dist: Delta, size=None) -> torch.Tensor:
    if size is None:
        size = torch.Size([])
    return torch.randn_like(dist.sample(size))


uniform_sample_delta = _uniform_sample_delta


def _err_compile_safetanh():
    raise RuntimeError(
        "safe_tanh=True in TanhNormal is not compatible with torch.compile with torch pre 2.6.0. "
        " To deactivate it, pass safe_tanh=False. "
        "If you are using a ProbabilisticTensorDictModule, this can be done via "
        "`distribution_kwargs={'safe_tanh': False}`. "
        "See https://github.com/pytorch/pytorch/issues/133529 for more details."
    )


_warn_compile_safetanh = assume_constant_result(_err_compile_safetanh)
