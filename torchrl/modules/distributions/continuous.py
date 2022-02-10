from numbers import Number
from typing import Union, Iterable

import numpy as np
import torch
from torch import distributions as D
from torch.distributions import constraints

from torchrl.modules.utils import mappings
from .truncated_normal import TruncatedNormal as _TruncatedNormal
from .utils import UNIFORM

__all__ = ["TanhNormal", "Delta", "TanhDelta"]


class SafeTanhTransform(D.TanhTransform):
    """
    TanhTransform subclass that ensured that the transformation is numerically invertible.

    """

    delta = 1e-4

    def _call(self, x: torch.Tensor) -> torch.Tensor:
        y = super()._call(x)
        y = y.clamp(-1 + self.delta, 1 - self.delta)
        return y


class TruncatedNormal(D.Independent):
    """
    Implements a Truncated Normal distribution with location scaling.
    Location scaling prevents the location to be "too far" from 0, which ultimately
    leads to numerically unstable samples and poor gradient computation (e.g. gradient explosion).
    In practice, the location is computed according to
        loc = (loc / upscale).tanh() * upscale.
    This behaviour can be disabled by switching off the tanh_loc parameter (see below).


    Args:
        net_output (torch.Tensor): tensor containing the mean and std data. The distribution mean will be taken to be
            the first half of net_output over the last dimension, and the std to be some (positive mapping of) the
            second half of that tensor.
            The mapping function for the std can be controlled via the scale_mapping argument;
        upscale (torch.Tensor or number): 'a' scaling factor in the formula:
             loc = (loc / upscale).tanh() * upscale;
        min (torch.Tensor or number): minimum value of the distribution. Default = -1.0;
        max (torch.Tensor or number): maximum value of the distribution. Default = 1.0;
        scale_mapping (str): positive mapping function to be used with the std.
            default = "biased_softplus_1.0" (i.e. softplus map with bias such that fn(0.0) = 1.0)
            choices: "softplus", "exp", "relu", "biased_softplus_1";
        event_dims (int): number of dimensions describing the action.
            default = 1;
        tanh_loc (bool): if True, the above formula is used for the location scaling, otherwise the raw value is kept.
            default: True;
        tanh_scale (bool): if True, the above formula is used for the standard deviation scaling before positive
            mapping, otherwise the raw value is kept.
            default: False.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.greater_than(1e-6),
    }

    def __init__(
        self,
        net_output: torch.Tensor,
        upscale: Union[torch.Tensor, Number] = 5.0,
        min: Union[torch.Tensor, Number] = -1.0,
        max: Union[torch.Tensor, Number] = 1.0,
        scale_mapping: str = "biased_softplus_1.0",
        tanh_loc: bool = True,
        tanh_scale: bool = False,
    ):
        err_msg = "TruncatedNormal max values must be strictly greater than min values"
        if isinstance(max, torch.Tensor) or isinstance(min, torch.Tensor):
            if not (max > min).all():
                raise RuntimeError(err_msg)
        elif isinstance(max, Number) and isinstance(min, Number):
            if not max > min:
                raise RuntimeError(err_msg)
        else:
            if not all(max > min):
                raise RuntimeError(err_msg)

        loc, scale = net_output.chunk(chunks=2, dim=-1)
        self.tanh_loc = tanh_loc
        if tanh_loc:
            if (isinstance(upscale, torch.Tensor) and (upscale != 1.0).any()) or (
                not isinstance(upscale, torch.Tensor) and upscale != 1.0
            ):
                upscale = (
                    upscale
                    if not isinstance(upscale, torch.Tensor)
                    else upscale.to(loc.device)
                )
            loc = loc / upscale
            loc = loc.tanh() * upscale
        if tanh_scale:
            if (isinstance(upscale, torch.Tensor) and (upscale != 1.0).any()) or (
                not isinstance(upscale, torch.Tensor) and upscale != 1.0
            ):
                upscale = (
                    upscale
                    if not isinstance(upscale, torch.Tensor)
                    else upscale.to(loc.device)
                )
            scale = scale / upscale
            scale = scale.tanh() * upscale

        if not isinstance(max, torch.Tensor):
            max = torch.tensor(max)
        if not isinstance(min, torch.Tensor):
            min = torch.tensor(min)
        max = max.to(loc.device)
        min = min.to(loc.device)

        loc = loc + (max - min) / 2 + min

        scale = mappings(scale_mapping)(scale).clamp_min(1e-4)
        self.upscale = upscale

        base_dist = _TruncatedNormal(
            loc, scale, min.expand_as(loc), max.expand_as(scale)
        )
        super().__init__(base_dist, 1, validate_args=False)

    @property
    def mode(self):
        m = self.base_dist.loc
        a = self.base_dist._non_std_a + self.base_dist._dtype_min_gt_0
        b = self.base_dist._non_std_b - self.base_dist._dtype_min_gt_0
        m = torch.min(torch.stack([m, b], -1), dim=-1)[0]
        return torch.max(torch.stack([m, a], -1), dim=-1)[0]

    def log_prob(self, value, **kwargs):
        a = self.base_dist._non_std_a + self.base_dist._dtype_min_gt_0
        a = a.expand_as(value)
        b = self.base_dist._non_std_b - self.base_dist._dtype_min_gt_0
        b = b.expand_as(value)
        value = torch.min(torch.stack([value, b], -1), dim=-1)[0]
        value = torch.max(torch.stack([value, a], -1), dim=-1)[0]
        return super().log_prob(value, **kwargs)


class TanhNormal(D.TransformedDistribution):
    """
    Implements a TanhNormal distribution with location scaling.
    Location scaling prevents the location to be "too far" from 0 when a TanhTransform is applied, which ultimately
    leads to numerically unstable samples and poor gradient computation (e.g. gradient explosion).
    In practice, the location is computed according to
        loc = (loc / upscale).tanh() * upscale.
    This behaviour can be disabled by switching off the tanh_loc parameter (see below).


    Args:
        net_output (torch.Tensor): tensor containing the mean and std data. The distribution mean will be taken to be
            the first half of net_output over the last dimension, and the std to be some (positive mapping of) the
            second half of that tensor.
            The mapping function for the std can be controlled via the scale_mapping argument;
        upscale (torch.Tensor or number): 'a' scaling factor in the formula:
             loc = (loc / upscale).tanh() * upscale;
        min (torch.Tensor or number): minimum value of the distribution. Default = -1.0;
        max (torch.Tensor or number): maximum value of the distribution. Default = 1.0;
        scale_mapping (str): positive mapping function to be used with the std.
            default = "biased_softplus_1.0" (i.e. softplus map with bias such that fn(0.0) = 1.0)
            choices: "softplus", "exp", "relu", "biased_softplus_1";
        event_dims (int): number of dimensions describing the action.
            default = 1;
        tanh_loc (bool): if True, the above formula is used for the location scaling, otherwise the raw value is kept.
            default: True;
        tanh_scale (bool): if True, the above formula is used for the standard deviation scaling before positive
            mapping, otherwise the raw value is kept.
            default: False.
    """

    arg_constraints = {
        "loc": constraints.real,
        "scale": constraints.greater_than(1e-6),
    }

    def __init__(
        self,
        net_output: torch.Tensor,
        upscale: Union[torch.Tensor, Number] = 5.0,
        min: Union[torch.Tensor, Number] = -1.0,
        max: Union[torch.Tensor, Number] = 1.0,
        scale_mapping: str = "biased_softplus_1.0",
        event_dims: int = 1,
        tanh_loc: bool = True,
        tanh_scale: bool = False,
    ):
        err_msg = "TanhNormal max values must be strictly greater than min values"
        if isinstance(max, torch.Tensor) or isinstance(min, torch.Tensor):
            if not (max > min).all():
                raise RuntimeError(err_msg)
        elif isinstance(max, Number) and isinstance(min, Number):
            if not max > min:
                raise RuntimeError(err_msg)
        else:
            if not all(max > min):
                raise RuntimeError(err_msg)

        loc, scale = net_output.chunk(chunks=2, dim=-1)
        self.tanh_loc = tanh_loc
        if tanh_loc:
            if (isinstance(upscale, torch.Tensor) and (upscale != 1.0).any()) or (
                not isinstance(upscale, torch.Tensor) and upscale != 1.0
            ):
                upscale = (
                    upscale
                    if not isinstance(upscale, torch.Tensor)
                    else upscale.to(loc.device)
                )
            loc = loc / upscale
            loc = loc.tanh() * upscale
        if tanh_scale:
            if (isinstance(upscale, torch.Tensor) and (upscale != 1.0).any()) or (
                not isinstance(upscale, torch.Tensor) and upscale != 1.0
            ):
                upscale = (
                    upscale
                    if not isinstance(upscale, torch.Tensor)
                    else upscale.to(loc.device)
                )
            scale = scale / upscale
            scale = scale.tanh() * upscale

        if isinstance(max, torch.Tensor):
            max = max.to(loc.device)
        if isinstance(min, torch.Tensor):
            min = min.to(loc.device)
        self.min = min
        self.max = max

        loc = loc + (max - min) / 2 + min

        self.loc = loc
        self.scale = mappings(scale_mapping)(scale)
        self.upscale = upscale

        t = SafeTanhTransform()
        non_trivial_min = (isinstance(min, torch.Tensor) and (min != 1.0).any()) or (
            not isinstance(min, torch.Tensor) and min != 1.0
        )
        non_trivial_max = (isinstance(max, torch.Tensor) and (max != 1.0).any()) or (
            not isinstance(max, torch.Tensor) and max != 1.0
        )
        if non_trivial_max or non_trivial_min:
            t = D.ComposeTransform(
                [t, D.AffineTransform(loc=(max + min) / 2, scale=(max - min) / 2)]
            )
        base = D.Independent(D.Normal(self.loc, self.scale), event_dims)

        super().__init__(base, t)

    @property
    def mode(self):
        m = self.base_dist.base_dist.mean
        for t in self.transforms:
            m = t(m)
        return m


def uniform_sample_tanhnormal(dist: TanhNormal, size=torch.Size([])) -> torch.Tensor:
    """
    Defines what uniform sampling looks like for a TanhNormal distribution.

    Args:
        dist (TanhNormal): distribution defining the space where the sampling should occur.
        size (torch.Size): batch-size of the output tensor

    Returns: a tensor sampled uniformly in the boundaries defined by the input distribution.

    """
    return torch.rand_like(dist.sample(size)) * (dist.max - dist.min) + dist.min


UNIFORM[TanhNormal] = uniform_sample_tanhnormal


class Delta(D.Distribution):
    """
    Delta distribution.

    Args:
        param (torch.Tensor): parameter of the delta distribution;
        atol (numbe): absolute tolerance to consider that a tensor matches the distribution parameter;
            default: 1e-6
        rtol (numbe): relative tolerance to consider that a tensor matches the distribution parameter;
            default: 1e-6
        batch_shape (torch.Size): batch shape;
        event_shape (torch.Size): shape of the outcome;
    """

    arg_constraints = {}

    def __init__(
        self,
        param: torch.Tensor,
        atol: Number = 1e-6,
        rtol: Number = 1e-6,
        batch_shape: Union[torch.Size, Iterable] = torch.Size([]),
        event_shape: Union[torch.Size, Iterable] = torch.Size([]),
    ):
        self.param = param
        self.atol = atol
        self.rtol = rtol
        if not len(batch_shape) and not len(event_shape):
            batch_shape = param.shape[:-1]
            event_shape = param.shape[-1:]
        super().__init__(batch_shape=batch_shape, event_shape=event_shape)

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
    def sample(self, size=torch.Size([])) -> torch.Tensor:
        return self.param.expand(*size, *self.param.shape)

    def rsample(self, size=torch.Size([])) -> torch.Tensor:
        return self.param.expand(*size, *self.param.shape)

    @property
    def mode(self) -> torch.Tensor:
        return self.param

    @property
    def mean(self) -> torch.Tensor:
        return self.param


class TanhDelta(D.TransformedDistribution):
    """
    Implements a Tanh transformed Delta distribution.

    Args:
        net_output (torch.Tensor): parameter of the delta distribution;
                min (torch.Tensor or number): minimum value of the distribution. Default = -1.0;
        max (torch.Tensor or number): maximum value of the distribution. Default = 1.0;
        event_dims (int): number of dimensions describing the action.
            default = 1;
        atol (numbe): absolute tolerance to consider that a tensor matches the distribution parameter;
            default: 1e-6
        rtol (numbe): relative tolerance to consider that a tensor matches the distribution parameter;
            default: 1e-6
        batch_shape (torch.Size): batch shape;
        event_shape (torch.Size): shape of the outcome;

    """

    arg_constraints = {
        "loc": constraints.real,
    }

    def __init__(
        self,
        net_output: torch.Tensor,
        min: Union[torch.Tensor, Number] = -1.0,
        max: Union[torch.Tensor, Number] = 1.0,
        event_dims: int = 1,
        atol: Number = 1e-4,
        rtol: Number = 1e-4,
        **kwargs,
    ):
        minmax_msg = "max value has been found to be equal or less than min value"
        if isinstance(max, torch.Tensor) or isinstance(min, torch.Tensor):
            if not (max > min).all():
                raise ValueError(minmax_msg)
        elif isinstance(max, Number) and isinstance(min, Number):
            if max <= min:
                raise ValueError(minmax_msg)
        else:
            if not all(max > min):
                raise ValueError(minmax_msg)

        loc = net_output
        loc = loc + (max - min) / 2 + min

        self.loc = loc

        t = D.TanhTransform()
        non_trivial_min = (isinstance(min, torch.Tensor) and (min != 1.0).any()) or (
            not isinstance(min, torch.Tensor) and min != 1.0
        )
        non_trivial_max = (isinstance(max, torch.Tensor) and (max != 1.0).any()) or (
            not isinstance(max, torch.Tensor) and max != 1.0
        )
        if non_trivial_max or non_trivial_min:
            t = D.ComposeTransform(
                [t, D.AffineTransform(loc=(max + min) / 2, scale=(max - min) / 2)]
            )
        event_shape = net_output.shape[-event_dims:]
        batch_shape = net_output.shape[:-event_dims]
        base = Delta(
            loc,
            atol=atol,
            rtol=rtol,
            batch_shape=batch_shape,
            event_shape=event_shape,
            **kwargs,
        )

        super().__init__(base, t)

    @property
    def mode(self) -> torch.Tensor:
        mode = self.base_dist.param
        for t in self.transforms:
            mode = t(mode)
        return mode

    @property
    def mean(self) -> torch.Tensor:
        raise AttributeError("TanhDelta mean has not analytical form.")


def uniform_sample_delta(dist: Delta, size=torch.Size([])) -> torch.Tensor:
    return torch.randn_like(dist.sample(size))


UNIFORM[Delta] = uniform_sample_delta
