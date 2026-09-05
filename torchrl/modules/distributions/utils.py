# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""Sampling, entropy and KL helpers for TorchRL distributions.

``has_analytic_entropy`` and ``has_analytic_kl`` detect closed-form
``entropy`` / ``kl_divergence`` implementations from the distribution
class (or the torch KL registry) so objectives can avoid ``try/except``
on the hot path.
"""

from __future__ import annotations

from typing import Any

import torch
from tensordict import is_tensor_collection, TensorDict, TensorDictBase
from tensordict.nn import composite_lp_aggregate, CompositeDistribution
from torch import autograd, distributions as d
from torch.distributions import Independent, Transform, TransformedDistribution
from torch.distributions.kl import _KL_REGISTRY

from torchrl._utils import logger as torchrl_logger, VERBOSE

try:
    from torch.compiler import is_dynamo_compiling
except ImportError:
    from torch._dynamo import is_compiling as is_dynamo_compiling

_ANALYTIC_ENTROPY_CACHE: dict[type, bool] = {}
_ANALYTIC_KL_CACHE: dict[tuple[type, type], bool] = {}
_MC_ENTROPY_WARNED: set[type] = set()
_MC_KL_WARNED: set[tuple[type, type]] = set()


def sample_and_log_prob(
    distribution: d.Distribution,
    sample_shape: torch.Size | tuple[int, ...] = (),
    *,
    reparameterize: bool = False,
) -> tuple[Any, torch.Tensor | TensorDictBase]:
    """Sample once and score the same draw atomically when supported.

    If the distribution implements ``sample_and_log_prob`` or
    ``rsample_and_log_prob``, the matching method is used so that the score is
    computed from the same latent draw as the sample. Otherwise, this function
    falls back to separate sampling and scoring. Composite distributions are
    handled component by component and respect
    :func:`~tensordict.nn.composite_lp_aggregate`.

    Args:
        distribution (Distribution): distribution to sample and score.
        sample_shape (torch.Size or tuple of int, optional): leading sample
            dimensions. Defaults to an empty shape.
        reparameterize (bool, optional): if ``True``, use reparameterized
            sampling. Defaults to ``False``.

    Returns:
        A tuple containing the sample and its log probability.
    """
    sample_shape = torch.Size(sample_shape)
    if isinstance(distribution, CompositeDistribution):
        samples = {}
        log_probs = {}
        for name, component in distribution.dists.items():
            sample, log_prob = sample_and_log_prob(
                component,
                sample_shape,
                reparameterize=reparameterize,
            )
            samples[name] = sample
            if isinstance(name, str):
                log_prob_name = name + "_log_prob"
            else:
                log_prob_name = name[:-1] + (name[-1] + "_log_prob",)
            log_probs[log_prob_name] = log_prob

        batch_size = sample_shape + distribution.batch_shape
        sample = TensorDict(samples, batch_size=batch_size)
        if not composite_lp_aggregate():
            return sample, TensorDict(log_probs, batch_size=batch_size)

        log_prob = 0.0
        for component_log_prob in log_probs.values():
            if is_tensor_collection(component_log_prob):
                component_log_prob = component_log_prob.sum(dim="feature", reduce=True)
            elif component_log_prob.ndim > sample.ndim:
                component_log_prob = component_log_prob.flatten(sample.ndim, -1).sum(-1)
            log_prob = log_prob + component_log_prob
        return sample, log_prob

    method_name = "rsample_and_log_prob" if reparameterize else "sample_and_log_prob"
    joint_sample = getattr(distribution, method_name, None)
    if joint_sample is not None:
        return joint_sample(sample_shape)
    sample_fn = distribution.rsample if reparameterize else distribution.sample
    sample = sample_fn(sample_shape)
    if isinstance(sample, torch.Tensor) or is_tensor_collection(sample):
        return sample, distribution.log_prob(sample)
    return sample, distribution.log_prob(*sample)


def rsample_and_log_prob(
    distribution: d.Distribution,
    sample_shape: torch.Size | tuple[int, ...] = (),
) -> tuple[Any, torch.Tensor | TensorDictBase]:
    """Reparameterize once and score the same draw atomically when supported.

    Args:
        distribution (Distribution): distribution to sample and score.
        sample_shape (torch.Size or tuple of int, optional): leading sample
            dimensions. Defaults to an empty shape.

    Returns:
        A tuple containing the reparameterized sample and its log probability.
    """
    return sample_and_log_prob(
        distribution,
        sample_shape,
        reparameterize=True,
    )


def has_analytic_entropy(dist: d.Distribution) -> bool:
    """Return whether ``dist`` implements a closed-form ``entropy()``.

    The check is class-level: ``type(dist).entropy is not
    torch.distributions.Distribution.entropy``. ``Independent`` is resolved
    through its base distribution because ``Independent.entropy`` always
    exists and only works when the base distribution implements entropy.
    ``CompositeDistribution`` is treated as not having a closed-form
    entropy: its ``entropy()`` may return a TensorDict and still relies on
    ``try/except`` internally. Use :func:`composite_entropy` for composites.

    Args:
        dist (torch.distributions.Distribution): distribution to inspect.

    Returns:
        bool: ``True`` if a closed-form entropy method is available.

    Examples:
        >>> import torch
        >>> from torch import distributions as d
        >>> from torchrl.modules.distributions.utils import has_analytic_entropy
        >>> has_analytic_entropy(d.Normal(torch.zeros(2), torch.ones(2)))
        True
        >>> has_analytic_entropy(d.Independent(d.Normal(torch.zeros(2), torch.ones(2)), 1))
        True
    """
    if isinstance(dist, CompositeDistribution):
        return False
    if isinstance(dist, Independent):
        return has_analytic_entropy(dist.base_dist)
    cls = type(dist)
    cached = _ANALYTIC_ENTROPY_CACHE.get(cls)
    if cached is not None:
        return cached
    entropy_fn = getattr(cls, "entropy", None)
    result = entropy_fn is not None and entropy_fn is not d.Distribution.entropy
    _ANALYTIC_ENTROPY_CACHE[cls] = result
    return result


def has_analytic_kl(p: d.Distribution, q: d.Distribution) -> bool:
    """Return whether ``kl_divergence(p, q)`` has a registered closed form.

    ``Independent`` and ``TransformedDistribution`` pairs are resolved
    through their bases, matching the registered torch KL implementations
    without calling them (those wrappers raise ``NotImplementedError`` when
    the inner pair is missing). Other pairs are looked up in
    ``torch.distributions.kl._KL_REGISTRY``.

    Args:
        p (torch.distributions.Distribution): left argument of
            ``kl_divergence(p, q)``.
        q (torch.distributions.Distribution): right argument of
            ``kl_divergence(p, q)``.

    Returns:
        bool: ``True`` if a closed-form KL is registered for this pair.

    Examples:
        >>> import torch
        >>> from torch import distributions as d
        >>> from torchrl.modules.distributions.utils import has_analytic_kl
        >>> loc = torch.zeros(2)
        >>> scale = torch.ones(2)
        >>> has_analytic_kl(d.Normal(loc, scale), d.Normal(loc, scale))
        True
    """
    if isinstance(p, Independent) and isinstance(q, Independent):
        if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
            return False
        return has_analytic_kl(p.base_dist, q.base_dist)
    if isinstance(p, TransformedDistribution) and isinstance(q, TransformedDistribution):
        if p.transforms != q.transforms or p.event_shape != q.event_shape:
            return False
        return has_analytic_kl(p.base_dist, q.base_dist)
    key = (type(p), type(q))
    cached = _ANALYTIC_KL_CACHE.get(key)
    if cached:
        return True
    result = False
    type_p, type_q = key
    for super_p, super_q in _KL_REGISTRY:
        if super_p is Independent or super_q is Independent:
            continue
        if super_p is TransformedDistribution or super_q is TransformedDistribution:
            continue
        if issubclass(type_p, super_p) and issubclass(type_q, super_q):
            result = True
            break
    if result:
        _ANALYTIC_KL_CACHE[key] = True
    return result


def _warn_mc_entropy(dist: d.Distribution) -> None:
    if not VERBOSE:
        return
    cls = type(dist)
    if cls in _MC_ENTROPY_WARNED:
        return
    _MC_ENTROPY_WARNED.add(cls)
    torchrl_logger.warning(
        f"Entropy not implemented for {cls}. Using Monte Carlo sampling."
    )


def _warn_mc_kl(p: d.Distribution, q: d.Distribution) -> None:
    if not VERBOSE:
        return
    key = (type(p), type(q))
    if key in _MC_KL_WARNED:
        return
    _MC_KL_WARNED.add(key)
    torchrl_logger.warning(
        f"KL divergence not implemented for {key}. Using Monte Carlo sampling."
    )


def composite_entropy(
    distribution: CompositeDistribution,
    samples_mc: int = 1,
) -> torch.Tensor | TensorDictBase:
    """Compute component entropy without inverse-scoring Monte Carlo samples.

    Analytic component entropies are used when available. Components without
    analytic entropy are estimated from atomic reparameterized samples.

    Args:
        distribution (CompositeDistribution): distribution whose component
            entropies are computed.
        samples_mc (int, optional): number of Monte Carlo samples used for
            components without analytic entropy. Defaults to ``1``.

    Returns:
        The aggregated entropy, or a TensorDict of component entropies when
        composite log-probability aggregation is disabled.
    """
    entropies = {}
    for name, component in distribution.dists.items():
        analytic_entropy = has_analytic_entropy(component)
        if analytic_entropy:
            entropy = component.entropy()
        compiling = is_dynamo_compiling()
        needs_mc = not analytic_entropy or compiling
        if analytic_entropy and not compiling and not entropy.isfinite().all():
            needs_mc = True
        if needs_mc:
            if not analytic_entropy and not component.has_rsample:
                raise NotImplementedError(
                    f"Entropy is not implemented for {type(component)} and "
                    "the component does not support reparameterized sampling."
                )
            if not compiling:
                _warn_mc_entropy(component)
            if analytic_entropy:
                _, log_prob = sample_and_log_prob(
                    component,
                    (samples_mc,),
                    reparameterize=component.has_rsample,
                )
            else:
                _, log_prob = rsample_and_log_prob(component, (samples_mc,))
            sampled_entropy = -log_prob.mean(0)
            if analytic_entropy and compiling:
                entropy = torch.where(
                    entropy.isfinite(), entropy, sampled_entropy
                )
            else:
                entropy = sampled_entropy
        if isinstance(name, str):
            entropy_name = name + "_entropy"
        else:
            entropy_name = name[:-1] + (name[-1] + "_entropy",)
        entropies[entropy_name] = entropy

    if not composite_lp_aggregate():
        return TensorDict(entropies, batch_size=distribution.batch_shape)

    entropy = 0.0
    batch_ndim = len(distribution.batch_shape)
    for component_entropy in entropies.values():
        if is_tensor_collection(component_entropy):
            component_entropy = component_entropy.sum(dim="feature", reduce=True)
        elif component_entropy.ndim > batch_ndim:
            component_entropy = component_entropy.flatten(batch_ndim, -1).sum(-1)
        entropy = entropy + component_entropy
    return entropy


def _cast_device(elt: torch.Tensor | float, device) -> torch.Tensor | float:
    if isinstance(elt, torch.Tensor):
        _non_blocking = device is not None and torch.device(device).type == "cuda"
        return elt.to(device, non_blocking=_non_blocking)
    return elt


def _cast_transform_device(transform, device):
    if transform is None:
        return transform
    _non_blocking = device is not None and torch.device(device).type == "cuda"
    if isinstance(transform, d.ComposeTransform):
        for i, t in enumerate(transform.parts):
            transform.parts[i] = _cast_transform_device(t, device)
    elif isinstance(transform, d.Transform):
        for attribute in dir(transform):
            value = getattr(transform, attribute)
            if isinstance(value, torch.Tensor):
                setattr(
                    transform, attribute, value.to(device, non_blocking=_non_blocking)
                )
        return transform
    else:
        raise TypeError(
            f"Cannot perform device casting for transform of type {type(transform)}"
        )


class FasterTransformedDistribution(TransformedDistribution):
    """A faster implementation of TransformedDistribution."""

    __doc__ = __doc__ + TransformedDistribution.__doc__

    def __init__(self, base_distribution, transforms, validate_args=None):
        if is_dynamo_compiling():
            return super().__init__(
                base_distribution, transforms, validate_args=validate_args
            )
        if isinstance(transforms, Transform):
            self.transforms = [transforms]
        elif isinstance(transforms, list):
            raise ValueError("Make a ComposeTransform first.")
        else:
            raise ValueError(
                f"transforms must be a Transform or list, but was {transforms}"
            )
        transform = self.transforms[0]
        # Reshape base_distribution according to transforms.
        base_shape = base_distribution.batch_shape + base_distribution.event_shape
        base_event_dim = len(base_distribution.event_shape)
        # transform = ComposeTransform(self.transforms)
        # if len(base_shape) < transform.domain.event_dim:
        #     raise ValueError("base_distribution needs to have shape with size at least {}, but got {}."
        #                      .format(transform.domain.event_dim, base_shape))
        transform_codomain_event_dim = transform.codomain.event_dim
        transform_domain_event_dim = transform.domain.event_dim

        forward_shape = transform.forward_shape(base_shape)
        expanded_base_shape = transform.inverse_shape(forward_shape)
        if base_shape != expanded_base_shape:
            base_batch_shape = expanded_base_shape[
                : len(expanded_base_shape) - base_event_dim
            ]
            base_distribution = base_distribution.expand(base_batch_shape)
        reinterpreted_batch_ndims = transform_domain_event_dim - base_event_dim
        if reinterpreted_batch_ndims > 0:
            base_distribution = Independent(
                base_distribution, reinterpreted_batch_ndims
            )
        self.base_dist = base_distribution

        # Compute shapes.
        transform_change_in_event_dim = (
            transform_codomain_event_dim - transform_domain_event_dim
        )
        event_dim = max(
            transform_codomain_event_dim,  # the transform is coupled
            base_event_dim + transform_change_in_event_dim,  # the base dist is coupled
        )
        cut = len(forward_shape) - event_dim
        batch_shape = forward_shape[:cut]
        event_shape = forward_shape[cut:]
        super(TransformedDistribution, self).__init__(
            batch_shape, event_shape, validate_args=validate_args
        )


def _safetanh(x, eps):  # noqa: D103
    lim = 1.0 - eps
    y = x.tanh()
    return y.clamp(-lim, lim)


def _safeatanh(y, eps):  # noqa: D103
    lim = 1.0 - eps
    return y.clamp(-lim, lim).atanh()


class _SafeTanh(autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(input, eps):
        output = input.tanh()
        lim = 1.0 - eps
        output = output.clamp(-lim, lim)
        # ctx.save_for_backward(output)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        # input, eps = inputs
        # ctx.mark_non_differentiable(ind, ind_inv)
        # # Tensors must be saved via ctx.save_for_backward. Please do not
        # # assign them directly onto the ctx object.
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, *grad):
        grad = grad[0]
        (output,) = ctx.saved_tensors
        return (grad * (1 - output.pow(2)), None)


class _SafeTanhNoEps(autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(input):
        output = input.tanh()
        eps = torch.finfo(input.dtype).resolution
        lim = 1.0 - eps
        output = output.clamp(-lim, lim)
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        ctx.save_for_backward(output)

    @staticmethod
    def backward(ctx, *grad):
        grad = grad[0]
        (output,) = ctx.saved_tensors
        return (grad * (1 - output.pow(2)),)


class _SafeaTanh(autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(tanh_val, eps):
        if eps is None:
            eps = torch.finfo(tanh_val.dtype).resolution
        lim = 1.0 - eps
        output = tanh_val.clamp(-lim, lim)
        # ctx.save_for_backward(output)
        output = output.atanh()
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        tanh_val, eps = inputs

        # ctx.mark_non_differentiable(ind, ind_inv)
        # # Tensors must be saved via ctx.save_for_backward. Please do not
        # # assign them directly onto the ctx object.
        ctx.save_for_backward(tanh_val)
        ctx.eps = eps

    @staticmethod
    def backward(ctx, *grad):
        grad = grad[0]
        (tanh_val,) = ctx.saved_tensors
        eps = ctx.eps
        lim = 1.0 - eps
        output = tanh_val.clamp(-lim, lim)
        return (grad / (1 - output.pow(2)), None)


class _SafeaTanhNoEps(autograd.Function):
    generate_vmap_rule = True

    @staticmethod
    def forward(tanh_val):
        eps = torch.finfo(tanh_val.dtype).resolution
        lim = 1.0 - eps
        output = tanh_val.clamp(-lim, lim)
        # ctx.save_for_backward(output)
        output = output.atanh()
        return output

    @staticmethod
    def setup_context(ctx, inputs, output):
        tanh_val = inputs[0]
        eps = torch.finfo(tanh_val.dtype).resolution

        # ctx.mark_non_differentiable(ind, ind_inv)
        # # Tensors must be saved via ctx.save_for_backward. Please do not
        # # assign them directly onto the ctx object.
        ctx.save_for_backward(tanh_val)
        ctx.eps = eps

    @staticmethod
    def backward(ctx, *grad):
        grad = grad[0]
        (tanh_val,) = ctx.saved_tensors
        eps = ctx.eps
        lim = 1.0 - eps
        output = tanh_val.clamp(-lim, lim)
        return (grad / (1 - output.pow(2)),)


safetanh = _SafeTanh.apply
safeatanh = _SafeaTanh.apply

safetanh_noeps = _SafeTanhNoEps.apply
safeatanh_noeps = _SafeaTanhNoEps.apply
