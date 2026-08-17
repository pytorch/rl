# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

from typing import Any

import torch
from tensordict import is_tensor_collection, TensorDict, TensorDictBase
from tensordict.nn import composite_lp_aggregate, CompositeDistribution
from torch import autograd, distributions as d
from torch.distributions import Independent, Transform, TransformedDistribution

try:
    from torch.compiler import is_dynamo_compiling
except ImportError:
    from torch._dynamo import is_compiling as is_dynamo_compiling


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
        try:
            entropy = component.entropy()
        except NotImplementedError:
            if not component.has_rsample:
                raise
            _, log_prob = rsample_and_log_prob(component, (samples_mc,))
            entropy = -log_prob.mean(0)
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
