# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
from packaging import version
from torch import autograd, distributions as d
from torch.distributions import Independent, Transform, TransformedDistribution


def _cast_device(elt: Union[torch.Tensor, float], device) -> Union[torch.Tensor, float]:
    if isinstance(elt, torch.Tensor):
        return elt.to(device)
    return elt


def _cast_transform_device(transform, device):
    if transform is None:
        return transform
    elif isinstance(transform, d.ComposeTransform):
        for i, t in enumerate(transform.parts):
            transform.parts[i] = _cast_transform_device(t, device)
    elif isinstance(transform, d.Transform):
        for attribute in dir(transform):
            value = getattr(transform, attribute)
            if isinstance(value, torch.Tensor):
                setattr(transform, attribute, value.to(device))
        return transform
    else:
        raise TypeError(
            f"Cannot perform device casting for transform of type {type(transform)}"
        )


class FasterTransformedDistribution(TransformedDistribution):
    """A faster implementation of TransformedDistribution."""

    __doc__ = __doc__ + TransformedDistribution.__doc__

    def __init__(self, base_distribution, transforms, validate_args=None):
        if isinstance(transforms, Transform):
            self.transforms = [
                transforms,
            ]
        elif isinstance(transforms, list):
            raise ValueError("Make a ComposeTransform first.")
        else:
            raise ValueError(
                "transforms must be a Transform or list, but was {}".format(transforms)
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


if version.parse(torch.__version__) >= version.parse("2.0.0"):

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

    class _SafeaTanh(autograd.Function):
        generate_vmap_rule = True

        @staticmethod
        def setup_context(ctx, inputs, output):
            tanh_val, eps = inputs
            # ctx.mark_non_differentiable(ind, ind_inv)
            # # Tensors must be saved via ctx.save_for_backward. Please do not
            # # assign them directly onto the ctx object.
            ctx.save_for_backward(tanh_val)
            ctx.eps = eps

        @staticmethod
        def forward(tanh_val, eps):
            lim = 1.0 - eps
            output = tanh_val.clamp(-lim, lim)
            # ctx.save_for_backward(output)
            output = output.atanh()
            return output

        @staticmethod
        def backward(ctx, *grad):
            grad = grad[0]
            (tanh_val,) = ctx.saved_tensors
            eps = ctx.eps
            lim = 1.0 - eps
            output = tanh_val.clamp(-lim, lim)
            return (grad / (1 - output.pow(2)), None)

    safetanh = _SafeTanh.apply
    safeatanh = _SafeaTanh.apply

else:

    def safetanh(x, eps):  # noqa: D103
        lim = 1.0 - eps
        y = x.tanh()
        return y.clamp(-lim, lim)

    def safeatanh(y, eps):  # noqa: D103
        lim = 1.0 - eps
        return y.clamp(-lim, lim).atanh()
