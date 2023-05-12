# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union

import torch
from torch import distributions as d
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
            raise ValueError("Mae a ComposeTransform first.")
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
