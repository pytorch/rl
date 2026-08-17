# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import warnings
from collections.abc import Sequence
from copy import copy
from typing import Any, Literal, TYPE_CHECKING

import torch

from tensordict import set_lazy_legacy, TensorDictBase, unravel_key
from tensordict.utils import _zip_strict, expand_right, NestedKey

from torchrl.data.tensor_specs import ContinuousBox, TensorSpec
from torchrl.envs.transforms import functional as F
from torchrl.envs.transforms.utils import _get_reset, _set_missing_tolerance

if TYPE_CHECKING:
    pass

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

from torchrl.envs.transforms._base import (
    _apply_to_composite,
    _apply_to_composite_inv,
    _has_tv,
    Compose,
    IMAGE_KEYS,
    ObservationTransform,
    Transform,
)
from torchrl.envs.transforms._keys import ExcludeTransform

__all__ = [
    "CatFrames",
    "CenterCrop",
    "Crop",
    "FlattenObservation",
    "GrayScale",
    "NextObservationDelta",
    "PermuteTransform",
    "Resize",
    "SqueezeTransform",
    "ToTensorImage",
    "UnsqueezeTransform",
]


class ToTensorImage(ObservationTransform):
    """Transforms a numpy-like image (W x H x C) to a pytorch image (C x W x H).

    Transforms an observation image from a (... x W x H x C) tensor to a
    (... x C x W x H) tensor. Optionally, scales the input tensor from the range
    [0, 255] to the range [0.0, 1.0] (see ``from_int`` for more details).

    In the other cases, tensors are returned without scaling.

    Args:
        from_int (bool, optional): if ``True``, the tensor will be scaled from
            the range [0, 255] to the range [0.0, 1.0]. if `False``, the tensor
            will not be scaled. if `None`, the tensor will be scaled if
            it's not a floating-point tensor. default=None.
        unsqueeze (bool): if ``True``, the observation tensor is unsqueezed
            along the first dimension. default=False.
        dtype (torch.dtype, optional): dtype to use for the resulting
            observations.

    Keyword arguments:
        in_keys (list of NestedKeys): keys to process.
        out_keys (list of NestedKeys): keys to write.
        shape_tolerant (bool, optional): if ``True``, the shape of the input
            images will be check. If the last channel is not `3`, the permutation
            will be ignored. Defaults to ``False``.

    Examples:
        >>> transform = ToTensorImage(in_keys=["pixels"])
        >>> ri = torch.randint(0, 255, (1 , 1, 10, 11, 3), dtype=torch.uint8)
        >>> td = TensorDict(
        ...     {"pixels": ri},
        ...     [1, 1])
        >>> _ = transform(td)
        >>> obs = td.get("pixels")
        >>> print(obs.shape, obs.dtype)
        torch.Size([1, 1, 3, 10, 11]) torch.float32
    """

    def __init__(
        self,
        from_int: bool | None = None,
        unsqueeze: bool = False,
        dtype: torch.device | None = None,
        *,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        shape_tolerant: bool = False,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.from_int = from_int
        self.unsqueeze = unsqueeze
        self.dtype = dtype if dtype is not None else torch.get_default_dtype()
        self.shape_tolerant = shape_tolerant

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _apply_transform(self, observation: torch.FloatTensor) -> torch.Tensor:
        if not self.shape_tolerant or observation.shape[-1] == 3:
            observation = observation.permute(
                *list(range(observation.ndimension() - 3)), -1, -3, -2
            )
        if self.from_int or (
            self.from_int is None and not torch.is_floating_point(observation)
        ):
            observation = observation.div(255)
        observation = observation.to(self.dtype)
        if self._should_unsqueeze(observation):
            observation = observation.unsqueeze(0)
        return observation

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec = self._pixel_observation(observation_spec)
        dim = [1] if self._should_unsqueeze(observation_spec) else []
        if not self.shape_tolerant or observation_spec.shape[-1] == 3:
            observation_spec.shape = torch.Size(
                [
                    *dim,
                    *observation_spec.shape[:-3],
                    observation_spec.shape[-1],
                    observation_spec.shape[-3],
                    observation_spec.shape[-2],
                ]
            )
        observation_spec.dtype = self.dtype
        return observation_spec

    def _should_unsqueeze(self, observation_like: torch.FloatTensor | TensorSpec):
        if isinstance(observation_like, torch.FloatTensor):
            has_3_dimensions = observation_like.ndimension() == 3
        else:
            has_3_dimensions = len(observation_like.shape) == 3
        return has_3_dimensions and self.unsqueeze

    def _pixel_observation(self, spec: TensorSpec) -> None:
        if isinstance(spec.space, ContinuousBox):
            spec.space.high = self._apply_transform(spec.space.high)
            spec.space.low = self._apply_transform(spec.space.low)
        return spec


class Resize(ObservationTransform):
    """Resizes a pixel observation.

    Args:
        w (int): resulting width.
        h (int, optional): resulting height. If not provided, the value of `w`
            is taken.
        interpolation (str): interpolation method

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> t = Resize(64, 84)
        >>> base_env = GymEnv("HalfCheetah-v4", from_pixels=True)
        >>> env = TransformedEnv(base_env, Compose(ToTensorImage(), t))
    """

    def __init__(
        self,
        w: int,
        h: int | None = None,
        interpolation: str = "bilinear",
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        # we also allow lists or tuples
        if isinstance(w, (list, tuple)):
            w, h = w
        if h is None:
            h = w
        if not _has_tv:
            raise ImportError(
                "Torchvision not found. The Resize transform relies on "
                "torchvision implementation. "
                "Consider installing this dependency."
            )
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.w = int(w)
        self.h = int(h)

        try:
            from torchvision.transforms.functional import InterpolationMode

            def interpolation_fn(interpolation):  # noqa: D103
                return InterpolationMode(interpolation)

        except ImportError:

            def interpolation_fn(interpolation):  # noqa: D103
                return interpolation

        self.interpolation = interpolation_fn(interpolation)

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        # flatten if necessary
        if observation.shape[-2:] == torch.Size([self.w, self.h]):
            return observation
        ndim = observation.ndimension()
        if ndim > 4:
            sizes = observation.shape[:-3]
            observation = torch.flatten(observation, 0, ndim - 4)
        try:
            from torchvision.transforms.functional import resize
        except ImportError:
            from torchvision.transforms.functional_tensor import resize
        observation = resize(
            observation,
            [self.w, self.h],
            interpolation=self.interpolation,
            antialias=True,
        )
        if ndim > 4:
            observation = observation.unflatten(0, sizes)

        return observation

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            observation_spec.shape = space.low.shape
        else:
            observation_spec.shape = self._apply_transform(
                torch.zeros(observation_spec.shape)
            ).shape

        return observation_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"w={int(self.w)}, h={int(self.h)}, "
            f"interpolation={self.interpolation}, keys={self.in_keys})"
        )

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class Crop(ObservationTransform):
    """Crops the input image at the specified location and output size.

    Args:
        w (int): resulting width
        h (int, optional): resulting height. If None, then w is used (square crop).
        top (int, optional): top pixel coordinate to start cropping. Default is 0, i.e. top of the image.
        left (int, optional): left pixel coordinate to start cropping. Default is 0, i.e. left of the image.
        in_keys (sequence of NestedKey, optional): the entries to crop. If none is provided,
            ``["pixels"]`` is assumed.
        out_keys (sequence of NestedKey, optional): the cropped images keys. If none is
            provided, ``in_keys`` is assumed.

    """

    def __init__(
        self,
        w: int,
        h: int | None = None,
        top: int = 0,
        left: int = 0,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.w = w
        self.h = h if h else w
        self.top = top
        self.left = left

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms.functional import crop

        observation = crop(observation, self.top, self.left, self.w, self.h)
        return observation

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            observation_spec.shape = space.low.shape
        else:
            observation_spec.shape = self._apply_transform(
                torch.zeros(observation_spec.shape)
            ).shape
        return observation_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"w={float(self.w):4.4f}, h={float(self.h):4.4f}, top={float(self.top):4.4f}, left={float(self.left):4.4f}, "
        )


class CenterCrop(ObservationTransform):
    """Crops the center of an image.

    Args:
        w (int): resulting width
        h (int, optional): resulting height. If None, then w is used (square crop).
        in_keys (sequence of NestedKey, optional): the entries to crop. If none is provided,
            :obj:`["pixels"]` is assumed.
        out_keys (sequence of NestedKey, optional): the cropped images keys. If none is
            provided, :obj:`in_keys` is assumed.

    """

    def __init__(
        self,
        w: int,
        h: int | None = None,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.w = w
        self.h = h if h else w

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        from torchvision.transforms.functional import center_crop

        observation = center_crop(observation, [self.w, self.h])
        return observation

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            observation_spec.shape = space.low.shape
        else:
            observation_spec.shape = self._apply_transform(
                torch.zeros(observation_spec.shape)
            ).shape
        return observation_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"w={float(self.w):4.4f}, h={float(self.h):4.4f}, "
        )


class FlattenObservation(ObservationTransform):
    """Flatten adjacent dimensions of a tensor.

    Args:
        first_dim (int): first dimension of the dimensions to flatten.
        last_dim (int): last dimension of the dimensions to flatten.
        in_keys (sequence of NestedKey, optional): the entries to flatten. If none is provided,
            :obj:`["pixels"]` is assumed.
        out_keys (sequence of NestedKey, optional): the flatten observation keys. If none is
            provided, :obj:`in_keys` is assumed.
        allow_positive_dim (bool, optional): if ``True``, positive dimensions are accepted.
            :obj:`FlattenObservation` will map these to the n^th feature dimension
            (ie n^th dimension after batch size of parent env) of the input tensor.
            Defaults to False, ie. non-negative dimensions are not permitted.
    """

    def __init__(
        self,
        first_dim: int,
        last_dim: int,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        allow_positive_dim: bool = False,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS  # default
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if not allow_positive_dim and first_dim >= 0:
            raise ValueError(
                "first_dim should be smaller than 0 to accommodate for "
                "envs of different batch_sizes."
            )
        if not allow_positive_dim and last_dim >= 0:
            raise ValueError(
                "last_dim should be smaller than 0 to accommodate for "
                "envs of different batch_sizes."
            )
        self._first_dim = first_dim
        self._last_dim = last_dim

    @property
    def first_dim(self) -> int:
        if self._first_dim >= 0 and self.parent is not None:
            return len(self.parent.batch_size) + self._first_dim
        return self._first_dim

    @property
    def last_dim(self) -> int:
        if self._last_dim >= 0 and self.parent is not None:
            return len(self.parent.batch_size) + self._last_dim
        return self._last_dim

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = torch.flatten(observation, self.first_dim, self.last_dim)
        return observation

    forward = ObservationTransform._call

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space

        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            observation_spec.shape = space.low.shape
        else:
            observation_spec.shape = self._apply_transform(
                torch.zeros(observation_spec.shape)
            ).shape
        return observation_spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"first_dim={int(self.first_dim)}, last_dim={int(self.last_dim)}, in_keys={self.in_keys}, out_keys={self.out_keys})"
        )


class UnsqueezeTransform(Transform):
    """Inserts a dimension of size one at the specified position.

    Args:
        dim (int): dimension to unsqueeze. Must be negative (or allow_positive_dim
            must be turned on).

    Keyword Args:
        allow_positive_dim (bool, optional): if ``True``, positive dimensions are accepted.
            `UnsqueezeTransform`` will map these to the n^th feature dimension
            (ie n^th dimension after batch size of parent env) of the input tensor,
            independently of the tensordict batch size (ie positive dims may be
            dangerous in contexts where tensordict of different batch dimension
            are passed).
            Defaults to False, ie. non-negative dimensions are not permitted.
        in_keys (list of NestedKeys): input entries (read).
        out_keys (list of NestedKeys): input entries (write). Defaults to ``in_keys`` if
            not provided.
        in_keys_inv (list of NestedKeys): input entries (read) during ``inv`` calls.
        out_keys_inv (list of NestedKeys): input entries (write) during ``inv`` calls.
            Defaults to ``in_keys_in`` if not provided.
    """

    invertible = True

    @classmethod
    def __new__(cls, *args, **kwargs):
        cls._dim = None
        return super().__new__(cls)

    def __init__(
        self,
        dim: int | None = None,
        *,
        allow_positive_dim: bool = False,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = []  # default
        if out_keys is None:
            out_keys = copy(in_keys)
        if in_keys_inv is None:
            in_keys_inv = []  # default
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )
        self.allow_positive_dim = allow_positive_dim
        if dim >= 0 and not allow_positive_dim:
            raise RuntimeError(
                "dim should be smaller than 0 to accommodate for "
                "envs of different batch_sizes. Turn allow_positive_dim to accommodate "
                "for positive dim."
            )
        self._dim = dim

    @property
    def unsqueeze_dim(self) -> int:
        return self.dim

    @property
    def dim(self) -> int:
        if self._dim >= 0 and self.parent is not None:
            return len(self.parent.batch_size) + self._dim
        return self._dim

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.unsqueeze(self.dim)
        return observation

    def _inv_apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = observation.squeeze(self.dim)
        return observation

    def _transform_spec(self, spec: TensorSpec):
        space = spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            spec.shape = space.low.shape
        else:
            spec.shape = self._apply_transform(torch.zeros(spec.shape)).shape
        return spec

    # To map the specs, we actually use the forward call, not the inv
    _inv_transform_spec = _transform_spec

    @_apply_to_composite_inv
    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        return self._inv_transform_spec(action_spec)

    @_apply_to_composite_inv
    def transform_state_spec(self, state_spec: TensorSpec) -> TensorSpec:
        return self._inv_transform_spec(state_spec)

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        reward_key = self.parent.reward_key if self.parent is not None else "reward"
        if reward_key in self.in_keys:
            reward_spec = self._transform_spec(reward_spec)
        return reward_spec

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return self._transform_spec(observation_spec)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def __repr__(self) -> str:
        s = (
            f"{self.__class__.__name__}(dim={self.dim}, in_keys={self.in_keys}, out_keys={self.out_keys},"
            f" in_keys_inv={self.in_keys_inv}, out_keys_inv={self.out_keys_inv})"
        )
        return s


class SqueezeTransform(UnsqueezeTransform):
    """Removes a dimension of size one at the specified position.

    Args:
        dim (int): dimension to squeeze.
    """

    invertible = True

    def __init__(
        self,
        dim: int | None = None,
        *args,
        in_keys: Sequence[str] | None = None,
        out_keys: Sequence[str] | None = None,
        in_keys_inv: Sequence[str] | None = None,
        out_keys_inv: Sequence[str] | None = None,
        **kwargs,
    ):
        if dim is None:
            if "squeeze_dim" in kwargs:
                warnings.warn(
                    f"squeeze_dim will be deprecated in favor of dim arg in {type(self).__name__}."
                )
                dim = kwargs.pop("squeeze_dim")
            else:
                raise TypeError(
                    f"dim must be passed to {type(self).__name__} constructor."
                )

        super().__init__(
            dim,
            *args,
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
            **kwargs,
        )

    @property
    def squeeze_dim(self) -> int:
        return super().dim

    _apply_transform = UnsqueezeTransform._inv_apply_transform
    _inv_apply_transform = UnsqueezeTransform._apply_transform


class PermuteTransform(Transform):
    """Permutation transform.

    Permutes input tensors along the desired dimensions. The permutations
    must be provided along the feature dimension (not batch dimension).

    Args:
        dims (list of int): the permuted order of the dimensions. Must be a reordering
            of the dims ``[-(len(dims)), ..., -1]``.
        in_keys (list of NestedKeys): input entries (read).
        out_keys (list of NestedKeys): input entries (write). Defaults to ``in_keys`` if
            not provided.
        in_keys_inv (list of NestedKeys): input entries (read) during ``inv`` calls.
        out_keys_inv (list of NestedKeys): input entries (write) during ``inv`` calls. Defaults to ``in_keys_in`` if
            not provided.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> base_env = GymEnv("ALE/Pong-v5")
        >>> base_env.rollout(2)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([2, 6]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        pixels: Tensor(shape=torch.Size([2, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=cpu,
                    is_shared=False),
                pixels: Tensor(shape=torch.Size([2, 210, 160, 3]), device=cpu, dtype=torch.uint8, is_shared=False)},
            batch_size=torch.Size([2]),
            device=cpu,
            is_shared=False)
        >>> env = TransformedEnv(base_env, PermuteTransform((-1, -3, -2), in_keys=["pixels"]))
        >>> env.rollout(2)  # channels are at the end
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([2, 6]), device=cpu, dtype=torch.int64, is_shared=False),
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        pixels: Tensor(shape=torch.Size([2, 3, 210, 160]), device=cpu, dtype=torch.uint8, is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=cpu,
                    is_shared=False),
                pixels: Tensor(shape=torch.Size([2, 3, 210, 160]), device=cpu, dtype=torch.uint8, is_shared=False)},
            batch_size=torch.Size([2]),
            device=cpu,
            is_shared=False)

    """

    def __init__(
        self,
        dims,
        in_keys=None,
        out_keys=None,
        in_keys_inv=None,
        out_keys_inv=None,
    ):
        if in_keys is None:
            in_keys = []
        if out_keys is None:
            out_keys = copy(in_keys)
        if in_keys_inv is None:
            in_keys_inv = []
        if out_keys_inv is None:
            out_keys_inv = copy(in_keys_inv)

        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )
        # check dims
        self.dims = dims
        if sorted(dims) != list(range(-len(dims), 0)):
            raise ValueError(
                f"Only tailing dims with negative indices are supported by {self.__class__.__name__}. Got {dims} instead."
            )

    @staticmethod
    def _invert_permute(p):
        def _find_inv(i):
            for j, _p in enumerate(p):
                if _p < 0:
                    inv = True
                    _p = len(p) + _p
                else:
                    inv = False
                if i == _p:
                    if inv:
                        return j - len(p)
                    else:
                        return j
            else:
                # unreachable
                raise RuntimeError

        return [_find_inv(i) for i in range(len(p))]

    def _apply_transform(self, observation: torch.FloatTensor) -> torch.Tensor:
        observation = observation.permute(
            *list(range(observation.ndimension() - len(self.dims))), *self.dims
        )
        return observation

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        permuted_dims = self._invert_permute(self.dims)
        state = state.permute(
            *list(range(state.ndimension() - len(self.dims))), *permuted_dims
        )
        return state

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        observation_spec = self._edit_space(observation_spec)
        observation_spec.shape = torch.Size(
            [
                *observation_spec.shape[: -len(self.dims)],
                *[observation_spec.shape[dim] for dim in self.dims],
            ]
        )
        return observation_spec

    @_apply_to_composite_inv
    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        permuted_dims = self._invert_permute(self.dims)
        input_spec = self._edit_space_inv(input_spec)
        input_spec.shape = torch.Size(
            [
                *input_spec.shape[: -len(permuted_dims)],
                *[input_spec.shape[dim] for dim in permuted_dims],
            ]
        )
        return input_spec

    def _edit_space(self, spec: TensorSpec) -> None:
        if isinstance(spec.space, ContinuousBox):
            spec.space.high = self._apply_transform(spec.space.high)
            spec.space.low = self._apply_transform(spec.space.low)
        return spec

    def _edit_space_inv(self, spec: TensorSpec) -> None:
        if isinstance(spec.space, ContinuousBox):
            spec.space.high = self._inv_apply_transform(spec.space.high)
            spec.space.low = self._inv_apply_transform(spec.space.low)
        return spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class GrayScale(ObservationTransform):
    """Turns a pixel observation to grayscale."""

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)

    def _apply_transform(self, observation: torch.Tensor) -> torch.Tensor:
        observation = F.rgb_to_grayscale(observation)
        return observation

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
            observation_spec.shape = space.low.shape
        else:
            observation_spec.shape = self._apply_transform(
                torch.zeros(observation_spec.shape)
            ).shape
        return observation_spec

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class CatFrames(ObservationTransform):
    """Concatenates successive observation frames into a single tensor.

    This transform is useful for creating a sense of movement or velocity in the observed features.
    It can also be used with models that require access to past observations such as transformers and the like.
    It was first proposed in "Playing Atari with Deep Reinforcement Learning" (https://arxiv.org/pdf/1312.5602.pdf).

    When used within a transformed environment,
    :class:`CatFrames` is a stateful class, and it can be reset to its native state by
    calling the ``reset`` method. This method accepts tensordicts with a
    ``"_reset"`` entry that indicates which buffer to reset.

    Args:
        N (int): number of observation to concatenate.
        dim (int): dimension along which concatenate the
            observations. Should be negative, to ensure that it is compatible
            with environments of different batch_size.
        in_keys (sequence of NestedKey, optional): keys pointing to the frames that have
            to be concatenated. Defaults to ["pixels"].
        out_keys (sequence of NestedKey, optional): keys pointing to where the output
            has to be written. Defaults to the value of `in_keys`.
        padding (str, optional): the padding method. One of ``"same"`` or ``"constant"``.
            Defaults to ``"same"``, ie. the first value is used for padding.
        padding_value (:obj:`float`, optional): the value to use for padding if ``padding="constant"``.
            Defaults to 0.
        as_inverse (bool, optional): if ``True``, the transform is applied as an inverse transform. Defaults to ``False``.
        reset_key (NestedKey, optional): the reset key to be used as partial
            reset indicator. Must be unique. If not provided, defaults to the
            only reset key of the parent environment (if it has only one)
            and raises an exception otherwise.
        done_key (NestedKey, optional): the done key to be used as partial
            done indicator. Must be unique. If not provided, defaults to ``"done"``.
        future (bool, optional): if ``True``, each step's window gathers the
            ``N`` *upcoming* frames ``[t, t + 1, ..., t + N - 1]`` instead of
            the ``N`` most recent ones ``[t - N + 1, ..., t]``. With
            ``padding="same"`` the slots that run past the end of the
            trajectory repeat the last in-trajectory frame. Forward-looking
            windows require the full trajectory, so this mode is only
            available offline (replay buffer / data pipelines): attaching the
            transform to an environment raises a ``RuntimeError`` on the step
            path. Defaults to ``False``.

            .. versionadded:: 0.14
        mask_key (NestedKey, optional): if provided, the offline (forward /
            unfolding) path also writes a boolean mask of shape
            ``[*batch, time, N]`` flagging, for each window, the slots that
            were fabricated by padding (``True`` = padded slot, either out of
            the trajectory or out of the sampled window). This is the
            convention of the ``action_is_pad`` entry of chunked-action
            datasets. The mask is not available on the online (env step)
            path. Defaults to ``None`` (no mask is written).

            .. versionadded:: 0.14

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(GymEnv('Pendulum-v1'),
        ...     Compose(
        ...         UnsqueezeTransform(-1, in_keys=["observation"]),
        ...         CatFrames(N=4, dim=-1, in_keys=["observation"]),
        ...     )
        ... )
        >>> print(env.rollout(3))

    The :class:`CatFrames` transform can also be used offline to reproduce the
    effect of the online frame concatenation at a different scale (or for the
    purpose of limiting the memory consumption). The following example
    gives the complete picture, together with the usage of a :class:`torchrl.data.ReplayBuffer`:

    Examples:
        >>> from torchrl.modules import RandomPolicy        >>>         >>>         >>> from torchrl.envs import UnsqueezeTransform, CatFrames
        >>> from torchrl.collectors import Collector
        >>> # Create a transformed environment with CatFrames: notice the usage of UnsqueezeTransform to create an extra dimension
        >>> env = TransformedEnv(
        ...     GymEnv("CartPole-v1", from_pixels=True),
        ...     Compose(
        ...         ToTensorImage(in_keys=["pixels"], out_keys=["pixels_trsf"]),
        ...         Resize(in_keys=["pixels_trsf"], w=64, h=64),
        ...         GrayScale(in_keys=["pixels_trsf"]),
        ...         UnsqueezeTransform(-4, in_keys=["pixels_trsf"]),
        ...         CatFrames(dim=-4, N=4, in_keys=["pixels_trsf"]),
        ...     )
        ... )
        >>> # we design a collector
        >>> collector = Collector(
        ...     env,
        ...     RandomPolicy(env.action_spec),
        ...     frames_per_batch=10,
        ...     total_frames=1000,
        ... )
        >>> for data in collector:
        ...     print(data)
        ...     break
        >>> # now let's create a transform for the replay buffer. We don't need to unsqueeze the data here.
        >>> # however, we need to point to both the pixel entry at the root and at the next levels:
        >>> t = Compose(
        ...         ToTensorImage(in_keys=["pixels", ("next", "pixels")], out_keys=["pixels_trsf", ("next", "pixels_trsf")]),
        ...         Resize(in_keys=["pixels_trsf", ("next", "pixels_trsf")], w=64, h=64),
        ...         GrayScale(in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
        ...         CatFrames(dim=-4, N=4, in_keys=["pixels_trsf", ("next", "pixels_trsf")]),
        ... )
        >>> from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage
        >>> rb = TensorDictReplayBuffer(storage=LazyMemmapStorage(1000), transform=t, batch_size=16)
        >>> data_exclude = data.exclude("pixels_trsf", ("next", "pixels_trsf"))
        >>> rb.add(data_exclude)
        >>> s = rb.sample(1) # the buffer has only one element
        >>> # let's check that our sample is the same as the batch collected during inference
        >>> assert (data.exclude("collector")==s.squeeze(0).exclude("index", "collector")).all()

    .. note:: :class:`~CatFrames` currently only supports ``"done"``
        signal at the root. Nested ``done``,
        such as those found in MARL settings, are currently not supported.
        If this feature is needed, please raise an issue on TorchRL repo.

    .. note:: Storing stacks of frames in the replay buffer can significantly increase memory consumption (by N times).
        To mitigate this, you can store trajectories directly in the replay buffer and apply :class:`CatFrames` at sampling time.
        This approach involves sampling slices of the stored trajectories and then applying the frame stacking transform.
        For convenience, :class:`CatFrames` provides a :meth:`~.make_rb_transform_and_sampler` method that creates:

        - A modified version of the transform suitable for use in replay buffers
        - A corresponding :class:`SliceSampler` to use with the buffer

    .. seealso:: The offline (contiguous trajectory slice) windowing performed
        by this transform is also available as a pure functional,
        :func:`torchrl.envs.transforms.functional.cat_frames`, which operates
        directly on a plain tensor.

    """

    inplace = False
    _CAT_DIM_ERR = (
        "dim must be < 0 to accommodate for tensordict of "
        "different batch-sizes (since negative dims are batch invariant)."
    )
    ACCEPTED_PADDING = {"same", "constant", "zeros"}
    # class-level defaults double as fallbacks for instances pickled before
    # these options existed
    future = False
    mask_key = None

    def __init__(
        self,
        N: int,
        dim: int,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        padding="same",
        padding_value=0,
        as_inverse=False,
        reset_key: NestedKey | None = None,
        done_key: NestedKey | None = None,
        future: bool = False,
        mask_key: NestedKey | None = None,
    ):
        if in_keys is None:
            in_keys = IMAGE_KEYS
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.N = N
        self.future = bool(future)
        self.mask_key = mask_key
        if dim >= 0:
            raise ValueError(self._CAT_DIM_ERR)
        self.dim = dim
        if padding not in self.ACCEPTED_PADDING:
            raise ValueError(f"padding must be one of {self.ACCEPTED_PADDING}")
        if padding == "zeros":
            raise RuntimeError("Padding option 'zeros' will is deprecated")
        self.padding = padding
        self.padding_value = padding_value
        for in_key in self.in_keys:
            buffer_name = f"_cat_buffers_{in_key}"
            self.register_buffer(
                buffer_name,
                torch.nn.parameter.UninitializedBuffer(
                    device=torch.device("cpu"), dtype=torch.get_default_dtype()
                ),
            )
        # keeps track of calls to _reset since it's only _call that will populate the buffer
        self.as_inverse = as_inverse
        self.reset_key = reset_key
        self.done_key = done_key

    def make_rb_transform_and_sampler(
        self, batch_size: int, **sampler_kwargs
    ) -> tuple[Transform, torchrl.data.replay_buffers.SliceSampler]:  # noqa: F821
        """Creates a transform and sampler to be used with a replay buffer when storing frame-stacked data.

        This method helps reduce redundancy in stored data by avoiding the need to
        store the entire stack of frames in the buffer. Instead, it creates a
        transform that stacks frames on-the-fly during sampling, and a sampler that
        ensures the correct sequence length is maintained.

        Args:
            batch_size (int): The batch size to use for the sampler.
            **sampler_kwargs: Additional keyword arguments to pass to the
                :class:`~torchrl.data.replay_buffers.SliceSampler` constructor.

        Returns:
            A tuple containing:

                - transform (Transform): A transform that stacks frames on-the-fly during sampling.
                - sampler (SliceSampler): A sampler that ensures the correct sequence length is maintained.

        Example:
            >>> env = TransformedEnv(...)
            >>> catframes = CatFrames(N=4, ...)
            >>> transform, sampler = catframes.make_rb_transform_and_sampler(batch_size=32)
            >>> rb = ReplayBuffer(..., sampler=sampler, transform=transform)

        .. note:: When working with images, it's recommended to use distinct ``in_keys`` and ``out_keys`` in the preceding
            :class:`~torchrl.envs.ToTensorImage` transform. This ensures that the tensors stored in the buffer are separate
            from their processed counterparts, which we don't want to store.
            For non-image data, consider inserting a :class:`~torchrl.envs.RenameTransform` before :class:`CatFrames` to create
            a copy of the data that will be stored in the buffer.

        .. note:: When adding the transform to the replay buffer, one should pay attention to also pass the transforms
            that precede CatFrames, such as :class:`~torchrl.envs.ToTensorImage` or :class:`~torchrl.envs.UnsqueezeTransform`
            in such a way that the :class:`~torchrl.envs.CatFrames` transforms sees data formatted as it was during data
            collection.

        .. note:: For a more complete example, refer to torchrl's github repo `examples` folder:
            https://github.com/pytorch/rl/tree/main/examples/replay-buffers/catframes-in-buffer.py

        """
        from torchrl.data.replay_buffers import SliceSampler

        in_keys = self.in_keys
        in_keys = in_keys + [unravel_key(("next", key)) for key in in_keys]
        out_keys = self.out_keys
        out_keys = out_keys + [unravel_key(("next", key)) for key in out_keys]
        catframes = type(self)(
            N=self.N,
            in_keys=in_keys,
            out_keys=out_keys,
            dim=self.dim,
            padding=self.padding,
            padding_value=self.padding_value,
            as_inverse=False,
            reset_key=self.reset_key,
            done_key=self.done_key,
            future=self.future,
            mask_key=self.mask_key,
        )
        sampler = SliceSampler(slice_len=self.N, **sampler_kwargs)
        sampler._batch_size_multiplier = self.N
        transform = Compose(
            lambda td: td.reshape(-1, self.N),
            catframes,
            lambda td: td[:, -1],
            # We only store "pixels" to the replay buffer to save memory
            ExcludeTransform(*out_keys, inverse=True),
        )
        return transform, sampler

    @property
    def done_key(self):
        done_key = self.__dict__.get("_done_key", None)
        if done_key is None:
            done_key = "done"
            self._done_key = done_key
        return done_key

    @done_key.setter
    def done_key(self, value):
        self._done_key = value

    @property
    def reset_key(self):
        reset_key = getattr(self, "_reset_key", None)
        if reset_key is not None:
            return reset_key
        reset_keys = self.parent.reset_keys
        if len(reset_keys) > 1:
            raise RuntimeError(
                f"Got more than one reset key in env {self.container}, cannot infer which one to use. "
                f"Consider providing the reset key in the {type(self)} constructor."
            )
        reset_key = reset_keys[0]
        return reset_key

    @reset_key.setter
    def reset_key(self, value):
        self._reset_key = value

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Resets _buffers."""
        _reset = _get_reset(self.reset_key, tensordict)
        if self.as_inverse and self.parent is not None:
            raise Exception(
                "CatFrames as inverse is not supported as a transform for environments, only for replay buffers."
            )

        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset, _reset=_reset)

        return tensordict_reset

    def _reset_on_native_autoreset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        tensordict_reset = tensordict_reset.copy()
        for in_key in self.in_keys:
            buffer_name = f"_cat_buffers_{in_key}"
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                continue
            data = tensordict_reset.get(in_key)
            if data.size(self.dim) != buffer.size(self.dim):
                continue
            d = data.size(self.dim) // self.N
            dim = data.ndim + self.dim if self.dim < 0 else self.dim
            index = [slice(None, None) for _ in range(data.ndim)]
            index[dim] = slice(-d, None)
            tensordict_reset.set(in_key, data[tuple(index)])
        return self._reset(tensordict, tensordict_reset)

    def _make_missing_buffer(self, data, buffer_name):
        shape = list(data.shape)
        d = shape[self.dim]
        shape[self.dim] = d * self.N
        shape = torch.Size(shape)
        getattr(self, buffer_name).materialize(shape)
        buffer = (
            getattr(self, buffer_name)
            .to(dtype=data.dtype, device=data.device)
            .fill_(self.padding_value)
        )
        setattr(self, buffer_name, buffer)
        return buffer

    def _inv_call(self, tensordict: TensorDictBase) -> torch.Tensor:
        if self.as_inverse:
            return self.unfolding(tensordict)
        else:
            return tensordict

    def _call(self, next_tensordict: TensorDictBase, _reset=None) -> TensorDictBase:
        """Update the episode tensordict with max pooled keys."""
        if self.future:
            raise RuntimeError(
                "CatFrames(future=True) cannot run on the environment step "
                "path: forward-looking windows require the full trajectory "
                "and are only available offline (replay buffer / data "
                "pipelines)."
            )
        if self.mask_key is not None:
            raise RuntimeError(
                "CatFrames(mask_key=...) is only available offline (forward "
                "/ unfolding): the online step path does not build "
                "per-window validity masks."
            )
        _just_reset = _reset is not None
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            # Lazy init of buffers
            buffer_name = f"_cat_buffers_{in_key}"
            data = next_tensordict.get(in_key)
            d = data.size(self.dim)
            buffer = getattr(self, buffer_name)
            if isinstance(buffer, torch.nn.parameter.UninitializedBuffer):
                buffer = self._make_missing_buffer(data, buffer_name)
            # shift obs 1 position to the right
            if _just_reset:
                if _reset.all():
                    _all = True
                    data_reset = data
                    buffer_reset = buffer
                    dim = self.dim
                else:
                    _all = False
                    data_reset = data[_reset]
                    buffer_reset = buffer[_reset]
                    dim = self.dim - _reset.ndim + 1
                shape = [1 for _ in buffer_reset.shape]
                if _all:
                    shape[dim] = self.N
                else:
                    shape[dim] = self.N

                if self.padding == "same":
                    if _all:
                        buffer.copy_(data_reset.repeat(shape).clone())
                    else:
                        buffer[_reset] = data_reset.repeat(shape).clone()
                elif self.padding == "constant":
                    if _all:
                        buffer.fill_(self.padding_value)
                    else:
                        buffer[_reset] = self.padding_value
                else:
                    # make linter happy. An exception has already been raised
                    raise NotImplementedError

                if self.dim < 0:
                    n = buffer_reset.ndimension() + self.dim
                else:
                    raise ValueError(self._CAT_DIM_ERR)
                idx = tuple([slice(None, None) for _ in range(n)] + [slice(-d, None)])
                if not _all:
                    buffer_reset = buffer[_reset]
                buffer_reset[idx] = data_reset
                if not _all:
                    buffer[_reset] = buffer_reset
            else:
                buffer.copy_(torch.roll(buffer, shifts=-d, dims=self.dim))
                # add new obs
                if self.dim < 0:
                    n = buffer.ndimension() + self.dim
                else:
                    raise ValueError(self._CAT_DIM_ERR)
                idx = tuple([slice(None, None) for _ in range(n)] + [slice(-d, None)])
                buffer[idx] = buffer[idx].copy_(data)
            # add to tensordict
            next_tensordict.set(out_key, buffer.clone())
        return next_tensordict

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if self.future:
            raise RuntimeError(
                "CatFrames(future=True) cannot be attached to an "
                "environment: forward-looking windows require the full "
                "trajectory and are only available offline (replay buffer / "
                "data pipelines)."
            )
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = torch.cat([space.low] * self.N, self.dim)
            space.high = torch.cat([space.high] * self.N, self.dim)
            observation_spec.shape = space.low.shape
        else:
            shape = list(observation_spec.shape)
            shape[self.dim] = self.N * shape[self.dim]
            observation_spec.shape = torch.Size(shape)
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.as_inverse:
            return tensordict
        else:
            return self.unfolding(tensordict)

    def _apply_same_padding(self, dim, data, done_mask):
        # Kept for backward compatibility; delegates to the functional core.
        return F._apply_same_padding(dim, data, done_mask)

    @set_lazy_legacy(False)
    def unfolding(self, tensordict: TensorDictBase) -> TensorDictBase:
        # it is assumed that the last dimension of the tensordict is the time dimension
        if not tensordict.ndim:
            raise ValueError(
                "CatFrames cannot process unbatched tensordict instances. "
                "Make sure your input has more than one dimension and "
                "the time dimension is marked as 'time', e.g., "
                "`tensordict.refine_names(None, 'time', None)`."
            )
        i = 0
        for i, name in enumerate(tensordict.names):  # noqa: B007
            if name == "time":
                break
        else:
            warnings.warn(
                "The last dimension of the tensordict should be marked as 'time'. "
                "CatFrames will unfold the data along the time dimension assuming that "
                "the time dimension is the last dimension of the input tensordict. "
                "Define a 'time' dimension name (e.g., `tensordict.refine_names(..., 'time')`) to skip this warning. ",
                category=UserWarning,
            )
        tensordict_orig = tensordict
        if i != tensordict.ndim - 1:
            tensordict = tensordict.transpose(tensordict.ndim - 1, i)
        # first sort the in_keys with strings and non-strings
        keys = [
            (in_key, out_key)
            for in_key, out_key in _zip_strict(self.in_keys, self.out_keys)
            if isinstance(in_key, str)
        ]
        keys += [
            (in_key, out_key)
            for in_key, out_key in _zip_strict(self.in_keys, self.out_keys)
            if not isinstance(in_key, str)
        ]

        def unfold_done(done, N):
            prefix = (slice(None),) * (tensordict.ndim - 1)
            # the leading no-reset block is built explicitly rather than by
            # slicing ``done`` (which would cap it at the time length and
            # break windows longer than the trajectory, N > T)
            zeros_shape = list(done.shape)
            zeros_shape[tensordict.ndim - 1] = self.N - 1
            reset = torch.cat(
                [
                    torch.zeros(zeros_shape, dtype=done.dtype, device=done.device),
                    torch.ones_like(done[prefix + (slice(1),)]),
                    done[prefix + (slice(None, -1),)],
                ],
                tensordict.ndim - 1,
            )
            reset_unfold = reset.unfold(tensordict.ndim - 1, self.N, 1)
            reset_unfold_slice = reset_unfold[..., -1]
            reset_unfold_list = [torch.zeros_like(reset_unfold_slice)]
            for r in reversed(reset_unfold.unbind(-1)):
                reset_unfold_list.append(r | reset_unfold_list[-1])
                # reset_unfold_slice = reset_unfold_list[-1]
            reset_unfold = torch.stack(list(reversed(reset_unfold_list))[1:], -1)
            reset = reset[prefix + (slice(self.N - 1, None),)]
            reset[prefix + (0,)] = 1
            return reset_unfold, reset

        # The time axis is the last batch dim of (the possibly transposed)
        # ``tensordict``; the same index addresses it in every entry since the
        # batch dims lead the tensors.
        tdim = tensordict.ndim - 1
        done = tensordict.get(("next", self.done_key), default=None)
        if done is None:
            if not self.future:
                raise KeyError(
                    f"CatFrames.unfolding requires the {('next', self.done_key)} "
                    "entry to delimit trajectories. Make sure the sampled data "
                    "carries its done state, or use forward-looking windows "
                    "(future=True) to treat each batch row as a single "
                    "contiguous trajectory."
                )
            # Absent done in future mode: each batch row is one contiguous
            # trajectory and only the windows that run past its end are padded.
            done = torch.zeros(
                (*tensordict.shape, 1),
                dtype=torch.bool,
                device=tensordict.get(keys[0][0]).device,
            )
        if self.future:
            # Forward windows are backward windows of the time-reversed data:
            # the chunk ``[t, ..., t + N - 1]`` is the reversed window at
            # ``T - 1 - t`` read backwards. A boundary between steps ``t`` and
            # ``t + 1`` (``done[t]``) sits between reversed steps ``T - 2 - t``
            # and ``T - 1 - t``, hence the flip + shift; the rolled-in last
            # entry is never read (``unfold_done`` drops the final done).
            done = done.flip(tdim).roll(-1, dims=tdim)
        done_mask, reset = unfold_done(done, self.N)

        if self.mask_key is not None:
            mask = done_mask
            if self.future:
                mask = mask.flip(tdim).flip(-1)
            tensordict.set(self.mask_key, mask.reshape(*tensordict.shape, self.N))

        for in_key, out_key in keys:
            # check if we have an obs in "next" that has already been processed.
            # If so, we must add an offset
            data_orig = data = tensordict.get(in_key)
            n_feat = data_orig.shape[data.ndim + self.dim]
            first_val = None
            if isinstance(in_key, tuple) and in_key[0] == "next":
                # let's get the out_key we have already processed
                prev_out_key = dict(_zip_strict(self.in_keys, self.out_keys)).get(
                    in_key[1], None
                )
                if prev_out_key is not None:
                    prev_val = tensordict.get(prev_out_key)
                    # n_feat = prev_val.shape[data.ndim + self.dim] // self.N
                    first_val = prev_val.unflatten(
                        data.ndim + self.dim, (self.N, n_feat)
                    )
            if first_val is not None and self.future:
                raise NotImplementedError(
                    "CatFrames(future=True) does not support processing a "
                    "('next', key) entry alongside its root counterpart: the "
                    "one-step-offset fixup is only implemented for "
                    "history (backward) windows."
                )

            # The time axis sits at ``tensordict.ndim - 1`` within ``data`` (the
            # tensordict batch dims lead the tensor). Expressed relative to
            # ``data`` it is the following negative ``time_dim``. Delegate the
            # pure padding + sliding-window + done-mask concatenation to the
            # ``cat_frames`` functional so that the offline transform stays
            # byte-for-byte identical to its stateless core.
            time_dim = (tensordict.ndim - 1) - data.ndim
            if self.future:
                data = data.flip(tdim)
            data = F._cat_frames_windows(
                data,
                self.N,
                self.dim,
                padding=self.padding,
                padding_value=self.padding_value,
                time_dim=time_dim,
                done_mask=done_mask,
            )
            if self.future:
                # Back to forward time, windows read oldest-to-newest: undo
                # the time reversal and flip the window axis (which
                # ``_cat_frames_windows`` placed just before the cat axis).
                data = data.flip(tdim).flip(data.ndim + self.dim - 1)

            if first_val is not None:
                data0_pad = torch.full_like(
                    data_orig[tuple([slice(None)] * (tensordict.ndim - 1) + [0])],
                    self.padding_value,
                ).unsqueeze(tensordict.ndim - 1)
                data0 = [data0_pad] * (self.N - 1)
                # Aggregate reset along last dim
                reset_any = reset.any(-1, False)
                rexp = expand_right(
                    reset_any, (*reset_any.shape, *data.shape[data.ndim + self.dim :])
                )
                rexp = torch.cat(
                    [
                        torch.zeros_like(
                            data0[0].repeat_interleave(
                                len(data0), dim=tensordict.ndim - 1
                            ),
                            dtype=torch.bool,
                        ),
                        rexp,
                    ],
                    tensordict.ndim - 1,
                )
                rexp = rexp.unfold(tensordict.ndim - 1, self.N, 1)
                rexp_orig = rexp
                rexp = torch.cat([rexp[..., 1:], torch.zeros_like(rexp[..., -1:])], -1)
                if self.padding == "same":
                    rexp_orig = rexp_orig.flip(-1).cumsum(-1).flip(-1).bool()
                    rexp = rexp.flip(-1).cumsum(-1).flip(-1).bool()
                rexp_orig = torch.cat(
                    [torch.zeros_like(rexp_orig[..., -1:]), rexp_orig[..., 1:]], -1
                )
                rexp = rexp.permute(
                    *range(0, rexp.ndim + self.dim - 1),
                    -1,
                    *range(rexp.ndim + self.dim - 1, rexp.ndim - 1),
                )
                rexp_orig = rexp_orig.permute(
                    *range(0, rexp_orig.ndim + self.dim - 1),
                    -1,
                    *range(rexp_orig.ndim + self.dim - 1, rexp_orig.ndim - 1),
                )
                data[rexp] = first_val[rexp_orig]
            data = data.flatten(data.ndim + self.dim - 1, data.ndim + self.dim)
            tensordict.set(out_key, data)
        if tensordict_orig is not tensordict:
            tensordict_orig = tensordict.transpose(tensordict.ndim - 1, i)
        return tensordict_orig

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(N={self.N}, dim"
            f"={self.dim}, keys={self.in_keys})"
        )


class NextObservationDelta(Transform):
    """Stores ``("next", obs)`` as a low-precision delta in a sibling key.

    A single transform handles both sides of the compression:

    - **Env side** (``_step`` + ``_post_step_mdp_hooks``): for each
      in-key ``k``, write ``(next_obs - obs).to(delta_dtype)`` under
      the sibling key ``("next", "delta", k)``, then drop the full
      ``("next", k)`` from the post-step tensordict that the collector
      stacks. The full slot survives only long enough for
      :func:`~torchrl.envs.utils.step_mdp` to promote it to root, so the
      policy sees a full-precision observation on the next step.
    - **RB side** (``forward``): on
      :meth:`~torchrl.data.ReplayBuffer.sample`, reconstruct
      ``("next", k) = data[k] + data[("next", "delta", k)]`` and
      (optionally) drop the delta key. Unlike
      :class:`~torchrl.envs.transforms.NextStateReconstructor`, the
      delta encodes the actual transition, so trajectory-boundary
      transitions reconstruct exactly within the round-trip precision
      of ``delta_dtype`` rather than falling back to ``NaN``.

    Use the **same instance** (or two instances with matching ``in_keys``)
    on the env and on the replay buffer; the env-side and RB-side methods
    are dispatched automatically.

    Args:
        in_keys (sequence of NestedKey, optional): observation keys to
            compress. Defaults to ``None``, in which case the transform
            lazily walks ``parent.observation_spec`` and picks every
            floating-point leaf whose dtype is not in ``excluded_dtypes``.
            When the transform is used on a replay buffer (no env parent),
            ``in_keys`` must be passed explicitly.

    Keyword Args:
        delta_dtype (torch.dtype, optional): dtype in which the delta is
            stored. Must be a floating dtype. Defaults to ``torch.float16``.
        restore_dtype (torch.dtype or ``"root"``, optional): dtype of the
            reconstructed ``("next", k)`` on the RB side. ``"root"``
            (default) matches the dtype of the corresponding root key in
            the sampled batch.
        drop_delta (bool, optional): if ``True`` (default), the
            ``("next", "delta", k)`` entry is removed from the sampled
            tensordict after RB-side reconstruction so downstream consumers
            see the same key layout as an uncompressed pipeline.
        excluded_dtypes (tuple of torch.dtype, optional): dtypes to skip
            when auto-inferring ``in_keys``. Defaults to the integer +
            bool family.

    .. warning::
        The compression is **lossy**: round-tripping through ``delta_dtype``
        loses precision, particularly for unnormalized observations whose
        magnitudes exceed the dtype range or fall below its smallest
        representable step.

    .. warning::
        The transform must live **outside** any batched env
        (``TransformedEnv(ParallelEnv(N, factory), NextObservationDelta())``).
        Building a :class:`~torchrl.envs.SerialEnv` /
        :class:`~torchrl.envs.ParallelEnv` whose worker contains a
        ``NextObservationDelta`` raises at construction time.

    Example:
        >>> import torch
        >>> from torchrl.envs import GymEnv, TransformedEnv
        >>> from torchrl.envs.transforms import NextObservationDelta
        >>> env = TransformedEnv(GymEnv("Pendulum-v1"), NextObservationDelta())
        >>> td_root = env.reset()
        >>> _ = td_root.set("action", env.action_spec.rand())
        >>> td, td_ = env.step_and_maybe_reset(td_root)
        >>> td["next", "delta", "observation"].dtype
        torch.float16
        >>> ("next", "observation") in td.keys(True, True)
        False
        >>> td_["observation"].dtype
        torch.float32
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        *,
        delta_dtype: torch.dtype = torch.float16,
        restore_dtype: torch.dtype | Literal["root"] = "root",
        drop_delta: bool = True,
        excluded_dtypes: tuple[torch.dtype, ...] = (
            torch.uint8,
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.bool,
        ),
    ):
        if not delta_dtype.is_floating_point:
            raise ValueError(
                f"delta_dtype must be a floating-point dtype, got {delta_dtype}."
            )
        if restore_dtype != "root" and not (
            isinstance(restore_dtype, torch.dtype) and restore_dtype.is_floating_point
        ):
            raise ValueError(
                f"restore_dtype must be a floating-point dtype or 'root', got "
                f"{restore_dtype!r}."
            )
        self.delta_dtype = delta_dtype
        self.restore_dtype = restore_dtype
        self.drop_delta = drop_delta
        self.excluded_dtypes = tuple(excluded_dtypes)
        super().__init__(in_keys=in_keys, out_keys=in_keys)

    @property
    def in_keys(self) -> Sequence[NestedKey] | None:
        in_keys = self.__dict__.get("_in_keys", None)
        if in_keys is None:
            parent = self.parent
            if parent is None:
                return None
            in_keys = []
            for key, spec in parent.observation_spec.items(True, True):
                dtype = spec.dtype
                if dtype is None:
                    continue
                if dtype in self.excluded_dtypes:
                    continue
                if not dtype.is_floating_point:
                    continue
                in_keys.append(unravel_key(key))
            self._in_keys = in_keys
            if self.__dict__.get("_out_keys", None) is None:
                self._out_keys = copy(in_keys)
        return in_keys

    @in_keys.setter
    def in_keys(self, value: Sequence[NestedKey] | None) -> None:
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(v) for v in value]
        self._in_keys = value

    @property
    def out_keys(self) -> Sequence[NestedKey] | None:
        out_keys = self.__dict__.get("_out_keys", None)
        if out_keys is None:
            in_keys = self.in_keys
            if in_keys is None:
                return None
            out_keys = self._out_keys = copy(in_keys)
        return out_keys

    @out_keys.setter
    def out_keys(self, value: Sequence[NestedKey] | None) -> None:
        if value is not None:
            if isinstance(value, (str, tuple)):
                value = [value]
            value = [unravel_key(v) for v in value]
        self._out_keys = value

    @staticmethod
    def _as_key_tuple(key: NestedKey) -> tuple[str, ...]:
        if isinstance(key, str):
            return (key,)
        return tuple(key)

    def _delta_key(self, key: NestedKey) -> tuple[str, ...]:
        # `key` is a root-level observation key; the delta lives under
        # ("next", "delta", *key).
        return ("delta",) + self._as_key_tuple(key)

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        in_keys = self.in_keys
        if not in_keys:
            return next_tensordict
        for key in in_keys:
            obs = tensordict.get(key, default=None)
            next_obs = next_tensordict.get(key, default=None)
            if obs is None or next_obs is None:
                continue
            # Subtract in the source (typically full-precision) dtype, then
            # cast once. This loses fewer significant bits than casting each
            # operand to ``delta_dtype`` first and subtracting in low precision
            # (which would risk catastrophic cancellation for nearby values).
            delta = (next_obs - obs).to(self.delta_dtype)
            # Store the delta in a sibling sub-tensordict so a downstream
            # consumer cannot mistake it for a full-precision observation.
            next_tensordict.set(self._delta_key(key), delta)
        return next_tensordict

    def _post_step_mdp_hooks(
        self,
        tensordict: TensorDictBase,
        tensordict_: TensorDictBase,
    ) -> tuple[TensorDictBase, TensorDictBase]:
        # ``step_mdp`` has already promoted the still-full ``("next", k)`` to
        # root in ``tensordict_`` (because the delta key is not in the env's
        # observation spec, step_mdp leaves it alone). So the flowing td needs
        # no further work for ``k``. We just drop the full ``("next", k)``
        # from the post-step td so only the compressed delta survives into
        # the stacked rollout.
        in_keys = self.in_keys
        if not in_keys:
            return tensordict, tensordict_
        next_td = tensordict.get("next", default=None)
        if next_td is None:
            return tensordict, tensordict_
        for key in in_keys:
            key_tuple = self._as_key_tuple(key)
            if key_tuple in next_td.keys(include_nested=True, leaves_only=True):
                next_td.pop(key_tuple)
        return tensordict, tensordict_

    def _check_batched_worker_compat(self) -> None:
        raise RuntimeError(
            f"{type(self).__name__} cannot live inside a SerialEnv/ParallelEnv "
            "worker: the post-step-mdp delta key drop relies on the outer "
            "env's `step_and_maybe_reset` invoking the hook, but a batched "
            "env's `step_and_maybe_reset` does not propagate the worker's "
            "transform hook. Place the transform OUTSIDE the batched env "
            "instead, e.g. `TransformedEnv(ParallelEnv(N, base_env_factory), "
            f"{type(self).__name__}(...))`."
        )

    def transform_fake_tensordict(
        self, fake_tensordict: TensorDictBase
    ) -> TensorDictBase:
        # The runtime tensordict produced by `_step` + `_post_step_mdp_hooks`
        # has ``("next", "delta", k)`` (delta_dtype) and no ``("next", k)``.
        # Mirror that here so spec-vs-runtime key checks match.
        in_keys = self.in_keys
        if not in_keys:
            return fake_tensordict
        next_td = fake_tensordict.get("next", default=None)
        if next_td is None:
            return fake_tensordict
        for key in in_keys:
            key_tuple = self._as_key_tuple(key)
            leaf = next_td.get(key_tuple, default=None)
            if leaf is None:
                continue
            next_td.set(("delta",) + key_tuple, leaf.to(self.delta_dtype))
            next_td.pop(key_tuple)
        return fake_tensordict

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Reconstruct ``("next", k)`` from the stored delta at sample time.

        Invoked by :meth:`~torchrl.data.ReplayBuffer.sample` when this
        transform is appended to a replay buffer. Reads ``data[k]`` (root
        observation at step ``i``) and ``data[("next", "delta", k)]`` (the
        casted delta produced on the env side), writes
        ``data[("next", k)] = (data[k] + delta).to(restore_dtype)``, and
        (when ``drop_delta=True``, the default) removes the delta key.
        Keys for which either side is missing are silently skipped.
        """
        in_keys = self.in_keys
        if in_keys is None:
            # No env parent in RB context: explicit in_keys are required
            # so we know what to reconstruct.
            return tensordict
        for key in in_keys:
            key_tuple = self._as_key_tuple(key)
            delta_key = ("next", "delta") + key_tuple
            obs = tensordict.get(key_tuple, default=None)
            delta = tensordict.get(delta_key, default=None)
            if obs is None or delta is None:
                continue
            dtype = obs.dtype if self.restore_dtype == "root" else self.restore_dtype
            tensordict.set(("next",) + key_tuple, obs.to(dtype) + delta.to(dtype))
            if self.drop_delta:
                tensordict.pop(delta_key)
        return tensordict

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(in_keys={self.__dict__.get('_in_keys', None)}, "
            f"delta_dtype={self.delta_dtype})"
        )
