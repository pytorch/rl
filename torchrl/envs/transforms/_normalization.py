# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import abc

import multiprocessing as mp
import warnings
from collections import OrderedDict
from collections.abc import Sequence
from copy import copy
from typing import Any, TYPE_CHECKING

import torch

from tensordict import TensorDict, TensorDictBase
from tensordict.utils import _zip_strict, NestedKey
from torch import nn

from torchrl._utils import _append_last

from torchrl.data.tensor_specs import Bounded, ContinuousBox, TensorSpec, Unbounded
from torchrl.envs.common import EnvBase, make_tensordict
from torchrl.envs.transforms.utils import _set_missing_tolerance

if TYPE_CHECKING:
    pass

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

from torchrl.envs.transforms._base import (
    _apply_to_composite,
    _apply_to_composite_inv,
    Compose,
    ObservationTransform,
    Transform,
)

__all__ = [
    "ObservationNorm",
    "RewardScaling",
    "VecNorm",
]


class ObservationNorm(ObservationTransform):
    """Observation affine transformation layer.

    Normalizes an observation according to

    .. math::
        obs = obs * scale + loc

    Args:
        loc (number or tensor): location of the affine transform
        scale (number or tensor): scale of the affine transform
        in_keys (sequence of NestedKey, optional): entries to be normalized. Defaults to ["observation", "pixels"].
            All entries will be normalized with the same values: if a different behavior is desired
            (e.g. a different normalization for pixels and states) different :obj:`ObservationNorm`
            objects should be used.
        out_keys (sequence of NestedKey, optional): output entries. Defaults to the value of `in_keys`.
        in_keys_inv (sequence of NestedKey, optional): ObservationNorm also supports inverse transforms. This will
            only occur if a list of keys is provided to :obj:`in_keys_inv`. If none is provided,
            only the forward transform will be called.
        out_keys_inv (sequence of NestedKey, optional): output entries for the inverse transform.
            Defaults to the value of `in_keys_inv`.
        standard_normal (bool, optional): if ``True``, the transform will be

            .. math::
                obs = (obs-loc)/scale

            as it is done for standardization. Default is `False`.

        eps (:obj:`float`, optional): epsilon increment for the scale in the ``standard_normal`` case.
            Defaults to ``1e-6`` if not recoverable directly from the scale dtype.

    Examples:
        >>> torch.set_default_tensor_type(torch.DoubleTensor)
        >>> r = torch.randn(100, 3)*torch.randn(3) + torch.randn(3)
        >>> td = TensorDict({'obs': r}, [100])
        >>> transform = ObservationNorm(
        ...     loc = td.get('obs').mean(0),
        ...     scale = td.get('obs').std(0),
        ...     in_keys=["obs"],
        ...     standard_normal=True)
        >>> _ = transform(td)
        >>> print(torch.isclose(td.get('obs').mean(0),
        ...     torch.zeros(3)).all())
        tensor(True)
        >>> print(torch.isclose(td.get('next_obs').std(0),
        ...     torch.ones(3)).all())
        tensor(True)

    The normalization stats can be automatically computed:
    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> torch.manual_seed(0)
        >>> env = GymEnv("Pendulum-v1")
        >>> env = TransformedEnv(env, ObservationNorm(in_keys=["observation"]))
        >>> env.set_seed(0)
        >>> env.transform.init_stats(100)
        >>> print(env.transform.loc, env.transform.scale)
        tensor([-1.3752e+01, -6.5087e-03,  2.9294e-03], dtype=torch.float32) tensor([14.9636,  2.5608,  0.6408], dtype=torch.float32)

    """

    _ERR_INIT_MSG = "Cannot have an mixed initialized and uninitialized loc and scale"

    def __init__(
        self,
        loc: float | torch.Tensor | None = None,
        scale: float | torch.Tensor | None = None,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
        standard_normal: bool = False,
        eps: float | None = None,
    ):
        if in_keys is None:
            raise RuntimeError(
                "Not passing in_keys to ObservationNorm is a deprecated behavior."
            )

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
        if not isinstance(standard_normal, torch.Tensor):
            standard_normal = torch.as_tensor(standard_normal)
        self.register_buffer("standard_normal", standard_normal)
        self.eps = (
            eps
            if eps is not None
            else torch.finfo(scale.dtype).eps
            if isinstance(scale, torch.Tensor) and scale.dtype.is_floating_point
            else 1e-6
        )

        if loc is not None and not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc, dtype=torch.get_default_dtype())
        elif loc is None:
            if scale is not None:
                raise ValueError(self._ERR_INIT_MSG)
            loc = nn.UninitializedBuffer()

        if scale is not None and not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.get_default_dtype())
            scale = scale.clamp_min(self.eps)
        elif scale is None:
            # check that loc is None too
            if not isinstance(loc, nn.UninitializedBuffer):
                raise ValueError(self._ERR_INIT_MSG)
            scale = nn.UninitializedBuffer()

        # self.observation_spec_key = observation_spec_key
        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale)

    @property
    def initialized(self) -> bool:
        return not isinstance(self.loc, nn.UninitializedBuffer)

    def init_stats(
        self,
        num_iter: int,
        reduce_dim: int | tuple[int] = 0,
        cat_dim: int | None = None,
        key: NestedKey | None = None,
        keep_dims: tuple[int] | None = None,
    ) -> None:
        """Initializes the loc and scale stats of the parent environment.

        Normalization constant should ideally make the observation statistics approach
        those of a standard Gaussian distribution. This method computes a location
        and scale tensor that will empirically compute the mean and standard
        deviation of a Gaussian distribution fitted on data generated randomly with
        the parent environment for a given number of steps.

        Args:
            num_iter (int): number of random iterations to run in the environment.
            reduce_dim (int or tuple of int, optional): dimension to compute the mean and std over.
                Defaults to 0.
            cat_dim (int, optional): dimension along which the batches collected will be concatenated.
                It must be part equal to reduce_dim (if integer) or part of the reduce_dim tuple.
                Defaults to the same value as reduce_dim.
            key (NestedKey, optional): if provided, the summary statistics will be
                retrieved from that key in the resulting tensordicts.
                Otherwise, the first key in :obj:`ObservationNorm.in_keys` will be used.
            keep_dims (tuple of int, optional): the dimensions to keep in the loc and scale.
                For instance, one may want the location and scale to have shape [C, 1, 1]
                when normalizing a 3D tensor over the last two dimensions, but not the
                third. Defaults to None.

        """
        if cat_dim is None:
            cat_dim = reduce_dim
            if not isinstance(cat_dim, int):
                raise ValueError(
                    "cat_dim must be specified if reduce_dim is not an integer."
                )
        if (isinstance(reduce_dim, tuple) and cat_dim not in reduce_dim) or (
            isinstance(reduce_dim, int) and cat_dim != reduce_dim
        ):
            raise ValueError("cat_dim must be part of or equal to reduce_dim.")
        if self.initialized:
            raise RuntimeError(
                f"Loc/Scale are already initialized: ({self.loc}, {self.scale})"
            )

        if len(self.in_keys) > 1 and key is None:
            raise RuntimeError(
                "Transform has multiple in_keys but no specific key was passed as an argument"
            )
        key = self.in_keys[0] if key is None else key

        def raise_initialization_exception(module):
            if isinstance(module, ObservationNorm) and not module.initialized:
                raise RuntimeError(
                    "ObservationNorms need to be initialized in the right order."
                    "Trying to initialize an ObservationNorm "
                    "while a parent ObservationNorm transform is still uninitialized"
                )

        parent = self.parent
        if parent is None:
            raise RuntimeError(
                "Cannot initialize the transform if parent env is not defined."
            )
        parent.apply(raise_initialization_exception)

        collected_frames = 0
        data = []
        while collected_frames < num_iter:
            tensordict = parent.rollout(max_steps=num_iter)
            collected_frames += tensordict.numel()
            data.append(tensordict.get(key))

        data = torch.cat(data, cat_dim)
        if isinstance(reduce_dim, int):
            reduce_dim = [reduce_dim]
        # make all reduce_dim and keep_dims negative
        reduce_dim = sorted(dim if dim < 0 else dim - data.ndim for dim in reduce_dim)

        if keep_dims is not None:
            keep_dims = sorted(dim if dim < 0 else dim - data.ndim for dim in keep_dims)
            if not all(k in reduce_dim for k in keep_dims):
                raise ValueError("keep_dim elements must be part of reduce_dim list.")
        else:
            keep_dims = []
        loc = data.mean(reduce_dim, keepdim=True)
        scale = data.std(reduce_dim, keepdim=True)
        for r in reduce_dim:
            if r not in keep_dims:
                loc = loc.squeeze(r)
                scale = scale.squeeze(r)

        if not self.standard_normal:
            scale = 1 / scale.clamp_min(self.eps)
            loc = -loc * scale

        if not torch.isfinite(loc).all():
            raise RuntimeError("Non-finite values found in loc")
        if not torch.isfinite(scale).all():
            raise RuntimeError("Non-finite values found in scale")
        self.loc.materialize(shape=loc.shape, dtype=loc.dtype)
        self.loc.copy_(loc)
        self.scale.materialize(shape=scale.shape, dtype=scale.dtype)
        self.scale.copy_(scale.clamp_min(self.eps))

    def _apply_transform(self, obs: torch.Tensor) -> torch.Tensor:
        if not self.initialized:
            raise RuntimeError(
                "Loc/Scale have not been initialized. Either pass in values in the constructor "
                "or call the init_stats method"
            )
        if self.standard_normal:
            loc = self.loc
            scale = self.scale
            return (obs - loc) / scale
        else:
            scale = self.scale
            loc = self.loc
            return obs * scale + loc

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        if self.loc is None or self.scale is None:
            raise RuntimeError(
                "Loc/Scale have not been initialized. Either pass in values in the constructor "
                "or call the init_stats method"
            )
        if not self.standard_normal:
            loc = self.loc
            scale = self.scale
            return (state - loc) / scale
        else:
            scale = self.scale
            loc = self.loc
            return state * scale + loc

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        space = observation_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
        return observation_spec

    # @_apply_to_composite_inv
    # def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
    #     space = input_spec.space
    #     if isinstance(space, ContinuousBox):
    #         space.low = self._apply_transform(space.low)
    #         space.high = self._apply_transform(space.high)
    #     return input_spec

    @_apply_to_composite_inv
    def transform_action_spec(self, action_spec: TensorSpec) -> TensorSpec:
        space = action_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
        return action_spec

    @_apply_to_composite_inv
    def transform_state_spec(self, state_spec: TensorSpec) -> TensorSpec:
        space = state_spec.space
        if isinstance(space, ContinuousBox):
            space.low = self._apply_transform(space.low)
            space.high = self._apply_transform(space.high)
        return state_spec

    def __repr__(self) -> str:
        if self.initialized and (self.loc.numel() == 1 and self.scale.numel() == 1):
            return (
                f"{self.__class__.__name__}("
                f"loc={float(self.loc):4.4f}, scale"
                f"={float(self.scale):4.4f}, keys={self.in_keys})"
            )
        else:
            return super().__repr__()

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset


class RewardScaling(Transform):
    """Affine transform of the reward.

     The reward is transformed according to:

    .. math::
        reward = reward * scale + loc

    Args:
        loc (number or torch.Tensor): location of the affine transform
        scale (number or torch.Tensor): scale of the affine transform
        standard_normal (bool, optional): if ``True``, the transform will be

            .. math::
                reward = (reward-loc)/scale

            as it is done for standardization. Default is `False`.
    """

    def __init__(
        self,
        loc: float | torch.Tensor,
        scale: float | torch.Tensor,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        standard_normal: bool = False,
    ):
        if in_keys is None:
            in_keys = ["reward"]
        if out_keys is None:
            out_keys = copy(in_keys)

        super().__init__(in_keys=in_keys, out_keys=out_keys)
        if not isinstance(standard_normal, torch.Tensor):
            standard_normal = torch.tensor(standard_normal)
        self.register_buffer("standard_normal", standard_normal)

        if not isinstance(loc, torch.Tensor):
            loc = torch.tensor(loc)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale)

        self.register_buffer("loc", loc)
        self.register_buffer("scale", scale.clamp_min(1e-6))

    def _apply_transform(self, reward: torch.Tensor) -> torch.Tensor:
        if self.standard_normal:
            loc = self.loc
            scale = self.scale
            reward = (reward - loc) / scale
            return reward
        else:
            scale = self.scale
            loc = self.loc
            reward = reward * scale + loc
            return reward

    @_apply_to_composite
    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        if isinstance(reward_spec, Unbounded):
            return reward_spec
        else:
            raise NotImplementedError(
                f"{self.__class__.__name__}.transform_reward_spec not "
                f"implemented for tensor spec of type"
                f" {type(reward_spec).__name__}"
            )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"loc={self.loc.item():4.4f}, scale={self.scale.item():4.4f}, "
            f"keys={self.in_keys})"
        )


def _sum_left(val, dest):
    while val.ndimension() > dest.ndimension():
        val = val.sum(0)
    return val


class _VecNormMeta(abc.ABCMeta):
    def __call__(cls, *args, **kwargs):
        new_api = kwargs.pop("new_api", None)
        if new_api is None:
            warnings.warn(
                "The VecNorm class is to be deprecated in favor of `torchrl.envs.VecNormV2` and will be replaced by "
                "that class in v0.10. You can adapt to these changes by using the `new_api` argument or importing "
                "the `VecNormV2` class from `torchrl.envs`.",
                category=FutureWarning,
            )
            new_api = False
        if new_api:
            from torchrl.envs import VecNormV2

            return VecNormV2(*args, **kwargs)
        return super().__call__(*args, **kwargs)


class VecNorm(Transform, metaclass=_VecNormMeta):
    """Moving average normalization layer for torchrl environments.

    .. warning:: This class is to be deprecated in favor of :class:`~torchrl.envs.VecNormV2` and will be replaced by
        that class in v0.10. You can adapt to these changes by using the `new_api` argument or importing the
        `VecNormV2` class from `torchrl.envs`.

    VecNorm keeps track of the summary statistics of a dataset to standardize
    it on-the-fly. If the transform is in 'eval' mode, the running
    statistics are not updated.

    If multiple processes are running a similar environment, one can pass a
    TensorDictBase instance that is placed in shared memory: if so, every time
    the normalization layer is queried it will update the values for all
    processes that share the same reference.

    To use VecNorm at inference time and avoid updating the values with the new
    observations, one should substitute this layer by :meth:`~.to_observation_norm`.
    This will provide a static version of `VecNorm` which will not be updated
    when the source transform is updated.
    To get a frozen copy of the VecNorm layer, see :meth:`~.frozen_copy`.

    Args:
        in_keys (sequence of NestedKey, optional): keys to be updated.
            default: ["observation", "reward"]
        out_keys (sequence of NestedKey, optional): destination keys.
            Defaults to ``in_keys``.
        shared_td (TensorDictBase, optional): A shared tensordict containing the
            keys of the transform.
        lock (mp.Lock): a lock to prevent race conditions between processes.
            Defaults to None (lock created during init).
        decay (number, optional): decay rate of the moving average.
            default: 0.99
        eps (number, optional): lower bound of the running standard
            deviation (for numerical underflow). Default is 1e-4.
        shapes (List[torch.Size], optional): if provided, represents the shape
            of each in_keys. Its length must match the one of ``in_keys``.
            Each shape must match the trailing dimension of the corresponding
            entry.
            If not, the feature dimensions of the entry (ie all dims that do
            not belong to the tensordict batch-size) will be considered as
            feature dimension.
        new_api (bool or None, optional): if ``True``, an instance of VecNormV2 will be returned.
            If not passed, a warning will be raised.
            Defaults to ``False``.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> t = VecNorm(decay=0.9)
        >>> env = GymEnv("Pendulum-v0")
        >>> env = TransformedEnv(env, t)
        >>> tds = []
        >>> for _ in range(1000):
        ...     td = env.rand_step()
        ...     if td.get("done"):
        ...         _ = env.reset()
        ...     tds += [td]
        >>> tds = torch.stack(tds, 0)
        >>> print((abs(tds.get(("next", "observation")).mean(0))<0.2).all())
        tensor(True)
        >>> print((abs(tds.get(("next", "observation")).std(0)-1)<0.2).all())
        tensor(True)

    To recover the original (denormalized) values from normalized data, use :meth:`~.denorm`:

        >>> denormed = t.denorm(tds)

    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        shared_td: TensorDictBase | None = None,
        lock: mp.Lock = None,
        decay: float = 0.9999,
        eps: float = 1e-4,
        shapes: list[torch.Size] = None,
        new_api: bool | None = None,
    ) -> None:

        warnings.warn(
            "This class is to be deprecated in favor of :class:`~torchrl.envs.VecNormV2`.",
            category=FutureWarning,
        )

        # Warn about shared memory limitations on older PyTorch
        from packaging.version import parse as parse_version

        if (
            parse_version(torch.__version__).base_version < "2.8.0"
            and shared_td is not None
        ):
            warnings.warn(
                "VecNorm with shared memory (shared_td) may not synchronize correctly "
                "across processes on PyTorch < 2.8 when using the 'spawn' multiprocessing "
                "start method. This is due to limitations in PyTorch's shared memory "
                "implementation with the 'file_system' sharing strategy. "
                "Consider upgrading to PyTorch >= 2.8 for full shared memory support.",
                category=UserWarning,
            )

        if lock is None:
            lock = mp.Lock()
        if in_keys is None:
            in_keys = ["observation", "reward"]
        if out_keys is None:
            out_keys = copy(in_keys)
        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self._td = shared_td
        if shared_td is not None and not (
            shared_td.is_shared() or shared_td.is_memmap()
        ):
            raise RuntimeError(
                "shared_td must be either in shared memory or a memmap " "tensordict."
            )
        if shared_td is not None:
            for key in in_keys:
                if (
                    (_append_last(key, "_sum") not in shared_td.keys())
                    or (_append_last(key, "_ssq") not in shared_td.keys())
                    or (_append_last(key, "_count") not in shared_td.keys())
                ):
                    raise KeyError(
                        f"key {key} not present in the shared tensordict "
                        f"with keys {shared_td.keys()}"
                    )

        self.lock = lock
        self.decay = decay
        self.shapes = shapes
        self.eps = eps
        self.frozen = False

    def freeze(self) -> VecNorm:
        """Freezes the VecNorm, avoiding the stats to be updated when called.

        See :meth:`~.unfreeze`.
        """
        self.frozen = True
        return self

    def unfreeze(self) -> VecNorm:
        """Unfreezes the VecNorm.

        See :meth:`~.freeze`.
        """
        self.frozen = False
        return self

    def frozen_copy(self) -> VecNorm:
        """Returns a copy of the Transform that keeps track of the stats but does not update them."""
        if self._td is None:
            raise RuntimeError(
                "Make sure the VecNorm has been initialized before creating a frozen copy."
            )
        clone = self.clone()
        # replace values
        clone._td = self._td.copy()
        # freeze
        return clone.freeze()

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # TODO: remove this decorator when trackers are in data
        with _set_missing_tolerance(self, True):
            return self._call(tensordict_reset)
        return tensordict_reset

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if self.lock is not None:
            self.lock.acquire()

        for key, key_out in _zip_strict(self.in_keys, self.out_keys):
            if key not in next_tensordict.keys(include_nested=True):
                # TODO: init missing rewards with this
                # for key_suffix in [_append_last(key, suffix) for suffix in ("_sum", "_ssq", "_count")]:
                #     tensordict.set(key_suffix, self.container.observation_spec[key_suffix].zero())
                continue
            self._init(next_tensordict, key)
            # update and standardize
            new_val = self._update(
                key, next_tensordict.get(key), N=max(1, next_tensordict.numel())
            )

            next_tensordict.set(key_out, new_val)

        if self.lock is not None:
            self.lock.release()

        return next_tensordict

    forward = _call

    def _init(self, tensordict: TensorDictBase, key: str) -> None:
        if self._td is None or _append_last(key, "_sum") not in self._td.keys(True):
            if key is not key and key in tensordict.keys():
                raise RuntimeError(
                    f"Conflicting key names: {key} from VecNorm and input tensordict keys."
                )
            if self.shapes is None:
                td_view = tensordict.view(-1)
                td_select = td_view[0]
                item = td_select.get(key)
                d = {_append_last(key, "_sum"): torch.zeros_like(item)}
                d.update({_append_last(key, "_ssq"): torch.zeros_like(item)})
            else:
                idx = 0
                for in_key in self.in_keys:
                    if in_key != key:
                        idx += 1
                    else:
                        break
                shape = self.shapes[idx]
                item = tensordict.get(key)
                d = {
                    _append_last(key, "_sum"): torch.zeros(
                        shape, device=item.device, dtype=item.dtype
                    )
                }
                d.update(
                    {
                        _append_last(key, "_ssq"): torch.zeros(
                            shape, device=item.device, dtype=item.dtype
                        )
                    }
                )

            d.update(
                {
                    _append_last(key, "_count"): torch.zeros(
                        1, device=item.device, dtype=torch.float
                    )
                }
            )
            if self._td is None:
                self._td = TensorDict(d, batch_size=[])
            else:
                self._td.update(d)
        else:
            pass

    def _update(self, key, value, N) -> torch.Tensor:
        # TODO: we should revert this and have _td be like: TensorDict{"sum": ..., "ssq": ..., "count"...})
        #  to facilitate the computation of the stats using TD internals.
        #  Moreover, _td can be locked so these ops will be very fast on CUDA.
        _sum = self._td.get(_append_last(key, "_sum"))
        _ssq = self._td.get(_append_last(key, "_ssq"))
        _count = self._td.get(_append_last(key, "_count"))

        value_sum = _sum_left(value, _sum)

        if not self.frozen:
            _sum *= self.decay
            _sum += value_sum
            self._td.set_(
                _append_last(key, "_sum"),
                _sum,
            )

        _ssq = self._td.get(_append_last(key, "_ssq"))
        value_ssq = _sum_left(value.pow(2), _ssq)
        if not self.frozen:
            _ssq *= self.decay
            _ssq += value_ssq
            self._td.set_(
                _append_last(key, "_ssq"),
                _ssq,
            )

        _count = self._td.get(_append_last(key, "_count"))
        if not self.frozen:
            _count *= self.decay
            _count += N
            self._td.set_(
                _append_last(key, "_count"),
                _count,
            )

        mean = _sum / _count
        std = (_ssq / _count - mean.pow(2)).clamp_min(self.eps).sqrt()
        return (value - mean) / std.clamp_min(self.eps)

    def denorm(self, tensordict: TensorDictBase) -> TensorDictBase:
        """Denormalize a tensordict using the inverse of the normalization transform.

        Applies the inverse of the normalization: ``original = normalized * scale + loc``.

        Reads normalized values from ``out_keys`` and writes denormalized values to ``in_keys``.

        Args:
            tensordict (TensorDictBase): the tensordict containing normalized values.

        Returns:
            A shallow copy of the tensordict with denormalized values written to ``in_keys``.

        Raises:
            RuntimeError: if the transform has not been initialized (no data seen yet).

        Examples:
            >>> from torchrl.envs import GymEnv, VecNorm
            >>> env = GymEnv("Pendulum-v1")
            >>> vecnorm = VecNorm(in_keys=["observation"], out_keys=["observation_norm"])
            >>> env = env.append_transform(vecnorm)
            >>> # Collect some data to initialize statistics
            >>> rollout = env.rollout(10)
            >>> # Denormalize the normalized observations
            >>> denormed = vecnorm.denorm(rollout)
            >>> # denormed["observation"] now contains the original scale values

        """
        if self._td is None:
            raise RuntimeError("VecNorm must be initialized before calling denorm.")

        tensordict = tensordict.copy()
        loc, scale = self._get_loc_scale()
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            if out_key not in tensordict.keys(include_nested=True):
                continue
            value = tensordict.get(out_key)
            # Denormalize: value * scale + loc
            original_value = value * scale.get(in_key) + loc.get(in_key)
            tensordict.set(in_key, original_value)
        return tensordict

    def to_observation_norm(self) -> Compose | ObservationNorm:
        """Converts VecNorm into an ObservationNorm class that can be used at inference time.

        The :class:`~torchrl.envs.ObservationNorm` layer can be updated using the :meth:`~torch.nn.Module.state_dict`
        API.

        Examples:
            >>> from torchrl.envs import GymEnv, VecNorm
            >>> vecnorm = VecNorm(in_keys=["observation"])
            >>> train_env = GymEnv("CartPole-v1", device=None).append_transform(
            ...     vecnorm)
            >>>
            >>> r = train_env.rollout(4)
            >>>
            >>> eval_env = GymEnv("CartPole-v1").append_transform(
            ...     vecnorm.to_observation_norm())
            >>> print(eval_env.transform.loc, eval_env.transform.scale)
            >>>
            >>> r = train_env.rollout(4)
            >>> # Update entries with state_dict
            >>> eval_env.transform.load_state_dict(
            ...     vecnorm.to_observation_norm().state_dict())
            >>> print(eval_env.transform.loc, eval_env.transform.scale)

        """
        out = []
        loc = self.loc
        scale = self.scale
        for key, key_out in _zip_strict(self.in_keys, self.out_keys):
            _out = ObservationNorm(
                loc=loc.get(key),
                scale=scale.get(key),
                standard_normal=True,
                in_keys=key,
                out_keys=key_out,
            )
            out += [_out]
        if len(self.in_keys) > 1:
            return Compose(*out)
        return _out

    def _get_loc_scale(
        self, loc_only=False, scale_only=False
    ) -> tuple[TensorDict | None, TensorDict | None]:
        loc = {}
        scale = {}
        for key in self.in_keys:
            _sum = self._td.get(_append_last(key, "_sum"))
            _ssq = self._td.get(_append_last(key, "_ssq"))
            _count = self._td.get(_append_last(key, "_count"))
            loc[key] = _sum / _count
            scale[key] = (_ssq / _count - loc[key].pow(2)).clamp_min(self.eps).sqrt()
        if not scale_only:
            loc = TensorDict(loc)
        else:
            loc = None
        if not loc_only:
            scale = TensorDict(scale)
        else:
            scale = None
        return loc, scale

    @property
    def standard_normal(self) -> bool:
        """Whether the affine transform given by `loc` and `scale` follows the standard normal equation.

        Similar to :class:`~torchrl.envs.ObservationNorm` standard_normal attribute.

        Always returns ``True``.
        """
        return True

    @property
    def loc(self):
        """Returns a TensorDict with the loc to be used for an affine transform."""
        # We can't cache that value bc the summary stats could be updated by a different process
        loc, _ = self._get_loc_scale(loc_only=True)
        return loc

    @property
    def scale(self):
        """Returns a TensorDict with the scale to be used for an affine transform."""
        # We can't cache that value bc the summary stats could be updated by a different process
        _, scale = self._get_loc_scale(scale_only=True)
        return scale

    @staticmethod
    def build_td_for_shared_vecnorm(
        env: EnvBase,
        keys: Sequence[str] | None = None,
        memmap: bool = False,
    ) -> TensorDictBase:
        """Creates a shared tensordict for normalization across processes.

        Args:
            env (EnvBase): example environment to be used to create the
                tensordict
            keys (sequence of NestedKey, optional): keys that
                have to be normalized. Default is `["next", "reward"]`
            memmap (bool): if ``True``, the resulting tensordict will be cast into
                memory map (using `memmap_()`). Otherwise, the tensordict
                will be placed in shared memory.

        Returns:
            A memory in shared memory to be sent to each process.

        Examples:
            >>> from torch import multiprocessing as mp
            >>> queue = mp.Queue()
            >>> env = make_env()
            >>> td_shared = VecNorm.build_td_for_shared_vecnorm(env,
            ...     ["next", "reward"])
            >>> assert td_shared.is_shared()
            >>> queue.put(td_shared)
            >>> # on workers
            >>> v = VecNorm(shared_td=queue.get())
            >>> env = TransformedEnv(make_env(), v)

        """
        raise NotImplementedError("this feature is currently put on hold.")
        sep = ".-|-."
        if keys is None:
            keys = ["next", "reward"]
        td = make_tensordict(env)
        keys = {key for key in td.keys() if key in keys}
        td_select = td.select(*keys)
        td_select = td_select.flatten_keys(sep)
        if td.batch_dims:
            raise RuntimeError(
                f"VecNorm should be used with non-batched environments. "
                f"Got batch_size={td.batch_size}"
            )
        keys = list(td_select.keys())
        for key in keys:
            td_select.set(_append_last(key, "_ssq"), td_select.get(key).clone())
            td_select.set(
                _append_last(key, "_count"),
                torch.zeros(
                    *td.batch_size,
                    1,
                    device=td_select.device,
                    dtype=torch.float,
                ),
            )
            td_select.rename_key_(key, _append_last(key, "_sum"))
        td_select.exclude(*keys).zero_()
        td_select = td_select.unflatten_keys(sep)
        if memmap:
            return td_select.memmap_()
        return td_select.share_memory_()

    # We use a different separator to ensure that keys can have points within them.
    SEP = "-<.>-"

    def get_extra_state(self) -> OrderedDict:
        if self._td is None:
            warnings.warn(
                "Querying state_dict on an uninitialized VecNorm transform will "
                "return a `None` value for the summary statistics. "
                "Loading such a state_dict on an initialized VecNorm will result in "
                "an error."
            )
            return
        return self._td.flatten_keys(self.SEP).to_dict()

    def set_extra_state(self, state: OrderedDict) -> None:
        if state is not None:
            td = TensorDict(state).unflatten_keys(self.SEP)
            if self._td is None and not td.is_shared():
                warnings.warn(
                    "VecNorm wasn't initialized and the tensordict is not shared. In single "
                    "process settings, this is ok, but if you need to share the statistics "
                    "between workers this should require some attention. "
                    "Make sure that the content of VecNorm is transmitted to the workers "
                    "after calling load_state_dict and not before, as other workers "
                    "may not have access to the loaded TensorDict."
                )
                td.share_memory_()
            if self._td is not None:
                self._td.update_(td)
            else:
                self._td = td
        elif self._td is not None:
            raise KeyError("Could not find a tensordict in the state_dict.")

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(decay={self.decay:4.4f},"
            f"eps={self.eps:4.4f}, in_keys={self.in_keys}, out_keys={self.out_keys})"
        )

    def __getstate__(self) -> dict[str, Any]:
        state = super().__getstate__()
        _lock = state.pop("lock", None)
        if _lock is not None:
            state["lock_placeholder"] = None
        return state

    def __setstate__(self, state: dict[str, Any]):
        if "lock_placeholder" in state:
            state.pop("lock_placeholder")
            _lock = mp.Lock()
            state["lock"] = _lock
        super().__setstate__(state)

    @_apply_to_composite
    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if isinstance(observation_spec, Bounded):
            return Unbounded(
                shape=observation_spec.shape,
                dtype=observation_spec.dtype,
                device=observation_spec.device,
            )
        return observation_spec

    # TODO: incorporate this when trackers are part of the data
    # def transform_output_spec(self, output_spec: TensorSpec) -> TensorSpec:
    #     observation_spec = output_spec["full_observation_spec"]
    #     reward_spec = output_spec["full_reward_spec"]
    #     for key in list(observation_spec.keys(True, True)):
    #         if key in self.in_keys:
    #             observation_spec[_append_last(key, "_sum")] = observation_spec[key].clone()
    #             observation_spec[_append_last(key, "_ssq")] = observation_spec[key].clone()
    #             observation_spec[_append_last(key, "_count")] = observation_spec[key].clone()
    #     for key in list(reward_spec.keys(True, True)):
    #         if key in self.in_keys:
    #             observation_spec[_append_last(key, "_sum")] = reward_spec[key].clone()
    #             observation_spec[_append_last(key, "_ssq")] = reward_spec[key].clone()
    #             observation_spec[_append_last(key, "_count")] = reward_spec[key].clone()
    #     return output_spec
