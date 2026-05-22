# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
from __future__ import annotations

import abc
import functools
import warnings
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from functools import wraps

import torch
from tensordict import is_tensor_collection, TensorDictBase
from tensordict.nn import (
    composite_lp_aggregate,
    dispatch,
    ProbabilisticTensorDictModule,
    set_composite_lp_aggregate,
    set_skip_existing,
    TensorDictModule,
    TensorDictModuleBase,
)
from tensordict.nn.probabilistic import interaction_type
from tensordict.utils import NestedKey, unravel_key
from torch import Tensor

from torchrl._utils import logger, rl_warnings
from torchrl.envs.utils import step_mdp
from torchrl.objectives.utils import (
    _maybe_get_or_select,
    _pseudo_vmap,
    _vmap_func,
    hold_out_net,
)
from torchrl.objectives.value.functional import (
    generalized_advantage_estimate,
    td0_return_estimate,
    td_lambda_return_estimate,
    vec_generalized_advantage_estimate,
    vec_td1_return_estimate,
    vec_td_lambda_return_estimate,
    vtrace_advantage_estimate,
)

try:
    from torch.compiler import is_dynamo_compiling
except ImportError:
    from torch._dynamo import is_compiling as is_dynamo_compiling

try:
    from torch import vmap
except ImportError as err:
    try:
        from functorch import vmap
    except ImportError:
        raise ImportError(
            "vmap couldn't be found. Make sure you have torch>2.0 installed."
        ) from err


def _self_set_grad_enabled(fun):
    @wraps(fun)
    def new_fun(self, *args, **kwargs):
        with torch.set_grad_enabled(self.differentiable):
            return fun(self, *args, **kwargs)

    return new_fun


def _self_set_skip_existing(fun):
    @functools.wraps(fun)
    def new_func(self, *args, **kwargs):
        if self.skip_existing is not None:
            with set_skip_existing(self.skip_existing):
                return fun(self, *args, **kwargs)
        return fun(self, *args, **kwargs)

    return new_func


def _call_actor_net(
    actor_net: ProbabilisticTensorDictModule,
    data: TensorDictBase,
    params: TensorDictBase,
    log_prob_key: NestedKey,
):
    dist = actor_net.get_dist(data.select(*actor_net.in_keys, strict=False))
    s = actor_net._dist_sample(dist, interaction_type=interaction_type())
    with set_composite_lp_aggregate(True):
        return dist.log_prob(s)


class ValueEstimatorBase(TensorDictModuleBase):
    """An abstract parent class for value function modules.

    Its :meth:`ValueFunctionBase.forward` method will compute the value (given
    by the value network) and the value estimate (given by the value estimator)
    as well as the advantage and write these values in the output tensordict.

    If only the value estimate is needed, the :meth:`ValueFunctionBase.value_estimate`
    should be used instead.

    """

    @dataclass
    class _AcceptedKeys:
        """Maintains default values for all configurable tensordict keys.

        This class defines which tensordict keys can be set using '.set_keys(key_name=key_value)' and their
        default values.

        Attributes:
            advantage (NestedKey): The input tensordict key where the advantage is written to.
                Will be used for the underlying value estimator. Defaults to ``"advantage"``.
            value_target (NestedKey): The input tensordict key where the target state value is written to.
                Will be used for the underlying value estimator Defaults to ``"value_target"``.
            value (NestedKey): The input tensordict key where the state value is expected.
                Will be used for the underlying value estimator. Defaults to ``"state_value"``.
            reward (NestedKey): The input tensordict key where the reward is written to.
                Defaults to ``"reward"``.
            done (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is done.  Defaults to ``"done"``.
            terminated (NestedKey): The key in the input TensorDict that indicates
                whether a trajectory is terminated.  Defaults to ``"terminated"``.
            steps_to_next_obs (NestedKey): The key in the input tensordict
                that indicates the number of steps to the next observation.
                Defaults to ``"steps_to_next_obs"``.
            sample_log_prob (NestedKey): The key in the input tensordict that
                indicates the log probability of the sampled action.
                Defaults to ``"sample_log_prob"`` when :func:`~tensordict.nn.composite_lp_aggregate` returns `True`,
                `"action_log_prob"`  otherwise.
        """

        advantage: NestedKey = "advantage"
        value_target: NestedKey = "value_target"
        value: NestedKey = "state_value"
        reward: NestedKey = "reward"
        done: NestedKey = "done"
        terminated: NestedKey = "terminated"
        steps_to_next_obs: NestedKey = "steps_to_next_obs"
        sample_log_prob: NestedKey | None = None

        def __post_init__(self):
            if self.sample_log_prob is None:
                if composite_lp_aggregate(nowarn=True):
                    self.sample_log_prob = "sample_log_prob"
                else:
                    self.sample_log_prob = "action_log_prob"

    default_keys = _AcceptedKeys
    tensor_keys: _AcceptedKeys
    value_network: TensorDictModule | Callable
    _vmap_randomness = None
    deactivate_vmap: bool = False

    @property
    def advantage_key(self):
        return self.tensor_keys.advantage

    @property
    def value_key(self):
        return self.tensor_keys.value

    @property
    def value_target_key(self):
        return self.tensor_keys.value_target

    @property
    def reward_key(self):
        return self.tensor_keys.reward

    @property
    def done_key(self):
        return self.tensor_keys.done

    @property
    def terminated_key(self):
        return self.tensor_keys.terminated

    @property
    def steps_to_next_obs_key(self):
        return self.tensor_keys.steps_to_next_obs

    @property
    def sample_log_prob_key(self):
        return self.tensor_keys.sample_log_prob

    @abc.abstractmethod
    def forward(
        self,
        tensordict: TensorDictBase,
        *,
        params: TensorDictBase | None = None,
        target_params: TensorDictBase | None = None,
    ) -> TensorDictBase:
        """Computes the advantage estimate given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, ``"action"``, ``("next", "reward")``,
                ``("next", "done")``, ``("next", "terminated")``,
                and ``"next"`` tensordict state as returned by the environment)
                necessary to compute the value estimates and the TDEstimate.
                The data passed to this module should be structured as
                :obj:`[*B, T, *F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the
                feature dimension(s). The tensordict must have shape ``[*B, T]``.

        Keyword Args:
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.
            device (torch.device, optional): the device where the buffers will be instantiated.
                Defaults to ``torch.get_default_device()``.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.
        """
        ...

    def __init__(
        self,
        *,
        value_network: TensorDictModule,
        shifted: bool = False,
        differentiable: bool = False,
        skip_existing: bool | None = None,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
        device: torch.device | None = None,
        deactivate_vmap: bool = False,
        value_chunk_size: int | None = None,
        shifted_budget: int = 1,
    ):
        super().__init__()
        if device is None:
            device = getattr(torch, "get_default_device", lambda: torch.device("cpu"))()
        # this is saved for tracking only and should not be used to cast anything else than buffers during
        # init.
        self._device = device
        self._tensor_keys = None
        self.differentiable = differentiable
        self.deactivate_vmap = deactivate_vmap
        self.value_chunk_size = value_chunk_size
        if shifted_budget < 1:
            raise ValueError(f"shifted_budget must be >= 1, got {shifted_budget}.")
        self.shifted_budget = shifted_budget
        self.skip_existing = skip_existing
        self.__dict__["value_network"] = value_network
        self.dep_keys = {}
        self.shifted = self._normalize_shifted(shifted)

        if advantage_key is not None:
            raise RuntimeError(
                "Setting 'advantage_key' via constructor is deprecated, use .set_keys(advantage_key='some_key') instead.",
            )
        if value_target_key is not None:
            raise RuntimeError(
                "Setting 'value_target_key' via constructor is deprecated, use .set_keys(value_target_key='some_key') instead.",
            )
        if value_key is not None:
            raise RuntimeError(
                "Setting 'value_key' via constructor is deprecated, use .set_keys(value_key='some_key') instead.",
            )

    @property
    def tensor_keys(self) -> _AcceptedKeys:
        if self._tensor_keys is None:
            self.set_keys()
        return self._tensor_keys

    @tensor_keys.setter
    def tensor_keys(self, value):
        if not isinstance(value, type(self._AcceptedKeys)):
            raise ValueError("value must be an instance of _AcceptedKeys")
        self._keys = value

    @property
    def in_keys(self):
        try:
            in_keys = (
                self.value_network.in_keys
                + [
                    ("next", self.tensor_keys.reward),
                    ("next", self.tensor_keys.done),
                    ("next", self.tensor_keys.terminated),
                ]
                + [("next", in_key) for in_key in self.value_network.in_keys]
            )
        except AttributeError:
            # value network does not have an `in_keys` attribute
            in_keys = []
        return in_keys

    @property
    def out_keys(self):
        return [
            self.tensor_keys.advantage,
            self.tensor_keys.value_target,
        ]

    def set_keys(self, **kwargs) -> None:
        """Set tensordict key names."""
        for key, value in list(kwargs.items()):
            if isinstance(value, list):
                value = [unravel_key(k) for k in value]
            elif not isinstance(value, (str, tuple)):
                if value is None:
                    raise ValueError("tensordict keys cannot be None")
                raise ValueError(
                    f"key name must be of type NestedKey (Union[str, Tuple[str]]) but got {type(value)}"
                )
            else:
                value = unravel_key(value)

            if key not in self._AcceptedKeys.__dict__:
                raise KeyError(
                    f"{key} is not an accepted tensordict key for advantages"
                )
            if (
                key == "value"
                and hasattr(self.value_network, "out_keys")
                and (value not in self.value_network.out_keys)
            ):
                raise KeyError(
                    f"value key '{value}' not found in value network out_keys {self.value_network.out_keys}"
                )
            kwargs[key] = value
        if self._tensor_keys is None:
            conf = asdict(self.default_keys())
            conf.update(self.dep_keys)
        else:
            conf = asdict(self._tensor_keys)
        conf.update(kwargs)
        self._tensor_keys = self._AcceptedKeys(**conf)

    def value_estimate(
        self,
        tensordict,
        target_params: TensorDictBase | None = None,
        next_value: torch.Tensor | None = None,
        **kwargs,
    ):
        """Gets a value estimate, usually used as a target value for the value network.

        If the state value key is present under ``tensordict.get(("next", self.tensor_keys.value))``
        then this value will be used without recurring to the value network.

        Args:
            tensordict (TensorDictBase): the tensordict containing the data to
                read.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.
            next_value (torch.Tensor, optional): the value of the next state
                or state-action pair. Exclusive with ``target_params``.
            **kwargs: the keyword arguments to be passed to the value network.

        Returns: a tensor corresponding to the state value.

        """
        raise NotImplementedError

    @property
    def is_functional(self):
        return False

    @property
    def is_stateless(self):
        return False

    def _next_value(self, tensordict, target_params, kwargs):
        step_td = step_mdp(tensordict, keep_other=False)
        if self.value_network is not None:
            with hold_out_net(
                self.value_network
            ) if target_params is None else target_params.to_module(self.value_network):
                self.value_network(step_td)
        next_value = step_td.get(self.tensor_keys.value)
        return next_value

    @property
    def vmap_randomness(self):
        if self._vmap_randomness is None:
            if is_dynamo_compiling():
                self._vmap_randomness = "different"
                return "different"
            do_break = False
            for val in self.__dict__.values():
                if isinstance(val, torch.nn.Module):
                    import torchrl.objectives.utils

                    for module in val.modules():
                        if isinstance(
                            module, torchrl.objectives.utils.RANDOM_MODULE_LIST
                        ):
                            self._vmap_randomness = "different"
                            do_break = True
                            break
                if do_break:
                    # double break
                    break
            else:
                self._vmap_randomness = "error"

        return self._vmap_randomness

    def set_vmap_randomness(self, value):
        self._vmap_randomness = value

    def _get_time_dim(self, time_dim: int | None, data: TensorDictBase):
        if time_dim is not None:
            if time_dim < 0:
                time_dim = data.ndim + time_dim
            return time_dim
        time_dim_attr = getattr(self, "time_dim", None)
        if time_dim_attr is not None:
            if time_dim_attr < 0:
                time_dim_attr = data.ndim + time_dim_attr
            return time_dim_attr
        if data._has_names():
            for i, name in enumerate(data.names):
                if name == "time":
                    return i
        return data.ndim - 1

    @staticmethod
    def _sanitize_next_obs_nan(
        data: TensorDictBase,
        in_keys: list[NestedKey],
    ) -> TensorDictBase:
        """Replace ``NaN`` entries in ``("next", k)`` with the corresponding root ``k``.

        Acts as a finite placeholder for "next observations" that are absent —
        as produced by
        :class:`~torchrl.envs.transforms.NextStateReconstructor` at trajectory
        ends in conjunction with
        :class:`~torchrl.collectors.SyncDataCollector` configured with
        ``compact_obs=True``. Without this step, ``V(NaN) = NaN`` propagates
        through the TD / GAE kernels (the multiplication by ``(1 - done)``
        does not zero NaN out because ``0 * NaN = NaN`` in IEEE 754).

        Semantics of the substitution:

        - At **terminated** steps the value at the next observation is masked
          out by ``(1 - done)`` downstream, so the substitute is discarded.
        - At **truncated-only** steps the substitute acts as an approximate
          bootstrap ``V(obs[t]) ≈ V(real_next_obs)`` — strictly an
          approximation, but finite and well-defined.

        Operates on a shallow copy so the caller's ``tensordict`` is not
        mutated.
        """
        copied = False
        for k in in_keys:
            root = data.get(k, default=None)
            if root is None or not root.is_floating_point():
                continue
            next_k = ("next", *k) if isinstance(k, tuple) else ("next", k)
            nxt = data.get(next_k, default=None)
            if nxt is None or nxt.shape != root.shape:
                continue
            nan_mask = torch.isnan(nxt)
            if not nan_mask.any():
                continue
            if not copied:
                data = data.copy()
                copied = True
            data.set(next_k, torch.where(nan_mask, root, nxt))
        return data

    @staticmethod
    def _fill_missing_next_inputs(
        next_data: TensorDictBase, root_data: TensorDictBase, in_keys: list[NestedKey]
    ) -> TensorDictBase:
        copied = False
        for key in in_keys:
            if next_data.get(key, default=None) is not None:
                continue
            value = root_data.get(key, default=None)
            if value is None:
                continue
            if not copied:
                next_data = next_data.copy()
                copied = True
            next_data.set(key, value)
        return next_data

    @staticmethod
    def _normalize_shifted(
        shifted: bool,
    ) -> bool:
        """Normalize the ``shifted`` argument.

        ``shifted=True`` uses the budgeted shifted backend.
        """
        if shifted is False:
            return False
        if shifted is True:
            return True
        raise ValueError(f"shifted must be a boolean, got {shifted!r}.")

    def _call_value_net_shifted(
        self,
        data: TensorDictBase,
        params: TensorDictBase | None,
        next_params: TensorDictBase | None,
        value_key: NestedKey,
        ndim: int,
        value_net: TensorDictModuleBase,
        _call_value_net,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compact single-call path that inserts reset next-observations.

        This backend keeps the value-network input length fixed to ``T + 1``.
        It inserts true ``("next", ...)`` entries after internal reset steps,
        shifts subsequent root entries to the right, and marks the displaced
        suffix as invalid. Retained samples have exact root and next values.
        """
        if next_params is not None and next_params is not params:
            raise ValueError(
                "the value at t and t+1 cannot be retrieved in a single call when both params and next params are passed."
            )
        in_keys = value_net.in_keys
        time_idx = ndim - 1
        T = data.shape[time_idx]
        root_part = data.select(*in_keys, value_key, strict=False)
        next_part = data.get("next").select(*in_keys, value_key, strict=False)
        next_part = self._fill_missing_next_inputs(next_part, root_part, in_keys)
        done = data.get(("next", self.tensor_keys.done))
        terminated = data.get(("next", self.tensor_keys.terminated), default=done)
        if done.shape[-1] == 1:
            reset = done.squeeze(-1) & ~terminated.squeeze(-1)
        else:
            reset = done.any(-1) & ~terminated.any(-1)
        reset = reset.clone()
        reset[(slice(None),) * time_idx + (-1,)] = False
        reset_long = reset.to(torch.long)
        reset_cs = reset_long.cumsum(time_idx)
        zero_shape = list(reset.shape)
        zero_shape[time_idx] = 1
        reset_before = torch.cat(
            [reset_cs.new_zeros(zero_shape), reset_cs.narrow(time_idx, 0, T - 1)],
            dim=time_idx,
        )
        arange_shape = [1] * reset.ndim
        arange_shape[time_idx] = T
        arange = torch.arange(T, device=done.device).view(arange_shape)
        root_slot = arange + reset_before
        next_slot = root_slot + 1
        L = T + self.shifted_budget
        valid = next_slot < L
        root_valid = root_slot < L
        reset_valid = reset & valid
        boundary_slot = T + reset_long.sum(time_idx, keepdim=True)
        boundary_valid = boundary_slot < L
        data_in_batch_size = list(root_part.batch_size)
        data_in_batch_size[time_idx] = L
        data_in = root_part.new_zeros(data_in_batch_size)

        def _expand_index(index: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
            while index.ndim < source.ndim:
                index = index.unsqueeze(-1)
            return index.expand_as(source)

        def _expand_mask(mask: torch.Tensor, source: torch.Tensor) -> torch.Tensor:
            while mask.ndim < source.ndim:
                mask = mask.unsqueeze(-1)
            return mask.expand_as(source)

        def _scatter_time(
            destination: torch.Tensor,
            index: torch.Tensor,
            source: torch.Tensor,
            mask: torch.Tensor,
        ) -> torch.Tensor:
            index = index.clamp_max(L - 1)
            index_expand = _expand_index(index, source)
            current = destination.gather(time_idx, index_expand)
            source = torch.where(_expand_mask(mask, source), source, current)
            return destination.scatter(time_idx, index_expand, source)

        boundary_index = (slice(None),) * time_idx + (slice(T - 1, T),)
        for key in in_keys:
            root_value = root_part.get(key, default=None)
            if root_value is None:
                continue
            data_value = data_in.get(key)
            data_value = _scatter_time(data_value, root_slot, root_value, root_valid)
            next_value = next_part.get(key, default=None)
            if next_value is not None:
                data_value = _scatter_time(
                    data_value, next_slot, next_value, reset_valid
                )
                data_value = _scatter_time(
                    data_value,
                    boundary_slot,
                    next_value[boundary_index],
                    boundary_valid,
                )
            data_in.set(key, data_value)
        if params is not None:
            with params.to_module(value_net):
                values_full = _call_value_net(data_in)
        else:
            values_full = _call_value_net(data_in)

        def _gather_time(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
            index_expand = index.clamp_max(L - 1)
            while index_expand.ndim < source.ndim:
                index_expand = index_expand.unsqueeze(-1)
            target_shape = list(source.shape)
            target_shape[time_idx] = index.shape[time_idx]
            index_expand = index_expand.expand(target_shape)
            return source.gather(time_idx, index_expand)

        value = _gather_time(values_full, root_slot)
        value_ = _gather_time(values_full, next_slot)
        try:
            value = value.view_as(done)
            value_ = value_.view_as(done)
        except RuntimeError:
            pass
        return value, value_, valid

    def _call_value_nets(
        self,
        data: TensorDictBase,
        params: TensorDictBase,
        next_params: TensorDictBase,
        single_call: bool,
        value_key: NestedKey,
        detach_next: bool,
        vmap_randomness: str = "error",
        *,
        value_net: TensorDictModuleBase | None = None,
    ):
        # ``single_call`` is either ``False`` or requests the budgeted shifted path.
        if value_net is None:
            value_net = self.value_network
        in_keys = value_net.in_keys
        if single_call:
            try:
                ndim = list(data.names).index("time") + 1
            except ValueError:
                if rl_warnings():
                    logger.warning(
                        "Got a tensordict without a time-marked dimension, assuming time is along the last dimension. "
                        "This warning can be turned off by setting the environment variable RL_WARNINGS to False."
                    )
                ndim = data.ndim
        else:
            ndim = None
        data = self._sanitize_next_obs_nan(data, in_keys)

        def _call_value_net(data_in: TensorDictBase) -> torch.Tensor:
            chunk_size = self.value_chunk_size
            if chunk_size is None or data_in.numel() <= chunk_size:
                return value_net(data_in).get(value_key)
            values = []
            for chunk in data_in.split(chunk_size, dim=0):
                values.append(value_net(chunk).get(value_key))
            return torch.cat(values, dim=0)

        valid = None
        if single_call:
            value, value_, valid = self._call_value_net_shifted(
                data=data,
                params=params,
                next_params=next_params,
                value_key=value_key,
                ndim=ndim,
                value_net=value_net,
                _call_value_net=_call_value_net,
            )
        else:
            data_root = data.select(*in_keys, value_key, strict=False)
            data_next = data.get("next").select(*in_keys, value_key, strict=False)
            if "is_init" in data_root.keys():
                # We need to mark the first element of the "next" td as being an init step for RNNs
                #  otherwise, consecutive elements in the sequence will be considered as part of the same
                #  trajectory, even if they're not.
                data_next["is_init"] = data_next["is_init"] | data_root["is_init"]
            data_in = torch.stack(
                [data_root, data_next],
                0,
            )
            if (params is not None) ^ (next_params is not None):
                raise ValueError(
                    "params and next_params must be either both provided or not."
                )
            elif (
                params is None
                and next_params is None
                and self.value_chunk_size is not None
            ):
                value = _call_value_net(data_root)
                value_ = _call_value_net(data_next)
            elif params is not None:
                params_stack = torch.stack([params, next_params], 0).contiguous()
                data_out = _vmap_func(
                    value_net,
                    (0, 0),
                    randomness=vmap_randomness,
                    pseudo_vmap=self.deactivate_vmap,
                )(data_in, params_stack)
            elif not self.deactivate_vmap:
                data_out = vmap(value_net, (0,), randomness=vmap_randomness)(data_in)
            else:
                data_out = _pseudo_vmap(value_net, (0,), randomness=vmap_randomness)(
                    data_in
                )
            if self.value_chunk_size is None or params is not None:
                value_est = data_out.get(value_key)
                value, value_ = value_est[0], value_est[1]
        data.set(value_key, value)
        data.set(("next", value_key), value_)
        if detach_next:
            value_ = value_.detach()
        return value, value_, valid

    def _shifted_last_valid(self, valid: torch.Tensor, time_dim: int):
        next_valid = torch.cat(
            [
                valid.narrow(time_dim, 1, valid.shape[time_dim] - 1),
                valid.new_ones(
                    [
                        *valid.shape[:time_dim],
                        1,
                        *valid.shape[time_dim + 1 :],
                    ]
                ),
            ],
            dim=time_dim,
        )
        return valid & ~next_valid

    @staticmethod
    def _expand_to_match(source: torch.Tensor, target: torch.Tensor):
        while source.ndim < target.ndim:
            source = source.unsqueeze(-1)
        return source.expand_as(target)

    def _prepare_shifted_tensordict(
        self, tensordict: TensorDictBase, valid: torch.Tensor | None, time_dim: int
    ):
        if valid is None:
            return tensordict
        last_valid = self._shifted_last_valid(valid, time_dim)
        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated), default=done)
        done_mask = self._expand_to_match(last_valid, done)
        tensordict = tensordict.copy()
        tensordict.set(
            ("next", self.tensor_keys.done),
            done.masked_fill(done_mask, True),
        )
        tensordict.set(
            ("next", self.tensor_keys.terminated),
            terminated.masked_fill(done_mask, False),
        )
        return tensordict

    def _mask_shifted_output(
        self, tensordict: TensorDictBase, valid: torch.Tensor | None
    ):
        if valid is None:
            return
        tensordict.set("shifted_valid", valid)
        for key in (self.tensor_keys.advantage, self.tensor_keys.value_target):
            value = tensordict.get(key)
            mask = self._expand_to_match(valid, value)
            tensordict.set(key, value.masked_fill(~mask, 0))


class TD0Estimator(ValueEstimatorBase):
    """Temporal Difference (TD(0)) estimate of advantage function.

    AKA bootstrapped temporal difference or 1-step return.

    Keyword Args:
        gamma (scalar): exponential mean discount.
        value_network (TensorDictModule): value operator used to retrieve
            the value estimates.
        shifted (bool or str, optional): controls how value and next-value
            are obtained from the value network. ``False`` (default) calls
            the value network twice (once on the root tensordict, once on
            ``"next"``), which is correct whenever ``"next"`` may differ
            non-trivially from ``obs[t+1]``. Truthy values request a single
            call:

            - ``True``: fixed-budget single-call path. Inserts true
              ``next_obs`` after internal resets and masks the displaced
              suffix samples via ``"shifted_valid"``. Retained samples use
              exact next observations while keeping the static compute budget
              configured by ``shifted_budget``.
            - ``True``: fixed-budget single-call path. Inserts true
              ``next_obs`` after internal resets and masks the displaced
              suffix samples via ``"shifted_valid"``. Retained samples
              use exact next observations while keeping the static compute
              budget configured by ``shifted_budget``.

            All single-call paths require that the parameters at time
            ``t`` and ``t+1`` are identical (i.e. ``target_params`` is not
            used) and that the ``"next"`` value is shifted by exactly one
            time step (no multi-step returns). Defaults to ``False``.
        average_rewards (bool, optional): if ``True``, rewards will be standardized
            before the TD is computed.
        differentiable (bool, optional): if ``True``, gradients are propagated through
            the computation of the value function. Default is ``False``.

            .. note::
              The proper way to make the function call non-differentiable is to
              decorate it in a `torch.no_grad()` context manager/decorator or
              pass detached parameters for functional modules.

        skip_existing (bool, optional): if ``True``, the value network will skip
            modules which outputs are already present in the tensordict.
            Defaults to ``None``, i.e., the value of :func:`tensordict.nn.skip_existing()`
            is not affected.
        advantage_key (str or tuple of str, optional): [Deprecated] the key of
            the advantage entry.  Defaults to ``"advantage"``.
        value_target_key (str or tuple of str, optional): [Deprecated] the key
            of the advantage entry.  Defaults to ``"value_target"``.
        value_key (str or tuple of str, optional): [Deprecated] the value key to
            read from the input tensordict.  Defaults to ``"state_value"``.
        device (torch.device, optional): the device where the buffers will be instantiated.
            Defaults to ``torch.get_default_device()``.
        deactivate_vmap (bool, optional): whether to deactivate vmap calls and replace them with a plain for loop.
            Defaults to ``False``.
        value_chunk_size (int, optional): if set, splits value-network calls
            into chunks of this many elements along the leading dimension.
            Defaults to ``None``.
        shifted_budget (int, optional): number of extra value-network time slots
            used when ``shifted=True``. ``1`` uses a ``T+1``
            budget, ``2`` can represent one internal reset plus the rollout
            boundary without dropping samples, and so on. Defaults to ``1``.

    """

    def __init__(
        self,
        *,
        gamma: float | torch.Tensor,
        value_network: TensorDictModule,
        shifted: bool = False,
        average_rewards: bool = False,
        differentiable: bool = False,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
        skip_existing: bool | None = None,
        device: torch.device | None = None,
        deactivate_vmap: bool = False,
        value_chunk_size: int | None = None,
        shifted_budget: int = 1,
    ):
        super().__init__(
            value_network=value_network,
            differentiable=differentiable,
            shifted=shifted,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            skip_existing=skip_existing,
            device=device,
            deactivate_vmap=deactivate_vmap,
            value_chunk_size=value_chunk_size,
            shifted_budget=shifted_budget,
        )
        self.register_buffer("gamma", torch.tensor(gamma, device=self._device))
        self.average_rewards = average_rewards

    @_self_set_skip_existing
    @_self_set_grad_enabled
    @dispatch
    def forward(
        self,
        tensordict: TensorDictBase,
        *,
        params: TensorDictBase | None = None,
        target_params: TensorDictBase | None = None,
    ) -> TensorDictBase:
        """Computes the TD(0) advantage given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, ``"action"``, ``("next", "reward")``,
                ``("next", "done")``, ``("next", "terminated")``, and ``"next"``
                tensordict state as returned by the environment) necessary to
                compute the value estimates and the TDEstimate.
                The data passed to this module should be structured as
                :obj:`[*B, T, *F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the
                feature dimension(s). The tensordict must have shape ``[*B, T]``.

        Keyword Args:
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> from tensordict import TensorDict
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDEstimate(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs, "done": done, "terminated": terminated, "reward": reward}}, [1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDEstimate(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> advantage, value_target = module(obs=obs, next_reward=reward, next_done=done, next_obs=next_obs, next_terminated=terminated)

        """
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got"
                f"tensordict.batch_size = {tensordict.batch_size}"
            )

        if self.is_stateless and params is None:
            raise RuntimeError(
                "Expected params to be passed to advantage module but got none."
            )
        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network) if (
                params is None and target_params is None
            ) else nullcontext():
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value, valid = self._call_value_nets(
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                    vmap_randomness=self.vmap_randomness,
                )
                if valid is not None:
                    tensordict.set("shifted_valid", valid)
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))

        valid = tensordict.get("shifted_valid", default=None)
        data_for_value = self._prepare_shifted_tensordict(
            tensordict, valid, self._get_time_dim(None, tensordict)
        )
        value_target = self.value_estimate(data_for_value, next_value=next_value)
        tensordict.set(self.tensor_keys.advantage, value_target - value)
        tensordict.set(self.tensor_keys.value_target, value_target)
        self._mask_shifted_output(tensordict, valid)
        return tensordict

    def value_estimate(
        self,
        tensordict,
        target_params: TensorDictBase | None = None,
        next_value: torch.Tensor | None = None,
        **kwargs,
    ):
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device

        if self.gamma.device != device:
            self.gamma = self.gamma.to(device)
        gamma = self.gamma
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        if self.average_rewards:
            reward = reward - reward.mean()
            reward = reward / reward.std().clamp_min(1e-5)
            tensordict.set(
                ("next", self.tensor_keys.reward), reward
            )  # we must update the rewards if they are used later in the code
        if next_value is None:
            next_value = self._next_value(tensordict, target_params, kwargs=kwargs)

        done = tensordict.get(("next", self.tensor_keys.done))
        terminated = tensordict.get(("next", self.tensor_keys.terminated), default=done)
        value_target = td0_return_estimate(
            gamma=gamma,
            next_state_value=next_value,
            reward=reward,
            done=done,
            terminated=terminated,
        )
        return value_target


class TD1Estimator(ValueEstimatorBase):
    r""":math:`\infty`-Temporal Difference (TD(1)) estimate of advantage function.

    Keyword Args:
        gamma (scalar): exponential mean discount.
        value_network (TensorDictModule): value operator used to retrieve the value estimates.
        average_rewards (bool, optional): if ``True``, rewards will be standardized
            before the TD is computed.
        differentiable (bool, optional): if ``True``, gradients are propagated through
            the computation of the value function. Default is ``False``.

            .. note::
              The proper way to make the function call non-differentiable is to
              decorate it in a `torch.no_grad()` context manager/decorator or
              pass detached parameters for functional modules.

        skip_existing (bool, optional): if ``True``, the value network will skip
            modules which outputs are already present in the tensordict.
            Defaults to ``None``, i.e., the value of :func:`tensordict.nn.skip_existing()`
            is not affected.
        advantage_key (str or tuple of str, optional): [Deprecated] the key of
            the advantage entry.  Defaults to ``"advantage"``.
        value_target_key (str or tuple of str, optional): [Deprecated] the key
            of the advantage entry.  Defaults to ``"value_target"``.
        value_key (str or tuple of str, optional): [Deprecated] the value key to
            read from the input tensordict.  Defaults to ``"state_value"``.
        shifted (bool or str, optional): controls how value and next-value
            are obtained from the value network. ``False`` (default) calls
            the value network twice (once on the root tensordict, once on
            ``"next"``), which is correct whenever ``"next"`` may differ
            non-trivially from ``obs[t+1]``. Truthy values request a single
            call:

            - ``True``: fixed-budget single-call path. Inserts true
              ``next_obs`` after internal resets and masks the displaced
              suffix samples via ``"shifted_valid"``. Retained samples use
              exact next observations while keeping the static compute budget
              configured by ``shifted_budget``.
            - ``True``: fixed-budget single-call path. Inserts true
              ``next_obs`` after internal resets and masks the displaced
              suffix samples via ``"shifted_valid"``. Retained samples
              use exact next observations while keeping the static compute
              budget configured by ``shifted_budget``.

            All single-call paths require that the parameters at time
            ``t`` and ``t+1`` are identical (i.e. ``target_params`` is not
            used) and that the ``"next"`` value is shifted by exactly one
            time step (no multi-step returns). Defaults to ``False``.
        device (torch.device, optional): the device where the buffers will be instantiated.
            Defaults to ``torch.get_default_device()``.
        time_dim (int, optional): the dimension corresponding to the time
            in the input tensordict. If not provided, defaults to the dimension
            marked with the ``"time"`` name if any, and to the last dimension
            otherwise. Can be overridden during a call to
            :meth:`~.value_estimate`.
            Negative dimensions are considered with respect to the input
            tensordict.
        deactivate_vmap (bool, optional): whether to deactivate vmap calls and replace them with a plain for loop.
            Defaults to ``False``.
        value_chunk_size (int, optional): if set, splits value-network calls
            into chunks of this many elements along the leading dimension.
            Defaults to ``None``.
        shifted_budget (int, optional): number of extra value-network time slots
            used when ``shifted=True``. ``1`` uses a ``T+1``
            budget, ``2`` can represent one internal reset plus the rollout
            boundary without dropping samples, and so on. Defaults to ``1``.

    """

    def __init__(
        self,
        *,
        gamma: float | torch.Tensor,
        value_network: TensorDictModule,
        average_rewards: bool = False,
        differentiable: bool = False,
        skip_existing: bool | None = None,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
        shifted: bool = False,
        device: torch.device | None = None,
        time_dim: int | None = None,
        deactivate_vmap: bool = False,
        value_chunk_size: int | None = None,
        shifted_budget: int = 1,
    ):
        super().__init__(
            value_network=value_network,
            differentiable=differentiable,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            shifted=shifted,
            skip_existing=skip_existing,
            device=device,
            deactivate_vmap=deactivate_vmap,
            value_chunk_size=value_chunk_size,
            shifted_budget=shifted_budget,
        )
        self.register_buffer("gamma", torch.tensor(gamma, device=self._device))
        self.average_rewards = average_rewards
        self.time_dim = time_dim

    @_self_set_skip_existing
    @_self_set_grad_enabled
    @dispatch
    def forward(
        self,
        tensordict: TensorDictBase,
        *,
        params: TensorDictBase | None = None,
        target_params: TensorDictBase | None = None,
    ) -> TensorDictBase:
        """Computes the TD(1) advantage given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, ``"action"``, ``("next", "reward")``,
                ``("next", "done")``, ``("next", "terminated")``,
                and ``"next"`` tensordict state as returned by the environment)
                necessary to compute the value estimates and the TDEstimate.
                The data passed to this module should be structured as :obj:`[*B, T, *F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the feature dimension(s).
                The tensordict must have shape ``[*B, T]``.

        Keyword Args:
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> from tensordict import TensorDict
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDEstimate(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs, "done": done, "reward": reward, "terminated": terminated}}, [1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDEstimate(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> advantage, value_target = module(obs=obs, next_reward=reward, next_done=done, next_obs=next_obs, next_terminated=terminated)

        """
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got"
                f"tensordict.batch_size = {tensordict.batch_size}"
            )

        if self.is_stateless and params is None:
            raise RuntimeError(
                "Expected params to be passed to advantage module but got none."
            )
        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network) if (
                params is None and target_params is None
            ) else nullcontext():
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value, valid = self._call_value_nets(
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                    vmap_randomness=self.vmap_randomness,
                )
                if valid is not None:
                    tensordict.set("shifted_valid", valid)
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))

        valid = tensordict.get("shifted_valid", default=None)
        data_for_value = self._prepare_shifted_tensordict(
            tensordict, valid, self._get_time_dim(None, tensordict)
        )
        value_target = self.value_estimate(data_for_value, next_value=next_value)

        tensordict.set(self.tensor_keys.advantage, value_target - value)
        tensordict.set(self.tensor_keys.value_target, value_target)
        self._mask_shifted_output(tensordict, valid)
        return tensordict

    def value_estimate(
        self,
        tensordict,
        target_params: TensorDictBase | None = None,
        next_value: torch.Tensor | None = None,
        time_dim: int | None = None,
        **kwargs,
    ):
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device
        if self.gamma.device != device:
            self.gamma = self.gamma.to(device)
        gamma = self.gamma
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        if self.average_rewards:
            reward = reward - reward.mean()
            reward = reward / reward.std().clamp_min(1e-5)
            tensordict.set(
                ("next", self.tensor_keys.reward), reward
            )  # we must update the rewards if they are used later in the code
        if next_value is None:
            next_value = self._next_value(tensordict, target_params, kwargs=kwargs)

        time_dim = self._get_time_dim(time_dim, tensordict)
        valid = tensordict.get("shifted_valid", default=None)
        data_for_value = self._prepare_shifted_tensordict(tensordict, valid, time_dim)
        reward = data_for_value.get(("next", self.tensor_keys.reward))
        done = data_for_value.get(("next", self.tensor_keys.done))
        terminated = data_for_value.get(
            ("next", self.tensor_keys.terminated), default=done
        )
        value_target = vec_td1_return_estimate(
            gamma,
            next_value,
            reward,
            done=done,
            terminated=terminated,
            time_dim=time_dim,
        )
        return value_target


class TDLambdaEstimator(ValueEstimatorBase):
    r"""TD(:math:`\lambda`) estimate of advantage function.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        value_network (TensorDictModule): value operator used to retrieve the value estimates.
        average_rewards (bool, optional): if ``True``, rewards will be standardized
            before the TD is computed.
        differentiable (bool, optional): if ``True``, gradients are propagated through
            the computation of the value function. Default is ``False``.

            .. note::
              The proper way to make the function call non-differentiable is to
              decorate it in a `torch.no_grad()` context manager/decorator or
              pass detached parameters for functional modules.

        vectorized (bool, optional): whether to use the vectorized version of the
            lambda return. Default is `True`.
        skip_existing (bool, optional): if ``True``, the value network will skip
            modules which outputs are already present in the tensordict.
            Defaults to ``None``, i.e., the value of :func:`tensordict.nn.skip_existing()`
            is not affected.
        advantage_key (str or tuple of str, optional): [Deprecated] the key of
            the advantage entry.  Defaults to ``"advantage"``.
        value_target_key (str or tuple of str, optional): [Deprecated] the key
            of the advantage entry.  Defaults to ``"value_target"``.
        value_key (str or tuple of str, optional): [Deprecated] the value key to
            read from the input tensordict.  Defaults to ``"state_value"``.
        shifted (bool or str, optional): controls how value and next-value
            are obtained from the value network. ``False`` (default) calls
            the value network twice (once on the root tensordict, once on
            ``"next"``), which is correct whenever ``"next"`` may differ
            non-trivially from ``obs[t+1]``. Truthy values request a single
            call:

            - ``True``: fixed-budget single-call path. Inserts true
              ``next_obs`` after internal resets and masks the displaced
              suffix samples via ``"shifted_valid"``. Retained samples use
              exact next observations while keeping the static compute budget
              configured by ``shifted_budget``.
            - ``True``: fixed-budget single-call path. Inserts true
              ``next_obs`` after internal resets and masks the displaced
              suffix samples via ``"shifted_valid"``. Retained samples
              use exact next observations while keeping the static compute
              budget configured by ``shifted_budget``.

            All single-call paths require that the parameters at time
            ``t`` and ``t+1`` are identical (i.e. ``target_params`` is not
            used) and that the ``"next"`` value is shifted by exactly one
            time step (no multi-step returns). Defaults to ``False``.
        device (torch.device, optional): the device where the buffers will be instantiated.
            Defaults to ``torch.get_default_device()``.
        time_dim (int, optional): the dimension corresponding to the time
            in the input tensordict. If not provided, defaults to the dimension
            marked with the ``"time"`` name if any, and to the last dimension
            otherwise. Can be overridden during a call to
            :meth:`~.value_estimate`.
            Negative dimensions are considered with respect to the input
            tensordict.
        deactivate_vmap (bool, optional): whether to deactivate vmap calls and replace them with a plain for loop.
            Defaults to ``False``.
        value_chunk_size (int, optional): if set, splits value-network calls
            into chunks of this many elements along the leading dimension.
            Defaults to ``None``.
        shifted_budget (int, optional): number of extra value-network time slots
            used when ``shifted=True``. ``1`` uses a ``T+1``
            budget, ``2`` can represent one internal reset plus the rollout
            boundary without dropping samples, and so on. Defaults to ``1``.

    """

    def __init__(
        self,
        *,
        gamma: float | torch.Tensor,
        lmbda: float | torch.Tensor,
        value_network: TensorDictModule,
        average_rewards: bool = False,
        differentiable: bool = False,
        vectorized: bool = True,
        skip_existing: bool | None = None,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
        shifted: bool = False,
        device: torch.device | None = None,
        time_dim: int | None = None,
        deactivate_vmap: bool = False,
        value_chunk_size: int | None = None,
        shifted_budget: int = 1,
    ):
        super().__init__(
            value_network=value_network,
            differentiable=differentiable,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            skip_existing=skip_existing,
            shifted=shifted,
            device=device,
            deactivate_vmap=deactivate_vmap,
            value_chunk_size=value_chunk_size,
            shifted_budget=shifted_budget,
        )
        self.register_buffer("gamma", torch.tensor(gamma, device=self._device))
        self.register_buffer("lmbda", torch.tensor(lmbda, device=self._device))
        self.average_rewards = average_rewards
        self.vectorized = vectorized
        self.time_dim = time_dim

    @property
    def vectorized(self):
        if is_dynamo_compiling():
            return False
        return self._vectorized

    @vectorized.setter
    def vectorized(self, value):
        self._vectorized = value

    @_self_set_skip_existing
    @_self_set_grad_enabled
    @dispatch
    def forward(
        self,
        tensordict: TensorDictBase,
        *,
        params: list[Tensor] | None = None,
        target_params: list[Tensor] | None = None,
    ) -> TensorDictBase:
        r"""Computes the TD(:math:`\lambda`) advantage given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, ``"action"``, ``("next", "reward")``,
                ``("next", "done")``, ``("next", "terminated")``,
                and ``"next"`` tensordict state as returned by the environment)
                necessary to compute the value estimates and the TDLambdaEstimate.
                The data passed to this module should be structured as :obj:`[*B, T, *F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the feature dimension(s).
                The tensordict must have shape ``[*B, T]``.

        Keyword Args:
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> from tensordict import TensorDict
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDLambdaEstimator(
            ...     gamma=0.98,
            ...     lmbda=0.94,
            ...     value_network=value_net,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs, "done": done, "reward": reward, "terminated": terminated}}, [1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = TDLambdaEstimator(
            ...     gamma=0.98,
            ...     lmbda=0.94,
            ...     value_network=value_net,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> advantage, value_target = module(obs=obs, next_reward=reward, next_done=done, next_obs=next_obs, next_terminated=terminated)

        """
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got"
                f"tensordict.batch_size = {tensordict.batch_size}"
            )
        if self.is_stateless and params is None:
            raise RuntimeError(
                "Expected params to be passed to advantage module but got none."
            )
        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network) if (
                params is None and target_params is None
            ) else nullcontext():
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value, valid = self._call_value_nets(
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                    vmap_randomness=self.vmap_randomness,
                )
                if valid is not None:
                    tensordict.set("shifted_valid", valid)
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))
        valid = tensordict.get("shifted_valid", default=None)
        data_for_value = self._prepare_shifted_tensordict(
            tensordict, valid, self._get_time_dim(None, tensordict)
        )
        value_target = self.value_estimate(data_for_value, next_value=next_value)

        tensordict.set(self.tensor_keys.advantage, value_target - value)
        tensordict.set(self.tensor_keys.value_target, value_target)
        self._mask_shifted_output(tensordict, valid)
        return tensordict

    def value_estimate(
        self,
        tensordict,
        target_params: TensorDictBase | None = None,
        next_value: torch.Tensor | None = None,
        time_dim: int | None = None,
        **kwargs,
    ):
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device

        if self.gamma.device != device:
            self.gamma = self.gamma.to(device)
        gamma = self.gamma
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        if self.lmbda.device != device:
            self.lmbda = self.lmbda.to(device)
        lmbda = self.lmbda
        if self.average_rewards:
            reward = reward - reward.mean()
            reward = reward / reward.std().clamp_min(1e-4)
            tensordict.set(
                ("next", self.tensor_keys.steps_to_next_obs), reward
            )  # we must update the rewards if they are used later in the code

        if next_value is None:
            next_value = self._next_value(tensordict, target_params, kwargs=kwargs)

        time_dim = self._get_time_dim(time_dim, tensordict)
        valid = tensordict.get("shifted_valid", default=None)
        data_for_value = self._prepare_shifted_tensordict(tensordict, valid, time_dim)
        reward = data_for_value.get(("next", self.tensor_keys.reward))
        done = data_for_value.get(("next", self.tensor_keys.done))
        terminated = data_for_value.get(
            ("next", self.tensor_keys.terminated), default=done
        )
        if self.vectorized:
            val = vec_td_lambda_return_estimate(
                gamma,
                lmbda,
                next_value,
                reward,
                done=done,
                terminated=terminated,
                time_dim=time_dim,
            )
        else:
            val = td_lambda_return_estimate(
                gamma,
                lmbda,
                next_value,
                reward,
                done=done,
                terminated=terminated,
                time_dim=time_dim,
            )
        return val


class GAE(ValueEstimatorBase):
    """A class wrapper around the generalized advantage estimate functional.

    Refer to "HIGH-DIMENSIONAL CONTINUOUS CONTROL USING GENERALIZED ADVANTAGE ESTIMATION"
    https://arxiv.org/pdf/1506.02438.pdf for more context.

    Args:
        gamma (scalar): exponential mean discount.
        lmbda (scalar): trajectory discount.
        value_network (TensorDictModule, optional): value operator used to retrieve the value estimates.
            If ``None``, this module will expect the ``"state_value"`` keys to be already filled, and
            will not call the value network to produce it.
        average_gae (bool): if ``True``, the resulting GAE values will be standardized.
            Default is ``False``.
        differentiable (bool, optional): if ``True``, gradients are propagated through
            the computation of the value function. Default is ``False``.

            .. note::
              The proper way to make the function call non-differentiable is to
              decorate it in a `torch.no_grad()` context manager/decorator or
              pass detached parameters for functional modules.

        vectorized (bool, optional): whether to use the vectorized version of the
            lambda return. Default is `True` if not compiling.
        skip_existing (bool, optional): if ``True``, the value network will skip
            modules which outputs are already present in the tensordict.
            Defaults to ``None``, i.e., the value of :func:`tensordict.nn.skip_existing()`
            is not affected.
            Defaults to "state_value".
        advantage_key (str or tuple of str, optional): [Deprecated] the key of
            the advantage entry.  Defaults to ``"advantage"``.
        value_target_key (str or tuple of str, optional): [Deprecated] the key
            of the advantage entry.  Defaults to ``"value_target"``.
        value_key (str or tuple of str, optional): [Deprecated] the value key to
            read from the input tensordict.  Defaults to ``"state_value"``.
        shifted (bool or str, optional): controls how value and next-value
            are obtained from the value network. ``False`` (default) calls
            the value network twice (once on the root tensordict, once on
            ``"next"``), which is correct whenever ``"next"`` may differ
            non-trivially from ``obs[t+1]``. Truthy values request a single
            call:

            - ``True``: fixed-budget single-call path. Inserts true
              ``next_obs`` after internal resets and masks the displaced
              suffix samples via ``"shifted_valid"``. Retained samples use
              exact next observations while keeping the static compute budget
              configured by ``shifted_budget``.
            - ``True``: fixed-budget single-call path. Inserts true
              ``next_obs`` after internal resets and masks the displaced
              suffix samples via ``"shifted_valid"``. Retained samples
              use exact next observations while keeping the static compute
              budget configured by ``shifted_budget``.

            All single-call paths require that the parameters at time
            ``t`` and ``t+1`` are identical (i.e. ``target_params`` is not
            used) and that the ``"next"`` value is shifted by exactly one
            time step (no multi-step returns). Defaults to ``False``.
        device (torch.device, optional): the device where the buffers will be instantiated.
            Defaults to ``torch.get_default_device()``.
        time_dim (int, optional): the dimension corresponding to the time
            in the input tensordict. If not provided, defaults to the dimension
            marked with the ``"time"`` name if any, and to the last dimension
            otherwise. Can be overridden during a call to
            :meth:`~.value_estimate`.
            Negative dimensions are considered with respect to the input
            tensordict.
        auto_reset_env (bool, optional): if ``True``, the last ``"next"`` state
            of the episode isn't valid, so the GAE calculation will use the ``value``
            instead of ``next_value`` to bootstrap truncated episodes.
        deactivate_vmap (bool, optional): if ``True``, no vmap call will be used, and
            vectorized maps will be replaced with simple for loops. Defaults to ``False``.
        value_chunk_size (int, optional): if set, splits value-network calls
            into chunks of this many elements along the leading dimension.
            Defaults to ``None``.
        shifted_budget (int, optional): number of extra value-network time slots
            used when ``shifted=True``. ``1`` uses a ``T+1``
            budget, ``2`` can represent one internal reset plus the rollout
            boundary without dropping samples, and so on. Defaults to ``1``.

    GAE will return an :obj:`"advantage"` entry containing the advantage value. It will also
    return a :obj:`"value_target"` entry with the return value that is to be used
    to train the value network. Finally, if :obj:`gradient_mode` is ``True``,
    an additional and differentiable :obj:`"value_error"` entry will be returned,
    which simply represents the difference between the return and the value network
    output (i.e. an additional distance loss should be applied to that signed value).

    .. note::
      As other advantage functions do, if the ``value_key`` is already present
      in the input tensordict, the GAE module will ignore the calls to the value
      network (if any) and use the provided value instead.

    .. note:: GAE can be used with value networks that rely on recurrent neural networks, provided that the
        init markers (`"is_init"`) and terminated / truncated markers are properly set.
        With ``shifted=True``, reset next-observations are inserted into a
        fixed-shape value-network call according to ``shifted_budget``. If ``shifted=False``,
        the root and ``"next"`` trajectories are stacked and the value network is called with ``vmap`` over the
        stack of trajectories. Because RNNs require a fair amount of control flow, they are currently not
        compatible with ``torch.vmap`` and, as such, the ``deactivate_vmap`` option must be turned on in these
        cases. Similarly, if ``shifted=False``, the ``"is_init"`` entry of the root tensordict will be copied
        onto the ``"is_init"`` of the ``"next"`` entry, such that trajectories are well separated both for root
        and ``"next"`` data.
    """

    value_network: TensorDictModule | None

    def __init__(
        self,
        *,
        gamma: float | torch.Tensor,
        lmbda: float | torch.Tensor,
        value_network: TensorDictModule | None,
        average_gae: bool = False,
        differentiable: bool = False,
        vectorized: bool | None = None,
        skip_existing: bool | None = None,
        advantage_key: NestedKey = None,
        value_target_key: NestedKey = None,
        value_key: NestedKey = None,
        shifted: bool = False,
        device: torch.device | None = None,
        time_dim: int | None = None,
        auto_reset_env: bool = False,
        deactivate_vmap: bool = False,
        value_chunk_size: int | None = None,
        shifted_budget: int = 1,
    ):
        super().__init__(
            shifted=shifted,
            value_network=value_network,
            differentiable=differentiable,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            skip_existing=skip_existing,
            device=device,
            deactivate_vmap=deactivate_vmap,
            value_chunk_size=value_chunk_size,
            shifted_budget=shifted_budget,
        )
        self.register_buffer(
            "gamma",
            gamma.to(self._device)
            if isinstance(gamma, Tensor)
            else torch.tensor(gamma, device=self._device),
        )
        self.register_buffer(
            "lmbda",
            lmbda.to(self._device)
            if isinstance(lmbda, Tensor)
            else torch.tensor(lmbda, device=self._device),
        )
        self.average_gae = average_gae
        self.vectorized = vectorized
        self.time_dim = time_dim
        self.auto_reset_env = auto_reset_env
        self.deactivate_vmap = deactivate_vmap

    @property
    def vectorized(self):
        if is_dynamo_compiling():
            return False
        return self._vectorized

    @vectorized.setter
    def vectorized(self, value):
        self._vectorized = value

    @_self_set_skip_existing
    @_self_set_grad_enabled
    @dispatch
    def forward(
        self,
        tensordict: TensorDictBase,
        *,
        params: list[Tensor] | None = None,
        target_params: list[Tensor] | None = None,
        time_dim: int | None = None,
    ) -> TensorDictBase:
        """Computes the GAE given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, ``"action"``, ``("next", "reward")``,
                ``("next", "done")``, ``("next", "terminated")``,
                and ``"next"`` tensordict state as returned by the environment)
                necessary to compute the value estimates and the GAE.
                The data passed to this module should be structured as :obj:`[*B, T, *F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the feature dimension(s).
                The tensordict must have shape ``[*B, T]``.

        Keyword Args:
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.
            time_dim (int, optional): the dimension corresponding to the time
                in the input tensordict. If not provided, defaults to the dimension
                marked with the ``"time"`` name if any, and to the last dimension
                otherwise.
                Negative dimensions are considered with respect to the input
                tensordict.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> from tensordict import TensorDict
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = GAE(
            ...     gamma=0.98,
            ...     lmbda=0.94,
            ...     value_network=value_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> tensordict = TensorDict({"obs": obs, "next": {"obs": next_obs}, "done": done, "reward": reward, "terminated": terminated}, [1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = TensorDictModule(
            ...     nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"]
            ... )
            >>> module = GAE(
            ...     gamma=0.98,
            ...     lmbda=0.94,
            ...     value_network=value_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> advantage, value_target = module(obs=obs, next_reward=reward, next_done=done, next_obs=next_obs, next_terminated=terminated)

        """
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimension, got "
                f"tensordict.batch_size = {tensordict.batch_size}"
            )
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device
        if self.gamma.device != device:
            self.gamma = self.gamma.to(device)
        gamma = self.gamma
        if self.lmbda.device != device:
            self.lmbda = self.lmbda.to(device)
        lmbda = self.lmbda
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network) if (
                params is None and target_params is None
            ) else nullcontext():
                # with torch.no_grad():
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value, valid = self._call_value_nets(
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                    vmap_randomness=self.vmap_randomness,
                )
                if valid is not None:
                    tensordict.set("shifted_valid", valid)
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))

            if value is None:
                raise ValueError(
                    f"The tensor with key {self.tensor_keys.value} is missing, and no value network was provided."
                )
            if next_value is None:
                raise ValueError(
                    f"The tensor with key {('next', self.tensor_keys.value)} is missing, and no value network was provided."
                )

        time_dim = self._get_time_dim(time_dim, tensordict)
        valid = tensordict.get("shifted_valid", default=None)
        data_for_value = self._prepare_shifted_tensordict(tensordict, valid, time_dim)
        reward = data_for_value.get(("next", self.tensor_keys.reward))
        done = data_for_value.get(("next", self.tensor_keys.done))
        terminated = data_for_value.get(
            ("next", self.tensor_keys.terminated), default=done
        )

        # Subclass extension hook: lets subclasses reshape / broadcast the
        # reward and done signals to match the value tensor before the
        # advantage recursion is run. Default: identity.
        reward, done, terminated = self._prepare_signals(
            reward, done, terminated, value
        )

        if self.auto_reset_env:
            truncated = tensordict.get(("next", "truncated"))
            truncated = self._broadcast_optional(truncated, value)
            if truncated.any():
                reward = reward + gamma * value * truncated

        if self.vectorized:
            adv, value_target = vec_generalized_advantage_estimate(
                gamma,
                lmbda,
                value,
                next_value,
                reward,
                done=done,
                terminated=terminated if not self.auto_reset_env else done,
                time_dim=time_dim,
            )
        else:
            adv, value_target = generalized_advantage_estimate(
                gamma,
                lmbda,
                value,
                next_value,
                reward,
                done=done,
                terminated=terminated if not self.auto_reset_env else done,
                time_dim=time_dim,
            )

        if self.average_gae:
            adv = self._normalize_advantage(adv, valid)

        tensordict.set(self.tensor_keys.advantage, adv)
        tensordict.set(self.tensor_keys.value_target, value_target)
        self._mask_shifted_output(tensordict, valid)

        return tensordict

    # -- extension hooks -----------------------------------------------------

    def _prepare_signals(
        self,
        reward: Tensor,
        done: Tensor,
        terminated: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Hook to reshape reward / done / terminated before the recursion.

        Default implementation is identity. :class:`MultiAgentGAE` overrides
        this to broadcast team-shared signals across the agent dim.
        """
        return reward, done, terminated

    def _broadcast_optional(self, tensor: Tensor, value: Tensor) -> Tensor:
        """Optional broadcast for the truncated signal used in auto_reset_env.

        Default: return ``tensor`` unchanged. Subclasses that broadcast
        rewards / done flags should typically override this with the same
        broadcasting policy.
        """
        return tensor

    def _normalize_advantage(
        self, adv: Tensor, valid: torch.Tensor | None = None
    ) -> Tensor:
        """Standardise the advantage tensor.

        Default standardises globally (single mean/std over the whole tensor).
        :class:`MultiAgentGAE` overrides this to leave the agent dim
        independent.
        """
        if valid is None:
            loc = adv.mean()
            scale = adv.std().clamp_min(1e-4)
            return (adv - loc) / scale
        mask = self._expand_to_match(valid, adv).to(adv.dtype)
        count = mask.sum().clamp_min(1)
        loc = (adv * mask).sum() / count
        scale = (((adv - loc).pow(2) * mask).sum() / count).sqrt().clamp_min(1e-4)
        return (adv - loc) / scale

    def value_estimate(
        self,
        tensordict,
        params: TensorDictBase | None = None,
        target_params: TensorDictBase | None = None,
        time_dim: int | None = None,
        **kwargs,
    ):
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got"
                f"tensordict.batch_size = {tensordict.batch_size}"
            )
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device
        if self.gamma.device != device:
            self.gamma = self.gamma.to(device)
        gamma = self.gamma
        if self.lmbda.device != device:
            self.lmbda = self.lmbda.to(device)
        lmbda = self.lmbda
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        time_dim = self._get_time_dim(time_dim, tensordict)

        if self.is_stateless and params is None:
            raise RuntimeError(
                "Expected params to be passed to advantage module but got none."
            )
        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network) if (
                params is None and target_params is None
            ) else nullcontext():
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value, valid = self._call_value_nets(
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                    vmap_randomness=self.vmap_randomness,
                )
                if valid is not None:
                    tensordict.set("shifted_valid", valid)
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))
        valid = tensordict.get("shifted_valid", default=None)
        data_for_value = self._prepare_shifted_tensordict(tensordict, valid, time_dim)
        reward = data_for_value.get(("next", self.tensor_keys.reward))
        done = data_for_value.get(("next", self.tensor_keys.done))
        terminated = data_for_value.get(
            ("next", self.tensor_keys.terminated), default=done
        )
        reward, done, terminated = self._prepare_signals(
            reward, done, terminated, value
        )
        _, value_target = vec_generalized_advantage_estimate(
            gamma,
            lmbda,
            value,
            next_value,
            reward,
            done=done,
            terminated=terminated,
            time_dim=time_dim,
        )
        return value_target


class MultiAgentGAE(GAE):
    """Multi-agent Generalized Advantage Estimator.

    Drop-in replacement for :class:`GAE` when the value network produces per-agent
    state values (shape ``[*B, T, n_agents, 1]``) but the reward / done /
    terminated signals are shared across agents at the team level
    (shape ``[*B, T, 1]``) — the standard cooperative-MARL layout in torchrl
    (see e.g. ``torchrl/envs/libs/vmas.py`` and
    ``torchrl/envs/libs/pettingzoo.py``).

    The estimator detects whether the reward/done/terminated tensors are missing
    the agent dimension relative to the value tensor, and broadcasts them along
    that dimension before running the standard vectorised GAE recursion. If the
    reward is already per-agent (e.g. a competitive setting), it is passed
    through unchanged.

    The output ``"advantage"`` and ``"value_target"`` entries match the shape
    of the value tensor (``[*B, T, n_agents, 1]``), which is what
    :class:`~torchrl.objectives.multiagent.MAPPOLoss` expects.

    Keyword Args:
        agent_dim (int, optional): the dimension that holds the agent index in
            the value tensor. Negative dimensions are taken modulo
            ``value.ndim``. Defaults to ``-2`` (penultimate), matching the
            convention used by :class:`~torchrl.modules.MultiAgentMLP`.

    Other args/kwargs are forwarded to :class:`GAE`.
    """

    def __init__(self, *, agent_dim: int = -2, **kwargs):
        super().__init__(**kwargs)
        self.agent_dim = agent_dim

    @staticmethod
    def _broadcast_to_agents(
        tensor: torch.Tensor, target: torch.Tensor, agent_dim: int
    ) -> torch.Tensor:
        """Expand ``tensor`` along ``agent_dim`` to match ``target``'s shape.

        If ``tensor`` already has the same number of dims as ``target`` we
        assume it is per-agent and return it unchanged. Otherwise we unsqueeze
        at ``agent_dim`` and expand.
        """
        if tensor.ndim == target.ndim:
            return tensor
        if tensor.ndim != target.ndim - 1:
            raise ValueError(
                f"MultiAgentGAE expected the reward/done/terminated tensor to "
                f"have either the same number of dims as the value tensor "
                f"(per-agent) or one fewer (team-shared). Got "
                f"tensor.shape={tuple(tensor.shape)}, "
                f"value.shape={tuple(target.shape)}."
            )
        dim = agent_dim if agent_dim >= 0 else target.ndim + agent_dim
        n_agents = target.shape[dim]
        unsqueezed = tensor.unsqueeze(dim)
        expand_shape = list(unsqueezed.shape)
        expand_shape[dim] = n_agents
        return unsqueezed.expand(expand_shape)

    # -- GAE extension hooks -------------------------------------------------

    def _prepare_signals(
        self,
        reward: Tensor,
        done: Tensor,
        terminated: Tensor,
        value: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        return (
            self._broadcast_to_agents(reward, value, self.agent_dim),
            self._broadcast_to_agents(done, value, self.agent_dim),
            self._broadcast_to_agents(terminated, value, self.agent_dim),
        )

    def _broadcast_optional(self, tensor: Tensor, value: Tensor) -> Tensor:
        # Used by GAE for the auto_reset_env ``truncated`` tensor — same
        # broadcasting policy as the other team signals.
        return self._broadcast_to_agents(tensor, value, self.agent_dim)

    def _normalize_advantage(
        self, adv: Tensor, valid: torch.Tensor | None = None
    ) -> Tensor:
        # Per-agent standardisation: normalise over batch + time but keep the
        # agent dim independent so high-variance agents are not flattened.
        agent_dim = self.agent_dim if self.agent_dim >= 0 else adv.ndim + self.agent_dim
        reduce_dims = [d for d in range(adv.ndim) if d != agent_dim]
        if valid is None:
            loc = adv.mean(dim=reduce_dims, keepdim=True)
            scale = adv.std(dim=reduce_dims, keepdim=True).clamp_min(1e-4)
            return (adv - loc) / scale
        mask = self._expand_to_match(valid, adv).to(adv.dtype)
        count = mask.sum(dim=reduce_dims, keepdim=True).clamp_min(1)
        loc = (adv * mask).sum(dim=reduce_dims, keepdim=True) / count
        scale = (
            (((adv - loc).pow(2) * mask).sum(dim=reduce_dims, keepdim=True) / count)
            .sqrt()
            .clamp_min(1e-4)
        )
        return (adv - loc) / scale


class VTrace(ValueEstimatorBase):
    """A class wrapper around V-Trace estimate functional.

    Refer to "IMPALA: Scalable Distributed Deep-RL with Importance Weighted  Actor-Learner Architectures"
    :ref:`here <https://arxiv.org/abs/1802.01561>`_ for more context.

    Keyword Args:
        gamma (scalar): exponential mean discount.
        value_network (TensorDictModule): value operator used to retrieve the value estimates.
        actor_network (TensorDictModule): actor operator used to retrieve the log prob.
        rho_thresh (Union[float, Tensor]): rho clipping parameter for importance weights.
            Defaults to ``1.0``.
        c_thresh (Union[float, Tensor]): c clipping parameter for importance weights.
            Defaults to ``1.0``.
        average_adv (bool): if ``True``, the resulting advantage values will be standardized.
            Default is ``False``.
        differentiable (bool, optional): if ``True``, gradients are propagated through
            the computation of the value function. Default is ``False``.

            .. note::
              The proper way to make the function call non-differentiable is to
              decorate it in a `torch.no_grad()` context manager/decorator or
              pass detached parameters for functional modules.
        skip_existing (bool, optional): if ``True``, the value network will skip
            modules which outputs are already present in the tensordict.
            Defaults to ``None``, i.e., the value of :func:`tensordict.nn.skip_existing()`
            is not affected.
            Defaults to "state_value".
        advantage_key (str or tuple of str, optional): [Deprecated] the key of
            the advantage entry.  Defaults to ``"advantage"``.
        value_target_key (str or tuple of str, optional): [Deprecated] the key
            of the advantage entry.  Defaults to ``"value_target"``.
        value_key (str or tuple of str, optional): [Deprecated] the value key to
            read from the input tensordict.  Defaults to ``"state_value"``.
        shifted (bool or str, optional): controls how value and next-value
            are obtained from the value network. ``False`` (default) calls
            the value network twice (once on the root tensordict, once on
            ``"next"``), which is correct whenever ``"next"`` may differ
            non-trivially from ``obs[t+1]``. Truthy values request a single
            call:

            - ``True``: fixed-budget single-call path. Inserts true
              ``next_obs`` after internal resets and masks the displaced
              suffix samples via ``"shifted_valid"``. Retained samples use
              exact next observations while keeping the static compute budget
              configured by ``shifted_budget``.
            - ``True``: fixed-budget single-call path. Inserts true
              ``next_obs`` after internal resets and masks the displaced
              suffix samples via ``"shifted_valid"``. Retained samples
              use exact next observations while keeping the static compute
              budget configured by ``shifted_budget``.

            All single-call paths require that the parameters at time
            ``t`` and ``t+1`` are identical (i.e. ``target_params`` is not
            used) and that the ``"next"`` value is shifted by exactly one
            time step (no multi-step returns). Defaults to ``False``.
        device (torch.device, optional): the device where the buffers will be instantiated.
            Defaults to ``torch.get_default_device()``.
        time_dim (int, optional): the dimension corresponding to the time
            in the input tensordict. If not provided, defaults to the dimension
            marked with the ``"time"`` name if any, and to the last dimension
            otherwise. Can be overridden during a call to
            :meth:`~.value_estimate`.
            Negative dimensions are considered with respect to the input
            tensordict.
        value_chunk_size (int, optional): if set, splits value-network calls
            into chunks of this many elements along the leading dimension.
            Defaults to ``None``.
        shifted_budget (int, optional): number of extra value-network time slots
            used when ``shifted=True``. ``1`` uses a ``T+1``
            budget, ``2`` can represent one internal reset plus the rollout
            boundary without dropping samples, and so on. Defaults to ``1``.

    VTrace will return an :obj:`"advantage"` entry containing the advantage value. It will also
    return a :obj:`"value_target"` entry with the V-Trace target value.

    .. note::
      As other advantage functions do, if the ``value_key`` is already present
      in the input tensordict, the VTrace module will ignore the calls to the value
      network (if any) and use the provided value instead.

    """

    def __init__(
        self,
        *,
        gamma: float | torch.Tensor,
        actor_network: TensorDictModule,
        value_network: TensorDictModule,
        rho_thresh: float | torch.Tensor = 1.0,
        c_thresh: float | torch.Tensor = 1.0,
        average_adv: bool = False,
        differentiable: bool = False,
        skip_existing: bool | None = None,
        advantage_key: NestedKey | None = None,
        value_target_key: NestedKey | None = None,
        value_key: NestedKey | None = None,
        shifted: bool = False,
        device: torch.device | None = None,
        time_dim: int | None = None,
        value_chunk_size: int | None = None,
        shifted_budget: int = 1,
    ):
        super().__init__(
            shifted=shifted,
            value_network=value_network,
            differentiable=differentiable,
            advantage_key=advantage_key,
            value_target_key=value_target_key,
            value_key=value_key,
            skip_existing=skip_existing,
            device=device,
            value_chunk_size=value_chunk_size,
            shifted_budget=shifted_budget,
        )
        if not isinstance(gamma, torch.Tensor):
            gamma = torch.tensor(gamma, device=self._device)
        if not isinstance(rho_thresh, torch.Tensor):
            rho_thresh = torch.tensor(rho_thresh, device=self._device)
        if not isinstance(c_thresh, torch.Tensor):
            c_thresh = torch.tensor(c_thresh, device=self._device)

        self.register_buffer("gamma", gamma)
        self.register_buffer("rho_thresh", rho_thresh)
        self.register_buffer("c_thresh", c_thresh)
        self.average_adv = average_adv
        self.actor_network = actor_network
        self.time_dim = time_dim

        if isinstance(gamma, torch.Tensor) and gamma.shape != ():
            raise NotImplementedError(
                "Per-value gamma is not supported yet. Gamma must be a scalar."
            )

    @property
    def in_keys(self):
        parent_in_keys = super().in_keys
        extended_in_keys = parent_in_keys + [self.tensor_keys.sample_log_prob]
        return extended_in_keys

    @_self_set_skip_existing
    @_self_set_grad_enabled
    @dispatch
    def forward(
        self,
        tensordict: TensorDictBase,
        *,
        params: list[Tensor] | None = None,
        target_params: list[Tensor] | None = None,
        time_dim: int | None = None,
    ) -> TensorDictBase:
        """Computes the V-Trace correction given the data in tensordict.

        If a functional module is provided, a nested TensorDict containing the parameters
        (and if relevant the target parameters) can be passed to the module.

        Args:
            tensordict (TensorDictBase): A TensorDict containing the data
                (an observation key, "action", "reward", "done" and "next" tensordict state
                as returned by the environment) necessary to compute the value estimates and the GAE.
                The data passed to this module should be structured as :obj:`[*B, T, F]` where :obj:`B` are
                the batch size, :obj:`T` the time dimension and :obj:`F` the feature dimension(s).

        Keyword Args:
            params (TensorDictBase, optional): A nested TensorDict containing the params
                to be passed to the functional value network module.
            target_params (TensorDictBase, optional): A nested TensorDict containing the
                target params to be passed to the functional value network module.
            time_dim (int, optional): the dimension corresponding to the time
                in the input tensordict. If not provided, defaults to the dimension
                marked with the ``"time"`` name if any, and to the last dimension
                otherwise.
                Negative dimensions are considered with respect to the input
                tensordict.

        Returns:
            An updated TensorDict with an advantage and a value_error keys as defined in the constructor.

        Examples:
            >>> value_net = TensorDictModule(nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"])
            >>> actor_net = TensorDictModule(nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"])
            >>> actor_net = ProbabilisticActor(
            ...     module=actor_net,
            ...     in_keys=["logits"],
            ...     out_keys=["action"],
            ...     distribution_class=OneHotCategorical,
            ...     return_log_prob=True,
            ... )
            >>> module = VTrace(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ...     actor_network=actor_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> sample_log_prob = torch.randn(1, 10, 1)
            >>> tensordict = TensorDict({
            ...     "obs": obs,
            ...     "done": done,
            ...     "terminated": terminated,
            ...     "sample_log_prob": sample_log_prob,
            ...     "next": {"obs": next_obs, "reward": reward, "done": done, "terminated": terminated},
            ... }, batch_size=[1, 10])
            >>> _ = module(tensordict)
            >>> assert "advantage" in tensordict.keys()

        The module supports non-tensordict (i.e. unpacked tensordict) inputs too:

        Examples:
            >>> value_net = TensorDictModule(nn.Linear(3, 1), in_keys=["obs"], out_keys=["state_value"])
            >>> actor_net = TensorDictModule(nn.Linear(3, 4), in_keys=["obs"], out_keys=["logits"])
            >>> actor_net = ProbabilisticActor(
            ...     module=actor_net,
            ...     in_keys=["logits"],
            ...     out_keys=["action"],
            ...     distribution_class=OneHotCategorical,
            ...     return_log_prob=True,
            ... )
            >>> module = VTrace(
            ...     gamma=0.98,
            ...     value_network=value_net,
            ...     actor_network=actor_net,
            ...     differentiable=False,
            ... )
            >>> obs, next_obs = torch.randn(2, 1, 10, 3)
            >>> reward = torch.randn(1, 10, 1)
            >>> done = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> terminated = torch.zeros(1, 10, 1, dtype=torch.bool)
            >>> sample_log_prob = torch.randn(1, 10, 1)
            >>> tensordict = TensorDict({
            ...     "obs": obs,
            ...     "done": done,
            ...     "terminated": terminated,
            ...     "sample_log_prob": sample_log_prob,
            ...     "next": {"obs": next_obs, "reward": reward, "done": done, "terminated": terminated},
            ... }, batch_size=[1, 10])
            >>> advantage, value_target = module(
            ...     obs=obs, next_reward=reward, next_done=done, next_obs=next_obs, next_terminated=terminated, sample_log_prob=sample_log_prob
            ... )

        """
        if tensordict.batch_dims < 1:
            raise RuntimeError(
                "Expected input tensordict to have at least one dimensions, got "
                f"tensordict.batch_size = {tensordict.batch_size}"
            )
        reward = tensordict.get(("next", self.tensor_keys.reward))
        device = reward.device

        if self.gamma.device != device:
            self.gamma = self.gamma.to(device)
        gamma = self.gamma
        steps_to_next_obs = tensordict.get(self.tensor_keys.steps_to_next_obs, None)
        if steps_to_next_obs is not None:
            gamma = gamma ** steps_to_next_obs.view_as(reward)

        # Make sure we have the value and next value
        if self.value_network is not None:
            if params is not None:
                params = params.detach()
                if target_params is None:
                    target_params = params.clone(False)
            with hold_out_net(self.value_network):
                # we may still need to pass gradient, but we don't want to assign grads to
                # value net params
                value, next_value, valid = self._call_value_nets(
                    data=tensordict,
                    params=params,
                    next_params=target_params,
                    single_call=self.shifted,
                    value_key=self.tensor_keys.value,
                    detach_next=True,
                    vmap_randomness=self.vmap_randomness,
                )
                if valid is not None:
                    tensordict.set("shifted_valid", valid)
        else:
            value = tensordict.get(self.tensor_keys.value)
            next_value = tensordict.get(("next", self.tensor_keys.value))

        lp = _maybe_get_or_select(tensordict, self.tensor_keys.sample_log_prob)
        if is_tensor_collection(lp):
            # Sum all values to match the batch size
            lp = lp.sum(dim="feature", reduce=True)
        log_mu = lp.view_as(value)

        # Compute log prob with current policy
        with hold_out_net(self.actor_network):
            log_pi = _call_actor_net(
                actor_net=self.actor_network,
                data=tensordict,
                params=None,
                log_prob_key=self.tensor_keys.sample_log_prob,
            )
            log_pi = log_pi.view_as(value)

        time_dim = self._get_time_dim(time_dim, tensordict)
        valid = tensordict.get("shifted_valid", default=None)
        data_for_value = self._prepare_shifted_tensordict(tensordict, valid, time_dim)
        reward = data_for_value.get(("next", self.tensor_keys.reward))

        # Compute the V-Trace correction
        done = data_for_value.get(("next", self.tensor_keys.done))
        terminated = data_for_value.get(("next", self.tensor_keys.terminated))

        adv, value_target = vtrace_advantage_estimate(
            gamma,
            log_pi,
            log_mu,
            value,
            next_value,
            reward,
            done,
            terminated,
            rho_thresh=self.rho_thresh,
            c_thresh=self.c_thresh,
            time_dim=time_dim,
        )

        if self.average_adv:
            loc = adv.mean()
            scale = adv.std().clamp_min(1e-5)
            adv = adv - loc
            adv = adv / scale

        tensordict.set(self.tensor_keys.advantage, adv)
        tensordict.set(self.tensor_keys.value_target, value_target)
        self._mask_shifted_output(tensordict, valid)

        return tensordict


def _deprecate_class(cls, new_cls):
    @wraps(cls.__init__)
    def new_init(self, *args, **kwargs):
        warnings.warn(f"class {cls} is deprecated, please use {new_cls} instead.")
        cls.__init__(self, *args, **kwargs)

    cls.__init__ = new_init


TD0Estimate = type("TD0Estimate", TD0Estimator.__bases__, dict(TD0Estimator.__dict__))
_deprecate_class(TD0Estimate, TD0Estimator)
TD1Estimate = type("TD1Estimate", TD1Estimator.__bases__, dict(TD1Estimator.__dict__))
_deprecate_class(TD1Estimate, TD1Estimator)
TDLambdaEstimate = type(
    "TDLambdaEstimate", TDLambdaEstimator.__bases__, dict(TDLambdaEstimator.__dict__)
)
_deprecate_class(TDLambdaEstimate, TDLambdaEstimator)
