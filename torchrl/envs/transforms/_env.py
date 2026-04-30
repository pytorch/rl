# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import multiprocessing as mp
import warnings
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TYPE_CHECKING

import torch

from tensordict import (
    LazyStackedTensorDict,
    NonTensorData,
    TensorDict,
    TensorDictBase,
    unravel_key,
)
from tensordict.base import _is_leaf_nontensor
from tensordict.nn import TensorDictModuleBase
from tensordict.utils import _zip_strict, expand_as_right, NestedKey

from torchrl._utils import _ends_with, _replace_last

from torchrl.data.tensor_specs import (
    Bounded,
    Categorical,
    Composite,
    TensorSpec,
    Unbounded,
)
from torchrl.envs.common import EnvBase
from torchrl.envs.transforms.utils import _get_reset
from torchrl.envs.utils import step_mdp

if TYPE_CHECKING:
    pass

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

from torchrl.envs.transforms._base import (
    _MAX_NOOPS_TRIALS,
    AutoResetEnv,
    FORWARD_NOT_IMPLEMENTED,
    Transform,
)

__all__ = [
    "AutoResetTransform",
    "BatchSizeTransform",
    "BurnInTransform",
    "FrameSkipTransform",
    "InitTracker",
    "NoopResetEnv",
    "RandomTruncationTransform",
    "StepCounter",
    "TensorDictPrimer",
    "TrajCounter",
    "gSDENoise",
]


class FrameSkipTransform(Transform):
    """A frame-skip transform.

    This transform applies the same action repeatedly in the parent environment,
    which improves stability on certain training sota-implementations.

    Args:
        frame_skip (int, optional): a positive integer representing the number
            of frames during which the same action must be applied.

    """

    def __init__(self, frame_skip: int = 1):
        super().__init__()
        if frame_skip < 1:
            raise ValueError("frame_skip should have a value greater or equal to one.")
        self.frame_skip = frame_skip

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        parent = self.parent
        if parent is None:
            raise RuntimeError("parent not found for FrameSkipTransform")
        reward_key = parent.reward_key
        reward = next_tensordict.get(reward_key)
        for _ in range(self.frame_skip - 1):
            next_tensordict = parent._step(tensordict)
            reward = reward + next_tensordict.get(reward_key)
        return next_tensordict.set(reward_key, reward)

    def forward(self, tensordict):
        raise RuntimeError(
            "FrameSkipTransform can only be used when appended to a transformed env."
        )


class NoopResetEnv(Transform):
    """Runs a series of random actions when an environment is reset.

    Args:
        env (EnvBase): env on which the random actions have to be
            performed. Can be the same env as the one provided to the
            TransformedEnv class
        noops (int, optional): upper-bound on the number of actions
            performed after reset. Default is `30`.
            If noops is too high such that it results in the env being
            done or truncated before the all the noops are applied,
            in multiple trials, the transform raises a RuntimeError.
        random (bool, optional): if False, the number of random ops will
            always be equal to the noops value. If True, the number of
            random actions will be randomly selected between 0 and noops.
            Default is `True`.

    """

    def __init__(self, noops: int = 30, random: bool = True):
        """Sample initial states by taking random number of no-ops on reset."""
        super().__init__()
        self.noops = noops
        self.random = random

    @property
    def base_env(self):
        return self.parent

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Do no-op action for a number of steps in [1, noop_max]."""
        parent = self.parent
        if parent is None:
            raise RuntimeError(
                "NoopResetEnv.parent not found. Make sure that the parent is set."
            )
        # Merge the two tensordicts
        tensordict = parent._reset_proc_data(tensordict.clone(False), tensordict_reset)
        # check that there is a single done state -- behavior is undefined for multiple dones
        done_keys = parent.done_keys
        reward_key = parent.reward_key
        if parent.batch_size.numel() > 1:
            raise ValueError(
                "The parent environment batch-size is non-null. "
                "NoopResetEnv is designed to work on single env instances, as partial reset "
                "is currently not supported. If you feel like this is a missing feature, submit "
                "an issue on TorchRL github repo. "
                "In case you are trying to use NoopResetEnv over a batch of environments, know "
                "that you can have a transformed batch of transformed envs, such as: "
                "`TransformedEnv(ParallelEnv(3, lambda: TransformedEnv(MyEnv(), NoopResetEnv(3))), OtherTransform())`."
            )

        noops = (
            self.noops if not self.random else torch.randint(self.noops, (1,)).item()
        )

        trial = 0
        while trial <= _MAX_NOOPS_TRIALS:
            i = 0

            while i < noops:
                i += 1
                tensordict = parent.rand_step(tensordict)
                reset = False
                # if any of the done_keys is True, we break
                for done_key in done_keys:
                    done = tensordict.get(("next", done_key))
                    if done.numel() > 1:
                        raise ValueError(
                            f"{type(self)} only supports scalar done states."
                        )
                    if done:
                        reset = True
                        break
                tensordict = step_mdp(tensordict, exclude_done=False)
                if reset:
                    tensordict = parent.reset(tensordict.clone(False))
                    break
            else:
                break

            trial += 1

        else:
            raise RuntimeError(
                f"Parent env was repeatedly done or truncated"
                f" before the sampled number of noops (={noops}) could be applied. "
            )
        tensordict_reset = tensordict
        return tensordict_reset.exclude(reward_key, inplace=True)

    def __repr__(self) -> str:
        random = self.random
        noops = self.noops
        class_name = self.__class__.__name__
        return f"{class_name}(noops={noops}, random={random})"


class TensorDictPrimer(Transform):
    """A primer for TensorDict initialization at reset time.

    This transform will populate the tensordict at reset with values drawn from
    the relative tensorspecs provided at initialization.
    If the transform is used out of the env context (e.g. as an nn.Module or
    appended to a replay buffer), a call to `forward` will also populate the
    tensordict with the desired features.

    Args:
        primers (dict or Composite, optional): a dictionary containing
            key-spec pairs which will be used to populate the input tensordict.
            :class:`~torchrl.data.Composite` instances are supported too.
        random (bool, optional): if ``True``, the values will be drawn randomly from
            the TensorSpec domain (or a unit Gaussian if unbounded). Otherwise a fixed value will be assumed.
            Defaults to `False`.
        default_value (:obj:`float`, Callable, Dict[NestedKey, float], Dict[NestedKey, Callable], optional): If non-random
            filling is chosen, `default_value` will be used to populate the tensors.

            - If `default_value` is a float or any other scala, all elements of the tensors will be set to that value.
            - If it is a callable and `single_default_value=False` (default), this callable is expected to return a tensor
              fitting the specs (ie, ``default_value()`` will be called independently for each leaf spec).
            - If it is a callable and ``single_default_value=True``, then the callable will be called just once and it is expected
              that the structure of its returned TensorDict instance or equivalent will match the provided specs.
              The ``default_value`` must accept an optional `reset` keyword argument indicating which envs are to be reset.
              The returned `TensorDict` must have as many elements as the number of envs to reset.

              .. seealso:: :class:`~torchrl.envs.DataLoadingPrimer`

            - Finally, if `default_value` is a dictionary of tensors or a dictionary of callables with keys matching
              those of the specs, these will be used to generate the corresponding tensors. Defaults to `0.0`.

        reset_key (NestedKey, optional): the reset key to be used as partial
            reset indicator. Must be unique. If not provided, defaults to the
            only reset key of the parent environment (if it has only one)
            and raises an exception otherwise.
        single_default_value (bool, optional): if ``True`` and `default_value` is a callable, it will be expected that
            ``default_value`` returns a single tensordict matching the specs. If `False`, `default_value()` will be
            called independently for each leaf. Defaults to ``False``.
        call_before_env_reset (bool, optional): if ``True``, the tensordict is populated before `env.reset` is called.
            Defaults to ``False``.
        **kwargs: each keyword argument corresponds to a key in the tensordict.
            The corresponding value has to be a TensorSpec instance indicating
            what the value must be.

    When used in a `TransformedEnv`, the spec shapes must match the environment's shape if
    the parent environment is batch-locked (`env.batch_locked=True`). If the spec shapes and
    parent shapes do not match, the spec shapes are modified in-place to match the leading
    dimensions of the parent's batch size. This adjustment is made for cases where the parent
    batch size dimension is not known during instantiation.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.envs import SerialEnv
        >>> base_env = SerialEnv(2, lambda: GymEnv("Pendulum-v1"))
        >>> env = TransformedEnv(base_env)
        >>> # the env is batch-locked, so the leading dims of the spec must match those of the env
        >>> env.append_transform(TensorDictPrimer(mykey=Unbounded([2, 3])))
        >>> td = env.reset()
        >>> print(td)
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                mykey: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([2]),
            device=cpu,
            is_shared=False)
        >>> # the entry is populated with 0s
        >>> print(td.get("mykey"))
        tensor([[0., 0., 0.],
                [0., 0., 0.]])

    When calling ``env.step()``, the current value of the key will be carried
    in the ``"next"`` tensordict __unless it already exists__.

    Examples:
        >>> td = env.rand_step(td)
        >>> print(td.get(("next", "mykey")))
        tensor([[0., 0., 0.],
                [0., 0., 0.]])
        >>> # with another value for "mykey", the previous value is not carried on
        >>> td = env.reset()
        >>> td = td.set(("next", "mykey"), torch.ones(2, 3))
        >>> td = env.rand_step(td)
        >>> print(td.get(("next", "mykey")))
        tensor([[1., 1., 1.],
                [1., 1., 1.]])

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> from torchrl.envs import SerialEnv, TransformedEnv
        >>> from torchrl.modules.utils import get_primers_from_module
        >>> from torchrl.modules import GRUModule
        >>> base_env = SerialEnv(2, lambda: GymEnv("Pendulum-v1"))
        >>> env = TransformedEnv(base_env)
        >>> model = GRUModule(input_size=2, hidden_size=2, in_key="observation", out_key="action")
        >>> primers = get_primers_from_module(model)
        >>> print(primers) # Primers shape is independent of the env batch size
        TensorDictPrimer(primers=Composite(
            recurrent_state: UnboundedContinuous(
                shape=torch.Size([1, 2]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([1, 2]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([1, 2]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous),
            device=None,
            shape=torch.Size([])), default_value={'recurrent_state': 0.0}, random=None)
        >>> env.append_transform(primers)
        >>> print(env.reset()) # The primers are automatically expanded to match the env batch size
        TensorDict(
            fields={
                done: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                observation: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                recurrent_state: Tensor(shape=torch.Size([2, 1, 2]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([2]),
            device=None,
            is_shared=False)

    .. note:: Some TorchRL modules rely on specific keys being present in the environment TensorDicts,
        like :class:`~torchrl.modules.models.LSTM` or :class:`~torchrl.modules.models.GRU`.
        To facilitate this process, the method :func:`~torchrl.modules.utils.get_primers_from_module`
        automatically checks for required primer transforms in a module and its submodules and
        generates them.
    """

    def __init__(
        self,
        primers: dict | Composite | None = None,
        random: bool | None = None,
        default_value: float
        | Callable
        | dict[NestedKey, float]
        | dict[NestedKey, Callable]
        | None = None,
        reset_key: NestedKey | None = None,
        expand_specs: bool | None = None,
        single_default_value: bool = False,
        call_before_env_reset: bool = False,
        **kwargs,
    ):
        self.device = kwargs.pop("device", None)
        if primers is not None:
            if kwargs:
                raise RuntimeError(
                    f"providing the primers as a dictionary is incompatible with extra keys "
                    f"'{kwargs.keys()}' provided as kwargs."
                )
            kwargs = primers
        if not isinstance(kwargs, Composite):
            shape = kwargs.pop("shape", None)
            device = self.device
            if "batch_size" in kwargs.keys():
                extra_kwargs = {"batch_size": kwargs.pop("batch_size")}
            else:
                extra_kwargs = {}
            primers = Composite(kwargs, device=device, shape=shape, **extra_kwargs)
        self.primers = primers
        self.expand_specs = expand_specs
        self.call_before_env_reset = call_before_env_reset

        if random and default_value:
            raise ValueError(
                "Setting random to True and providing a default_value are incompatible."
            )
        default_value = (
            default_value or 0.0
        )  # if not random and no default value, use 0.0
        self.random = random
        if isinstance(default_value, dict):
            default_value = TensorDict(default_value, [])
            default_value_keys = default_value.keys(
                True,
                True,
                is_leaf=lambda x: issubclass(x, (NonTensorData, torch.Tensor)),
            )
            if set(default_value_keys) != set(self.primers.keys(True, True)):
                raise ValueError(
                    "If a default_value dictionary is provided, it must match the primers keys."
                )
        elif single_default_value:
            pass
        else:
            default_value = {
                key: default_value for key in self.primers.keys(True, True)
            }
        self.single_default_value = single_default_value
        self.default_value = default_value
        self._validated = False
        self.reset_key = reset_key

        # sanity check
        for spec in self.primers.values(True, True):
            if not isinstance(spec, TensorSpec):
                raise ValueError(
                    "The values of the primers must be a subtype of the TensorSpec class. "
                    f"Got {type(spec)} instead."
                )
        super().__init__()

    @property
    def reset_key(self):
        reset_key = self.__dict__.get("_reset_key")
        if reset_key is None:
            if self.parent is None:
                raise RuntimeError(
                    "Missing parent, cannot infer reset_key automatically."
                )
            reset_keys = self.parent.reset_keys
            if len(reset_keys) > 1:
                raise RuntimeError(
                    f"Got more than one reset key in env {self.container}, cannot infer which one to use. "
                    f"Consider providing the reset key in the {type(self)} constructor."
                )
            reset_key = self._reset_key = reset_keys[0]
        return reset_key

    @reset_key.setter
    def reset_key(self, value):
        self._reset_key = value

    @property
    def device(self):
        device = self._device
        if device is None and hasattr(self, "parent") and self.parent is not None:
            device = self.parent.device
            self._device = device
        return device

    @device.setter
    def device(self, value):
        if value is None:
            self._device = None
            return
        self._device = torch.device(value)

    def to(self, *args, **kwargs):
        device, dtype, non_blocking, convert_to_format = torch._C._nn._parse_to(
            *args, **kwargs
        )
        if device is not None:
            self.device = device
            self.empty_cache()
            self.primers = self.primers.to(device)
        return super().to(*args, **kwargs)

    def _expand_shape(self, spec):
        return spec.expand((*self.parent.batch_size, *spec.shape))

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        if not isinstance(observation_spec, Composite):
            raise ValueError(
                f"observation_spec was expected to be of type Composite. Got {type(observation_spec)} instead."
            )

        if self.primers.shape[: observation_spec.ndim] != observation_spec.shape:
            if self.expand_specs:
                self.primers = self._expand_shape(self.primers)
            elif self.expand_specs is None:
                raise RuntimeError(
                    f"expand_specs wasn't specified in the {type(self).__name__} constructor, and the shape of the primers "
                    f"and observation specs mismatch ({self.primers.shape=} and {observation_spec.shape=}) - indicating a batch-size incongruency. Make sure the expand_specs arg "
                    f"is properly set or that the primer shape matches the environment batch-size."
                )
            else:
                self.primers.shape = observation_spec.shape

        device = observation_spec.device
        observation_spec.update(self.primers.clone().to(device))
        return observation_spec

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        new_state_spec = self.transform_observation_spec(input_spec["full_state_spec"])
        for action_key in list(input_spec["full_action_spec"].keys(True, True)):
            if action_key in new_state_spec.keys(True, True):
                input_spec["full_action_spec", action_key] = new_state_spec[action_key]
                del new_state_spec[action_key]
        input_spec["full_state_spec"] = new_state_spec
        return input_spec

    @property
    def _batch_size(self) -> torch.Size:
        return self.parent.batch_size

    def _validate_value_tensor(self, value, spec) -> bool:
        if not spec.is_in(value):
            raise ValueError(
                f"spec {spec}, spec.shape {spec.shape}, value.shape {value.shape}, spec.device {spec.device}, value.device {value.device}, spec.dtype {spec.dtype}, value.dtype {value.dtype}"
            )
            raise RuntimeError(f"Value ({value}) is not in the spec domain ({spec}).")
        return True

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.single_default_value and callable(self.default_value):
            tensordict.update(self.default_value())
            for key, spec in self.primers.items(True, True):
                if not self._validated:
                    self._validate_value_tensor(tensordict.get(key), spec)
            if not self._validated:
                self._validated = True
            return tensordict
        for key, spec in self.primers.items(True, True):
            if spec.shape[: len(tensordict.shape)] != tensordict.shape:
                raise RuntimeError(
                    "The leading shape of the spec must match the tensordict's, "
                    "but it does not: got "
                    f"tensordict.shape={tensordict.shape} whereas {key} spec's shape is "
                    f"{spec.shape}."
                )
            if self.random:
                value = spec.rand()
            else:
                value = self.default_value[key]
                if callable(value):
                    value = value()
                    if not self._validated:
                        self._validate_value_tensor(value, spec)
                else:
                    value = torch.full(
                        spec.shape,
                        value,
                        device=spec.device,
                    )

            tensordict.set(key, value)
        if not self._validated:
            self._validated = True
        return tensordict

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        for key in self.primers.keys(True, True):
            # We relax a bit the condition here, allowing nested but not leaf values to
            #  be checked against
            if key not in next_tensordict.keys(True, is_leaf=_is_leaf_nontensor):
                prev_val = tensordict.get(key)
                next_tensordict.set(key, prev_val)
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        """Sets the default values in the input tensordict.

        If the parent is batch-locked, we make sure the specs have the appropriate leading
        shape. We allow for execution when the parent is missing, in which case the
        spec shape is assumed to match the tensordict's.
        """
        if self.call_before_env_reset:
            return tensordict_reset
        return self._reset_func(tensordict, tensordict_reset)

    def _reset_env_preprocess(self, tensordict: TensorDictBase) -> TensorDictBase:
        if not self.call_before_env_reset:
            return tensordict
        if tensordict is None:
            parent = self.parent
            if parent is not None:
                device = parent.device
                batch_size = parent.batch_size
            else:
                device = getattr(self, "device", None)
                batch_size = getattr(self, "batch_size", ())
            tensordict = TensorDict(device=device, batch_size=batch_size)
        return self._reset_func(tensordict, tensordict)

    def _reset_func(
        self, tensordict, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        _reset = _get_reset(self.reset_key, tensordict)
        if (
            self.parent
            and self.parent.batch_locked
            and self.primers.shape[: len(self.parent.shape)] != self.parent.batch_size
        ):
            self.primers = self._expand_shape(self.primers)
        if _reset.any():
            if self.single_default_value and callable(self.default_value):
                if not _reset.all():
                    # FIXME: use masked op
                    # tensordict_reset = tensordict_reset.clone()
                    reset_val = self.default_value(reset=_reset)
                    # This is safE because env.reset calls _update_during_reset which will discard the new data
                    # tensordict_reset = (
                    #     self.container.full_observation_spec.zero().select(
                    #         *reset_val.keys(True)
                    #     )
                    # )
                    tensordict_reset = reset_val.new_zeros(
                        _reset.shape, empty_lazy=True
                    )
                    tensordict_reset[_reset] = reset_val
                else:
                    resets = self.default_value(reset=_reset)
                    tensordict_reset.update(resets)

                for key, spec in self.primers.items(True, True):
                    if not self._validated:
                        self._validate_value_tensor(tensordict_reset.get(key), spec)
                self._validated = True
                return tensordict_reset

            for key, spec in self.primers.items(True, True):
                if self.random:
                    shape = (
                        ()
                        if (not self.parent or self.parent.batch_locked)
                        else tensordict.batch_size
                    )
                    value = spec.rand(shape)
                else:
                    value = self.default_value[key]
                    if callable(value):
                        value = value()
                        if not self._validated:
                            self._validate_value_tensor(value, spec)
                    else:
                        value = torch.full(
                            spec.shape,
                            value,
                            device=spec.device,
                        )
                        prev_val = tensordict.get(key, 0.0)
                        value = torch.where(
                            expand_as_right(_reset, value), value, prev_val
                        )
                tensordict_reset.set(key, value)
            self._validated = True
        return tensordict_reset

    def __repr__(self) -> str:
        class_name = self.__class__.__name__
        if callable(self.default_value):
            default_value = self.default_value
        else:
            default_value = {
                key: value if isinstance(value, float) else "Callable"
                for key, value in self.default_value.items()
            }
        return f"{class_name}(primers={self.primers}, default_value={default_value}, random={self.random})"


class gSDENoise(TensorDictPrimer):
    """A gSDE noise initializer.

    See the :func:`~torchrl.modules.models.exploration.gSDEModule` for more info.
    """

    def __init__(
        self,
        state_dim=None,
        action_dim=None,
        shape=None,
        **kwargs,
    ) -> None:
        self.state_dim = state_dim
        self.action_dim = action_dim
        if shape is None:
            shape = ()
        tail_dim = (
            (1,) if state_dim is None or action_dim is None else (action_dim, state_dim)
        )
        random = state_dim is not None and action_dim is not None
        feat_shape = tuple(shape) + tail_dim
        primers = Composite({"_eps_gSDE": Unbounded(shape=feat_shape)}, shape=shape)
        super().__init__(primers=primers, random=random, **kwargs)


class StepCounter(Transform):
    """Counts the steps from a reset and optionally sets the truncated state to ``True`` after a certain number of steps.

    The ``"done"`` state is also adapted accordingly (as done is the disjunction
    of task completion and early truncation).

    Args:
        max_steps (int, optional): a positive integer that indicates the
            maximum number of steps to take before setting the ``truncated_key``
            entry to ``True``.
        truncated_key (str, optional): the key where the truncated entries
            should be written. Defaults to ``"truncated"``, which is recognised by
            data collectors as a reset signal.
            This argument can only be a string (not a nested key) as it will be
            matched to each of the leaf done key in the parent environment
            (eg, a ``("agent", "done")`` key will be accompanied by a
            ``("agent", "truncated")`` if the ``"truncated"`` key name is used).
        step_count_key (str, optional): the key where the step count entries
            should be written. Defaults to ``"step_count"``.
            This argument can only be a string (not a nested key) as it will be
            matched to each of the leaf done key in the parent environment
            (eg, a ``("agent", "done")`` key will be accompanied by a
            ``("agent", "step_count")`` if the ``"step_count"`` key name is used).
        update_done (bool, optional): if ``True``, the ``"done"`` boolean tensor
            at the level of ``"truncated"``
            will be updated.
            This signal indicates that the trajectory has reached its ends,
            either because the task is completed (``"completed"`` entry is
            ``True``) or because it has been truncated (``"truncated"`` entry
            is ``True``).
            Defaults to ``True``.

    .. note:: To ensure compatibility with environments that have multiple
        done_key(s), this transform will write a step_count entry for
        every done entry within the tensordict.

    Examples:
        >>> import gymnasium
        >>> from torchrl.envs import GymWrapper
        >>> base_env = GymWrapper(gymnasium.make("Pendulum-v1"))
        >>> env = TransformedEnv(base_env,
        ...     StepCounter(max_steps=5))
        >>> rollout = env.rollout(100)
        >>> print(rollout)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                completed: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        completed: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                        observation: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        step_count: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                        truncated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([5]),
                    device=cpu,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([5, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                step_count: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.int64, is_shared=False),
                truncated: Tensor(shape=torch.Size([5, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([5]),
            device=cpu,
            is_shared=False)
        >>> print(rollout["next", "step_count"])
        tensor([[1],
                [2],
                [3],
                [4],
                [5]])

    """

    invertible = False

    def __init__(
        self,
        max_steps: int | None = None,
        truncated_key: str | None = "truncated",
        step_count_key: str | None = "step_count",
        update_done: bool = True,
    ):
        if max_steps is not None and max_steps < 1:
            raise ValueError("max_steps should have a value greater or equal to one.")
        if not isinstance(truncated_key, str):
            raise ValueError("truncated_key must be a string.")
        if not isinstance(step_count_key, str):
            raise ValueError("step_count_key must be a string.")
        self.max_steps = max_steps
        self.truncated_key = truncated_key
        self.step_count_key = step_count_key
        self.update_done = update_done
        super().__init__()

    @property
    def truncated_keys(self) -> list[NestedKey]:
        truncated_keys = self.__dict__.get("_truncated_keys", None)
        if truncated_keys is None:
            # make the default truncated keys
            truncated_keys = []
            for reset_key in self.parent._filtered_reset_keys:
                if isinstance(reset_key, str):
                    key = self.truncated_key
                else:
                    key = (*reset_key[:-1], self.truncated_key)
                truncated_keys.append(key)
        self._truncated_keys = truncated_keys
        return truncated_keys

    @property
    def all_truncated_keys(self) -> list[NestedKey]:
        """Returns truncated keys for ALL reset keys (including nested ones).

        Used for propagating truncated to nested agent-level keys in MARL envs.
        """
        all_truncated_keys = self.__dict__.get("_all_truncated_keys", None)
        if all_truncated_keys is None:
            all_truncated_keys = []
            if self.parent is None:
                return self.truncated_keys
            for reset_key in self.parent.reset_keys:
                if isinstance(reset_key, str):
                    key = self.truncated_key
                else:
                    key = (*reset_key[:-1], self.truncated_key)
                all_truncated_keys.append(key)
        self.__dict__["_all_truncated_keys"] = all_truncated_keys
        return all_truncated_keys

    @property
    def done_keys(self) -> list[NestedKey]:
        done_keys = self.__dict__.get("_done_keys", None)
        if done_keys is None:
            # make the default done keys
            done_keys = []
            for reset_key in self.parent._filtered_reset_keys:
                if isinstance(reset_key, str):
                    key = "done"
                else:
                    key = (*reset_key[:-1], "done")
                done_keys.append(key)
        self.__dict__["_done_keys"] = done_keys
        return done_keys

    @property
    def all_done_keys(self) -> list[NestedKey]:
        """Returns done keys for ALL reset keys (including nested ones).

        Used for propagating done to nested agent-level keys in MARL envs.
        """
        all_done_keys = self.__dict__.get("_all_done_keys", None)
        if all_done_keys is None:
            all_done_keys = []
            if self.parent is None:
                return self.done_keys
            for reset_key in self.parent.reset_keys:
                if isinstance(reset_key, str):
                    key = "done"
                else:
                    key = (*reset_key[:-1], "done")
                all_done_keys.append(key)
        self.__dict__["_all_done_keys"] = all_done_keys
        return all_done_keys

    @property
    def terminated_keys(self) -> list[NestedKey]:
        terminated_keys = self.__dict__.get("_terminated_keys", None)
        if terminated_keys is None:
            # make the default terminated keys
            terminated_keys = []
            for reset_key in self.parent._filtered_reset_keys:
                if isinstance(reset_key, str):
                    key = "terminated"
                else:
                    key = (*reset_key[:-1], "terminated")
                terminated_keys.append(key)
        self.__dict__["_terminated_keys"] = terminated_keys
        return terminated_keys

    @property
    def step_count_keys(self) -> list[NestedKey]:
        step_count_keys = self.__dict__.get("_step_count_keys", None)
        if step_count_keys is None:
            # make the default step_count keys
            step_count_keys = []
            for reset_key in self.parent._filtered_reset_keys:
                if isinstance(reset_key, str):
                    key = self.step_count_key
                else:
                    key = (*reset_key[:-1], self.step_count_key)
                step_count_keys.append(key)
        self.__dict__["_step_count_keys"] = step_count_keys
        return step_count_keys

    @property
    def reset_keys(self) -> list[NestedKey]:
        if self.parent is not None:
            return self.parent._filtered_reset_keys
        # fallback on default "_reset"
        return ["_reset"]

    @property
    def full_done_spec(self) -> TensorSpec | None:
        return self.parent.output_spec["full_done_spec"] if self.parent else None

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        # get reset signal
        for (
            step_count_key,
            truncated_key,
            terminated_key,
            reset_key,
            done_key,
        ) in _zip_strict(
            self.step_count_keys,
            self.truncated_keys,
            self.terminated_keys,
            self.reset_keys,
            self.done_keys,
        ):
            reset = tensordict.get(reset_key, default=None)
            if reset is None:
                # get done status, just to inform the reset shape, dtype and device
                for entry_name in (terminated_key, truncated_key, done_key):
                    done = tensordict.get(entry_name, default=None)
                    if done is not None:
                        break
                else:
                    # It may be the case that reset did not provide a done state, in which case
                    # we fall back on the spec
                    done = self.parent.full_done_spec_unbatched[entry_name].zero(
                        tensordict_reset.shape
                    )
                reset = torch.ones_like(done)

            step_count = tensordict.get(step_count_key, default=None)
            if step_count is None:
                step_count = self.container.observation_spec[step_count_key].zero()
                if step_count.device != reset.device:
                    step_count = step_count.to(reset.device, non_blocking=True)

            # zero the step count if reset is needed
            step_count = torch.where(~reset, step_count.expand_as(reset), 0)
            tensordict_reset.set(step_count_key, step_count)
            if self.max_steps is not None:
                truncated = step_count >= self.max_steps
                truncated = truncated | tensordict_reset.get(truncated_key, False)
                if self.update_done:
                    # we assume no done after reset
                    tensordict_reset.set(done_key, truncated)
                tensordict_reset.set(truncated_key, truncated)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        for step_count_key, truncated_key, done_key in _zip_strict(
            self.step_count_keys, self.truncated_keys, self.done_keys
        ):
            step_count = tensordict.get(step_count_key)
            next_step_count = step_count + 1
            next_tensordict.set(step_count_key, next_step_count)

            if self.max_steps is not None:
                truncated = next_step_count >= self.max_steps
                truncated = truncated | next_tensordict.get(truncated_key, False)
                if self.update_done:
                    done = next_tensordict.get(done_key, None)

                    # we can have terminated and truncated
                    # terminated = next_tensordict.get(terminated_key, None)
                    # if terminated is not None:
                    #     truncated = truncated & ~terminated

                    done = truncated | done  # we assume no done after reset
                    next_tensordict.set(done_key, done)
                next_tensordict.set(truncated_key, truncated)

        # Propagate truncated/done to nested agent-level keys in MARL envs
        # This ensures that when max_steps is reached, all agent truncated/done keys are updated
        if self.max_steps is not None:
            self._propagate_to_nested_keys(next_tensordict)

        return next_tensordict

    def _propagate_to_nested_keys(self, next_tensordict: TensorDictBase) -> None:
        """Propagate truncated and done values to nested agent-level keys.

        In MARL envs, there may be nested agent-level truncated/done keys that
        are children of the root truncated/done. When StepCounter sets truncated
        at the root level, we need to propagate this to nested keys.
        """
        # Get the set of keys we already updated (filtered keys)
        updated_truncated = set(self.truncated_keys)
        updated_done = set(self.done_keys)

        # Propagate truncated to nested keys
        for nested_key in self.all_truncated_keys:
            if nested_key in updated_truncated:
                continue
            # Find the parent truncated key that should be propagated
            nested_truncated = next_tensordict.get(nested_key, None)
            if nested_truncated is None:
                continue
            # Find a parent truncated key to propagate from
            for parent_key in self.truncated_keys:
                parent_truncated = next_tensordict.get(parent_key, None)
                if parent_truncated is not None:
                    # Insert extra dims (e.g. agent dims) so the parent is
                    # broadcastable to the nested agent-level shape.
                    parent_val = parent_truncated
                    while parent_val.ndim < nested_truncated.ndim:
                        parent_val = parent_val.unsqueeze(-2)
                    expanded = parent_val.expand_as(nested_truncated)
                    next_tensordict.set(nested_key, nested_truncated | expanded)
                    break

        # Propagate done to nested keys if update_done is True
        if self.update_done:
            for nested_key in self.all_done_keys:
                if nested_key in updated_done:
                    continue
                nested_done = next_tensordict.get(nested_key, None)
                if nested_done is None:
                    continue
                # Find a parent done key to propagate from
                for parent_key in self.done_keys:
                    parent_done = next_tensordict.get(parent_key, None)
                    if parent_done is not None:
                        parent_val = parent_done
                        while parent_val.ndim < nested_done.ndim:
                            parent_val = parent_val.unsqueeze(-2)
                        expanded = parent_val.expand_as(nested_done)
                        next_tensordict.set(nested_key, nested_done | expanded)
                        break

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        if not isinstance(observation_spec, Composite):
            raise ValueError(
                f"observation_spec was expected to be of type Composite. Got {type(observation_spec)} instead."
            )
        full_done_spec = self.parent.output_spec["full_done_spec"]
        for step_count_key in self.step_count_keys:
            step_count_key = unravel_key(step_count_key)
            # find a matching done key (there might be more than one)
            for done_key in self.done_keys:
                # check root
                if type(done_key) is not type(step_count_key):
                    continue
                if isinstance(done_key, tuple):
                    if done_key[:-1] == step_count_key[:-1]:
                        shape = full_done_spec[done_key].shape
                        break
                if isinstance(done_key, str):
                    shape = full_done_spec[done_key].shape
                    break

            else:
                raise KeyError(
                    f"Could not find root of step_count_key {step_count_key} in done keys {self.done_keys}."
                )
            observation_spec[step_count_key] = Bounded(
                shape=shape,
                dtype=torch.int64,
                device=observation_spec.device,
                low=0,
                high=torch.iinfo(torch.int64).max,
            )
        return super().transform_observation_spec(observation_spec)

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        if self.max_steps:
            full_done_spec = self.parent.output_spec["full_done_spec"]
            for truncated_key in self.truncated_keys:
                truncated_key = unravel_key(truncated_key)
                # find a matching done key (there might be more than one)
                for done_key in self.done_keys:
                    # check root
                    if type(done_key) is not type(truncated_key):
                        continue
                    if isinstance(done_key, tuple):
                        if done_key[:-1] == truncated_key[:-1]:
                            shape = full_done_spec[done_key].shape
                            break
                    if isinstance(done_key, str):
                        shape = full_done_spec[done_key].shape
                        break

                else:
                    raise KeyError(
                        f"Could not find root of truncated_key {truncated_key} in done keys {self.done_keys}."
                    )
                full_done_spec[truncated_key] = Categorical(
                    2, dtype=torch.bool, device=output_spec.device, shape=shape
                )
            if self.update_done:
                for done_key in self.done_keys:
                    done_key = unravel_key(done_key)
                    # find a matching done key (there might be more than one)
                    for done_key in self.done_keys:
                        # check root
                        if type(done_key) is not type(done_key):
                            continue
                        if isinstance(done_key, tuple):
                            if done_key[:-1] == done_key[:-1]:
                                shape = full_done_spec[done_key].shape
                                break
                        if isinstance(done_key, str):
                            shape = full_done_spec[done_key].shape
                            break

                    else:
                        raise KeyError(
                            f"Could not find root of stop_key {done_key} in done keys {self.done_keys}."
                        )
                    full_done_spec[done_key] = Categorical(
                        2, dtype=torch.bool, device=output_spec.device, shape=shape
                    )
            output_spec["full_done_spec"] = full_done_spec
        return super().transform_output_spec(output_spec)

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        if not isinstance(input_spec, Composite):
            raise ValueError(
                f"input_spec was expected to be of type Composite. Got {type(input_spec)} instead."
            )
        if input_spec["full_state_spec"] is None:
            input_spec["full_state_spec"] = Composite(
                shape=input_spec.shape, device=input_spec.device
            )

        full_done_spec = self.parent.output_spec["full_done_spec"]
        for step_count_key in self.step_count_keys:
            step_count_key = unravel_key(step_count_key)
            # find a matching done key (there might be more than one)
            for done_key in self.done_keys:
                # check root
                if type(done_key) is not type(step_count_key):
                    continue
                if isinstance(done_key, tuple):
                    if done_key[:-1] == step_count_key[:-1]:
                        shape = full_done_spec[done_key].shape
                        break
                if isinstance(done_key, str):
                    shape = full_done_spec[done_key].shape
                    break

            else:
                raise KeyError(
                    f"Could not find root of step_count_key {step_count_key} in done keys {self.done_keys}."
                )

            input_spec[unravel_key(("full_state_spec", step_count_key))] = Bounded(
                shape=shape,
                dtype=torch.int64,
                device=input_spec.device,
                low=0,
                high=torch.iinfo(torch.int64).max,
            )

        return input_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            "StepCounter cannot be called independently, only its step and reset methods "
            "are functional. The reason for this is that it is hard to consider using "
            "StepCounter with non-sequential data, such as those collected by a replay buffer "
            "or a dataset. If you need StepCounter to work on a batch of sequential data "
            "(ie as LSTM would work over a whole sequence of data), file an issue on "
            "TorchRL requesting that feature."
        )


class RandomTruncationTransform(Transform):
    """Randomly truncate episodes to decorrelate synchronized batched envs.

    When many batched environments share the same ``max_episode_steps``, all
    environments hit truncation at nearly the same step, creating correlated
    waves of start-of-episode data in the replay buffer.  This transform
    breaks that synchronisation by assigning each environment a random horizon.

    On the **first reset** every environment receives a horizon drawn from
    ``Uniform(1, max_horizon)`` so they immediately spread across different
    phases of the episode.  On **subsequent resets**, with probability
    ``prob`` a new horizon is sampled from ``Uniform(min_horizon, max_horizon)``;
    otherwise the full ``max_horizon`` is used.

    ``first_episode_prob`` controls the truncation probability for each
    environment's first episode after the initial spread. By default it matches
    ``prob`` so that ``prob=0.0`` disables all subsequent random truncation
    after the initial spread. Setting it higher than ``prob`` can accelerate
    decorrelation when batch sizes are large relative to ``max_horizon``.

    .. note:: This transform must be placed **after** :class:`~torchrl.envs.StepCounter`
        in the transform chain, as it relies on the ``"step_count"`` key.

    Args:
        min_horizon (int): minimum horizon for random truncation
            (inclusive).
        max_horizon (int): maximum horizon for random truncation
            (inclusive). Also used as the full-length horizon when no random
            truncation is applied. This should typically match the
            environment's ``max_episode_steps``, which unfortunately cannot
            be retrieved automatically in general.
        prob (float, optional): probability of sampling a random horizon on
            each subsequent reset. Defaults to ``0.0`` (only the initial
            spread is applied). When nonzero, a low value (e.g. ``0.01``) is
            recommended -- frequent truncation can negatively impact training.
        first_episode_prob (float, optional): truncation probability for each
            environment's first episode after the initial spread. Defaults to
            ``prob`` when omitted.

    Examples:
        >>> from torchrl.envs import GymEnv, TransformedEnv, StepCounter
        >>> base_env = GymEnv("Pendulum-v1")
        >>> env = TransformedEnv(
        ...     base_env,
        ...     Compose(
        ...         StepCounter(),
        ...         RandomTruncationTransform(
        ...             prob=0.1, min_horizon=50, max_horizon=200
        ...         ),
        ...     ),
        ... )
        >>> rollout = env.rollout(300)
        >>> # Episode length will be at most 200 steps
        >>> print(rollout.shape)
        torch.Size([...])
    """

    invertible = False

    def __init__(
        self,
        min_horizon: int,
        max_horizon: int,
        prob: float = 0.0,
        first_episode_prob: float | None = None,
    ):
        super().__init__()
        if first_episode_prob is None:
            first_episode_prob = prob
        if not 0.0 <= prob <= 1.0:
            raise ValueError(f"prob must be in [0, 1], got {prob}")
        if not 0.0 <= first_episode_prob <= 1.0:
            raise ValueError(
                f"first_episode_prob must be in [0, 1], got {first_episode_prob}"
            )
        if min_horizon < 1:
            raise ValueError(f"min_horizon must be >= 1, got {min_horizon}")
        if max_horizon < 1:
            raise ValueError(f"max_horizon must be >= 1, got {max_horizon}")
        if min_horizon > max_horizon:
            raise ValueError(
                f"min_horizon ({min_horizon}) must be <= max_horizon ({max_horizon})"
            )
        self.prob = prob
        self.first_episode_prob = first_episode_prob
        self.min_horizon = min_horizon
        self.max_horizon = max_horizon
        self._horizons: torch.Tensor | None = None
        self._first_episode: torch.Tensor | None = None
        self._initialized = False

    def set_container(self, container: Transform | EnvBase) -> None:
        super().set_container(container)
        self._validate_step_counter_registration()

    def _validate_step_counter_registration(self) -> None:
        parent = self.parent
        if parent is None:
            return
        observation_spec = getattr(parent, "observation_spec", None)
        if observation_spec is None:
            return
        keys = observation_spec.keys(True, True)
        has_step_count = any(
            key == "step_count" or (isinstance(key, tuple) and key[-1] == "step_count")
            for key in keys
        )
        if not has_step_count:
            raise RuntimeError(
                "RandomTruncationTransform requires a StepCounter earlier in the "
                "transform chain. Use:\n"
                "  Compose(StepCounter(), RandomTruncationTransform(...))\n"
                "or add StepCounter() before RandomTruncationTransform in your "
                "transform pipeline."
            )

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        step_count = next_tensordict.get("step_count", None)
        if step_count is None or self._horizons is None:
            return next_tensordict

        should_truncate = step_count >= self._horizons
        truncated = next_tensordict.get("truncated", torch.zeros_like(should_truncate))
        done = next_tensordict.get("done", torch.zeros_like(should_truncate))
        next_tensordict.set("truncated", truncated | should_truncate)
        next_tensordict.set("done", done | should_truncate)
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        step_count = tensordict_reset.get("step_count", None)
        if step_count is None:
            return tensordict_reset

        # Ensure truncated is False after reset
        done = tensordict_reset.get("done", None)
        if done is not None:
            tensordict_reset.set(
                "truncated",
                torch.zeros_like(done),
            )

        if not self._initialized:
            # First reset: uniform spread for immediate decorrelation
            self._horizons = torch.randint(
                1,
                self.max_horizon + 1,
                step_count.shape,
                device=step_count.device,
            )
            self._first_episode = torch.ones(
                step_count.shape, dtype=torch.bool, device=step_count.device
            )
            self._initialized = True
            return tensordict_reset

        # Resample horizons for envs that just reset
        reset_mask = tensordict.get("_reset", None)
        if reset_mask is not None:
            mask = reset_mask.view_as(self._horizons).bool()
            if mask.any():
                if self.prob == 0.0 and self.first_episode_prob == 0.0:
                    self._horizons[mask] = self.max_horizon
                    self._first_episode[mask] = False
                    return tensordict_reset
                n = int(mask.sum())
                new_h = torch.randint(
                    self.min_horizon,
                    self.max_horizon + 1,
                    (n,),
                    device=self._horizons.device,
                )
                # Use first_episode_prob for envs still in their first
                # episode, prob for all subsequent episodes
                first_ep = self._first_episode[mask]
                effective_prob = torch.where(
                    first_ep,
                    torch.tensor(self.first_episode_prob, device=self._horizons.device),
                    torch.tensor(self.prob, device=self._horizons.device),
                )
                keep_full = torch.rand(n, device=self._horizons.device) > effective_prob
                new_h[keep_full] = self.max_horizon
                self._horizons[mask] = new_h.view_as(self._horizons[mask])
                # First episode is over for these envs
                self._first_episode[mask] = False
        return tensordict_reset

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        full_done_spec = self.parent.output_spec["full_done_spec"]
        # Ensure truncated and done keys exist in the spec
        if "truncated" not in full_done_spec.keys():
            done_shape = full_done_spec["done"].shape
            full_done_spec["truncated"] = Categorical(
                2, dtype=torch.bool, device=output_spec.device, shape=done_shape
            )
        output_spec["full_done_spec"] = full_done_spec
        return super().transform_output_spec(output_spec)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            "RandomTruncationTransform cannot be called independently, only its "
            "step and reset methods are functional."
        )


class InitTracker(Transform):
    """Reset tracker.

    This transform populates the step/reset tensordict with a reset tracker entry
    that is set to ``True`` whenever ``reset`` is called.

    Args:
         init_key (NestedKey, optional): the key to be used for the tracker entry.
            In case of multiple _reset flags, this key is used as the leaf replacement for each.

    Examples:
        >>> from torchrl.envs.libs.gym import GymEnv
        >>> env = TransformedEnv(GymEnv("Pendulum-v1"), InitTracker())
        >>> td = env.reset()
        >>> print(td["is_init"])
        tensor(True)
        >>> td = env.rand_step(td)
        >>> print(td["next", "is_init"])
        tensor(False)

    """

    def __init__(self, init_key: str = "is_init"):
        if not isinstance(init_key, str):
            raise ValueError(
                "init_key can only be of type str as it will be the leaf key associated to each reset flag."
            )
        self.init_key = init_key
        super().__init__()

    def set_container(self, container: Transform | EnvBase) -> None:
        self._init_keys = None
        return super().set_container(container)

    @property
    def out_keys(self) -> Sequence[NestedKey]:
        return self.init_keys

    @out_keys.setter
    def out_keys(self, value):
        if value in (None, []):
            return
        raise ValueError(
            "Cannot set non-empty out-keys when out-keys are defined by the init_key value."
        )

    @property
    def init_keys(self) -> Sequence[NestedKey]:
        init_keys = self.__dict__.get("_init_keys", None)
        if init_keys is not None:
            return init_keys
        init_keys = []
        if self.parent is None:
            raise NotImplementedError(
                FORWARD_NOT_IMPLEMENTED.format(self.__class__.__name__)
            )
        for reset_key in self.parent._filtered_reset_keys:
            if isinstance(reset_key, str):
                init_key = self.init_key
            else:
                init_key = unravel_key((reset_key[:-1], self.init_key))
            init_keys.append(init_key)
        self._init_keys = init_keys
        return self._init_keys

    @property
    def reset_keys(self) -> Sequence[NestedKey]:
        return self.parent._filtered_reset_keys

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        for init_key in self.init_keys:
            done_key = _replace_last(init_key, "done")
            if init_key not in next_tensordict.keys(True, True):
                device = next_tensordict.device
                if device is None:
                    device = torch.device("cpu")
                shape = self.parent.full_done_spec[done_key].shape
                next_tensordict.set(
                    init_key,
                    torch.zeros(shape, device=device, dtype=torch.bool),
                )
        return next_tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        device = tensordict.device
        if device is None:
            device = torch.device("cpu")
        for reset_key, init_key in _zip_strict(self.reset_keys, self.init_keys):
            _reset = tensordict.get(reset_key, None)
            if _reset is None:
                done_key = _replace_last(init_key, "done")
                shape = self.parent.full_done_spec[done_key]._safe_shape
                tensordict_reset.set(
                    init_key,
                    torch.ones(
                        shape,
                        device=device,
                        dtype=torch.bool,
                    ),
                )
            else:
                init_val = _reset.clone()
                parent_td = (
                    tensordict_reset
                    if isinstance(init_key, str)
                    else tensordict_reset.get(init_key[:-1])
                )
                if init_val.ndim == parent_td.ndim:
                    # unsqueeze, to match the done shape
                    init_val = init_val.unsqueeze(-1)
                tensordict_reset.set(init_key, init_val)
        return tensordict_reset

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        full_done_spec = self.parent.output_spec["full_done_spec"]
        for init_key in self.init_keys:
            for done_key in self.parent.done_keys:
                # check root
                if type(done_key) is not type(init_key):
                    continue
                if isinstance(done_key, tuple):
                    if done_key[:-1] == init_key[:-1]:
                        shape = full_done_spec[done_key].shape
                        break
                if isinstance(done_key, str):
                    shape = full_done_spec[done_key].shape
                    break
            else:
                raise KeyError(
                    f"Could not find root of init_key {init_key} within done_keys {self.parent.done_keys}."
                )
            observation_spec[init_key] = Categorical(
                2,
                dtype=torch.bool,
                device=self.parent.device,
                shape=shape,
            )
        return observation_spec

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise NotImplementedError(
            FORWARD_NOT_IMPLEMENTED.format(self.__class__.__name__)
        )


class BurnInTransform(Transform):
    """Transform to partially burn-in data sequences.

    This transform is useful to obtain up-to-date recurrent states when
    they are not available. It burns-in a number of steps along the time dimension
    from sampled sequential data slices and returns the remaining data sequence with
    the burnt-in data in its initial time step. This transform is intended to be used as a
    replay buffer transform, not as an environment transform.

    Args:
        modules (sequence of TensorDictModule): A list of modules used to burn-in data sequences.
        burn_in (int): The number of time steps to burn in.
        out_keys (sequence of NestedKey, optional): destination keys. Defaults to
        all the modules `out_keys` that point to the next time step (e.g. `"hidden"` if `
        ("next", "hidden")` is part of the `out_keys` of a module).

    .. note::
        This transform expects as inputs TensorDicts with its last dimension being the
        time dimension. It also assumes that all provided modules can process
        sequential data.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs.transforms import BurnInTransform
        >>> from torchrl.modules import GRUModule
        >>> gru_module = GRUModule(
        ...     input_size=10,
        ...     hidden_size=10,
        ...     in_keys=["observation", "hidden"],
        ...     out_keys=["intermediate", ("next", "hidden")],
        ...     default_recurrent_mode=True,
        ... )
        >>> burn_in_transform = BurnInTransform(
        ...     modules=[gru_module],
        ...     burn_in=5,
        ... )
        >>> td = TensorDict({
        ...     "observation": torch.randn(2, 10, 10),
        ...      "hidden": torch.randn(2, 10, gru_module.gru.num_layers, 10),
        ...      "is_init": torch.zeros(2, 10, 1),
        ... }, batch_size=[2, 10])
        >>> td = burn_in_transform(td)
        >>> td.shape
        torch.Size([2, 5])
        >>> td.get("hidden").abs().sum()
        tensor(86.3008)

        >>> from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
        >>> buffer = TensorDictReplayBuffer(
        ...     storage=LazyMemmapStorage(2),
        ...     batch_size=1,
        ... )
        >>> buffer.append_transform(burn_in_transform)
        >>> td = TensorDict({
        ...     "observation": torch.randn(2, 10, 10),
        ...      "hidden": torch.randn(2, 10, gru_module.gru.num_layers, 10),
        ...      "is_init": torch.zeros(2, 10, 1),
        ... }, batch_size=[2, 10])
        >>> buffer.extend(td)
        >>> td = buffer.sample(1)
        >>> td.shape
        torch.Size([1, 5])
        >>> td.get("hidden").abs().sum()
        tensor(37.0344)
    """

    invertible = False

    def __init__(
        self,
        modules: Sequence[TensorDictModuleBase],
        burn_in: int,
        out_keys: Sequence[NestedKey] | None = None,
    ):
        if not isinstance(modules, Sequence):
            modules = [modules]

        for module in modules:
            if not isinstance(module, TensorDictModuleBase):
                raise ValueError(
                    f"All modules must be TensorDictModules, but a {type(module)} was provided."
                )

        in_keys = set()
        for module in modules:
            in_keys.update(module.in_keys)

        if out_keys is None:
            out_keys = set()
            for module in modules:
                for key in module.out_keys:
                    if key[0] == "next":
                        out_keys.add(key[1])
        else:
            out_keys_ = set()
            for key in out_keys:
                if isinstance(key, tuple) and key[0] == "next":
                    key = key[1]
                    warnings.warn(
                        f"The 'next' key is not needed in the BurnInTransform `out_key` {key} and "
                        f"will be ignored. This transform already assumes that `out_keys` will be "
                        f"retrieved from the next time step of the burnt-in data."
                    )
                out_keys_.add(key)
            out_keys = out_keys_

        super().__init__(in_keys=in_keys, out_keys=out_keys)
        self.modules = modules
        self.burn_in = burn_in

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError("BurnInTransform can only be appended to a ReplayBuffer")

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        raise RuntimeError("BurnInTransform can only be appended to a ReplayBuffer.")

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:

        if self.burn_in == 0:
            return tensordict

        td_device = tensordict.device
        B, T, *extra_dims = tensordict.batch_size

        # Split the tensor dict into burn-in data and the rest.
        td_burn_in = tensordict[..., : self.burn_in]
        td_out = tensordict[..., self.burn_in :]

        # Burn in the recurrent state.
        with torch.no_grad():
            for module in self.modules:
                module_device = next(module.parameters()).device or None
                td_burn_in = td_burn_in.to(module_device)
                td_burn_in = module(td_burn_in)
        td_burn_in = td_burn_in.to(td_device)

        # Update out TensorDict with the burnt-in data.
        for out_key in self.out_keys:
            if out_key not in td_out.keys(include_nested=True):
                td_out.set(
                    out_key,
                    torch.zeros(
                        B, T - self.burn_in, *tensordict.get(out_key).shape[2:]
                    ),
                )
            td_out[..., 0][out_key].copy_(td_burn_in["next"][..., -1][out_key])

        return td_out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(burn_in={self.burn_in}, in_keys={self.in_keys}, out_keys={self.out_keys})"


class BatchSizeTransform(Transform):
    """A transform to modify the batch-size of an environment.

    This transform has two distinct usages: it can be used to set the
    batch-size for non-batch-locked (e.g. stateless) environments to
    enable data collection using data collectors. It can also be used
    to modify the batch-size of an environment (e.g. squeeze, unsqueeze or
    reshape).

    This transform modifies the environment batch-size to match the one provided.
    It expects the parent environment batch-size to be expandable to the
    provided one.

    Keyword Args:
        batch_size (torch.Size or equivalent, optional): the new batch-size of the environment.
            Exclusive with ``reshape_fn``.
        reshape_fn (callable, optional): a callable to modify the environment batch-size.
            Exclusive with ``batch_size``.

            .. note:: Currently, transformations involving
                ``reshape``, ``flatten``, ``unflatten``, ``squeeze`` and ``unsqueeze``
                are supported. If another reshape operation is required, please submit
                a feature request on TorchRL github.

        reset_func (callable, optional): a function that produces a reset tensordict.
            The signature must match ``Callable[[TensorDictBase, TensorDictBase], TensorDictBase]``
            where the first input argument is the optional tensordict passed to the
            environment during the call to :meth:`~EnvBase.reset` and the second
            is the output of ``TransformedEnv.base_env.reset``. It can also support an
            optional ``env`` keyword argument if ``env_kwarg=True``.
        env_kwarg (bool, optional): if ``True``, ``reset_func`` must support a
            ``env`` keyword argument. Defaults to ``False``. The env passed will
            be the env accompanied by its transform.

    Example:
        >>> # Changing the batch-size with a function
        >>> from torchrl.envs import GymEnv
        >>> base_env = GymEnv("CartPole-v1")
        >>> env = TransformedEnv(base_env, BatchSizeTransform(reshape_fn=lambda data: data.reshape(1, 1)))
        >>> env.rollout(4)
        >>> # Setting the shape of a stateless environment
        >>> class MyEnv(EnvBase):
        ...     batch_locked = False
        ...     def __init__(self):
        ...         super().__init__()
        ...         self.observation_spec = Composite(observation=Unbounded(3))
        ...         self.reward_spec = Unbounded(1)
        ...         self.action_spec = Unbounded(1)
        ...
        ...     def _reset(self, tensordict: TensorDictBase, **kwargs) -> TensorDictBase:
        ...         tensordict_batch_size = tensordict.batch_size if tensordict is not None else torch.Size([])
        ...         result = self.observation_spec.rand(tensordict_batch_size)
        ...         result.update(self.full_done_spec.zero(tensordict_batch_size))
        ...         return result
        ...
        ...     def _step(
        ...         self,
        ...         tensordict: TensorDictBase,
        ...     ) -> TensorDictBase:
        ...         result = self.observation_spec.rand(tensordict.batch_size)
        ...         result.update(self.full_done_spec.zero(tensordict.batch_size))
        ...         result.update(self.full_reward_spec.zero(tensordict.batch_size))
        ...         return result
        ...
        ...     def _set_seed(self, seed: Optional[int]) -> None:
        ...         pass
        ...
        >>> env = TransformedEnv(MyEnv(), BatchSizeTransform([5]))
        >>> assert env.batch_size == torch.Size([5])
        >>> assert env.rollout(10).shape == torch.Size([5, 10])

    The ``reset_func`` can create a tensordict with the desired batch-size, allowing for
    a fine-grained reset call:

        >>> def reset_func(tensordict, tensordict_reset, env):
        ...     result = env.observation_spec.rand()
        ...     result.update(env.full_done_spec.zero())
        ...     assert result.batch_size != torch.Size([])
        ...     return result
        >>> env = TransformedEnv(MyEnv(), BatchSizeTransform([5], reset_func=reset_func, env_kwarg=True))
        >>> print(env.rollout(2))
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([5, 2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([5, 2]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([5, 2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([5, 2]),
            device=None,
            is_shared=False)

    This transform can be used to deploy non-batch-locked environments within data
    collectors:

        >>> from torchrl.collectors import Collector
        >>> collector = Collector(env, lambda td: env.rand_action(td), frames_per_batch=10, total_frames=-1)
        >>> for data in collector:
        ...     print(data)
        ...     break
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                collector: TensorDict(
                    fields={
                        traj_ids: Tensor(shape=torch.Size([5, 2]), device=cpu, dtype=torch.int64, is_shared=False)},
                    batch_size=torch.Size([5, 2]),
                    device=None,
                    is_shared=False),
                done: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([5, 2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([5, 2]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([5, 2, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                terminated: Tensor(shape=torch.Size([5, 2, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([5, 2]),
            device=None,
            is_shared=False)
        >>> collector.shutdown()
    """

    _ENV_ERR = "BatchSizeTransform.{} requires a parent env."

    def __init__(
        self,
        *,
        batch_size: torch.Size | None = None,
        reshape_fn: Callable[[TensorDictBase], TensorDictBase] | None = None,
        reset_func: Callable[[TensorDictBase, TensorDictBase], TensorDictBase]
        | None = None,
        env_kwarg: bool = False,
    ):
        super().__init__()
        if not ((batch_size is None) ^ (reshape_fn is None)):
            raise ValueError(
                "One and only one of batch_size OR reshape_fn must be provided."
            )
        if batch_size is not None:
            self.batch_size = torch.Size(batch_size)
            self.reshape_fn = None
        else:
            self.reshape_fn = reshape_fn
            self.batch_size = None
        self.reshape_fn = reshape_fn
        self.reset_func = reset_func
        self.env_kwarg = env_kwarg

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        if self.reset_func is not None:
            if self.env_kwarg:
                tensordict_reset = self.reset_func(
                    tensordict, tensordict_reset, env=self.container
                )
            else:
                tensordict_reset = self.reset_func(tensordict, tensordict_reset)
        if self.batch_size is not None:
            return tensordict_reset.expand(self.batch_size)
        return self.reshape_fn(tensordict_reset)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if self.reshape_fn is not None:
            next_tensordict = self.reshape_fn(next_tensordict)
        return next_tensordict

    forward = _call

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.reshape_fn is not None:
            parent = self.parent
            if parent is not None:
                parent_batch_size = parent.batch_size
                tensordict = tensordict.reshape(parent_batch_size)
        return tensordict

    def transform_env_batch_size(self, batch_size: torch.Size):
        if self.batch_size is not None:
            return self.batch_size
        return self.reshape_fn(torch.zeros(batch_size, device="meta")).shape

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        if self.batch_size is not None:
            return output_spec.expand(self.batch_size)
        return self.reshape_fn(output_spec)

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        if self.batch_size is not None:
            return input_spec.expand(self.batch_size)
        return self.reshape_fn(input_spec)


class AutoResetTransform(Transform):
    """A transform for auto-resetting environments.

    This transform can be appended to any auto-resetting environment, or automatically
    appended using ``env = SomeEnvClass(..., auto_reset=True)``. If the transform is explicitly
    appended to an env, a :class:`~torchrl.envs.transforms.AutoResetEnv` must be used.

    An auto-reset environment must have the following properties (differences from this
    description should be accounted for by subclassing this class):

      - the reset function can be called once at the beginning (after instantiation) with
        or without effect. Whether calls to `reset` are allowed after that is up to the
        environment itself.
      - During a rollout, any ``done`` state will result in a reset and produce an observation
        that isn't the last observation of the current episode, but the first observation
        of the next episode (this transform will extract and cache this observation
        and fill the obs with some arbitrary value).

    Keyword Args:
        replace (bool, optional): if ``False``, values are just placed as they are in the
            ``"next"`` entry even if they are not valid. Defaults to ``True``. A value of
            ``False`` overrides any subsequent filling keyword argument.
            This argument can also be passed with the constructor method by passing a
            ``auto_reset_replace`` argument: ``env = FooEnv(..., auto_reset=True, auto_reset_replace=False)``.
        fill_float (:obj:`float` or str, optional): The filling value for floating point tensors
            that terminate an episode. A value of ``None`` means no replacement (values are just
            placed as they are in the ``"next"`` entry even if they are not valid).
        fill_int (int, optional): The filling value for signed integer tensors
            that terminate an episode.  A value of ``None`` means no replacement (values are just
            placed as they are in the ``"next"`` entry even if they are not valid).
        fill_bool (bool, optional): The filling value for boolean tensors
            that terminate an episode.  A value of ``None`` means no replacement (values are just
            placed as they are in the ``"next"`` entry even if they are not valid).

    Arguments are only available when the transform is explicitly instantiated (not through `EnvType(..., auto_reset=True)`).

    Examples:
        >>> from torchrl.envs import GymEnv
        >>> from torchrl.envs import set_gym_backend
        >>> import torch
        >>> torch.manual_seed(0)
        >>>
        >>> class AutoResettingGymEnv(GymEnv):
        ...     def _step(self, tensordict):
        ...         tensordict = super()._step(tensordict)
        ...         if tensordict["done"].any():
        ...             td_reset = super().reset()
        ...             tensordict.update(td_reset.exclude(*self.done_keys))
        ...         return tensordict
        ...
        ...     def _reset(self, tensordict=None):
        ...         if tensordict is not None and "_reset" in tensordict:
        ...             return tensordict.copy()
        ...         return super()._reset(tensordict)
        >>>
        >>> with set_gym_backend("gym"):
        ...     env = AutoResettingGymEnv("CartPole-v1", auto_reset=True, auto_reset_replace=True)
        ...     env.set_seed(0)
        ...     r = env.rollout(30, break_when_any_done=False)
        >>> print(r["next", "done"].squeeze())
        tensor([False, False, False, False, False, False, False, False, False, False,
                False, False, False,  True, False, False, False, False, False, False,
                False, False, False, False, False,  True, False, False, False, False])
        >>> print("observation after reset are set as nan", r["next", "observation"])
        observation after reset are set as nan tensor([[-4.3633e-02, -1.4877e-01,  1.2849e-02,  2.7584e-01],
                [-4.6609e-02,  4.6166e-02,  1.8366e-02, -1.2761e-02],
                [-4.5685e-02,  2.4102e-01,  1.8111e-02, -2.9959e-01],
                [-4.0865e-02,  4.5644e-02,  1.2119e-02, -1.2542e-03],
                [-3.9952e-02,  2.4059e-01,  1.2094e-02, -2.9009e-01],
                [-3.5140e-02,  4.3554e-01,  6.2920e-03, -5.7893e-01],
                [-2.6429e-02,  6.3057e-01, -5.2867e-03, -8.6963e-01],
                [-1.3818e-02,  8.2576e-01, -2.2679e-02, -1.1640e+00],
                [ 2.6972e-03,  1.0212e+00, -4.5959e-02, -1.4637e+00],
                [ 2.3121e-02,  1.2168e+00, -7.5232e-02, -1.7704e+00],
                [ 4.7457e-02,  1.4127e+00, -1.1064e-01, -2.0854e+00],
                [ 7.5712e-02,  1.2189e+00, -1.5235e-01, -1.8289e+00],
                [ 1.0009e-01,  1.0257e+00, -1.8893e-01, -1.5872e+00],
                [        nan,         nan,         nan,         nan],
                [-3.9405e-02, -1.7766e-01, -1.0403e-02,  3.0626e-01],
                [-4.2959e-02, -3.7263e-01, -4.2775e-03,  5.9564e-01],
                [-5.0411e-02, -5.6769e-01,  7.6354e-03,  8.8698e-01],
                [-6.1765e-02, -7.6292e-01,  2.5375e-02,  1.1820e+00],
                [-7.7023e-02, -9.5836e-01,  4.9016e-02,  1.4826e+00],
                [-9.6191e-02, -7.6387e-01,  7.8667e-02,  1.2056e+00],
                [-1.1147e-01, -9.5991e-01,  1.0278e-01,  1.5219e+00],
                [-1.3067e-01, -7.6617e-01,  1.3322e-01,  1.2629e+00],
                [-1.4599e-01, -5.7298e-01,  1.5848e-01,  1.0148e+00],
                [-1.5745e-01, -7.6982e-01,  1.7877e-01,  1.3527e+00],
                [-1.7285e-01, -9.6668e-01,  2.0583e-01,  1.6956e+00],
                [        nan,         nan,         nan,         nan],
                [-4.3962e-02,  1.9845e-01, -4.5015e-02, -2.5903e-01],
                [-3.9993e-02,  3.9418e-01, -5.0196e-02, -5.6557e-01],
                [-3.2109e-02,  5.8997e-01, -6.1507e-02, -8.7363e-01],
                [-2.0310e-02,  3.9574e-01, -7.8980e-02, -6.0090e-01]])

    """

    def __init__(
        self,
        *,
        replace: bool | None = None,
        fill_float="nan",
        fill_int=-1,
        fill_bool=False,
    ):
        super().__init__()
        if replace is False:
            fill_float = fill_int = fill_bool = None
        if fill_float == "nan":
            fill_float = float("nan")
        self.fill_float = fill_float
        self.fill_int = fill_int
        self.fill_bool = fill_bool
        self._validated = False

    def _validate_container(self):
        if self._validated:
            return
        if type(self.container) is not AutoResetEnv:
            raise RuntimeError(
                f"The {self.__class__.__name__} container must be of type AutoResetEnv."
            )
        self._validated = True

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        self._validate_container()
        return self._replace_auto_reset_vals(tensordict_reset=tensordict_reset)

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        return self._correct_auto_reset_vals(next_tensordict)

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError

    @property
    def _simple_done(self):
        return self.parent._simple_done

    def _correct_auto_reset_vals(self, tensordict):
        # we need to move the data from tensordict to tensordict_
        def replace_and_set(key, val, mask, saved_td_autoreset, agent=tensordict):
            saved_td_autoreset.set(key, val)
            if val.dtype.is_floating_point:
                if self.fill_float is None:
                    val_set_nan = val.clone()
                else:
                    val_set_nan = torch.where(
                        expand_as_right(mask, val),
                        torch.full_like(val, self.fill_float),
                        val,
                    )
            elif val.dtype.is_signed:
                if self.fill_int is None:
                    val_set_nan = val.clone()
                else:
                    val_set_nan = torch.where(
                        expand_as_right(mask, val),
                        torch.full_like(val, self.fill_int),
                        val,
                    )
            else:
                if self.fill_bool is None:
                    val_set_nan = val.clone()
                else:
                    val_set_nan = torch.where(
                        expand_as_right(mask, val),
                        torch.full_like(val, self.fill_bool),
                        val,
                    )
            agent.set(key, val_set_nan)

        if self._simple_done:
            done = tensordict.get("done")
            if done.any():
                mask = done.squeeze(-1)
                self._saved_td_autorest = TensorDict()
                for key in self.parent.full_observation_spec.keys(True, True):
                    val = tensordict.get(key)
                    replace_and_set(
                        key, val, mask, saved_td_autoreset=self._saved_td_autorest
                    )
        else:
            parents = []
            # Go through each "done" key and get the corresponding agent.
            _saved_td_autorest = None
            obs_keys = list(self.parent.full_observation_spec.keys(True, True))
            for done_key in self.parent.done_keys:
                if _ends_with(done_key, "done"):
                    if isinstance(done_key, str):
                        raise TypeError(
                            "A 'done' key was a string but a tuple was expected."
                        )
                    agent_key = done_key[:-1]
                    done = tensordict.get(done_key)
                    mask = done.squeeze(-1)
                    if done.any():
                        if _saved_td_autorest is None:
                            _saved_td_autorest = TensorDict()
                        agent = tensordict.get(agent_key)
                        if isinstance(agent, LazyStackedTensorDict):
                            agents = agent.tensordicts
                            masks = mask.unbind(agent.stack_dim)
                            saved_td_autorest_agent = LazyStackedTensorDict(
                                *[td.empty() for td in agents],
                                stack_dim=agent.stack_dim,
                            )
                            saved_td_autorest_agents = (
                                saved_td_autorest_agent.tensordicts
                            )
                        else:
                            agents = [agent]
                            masks = [mask]
                            saved_td_autorest_agent = _saved_td_autorest.setdefault(
                                agent_key, agent.empty()
                            )
                            saved_td_autorest_agents = [saved_td_autorest_agent]
                        for key in obs_keys:
                            if (
                                isinstance(key, tuple)
                                and key[: len(agent_key)] == agent_key
                            ):
                                for _agent, _mask, _saved_td_autorest_agent in zip(
                                    agents, masks, saved_td_autorest_agents
                                ):
                                    val = _agent.get(key[len(agent_key) :])
                                    replace_and_set(
                                        key[len(agent_key) :],
                                        val,
                                        _mask,
                                        saved_td_autoreset=_saved_td_autorest_agent,
                                        agent=_agent,
                                    )
                    parents.append(done_key[:-1])
            if _saved_td_autorest is not None:
                self.__dict__["_saved_td_autorest"] = _saved_td_autorest

        return tensordict

    def _replace_auto_reset_vals(self, *, tensordict_reset):
        _saved_td_autorest = self.__dict__.get("_saved_td_autorest", None)
        if _saved_td_autorest is None:
            return tensordict_reset
        if self._simple_done:
            for key, val in self._saved_td_autorest.items(True, True):
                if _ends_with(key, "_reset"):
                    continue
                val_set_reg = val
                tensordict_reset.set(key, val_set_reg)
        else:
            for done_key in self.parent.done_keys:
                if _ends_with(done_key, "done"):
                    agent_key = done_key[:-1]
                    mask = self._saved_td_autorest.pop(
                        _replace_last(done_key, "__mask__"), None
                    )
                    if mask is not None:
                        agent = self._saved_td_autorest.get(agent_key)

                        if isinstance(agent, LazyStackedTensorDict):
                            agents = agent.tensordicts
                            masks = mask.unbind(agent.stack_dim)
                            dests = tensordict_reset.setdefault(
                                agent_key,
                                LazyStackedTensorDict(
                                    *[td.empty() for td in agents],
                                    stack_dim=agent.stack_dim,
                                ),
                            )
                        else:
                            agents = [agent]
                            masks = [mask]
                            dests = [
                                tensordict_reset.setdefault(agent_key, agent.empty())
                            ]
                        for _agent, _mask, _dest in zip(agents, masks, dests):
                            for key, val in _agent.items(True, True):
                                if _ends_with(key, "_reset"):
                                    continue
                                if not _mask.all():
                                    val_not_reset = _dest.get(key)
                                    val_set_reg = torch.where(
                                        expand_as_right(mask, val), val, val_not_reset
                                    )
                                else:
                                    val_set_reg = val
                                _dest.set(key, val_set_reg)
        delattr(self, "_saved_td_autorest")
        return tensordict_reset


class TrajCounter(Transform):
    """Global trajectory counter transform.

    TrajCounter can be used to count the number of trajectories (i.e., the number of times `reset` is called) in any
    TorchRL environment.
    This transform will work within a single node across multiple processes (see note below).
    A single transform can only count the trajectories associated with a single done state, but nested done states are
    accepted as long as their prefix matches the prefix of the counter key.

    Args:
        out_key (NestedKey, optional): The entry name of the trajectory counter. Defaults to ``"traj_count"``.

    Examples:
        >>> from torchrl.envs import GymEnv, StepCounter, TrajCounter
        >>> env = GymEnv("Pendulum-v1").append_transform(StepCounter(6))
        >>> env = env.append_transform(TrajCounter())
        >>> r = env.rollout(18, break_when_any_done=False)  # 18 // 6 = 3 trajectories
        >>> r["next", "traj_count"]
        tensor([[0],
                [0],
                [0],
                [0],
                [0],
                [0],
                [1],
                [1],
                [1],
                [1],
                [1],
                [1],
                [2],
                [2],
                [2],
                [2],
                [2],
                [2]])

    .. note::
        Sharing a trajectory counter among workers can be done in multiple ways, but it will usually involve wrapping the environment in a :class:`~torchrl.envs.EnvCreator`. Not doing so may result in an error during serialization of the transform. The counter will be shared among the workers, meaning that at any point in time, it is guaranteed that there will not be two environments that will share the same trajectory count (and each (step-count, traj-count) pair will be unique).
        Here are examples of valid ways of sharing a ``TrajCounter`` object between processes:

            >>> # Option 1: Create the trajectory counter outside the environment.
            >>> #  This requires the counter to be cloned within the transformed env, as a single transform object cannot have two parents.
            >>> t = TrajCounter()
            >>> def make_env(max_steps=4, t=t):
            ...     # See CountingEnv in torchrl.test.mocking_classes
            ...     env = TransformedEnv(CountingEnv(max_steps=max_steps), t.clone())
            ...     env.transform.transform_observation_spec(env.base_env.observation_spec)
            ...     return env
            >>> penv = ParallelEnv(
            ...     2,
            ...     [EnvCreator(make_env, max_steps=4), EnvCreator(make_env, max_steps=5)],
            ...     mp_start_method="spawn",
            ... )
            >>> # Option 2: Create the transform within the constructor.
            >>> #  In this scenario, we still need to tell each sub-env what kwarg has to be used.
            >>> #  Both EnvCreator and ParallelEnv offer that possibility.
            >>> def make_env(max_steps=4):
            ...     t = TrajCounter()
            ...     env = TransformedEnv(CountingEnv(max_steps=max_steps), t)
            ...     env.transform.transform_observation_spec(env.base_env.observation_spec)
            ...     return env
            >>> make_env_c0 = EnvCreator(make_env)
            >>> # Create a variant of the env with different kwargs
            >>> make_env_c1 = make_env_c0.make_variant(max_steps=5)
            >>> penv = ParallelEnv(
            ...     2,
            ...     [make_env_c0, make_env_c1],
            ...     mp_start_method="spawn",
            ... )
            >>> # Alternatively, pass the kwargs to the ParallelEnv
            >>> penv = ParallelEnv(
            ...     2,
            ...     [make_env_c0, make_env_c0],
            ...     create_env_kwargs=[{"max_steps": 5}, {"max_steps": 4}],
            ...     mp_start_method="spawn",
            ... )

    """

    def __init__(
        self, out_key: NestedKey = "traj_count", *, repeats: int | None = None
    ):
        super().__init__(in_keys=[], out_keys=[out_key])
        self._make_shared_value()
        self._initialized = False
        if repeats is None:
            repeats = 0
        self.repeats = repeats

    def _make_shared_value(self):
        self._traj_count = mp.Value("i", 0)

    def __getstate__(self):
        state = super().__getstate__()
        state["_traj_count"] = None
        return state

    def clone(self) -> Self:
        clone = super().clone()
        # All clones share the same _traj_count and lock
        clone._traj_count = self._traj_count
        return clone

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        if not self._initialized:
            self._initialized = True
        rk = self.parent.reset_keys
        traj_count_key = self.out_keys[0]
        is_str = isinstance(traj_count_key, str)
        for _rk in rk:
            if is_str and isinstance(_rk, str):
                rk = _rk
                break
            elif (
                not is_str
                and isinstance(_rk, tuple)
                and _rk[:-1] == traj_count_key[:-1]
            ):
                rk = _rk
                break
        else:
            raise RuntimeError(
                f"Did not find reset key that matched the prefix of the traj counter key. Reset keys: {rk}, traj count: {traj_count_key}"
            )
        reset = None
        if tensordict is not None:
            reset = tensordict.get(rk, default=None)
        if reset is None:
            reset = torch.ones(
                self.container.observation_spec[self.out_keys[0]].shape,
                device=tensordict_reset.device,
                dtype=torch.bool,
            )
        with (self._traj_count):
            tc = int(self._traj_count.value)
            self._traj_count.value = self._traj_count.value + reset.sum().item()
            episodes = torch.arange(tc, tc + reset.sum(), device=self.parent.device)
            episodes = torch.masked_scatter(
                torch.zeros_like(reset, dtype=episodes.dtype), reset, episodes
            )
            tensordict_reset.set(traj_count_key, episodes)
        return tensordict_reset

    def _step(
        self, tensordict: TensorDictBase, next_tensordict: TensorDictBase
    ) -> TensorDictBase:
        if not self._initialized:
            raise RuntimeError("_step was called before _reset was called.")
        next_tensordict.set(self.out_keys[0], tensordict.get(self.out_keys[0]))
        return next_tensordict

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError(
            f"{type(self).__name__} can only be called within an environment step or reset."
        )

    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        raise RuntimeError(
            f"{type(self).__name__} can only be called within an environment step or reset."
        )

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        return {
            "traj_count": int(self._traj_count.value),
        }

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False
    ):
        self._traj_count.value *= 0
        self._traj_count.value += state_dict["traj_count"]

    def transform_observation_spec(self, observation_spec: Composite) -> Composite:
        if not isinstance(observation_spec, Composite):
            raise ValueError(
                f"observation_spec was expected to be of type Composite. Got {type(observation_spec)} instead."
            )
        full_done_spec = self.parent.output_spec["full_done_spec"]
        traj_count_key = self.out_keys[0]
        # find a matching done key (there might be more than one)
        for done_key in self.parent.done_keys:
            # check root
            if type(done_key) is not type(traj_count_key):
                continue
            if isinstance(done_key, tuple):
                if done_key[:-1] == traj_count_key[:-1]:
                    shape = full_done_spec[done_key].shape
                    break
            if isinstance(done_key, str):
                shape = full_done_spec[done_key].shape
                break

        else:
            raise KeyError(
                f"Could not find root of traj_count key {traj_count_key} in done keys {self.done_keys}."
            )
        observation_spec[traj_count_key] = Bounded(
            shape=shape,
            dtype=torch.int64,
            device=observation_spec.device,
            low=0,
            high=torch.iinfo(torch.int64).max,
        )
        return super().transform_observation_spec(observation_spec)
