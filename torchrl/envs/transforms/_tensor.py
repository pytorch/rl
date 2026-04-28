# Copyright (c) Meta Plobs_dictnc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from copy import copy
from typing import Any, TYPE_CHECKING

import torch

from tensordict import NonTensorData, NonTensorStack, TensorDictBase
from tensordict.nn import dispatch
from tensordict.utils import _zip_strict, NestedKey
from torch import Tensor

from torchrl._utils import _replace_last

from torchrl.data.tensor_specs import Bounded, Composite, TensorSpec, Unbounded
from torchrl.envs.transforms.utils import _set_missing_tolerance
from torchrl.envs.utils import _sort_keys, make_composite_from_td

if TYPE_CHECKING:
    import transformers

if TYPE_CHECKING:
    from typing import Self
else:
    Self = Any

from torchrl.envs.transforms._base import Transform

__all__ = [
    "CatTensors",
    "Hash",
    "Stack",
    "Tokenizer",
    "UnaryTransform",
]


class CatTensors(Transform):
    """Concatenates several keys in a single tensor.

    This is especially useful if multiple keys describe a single state (e.g.
    "observation_position" and
    "observation_velocity")

    Args:
        in_keys (sequence of NestedKey): keys to be concatenated. If `None` (or not provided)
            the keys will be retrieved from the parent environment the first time
            the transform is used. This behavior will only work if a parent is set.
        out_key (NestedKey): key of the resulting tensor.
        dim (int, optional): dimension along which the concatenation will occur.
            Default is ``-1``.

    Keyword Args:
        del_keys (bool, optional): if ``True``, the input values will be deleted after
            concatenation. Default is ``True``.
        unsqueeze_if_oor (bool, optional): if ``True``, CatTensor will check that
            the indicated dimension exists for the tensors to concatenate. If not,
            the tensors will be unsqueezed along that dimension.
            Default is ``False``.
        sort (bool, optional): if ``True``, the keys will be sorted in the
            transform. Otherwise, the order provided by the user will prevail.
            Defaults to ``True``.

    Examples:
        >>> transform = CatTensors(in_keys=["key1", "key2"])
        >>> td = TensorDict({"key1": torch.zeros(1, 1),
        ...     "key2": torch.ones(1, 1)}, [1])
        >>> _ = transform(td)
        >>> print(td.get("observation_vector"))
        tensor([[0., 1.]])
        >>> transform = CatTensors(in_keys=["key1", "key2"], dim=-2, unsqueeze_if_oor=True)
        >>> td = TensorDict({"key1": torch.zeros(1),
        ...     "key2": torch.ones(1)}, [])
        >>> _ = transform(td)
        >>> print(td.get("observation_vector").shape)
        torch.Size([2, 1])

    """

    invertible = False

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_key: NestedKey = "observation_vector",
        dim: int = -1,
        *,
        del_keys: bool = True,
        unsqueeze_if_oor: bool = False,
        sort: bool = True,
    ):
        self._initialized = in_keys is not None
        if not self._initialized:
            if dim != -1:
                raise ValueError(
                    "Lazy call to CatTensors is only supported when `dim=-1`."
                )
        elif sort:
            in_keys = sorted(in_keys, key=_sort_keys)
        if not isinstance(out_key, (str, tuple)):
            raise Exception("CatTensors requires out_key to be of type NestedKey")
        super().__init__(in_keys=in_keys, out_keys=[out_key])
        self.dim = dim
        self._del_keys = del_keys
        self._keys_to_exclude = None
        self.unsqueeze_if_oor = unsqueeze_if_oor

    @property
    def keys_to_exclude(self) -> list[NestedKey]:
        if self._keys_to_exclude is None:
            self._keys_to_exclude = [
                key for key in self.in_keys if key != self.out_keys[0]
            ]
        return self._keys_to_exclude

    def _find_in_keys(self):
        """Gathers all the entries from observation spec which shape is 1d."""
        parent = self.parent
        obs_spec = parent.observation_spec
        in_keys = []
        for key, value in obs_spec.items(True, True):
            if len(value.shape) == 1:
                in_keys.append(key)
        return sorted(in_keys, key=_sort_keys)

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        if not self._initialized:
            self.in_keys = self._find_in_keys()
            self._initialized = True

        values = [next_tensordict.get(key, None) for key in self.in_keys]
        if any(value is None for value in values):
            raise Exception(
                f"CatTensor failed, as it expected input keys ="
                f" {sorted(self.in_keys, key=_sort_keys)} but got a TensorDict with keys"
                f" {sorted(next_tensordict.keys(include_nested=True), key=_sort_keys)}"
            )
        if self.unsqueeze_if_oor:
            pos_idx = self.dim > 0
            abs_idx = self.dim if pos_idx else -self.dim - 1
            values = [
                v
                if abs_idx < v.ndimension()
                else v.unsqueeze(0)
                if not pos_idx
                else v.unsqueeze(-1)
                for v in values
            ]

        out_tensor = torch.cat(values, dim=self.dim)
        next_tensordict.set(self.out_keys[0], out_tensor)
        if self._del_keys:
            next_tensordict.exclude(*self.keys_to_exclude, inplace=True)
        return next_tensordict

    forward = _call

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        if not self._initialized:
            self.in_keys = self._find_in_keys()
            self._initialized = True

        # check that all keys are in observation_spec
        if len(self.in_keys) > 1 and not isinstance(observation_spec, Composite):
            raise ValueError(
                "CatTensor cannot infer the output observation spec as there are multiple input keys but "
                "only one observation_spec."
            )

        if isinstance(observation_spec, Composite) and len(
            [key for key in self.in_keys if key not in observation_spec.keys(True)]
        ):
            raise ValueError(
                "CatTensor got a list of keys that does not match the keys in observation_spec. "
                "Make sure the environment has an observation_spec attribute that includes all the specs needed for CatTensor."
            )

        if not isinstance(observation_spec, Composite):
            # by def, there must be only one key
            return observation_spec

        keys = [key for key in observation_spec.keys(True, True) if key in self.in_keys]

        sum_shape = sum(
            [
                observation_spec[key].shape[self.dim]
                if observation_spec[key].shape
                else 1
                for key in keys
            ]
        )
        spec0 = observation_spec[keys[0]]
        out_key = self.out_keys[0]
        shape = list(spec0.shape)
        device = spec0.device
        shape[self.dim] = sum_shape
        shape = torch.Size(shape)
        observation_spec[out_key] = Unbounded(
            shape=shape,
            dtype=spec0.dtype,
            device=device,
        )
        if self._del_keys:
            for key in self.keys_to_exclude:
                if key in observation_spec.keys(True):
                    del observation_spec[key]
        return observation_spec

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(in_keys={self.in_keys}, out_key"
            f"={self.out_keys[0]})"
        )


class UnaryTransform(Transform):
    r"""Applies a unary operation on the specified inputs.

    Args:
        in_keys (sequence of NestedKey): the keys of inputs to the unary operation.
        out_keys (sequence of NestedKey): the keys of the outputs of the unary operation.
        in_keys_inv (sequence of NestedKey, optional): the keys of inputs to the unary operation during inverse call.
        out_keys_inv (sequence of NestedKey, optional): the keys of the outputs of the unary operation during inverse call.

    Keyword Args:
        fn (Callable[[Any], Tensor | TensorDictBase]): the function to use as the unary operation. If it accepts
            a non-tensor input, it must also accept ``None``.
        inv_fn (Callable[[Any], Any], optional): the function to use as the unary operation during inverse calls.
            If it accepts a non-tensor input, it must also accept ``None``.
            Can be omitted, in which case :attr:`fn` will be used for inverse maps.
        use_raw_nontensor (bool, optional): if ``False``, data is extracted from
            :class:`~tensordict.NonTensorData`/:class:`~tensordict.NonTensorStack` inputs before ``fn`` is called
            on them. If ``True``, the raw :class:`~tensordict.NonTensorData`/:class:`~tensordict.NonTensorStack`
            inputs are given directly to ``fn``, which must support those
            inputs. Default is ``False``.

    Example:
        >>> from torchrl.envs import GymEnv, UnaryTransform
        >>> env = GymEnv("Pendulum-v1")
        >>> env = env.append_transform(
        ...     UnaryTransform(
        ...         in_keys=["observation"],
        ...         out_keys=["observation_trsf"],
        ...             fn=lambda tensor: str(tensor.numpy().tobytes())))
        >>> env.observation_spec
        Composite(
            observation: BoundedContinuous(
                shape=torch.Size([3]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous),
            observation_trsf: NonTensor(
                shape=torch.Size([]),
                space=None,
                device=cpu,
                dtype=None,
                domain=None),
            device=None,
            shape=torch.Size([]))
        >>> env.rollout(3)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        observation_trsf: NonTensorStack(
                            ["b'\\xbe\\xbc\\x7f?8\\x859=/\\x81\\xbe;'", "b'\\x...,
                            batch_size=torch.Size([3]),
                            device=None),
                        reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                observation_trsf: NonTensorStack(
                    ["b'\\x9a\\xbd\\x7f?\\xb8T8=8.c>'", "b'\\xbe\\xbc\...,
                    batch_size=torch.Size([3]),
                    device=None),
                terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> env.check_env_specs()
        [torchrl][INFO] check_env_specs succeeded!

    """
    enable_inv_on_reset = True

    def __init__(
        self,
        in_keys: Sequence[NestedKey],
        out_keys: Sequence[NestedKey],
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
        *,
        fn: Callable[[Any], Tensor | TensorDictBase],
        inv_fn: Callable[[Any], Any] | None = None,
        use_raw_nontensor: bool = False,
    ):
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
        )
        self._fn = fn
        self._inv_fn = inv_fn
        self._use_raw_nontensor = use_raw_nontensor

    def _apply_transform(self, value):
        if not self._use_raw_nontensor:
            if isinstance(value, NonTensorData):
                if value.dim() == 0:
                    value = value.get("data")
                else:
                    value = value.tolist()
            elif isinstance(value, NonTensorStack):
                value = value.tolist()
        return self._fn(value)

    def _inv_apply_transform(self, state: torch.Tensor) -> torch.Tensor:
        if not self._use_raw_nontensor:
            if isinstance(state, NonTensorData):
                if state.dim() == 0:
                    state = state.get("data")
                else:
                    state = state.tolist()
            elif isinstance(state, NonTensorStack):
                state = state.tolist()
        if self._inv_fn is not None:
            return self._inv_fn(state)
        return self._fn(state)

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        input_spec = input_spec.clone()

        # Make a generic input from the spec, call the transform with that
        # input, and then generate the output spec from the output.
        zero_input_ = input_spec.zero()
        test_input = zero_input_["full_action_spec"].update(
            zero_input_["full_state_spec"]
        )
        # We use forward and not inv because the spec comes from the base env and
        # we are trying to infer what the spec looks like from the outside.
        for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
            data = test_input.get(in_key, None)
            if data is not None:
                data = self._apply_transform(data)
                test_input.set(out_key, data)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {test_input}")
        test_output = test_input
        # test_output = self.inv(test_input)
        test_input_spec = make_composite_from_td(
            test_output, unsqueeze_null_shapes=False
        )

        input_spec["full_action_spec"] = self.transform_action_spec(
            input_spec["full_action_spec"],
            test_input_spec,
        )
        if "full_state_spec" in input_spec.keys():
            input_spec["full_state_spec"] = self.transform_state_spec(
                input_spec["full_state_spec"],
                test_input_spec,
            )
        return input_spec

    def transform_output_spec(self, output_spec: Composite) -> Composite:
        output_spec = output_spec.clone()

        # Make a generic input from the spec, call the transform with that
        # input, and then generate the output spec from the output.
        zero_input_ = output_spec.zero()
        test_input = (
            zero_input_["full_observation_spec"]
            .update(zero_input_["full_reward_spec"])
            .update(zero_input_["full_done_spec"])
        )
        test_output = self.forward(test_input)
        test_output_spec = make_composite_from_td(
            test_output, unsqueeze_null_shapes=False
        )

        output_spec["full_observation_spec"] = self.transform_observation_spec(
            output_spec["full_observation_spec"],
            test_output_spec,
        )
        if "full_reward_spec" in output_spec.keys():
            output_spec["full_reward_spec"] = self.transform_reward_spec(
                output_spec["full_reward_spec"],
                test_output_spec,
            )
        if "full_done_spec" in output_spec.keys():
            output_spec["full_done_spec"] = self.transform_done_spec(
                output_spec["full_done_spec"],
                test_output_spec,
            )
        return output_spec

    def _transform_spec(
        self, spec: TensorSpec, test_output_spec: TensorSpec, inverse: bool = False
    ) -> TensorSpec:
        if not isinstance(spec, Composite):
            raise TypeError(f"{self}: Only specs of type Composite can be transformed")

        spec_keys = set(spec.keys(include_nested=True))

        iterator = (
            zip(self.in_keys, self.out_keys)
            if not inverse
            else zip(self.in_keys_inv, self.out_keys_inv)
        )
        for in_key, out_key in iterator:
            if in_key in spec_keys:
                spec.set(out_key, test_output_spec[out_key])
        return spec

    def transform_observation_spec(
        self, observation_spec: TensorSpec, test_output_spec: TensorSpec
    ) -> TensorSpec:
        return self._transform_spec(observation_spec, test_output_spec)

    def transform_reward_spec(
        self, reward_spec: TensorSpec, test_output_spec: TensorSpec
    ) -> TensorSpec:
        return self._transform_spec(reward_spec, test_output_spec)

    def transform_done_spec(
        self, done_spec: TensorSpec, test_output_spec: TensorSpec
    ) -> TensorSpec:
        return self._transform_spec(done_spec, test_output_spec)

    def transform_action_spec(
        self, action_spec: TensorSpec, test_input_spec: TensorSpec
    ) -> TensorSpec:
        return self._transform_spec(action_spec, test_input_spec, inverse=True)

    def transform_state_spec(
        self, state_spec: TensorSpec, test_input_spec: TensorSpec
    ) -> TensorSpec:
        return self._transform_spec(state_spec, test_input_spec, inverse=True)


class Hash(UnaryTransform):
    r"""Adds a hash value to a tensordict.

    Args:
        in_keys (sequence of NestedKey): the keys of the values to hash.
        out_keys (sequence of NestedKey): the keys of the resulting hashes.
        in_keys_inv (sequence of NestedKey, optional): the keys of the values to hash during inv call.
        out_keys_inv (sequence of NestedKey, optional): the keys of the resulting hashes during inv call.

    Keyword Args:
        hash_fn (Callable, optional): the hash function to use. The function
            signature must be
            ``(input: Any, seed: Any | None) -> torch.Tensor``.
            ``seed`` is only used if this transform is initialized with the
            ``seed`` argument.  Default is ``Hash.reproducible_hash``.
        seed (optional): seed to use for the hash function, if it requires one.
        use_raw_nontensor (bool, optional): if ``False``, data is extracted from
            :class:`~tensordict.NonTensorData`/:class:`~tensordict.NonTensorStack` inputs before ``fn`` is called
            on them. If ``True``, the raw :class:`~tensordict.NonTensorData`/:class:`~tensordict.NonTensorStack`
            inputs are given directly to ``fn``, which must support those
            inputs. Default is ``False``.
        repertoire (Dict[Tuple[int], Any], optional): If given, this dict stores
            the inverse mappings from hashes to inputs. This repertoire isn't
            copied, so it can be modified in the same workspace after the
            transform instantiation and these modifications will be reflected in
            the map. Missing hashes will be mapped to ``None``. Default: ``None``

    Examples:
        >>> from torchrl.envs import GymEnv, UnaryTransform, Hash
        >>> env = GymEnv("Pendulum-v1")
        >>> # Add a string output
        >>> env = env.append_transform(
        ...     UnaryTransform(
        ...         in_keys=["observation"],
        ...         out_keys=["observation_str"],
        ...             fn=lambda tensor: str(tensor.numpy().tobytes())))
        >>> # process the string output
        >>> env = env.append_transform(
        ...     Hash(
        ...         in_keys=["observation_str"],
        ...         out_keys=["observation_hash"],)
        ... )
        >>> env.observation_spec
        Composite(
            observation: BoundedContinuous(
                shape=torch.Size([3]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True),
                    high=Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, contiguous=True)),
                device=cpu,
                dtype=torch.float32,
                domain=continuous),
            observation_str: NonTensor(
                shape=torch.Size([]),
                space=None,
                device=cpu,
                dtype=None,
                domain=None),
            observation_hash: UnboundedDiscrete(
                shape=torch.Size([32]),
                space=ContinuousBox(
                    low=Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.uint8, contiguous=True),
                    high=Tensor(shape=torch.Size([32]), device=cpu, dtype=torch.uint8, contiguous=True)),
                device=cpu,
                dtype=torch.uint8,
                domain=discrete),
            device=None,
            shape=torch.Size([]))
        >>> env.rollout(3)
        TensorDict(
            fields={
                action: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                next: TensorDict(
                    fields={
                        done: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                        observation_hash: Tensor(shape=torch.Size([3, 32]), device=cpu, dtype=torch.uint8, is_shared=False),
                        observation_str: NonTensorStack(
                            ["b'g\\x08\\x8b\\xbexav\\xbf\\x00\\xee(>'", "b'\\x...,
                            batch_size=torch.Size([3]),
                            device=None),
                        reward: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.float32, is_shared=False),
                        terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                        truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
                    batch_size=torch.Size([3]),
                    device=None,
                    is_shared=False),
                observation: Tensor(shape=torch.Size([3, 3]), device=cpu, dtype=torch.float32, is_shared=False),
                observation_hash: Tensor(shape=torch.Size([3, 32]), device=cpu, dtype=torch.uint8, is_shared=False),
                observation_str: NonTensorStack(
                    ["b'\\xb5\\x17\\x8f\\xbe\\x88\\xccu\\xbf\\xc0Vr?'"...,
                    batch_size=torch.Size([3]),
                    device=None),
                terminated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False),
                truncated: Tensor(shape=torch.Size([3, 1]), device=cpu, dtype=torch.bool, is_shared=False)},
            batch_size=torch.Size([3]),
            device=None,
            is_shared=False)
        >>> env.check_env_specs()
        [torchrl][INFO] check_env_specs succeeded!
    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey],
        out_keys: Sequence[NestedKey],
        in_keys_inv: Sequence[NestedKey] = None,
        out_keys_inv: Sequence[NestedKey] = None,
        *,
        hash_fn: Callable = None,
        seed: Any | None = None,
        use_raw_nontensor: bool = False,
        repertoire: tuple[tuple[int], Any] = None,
    ):
        if hash_fn is None:
            hash_fn = Hash.reproducible_hash

        if repertoire is None and in_keys_inv is not None and len(in_keys_inv) > 0:
            self._repertoire = {}
        else:
            self._repertoire = repertoire

        self._seed = seed
        self._hash_fn = hash_fn
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
            fn=self.call_hash_fn,
            inv_fn=self.get_input_from_hash,
            use_raw_nontensor=use_raw_nontensor,
        )

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        return {"_repertoire": self._repertoire}

    @classmethod
    def hash_to_repertoire_key(cls, hash_tensor):
        if isinstance(hash_tensor, torch.Tensor):
            if hash_tensor.dim() == 0:
                return hash_tensor.tolist()
            return tuple(cls.hash_to_repertoire_key(t) for t in hash_tensor.tolist())
        elif isinstance(hash_tensor, list):
            return tuple(cls.hash_to_repertoire_key(t) for t in hash_tensor)
        else:
            return hash_tensor

    def get_input_from_hash(self, hash_tensor) -> Any:
        """Look up the input that was given for a particular hash output.

        This feature is only available if, during initialization, either the
        ``repertoire`` argument was given or both the ``in_keys_inv`` and
        ``out_keys_inv`` arguments were given.

        Args:
            hash_tensor (Tensor): The hash output.

        Returns:
            Any: The input that the hash was generated from.
        """
        if self._repertoire is None:
            raise RuntimeError(
                "An inverse transform was queried but the repertoire is None."
            )
        return self._repertoire[self.hash_to_repertoire_key(hash_tensor)]

    def call_hash_fn(self, value):
        if self._seed is None:
            hash_tensor = self._hash_fn(value)
        else:
            hash_tensor = self._hash_fn(value, self._seed)
        if not torch.is_tensor(hash_tensor):
            raise ValueError(
                f"Hash function must return a tensor, but got {type(hash_tensor)}"
            )
        if self._repertoire is not None:
            self._repertoire[self.hash_to_repertoire_key(hash_tensor)] = copy(value)
        return hash_tensor

    @classmethod
    def reproducible_hash(cls, string, seed=None):
        """Creates a reproducible 256-bit hash from a string using a seed.

        Args:
            string (str or None): The input string. If ``None``, null string ``""`` is used.
            seed (str, optional): The seed value. Default is ``None``.

        Returns:
            Tensor: Shape ``(32,)`` with dtype ``torch.uint8``.
        """
        if string is None:
            string = ""

        # Prepend the seed to the string
        if seed is not None:
            seeded_string = seed + string
        else:
            seeded_string = str(string)

        # Create a new SHA-256 hash object
        hash_object = hashlib.sha256()

        # Update the hash object with the seeded string
        hash_object.update(seeded_string.encode("utf-8"))

        # Get the hash value as bytes
        hash_bytes = bytearray(hash_object.digest())

        return torch.frombuffer(hash_bytes, dtype=torch.uint8)


class Tokenizer(UnaryTransform):
    r"""Applies a tokenization operation on the specified inputs.

    Args:
        in_keys (sequence of NestedKey): the keys of inputs to the tokenization operation.
        out_keys (sequence of NestedKey): the keys of the outputs of the tokenization operation.
        in_keys_inv (sequence of NestedKey, optional): the keys of inputs to the tokenization operation during inverse call.
        out_keys_inv (sequence of NestedKey, optional): the keys of the outputs of the tokenization operation during inverse call.

    Keyword Args:
        tokenizer (transformers.PretrainedTokenizerBase or str, optional): the tokenizer to use. If ``None``,
            "bert-base-uncased" will be used by default. If a string is provided, it should be the name of a
            pre-trained tokenizer.
        use_raw_nontensor (bool, optional): if ``False``, data is extracted from
            :class:`~tensordict.NonTensorData`/:class:`~tensordict.NonTensorStack` inputs before the tokenization
            function is called on them. If ``True``, the raw :class:`~tensordict.NonTensorData`/:class:`~tensordict.NonTensorStack`
            inputs are given directly to the tokenization function, which must support those inputs. Default is ``False``.
        additional_tokens (List[str], optional): list of additional tokens to add to the tokenizer's vocabulary.

    .. note:: This transform can be used both to transform output strings into tokens and to transform back tokenized
        actions or states into strings. If the environment has a string state-spec, the transformed version will have
        a tokenized state-spec. If it is a string action spec, it will result in a tokenized action spec.

    """

    def __init__(
        self,
        in_keys: Sequence[NestedKey] | None = None,
        out_keys: Sequence[NestedKey] | None = None,
        in_keys_inv: Sequence[NestedKey] | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
        *,
        tokenizer: transformers.PretrainedTokenizerBase = None,  # noqa: F821
        use_raw_nontensor: bool = False,
        additional_tokens: list[str] | None = None,
        skip_special_tokens: bool = True,
        add_special_tokens: bool = False,
        padding: bool = True,
        max_length: int | None = None,
        return_attention_mask: bool = True,
        missing_tolerance: bool = True,
        call_before_reset: bool = False,
    ):
        if tokenizer is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained("google-bert/bert-base-uncased")
        elif isinstance(tokenizer, str):
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(tokenizer)

        self.tokenizer = tokenizer
        self.add_special_tokens = add_special_tokens
        self.skip_special_tokens = skip_special_tokens
        self.padding = padding
        self.max_length = max_length
        self.return_attention_mask = return_attention_mask
        self.call_before_reset = call_before_reset
        if additional_tokens:
            self.tokenizer.add_tokens(additional_tokens)
        super().__init__(
            in_keys=in_keys,
            out_keys=out_keys,
            in_keys_inv=in_keys_inv,
            out_keys_inv=out_keys_inv,
            fn=self.call_tokenizer_fn,
            inv_fn=self.call_tokenizer_inv_fn,
            use_raw_nontensor=use_raw_nontensor,
        )
        self._missing_tolerance = missing_tolerance

    @property
    def device(self) -> torch.device | None:
        if "_device" in self.__dict__:
            return self._device
        parent = self.parent
        if parent is None:
            return None
        device = parent.device
        self._device = device
        return device

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        # Specialized for attention mask
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            value = next_tensordict.get(in_key, default=None)
            if value is not None:
                observation = self._apply_transform(value)
                if self.return_attention_mask:
                    observation, attention_mask = observation
                    next_tensordict.set(
                        _replace_last(out_key, "attention_mask"),
                        attention_mask,
                    )
                next_tensordict.set(
                    out_key,
                    observation,
                )
            elif (
                self.missing_tolerance
                and self.return_attention_mask
                and out_key in next_tensordict.keys(True)
            ):
                attention_key = _replace_last(out_key, "attention_mask")
                if attention_key not in next_tensordict:
                    next_tensordict[attention_key] = torch.ones_like(
                        next_tensordict.get(out_key)
                    )
            elif not self.missing_tolerance:
                raise KeyError(
                    f"{self}: '{in_key}' not found in tensordict {next_tensordict}"
                )
        return next_tensordict

    @dispatch(source="in_keys", dest="out_keys")
    def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            data = tensordict.get(in_key, None)
            if data is not None:
                data = self._apply_transform(data)
                if self.return_attention_mask:
                    data, attention_mask = data
                    tensordict.set(
                        _replace_last(out_key, "attention_mask"),
                        attention_mask,
                    )
                tensordict.set(out_key, data)
            elif not self.missing_tolerance:
                raise KeyError(f"'{in_key}' not found in tensordict {tensordict}")
        return tensordict

    def _reset_env_preprocess(self, tensordict: TensorDictBase) -> TensorDictBase:
        if self.call_before_reset:
            with _set_missing_tolerance(self, True):
                tensordict = self._call(tensordict)
        return tensordict

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        if self.call_before_reset:
            return tensordict_reset
        return super()._reset(tensordict, tensordict_reset)

    def call_tokenizer_fn(self, value: str | list[str]):
        device = self.device
        kwargs = {"add_special_tokens": self.add_special_tokens}
        if self.max_length is not None:
            kwargs["padding"] = "max_length"
            kwargs["max_length"] = self.max_length
        if isinstance(value, str):
            out = self.tokenizer.encode(value, return_tensors="pt", **kwargs)[0]
            # TODO: incorporate attention mask
            if self.return_attention_mask:
                attention_mask = torch.ones_like(out, dtype=torch.int64)
        else:
            kwargs["padding"] = (
                self.padding if self.max_length is None else "max_length"
            )
            kwargs["return_attention_mask"] = self.return_attention_mask
            # kwargs["return_token_type_ids"] = False
            out = self.tokenizer(value, return_tensors="pt", **kwargs)
            if self.return_attention_mask:
                attention_mask = out["attention_mask"]
            out = out["input_ids"]

        if device is not None and out.device != device:
            out = out.to(device)
            if self.return_attention_mask:
                attention_mask = attention_mask.to(device)
        if self.return_attention_mask:
            return out, attention_mask
        return out

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        # Override _inv_call to account for ragged dims
        if not self.in_keys_inv:
            return tensordict
        for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
            data = tensordict.get(out_key, None, as_padded_tensor=True)
            if data is not None:
                item = self._inv_apply_transform(data)
                tensordict.set(in_key, item)
            elif not self.missing_tolerance:
                raise KeyError(f"'{out_key}' not found in tensordict {tensordict}")
        return tensordict

    def call_tokenizer_inv_fn(self, value: Tensor):
        if value.ndim == 1:
            out = self.tokenizer.decode(
                value.int(), skip_special_tokens=self.skip_special_tokens
            )
        else:
            out = self.tokenizer.batch_decode(
                value.int(), skip_special_tokens=self.skip_special_tokens
            )
        device = self._str_device
        if isinstance(out, list):
            result = NonTensorStack(*out)
            if device:
                result = result.to(device)
            return result
        return NonTensorData(out, device=device)

    @property
    def _str_device(self):
        parent = self.parent
        if parent is None:
            return None
        if self.in_keys:
            in_key = self.in_keys[0]
        elif self.in_keys_inv:
            in_key = self.in_keys_inv[0]
        else:
            return None
        if in_key in parent.observation_keys:
            return parent.full_observation_spec[in_key].device
        if in_key in parent.action_keys:
            return parent.full_action_spec[in_key].device
        if in_key in parent.state_keys:
            return parent.full_state_spec[in_key].device
        return None

    def transform_input_spec(self, input_spec: Composite) -> Composite:
        # We need to cap the spec to generate valid random strings
        for in_key, out_key in _zip_strict(self.in_keys_inv, self.out_keys_inv):
            if in_key in input_spec["full_state_spec"].keys(True, True):
                spec = input_spec["full_state_spec"]
            elif in_key in input_spec["full_action_spec"].keys(False, True):
                spec = input_spec["full_action_spec"]
            else:
                raise KeyError(
                    f"The input keys {in_key} wasn't found in the env input specs."
                )
            local_spec = spec.pop(in_key)
            local_dtype = local_spec.dtype
            if local_dtype is None or local_dtype.is_floating_point:
                local_dtype = torch.int64
            new_shape = spec.shape
            if self.max_length is None:
                # Then we can't tell what the shape will be
                new_shape = new_shape + torch.Size((-1,))
            else:
                new_shape = new_shape + torch.Size((self.max_length,))
            spec[out_key] = Bounded(
                0,
                self.tokenizer.vocab_size,
                shape=new_shape,
                device=local_spec.device,
                dtype=local_dtype,
            )
        return input_spec

    transform_output_spec = Transform.transform_output_spec
    transform_reward_spec = Transform.transform_reward_spec
    transform_done_spec = Transform.transform_done_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        attention_mask_keys = set()
        for in_key, out_key in _zip_strict(self.in_keys, self.out_keys):
            new_shape = observation_spec.shape + torch.Size((-1,))
            try:
                in_spec = observation_spec[in_key]
                obs_dtype = in_spec.dtype
                device = in_spec.device
            except KeyError:
                # In some cases (eg, the tokenizer is applied during reset on data that
                #  originates from a dataloader) we don't have an in_spec
                in_spec = None
                obs_dtype = None
                device = observation_spec.device
            if obs_dtype is None or obs_dtype.is_floating_point:
                obs_dtype = torch.int64
            observation_spec[out_key] = Bounded(
                0,
                self.tokenizer.vocab_size,
                shape=new_shape,
                device=device,
                dtype=obs_dtype,
            )
            if self.return_attention_mask:
                attention_mask_key = _replace_last(out_key, "attention_mask")
                if attention_mask_key in attention_mask_keys:
                    raise KeyError(
                        "Conflicting attention_mask keys. Make sure the token tensors are "
                        "nested at different places in the tensordict such that `(*root, 'attention_mask')` "
                        "entries are unique."
                    )
                attention_mask_keys.add(attention_mask_key)
                attention_dtype = obs_dtype
                if attention_dtype is None or attention_dtype.is_floating_point:
                    attention_dtype = torch.int64
                observation_spec[attention_mask_key] = Bounded(
                    0,
                    2,
                    shape=new_shape,
                    device=device,
                    dtype=attention_dtype,
                )
        return observation_spec


class Stack(Transform):
    """Stacks tensors and tensordicts.

    Concatenates a sequence of tensors or tensordicts along a new dimension.
    The tensordicts or tensors under ``in_keys`` must all have the same shapes.

    This transform only stacks the inputs into one output key. Stacking multiple
    groups of input keys into different output keys requires multiple
    transforms.

    This transform can be useful for environments that have multiple agents with
    identical specs under different keys. The specs and tensordicts for the
    agents can be stacked together under a shared key, in order to run MARL
    algorithms that expect the tensors for observations, rewards, etc. to
    contain batched data for all the agents.

    Args:
        in_keys (sequence of NestedKey): keys to be stacked.
        out_key (NestedKey): key of the resulting stacked entry.
        in_key_inv (NestedKey, optional): key to unstack during ``inv``
            calls. Default is ``None``.
        out_keys_inv (sequence of NestedKey, optional): keys of the resulting
            unstacked entries after ``inv`` calls. Default is ``None``.
        dim (int, optional): dimension to insert. Default is ``-1``.
        allow_positive_dim (bool, optional): if ``True``, positive dimensions
            are accepted.  Defaults to ``False``, ie. non-negative dimensions are
            not permitted.

    Keyword Args:
        del_keys (bool, optional): if ``True``, the input values will be deleted
            after stacking. Default is ``True``.

    Examples:
        >>> import torch
        >>> from tensordict import TensorDict
        >>> from torchrl.envs import Stack
        >>> td = TensorDict({"key1": torch.zeros(3), "key2": torch.ones(3)}, [])
        >>> td
        TensorDict(
            fields={
                key1: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False),
                key2: Tensor(shape=torch.Size([3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> transform = Stack(in_keys=["key1", "key2"], out_key="out", dim=-2)
        >>> transform(td)
        TensorDict(
            fields={
                out: Tensor(shape=torch.Size([2, 3]), device=cpu, dtype=torch.float32, is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
        >>> td["out"]
        tensor([[0., 0., 0.],
                [1., 1., 1.]])

        >>> agent_0 = TensorDict({"obs": torch.rand(4, 5), "reward": torch.zeros(1)})
        >>> agent_1 = TensorDict({"obs": torch.rand(4, 5), "reward": torch.zeros(1)})
        >>> td = TensorDict({"agent_0": agent_0, "agent_1": agent_1})
        >>> transform = Stack(in_keys=["agent_0", "agent_1"], out_key="agents")
        >>> transform(td)
        TensorDict(
            fields={
                agents: TensorDict(
                    fields={
                        obs: Tensor(shape=torch.Size([2, 4, 5]), device=cpu, dtype=torch.float32, is_shared=False),
                        reward: Tensor(shape=torch.Size([2, 1]), device=cpu, dtype=torch.float32, is_shared=False)},
                    batch_size=torch.Size([2]),
                    device=None,
                    is_shared=False)},
            batch_size=torch.Size([]),
            device=None,
            is_shared=False)
    """

    invertible = True

    def __init__(
        self,
        in_keys: Sequence[NestedKey],
        out_key: NestedKey,
        in_key_inv: NestedKey | None = None,
        out_keys_inv: Sequence[NestedKey] | None = None,
        dim: int = -1,
        allow_positive_dim: bool = False,
        *,
        del_keys: bool = True,
    ):
        if not allow_positive_dim and dim >= 0:
            raise ValueError(
                "dim should be negative to accommodate for envs of different "
                "batch_sizes. If you need dim to be positive, set "
                "allow_positive_dim=True."
            )

        if in_key_inv is None and out_keys_inv is not None:
            raise ValueError("out_keys_inv was specified, but in_key_inv was not")
        elif in_key_inv is not None and out_keys_inv is None:
            raise ValueError("in_key_inv was specified, but out_keys_inv was not")

        super().__init__(
            in_keys=in_keys,
            out_keys=[out_key],
            in_keys_inv=None if in_key_inv is None else [in_key_inv],
            out_keys_inv=out_keys_inv,
        )

        for in_key in self.in_keys:
            if len(in_key) == len(self.out_keys[0]):
                if all(k1 == k2 for k1, k2 in zip(in_key, self.out_keys[0])):
                    raise ValueError(f"{self}: out_key cannot be in in_keys")
        parent_keys = []
        for key in self.in_keys:
            if isinstance(key, (list, tuple)):
                for parent_level in range(1, len(key)):
                    parent_key = tuple(key[:-parent_level])
                    if parent_key not in parent_keys:
                        parent_keys.append(parent_key)
        self._maybe_del_parent_keys = sorted(parent_keys, key=len, reverse=True)
        self.dim = dim
        self._del_keys = del_keys
        self._keys_to_exclude = None

    def _call(self, next_tensordict: TensorDictBase) -> TensorDictBase:
        values = []
        for in_key in self.in_keys:
            value = next_tensordict.get(in_key, default=None)
            if value is not None:
                values.append(value)
            elif not self.missing_tolerance:
                raise KeyError(
                    f"{self}: '{in_key}' not found in tensordict {next_tensordict}"
                )

        out_tensor = torch.stack(values, dim=self.dim)
        next_tensordict.set(self.out_keys[0], out_tensor)
        if self._del_keys:
            next_tensordict.exclude(*self.in_keys, inplace=True)
            for parent_key in self._maybe_del_parent_keys:
                if len(next_tensordict[parent_key].keys()) == 0:
                    next_tensordict.exclude(parent_key, inplace=True)
        return next_tensordict

    forward = _call

    def _inv_call(self, tensordict: TensorDictBase) -> TensorDictBase:
        if len(self.in_keys_inv) == 0:
            return tensordict

        if self.in_keys_inv[0] not in tensordict.keys(include_nested=True):
            return tensordict
        values = torch.unbind(tensordict[self.in_keys_inv[0]], dim=self.dim)
        for value, out_key_inv in _zip_strict(values, self.out_keys_inv):
            tensordict = tensordict.set(out_key_inv, value)
        return tensordict.exclude(self.in_keys_inv[0])

    def _reset(
        self, tensordict: TensorDictBase, tensordict_reset: TensorDictBase
    ) -> TensorDictBase:
        with _set_missing_tolerance(self, True):
            tensordict_reset = self._call(tensordict_reset)
        return tensordict_reset

    def _transform_spec(self, spec: TensorSpec) -> TensorSpec:
        if not isinstance(spec, Composite):
            raise TypeError(f"{self}: Only specs of type Composite can be transformed")

        spec_keys = spec.keys(include_nested=True)
        keys_to_stack = [key for key in spec_keys if key in self.in_keys]
        specs_to_stack = [spec[key] for key in keys_to_stack]

        if len(specs_to_stack) == 0:
            return spec

        stacked_specs = torch.stack(specs_to_stack, dim=self.dim)
        spec.set(self.out_keys[0], stacked_specs)

        if self._del_keys:
            for key in keys_to_stack:
                del spec[key]
            for parent_key in self._maybe_del_parent_keys:
                if len(spec[parent_key]) == 0:
                    del spec[parent_key]

        return spec

    def transform_input_spec(self, input_spec: TensorSpec) -> TensorSpec:
        self._transform_spec(input_spec["full_state_spec"])
        self._transform_spec(input_spec["full_action_spec"])
        return input_spec

    def transform_observation_spec(self, observation_spec: TensorSpec) -> TensorSpec:
        return self._transform_spec(observation_spec)

    def transform_reward_spec(self, reward_spec: TensorSpec) -> TensorSpec:
        return self._transform_spec(reward_spec)

    def transform_done_spec(self, done_spec: TensorSpec) -> TensorSpec:
        return self._transform_spec(done_spec)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_keys={self.in_keys}, "
            f"out_key={self.out_keys[0]}, "
            f"dim={self.dim}"
            ")"
        )
