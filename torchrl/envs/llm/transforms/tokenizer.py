# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import annotations

from typing import Sequence

import torch
from tensordict import NonTensorData, NonTensorStack, TensorDictBase
from tensordict.nn import dispatch
from tensordict.utils import _zip_strict, NestedKey
from torch import Tensor
from torchrl._utils import _replace_last
from torchrl.data.tensor_specs import Bounded, Composite, TensorSpec
from torchrl.envs import Transform, UnaryTransform
from torchrl.envs.transforms.utils import _set_missing_tolerance


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
    def device(self):
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
            out = self.tokenizer.batch_encode_plus(value, return_tensors="pt", **kwargs)
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
