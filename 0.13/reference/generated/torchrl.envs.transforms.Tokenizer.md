# Tokenizer

*class*torchrl.envs.transforms.Tokenizer(*in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *in_keys_inv: Sequence[NestedKey] | None = None*, *out_keys_inv: Sequence[NestedKey] | None = None*, ***, *tokenizer: transformers.PretrainedTokenizerBase = None*, *use_raw_nontensor: bool = False*, *additional_tokens: list[str] | None = None*, *skip_special_tokens: bool = True*, *add_special_tokens: bool = False*, *padding: bool = True*, *max_length: int | None = None*, *return_attention_mask: bool = True*, *missing_tolerance: bool = True*, *call_before_reset: bool = False*)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#Tokenizer)

Applies a tokenization operation on the specified inputs.

Parameters:

- **in_keys** (*sequence**of**NestedKey*) - the keys of inputs to the tokenization operation.
- **out_keys** (*sequence**of**NestedKey*) - the keys of the outputs of the tokenization operation.
- **in_keys_inv** (*sequence**of**NestedKey**,**optional*) - the keys of inputs to the tokenization operation during inverse call.
- **out_keys_inv** (*sequence**of**NestedKey**,**optional*) - the keys of the outputs of the tokenization operation during inverse call.

Keyword Arguments:

- **tokenizer** (*transformers.PretrainedTokenizerBase**or**str**,**optional*) - the tokenizer to use. If `None`,
"bert-base-uncased" will be used by default. If a string is provided, it should be the name of a
pre-trained tokenizer.
- **use_raw_nontensor** (*bool**,**optional*) - if `False`, data is extracted from
[`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData)/[`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack) inputs before the tokenization
function is called on them. If `True`, the raw [`NonTensorData`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorData.html#tensordict.NonTensorData)/[`NonTensorStack`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.NonTensorStack.html#tensordict.NonTensorStack)
inputs are given directly to the tokenization function, which must support those inputs. Default is `False`.
- **additional_tokens** (*List**[**str**]**,**optional*) - list of additional tokens to add to the tokenizer's vocabulary.

Note

This transform can be used both to transform output strings into tokens and to transform back tokenized
actions or states into strings. If the environment has a string state-spec, the transformed version will have
a tokenized state-spec. If it is a string action spec, it will result in a tokenized action spec.

forward(*tensordict: [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase) = None*) → [TensorDictBase](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDictBase.html#tensordict.TensorDictBase)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#Tokenizer.forward)

Reads the input tensordict, and for the selected keys, applies the transform.

By default, this method:

- calls directly `_apply_transform()`.
- does not call `_step()` or `_call()`.

This method is not called within env.step at any point. However, is is called within
[`sample()`](torchrl.data.ReplayBuffer.html#torchrl.data.ReplayBuffer.sample).

Note

`forward` also works with regular keyword arguments using [`dispatch`](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.nn.dispatch.html#tensordict.nn.dispatch) to cast the args
names to the keys.

Examples

```
>>> class TransformThatMeasuresBytes(Transform):
... '''Measures the number of bytes in the tensordict, and writes it under `"bytes"`.'''
... def __init__(self):
... super().__init__(in_keys=[], out_keys=["bytes"])
...
... def forward(self, tensordict: TensorDictBase) -> TensorDictBase:
... bytes_in_td = tensordict.bytes()
... tensordict["bytes"] = bytes
... return tensordict
>>> t = TransformThatMeasuresBytes()
>>> env = env.append_transform(t) # works within envs
>>> t(TensorDict(a=0)) # Works offline too.
```

transform_done_spec(*done_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)

Transforms the done spec such that the resulting spec matches transform mapping.

Parameters:

**done_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_input_spec(*input_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#Tokenizer.transform_input_spec)

Transforms the input spec such that the resulting spec matches transform mapping.

Parameters:

**input_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_tensor.html#Tokenizer.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform

transform_output_spec(*output_spec: [Composite](torchrl.data.Composite.html#torchrl.data.Composite)*) → [Composite](torchrl.data.Composite.html#torchrl.data.Composite)

Transforms the output spec such that the resulting spec matches transform mapping.

This method should generally be left untouched. Changes should be implemented using
`transform_observation_spec()`, `transform_reward_spec()` and `transform_full_done_spec()`.
:param output_spec: spec before the transform
:type output_spec: TensorSpec

Returns:

expected spec after the transform

transform_reward_spec(*reward_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)

Transforms the reward spec such that the resulting spec matches transform mapping.

Parameters:

**reward_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform