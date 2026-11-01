# ToTensorImage

*class*torchrl.envs.transforms.ToTensorImage(*from_int: bool | None = None*, *unsqueeze: bool = False*, *dtype: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | None = None*, ***, *in_keys: Sequence[NestedKey] | None = None*, *out_keys: Sequence[NestedKey] | None = None*, *shape_tolerant: bool = False*)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#ToTensorImage)

Transforms a numpy-like image (W x H x C) to a pytorch image (C x W x H).

Transforms an observation image from a (... x W x H x C) tensor to a
(... x C x W x H) tensor. Optionally, scales the input tensor from the range
[0, 255] to the range [0.0, 1.0] (see `from_int` for more details).

In the other cases, tensors are returned without scaling.

Parameters:

- **from_int** (*bool**,**optional*) - if `True`, the tensor will be scaled from
the range [0, 255] to the range [0.0, 1.0]. if False`, the tensor
will not be scaled. if None, the tensor will be scaled if
it's not a floating-point tensor. default=None.
- **unsqueeze** (*bool*) - if `True`, the observation tensor is unsqueezed
along the first dimension. default=False.
- **dtype** ([*torch.dtype*](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*,**optional*) - dtype to use for the resulting
observations.

Keyword Arguments:

- **in_keys** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**NestedKeys*) - keys to process.
- **out_keys** ([*list*](torchrl.services.RayService.html#torchrl.services.RayService.list)*of**NestedKeys*) - keys to write.
- **shape_tolerant** (*bool**,**optional*) - if `True`, the shape of the input
images will be check. If the last channel is not 3, the permutation
will be ignored. Defaults to `False`.

Examples

```
>>> transform = ToTensorImage(in_keys=["pixels"])
>>> ri = torch.randint(0, 255, (1 , 1, 10, 11, 3), dtype=torch.uint8)
>>> td = TensorDict(
... {"pixels": ri},
... [1, 1])
>>> _ = transform(td)
>>> obs = td.get("pixels")
>>> print(obs.shape, obs.dtype)
torch.Size([1, 1, 3, 10, 11]) torch.float32
```

transform_observation_spec(*observation_spec: [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)*) → [TensorSpec](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)[[source]](../../_modules/torchrl/envs/transforms/_observation.html#ToTensorImage.transform_observation_spec)

Transforms the observation spec such that the resulting spec matches transform mapping.

Parameters:

**observation_spec** ([*TensorSpec*](torchrl.data.TensorSpec.html#torchrl.data.TensorSpec)) - spec before the transform

Returns:

expected spec after the transform