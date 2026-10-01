# VIPTransform

*class*torchrl.envs.transforms.VIPTransform(**args*, ***kwargs*)[[source]](../../_modules/torchrl/envs/transforms/vip.html#VIPTransform)

VIP Transform class.

VIP provides pre-trained ResNet weights aimed at facilitating visual
embedding and reward for robotic tasks. The models are trained using Ego4d.

See the paper: VIP: Towards Universal Visual Reward and Representation via Value-Implicit Pre-Training
(Jason Ma, Shagun Sodhani, Dinesh Jayaraman, Osbert Bastani, Vikash Kumar*, Amy Zhang*)

Parameters:

- **model_name** (*str*) - one of resnet50
- **in_keys** (*list**of**str**,**optional*) - list of input keys. If left empty, the
"pixels" key is assumed.
- **out_keys** (*list**of**str**,**optional*) - list of output keys. If left empty,
"vip_vec" is assumed.
- **size** (*int**,**optional*) - Size of the image to feed to resnet.
Defaults to 244.
- **stack_images** (*bool**,**optional*) - if False, the images given in the `in_keys`
argument will be treaded separately and each will be given a single,
separated entry in the output tensordict. Defaults to `True`.
- **download** (*bool**,**torchvision Weights config**or**corresponding string*) - if `True`, the weights will be downloaded using the torch.hub download
API (i.e. weights will be cached for future use).
These weights are the original weights from the VIP publication.
If the torchvision weights are needed, there are two ways they can be
obtained: `download=ResNet50_Weights.IMAGENET1K_V1` or `download="IMAGENET1K_V1"`
where `ResNet50_Weights` can be imported via `from torchvision.models import resnet50, ResNet50_Weights`.
Defaults to False.
- **download_path** (*str**,**optional*) - path where to download the models.
Default is None (cache path determined by torch.hub utils).
- **tensor_pixels_keys** (*list**of**str**,**optional*) - Optionally, one can keep the
original images (as collected from the env) in the output tensordict.
If no value is provided, this won't be collected.

to(*dest: [device](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device) | str | int | [dtype](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)*)[[source]](../../_modules/torchrl/envs/transforms/vip.html#VIPTransform.to)

Move and/or cast the parameters and buffers.

This can be called as

to(*device=None*, *dtype=None*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/vip.html#VIPTransform.to)

to(*dtype*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/vip.html#VIPTransform.to)

to(*tensor*, *non_blocking=False*)[[source]](../../_modules/torchrl/envs/transforms/vip.html#VIPTransform.to)

to(*memory_format=torch.channels_last*)[[source]](../../_modules/torchrl/envs/transforms/vip.html#VIPTransform.to)

Its signature is similar to [`torch.Tensor.to()`](https://docs.pytorch.org/docs/stable/generated/torch.Tensor.to.html#torch.Tensor.to), but only accepts
floating point or complex `dtype`s. In addition, this method will
only cast the floating point or complex parameters and buffers to `dtype`
(if given). The integral parameters and buffers will be moved
`device`, if that is given, but with dtypes unchanged. When
`non_blocking` is set, it tries to convert/move asynchronously
with respect to the host if possible, e.g., moving CPU Tensors with
pinned memory to CUDA devices.

See below for examples.

Note

This method modifies the module in-place.

Parameters:

- **device** ([`torch.device`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.device)) - the desired device of the parameters
and buffers in this module
- **dtype** ([`torch.dtype`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.dtype)) - the desired floating point or complex dtype of
the parameters and buffers in this module
- **tensor** ([*torch.Tensor*](https://docs.pytorch.org/docs/stable/tensors.html#torch.Tensor)) - Tensor whose dtype and device are the desired
dtype and device for all parameters and buffers in this module
- **memory_format** ([`torch.memory_format`](https://docs.pytorch.org/docs/stable/tensor_attributes.html#torch.memory_format)) - the desired memory
format for 4D parameters and buffers in this module (keyword
only argument)

Returns:

self

Return type:

Module

Examples:

```
>>> # xdoctest: +IGNORE_WANT("non-deterministic")
>>> linear = nn.Linear(2, 2)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
 [-0.5113, -0.2325]])
>>> linear.to(torch.double)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1913, -0.3420],
 [-0.5113, -0.2325]], dtype=torch.float64)
>>> # xdoctest: +REQUIRES(env:TORCH_DOCTEST_CUDA1)
>>> gpu1 = torch.device("cuda:1")
>>> linear.to(gpu1, dtype=torch.half, non_blocking=True)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
 [-0.5112, -0.2324]], dtype=torch.float16, device='cuda:1')
>>> cpu = torch.device("cpu")
>>> linear.to(cpu)
Linear(in_features=2, out_features=2, bias=True)
>>> linear.weight
Parameter containing:
tensor([[ 0.1914, -0.3420],
 [-0.5112, -0.2324]], dtype=torch.float16)

>>> linear = nn.Linear(2, 2, bias=None).to(torch.cdouble)
>>> linear.weight
Parameter containing:
tensor([[ 0.3741+0.j, 0.2382+0.j],
 [ 0.5593+0.j, -0.4443+0.j]], dtype=torch.complex128)
>>> linear(torch.ones(3, 2, dtype=torch.cdouble))
tensor([[0.6122+0.j, 0.1150+0.j],
 [0.6122+0.j, 0.1150+0.j],
 [0.6122+0.j, 0.1150+0.j]], dtype=torch.complex128)
```