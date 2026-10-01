# make_composite_from_td

torchrl.envs.make_composite_from_td(*data*, ***, *unsqueeze_null_shapes: bool = True*, *dynamic_shape: bool = False*)[[source]](../../_modules/torchrl/envs/utils.html#make_composite_from_td)

Creates a Composite instance from a tensordict, assuming all values are unbounded.

Parameters:

**data** ([*tensordict.TensorDict*](https://docs.pytorch.org/tensordict/stable/reference/generated/tensordict.TensorDict.html#tensordict.TensorDict)) - a tensordict to be mapped onto a Composite.

Keyword Arguments:

- **unsqueeze_null_shapes** (*bool**,**optional*) - if `True`, every empty shape will be
unsqueezed to (1,). Defaults to `True`.
- **dynamic_shape** (*bool**,**optional*) - if `True`, all tensors will be assumed to have a dynamic shape
along the last dimension. Defaults to `False`.

Examples

```
>>> from tensordict import TensorDict
>>> data = TensorDict({
... "obs": torch.randn(3),
... "action": torch.zeros(2, dtype=torch.int),
... "next": {"obs": torch.randn(3), "reward": torch.randn(1)}
... }, [])
>>> spec = make_composite_from_td(data)
>>> print(spec)
Composite(
 obs: UnboundedContinuous(
 shape=torch.Size([3]), space=None, device=cpu, dtype=torch.float32, domain=continuous),
 action: UnboundedContinuous(
 shape=torch.Size([2]), space=None, device=cpu, dtype=torch.int32, domain=continuous),
 next: Composite(
 obs: UnboundedContinuous(
 shape=torch.Size([3]), space=None, device=cpu, dtype=torch.float32, domain=continuous),
 reward: UnboundedContinuous(
 shape=torch.Size([1]), space=ContinuousBox(low=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True), high=Tensor(shape=torch.Size([]), device=cpu, dtype=torch.float32, contiguous=True)), device=cpu, dtype=torch.float32, domain=continuous), device=cpu, shape=torch.Size([])), device=cpu, shape=torch.Size([]))
>>> assert (spec.zero() == data.zero_()).all()
```