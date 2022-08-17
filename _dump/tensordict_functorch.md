```python
from torchrl.data import TensorDict
from torchrl.data.tensordict.tensordict import TensorDictBase
import functorch
from torch import nn
import torch
from copy import copy, deepcopy

_RESET_OLD_TENSORDICT = True
```


```python
from functorch._src.vmap import _add_batch_dim, tree_unflatten, tree_flatten
```


```python
class FunctionalModule(nn.Module):
    """
    This is the callable object returned by :func:`make_functional`.
    """

    def __init__(self, stateless_model):
        super(FunctionalModule, self).__init__()
        self.stateless_model = stateless_model

    @staticmethod
    def _create_from(model, disable_autograd_tracking=False):
        # TODO: We don't need to copy the model to create a stateless copy
        model_copy = deepcopy(model)
        param_tensordict = extract_weights(model_copy)
        if disable_autograd_tracking:
            tensordict_weights.apply(lambda x: x.requires_grad_(False), inplace=True)
        return FunctionalModule(model_copy), param_tensordict

    def forward(self, params, *args, **kwargs):
        # Temporarily load the state back onto self.stateless_model
        old_state = _swap_state(self.stateless_model, params, return_old_tensordict=_RESET_OLD_TENSORDICT)
        try:
            return self.stateless_model(*args, **kwargs)
        finally:
            # Remove the loaded state on self.stateless_model
            if _RESET_OLD_TENSORDICT:
                _swap_state(self.stateless_model, old_state)

```



```python
def extract_weights(model):
    tensordict = TensorDict({}, [])
    for name, param in list(model.named_parameters(recurse=False)):
        setattr(model, name, None)
        tensordict[name] = param
    for name, module in model.named_children():
        module_tensordict = extract_weights(module)
        if module_tensordict is not None:
            tensordict[name] = module_tensordict
    if len(tensordict.keys()):
        return tensordict
    else:
        return None

def _swap_state(model, tensordict, return_old_tensordict=False):
#     if return_old_tensordict:
#         old_tensordict = tensordict.clone(recursive=False)
#         old_tensordict.batch_size = []
    
    if return_old_tensordict:
        old_tensordict = TensorDict({}, [], device=tensordict._device_safe())

    for key, value in list(tensordict.items()):
        if isinstance(value, TensorDictBase):
            _swap_state(getattr(model, key), value)
        else:
            if return_old_tensordict:
                old_attr = getattr(model, key)
                if old_attr is None:
                    old_attr = torch.tensor([]).view(*value.shape, 0)
            delattr(model, key)
            setattr(model, key, value)
            if return_old_tensordict:
                old_tensordict.set(key, old_attr)
    if return_old_tensordict:
        return old_tensordict
```


```python
model = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3), nn.Sequential(nn.Linear(3, 4)))
print(model)
```

    Sequential(
      (0): Linear(in_features=1, out_features=2, bias=True)
      (1): Linear(in_features=2, out_features=3, bias=True)
      (2): Sequential(
        (0): Linear(in_features=3, out_features=4, bias=True)
      )
    )



```python
tensordict_weights = extract_weights(model)
print(tensordict_weights)
```

    TensorDict(
        fields={
            0: TensorDict(
                fields={
                    bias: Tensor(torch.Size([2]), dtype=torch.float32),
                    weight: Tensor(torch.Size([2, 1]), dtype=torch.float32)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False),
            1: TensorDict(
                fields={
                    bias: Tensor(torch.Size([3]), dtype=torch.float32),
                    weight: Tensor(torch.Size([3, 2]), dtype=torch.float32)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False),
            2: TensorDict(
                fields={
                    0: TensorDict(
                        fields={
                            bias: Tensor(torch.Size([4]), dtype=torch.float32),
                            weight: Tensor(torch.Size([4, 3]), dtype=torch.float32)},
                        batch_size=torch.Size([]),
                        device=cpu,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False)},
        batch_size=torch.Size([]),
        device=cpu,
        is_shared=False)



```python
# accessing weights
tensordict_weights["0", "bias"]
```




    Parameter containing:
    tensor([0.1881, 0.8179], requires_grad=True)




```python
tensordict_weights["0"]["bias"]
```




    Parameter containing:
    tensor([0.1881, 0.8179], requires_grad=True)




```python

```


```python
# flatten - unflatten
tensordict_weights_flatten = tensordict_weights.flatten_keys(separator=".", inplace=False)
print(tensordict_weights_flatten)
```

    TensorDict(
        fields={
            0.bias: Tensor(torch.Size([2]), dtype=torch.float32),
            0.weight: Tensor(torch.Size([2, 1]), dtype=torch.float32),
            1.bias: Tensor(torch.Size([3]), dtype=torch.float32),
            1.weight: Tensor(torch.Size([3, 2]), dtype=torch.float32),
            2.0.bias: Tensor(torch.Size([4]), dtype=torch.float32),
            2.0.weight: Tensor(torch.Size([4, 3]), dtype=torch.float32)},
        batch_size=torch.Size([]),
        device=cpu,
        is_shared=False)



```python
tensordict_weights_unflatten = tensordict_weights_flatten.unflatten_keys(separator=".", inplace=False)
print(tensordict_weights_unflatten)
```

    TensorDict(
        fields={
            0: TensorDict(
                fields={
                    bias: Tensor(torch.Size([2]), dtype=torch.float32),
                    weight: Tensor(torch.Size([2, 1]), dtype=torch.float32)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False),
            1: TensorDict(
                fields={
                    bias: Tensor(torch.Size([3]), dtype=torch.float32),
                    weight: Tensor(torch.Size([3, 2]), dtype=torch.float32)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False),
            2: TensorDict(
                fields={
                    0: TensorDict(
                        fields={
                            bias: Tensor(torch.Size([4]), dtype=torch.float32),
                            weight: Tensor(torch.Size([4, 3]), dtype=torch.float32)},
                        batch_size=torch.Size([]),
                        device=cpu,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False)},
        batch_size=torch.Size([]),
        device=cpu,
        is_shared=False)



```python
# BatchedTensor
t = TensorDict({"a": torch.randn(3, 1), "b": TensorDict({"c": torch.randn(3, 1)}, [])}, [])
t = t.apply(lambda x: _add_batch_dim(x, 0, 0))
t["b", "c"]
```




    BatchedTensor(lvl=0, bdim=0, value=
        tensor([[ 1.9844],
                [-2.1292],
                [ 1.4221]])
    )




```python
# requires_grad to False
tensordict_weights.apply(lambda x: x.requires_grad_(False), inplace=True)
tensordict_weights["0", "bias"]
```




    Parameter containing:
    tensor([0.1881, 0.8179])




```python
model = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3), nn.Sequential(nn.Linear(3, 4)))

fmodel, params = FunctionalModule._create_from(model)
params
```




    TensorDict(
        fields={
            0: TensorDict(
                fields={
                    bias: Tensor(torch.Size([2]), dtype=torch.float32),
                    weight: Tensor(torch.Size([2, 1]), dtype=torch.float32)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False),
            1: TensorDict(
                fields={
                    bias: Tensor(torch.Size([3]), dtype=torch.float32),
                    weight: Tensor(torch.Size([3, 2]), dtype=torch.float32)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False),
            2: TensorDict(
                fields={
                    0: TensorDict(
                        fields={
                            bias: Tensor(torch.Size([4]), dtype=torch.float32),
                            weight: Tensor(torch.Size([4, 3]), dtype=torch.float32)},
                        batch_size=torch.Size([]),
                        device=cpu,
                        is_shared=False)},
                batch_size=torch.Size([]),
                device=cpu,
                is_shared=False)},
        batch_size=torch.Size([]),
        device=cpu,
        is_shared=False)




```python
fmodel(params, torch.randn(1))
```




    tensor([-0.3595,  0.5177, -0.0109, -0.6153], grad_fn=<AddBackward0>)




```python
fmodel(params, torch.randn(1, 1))
```




    tensor([[-0.3915,  0.1743, -0.0517, -0.6969]], grad_fn=<AddmmBackward0>)




```python
functorch.vmap(torch.add, (0, 0))(torch.ones(10, 1), torch.ones(10, 1)).shape
```




    torch.Size([10, 1])




```python
x = torch.randn(10, 1, 1)
out = functorch.vmap(fmodel, (None, 0))(params, x)  # works
print(out.shape)
```

    torch.Size([10, 1, 4])



```python
out = functorch.vmap(fmodel, (0, 0))(params.expand(10), x)  # works
print(out.shape)
```

    torch.Size([10, 1, 4])



```python
# benchmarking
from functorch._src.make_functional import FunctionalModule as FunctionalModule_orig

model = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3), nn.Sequential(nn.Linear(3, 4)))
%timeit FunctionalModule_orig._create_from(model)
%timeit FunctionalModule._create_from(model)
```

    494 µs ± 1.03 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    2.02 ms ± 26.8 µs per loop (mean ± std. dev. of 7 runs, 100 loops each)



```python
module_orig, params_orig = FunctionalModule_orig._create_from(model)
module, params = FunctionalModule._create_from(model)

# fair comparison
_RESET_OLD_TENSORDICT = True
x = torch.randn(1)
%timeit module_orig(params_orig, x)
%timeit module(params, x)
```

    228 µs ± 1.32 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    168 µs ± 20.8 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)



```python
# unfair comparison -- does not swap back the params
_RESET_OLD_TENSORDICT = False
x = torch.randn(1)
%timeit module_orig(params_orig, x)
%timeit module(params, x)
```

    231 µs ± 1.38 µs per loop (mean ± std. dev. of 7 runs, 1000 loops each)
    146 µs ± 10.6 µs per loop (mean ± std. dev. of 7 runs, 10000 loops each)


