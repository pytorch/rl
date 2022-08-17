#!/usr/bin/env python
# coding: utf-8

# In[1]:


from torchrl.data import TensorDict
from torchrl.data.tensordict.tensordict import TensorDictBase
import functorch
from torch import nn
import torch
from copy import copy, deepcopy

_RESET_OLD_TENSORDICT = True


# In[2]:


from functorch._src.vmap import _add_batch_dim, tree_unflatten, tree_flatten


# In[3]:


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


# In[4]:


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


# In[5]:


model = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3), nn.Sequential(nn.Linear(3, 4)))
print(model)


# In[6]:


tensordict_weights = extract_weights(model)
print(tensordict_weights)


# In[7]:


# accessing weights
tensordict_weights["0", "bias"]


# In[8]:


tensordict_weights["0"]["bias"]


# In[ ]:





# In[9]:


# flatten - unflatten
tensordict_weights_flatten = tensordict_weights.flatten_keys(separator=".", inplace=False)
print(tensordict_weights_flatten)


# In[10]:


tensordict_weights_unflatten = tensordict_weights_flatten.unflatten_keys(separator=".", inplace=False)
print(tensordict_weights_unflatten)


# In[11]:


# BatchedTensor
t = TensorDict({"a": torch.randn(3, 1), "b": TensorDict({"c": torch.randn(3, 1)}, [])}, [])
t = t.apply(lambda x: _add_batch_dim(x, 0, 0))
t["b", "c"]


# In[12]:


# requires_grad to False
tensordict_weights.apply(lambda x: x.requires_grad_(False), inplace=True)
tensordict_weights["0", "bias"]


# In[13]:


model = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3), nn.Sequential(nn.Linear(3, 4)))

fmodel, params = FunctionalModule._create_from(model)
params


# In[14]:


fmodel(params, torch.randn(1))


# In[15]:


fmodel(params, torch.randn(1, 1))


# In[16]:


functorch.vmap(torch.add, (0, 0))(torch.ones(10, 1), torch.ones(10, 1)).shape


# In[17]:


x = torch.randn(10, 1, 1)
out = functorch.vmap(fmodel, (None, 0))(params, x)  # works
print(out.shape)


# In[18]:


out = functorch.vmap(fmodel, (0, 0))(params.expand(10), x)  # works
print(out.shape)


# In[19]:


# benchmarking
from functorch._src.make_functional import FunctionalModule as FunctionalModule_orig

model = nn.Sequential(nn.Linear(1, 2), nn.Linear(2, 3), nn.Sequential(nn.Linear(3, 4)))
get_ipython().run_line_magic('timeit', 'FunctionalModule_orig._create_from(model)')
get_ipython().run_line_magic('timeit', 'FunctionalModule._create_from(model)')


# In[20]:


module_orig, params_orig = FunctionalModule_orig._create_from(model)
module, params = FunctionalModule._create_from(model)

# fair comparison
_RESET_OLD_TENSORDICT = True
x = torch.randn(1)
get_ipython().run_line_magic('timeit', 'module_orig(params_orig, x)')
get_ipython().run_line_magic('timeit', 'module(params, x)')


# In[21]:


# unfair comparison -- does not swap back the params
_RESET_OLD_TENSORDICT = False
x = torch.randn(1)
get_ipython().run_line_magic('timeit', 'module_orig(params_orig, x)')
get_ipython().run_line_magic('timeit', 'module(params, x)')


# In[ ]:




